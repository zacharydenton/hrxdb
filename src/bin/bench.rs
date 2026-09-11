//! Reproducible host-completion benchmark. No compilation or ingestion is timed.
use hrxdb::{FlatIndex, Measurement, ScanConfig};
use serde::Serialize;
use std::{path::PathBuf, time::Instant};

#[derive(Debug)]
struct Options {
    rows: usize,
    dimensions: usize,
    samples: usize,
    k: usize,
    exclude_first: usize,
    batch: usize,
    sweep: bool,
    output: Option<PathBuf>,
    config: ScanConfig,
}

fn options() -> Result<Options, String> {
    let mut o = Options {
        rows: 10_000_000,
        dimensions: 384,
        samples: 30,
        k: 10,
        exclude_first: 0,
        batch: 1,
        sweep: false,
        output: None,
        config: ScanConfig::default(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!(
                "hrxdb-bench [--rows 10000000] [--dimensions 384] [--samples 30] [--k 10] [--exclude-first 0] [--batch 1]\n             [--sweep] [--threads 128] [--rows-per-wave 2] [--load-width 4] [--output results.json]\n             Builds once, checks sample scores, then measures completed single-query work.\n             --sweep tests all 16 schedules on the same corpus."
            );
            std::process::exit(0);
        }
        if key == "--sweep" {
            o.sweep = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if key == "--output" {
            o.output = Some(value.into());
            continue;
        }
        let value: usize = value
            .parse()
            .map_err(|_| format!("invalid value for {key}"))?;
        match key.as_str() {
            "--rows" => o.rows = value,
            "--dimensions" => o.dimensions = value,
            "--samples" => o.samples = value,
            "--k" => o.k = value,
            "--exclude-first" => o.exclude_first = value,
            "--batch" => o.batch = value,
            "--threads" => o.config.threads = value,
            "--rows-per-wave" => o.config.rows_per_wave = value,
            "--load-width" => o.config.load_width = value,
            _ => return Err(format!("unknown option {key}")),
        }
    }
    if o.rows == 0 || o.samples == 0 || !(1..=1024).contains(&o.k) || o.exclude_first >= o.rows {
        return Err("rows and samples must be positive; k must be in 1..=1024; exclude-first must be less than rows".into());
    }
    if !(1..=64).contains(&o.batch) || (o.batch > 1 && o.sweep) {
        return Err("batch must be in 1..=64; batched GEMM does not use --sweep".into());
    }
    Ok(o)
}

fn row(i: usize, d: usize) -> Vec<f32> {
    let mut x = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
    (0..d)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // Keep corpus rows nonzero even for an accidental zero RNG seed.
            (x >> 8) as f32 / 8388608.0 - 1.0
        })
        .collect()
}

fn score(id: usize, d: usize, query: &[f32], quantize: bool) -> f64 {
    let mut values = row(id, d);
    let norm = values
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    for value in &mut values {
        *value = if quantize {
            half::f16::from_f64(*value as f64 / norm).to_f32()
        } else {
            (*value as f64 / norm) as f32
        };
    }
    let inverse = (1.0
        / values
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt()) as f32;
    let qnorm = query
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    values
        .iter()
        .zip(query)
        .map(|(&x, &q)| x as f64 * ((q as f64 / qnorm) as f32) as f64)
        .sum::<f64>()
        * inverse as f64
}

fn percentile(v: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mut values: Vec<_> = v.collect();
    values.sort_by(f64::total_cmp);
    values[((values.len() as f64 * p).ceil() as usize).saturating_sub(1)]
}

#[derive(Serialize)]
struct Trial {
    config: ScanConfig,
    scan_median_ms: f64,
    scan_p95_ms: f64,
    search_median_ms: f64,
    search_p95_ms: f64,
    read_control_median_ms: f64,
    scan_gb_s: f64,
    search_gb_s: f64,
    read_control_gb_s: f64,
    fraction_of_read_control: f64,
    fraction_of_256_gb_s: f64,
    scan_target_met: bool,
    resources: Option<Resources>,
    compiler: Vec<hrxdb::Compilation>,
    samples: Vec<Measurement>,
}

#[derive(Serialize)]
struct Resources {
    vgpr_count: u64,
    sgpr_count: u64,
    private_segment_bytes: u64,
    workgroup_segment_bytes: u64,
}

// LLVM tools are optional benchmark diagnostics, never a library dependency.
fn resources(artifact: &str) -> Option<Resources> {
    let tool = std::env::var_os("HRXDB_LLVM_READOBJ").unwrap_or_else(|| "llvm-readobj".into());
    let output = std::process::Command::new(tool)
        .args(["--notes", artifact])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let notes = String::from_utf8(output.stdout).ok()?;
    let value = |key: &str| {
        notes.lines().find_map(|line| {
            let (name, value) = line.trim().split_once(':')?;
            (name == key)
                .then(|| value.trim().parse::<u64>().ok())
                .flatten()
        })
    };
    Some(Resources {
        vgpr_count: value(".vgpr_count")?,
        sgpr_count: value(".sgpr_count")?,
        private_segment_bytes: value(".private_segment_fixed_size")?,
        workgroup_segment_bytes: value(".group_segment_fixed_size")?,
    })
}

#[derive(Serialize)]
struct Report {
    target: String,
    rows: usize,
    dimensions: usize,
    padded_dimensions: usize,
    k: usize,
    excluded_rows: usize,
    shards: usize,
    vector_bytes: usize,
    norm_bytes: usize,
    score_bytes: usize,
    ingestion_and_compile_seconds: f64,
    timing: &'static str,
    validation: &'static str,
    quantization_sample_rows: usize,
    quantization_sample_top_k_overlap: f64,
    selected_config: ScanConfig,
    trials: Vec<Trial>,
}

fn run(o: Options) -> hrxdb::Result<()> {
    if o.batch > 1 {
        return run_batch(o);
    }
    let device = hrx::Device::open(0)?;
    eprintln!(
        "Building {} x {} on {}",
        o.rows,
        o.dimensions,
        device.target().as_str()
    );
    let start = Instant::now();
    let mut db = FlatIndex::build_with_config(
        &device,
        o.dimensions,
        (0..o.rows).map(|i| row(i, o.dimensions)),
        o.config,
    )?;
    let build_seconds = start.elapsed().as_secs_f64();
    db.reserve_search(o.k)?;
    let excluded: Vec<u32> = (0..o.exclude_first as u32).collect();
    eprintln!("Ingestion and compilation: {build_seconds:.2}s");
    let configs: Vec<_> = if o.sweep {
        ScanConfig::configurations().collect()
    } else {
        vec![o.config]
    };
    let vector_bytes = o.rows * o.dimensions * 2;
    let mut trials = Vec::new();
    for config in configs {
        db.configure(config)?;
        let query = row(o.rows + 888, o.dimensions);
        // Check evenly spaced rows, allocation tail, and both sides of 4 GiB.
        let scores = db.scores(&query)?;
        let mut checks: Vec<_> = (0..64).map(|j| j * (o.rows - 1) / 63).collect();
        let boundary = (1usize << 32) / (db.padded_dimensions() * 2);
        for id in boundary.saturating_sub(1)..=boundary + 1 {
            if id < o.rows {
                checks.push(id);
            }
        }
        for id in checks {
            let reference = score(id, o.dimensions, &query, true);
            if !scores[id].is_finite() || (scores[id] as f64 - reference).abs() > 3e-6 {
                return Err(hrxdb::Error::Message(format!(
                    "score mismatch at {id}: GPU={} CPU={reference}",
                    scores[id]
                )));
            }
        }
        // Validate selection against every GPU score without an O(N log N) sort.
        // Traversal is by ascending ID, so equal scores retain the earlier ID.
        let k = o.k.min(o.rows - o.exclude_first);
        let mut expected: Vec<hrxdb::Neighbor> = Vec::with_capacity(k + 1);
        for (id, &similarity) in scores.iter().enumerate() {
            if !similarity.is_finite() {
                return Err(hrxdb::Error::Message(format!("nonfinite score at {id}")));
            }
            if id < o.exclude_first {
                continue;
            }
            if expected.len() == k && similarity <= expected[k - 1].similarity {
                continue;
            }
            let at = expected.partition_point(|n| n.similarity >= similarity);
            expected.insert(
                at,
                hrxdb::Neighbor {
                    id: id as u32,
                    similarity,
                },
            );
            expected.truncate(k);
        }
        let selected = db.search_excluding(&query, o.k, &excluded)?;
        if selected != expected {
            return Err(hrxdb::Error::Message(
                "GPU top-k differs from CPU selection over the full score array".into(),
            ));
        }
        drop(scores);
        let mut samples = Vec::new();
        for i in 0..o.samples + 3 {
            let query = row(o.rows + i + 1, o.dimensions);
            let sample = db.measure_excluding(&query, o.k, &excluded)?;
            if sample.neighbors.len() != k {
                return Err(hrxdb::Error::Message("wrong result count".into()));
            }
            for pair in sample.neighbors.windows(2) {
                if pair[0].similarity < pair[1].similarity
                    || (pair[0].similarity == pair[1].similarity && pair[0].id >= pair[1].id)
                {
                    return Err(hrxdb::Error::Message(
                        "unsorted or duplicate neighbors".into(),
                    ));
                }
            }
            for n in &sample.neighbors {
                if (n.id as usize) < o.exclude_first
                    || n.id as usize >= o.rows
                    || !n.similarity.is_finite()
                {
                    return Err(hrxdb::Error::Message("invalid neighbor returned".into()));
                }
                let reference = score(n.id as usize, o.dimensions, &query, true);
                if (n.similarity as f64 - reference).abs() > 3e-6 {
                    return Err(hrxdb::Error::Message(
                        "returned score differs from CPU reference".into(),
                    ));
                }
            }
            if i >= 3 {
                samples.push(sample);
            }
        }
        let scan = percentile(samples.iter().map(|s| s.scan_ms), 0.5);
        let search = percentile(samples.iter().map(|s| s.search_ms), 0.5);
        let read = percentile(samples.iter().map(|s| s.read_control_ms), 0.5);
        let rate = |ms| vector_bytes as f64 / (ms * 1e6);
        eprintln!(
            "{config:?}: scan {:.2} GB/s, search {:.2} ms, {:.1}% of read control",
            rate(scan),
            search,
            read / scan * 100.0
        );
        trials.push(Trial {
            config,
            scan_median_ms: scan,
            search_median_ms: search,
            scan_p95_ms: percentile(samples.iter().map(|s| s.scan_ms), 0.95),
            search_p95_ms: percentile(samples.iter().map(|s| s.search_ms), 0.95),
            read_control_median_ms: read,
            scan_gb_s: rate(scan),
            search_gb_s: rate(search),
            read_control_gb_s: rate(read),
            fraction_of_read_control: read / scan,
            fraction_of_256_gb_s: rate(scan) / 256.0,
            scan_target_met: read / scan >= 0.9,
            resources: resources(&db.compilation_reports()[0].artifact),
            compiler: db.compilation_reports().to_vec(),
            samples,
        });
    }
    let fastest = trials
        .iter()
        .map(|t| t.search_median_ms)
        .min_by(f64::total_cmp)
        .unwrap();
    // Prefer fewer VGPRs for results within 1%; without LLVM diagnostics,
    // select purely by latency and leave resources explicitly null.
    let best = trials
        .iter()
        .filter(|t| t.search_median_ms <= fastest * 1.01)
        .min_by(|a, b| {
            let registers = |t: &Trial| {
                t.resources
                    .as_ref()
                    .map(|r| r.vgpr_count)
                    .unwrap_or(u64::MAX)
            };
            registers(a)
                .cmp(&registers(b))
                .then(a.search_median_ms.total_cmp(&b.search_median_ms))
        })
        .unwrap()
        .config;
    let query = row(o.rows + 99, o.dimensions);
    let sample_n = o.rows.min(8192);
    let ranked = |quantize| {
        let mut rows: Vec<_> = (0..sample_n)
            .map(|i| (i, score(i, o.dimensions, &query, quantize)))
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        rows.truncate(o.k.min(sample_n));
        rows
    };
    let original = ranked(false);
    let quantized = ranked(true);
    let overlap = quantized
        .iter()
        .filter(|(id, _)| original.iter().any(|(other, _)| id == other))
        .count() as f64
        / original.len() as f64;
    let mut report = Report {
        target: device.target().as_str().to_string(),
        rows: o.rows,
        dimensions: o.dimensions,
        padded_dimensions: db.padded_dimensions(),
        k: o.k,
        excluded_rows: o.exclude_first,
        shards: db.shard_count(),
        vector_bytes,
        norm_bytes: o.rows * 4,
        score_bytes: o.rows * 4,
        ingestion_and_compile_seconds: build_seconds,
        timing: "Host wall time with explicit completion; three warmups; serialized changing queries; scratch reserved before timing; scan/control exclude query preparation; search includes normalization, query publication, optional exclusion bitmap preparation and masking, selection and readback. GB/s uses unpadded FP16 vector bytes and decimal GB.",
        validation: "GPU selection checked against a CPU top-k over the entire GPU score array after exclusions for each configuration. Sampled scores include the allocation tail and 4 GiB boundary; all returned scores checked against quantized CPU reference. This is not an exhaustive CPU recomputation of every dot product. Quantization overlap is measured without exclusions on a separate subset.",
        quantization_sample_rows: sample_n,
        quantization_sample_top_k_overlap: overlap,
        selected_config: best,
        trials,
    };
    // Preserve content hashes while keeping user-specific cache paths out of
    // reports intended for sharing. Resource inspection above used local paths.
    for trial in &mut report.trials {
        for compilation in &mut trial.compiler {
            let path = std::path::Path::new(&compilation.artifact);
            if let (Some(key), Some(file)) = (
                path.parent().and_then(std::path::Path::file_name),
                path.file_name(),
            ) {
                compilation.artifact = format!(
                    "hrx-cache/kernels/{}/{}",
                    key.to_string_lossy(),
                    file.to_string_lossy()
                );
            }
        }
    }
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = o.output {
        std::fs::write(path, &json).map_err(|e| hrxdb::Error::Message(e.to_string()))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn run_batch(o: Options) -> hrxdb::Result<()> {
    let device = hrx::Device::open(0)?;
    eprintln!(
        "Building {} x {} for {}-query batches",
        o.rows, o.dimensions, o.batch
    );
    let start = Instant::now();
    let mut db = FlatIndex::build_with_config(
        &device,
        o.dimensions,
        (0..o.rows).map(|i| row(i, o.dimensions)),
        o.config,
    )?;
    let build_seconds = start.elapsed().as_secs_f64();
    db.reserve_search(o.k)?;
    let start = Instant::now();
    db.reserve_batch(o.batch, o.k)?;
    let reserve_seconds = start.elapsed().as_secs_f64();
    let excluded: Vec<u32> = (0..o.exclude_first as u32).collect();
    let mut samples = Vec::new();
    let mut near_tie_rank_differences = 0;
    let mut max_score_error = 0.0f64;
    for sample in 0..o.samples + 3 {
        let queries: Vec<_> = (0..o.batch)
            .flat_map(|q| row(o.rows + 888 + sample * o.batch + q, o.dimensions))
            .collect();
        let mut run = |batched| -> hrxdb::Result<_> {
            let start = Instant::now();
            let neighbors = if batched {
                db.search_batch_excluding(&queries, o.k, &excluded)?
            } else {
                queries
                    .chunks_exact(o.dimensions)
                    .map(|q| db.search_excluding(q, o.k, &excluded))
                    .collect::<hrxdb::Result<Vec<_>>>()?
            };
            Ok((start.elapsed().as_secs_f64() * 1000.0, neighbors))
        };
        let (sequential, batched) = if sample % 2 == 0 {
            (run(false)?, run(true)?)
        } else {
            let b = run(true)?;
            (run(false)?, b)
        };
        if batched.1.len() != o.batch {
            return Err(hrxdb::Error::Message("batch query count mismatch".into()));
        }
        for ((actual, expected), query) in batched
            .1
            .iter()
            .zip(&sequential.1)
            .zip(queries.chunks_exact(o.dimensions))
        {
            if actual.len() != expected.len() {
                return Err(hrxdb::Error::Message("batch result count mismatch".into()));
            }
            if actual.windows(2).any(|w| {
                w[0].similarity < w[1].similarity
                    || (w[0].similarity == w[1].similarity && w[0].id >= w[1].id)
            }) {
                return Err(hrxdb::Error::Message(
                    "unsorted or duplicate batch neighbors".into(),
                ));
            }
            for (got, want) in actual.iter().zip(expected) {
                if got.id != want.id {
                    near_tie_rank_differences += 1;
                    if (got.similarity - want.similarity).abs() > 3e-6 {
                        return Err(hrxdb::Error::Message(
                            "batch ranking differs beyond rounding tolerance".into(),
                        ));
                    }
                }
                if (got.id as usize) < o.exclude_first
                    || got.id as usize >= o.rows
                    || !got.similarity.is_finite()
                {
                    return Err(hrxdb::Error::Message(
                        "batch returned invalid or excluded ID".into(),
                    ));
                }
                let error = (got.similarity as f64
                    - score(got.id as usize, o.dimensions, query, true))
                .abs();
                max_score_error = max_score_error.max(error);
                if error > 3e-6 {
                    return Err(hrxdb::Error::Message(
                        "batch score differs from CPU reference".into(),
                    ));
                }
            }
        }
        eprintln!(
            "batch {sample}: sequential {:.2} ms, batched {:.2} ms",
            sequential.0, batched.0
        );
        if sample >= 3 {
            samples.push(serde_json::json!({"sequential_ms": sequential.0, "batch_ms": batched.0}));
        }
    }
    let median = |key: &str| percentile(samples.iter().map(|s| s[key].as_f64().unwrap()), 0.5);
    let sequential = median("sequential_ms");
    let batched = median("batch_ms");
    let mut reports = db.compilation_reports().to_vec();
    for report in &mut reports {
        let path = std::path::Path::new(&report.artifact);
        if let (Some(key), Some(file)) =
            (path.parent().and_then(|p| p.file_name()), path.file_name())
        {
            report.artifact = format!(
                "hrx-cache/kernels/{}/{}",
                key.to_string_lossy(),
                file.to_string_lossy()
            );
        }
    }
    let report = serde_json::json!({
        "target": device.target().as_str(), "rows": o.rows, "dimensions": o.dimensions,
        "batch": o.batch, "k": o.k, "excluded_rows": o.exclude_first, "shards": db.shard_count(),
        "ingestion_and_compile_seconds": build_seconds, "batch_reserve_seconds": reserve_seconds,
        "batch_workspace_bytes": db.batch_workspace_bytes(), "sequential_median_ms": sequential,
        "batch_median_ms": batched, "speedup": sequential / batched,
        "batch_p95_ms": percentile(samples.iter().map(|s| s["batch_ms"].as_f64().unwrap()), 0.95),
        "max_returned_score_error": max_score_error, "near_tie_rank_differences": near_tie_rank_differences,
        "timing": "Host completion. Three warmups, changing queries, alternating sequential/batch timing order. Compilation and workspace reservation excluded; query preparation, selection, masking and readback included. No concurrent hrxdb benchmark or GPU tests.",
        "validation": "Each batch compared with individual GPU searches. ID differences accepted only within 3e-6 score tolerance and counted. All returned scores checked against quantized CPU reference.",
        "compiler": reports, "samples": samples,
    });
    eprintln!(
        "Batch median {batched:.2} ms vs {sequential:.2} ms sequential: {:.2}x",
        sequential / batched
    );
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = o.output {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn main() -> hrxdb::Result<()> {
    run(options().map_err(hrxdb::Error::Message)?)
}
