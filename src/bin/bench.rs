//! Reproducible host-completion benchmark. No compilation or ingestion is timed.
use hrx::benchmark::Distribution;
use hrxdb::{Measurement, ScanConfig, Searcher};
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
    append: usize,
    appends: usize,
    updates: usize,
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
        append: 0,
        appends: 1_000,
        updates: 100,
        sweep: false,
        output: None,
        config: ScanConfig::default(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!(
                "hrxdb-bench [--rows 10000000] [--dimensions 384] [--samples 30] [--k 10] [--exclude-first 0] [--batch 1]\n             [--sweep] [--threads 128] [--rows-per-wave 2] [--load-width 4] [--output results.json]\n             [--append 50 [--appends 1000] [--updates 100]]\n             Builds once, checks sample scores, then measures completed single-query work.\n             --sweep tests all 16 schedules on the same corpus.\n             --append times appends of that many rows onto --rows, searcher handover, and\n             search against a single build of the same rows, then scattered updates."
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
            "--append" => o.append = value,
            "--appends" => o.appends = value,
            "--updates" => o.updates = value,
            "--threads" => o.config.threads = value,
            "--rows-per-wave" => o.config.rows_per_wave = value,
            "--load-width" => o.config.load_width = value,
            _ => return Err(format!("unknown option {key}")),
        }
    }
    if o.rows == 0
        || o.samples == 0
        || !(1..=hrxdb::MAX_K).contains(&o.k)
        || o.exclude_first >= o.rows
    {
        return Err("rows and samples must be positive; k must be in 1..=1024; exclude-first must be less than rows".into());
    }
    if !(1..=hrxdb::MAX_BATCH).contains(&o.batch) || (o.batch > 1 && o.sweep) {
        return Err("batch must be in 1..=64; batched GEMM does not use --sweep".into());
    }
    if o.append > 0 && o.appends == 0 {
        return Err("appends must be positive when --append is enabled".into());
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
    score_row(row(id, d), query, quantize)
}

fn score_row(mut values: Vec<f32>, query: &[f32], quantize: bool) -> f64 {
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

fn distribution(values: impl Iterator<Item = f64>) -> hrxdb::Result<Distribution> {
    Distribution::from_samples(values.collect())
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
    if o.append > 0 {
        return run_append(o);
    }
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
    let mut db = Searcher::build_with_config(
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
        let scan_distribution = distribution(samples.iter().map(|s| s.scan_ms))?;
        let search_distribution = distribution(samples.iter().map(|s| s.search_ms))?;
        let read_distribution = distribution(samples.iter().map(|s| s.read_control_ms))?;
        let scan = scan_distribution.median_ms;
        let search = search_distribution.median_ms;
        let read = read_distribution.median_ms;
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
            scan_p95_ms: scan_distribution.p95_ms,
            search_p95_ms: search_distribution.p95_ms,
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
    let mut db = Searcher::build_with_config(
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
    let sequential_distribution = distribution(
        samples
            .iter()
            .map(|sample| sample["sequential_ms"].as_f64().unwrap()),
    )?;
    let batch_distribution = distribution(
        samples
            .iter()
            .map(|sample| sample["batch_ms"].as_f64().unwrap()),
    )?;
    let sequential = sequential_distribution.median_ms;
    let batched = batch_distribution.median_ms;
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
        "batch_p95_ms": batch_distribution.p95_ms,
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

/// Timed searches over two searchers, alternating which runs first.
struct Comparison {
    single: [Vec<f64>; 2],
    batch: [Vec<f64>; 2],
}

const APPEND_BATCH: usize = 60;

fn compare(
    searchers: [&mut Searcher; 2],
    o: &Options,
    seed: usize,
    same_rows: bool,
) -> hrxdb::Result<Comparison> {
    let [a, b] = searchers;
    let mut single = [Vec::new(), Vec::new()];
    let mut batch = [Vec::new(), Vec::new()];
    for sample in 0..o.samples + 3 {
        let base = seed + sample * (APPEND_BATCH + 1);
        let query = row(base, o.dimensions);
        let queries: Vec<f32> = (1..=APPEND_BATCH)
            .flat_map(|q| row(base + q, o.dimensions))
            .collect();
        let mut results = [None, None];
        let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
        for side in order {
            let searcher = if side == 0 { &mut *a } else { &mut *b };
            let start = Instant::now();
            let one = searcher.search(&query, o.k)?;
            let single_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let many = searcher.search_batch(&queries, o.k)?;
            let batch_ms = start.elapsed().as_secs_f64() * 1000.0;
            if sample >= 3 {
                single[side].push(single_ms);
                batch[side].push(batch_ms);
            }
            results[side] = Some((one, many));
        }
        if same_rows && results[0] != results[1] {
            return Err(hrxdb::Error::Message(
                "searches over the same rows differ".into(),
            ));
        }
    }
    Ok(Comparison { single, batch })
}

fn comparison_json(c: &Comparison, names: [&str; 2]) -> hrxdb::Result<serde_json::Value> {
    let mut sides = serde_json::Map::new();
    let mut medians = [[0.0; 2]; 2];
    for (i, name) in names.into_iter().enumerate() {
        let single = distribution(c.single[i].iter().copied())?;
        let batch = distribution(c.batch[i].iter().copied())?;
        medians[i] = [single.median_ms, batch.median_ms];
        sides.insert(
            name.into(),
            serde_json::json!({
                "single_median_ms": single.median_ms,
                "single_p95_ms": single.p95_ms,
                "batch_median_ms": batch.median_ms,
                "batch_p95_ms": batch.p95_ms,
                "single_samples_ms": c.single[i],
                "batch_samples_ms": c.batch[i],
            }),
        );
    }
    let [[s0, b0], [s1, b1]] = medians;
    eprintln!(
        "  single {s0:.3} vs {s1:.3} ms ({:+.2}%), batch {b0:.2} vs {b1:.2} ms ({:+.2}%) [{} vs {}]",
        (s0 / s1 - 1.0) * 100.0,
        (b0 / b1 - 1.0) * 100.0,
        names[0],
        names[1],
    );
    Ok(serde_json::Value::Object(sides))
}

fn run_append(o: Options) -> hrxdb::Result<()> {
    use hrxdb::Corpus;
    let device = hrx::Device::open(0)?;
    let d = o.dimensions;
    let total = o.rows + o.append * o.appends;
    eprintln!(
        "Building {} x {d}, then {} appends of {} rows",
        o.rows, o.appends, o.append
    );
    let start = Instant::now();
    let mut corpus = Corpus::build(&device, d, (0..o.rows).map(|i| row(i, d)))?;
    let build_seconds = start.elapsed().as_secs_f64();
    let mut searcher = corpus.searcher_with_config(o.config)?;
    searcher.reserve_search(o.k)?;
    searcher.reserve_batch(APPEND_BATCH, o.k)?;
    let (mut append_ms, mut handover_ms, mut searcher_ms) = (Vec::new(), Vec::new(), Vec::new());
    let mut max_shards = 0;
    for step in 0..o.appends {
        let first = o.rows + step * o.append;
        let rows: Vec<_> = (first..first + o.append).map(|i| row(i, d)).collect();
        let start = Instant::now();
        corpus = corpus.append(&rows)?;
        append_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        let start = Instant::now();
        searcher.set_corpus(corpus.clone())?;
        handover_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        if step % 100 == 99 || (step + 1 == o.appends && searcher_ms.is_empty()) {
            let start = Instant::now();
            drop(corpus.searcher_with_config(o.config)?);
            searcher_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        max_shards = max_shards.max(corpus.shards().len());
    }
    let appended_shards = corpus.shards().len();
    let appended_memory = corpus.memory_usage();
    let append = distribution(append_ms.iter().copied())?;
    let handover = distribution(handover_ms.iter().copied())?;
    let fresh = distribution(searcher_ms.iter().copied())?;
    eprintln!(
        "append median {:.3} ms p95 {:.3} ms max {:.3} ms; set_corpus median {:.3} ms; new searcher median {:.3} ms; shards {appended_shards}",
        append.median_ms,
        append.p95_ms,
        append_ms.iter().copied().fold(0.0, f64::max),
        handover.median_ms,
        fresh.median_ms,
    );

    eprintln!("Building the reference: {total} x {d} in one build");
    let start = Instant::now();
    let reference = Corpus::build(&device, d, (0..total).map(|i| row(i, d)))?;
    let reference_seconds = start.elapsed().as_secs_f64();
    let mut single = reference.searcher_with_config(o.config)?;
    single.reserve_search(o.k)?;
    single.reserve_batch(APPEND_BATCH, o.k)?;
    for q in 0..4 {
        let query = row(total + 17 + q, d);
        if searcher.scores(&query)? != single.scores(&query)? {
            return Err(hrxdb::Error::Message(
                "appended scores differ from a single build".into(),
            ));
        }
    }
    eprintln!("Appended corpus vs single build:");
    let appended = compare([&mut searcher, &mut single], &o, total + 1_000, true)?;
    let appended_json = comparison_json(&appended, ["appended", "single_build"])?;

    // Consecutive IDs, as when one album's photos are replaced, then scattered
    // updates; the appended snapshot is still held, so both copy on write.
    let cluster: Vec<u32> = (0..o.updates as u32)
        .map(|i| total as u32 / 2 + i)
        .collect();
    let rows: Vec<_> = cluster
        .iter()
        .map(|&id| row(id as usize + 40_000_000, d))
        .collect();
    let start = Instant::now();
    let clustered = corpus.clone().update(&cluster, &rows)?;
    let update_clustered_ms = start.elapsed().as_secs_f64() * 1000.0;
    let clustered_shards = clustered.shards().len();
    let clustered_bytes = clustered.memory_usage().total() - appended_memory.total();
    eprintln!(
        "copy-on-write update of {} consecutive rows: {update_clustered_ms:.2} ms, {clustered_shards} shards, +{} MB",
        cluster.len(),
        clustered_bytes / 1_000_000
    );
    drop(clustered);

    // Scattered updates while the appended snapshot is still held.
    let mut ids: Vec<u32> = (0..o.updates)
        .map(|i| ((i as u64 * 2_654_435_761 + 12_345) % total as u64) as u32)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let replacement = |id: u32| row(id as usize + 50_000_000, d);
    let rows: Vec<_> = ids.iter().map(|&id| replacement(id)).collect();
    let start = Instant::now();
    let updated = corpus.clone().update(&ids, &rows)?;
    let update_copy_ms = start.elapsed().as_secs_f64() * 1000.0;
    let updated_shards = updated.shards().len();
    let updated_memory = updated.memory_usage();
    eprintln!(
        "copy-on-write update of {} rows: {update_copy_ms:.2} ms, {updated_shards} shards, +{} MB",
        ids.len(),
        (updated_memory.total() - appended_memory.total()) / 1_000_000
    );
    searcher.set_corpus(updated.clone())?;
    let query = row(total + 5, d);
    let scores = searcher.scores(&query)?;
    for &id in &ids {
        let reference = score_row(replacement(id), &query, true);
        if (scores[id as usize] as f64 - reference).abs() > 3e-6 {
            return Err(hrxdb::Error::Message(
                "updated score differs from CPU reference".into(),
            ));
        }
    }
    eprintln!("Updated (fragmented) corpus vs single build:");
    let fragmented = compare([&mut searcher, &mut single], &o, total + 9_000, false)?;
    let fragmented_json = comparison_json(&fragmented, ["updated", "single_build"])?;

    let start = Instant::now();
    let compact = updated.compact()?;
    let compact_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut compacted = compact.searcher_with_config(o.config)?;
    if compacted.scores(&query)? != scores {
        return Err(hrxdb::Error::Message("compaction changed scores".into()));
    }
    eprintln!(
        "compact: {compact_ms:.1} ms, {} shards",
        compact.shards().len()
    );

    // Exclusive handle: drop every other reference, then rewrite in place.
    drop(compacted);
    drop(corpus);
    searcher.set_corpus(compact.clone())?;
    drop(updated);
    drop(searcher);
    let memory = compact.memory_usage();
    let second: Vec<_> = ids
        .iter()
        .map(|&id| row(id as usize + 60_000_000, d))
        .collect();
    let start = Instant::now();
    let compact = compact.update(&ids, &second)?;
    let update_in_place_ms = start.elapsed().as_secs_f64() * 1000.0;
    if compact.memory_usage() != memory {
        return Err(hrxdb::Error::Message("exclusive update allocated".into()));
    }
    eprintln!(
        "in-place update of {} rows: {update_in_place_ms:.2} ms",
        ids.len()
    );

    let report = serde_json::json!({
        "target": device.target().as_str(),
        "base_rows": o.rows, "dimensions": d, "padded_dimensions": reference.padded_dimensions(),
        "append_rows": o.append, "appends": o.appends, "final_rows": total, "k": o.k,
        "batch": APPEND_BATCH, "config": o.config,
        "build_seconds": build_seconds, "reference_build_seconds": reference_seconds,
        "append_median_ms": append.median_ms, "append_p95_ms": append.p95_ms,
        "append_max_ms": append_ms.iter().copied().fold(0.0, f64::max),
        "set_corpus_median_ms": handover.median_ms, "set_corpus_p95_ms": handover.p95_ms,
        "new_searcher_median_ms": fresh.median_ms,
        "appended_shards": appended_shards, "max_shards_during_appends": max_shards,
        "reference_shards": reference.shards().len(),
        "appended_memory": appended_memory, "reference_memory": reference.memory_usage(),
        "search_after_appends": appended_json,
        "clustered_update_rows": cluster.len(), "clustered_update_copy_on_write_ms": update_clustered_ms,
        "clustered_update_shards": clustered_shards, "clustered_update_added_bytes": clustered_bytes,
        "updated_rows": ids.len(), "update_copy_on_write_ms": update_copy_ms,
        "updated_shards": updated_shards, "updated_memory": updated_memory,
        "search_after_updates": fragmented_json,
        "compact_ms": compact_ms, "update_in_place_ms": update_in_place_ms,
        "append_samples_ms": append_ms, "set_corpus_samples_ms": handover_ms,
        "timing": "Host wall time with explicit completion. Appends include host conversion, upload and tail moves. set_corpus includes allocation and scan compilation when a tail move changes capacity; new searchers compile kernels the preceding set_corpus already cached in-process. Searches: three warmups, changing queries, alternating which corpus runs first; single queries and 60-query batches, compilation and workspace reserved beforehand.",
        "validation": "Appended scores bitwise equal to a single build for four queries; every timed search result over the appended corpus equal to the single build's. Updated rows' scores checked against the quantized CPU reference; compaction checked bitwise against the updated snapshot.",
    });
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = o.output {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}
