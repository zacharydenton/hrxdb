//! Opt-in interleaved comparisons of kernels and batch submission.
//! Both searchers share one corpus; run alone to measure GPU performance.
use super::*;
use std::time::Instant;

fn var(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| v.parse().expect("integer benchmark option"))
}

fn row(mut x: u32, dim: usize) -> Vec<f32> {
    x = x.wrapping_mul(747796405).wrapping_add(2891336453);
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as f32 / 8388608.0 - 1.0
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5
}

// Precision is inferred from the actual compiled scan, with explicit overrides
// for custom sources. Selection-only comparisons keep the production scan.
fn fp16_queries(option: &str, report: &Compilation, selection: bool) -> Result<bool> {
    match std::env::var(option) {
        Ok(value) => match value.as_str() {
            "fp16" => Ok(true),
            "fp32" => Ok(false),
            _ => Err(invalid("query precision must be fp16 or fp32")),
        },
        Err(std::env::VarError::NotPresent) if selection => Ok(true),
        Err(std::env::VarError::NotPresent) => report
            .report
            .as_ref()
            .and_then(|r| r.pointer("/static_instruction_mix/wmma_count"))
            .and_then(serde_json::Value::as_u64)
            .map(|count| count != 0)
            .ok_or_else(|| {
                invalid("missing scan instruction report; set query precision explicitly")
            }),
        Err(_) => Err(invalid("invalid query precision option")),
    }
}

fn unit_query(query: &[f32], half: bool) -> Vec<f64> {
    let length = norm(query, query.len()).unwrap();
    query
        .iter()
        .map(|&v| {
            let unit = (v as f64 / length) as f32;
            if half {
                f16::from_f32(unit).to_f64()
            } else {
                unit as f64
            }
        })
        .collect()
}

fn reference_score(id: u32, query: &[f64]) -> f64 {
    let source = row(id, query.len());
    let length = norm(&source, source.len()).unwrap();
    let stored: Vec<_> = source
        .iter()
        .map(|&v| f16::from_f64(v as f64 / length).to_f64())
        .collect();
    let inverse = (1.0 / stored.iter().map(|v| v * v).sum::<f64>().sqrt()) as f32;
    stored.iter().zip(query).map(|(x, y)| x * y).sum::<f64>() * inverse as f64
}

#[derive(Default)]
struct ComparisonChecks {
    neighbors: usize,
    rank_differences: usize,
    score_bit_differences: usize,
    max_rank_score_difference: f64,
    max_reference_error: [f64; 2],
}

impl ComparisonChecks {
    fn check(
        &mut self,
        rows: usize,
        dim: usize,
        k: usize,
        query: &[f32],
        output: &[Vec<Vec<Neighbor>>; 2],
        half: [bool; 2],
    ) {
        let batch = query.len() / dim;
        for arm in 0..2 {
            assert_eq!(output[arm].len(), batch);
            for (q, result) in output[arm].iter().enumerate() {
                assert_eq!(result.len(), k.min(rows));
                let unit = unit_query(&query[q * dim..(q + 1) * dim], half[arm]);
                let mut ids = std::collections::HashSet::new();
                for got in result {
                    assert!(
                        (got.id as usize) < rows && ids.insert(got.id),
                        "invalid or duplicate ID"
                    );
                    assert!(got.similarity.is_finite());
                    let error = (got.similarity as f64 - reference_score(got.id, &unit)).abs();
                    self.max_reference_error[arm] = self.max_reference_error[arm].max(error);
                    assert!(
                        error <= 3e-6,
                        "arm {arm}: score disagrees with CPU reference by {error}"
                    );
                }
                assert!(
                    result.windows(2).all(|w| w[0].similarity > w[1].similarity
                        || (w[0].similarity == w[1].similarity && w[0].id < w[1].id)),
                    "unsorted results"
                );
            }
        }
        for (a, b) in output[0].iter().flatten().zip(output[1].iter().flatten()) {
            let error = (a.similarity as f64 - b.similarity as f64).abs();
            self.max_rank_score_difference = self.max_rank_score_difference.max(error);
            self.rank_differences += usize::from(a.id != b.id);
            self.score_bit_differences +=
                usize::from(a.similarity.to_bits() != b.similarity.to_bits());
            if half[0] == half[1] {
                assert_eq!(a.id, b.id, "ranking mismatch");
                assert_eq!(
                    a.similarity.to_bits(),
                    b.similarity.to_bits(),
                    "score mismatch"
                );
            } else {
                // Sorting cannot increase a uniform per-row score perturbation.
                // IDs may swap only within this score envelope. Each returned
                // ID's own score was independently checked above.
                assert!(
                    error <= 2f64.powi(-11),
                    "rank score difference exceeds FP16-query bound: {error}"
                );
            }
            self.neighbors += 1;
        }
    }
}

#[test]
fn comparison_checks_validate_each_precision_and_reject_matching_corruption() {
    let query = row(888, 129);
    let outputs = [false, true].map(|half| {
        let unit = unit_query(&query, half);
        let mut neighbors: Vec<_> = (0..4)
            .map(|id| Neighbor {
                id,
                similarity: reference_score(id, &unit) as f32,
            })
            .collect();
        neighbors.sort_by(|a, b| b.similarity.total_cmp(&a.similarity).then(a.id.cmp(&b.id)));
        vec![neighbors]
    });
    ComparisonChecks::default().check(4, 129, 4, &query, &outputs, [false, true]);

    // Equal arms alone cannot establish correctness: reject the same wrong
    // score or duplicate ID in both, even though parity would pass.
    for duplicate in [false, true] {
        let mut bad = outputs[0].clone();
        if duplicate {
            bad[0][1] = bad[0][0];
        } else {
            bad[0][0].similarity += 0.01;
        }
        assert!(
            std::panic::catch_unwind(|| {
                ComparisonChecks::default().check(
                    4,
                    129,
                    4,
                    &query,
                    &[bad.clone(), bad],
                    [false, false],
                );
            })
            .is_err()
        );
    }
}

#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_scan() -> Result<()> {
    compare_batch_kernels(Comparison::Scan)
}

#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_selection() -> Result<()> {
    compare_batch_kernels(Comparison::Selection)
}

#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_replay() -> Result<()> {
    compare_batch_kernels(Comparison::Replay)
}

#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_optimized() -> Result<()> {
    compare_batch_kernels(Comparison::Combined)
}

#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_pipeline() -> Result<()> {
    compare_batch_kernels(Comparison::Pipeline)
}

#[derive(PartialEq)]
enum Comparison {
    Scan,
    Selection,
    Replay,
    Combined,
    Pipeline,
}

fn compare_batch_kernels(comparison: Comparison) -> Result<()> {
    let selection = matches!(comparison, Comparison::Selection | Comparison::Combined);
    let replay = matches!(comparison, Comparison::Replay | Comparison::Combined);
    let pipeline = comparison == Comparison::Pipeline;
    let fused_baseline = pipeline && var("BASELINE_FUSED", 0) != 0;
    let cached = pipeline || var("CACHED_GRAPHS", 0) != 0;
    let padded_baseline = var("PADDED_SELECTION_BASELINE", 0) != 0;
    let rows = var("ROWS", 6_909_092);
    let dim = var("DIM", 384);
    let batch = var("BATCH", 60);
    let k = var("K", 5);
    let samples = var("SAMPLES", 15);
    assert!(rows > 0 && samples > 0 && (2..=MAX_BATCH).contains(&batch));
    assert!(
        batch <= 64
            || selection
            || replay
            || pipeline
            || std::env::var_os("BASELINE_SOURCE").is_some(),
        "the historical scan supports up to 64 queries; use compare_batch_chunks or supply a compatible BASELINE_SOURCE"
    );
    let device = Device::open(0)?;
    let mut candidate = Searcher::build(&device, dim, (0..rows).map(|i| row(i as u32, dim)))?;
    let mut baseline = candidate.corpus().searcher()?;
    for db in [&mut candidate, &mut baseline] {
        db.reserve_batch(batch, k)?;
    }
    baseline.batch.as_mut().unwrap().immediate = !cached;
    candidate.batch.as_mut().unwrap().immediate = !(replay || cached);
    baseline.batch.as_mut().unwrap().padded_selection = padded_baseline;
    // A scan-only experiment must replace every tile's scan. A pipeline
    // comparison also retains the unfused reference's separate compaction.
    baseline.batch.as_mut().unwrap().separate_compact =
        comparison == Comparison::Scan || (pipeline && !fused_baseline);
    candidate.batch.as_mut().unwrap().separate_compact = comparison == Comparison::Scan;
    let width = batch.next_power_of_two().max(8);
    let source = |name: &str, default: &str| -> Result<String> {
        match std::env::var(name) {
            Ok(path) => Ok(std::fs::read_to_string(path)?),
            Err(_) => Ok(default.to_owned()),
        }
    };
    assert!(
        !selection || k <= 32,
        "selection comparison requires k <= 32"
    );
    let candidate_source = source(
        "CANDIDATE_SOURCE",
        if selection {
            kernels::SELECT
        } else {
            kernels::BATCH_SCAN
        },
    )?;
    let baseline_source = source(
        "BASELINE_SOURCE",
        if selection {
            include_str!("../tests/fixtures/select_two_reductions.loom")
        } else if replay || pipeline {
            kernels::BATCH_SCAN
        } else {
            include_str!("../tests/fixtures/batch_scan_columns.loom")
        },
    )?;
    let mut reports = Vec::new();
    let packed = [
        var(
            "BASELINE_PACKED_QUERIES",
            usize::from(baseline_source == kernels::BATCH_SCAN),
        ) != 0,
        var(
            "CANDIDATE_PACKED_QUERIES",
            usize::from(candidate_source == kernels::BATCH_SCAN),
        ) != 0,
    ];
    for (db, source, packed) in [
        (&mut baseline, &baseline_source, packed[0]),
        (&mut candidate, &candidate_source, packed[1]),
    ] {
        let mut arm = Vec::new();
        for first in [true, false]
            .into_iter()
            .take(if selection { 2 } else { 1 })
        {
            let mut spec = if selection {
                kernels::select_spec(first)
            } else {
                kernels::named_spec("batch_scan")
            };
            spec.set_report(hrx::loom::ReportMode::Details);
            spec.set_config("db.batch", width.to_string());
            if selection {
                spec.set_config("db.select.limit", TILE_ROWS.to_string());
            } else {
                spec.set_config("db.scan.dimensions", db.padded.to_string());
            }
            let (kernel, report) = compile(&db.compiler, &db.stream, source, spec)?;
            let plan = db
                .batch
                .as_mut()
                .unwrap()
                .plans
                .iter_mut()
                .find(|p| p.width == width)
                .unwrap();
            if selection {
                let plan = plan.selections.get_mut(&width).unwrap();
                if first {
                    plan.first = kernel;
                } else {
                    plan.merge = kernel;
                }
            } else {
                plan.scan = kernel;
                if !packed {
                    plan.query_pack = None;
                }
            }
            arm.push(report);
        }
        reports.push(arm);
    }
    let mut candidate_report = reports.pop().unwrap();
    let mut baseline_report = reports.pop().unwrap();
    if fused_baseline {
        let mut spec =
            kernels::named_spec("batch_scan_pruned").with_report(hrx::loom::ReportMode::Details);
        spec.set_config("db.batch", width.to_string());
        spec.set_config("db.scan.dimensions", baseline.padded.to_string());
        let (kernel, report) =
            compile(&baseline.compiler, &baseline.stream, &baseline_source, spec)?;
        baseline
            .batch
            .as_mut()
            .unwrap()
            .plans
            .iter_mut()
            .find(|p| p.width == width)
            .unwrap()
            .scan_pruned = kernel;
        baseline_report.push(report);
    }
    if pipeline {
        let threshold = source("BASELINE_THRESHOLD_SOURCE", kernels::THRESHOLD)?;
        for symbol in ["threshold_compact", "candidate_merge", "overflow_reduce"] {
            let spec =
                hrx::loom::Specialization::new(symbol).with_report(hrx::loom::ReportMode::Details);
            let (kernel, report) = compile(&baseline.compiler, &baseline.stream, &threshold, spec)?;
            let plan = baseline
                .batch
                .as_mut()
                .unwrap()
                .plans
                .iter_mut()
                .find(|p| p.width == width)
                .unwrap();
            match symbol {
                "threshold_compact" => plan.compact = kernel,
                "candidate_merge" => plan.candidate_merge = kernel,
                _ => plan.overflow_reduce = kernel,
            }
            baseline_report.push(report);
        }
        candidate_report.extend(
            candidate
                .detailed_compilation_reports()?
                .into_iter()
                .filter(|r| {
                    matches!(
                        r.symbol.as_str(),
                        "batch_scan_pruned"
                            | "batch_query_pack"
                            | "candidate_merge"
                            | "overflow_reduce"
                            | "threshold_compact"
                    )
                }),
        );
    }
    let mut times = [Vec::new(), Vec::new()];
    let half = [
        fp16_queries("BASELINE_QUERY_PRECISION", &baseline_report[0], selection)?,
        fp16_queries("CANDIDATE_QUERY_PRECISION", &candidate_report[0], selection)?,
    ];
    let scan_blocks: Vec<_> = [&baseline, &candidate]
        .iter()
        .map(|db| {
            db.batch
                .as_ref()
                .unwrap()
                .plans
                .iter()
                .find(|p| p.width == width)
                .unwrap()
                .scan
                .info()
                .workgroup_size
        })
        .collect();
    let mut observations = Vec::new();
    for iteration in 0..samples + 3 {
        let query: Vec<_> = (0..batch)
            .flat_map(|q| row((rows + 888 + iteration * batch + q) as u32, dim))
            .collect();
        let mut output = [Vec::new(), Vec::new()];
        for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
            let db = if index == 0 {
                &mut baseline
            } else {
                &mut candidate
            };
            let clock = Instant::now();
            output[index] = db.search_batch(&query, k)?;
            if iteration >= 3 {
                times[index].push(clock.elapsed().as_secs_f64() * 1000.0);
            }
        }
        observations.push((query, output));
    }
    // CPU and precision-aware validation happens after the complete ordinary
    // timing window, so it cannot become part of either arm's search latency.
    let mut checks = ComparisonChecks::default();
    for (query, output) in &observations {
        checks.check(rows, dim, k, query, output, half);
    }
    // Check every materialized score in the final tile, outside timing. Small
    // corpora fit in one tile, so this also supports complete matrix comparisons.
    let final_rows = (candidate.corpus.inner.shards.last().unwrap().count - 1)
        % candidate.batch.as_ref().unwrap().tile_rows
        + 1;
    let score_count = width * final_rows;
    let mut scores = [vec![0; score_count * 4], vec![0; score_count * 4]];
    for (db, bytes) in [&mut baseline, &mut candidate].into_iter().zip(&mut scores) {
        db.stream.read_blocking(
            db.batch
                .as_ref()
                .unwrap()
                .scores
                .try_slice(0, bytes.len())?,
            bytes,
        )?;
    }
    let mut matrix_bit_differences = 0;
    let mut max_matrix_score_difference = 0.0f64;
    for (at, (expected, actual)) in scores[0]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(scores[1].as_chunks::<4>().0)
        .enumerate()
    {
        matrix_bit_differences += usize::from(actual != expected);
        let a = f32::from_le_bytes(*actual);
        let b = f32::from_le_bytes(*expected);
        assert!(a.is_finite() && b.is_finite());
        let error = (a as f64 - b as f64).abs();
        max_matrix_score_difference = max_matrix_score_difference.max(error);
        if half[0] == half[1] {
            assert_eq!(actual, expected, "final tile score mismatch at {at}");
        } else {
            assert!(
                error <= 2f64.powi(-11),
                "final tile score error {error} at {at}"
            );
        }
    }
    let query: Vec<_> = (0..batch)
        .flat_map(|q| row((rows + 999 + q) as u32, dim))
        .collect();
    let mut profiles = [Vec::new(), Vec::new()];
    for iteration in 0..3 {
        for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
            let db = if index == 0 {
                &mut baseline
            } else {
                &mut candidate
            };
            let expected = db.search_batch(&query, k)?;
            let actual = db.profile_search(&query, k, &[])?;
            assert_eq!(actual.neighbors, expected);
            profiles[index].push(actual.execution.unwrap());
        }
    }
    let baseline_ms = median(&times[0]);
    let candidate_ms = median(&times[1]);
    // Preserve artifact hashes without exporting machine-specific cache paths.
    for compilation in baseline_report.iter_mut().chain(&mut candidate_report) {
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
    let report = serde_json::json!({
        "rows": rows, "dimensions": dim, "batch": batch, "compiled_width": width, "k": k,
        "warmups": 3, "samples": samples,
        "baseline_ms": baseline_ms, "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "compared_neighbors": checks.neighbors, "final_tile_scores_compared": score_count,
        "scores_bitwise_equal": checks.score_bit_differences == 0 && matrix_bit_differences == 0,
        "ids_equal": checks.rank_differences == 0,
        "rank_differences": checks.rank_differences,
        "returned_score_bit_differences": checks.score_bit_differences,
        "final_tile_score_bit_differences": matrix_bit_differences,
        "max_rank_score_difference": checks.max_rank_score_difference,
        "max_final_tile_score_difference": max_matrix_score_difference,
        "max_cpu_reference_error": checks.max_reference_error,
        "baseline_query_precision": if half[0] { "fp16" } else { "fp32" },
        "candidate_query_precision": if half[1] { "fp16" } else { "fp32" },
        "baseline_scan_workgroup": scan_blocks[0],
        "candidate_scan_workgroup": scan_blocks[1],
        "validation": "All returned scores versus FP64 CPU reference for each arm's query precision (3e-6); finite, unique, ordered IDs. Same precision requires exact IDs/score bits. Different precisions allow rank scores and every final-tile score to differ by at most 2^-11.",
        "baseline_samples_ms": times[0], "candidate_samples_ms": times[1],
        "kernel_family": if pipeline { "batch_pipeline" } else if selection { "select" } else { "batch_scan" },
        "baseline_submission": if cached { "cached_graph" } else { "immediate" },
        "candidate_submission": if replay || cached { "cached_graph" } else { "immediate" },
        "baseline_padded_selection": padded_baseline,
        "candidate_padded_selection": false,
        "baseline_compiler": baseline_report[0], "candidate_compiler": candidate_report[0],
        "baseline_compilers": baseline_report, "candidate_compilers": candidate_report,
        "baseline_source_digest": hrx::bundle::digest(baseline_source.as_bytes()),
        "candidate_source_digest": hrx::bundle::digest(candidate_source.as_bytes()),
        "baseline_profiles": profiles[0], "candidate_profiles": profiles[1],
        "timing": "Ordinary completed searches, alternating order on one corpus; three warmups. Separate profiled replays after all timed windows; instrumentation is not part of ordinary latency.",
    });
    println!(
        "baseline {baseline_ms:.3} ms, candidate {candidate_ms:.3} ms, speedup {:.3}x; {score_count} final-tile scores checked",
        baseline_ms / candidate_ms
    );
    if let Ok(path) = std::env::var("OUTPUT") {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

/// Compare production tile sizes on one shared corpus and workspace.
#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_tiles() -> Result<()> {
    let rows = var("ROWS", 6_909_092);
    let dim = var("DIM", 384);
    let batch = var("BATCH", 60);
    let k = var("K", 5);
    let samples = var("SAMPLES", 15);
    assert!(rows > 0 && samples > 0 && (2..=MAX_BATCH).contains(&batch));
    let device = Device::open(0)?;
    let mut db = Searcher::build(&device, dim, (0..rows).map(|i| row(i as u32, dim)))?;
    db.reserve_batch(batch, k)?;
    // This experiment deliberately includes tiles beyond the production cap.
    // Allocate the largest variant before timing so every dispatch fits.
    let old = db.batch.as_mut().unwrap();
    let mut scratch =
        BatchScratch::new(&db.stream, old.width, old.k, rows.min(TILE_ROWS), db.padded)?;
    scratch.plans = std::mem::take(&mut old.plans);
    db.batch = Some(scratch);
    db.batch.as_mut().unwrap().immediate = true;
    let tiles = [16_384, 32_768, 65_536, 131_072, 262_144];
    let mut times = vec![Vec::new(); tiles.len()];
    for iteration in 0..samples + 3 {
        let queries: Vec<_> = (0..batch)
            .flat_map(|q| row((rows + 888 + iteration * batch + q) as u32, dim))
            .collect();
        let mut reference: Option<Vec<Vec<Neighbor>>> = None;
        // Rotate first/last positions so drift is shared by every variant.
        for position in 0..tiles.len() {
            let index = (position + iteration) % tiles.len();
            db.batch.as_mut().unwrap().tile_rows = rows.min(tiles[index]);
            let clock = Instant::now();
            let got = db.search_batch(&queries, k)?;
            if iteration >= 3 {
                times[index].push(clock.elapsed().as_secs_f64() * 1000.0);
            }
            if let Some(expected) = &reference {
                assert_eq!(got.len(), expected.len());
                for (a, b) in got.iter().zip(expected) {
                    assert_eq!(a.len(), b.len());
                }
                for (a, b) in got.iter().flatten().zip(expected.iter().flatten()) {
                    assert_eq!(a.id, b.id, "tile size changed ranking");
                    assert_eq!(
                        a.similarity.to_bits(),
                        b.similarity.to_bits(),
                        "tile size changed score"
                    );
                }
            } else {
                reference = Some(got);
            }
        }
    }
    let variants: Vec<_> = tiles.iter().zip(&times).map(|(&tile_rows, samples_ms)| {
        let median_ms = median(samples_ms);
        println!("tile rows {tile_rows}: {median_ms:.3} ms");
        serde_json::json!({"tile_rows": tile_rows, "median_ms": median_ms, "samples_ms": samples_ms})
    }).collect();
    let report = serde_json::json!({
        "rows": rows, "dimensions": dim, "batch": batch, "k": k,
        "warmups": 3, "samples": samples, "variants": variants,
        "scores_bitwise_equal": true, "ids_equal": true,
        "timing": "Host completion; changing queries and rotating variant order on one corpus. Compilation and ingestion excluded. Same workspace capacity for every tile size. GPU clocks and other system activity were not controlled.",
    });
    if let Ok(path) = std::env::var("OUTPUT") {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

/// Compare a production wide batch with repeated production 64-query calls.
#[test]
#[ignore = "requires gfx1151; run alone to measure performance"]
fn compare_batch_chunks() -> Result<()> {
    let rows = var("ROWS", 1_000_000);
    let dim = var("DIM", 384);
    let batch = var("BATCH", 256);
    let k = var("K", 5);
    let samples = var("SAMPLES", 15);
    assert!(
        rows > 0 && samples > 0 && (128..=MAX_BATCH).contains(&batch) && batch.is_multiple_of(64)
    );
    let device = Device::open(0)?;
    let mut wide = Searcher::build(&device, dim, (0..rows).map(|i| row(i as u32, dim)))?;
    let mut chunks = wide.corpus().searcher()?;
    wide.reserve_batch(batch, k)?;
    chunks.reserve_batch(64, k)?;
    let mut times = [Vec::new(), Vec::new()];
    let mut observations = Vec::new();
    for iteration in 0..samples + 3 {
        let queries: Vec<_> = (0..batch)
            .flat_map(|q| row((rows + 888 + iteration * batch + q) as u32, dim))
            .collect();
        let mut outputs = [Vec::new(), Vec::new()];
        for arm in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
            let now = Instant::now();
            if arm == 0 {
                for query in queries.chunks_exact(64 * dim) {
                    outputs[arm].extend(chunks.search_batch(query, k)?);
                }
            } else {
                outputs[arm] = wide.search_batch(&queries, k)?;
            }
            if iteration >= 3 {
                times[arm].push(now.elapsed().as_secs_f64() * 1000.0);
            }
        }
        observations.push((queries, outputs));
    }
    let mut checks = ComparisonChecks::default();
    for (queries, outputs) in &observations {
        checks.check(rows, dim, k, queries, outputs, [true, true]);
    }
    let queries = &observations.last().unwrap().0;
    let mut profiles = [Vec::new(), Vec::new()];
    for iteration in 0..3 {
        for arm in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
            let db = if arm == 0 { &mut chunks } else { &mut wide };
            let mut stages = std::collections::BTreeMap::<String, StageProfile>::new();
            for query in queries.chunks(if arm == 0 { 64 * dim } else { batch * dim }) {
                let expected = db.search_batch(query, k)?;
                let actual = db.profile_search(query, k, &[])?;
                assert_eq!(actual.neighbors, expected);
                for (name, stage) in actual.execution.unwrap().stages {
                    let sum = stages.entry(name).or_default();
                    sum.commands += stage.commands;
                    sum.device_ms += stage.device_ms;
                }
            }
            profiles[arm].push(stages);
        }
    }
    let mut reports = [
        chunks.detailed_compilation_reports()?,
        wide.detailed_compilation_reports()?,
    ];
    for report in reports.iter_mut().flatten() {
        let path = std::path::Path::new(&report.artifact);
        report.artifact = format!(
            "hrx-cache/kernels/{}/{}",
            path.parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy(),
            path.file_name().unwrap().to_string_lossy()
        );
    }
    let baseline_ms = median(&times[0]);
    let candidate_ms = median(&times[1]);
    let report = serde_json::json!({
        "rows": rows, "dimensions": dim, "batch": batch, "k": k, "samples": samples, "warmups": 3,
        "baseline_ms": baseline_ms, "candidate_ms": candidate_ms, "speedup": baseline_ms / candidate_ms,
        "baseline_samples_ms": times[0], "candidate_samples_ms": times[1],
        "baseline_tile_rows": chunks.batch.as_ref().unwrap().tile_rows,
        "candidate_tile_rows": wide.batch.as_ref().unwrap().tile_rows,
        "baseline_workspace_bytes": chunks.batch_workspace_bytes(), "candidate_workspace_bytes": wide.batch_workspace_bytes(),
        "scores_bitwise_equal": checks.score_bit_differences == 0, "ids_equal": checks.rank_differences == 0,
        "compared_neighbors": checks.neighbors, "max_cpu_reference_error": checks.max_reference_error,
        "baseline_compilers": reports[0], "candidate_compilers": reports[1],
        "baseline_profiles": profiles[0], "candidate_profiles": profiles[1],
        "timing": "Alternating completed searches on one corpus with changing queries and cached graphs. Baseline reuses one 64-query worker. CPU checks, detailed compilation and three instrumented stage replays occur after ordinary timings. External GPU activity is uncontrolled."
    });
    println!(
        "64-query chunks {baseline_ms:.3} ms, batch {batch} {candidate_ms:.3} ms; {:.3}x",
        baseline_ms / candidate_ms
    );
    if let Ok(path) = std::env::var("OUTPUT") {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}
