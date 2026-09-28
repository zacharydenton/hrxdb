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

#[derive(PartialEq)]
enum Comparison {
    Scan,
    Selection,
    Replay,
    Combined,
}

fn compare_batch_kernels(comparison: Comparison) -> Result<()> {
    let selection = matches!(comparison, Comparison::Selection | Comparison::Combined);
    let replay = matches!(comparison, Comparison::Replay | Comparison::Combined);
    let rows = var("ROWS", 6_909_092);
    let dim = var("DIM", 384);
    let batch = var("BATCH", 60);
    let k = var("K", 5);
    let samples = var("SAMPLES", 15);
    assert!(rows > 0 && samples > 0 && (2..=64).contains(&batch));
    let device = Device::open(0)?;
    let mut candidate = Searcher::build(&device, dim, (0..rows).map(|i| row(i as u32, dim)))?;
    let mut baseline = candidate.corpus().searcher()?;
    for db in [&mut candidate, &mut baseline] {
        db.reserve_batch(batch, k)?;
    }
    baseline.batch.as_mut().unwrap().immediate = true;
    candidate.batch.as_mut().unwrap().immediate = !replay;
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
        } else if replay {
            kernels::BATCH_SCAN
        } else {
            include_str!("../tests/fixtures/batch_scan_columns.loom")
        },
    )?;
    let mut reports = Vec::new();
    for (db, source) in [
        (&mut baseline, &baseline_source),
        (&mut candidate, &candidate_source),
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
            }
            arm.push(report);
        }
        reports.push(arm);
    }
    let mut candidate_report = reports.pop().unwrap();
    let mut baseline_report = reports.pop().unwrap();
    let mut times = [Vec::new(), Vec::new()];
    let mut compared_neighbors = 0;
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
        assert_eq!(output[0].len(), batch);
        assert_eq!(output[1].len(), batch);
        for (expected, actual) in output[0].iter().zip(&output[1]) {
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!(a.similarity.is_finite());
                assert_eq!(a.id, b.id, "ranking mismatch");
                assert_eq!(
                    a.similarity.to_bits(),
                    b.similarity.to_bits(),
                    "score mismatch"
                );
                compared_neighbors += 1;
            }
        }
    }
    // Check every materialized score in the final tile, outside timing. Small
    // corpora fit in one tile, so this also supports complete matrix comparisons.
    let final_rows = (candidate.corpus.inner.shards.last().unwrap().count - 1) % TILE_ROWS + 1;
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
    for (at, (expected, actual)) in scores[0]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(scores[1].as_chunks::<4>().0)
        .enumerate()
    {
        assert_eq!(actual, expected, "final tile score mismatch at {at}");
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
        "compared_neighbors": compared_neighbors, "final_tile_scores_compared": score_count,
        "scores_bitwise_equal": true, "ids_equal": true,
        "baseline_samples_ms": times[0], "candidate_samples_ms": times[1],
        "kernel_family": if selection { "select" } else { "batch_scan" },
        "baseline_submission": "immediate",
        "candidate_submission": if replay { "cached_graph" } else { "immediate" },
        "baseline_compiler": baseline_report[0], "candidate_compiler": candidate_report[0],
        "baseline_compilers": baseline_report, "candidate_compilers": candidate_report,
        "baseline_source_digest": hrx::bundle::digest(baseline_source.as_bytes()),
        "candidate_source_digest": hrx::bundle::digest(candidate_source.as_bytes()),
        "baseline_profiles": profiles[0], "candidate_profiles": profiles[1],
        "timing": "Ordinary completed searches, alternating order on one corpus; three warmups. Separate profiled replays after all timed windows; instrumentation is not part of ordinary latency.",
    });
    println!(
        "baseline {baseline_ms:.3} ms, candidate {candidate_ms:.3} ms, speedup {:.3}x; {score_count} final-tile scores bitwise equal",
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
    assert!(rows > 0 && samples > 0 && (2..=64).contains(&batch));
    let device = Device::open(0)?;
    let mut db = Searcher::build(&device, dim, (0..rows).map(|i| row(i as u32, dim)))?;
    db.reserve_batch(batch, k)?;
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
