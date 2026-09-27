//! Experimental GPU-independent NPU capability and retrieval measurements.
//! Run with `--features npu`; see docs/npu.md for limits and interpretation.
#[path = "support/npu.rs"]
mod npu;
use hrx::{Error, Result};
use std::time::Instant;

fn main() -> Result<()> {
    let mut rows = 1033usize;
    let mut dimensions = 129usize;
    let mut batch = 1usize;
    let mut k = 10usize;
    let mut samples = 10usize;
    let mut warmups = 3usize;
    let mut max_seconds = None;
    let mut binding_offset = 0;
    let mut output = None;
    let mut primitives = false;
    let mut direct_fp16 = false;
    let mut compare_gpu = false;
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        match key.as_str() {
            "--help" => {
                println!(
                    "npu_search_probe [--primitives [--binding-offset 0]] [--rows 1033] [--dimensions 129] [--batch 1] [--k 10] [--samples 10] [--warmups 3] [--max-seconds N] [--fp16] [--compare-gpu] [--output report.json]\nDefault path opens only the NPU. --compare-gpu explicitly enables a separate GPU reference. --max-seconds stops between completed chunks without cancelling native work."
                );
                return Ok(());
            }
            "--primitives" => {
                primitives = true;
                continue;
            }
            "--fp16" => {
                direct_fp16 = true;
                continue;
            }
            "--compare-gpu" => {
                compare_gpu = true;
                continue;
            }
            _ => {}
        }
        let value = args
            .next()
            .ok_or_else(|| Error::Message(format!("missing value for {key}")))?;
        if key == "--output" {
            output = Some(value);
            continue;
        }
        let value: usize = value
            .parse()
            .map_err(|_| Error::Message(format!("invalid value for {key}")))?;
        match key.as_str() {
            "--rows" => rows = value,
            "--dimensions" => dimensions = value,
            "--batch" => batch = value,
            "--k" => k = value,
            "--samples" => samples = value,
            "--warmups" => warmups = value,
            "--max-seconds" => max_seconds = Some(value),
            "--binding-offset" => binding_offset = value,
            _ => return Err(Error::Message(format!("unknown option {key}"))),
        }
    }
    let report = if primitives {
        if binding_offset == 0 {
            npu::primitives()?
        } else {
            npu::primitives_with_offset(binding_offset)?
        }
    } else {
        if samples == 0 || !(1..=64).contains(&batch) {
            return Err(Error::Message(
                "samples must be positive and batch in 1..=64".into(),
            ));
        }
        let t = Instant::now();
        let stored = npu::StoredRows::build(
            dimensions,
            (0..rows).map(|i| npu::generated_row(i, dimensions)),
            direct_fp16,
        )?;
        let host_encoding_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let mut search = npu::Search::new(&stored, k, None)?;
        let prepare_ms = t.elapsed().as_secs_f64() * 1000.0;
        search.deadline = max_seconds
            .map(|seconds| Instant::now() + std::time::Duration::from_secs(seconds as u64));
        eprintln!(
            "Prepared NPU-only {rows} x {dimensions}, batch {batch}, k {k}: {prepare_ms:.1} ms"
        );
        let queries: Vec<_> = (0..batch)
            .flat_map(|q| npu::generated_row(rows + 888 + q, dimensions))
            .collect();
        search
            .runtime
            .start_trace((rows.div_ceil(1024) * batch * 3).max(1))?;
        let actual = search.search(&queries, &[], true)?;
        let trace = search.runtime.finish_trace().unwrap();
        if trace
            .events
            .iter()
            .any(|event| event.lane != 3 || event.failed)
        {
            return Err(Error::Message("non-NPU or failed execution region".into()));
        }
        let max_error = npu::validate(&stored, &queries, &actual, &[], k)?;
        eprintln!(
            "Validation passed: max score error {max_error:e}; diagnostic search {:.1} ms",
            actual.timings.total_ms
        );
        let mut gpu = if compare_gpu {
            let device = hrxdb::Device::open(0)?;
            let corpus = hrxdb::Corpus::build_fp16(
                &device,
                dimensions,
                stored
                    .vectors
                    .chunks_exact(stored.padded * 2)
                    .take(rows)
                    .map(|r| &r[..dimensions * 2]),
            )?;
            let mut worker = corpus.searcher()?;
            worker.reserve_search(k)?;
            worker.reserve_batch(batch, k)?;
            Some(worker)
        } else {
            None
        };
        let gpu_overlap = if let Some(worker) = &mut gpu {
            let reference = worker.search_batch(&queries, k)?;
            let common: usize = reference
                .iter()
                .zip(&actual.neighbors)
                .map(|(a, b)| a.iter().filter(|v| b.iter().any(|w| w.id == v.id)).count())
                .sum();
            Some(
                serde_json::json!({"common_ids": common, "total_ids": reference.iter().map(Vec::len).sum::<usize>()}),
            )
        } else {
            None
        };
        let mut timings = Vec::new();
        let mut gpu_ms = Vec::new();
        for sample in 0..samples + warmups {
            let queries: Vec<_> = (0..batch)
                .flat_map(|q| npu::generated_row(rows + 888 + (sample + 1) * batch + q, dimensions))
                .collect();
            let t = Instant::now();
            let actual = search.search(&queries, &[], false)?;
            eprintln!(
                "sample {sample}: NPU {:.2} ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
            let reference_ms = if let Some(worker) = &mut gpu {
                let t = Instant::now();
                worker.search_batch(&queries, k)?;
                Some(t.elapsed().as_secs_f64() * 1000.0)
            } else {
                None
            };
            if sample >= warmups {
                timings.push(actual.timings);
                if let Some(ms) = reference_ms {
                    gpu_ms.push(ms);
                }
            }
        }
        let distribution = hrx::benchmark::Distribution::from_samples(
            timings.iter().map(|t| t.total_ms).collect(),
        )?;
        serde_json::json!({"hrx_version": "0.8.11", "rows": rows, "dimensions": dimensions, "batch": batch, "k": k,
            "direct_fp16": direct_fp16, "host_encoding_ms": host_encoding_ms, "prepare_ms": prepare_ms,
            "max_score_error": max_error, "top_k_matches_sort_of_npu_scores": true,
            "npu_only_trace": true, "trace_regions": trace.events.len(), "median_ms": distribution.median_ms,
            "p95_ms": distribution.p95_ms, "samples": timings, "gpu_samples_ms": gpu_ms, "gpu_top_k_overlap": gpu_overlap,
            "statistics": search.runtime.statistics(),
            "scope": "Host wall time with completion: sequential per-query NPU scan, select, merge, final readback; no compilation or ingestion. Scan/selection timings include submission and waits. Diagnostic scores read only during validation. CPU performs ingestion, query normalization and validation."})
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}
