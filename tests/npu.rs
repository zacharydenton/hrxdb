//! Opt-in NPU-only prototype qualification. Run with `--features npu --ignored`.
#![cfg(feature = "npu")]
#[path = "../examples/support/npu.rs"]
mod npu;
use hrx::{Result, residency::ResidencyManager};
use npu::{Search, StoredRows, generated_row, validate};

#[test]
fn validates_host_inputs_without_opening_hardware() {
    assert!(StoredRows::build(0, [[1.0]], false).is_err());
    assert!(StoredRows::build(1025, [vec![1.0; 1025]], false).is_err());
    assert!(StoredRows::build(1, [[0.0]], false).is_err());
    assert!(StoredRows::build(1, [[f32::NAN]], false).is_err());
    assert!(StoredRows::build(2, [[1.0]], false).is_err());
    assert!(StoredRows::build(1, [[f32::MAX]], true).is_err());
    assert!(StoredRows::build(1, [[f32::MIN_POSITIVE]], true).is_err());
}

#[test]
#[ignore = "requires published HRX compiler and Strix Halo NPU5"]
fn every_finite_half_and_fp32_primitives_are_exact() -> Result<()> {
    let report = npu::primitives()?;
    assert_eq!(report["finite_half_encodings"], 63_488);
    Ok(())
}

#[test]
#[ignore = "requires published HRX compiler and Strix Halo NPU5"]
fn tails_exclusions_replays_and_precisions() -> Result<()> {
    for direct in [false, true] {
        for d in [1, 3, 129, 384, 769, 1024] {
            let n = if d == 129 { 1033 } else { 17 };
            let stored = StoredRows::build(d, (0..n).map(|i| generated_row(i, d)), direct)?;
            let mut search = Search::new(&stored, 10, None)?;
            search.runtime.start_trace(256)?;
            for pass in 0..3 {
                let query = generated_row(12345 + pass, d);
                let excluded = if pass == 1 {
                    vec![0, 0, (n - 1) as u32]
                } else {
                    vec![]
                };
                let output = search.search(&query, &excluded, true)?;
                validate(&stored, &query, &output, &excluded, 10)?;
            }
            assert!(
                search
                    .runtime
                    .finish_trace()
                    .unwrap()
                    .events
                    .iter()
                    .all(|e| e.lane == 3 && !e.failed)
            );
            let stats = search.runtime.statistics();
            assert_eq!(stats.native_graphs_prepared, 0);
            assert_eq!(stats.copy_streams_created, 0);
            assert_eq!(stats.imports, 0);
            assert_eq!(stats.copied_bytes, 0);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires published HRX compiler and Strix Halo NPU5"]
fn all_batch_widths_and_large_k_ties() -> Result<()> {
    let stored = StoredRows::build(3, [[-1.0, 0.0, 0.0]; 17], false)?;
    for k in [1, 10, 32, 1024] {
        let mut search = Search::new(&stored, k, None)?;
        for batch in 1..=64 {
            let queries: Vec<_> = (0..batch).flat_map(|_| [1.0, 0.0, 0.0]).collect();
            let output = search.search(&queries, &[0, 16], true)?;
            validate(&stored, &queries, &output, &[0, 16], k)?;
            assert!(
                output
                    .neighbors
                    .iter()
                    .all(|row| row.iter().enumerate().all(|(i, v)| v.id as usize == i + 1))
            );
        }
    }
    // The running merge must also preserve all 1024 candidates across chunks.
    let stored = StoredRows::build(3, (0..1033).map(|i| generated_row(i, 3)), false)?;
    let mut search = Search::new(&stored, 1024, None)?;
    let query = [1.0, 0.0, 0.0];
    let output = search.search(&query, &[0, 1032], true)?;
    validate(&stored, &query, &output, &[0, 1032], 1024)?;
    Ok(())
}

#[test]
#[ignore = "requires published HRX compiler and Strix Halo NPU5"]
fn empty_invalid_extreme_and_budgeted() -> Result<()> {
    let empty = StoredRows::build(3, std::iter::empty::<[f32; 3]>(), false)?;
    let mut search = Search::new(&empty, 1, None)?;
    assert!(search.search(&[1.0, 0.0, 0.0], &[], false)?.neighbors[0].is_empty());
    assert!(search.search(&[], &[], false)?.neighbors.is_empty());
    assert!(search.search(&[0.0; 3], &[], false).is_err());
    assert!(search.search(&[f32::NAN; 3], &[], false).is_err());
    assert!(search.search(&[f32::INFINITY; 3], &[], false).is_err());
    assert!(search.search(&[1.0; 2], &[], false).is_err());
    assert!(search.search(&[1.0; 195], &[], false).is_err());
    assert!(search.search(&[1.0; 3], &[0], false).is_err());
    let tiny = half::f16::from_bits(1).to_f32();
    let rows = [
        [tiny, 0.0, 0.0],
        [65504.0, -65504.0, tiny],
        [-tiny, tiny, 0.0],
    ];
    let stored = StoredRows::build(3, rows, true)?;
    let manager = ResidencyManager::new(1_000_000)?;
    let budget = manager.budget();
    {
        let mut search = Search::new(&stored, 10, Some(budget.clone()))?;
        assert!(budget.reserved_bytes() > 0);
        for excluded in [vec![], vec![0, 1, 2], vec![]] {
            let query = [1.0, -0.5, 0.25];
            let output = search.search(&query, &excluded, true)?;
            validate(&stored, &query, &output, &excluded, 10)?;
        }
        search.deadline = Some(std::time::Instant::now());
        assert!(search.search(&[1.0, 0.0, 0.0], &[], false).is_err());
        search.deadline = None;
        assert_eq!(
            search.search(&[1.0, 0.0, 0.0], &[], false)?.neighbors[0].len(),
            3
        );
    }
    assert_eq!(budget.reserved_bytes(), 0);
    let small = ResidencyManager::new(128)?;
    assert!(Search::new(&stored, 10, Some(small.budget())).is_err());
    assert_eq!(small.budget().reserved_bytes(), 0);
    Ok(())
}
