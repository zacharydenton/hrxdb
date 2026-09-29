//! Device-clock diagnostic replays and detailed compiler evidence.
use hrxdb::{Corpus, Device, ExecutionProfile, Neighbor, Result, Searcher};

fn row(id: usize, dimensions: usize) -> Vec<f32> {
    (0..dimensions)
        .map(|c| (((id * 17 + c * 13) % 97) as f32 - 48.0) / 49.0)
        .collect()
}

fn same_results(actual: &[Vec<Neighbor>], expected: &[Vec<Neighbor>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.id, expected.id);
            assert_eq!(actual.similarity.to_bits(), expected.similarity.to_bits());
        }
    }
}

fn check_profile(profile: &ExecutionProfile) {
    assert!(profile.device.frequency_hz > 0);
    assert!(profile.replay_host_ms > 0.0);
    assert!(profile.device.span_ms <= profile.replay_host_ms + 0.05);
    assert_eq!(profile.commands.len(), profile.device.intervals.len());
    assert!(!profile.commands.is_empty());
    let mut last = 0;
    let mut total = 0.0;
    for (command, interval) in profile.commands.iter().zip(&profile.device.intervals) {
        assert_eq!(command.label, interval.label);
        assert!(interval.start_tick >= last);
        assert!(interval.end_tick >= interval.start_tick);
        last = interval.end_tick;
        total += (interval.end_tick - interval.start_tick) as f64 * 1000.0
            / profile.device.frequency_hz as f64;
    }
    assert!((total - profile.device.interval_union_ms).abs() < 1e-6);
    assert!((profile.stages.values().map(|s| s.device_ms).sum::<f64>() - total).abs() < 1e-6);
    assert_eq!(
        profile.stages.values().map(|s| s.commands).sum::<usize>(),
        profile.commands.len()
    );
    assert!(
        (profile.device.interval_union_ms + profile.device.gaps_ms - profile.device.span_ms).abs()
            < 1e-6
    );
}

#[test]
#[ignore = "requires gfx1151 and native timestamp support"]
fn profile_matches_search_for_single_batch_large_k_and_exclusions() -> Result<()> {
    let device = Device::open(0)?;
    let mut db = Searcher::build(&device, 129, (0..2051).map(|i| row(i, 129)))?;
    for k in [10, 33, 1024] {
        for batch in [1, 3, 8, 33, 60] {
            for pass in 0..2 {
                let queries: Vec<_> = (0..batch)
                    .flat_map(|q| row(9000 + pass * 71 + q, 129))
                    .collect();
                let excluded = if pass == 0 {
                    vec![0, 0, 1023, 2050]
                } else {
                    vec![]
                };
                let expected = db.search_batch_excluding(&queries, k, &excluded)?;
                let actual = db.profile_search(&queries, k, &excluded)?;
                same_results(&actual.neighbors, &expected);
                let profile = actual.execution.unwrap();
                check_profile(&profile);
                assert!(profile.stages.contains_key(if batch == 1 {
                    "scan"
                } else {
                    "batch_scan"
                }));
                assert!(profile.stages.contains_key(if k <= 32 {
                    "select"
                } else {
                    "sort_select"
                }));
                assert!(profile.stages.contains_key("copy"));
                if batch > 1 {
                    assert_eq!(profile.stages["copy"].commands, 4);
                    for command in &profile.commands {
                        if matches!(
                            command.operation.as_str(),
                            "select" | "sort_select" | "sorted_merge"
                        ) {
                            assert_eq!(command.grid.unwrap()[1], batch as u32);
                        }
                    }
                }
                if batch == 1 && !excluded.is_empty() {
                    assert!(profile.stages.contains_key("mask_scores"));
                }
                // Diagnostic recording must not replace or poison ordinary execution.
                same_results(
                    &db.search_batch_excluding(&queries, k, &excluded)?,
                    &expected,
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and native timestamp support"]
fn profile_tracks_tiles_and_snapshot_changes_and_empty_work() -> Result<()> {
    let device = Device::open(0)?;
    let base = Corpus::build(&device, 3, (0..262_144).map(|i| row(i, 3)))?;
    let mut db = base.searcher()?;
    let appended = base.append((262_144..262_209).map(|i| row(i, 3)).collect::<Vec<_>>())?;
    assert!(appended.shards().len() > 1);
    let updated = appended
        .clone()
        .update(&[0, 262_208], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    for corpus in [appended, updated, base] {
        db.set_corpus(corpus)?;
        let queries: Vec<_> = (0..3).flat_map(|q| row(9999 + q, 3)).collect();
        let expected = db.search_batch(&queries, 10)?;
        let actual = db.profile_search(&queries, 10, &[])?;
        same_results(&actual.neighbors, &expected);
        let profile = actual.execution.unwrap();
        check_profile(&profile);
        assert_eq!(profile.stages["copy"].commands, 4);
        for command in &profile.commands {
            if command.operation == "candidate_merge" {
                assert_eq!(command.grid.unwrap()[0], 3);
            }
        }
        if db.len() > 262_144 {
            assert!(profile.stages["batch_scan"].commands > 1);
            assert!(profile.stages.contains_key("threshold_compact"));
            assert!(profile.stages.contains_key("candidate_merge"));
        }
    }
    let mut small = Searcher::build(&device, 3, [[1.0, 0.0, 0.0]])?;
    assert!(
        small
            .profile_search(&[1.0, 0.0, 0.0], 10, &[0, 0])?
            .execution
            .is_none()
    );
    assert!(small.profile_search(&[], 10, &[])?.execution.is_none());
    assert!(small.profile_search(&[0.0; 3], 10, &[]).is_err());
    assert!(small.profile_search(&[f32::NAN; 3], 10, &[]).is_err());
    assert!(small.profile_search(&[1.0; 2], 10, &[]).is_err());
    assert!(small.profile_search(&[1.0; 3], 0, &[]).is_err());
    assert!(small.profile_search(&[1.0; 3], 10, &[1]).is_err());
    let empty = Corpus::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
    small.set_corpus(empty)?;
    let actual = small.profile_search(&[1.0, 0.0, 0.0], 10, &[])?;
    assert_eq!(actual.neighbors, vec![vec![]]);
    assert!(actual.execution.is_none());
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and published Loom compiler"]
fn detailed_reports_identify_resources_waits_and_unchanged_code() -> Result<()> {
    let device = Device::open(0)?;
    let mut db = Searcher::build(&device, 129, (0..1033).map(|i| row(i, 129)))?;
    db.reserve_batch(3, 33)?;
    let detailed = db.detailed_compilation_reports()?;
    assert_eq!(detailed.len(), db.compilation_reports().len());
    for (ordinary, detailed) in db.compilation_reports().iter().zip(&detailed) {
        assert_eq!(ordinary.compiler_identity, detailed.compiler_identity);
        assert_eq!(ordinary.target, detailed.target);
        assert_eq!(ordinary.configuration, detailed.configuration);
        assert_eq!(detailed.report_mode, hrx::loom::ReportMode::Details);
        assert!(detailed.wait_reasons.is_some());
        assert!(!detailed.resources.is_empty());
        assert!(detailed.resources[0].vector_registers.is_some());
        assert_eq!(detailed.report.as_ref().unwrap()["schema_version"], 0);
        // Collecting additional compiler evidence must not change emitted code.
        assert_eq!(
            std::fs::read(&ordinary.artifact)?,
            std::fs::read(&detailed.artifact)?
        );
        let serialized = serde_json::to_value(detailed)?;
        assert!(serialized.get("source").is_none());
        assert!(serialized["report"]["wait_reason_summary_rows"].is_object());
    }
    Ok(())
}
