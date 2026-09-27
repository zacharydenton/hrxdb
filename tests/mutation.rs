//! Appended and updated snapshots against from-scratch builds of the same rows.
//! Run explicitly with: cargo test --release --test mutation -- --ignored --test-threads=1
use half::f16;
use hrx::residency::ResidencyManager;
use hrxdb::{Corpus, Device, Neighbor, Result, Searcher};

fn row(i: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| {
            let x = (i as u64 * 2_654_435_761 + j as u64 * 40_503) % 1_000_003;
            x as f32 / 1_000_003.0 - 0.5
        })
        .collect()
}

fn rows(range: std::ops::Range<usize>, d: usize) -> Vec<Vec<f32>> {
    range.map(|i| row(i, d)).collect()
}

fn fp16(row: &[f32]) -> Vec<u8> {
    row.iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Every score, a top-k and a batch of each corpus must be bitwise equal.
fn assert_same(actual: &Corpus, expected: &Corpus, d: usize) -> Result<()> {
    assert_eq!(actual.len(), expected.len());
    let mut a = actual.searcher()?;
    let mut e = expected.searcher()?;
    let queries: Vec<f32> = (0..9).flat_map(|q| row(1_000_000 + q, d)).collect();
    for query in queries.chunks_exact(d) {
        assert_eq!(a.scores(query)?, e.scores(query)?);
        assert_eq!(a.search(query, 33)?, e.search(query, 33)?);
    }
    assert_eq!(a.search_batch(&queries, 7)?, e.search_batch(&queries, 7)?);
    assert_shard_views(actual);
    Ok(())
}

/// Shards partition the IDs in order; bindings cover 256-row capacities.
fn assert_shard_views(corpus: &Corpus) {
    let mut next = 0;
    let mut end = 0;
    let stride = corpus.padded_dimensions() * 2;
    for shard in corpus.shards() {
        assert_eq!(shard.row_range().start, next);
        assert!(!shard.row_range().is_empty());
        next = shard.row_range().end;
        assert!(shard.capacity_rows() >= shard.row_range().len());
        assert_eq!(shard.capacity_rows() % 256, 0);
        assert_eq!(shard.vectors().len(), shard.capacity_rows() * stride);
        assert_eq!(shard.inverse_norms().len(), shard.capacity_rows() * 4);
        assert_eq!(shard.vectors().offset() % (256 * stride), 0);
        end = end.max(shard.capacity_range().end);
    }
    assert_eq!(next, corpus.len());
    assert_eq!(corpus.capacity_range(), 0..end);
}

#[test]
#[ignore = "requires gfx1151"]
fn appended_rows_score_exactly_as_built_rows() -> Result<()> {
    let device = Device::open(0)?;
    let d = 129;
    let all = rows(0..3_000, d);
    for direct_fp16 in [false, true] {
        let build = |rows: &[Vec<f32>]| -> Result<Corpus> {
            if direct_fp16 {
                Corpus::build_fp16(&device, d, rows.iter().map(|r| fp16(r)))
            } else {
                Corpus::build(&device, d, rows)
            }
        };
        let append = |corpus: &Corpus, rows: &[Vec<f32>]| -> Result<Corpus> {
            if direct_fp16 {
                corpus.append_fp16(rows.iter().map(|r| fp16(r)))
            } else {
                corpus.append(rows)
            }
        };
        let base = build(&all[..1_000])?;
        let mut old = base.searcher()?;
        let query = row(7, d);
        let before = old.scores(&query)?;
        // Fill the base reserve, spill into a tail, then grow the tail.
        let mut corpus = base.clone();
        for range in [1_000..1_100, 1_100..1_300, 1_300..3_000] {
            corpus = append(&corpus, &all[range])?;
        }
        assert_same(&corpus, &build(&all)?, d)?;
        assert_eq!(corpus.shards().len(), 2);
        // The original snapshot and its searcher still see exactly their rows.
        assert_eq!(base.len(), 1_000);
        assert_eq!(old.scores(&query)?, before);
        assert_same(&base, &build(&all[..1_000])?, d)?;
        assert_eq!(corpus.append(std::iter::empty::<Vec<f32>>())?.len(), 3_000);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn appends_from_one_snapshot_do_not_see_each_other() -> Result<()> {
    let device = Device::open(0)?;
    let d = 3;
    let base = Corpus::build(&device, d, rows(0..100, d))?;
    let left = base.append(rows(100..150, d))?;
    // The base's reserve now holds `left`'s rows; this append must not reuse it.
    let right = base.append(rows(500..560, d))?;
    let left_again = left.append(rows(150..170, d))?;
    let right_rows: Vec<_> = rows(0..100, d)
        .into_iter()
        .chain(rows(500..560, d))
        .collect();
    assert_same(&left, &Corpus::build(&device, d, rows(0..150, d))?, d)?;
    assert_same(&right, &Corpus::build(&device, d, &right_rows)?, d)?;
    assert_same(&left_again, &Corpus::build(&device, d, rows(0..170, d))?, d)?;
    assert_same(&base, &Corpus::build(&device, d, rows(0..100, d))?, d)?;
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn many_small_appends_keep_one_tail() -> Result<()> {
    let device = Device::open(0)?;
    let d = 64;
    let mut corpus = Corpus::build(&device, d, rows(0..10_000, d))?;
    let mut kept = Vec::new();
    for step in 0..400 {
        let start = 10_000 + step * 50;
        corpus = corpus.append(rows(start..start + 50, d))?;
        assert!(corpus.shards().len() <= 2, "step {step}");
        if [7, 150].contains(&step) {
            kept.push((corpus.clone(), corpus.searcher()?.scores(&row(3, d))?));
        }
    }
    assert_eq!(corpus.len(), 30_000);
    assert_same(&corpus, &Corpus::build(&device, d, rows(0..30_000, d))?, d)?;
    // The tail's reserve is at most its own size, not the corpus's.
    let slack = corpus.memory_usage().vector_slack / (corpus.padded_dimensions() * 2);
    assert!(slack <= 20_000 + 256, "slack rows {slack}");
    for (snapshot, scores) in kept {
        assert_eq!(snapshot.searcher()?.scores(&row(3, d))?, scores);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn updates_change_only_their_rows_and_leave_old_snapshots() -> Result<()> {
    let device = Device::open(0)?;
    let d = 3;
    let n = 1_000_000;
    let mut expected = rows(0..n, d);
    let base = Corpus::build(&device, d, &expected)?;
    let mut old = base.searcher()?;
    let query = row(11, d);
    let before = old.scores(&query)?;
    // Edges of adjacent pages, two pages less than a tile apart (copied as
    // one run with the gap between them), and the last row.
    let ids = [16_383u32, 0, 16_384, 1, 500_000, 600_000, 999_999];
    let replacements: Vec<_> = ids.iter().map(|&i| row(900_000 + i as usize, d)).collect();
    let updated = base.clone().update(&ids, &replacements)?;
    for (&id, replacement) in ids.iter().zip(&replacements) {
        expected[id as usize] = replacement.clone();
    }
    assert_same(&updated, &Corpus::build(&device, d, &expected)?, d)?;
    // Copy-on-write: three page runs and the two untouched pieces between.
    assert_eq!(updated.shards().len(), 5);
    assert_eq!(old.scores(&query)?, before);
    let after = updated.searcher()?.scores(&query)?;
    for (id, (old, new)) in before.iter().zip(&after).enumerate() {
        if !ids.contains(&(id as u32)) {
            assert_eq!(old, new, "row {id}");
        }
    }
    assert!(
        ids.iter()
            .any(|&id| before[id as usize] != after[id as usize])
    );

    // An exclusive handle is rewritten in place: no allocation, same shards.
    drop(old);
    drop(base);
    let memory = updated.memory_usage();
    let shards = updated.shards().len();
    let replacement = row(123_456, d);
    let updated = updated.update(&[200_000], [&replacement])?;
    expected[200_000] = replacement;
    assert_eq!(updated.memory_usage(), memory);
    assert_eq!(updated.shards().len(), shards);
    assert_same(&updated, &Corpus::build(&device, d, &expected)?, d)?;

    // Compaction restores the build layout with identical scores.
    let compact = updated.compact()?;
    assert_eq!(compact.shards().len(), 1);
    assert_eq!(
        compact.memory_usage(),
        Corpus::build(&device, d, &expected)?.memory_usage()
    );
    assert_same(&compact, &Corpus::build(&device, d, &expected)?, d)?;
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn updates_and_appends_compose_at_the_tail() -> Result<()> {
    let device = Device::open(0)?;
    let d = 17;
    let mut expected = rows(0..5_000, d);
    let mut corpus = Corpus::build(&device, d, &expected)?;
    let mut held = Vec::new();
    for step in 0..30 {
        let start = expected.len();
        let fresh = rows(start..start + 40, d);
        corpus = corpus.append(&fresh)?;
        expected.extend(fresh);
        // Replace a recent row and an old one while an old snapshot is held.
        let ids = [start as u32 + 3, (step * 97) as u32];
        let replacements: Vec<_> = ids.iter().map(|&i| row(i as usize + 77_777, d)).collect();
        held.push(corpus.clone());
        corpus = corpus.update(&ids, &replacements)?;
        for (&id, replacement) in ids.iter().zip(replacements) {
            expected[id as usize] = replacement;
        }
    }
    // Updates copy the pages they touch; the tail keeps absorbing appends.
    assert_eq!(corpus.shards().len(), 2);
    assert_same(&corpus, &Corpus::build(&device, d, &expected)?, d)?;
    assert_eq!(held[0].len(), 5_040);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn a_searcher_moves_between_snapshots() -> Result<()> {
    let device = Device::open(0)?;
    let d = 33;
    let base = Corpus::build(&device, d, rows(0..700, d))?;
    let mut searcher = base.searcher()?;
    let queries: Vec<f32> = (0..20).flat_map(|q| row(50_000 + q, d)).collect();
    // Reserve batch storage for the small corpus before it grows.
    searcher.search_batch(&queries, 9)?;
    let grown = base.append(rows(700..5_000, d))?;
    let excluded = [3, 699, 700, 4_999];
    searcher.set_corpus(grown.clone())?;
    let mut fresh = grown.searcher()?;
    assert_eq!(searcher.len(), 5_000);
    assert_eq!(searcher.shard_count(), 2);
    assert_eq!(
        searcher.search_batch(&queries, 9)?,
        fresh.search_batch(&queries, 9)?
    );
    assert_eq!(
        searcher.search_batch_excluding(&queries, 40, &excluded)?,
        fresh.search_batch_excluding(&queries, 40, &excluded)?
    );
    for query in queries.chunks_exact(d) {
        assert_eq!(
            searcher.search_excluding(query, 5, &excluded)?,
            fresh.search_excluding(query, 5, &excluded)?
        );
        assert_eq!(searcher.scores(query)?, fresh.scores(query)?);
    }
    // Moving back to a smaller snapshot also works.
    searcher.set_corpus(base.clone())?;
    let mut original: Searcher = base.searcher()?;
    assert_eq!(
        searcher.search_batch(&queries, 9)?,
        original.search_batch(&queries, 9)?
    );
    let other = Corpus::build(&device, d + 1, rows(0..10, d + 1))?;
    assert!(searcher.set_corpus(other).is_err());
    assert_eq!(
        searcher.search(&queries[..d], 1)?,
        original.search(&queries[..d], 1)?
    );
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn invalid_input_changes_nothing() -> Result<()> {
    let device = Device::open(0)?;
    let base = Corpus::build(&device, 3, rows(0..10, 3))?;
    let mut searcher = base.searcher()?;
    let before = searcher.scores(&[1.0, 2.0, 3.0])?;
    for bad in [
        vec![f32::NAN, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 2.0],
        vec![f32::INFINITY, 1.0, 1.0],
    ] {
        assert!(base.append([vec![1.0, 1.0, 1.0], bad.clone()]).is_err());
        assert!(base.clone().update(&[4], [bad]).is_err());
    }
    assert!(base.append_fp16([vec![0u8; 6]]).is_err());
    assert!(base.append_fp16([vec![0u8; 5]]).is_err());
    let one = [[1.0f32, 0.0, 0.0]];
    assert!(base.clone().update(&[10], one).is_err());
    assert!(base.clone().update(&[1, 1], [one[0], one[0]]).is_err());
    assert!(base.clone().update(&[1, 2], one).is_err());
    assert_eq!(searcher.scores(&[1.0, 2.0, 3.0])?, before);
    // A failed append's claim does not stop the next one.
    let appended = base.append(rows(10..20, 3))?;
    assert_same(&appended, &Corpus::build(&device, 3, rows(0..20, 3))?, 3)?;
    let empty = Corpus::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
    assert_same(
        &empty.append(rows(0..5, 3))?,
        &Corpus::build(&device, 3, rows(0..5, 3))?,
        3,
    )?;
    let unchanged: Vec<Neighbor> = base
        .clone()
        .update(&[], std::iter::empty::<[f32; 3]>())?
        .searcher()?
        .search(&[1.0, 0.0, 0.0], 3)?;
    assert_eq!(unchanged, searcher.search(&[1.0, 0.0, 0.0], 3)?);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn mutations_use_each_handles_budget() -> Result<()> {
    let device = Device::open(0)?;
    // Initialize the shared writer without a budget and leave no tile slack.
    let base = Corpus::build(&device, 3, rows(0..256, 3))?.compact()?;
    let small = ResidencyManager::new(1024)?;
    let limited = base.clone().with_workspace_budget(small.budget());
    assert!(limited.append([[1.0, 0.0, 0.0]]).is_err());
    assert_eq!(small.budget().reserved_bytes(), 0);

    // A sibling without a budget must not inherit the failed call's ceiling.
    let unbudgeted = base.append([[1.0, 0.0, 0.0]])?;
    let tail_bytes = unbudgeted.memory_usage().total() - base.memory_usage().total();
    assert!(tail_bytes > 1024);
    let first = ResidencyManager::new(4 * tail_bytes)?;
    let second = ResidencyManager::new(4 * tail_bytes)?;
    let a = base
        .clone()
        .with_workspace_budget(first.budget())
        .append([[0.0, 1.0, 0.0]])?;
    assert_eq!(first.budget().reserved_bytes(), tail_bytes);
    let b = base
        .clone()
        .with_workspace_budget(second.budget())
        .append([[0.0, 0.0, 1.0]])?;
    assert_eq!(first.budget().reserved_bytes(), tail_bytes);
    assert_eq!(second.budget().reserved_bytes(), tail_bytes);
    let c = base.append([[1.0, 1.0, 0.0]])?;
    assert_eq!(first.budget().reserved_bytes(), tail_bytes);
    assert_eq!(second.budget().reserved_bytes(), tail_bytes);
    drop((a, b, c));
    assert_eq!(first.budget().reserved_bytes(), 0);
    assert_eq!(second.budget().reserved_bytes(), 0);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn updates_reserve_conversion_before_consuming_rows() -> Result<()> {
    use std::cell::Cell;
    let device = Device::open(0)?;
    let base = Corpus::build(&device, 3, rows(0..256, 3))?;
    let ids = [0, 1, 2, 3];
    let conversion_bytes =
        ids.len() * (base.padded_dimensions() * 2 + 4) + base.padded_dimensions() * 2;
    let small = ResidencyManager::new(conversion_bytes - 1)?;
    let consumed = Cell::new(0);
    let replacements = || {
        ids.iter().map(|_| {
            consumed.set(consumed.get() + 1);
            [1.0, 0.0, 0.0]
        })
    };
    assert!(
        base.clone()
            .with_workspace_budget(small.budget())
            .update(&ids, replacements())
            .is_err()
    );
    assert_eq!(consumed.get(), 0);
    assert_eq!(small.budget().reserved_bytes(), 0);

    // Conversion fits, but GPU staging does not; the host charge rolls back.
    let staging_limit = ResidencyManager::new(conversion_bytes)?;
    assert!(
        base.clone()
            .with_workspace_budget(staging_limit.budget())
            .update(&ids, replacements())
            .is_err()
    );
    assert_eq!(consumed.get(), ids.len());
    assert_eq!(staging_limit.budget().reserved_bytes(), 0);

    let enough = ResidencyManager::new(1_000_000)?;
    let budget = enough.budget();
    let corpus = base.with_workspace_budget(budget.clone());
    // An invalid row also releases the reservation.
    assert!(corpus.clone().update(&[0], [[0.0; 3]]).is_err());
    assert_eq!(budget.reserved_bytes(), 0);
    let replacements = ids.iter().map(|_| {
        assert_eq!(budget.reserved_bytes(), conversion_bytes);
        fp16(&[1.0, 0.0, 0.0])
    });
    let updated = corpus.update_fp16(&ids, replacements)?;
    assert_eq!(budget.reserved_bytes(), 0);
    let scores = updated.searcher()?.scores(&[1.0, 0.0, 0.0])?;
    assert_eq!(&scores[..ids.len()], &[1.0; 4]);
    Ok(())
}
