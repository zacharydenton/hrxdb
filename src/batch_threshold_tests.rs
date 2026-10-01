use super::*;

#[test]
#[ignore = "requires gfx1151"]
fn threshold_capacity_boundaries_overflow_and_replay_match_cpu() -> Result<()> {
    let device = Device::open(0)?;
    // At/below/above capacity, a partial reduction block, and a full tile that
    // needs three reduction levels at k=1024. All scores are exactly -1/0/1.
    for (tile, survivors) in [
        (8192, 0),
        (8192, 1),
        (8192, 63),
        (8192, 64),
        (8192, 65),
        (8192, 4095),
        (8192, 4096),
        (8192, 4097),
        (8192, 8191),
        (TILE_ROWS, TILE_ROWS - 1),
    ] {
        let rows: Vec<_> = (0..2 * tile)
            .map(|i| {
                if i < tile {
                    [0.0, 0.0, 1.0]
                } else if i < tile + survivors {
                    [1.0, 0.0, 0.0]
                } else if i == tile + survivors {
                    [0.0, 1.0, 0.0]
                } else {
                    [0.0, 0.0, -1.0]
                }
            })
            .collect();
        let mut db = Searcher::build(&device, 3, &rows)?;
        db.reserve_batch(3, 1024)?;
        db.batch.as_mut().unwrap().tile_rows = tile;
        for k in [1, 5, 32, 33, 50, 64, 65, 1024] {
            // Reorder queries between replays of the same graph. Counts move
            // between overflowing, compacted and empty query slots.
            for axes in [[0, 1, 2], [2, 0, 1]] {
                let queries: Vec<_> = axes
                    .iter()
                    .flat_map(|&axis| {
                        let mut q = [0.0; 3];
                        q[axis] = 1.0;
                        q
                    })
                    .collect();
                let result = db.search_batch(&queries, k)?;
                for (&axis, actual) in axes.iter().zip(&result) {
                    let mut expected: Vec<_> = rows
                        .iter()
                        .enumerate()
                        .map(|(id, r)| Neighbor {
                            id: id as u32,
                            similarity: r[axis],
                        })
                        .collect();
                    expected.sort_by(|a, b| {
                        b.similarity
                            .partial_cmp(&a.similarity)
                            .unwrap()
                            .then(a.id.cmp(&b.id))
                    });
                    expected.truncate(k);
                    assert_eq!(
                        *actual, expected,
                        "tile={tile} survivors={survivors} k={k} axis={axis}"
                    );
                }
                let mut counts = [0; 12];
                db.stream.read_blocking(
                    db.batch.as_ref().unwrap().pruned_counts.try_slice(0, 12)?,
                    &mut counts,
                )?;
                assert_eq!(counts, [0; 12], "replay must reset every query's counter");
            }
        }
        // A fully excluded first tile leaves a -infinity threshold for every
        // query. Overflow must preserve masks, global IDs and deterministic ties.
        let excluded: Vec<_> = (0..tile as u32).collect();
        let queries = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = db.search_batch_excluding(&queries, 33, &excluded)?;
        for (axis, actual) in result.iter().enumerate() {
            let mut expected: Vec<_> = rows
                .iter()
                .enumerate()
                .skip(tile)
                .map(|(id, r)| Neighbor {
                    id: id as u32,
                    similarity: r[axis],
                })
                .collect();
            expected.sort_by(|a, b| {
                b.similarity
                    .partial_cmp(&a.similarity)
                    .unwrap()
                    .then(a.id.cmp(&b.id))
            });
            expected.truncate(33);
            assert_eq!(*actual, expected);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn overflow_preserves_varied_scores_across_reduction_blocks() -> Result<()> {
    let device = Device::open(0)?;
    let tile = 8192;
    let mut db = Searcher::build(
        &device,
        3,
        (0..tile * 2).map(|i| {
            if i < tile {
                [-1.0, 0.0, 0.0]
            } else {
                [1.0, ((i * 37) % 101) as f32, ((i * 53) % 97) as f32]
            }
        }),
    )?;
    db.reserve_batch(256, 1024)?;
    db.batch.as_mut().unwrap().tile_rows = tile;
    // Axis queries are represented exactly in FP16; their scores also match
    // the independent single-query scan and selection path bit for bit.
    let queries: Vec<_> = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        .into_iter()
        .cycle()
        .take(256 * 3)
        .collect();
    for k in [1, 5, 33, 1024] {
        let batch = db.search_batch(&queries, k)?;
        for (query, actual) in queries.as_chunks::<3>().0.iter().zip(batch) {
            assert_eq!(actual, db.search(query, k)?, "k={k}");
        }
    }
    Ok(())
}
