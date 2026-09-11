//! Run explicitly with: cargo test --release --test gpu -- --ignored --test-threads=1
use half::f16;
use hrxdb::{FlatIndex, Neighbor, ScanConfig};

#[test]
#[ignore = "requires gfx1151"]
fn index_moves_to_worker_after_query() -> hrxdb::Result<()> {
    let device = hrxdb::Device::open(0)?;
    let mut db = FlatIndex::build(&device, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    assert_eq!(db.search(&[1.0, 0.0, 0.0], 1)?[0].id, 0);
    std::thread::spawn(move || -> hrxdb::Result<()> {
        assert_eq!(db.search(&[0.0, 1.0, 0.0], 1)?[0].id, 1);
        // Exercise synchronization, deregistration, and host deallocation on
        // a different thread from the one that created the index.
        drop(db);
        Ok(())
    })
    .join()
    .expect("query worker panicked")
}

fn row(i: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| {
            let mut x = (i as u32)
                .wrapping_mul(747796405)
                .wrapping_add(j as u32)
                .wrapping_add(2891336453);
            x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
            ((x >> 22) ^ x) as f64 as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect()
}

fn reference(rows: &[Vec<f32>], query: &[f32]) -> Vec<Neighbor> {
    let qnorm = query
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let q: Vec<f32> = query.iter().map(|&v| (v as f64 / qnorm) as f32).collect();
    let mut result: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(id, row)| {
            let length = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            let quantized: Vec<f32> = row
                .iter()
                .map(|&v| f16::from_f64(v as f64 / length).to_f32())
                .collect();
            let inverse = (1.0
                / quantized
                    .iter()
                    .map(|&v| (v as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()) as f32;
            let dot = quantized
                .iter()
                .zip(&q)
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum::<f64>();
            Neighbor {
                id: id as u32,
                similarity: (dot * inverse as f64) as f32,
            }
        })
        .collect();
    result.sort_by(|a, b| b.similarity.total_cmp(&a.similarity).then(a.id.cmp(&b.id)));
    result
}

#[test]
#[ignore = "requires gfx1151"]
fn all_schedules_match_cpu() -> hrxdb::Result<()> {
    let device = hrx::Device::open(0)?;
    for dimensions in [1, 127, 384, 769] {
        let rows: Vec<_> = (0..1057).map(|i| row(i, dimensions)).collect();
        let mut db = FlatIndex::build(&device, dimensions, &rows)?;
        let query = row(88_888, dimensions);
        let expected = reference(&rows, &query);
        for config in ScanConfig::configurations() {
            db.configure(config)?;
            let scores = db.scores(&query)?;
            for e in &expected {
                assert!(
                    (scores[e.id as usize] - e.similarity).abs() < 3e-6,
                    "dim={dimensions} config={config:?} id={} GPU={} CPU={}",
                    e.id,
                    scores[e.id as usize],
                    e.similarity
                );
            }
            for k in [1, 10, 32, 33, 1000, 1024] {
                let actual = db.search(&query, k)?;
                assert_eq!(
                    actual.iter().map(|n| n.id).collect::<Vec<_>>(),
                    expected[..k.min(expected.len())]
                        .iter()
                        .map(|n| n.id)
                        .collect::<Vec<_>>(),
                    "dim={dimensions}, config={config:?}, k={k}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn small_empty_invalid_and_ties() -> hrxdb::Result<()> {
    let device = hrx::Device::open(0)?;
    for n in [0, 1, 3, 31, 32, 33, 1023, 1024, 1025, 32769] {
        let rows = vec![vec![1.0, 0.0, 0.0]; n];
        let mut db = FlatIndex::build(&device, 3, &rows)?;
        assert_eq!(db.len(), n);
        for query in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for k in [1, 10, 32, 33, 1000, 1024] {
                let actual = db.search(&query, k)?;
                assert_eq!(actual.len(), n.min(k));
                for (id, neighbor) in actual.iter().enumerate() {
                    assert_eq!(neighbor.id, id as u32);
                    assert_eq!(neighbor.similarity, query[0]);
                }
            }
        }
        assert!(db.search(&[1.0, 0.0, 0.0], 0).is_err());
        assert!(db.search(&[1.0, 0.0, 0.0], 1025).is_err());
        assert!(db.search(&[0.0, 0.0, 0.0], 1).is_err());
        assert!(db.search(&[f32::NAN, 0.0, 0.0], 1).is_err());
        assert!(db.search(&[1.0], 1).is_err());
    }
    for bad in [vec![0.0; 3], vec![f32::INFINITY; 3], vec![1.0; 2]] {
        assert!(FlatIndex::build(&device, 3, [bad]).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn fp16_ingestion_and_reusable_scores() -> hrxdb::Result<()> {
    let device = hrxdb::Device::open(0)?;
    // Cross an ingestion chunk boundary and a selection partition boundary.
    let rows: Vec<[f32; 3]> = (0..16_389)
        .map(|i| match i % 4 {
            0 => [3.0, 4.0, 0.0],
            1 => [-6.0, 0.0, 8.0],
            2 => [0.0, 0.0, f16::from_bits(1).to_f32()],
            _ => [65504.0, -65504.0, 0.0],
        })
        .collect();
    let bytes: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect();
    let mut db = FlatIndex::build_fp16_with_config(
        &device,
        3,
        bytes.chunks_exact(6),
        ScanConfig::default(),
    )?;
    assert_eq!(db.len(), rows.len());
    assert_eq!(db.dimensions(), 3);
    assert_eq!(db.padded_dimensions(), 128);
    let mut output = vec![f32::NAN; db.len()];
    for query in [[1.0, 0.0, 0.0], [-1.0, 2.0, 3.0], [0.0, 0.0, -1.0]] {
        db.scores_into(&query, &mut output)?;
        assert_eq!(output, db.scores(&query)?);
        let qnorm = query
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        for (row, &score) in rows.iter().zip(&output) {
            let rnorm = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            let expected = row
                .iter()
                .zip(query)
                .map(|(&x, y)| x as f64 * y as f64)
                .sum::<f64>()
                / (qnorm * rnorm);
            assert!((score as f64 - expected).abs() < 3e-6);
        }
        let mut ranked: Vec<_> = output.iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then(a.0.cmp(&b.0)));
        for (actual, &(id, &similarity)) in db.search(&query, 32)?.iter().zip(&ranked) {
            assert_eq!(
                *actual,
                Neighbor {
                    id: id as u32,
                    similarity
                }
            );
        }
    }
    output.fill(42.0);
    for query in [vec![0.0; 3], vec![f32::NAN; 3], vec![1.0; 2]] {
        assert!(db.scores_into(&query, &mut output).is_err());
        assert!(output.iter().all(|&x| x == 42.0));
    }
    for size in [db.len() - 1, db.len() + 1] {
        let mut wrong = vec![42.0; size];
        assert!(db.scores_into(&[1.0, 0.0, 0.0], &mut wrong).is_err());
        assert!(wrong.iter().all(|&x| x == 42.0));
    }
    let mut empty = FlatIndex::build_fp16(&device, 3, std::iter::empty::<&[u8]>())?;
    empty.scores_into(&[1.0, 0.0, 0.0], &mut [])?;
    assert!(empty.scores(&[1.0, 0.0, 0.0])?.is_empty());
    assert!(empty.scores_into(&[0.0; 3], &mut []).is_err());
    for bad in [
        vec![0; 6],
        vec![0; 5],
        vec![0; 7],
        vec![0, 0x7c, 0, 0, 0, 0],
    ] {
        assert!(FlatIndex::build_fp16(&device, 3, [bad]).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151, allocates 7.8 GB"]
fn ten_million_rows_cross_four_gib() -> hrxdb::Result<()> {
    const N: usize = 10_000_000;
    const D: usize = 384;
    // Place unique unit-vector matches around 4 GiB and at the allocation tail.
    let crossing = (1usize << 32) / (D * 2);
    let special = [crossing - 1, crossing, crossing + 1, N - 1];
    let device = hrx::Device::open(0)?;
    let mut db = FlatIndex::build(
        &device,
        D,
        (0..N).map(|i| {
            let mut v = [0.0; D];
            let component = special.iter().position(|&r| r == i).map_or(0, |j| j + 1);
            v[component] = 1.0;
            v
        }),
    )?;
    for (j, &id) in special.iter().enumerate() {
        let mut query = [0.0; D];
        query[j + 1] = 1.0;
        let actual = db.search(&query, 10)?;
        assert_eq!(
            actual[0],
            Neighbor {
                id: id as u32,
                similarity: 1.0
            }
        );
        assert_eq!(actual[1].similarity, 0.0);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn exclusions_compose_with_large_k_and_reset_between_queries() -> hrxdb::Result<()> {
    let device = hrxdb::Device::open(0)?;
    let rows: Vec<_> = (0..4099).map(|i| row(i, 3)).collect();
    let mut db = FlatIndex::build(&device, 3, &rows)?;
    let query = [1.0, -2.0, 3.0];
    let scores = db.scores(&query)?;
    let mut ranked: Vec<_> = scores
        .iter()
        .enumerate()
        .map(|(id, &similarity)| Neighbor {
            id: id as u32,
            similarity,
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap()
            .then(a.id.cmp(&b.id))
    });
    let all: Vec<_> = (0..db.len() as u32).collect();
    let alternate: Vec<_> = all.iter().copied().step_by(2).collect();
    let few_remaining: Vec<_> = all[3..].to_vec();
    let across_words = [0, 7, 8, 31, 32, 33, 1023, 1024, 4098, 32, 0];
    for excluded in [&[][..], &across_words, &alternate, &all, &few_remaining] {
        let remaining: Vec<_> = ranked
            .iter()
            .copied()
            .filter(|n| !excluded.contains(&n.id))
            .collect();
        for k in [1, 10, 32, 33, 63, 255, 1000, 1024] {
            let actual = db.search_excluding(&query, k, excluded)?;
            assert_eq!(
                actual,
                remaining[..k.min(remaining.len())],
                "k={k} excluded={}",
                excluded.len()
            );
        }
        assert_eq!(db.search(&query, 1024)?, ranked[..1024]);
        assert_eq!(db.scores(&query)?, scores);
    }
    let mut visited = Vec::new();
    for _ in 0..40 {
        let q = visited
            .last()
            .map_or(query.as_slice(), |&id: &u32| rows[id as usize].as_slice());
        let next = db.search_excluding(q, 1, &visited)?[0].id;
        assert!(!visited.contains(&next));
        visited.push(next);
    }
    assert!(db.search_excluding(&query, 10, &[db.len() as u32]).is_err());
    assert!(db.search_excluding(&query, 10, &[u32::MAX]).is_err());
    assert!(db.search_excluding(&[0.0; 3], 10, &[0]).is_err());
    assert!(db.search_excluding(&query, 0, &[]).is_err());
    assert!(db.search_excluding(&query, 1025, &[]).is_err());
    assert_eq!(db.search(&query, 32)?, ranked[..32]);
    let mut empty = FlatIndex::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
    assert!(empty.search_excluding(&query, 1024, &[])?.is_empty());
    assert!(empty.search_excluding(&query, 1, &[0]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires gfx1151, allocates about 9.4 GB"]
fn faces_corpus_shards_past_two_to_32_elements() -> hrxdb::Result<()> {
    const N: usize = 8_942_135;
    const D: usize = 512;
    let boundary = (1usize << 32) / D;
    let special = [boundary - 1, boundary, boundary + 1, N - 1];
    let device = hrxdb::Device::open(0)?;
    let mut db = FlatIndex::build_fp16(
        &device,
        D,
        (0..N).map(|i| {
            let mut bytes = [0; D * 2];
            let component = special.iter().position(|&id| id == i).map_or(0, |j| j + 1);
            bytes[component * 2 + 1] = 0x3c; // FP16 1.0
            bytes
        }),
    )?;
    assert_eq!(db.shard_count(), 2);
    db.reserve_search(1024)?;
    for (j, &id) in special.iter().enumerate() {
        let mut query = [0.0; D];
        query[j + 1] = 1.0;
        let actual = db.search(&query, 1024)?;
        assert_eq!(actual.len(), 1024);
        assert_eq!(
            actual[0],
            Neighbor {
                id: id as u32,
                similarity: 1.0
            }
        );
        assert_eq!(
            actual[1],
            Neighbor {
                id: 0,
                similarity: 0.0
            }
        );
        let excluded = [id as u32, 0, 31, 32];
        let remaining = db.search_excluding(&query, 1024, &excluded)?;
        assert_eq!(remaining.len(), 1024);
        assert!(
            remaining
                .iter()
                .all(|n| n.similarity == 0.0 && !excluded.contains(&n.id))
        );
        assert_eq!(remaining[0].id, 1);
    }
    let mut query = [0.0; D];
    query[2] = 1.0;
    let scores = db.scores(&query)?;
    assert_eq!(scores[boundary], 1.0);
    assert_eq!(scores[boundary - 1], 0.0);
    assert_eq!(scores[N - 1], 0.0);
    Ok(())
}
