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
            for k in [1, 10, 32] {
                let actual = db.search(&query, k)?;
                assert_eq!(
                    actual.iter().map(|n| n.id).collect::<Vec<_>>(),
                    expected[..k].iter().map(|n| n.id).collect::<Vec<_>>(),
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
            for k in [1, 10, 32] {
                let actual = db.search(&query, k)?;
                assert_eq!(actual.len(), n.min(k));
                for (id, neighbor) in actual.iter().enumerate() {
                    assert_eq!(neighbor.id, id as u32);
                    assert_eq!(neighbor.similarity, query[0]);
                }
            }
        }
        assert!(db.search(&[1.0, 0.0, 0.0], 0).is_err());
        assert!(db.search(&[1.0, 0.0, 0.0], 33).is_err());
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
