//! Device composition and shared ownership tests.
use hrxdb::{Corpus, Device, DeviceNeighbors, ScoreBatch, TopK};

#[test]
fn shared_corpus_is_send_sync() {
    fn traits<T: Send + Sync + Clone>() {}
    traits::<Corpus>();
}

#[test]
#[ignore = "requires gfx1151"]
fn standalone_topk_handles_batches_ties_missing_and_reuse() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    let mut topk = TopK::new(&stream)?;
    for n in [0usize, 3, 2057] {
        let batch = 3;
        let values: Vec<f32> = (0..batch * n)
            .map(|i| match i % 17 {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                2 => f32::INFINITY,
                _ => ((i * 13) % 31) as f32 - 15.0,
            })
            .collect();
        let buffer = stream.allocate(values.len() * 4)?;
        stream.upload(
            buffer.binding(),
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        for k in [1, 5, 33, 1024] {
            let mut output = DeviceNeighbors::new(&stream, batch, k)?;
            let _completion = topk.select(
                &mut stream,
                ScoreBatch::new(buffer.binding(), batch, n)?,
                &mut output,
            )?;
            let actual = output.read(&mut stream).map_err(|e| {
                eprintln!("n={n} k={k}: {e}");
                e
            })?;
            for q in 0..batch {
                let mut expected: Vec<_> = (0..n)
                    .filter_map(|i| {
                        let s = values[q * n + i];
                        (!s.is_nan() && s != f32::NEG_INFINITY).then_some(hrxdb::Neighbor {
                            id: i as u32,
                            similarity: s,
                        })
                    })
                    .collect();
                expected
                    .sort_by(|a, b| b.similarity.total_cmp(&a.similarity).then(a.id.cmp(&b.id)));
                expected.truncate(k);
                assert_eq!(actual[q], expected, "n={n} k={k} q={q}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn shared_storage_supports_independent_workers_and_snapshots() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    let usage = corpus.memory_usage();
    let ptr = corpus.shards().next().unwrap().vectors().owner() as *const _ as usize;
    let handles: Vec<_> = (0..2)
        .map(|which| {
            let shared = corpus.clone();
            std::thread::spawn(move || -> hrxdb::Result<()> {
                assert_eq!(
                    shared.shards().next().unwrap().vectors().owner() as *const _ as usize,
                    ptr
                );
                let mut worker = shared.searcher()?;
                assert_eq!(worker.memory_usage().corpus, usage);
                let query = if which == 0 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let mut out = Vec::with_capacity(32);
                let pointer = out.as_ptr();
                for _ in 0..3 {
                    worker.search_into(&query, 2, &mut out)?;
                    assert_eq!(out[0].id, which);
                    assert_eq!(out.as_ptr(), pointer);
                }
                Ok(())
            })
        })
        .collect();
    drop(corpus);
    drop(device);
    for h in handles {
        h.join().unwrap()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn resident_import_validates_without_copying_vectors() -> hrxdb::Result<()> {
    use hrxdb::ResidentShard;
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    for invalid in [false, true] {
        let buffer = stream.allocate(256 * 128 * 2)?;
        let mut bytes = vec![0; 256 * 128 * 2];
        if !invalid {
            bytes[..2].copy_from_slice(&half::f16::from_f32(3.0).to_le_bytes());
            bytes[2..4].copy_from_slice(&half::f16::from_f32(4.0).to_le_bytes());
        }
        stream.upload(buffer.binding(), &bytes)?;
        let corpus = Corpus::from_device(
            &device,
            &mut stream,
            3,
            vec![ResidentShard {
                vectors: buffer,
                rows: 1,
            }],
        );
        if invalid {
            assert!(corpus.is_err());
        } else {
            let corpus = corpus?;
            let mut read = vec![0; bytes.len()];
            stream.read_blocking(corpus.shards().next().unwrap().vectors(), &mut read)?;
            assert_eq!(read, bytes);
            let mut searcher = corpus.searcher()?;
            assert!((searcher.search(&[1.0, 0.0, 0.0], 1)?[0].similarity - 0.6).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn device_queries_normalize_validate_and_chain() -> hrxdb::Result<()> {
    use hrxdb::DeviceQueries;
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )?;
    let mut searcher = corpus.searcher()?;
    for queries in [
        vec![3.0f32, 4.0, 0.0],
        vec![3.0, 4.0, 0.0, 0.0, 9.0, 0.0],
        vec![f32::MAX, 0.0, 0.0, 0.0, f32::from_bits(1), 0.0],
    ] {
        let batch = queries.len() / 3;
        let input = searcher.stream().allocate(queries.len() * 4)?;
        searcher.stream().upload(
            input.binding(),
            &queries
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut output = DeviceNeighbors::new(searcher.stream(), batch, 5)?;
        let done = searcher.search_device(
            DeviceQueries::new(input.binding(), batch, 3, 3)?,
            None,
            &mut output,
        )?;
        let mut consumer = corpus.stream()?;
        consumer.wait_event(&done)?;
        let actual = output.read(&mut consumer)?;
        let expected = searcher.search_batch(&queries, 5)?;
        for (a, e) in actual.iter().zip(expected) {
            for (a, e) in a.iter().zip(e) {
                assert_eq!(a.id, e.id);
                assert!((a.similarity - e.similarity).abs() < 3e-6);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn gpu_walk_updates_exclusions_without_host_roundtrips() -> hrxdb::Result<()> {
    use hrxdb::{DeviceExclusions, DeviceQueries};
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )?;
    let mut searcher = corpus.searcher()?;
    searcher.reserve_device(1, 1)?;
    let input = searcher.stream().allocate(12)?;
    searcher.stream().upload(
        input.binding(),
        &[1.0f32, 0.0, 0.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut excluded = DeviceExclusions::new(searcher.stream(), corpus.len())?;
    let mut outputs = Vec::new();
    for _ in 0..4 {
        let mut out = DeviceNeighbors::new(searcher.stream(), 1, 1)?;
        let _done = searcher.search_device(
            DeviceQueries::new(input.binding(), 1, 3, 3)?,
            Some(excluded.binding()),
            &mut out,
        )?;
        excluded.insert_device(searcher.stream(), out.ids(), 1)?;
        outputs.push(out);
    }
    for (i, out) in outputs.iter_mut().enumerate() {
        let result = out.read(searcher.stream())?;
        if i < 3 {
            assert_eq!(result[0][0].id, i as u32);
        } else {
            assert!(result[0].is_empty());
        }
    }
    excluded.remove(searcher.stream(), &[1, 1])?;
    let mut out = DeviceNeighbors::new(searcher.stream(), 1, 3)?;
    let _done = searcher.search_device(
        DeviceQueries::new(input.binding(), 1, 3, 3)?,
        Some(excluded.binding()),
        &mut out,
    )?;
    assert_eq!(
        out.read(searcher.stream())?[0]
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>(),
        [1]
    );
    excluded.clear(searcher.stream())?;
    excluded.insert(searcher.stream(), &[0, 0, 2])?;
    let _done = searcher.search_device(
        DeviceQueries::new(input.binding(), 1, 3, 3)?,
        Some(excluded.binding()),
        &mut out,
    )?;
    assert_eq!(
        out.read(searcher.stream())?[0]
            .iter()
            .map(|n| n.id)
            .collect::<Vec<_>>(),
        [1]
    );
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn topk_merges_large_matrix_tiles_and_global_ids() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    let n = 262_145;
    let batch = 2;
    let mut values = vec![-2.0f32; batch * n];
    values[n - 1] = 3.0;
    values[n + 17] = 2.0;
    values[2 * n - 1] = 4.0;
    let buffer = stream.allocate(values.len() * 4)?;
    stream.upload(
        buffer.binding(),
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut topk = TopK::new(&stream)?;
    for k in [5, 1024] {
        let mut out = DeviceNeighbors::new(&stream, batch, k)?;
        let _done = topk.select(
            &mut stream,
            ScoreBatch::new(buffer.binding(), batch, n)?,
            &mut out,
        )?;
        let got = out.read(&mut stream)?;
        assert_eq!(got[0][0].id, n as u32 - 1);
        assert_eq!(got[1][0].id, n as u32 - 1);
        assert_eq!(got[1][1].id, 17);
        assert_eq!(got[0][1].id, 0);
        assert_eq!(got[0].len(), k);
        assert_eq!(got[1].len(), k);
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn device_query_strides_widths_and_invalid_rows() -> hrxdb::Result<()> {
    use hrxdb::DeviceQueries;
    let device = Device::open(0)?;
    for d in [1usize, 127, 129, 769] {
        let rows: Vec<Vec<f32>> = (0..257)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 37 + j * 13) % 101) as f32 - 49.25)
                    .collect()
            })
            .collect();
        let corpus = Corpus::build(&device, d, &rows)?;
        let mut searcher = corpus.searcher()?;
        for batch in [1usize, 3, 9, 33, 64] {
            let queries: Vec<f32> = (0..batch)
                .flat_map(|i| (0..d).map(move |j| ((i * 19 + j * 7) % 97) as f32 - 45.5))
                .collect();
            let stride = d + 3;
            let mut padded = vec![f32::NAN; batch * stride];
            for (source, dest) in queries.chunks_exact(d).zip(padded.chunks_exact_mut(stride)) {
                dest[..d].copy_from_slice(source);
            }
            let buffer = searcher.stream().allocate(padded.len() * 4)?;
            searcher.stream().upload(
                buffer.binding(),
                &padded
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut output = DeviceNeighbors::new(searcher.stream(), batch, 33)?;
            searcher.reserve_device(batch, 33)?;
            let reserved = searcher.memory_usage();
            let _done = searcher.search_device(
                DeviceQueries::new(buffer.binding(), batch, d, stride)?,
                None,
                &mut output,
            )?;
            let actual = output.read(searcher.stream())?;
            let expected = searcher.search_batch(&queries, 33)?;
            for (q, (got, want)) in actual.iter().zip(expected).enumerate() {
                let scores = searcher.scores(&queries[q * d..(q + 1) * d])?;
                assert_eq!(got.len(), want.len());
                for (a, b) in got.iter().zip(want) {
                    assert!(
                        (a.similarity - b.similarity).abs() < 3e-6,
                        "d={d} batch={batch} q={q}: {a:?} vs {b:?}"
                    );
                    assert!((a.similarity - scores[a.id as usize]).abs() < 3e-6);
                }
            }
            assert_eq!(searcher.memory_usage(), reserved);
        }
    }
    let corpus = Corpus::build(&device, 3, [[1.0, 0.0, 0.0]])?;
    let mut searcher = corpus.searcher()?;
    let input = [
        1.0f32,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        f32::NAN,
        1.0,
        0.0,
        f32::INFINITY,
        0.0,
        0.0,
    ];
    let buffer = searcher.stream().allocate(input.len() * 4)?;
    searcher.stream().upload(
        buffer.binding(),
        &input
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut output = DeviceNeighbors::new(searcher.stream(), 4, 3)?;
    let _done = searcher.search_device(
        DeviceQueries::new(buffer.binding(), 4, 3, 3)?,
        None,
        &mut output,
    )?;
    let mut status = [0; 16];
    let mut counts = [0; 16];
    searcher
        .stream()
        .read_blocking(output.status(), &mut status)?;
    searcher
        .stream()
        .read_blocking(output.counts(), &mut counts)?;
    assert_eq!(
        status
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect::<Vec<_>>(),
        [0, 1, 1, 1]
    );
    assert_eq!(
        counts
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect::<Vec<_>>(),
        [1, 0, 0, 0]
    );
    let mut reused = vec![vec![hrxdb::Neighbor {
        id: 42,
        similarity: 0.5,
    }]];
    assert!(output.read_into(searcher.stream(), &mut reused).is_err());
    assert_eq!(reused[0][0].id, 42);
    assert!(DeviceQueries::new(buffer.binding(), 65, 3, 3).is_err());
    assert!(DeviceQueries::new(buffer.binding(), 4, 3, 2).is_err());
    assert!(DeviceQueries::new(buffer.binding(), 4, 3, 4).is_err());
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn resident_import_rejects_invalid_data_and_preserves_allocation() -> hrxdb::Result<()> {
    use hrxdb::ResidentShard;
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    for kind in 0..6 {
        let vectors = stream.allocate_shared(256 * 128 * 2)?;
        let pointer = vectors.device_ptr()?;
        let mut bytes = vec![0; 256 * 128 * 2];
        bytes[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
        match kind {
            0 => {}
            1 => bytes[..2].copy_from_slice(&half::f16::NAN.to_le_bytes()),
            2 => bytes[..2].copy_from_slice(&half::f16::INFINITY.to_le_bytes()),
            3 => bytes[..2].fill(0),
            4 => bytes[6..8].copy_from_slice(&half::f16::ONE.to_le_bytes()),
            _ => bytes[128 * 2..].fill(0xff), // Unspecified slack may contain NaNs.
        }
        stream.upload(vectors.binding(), &bytes)?;
        let result = Corpus::from_device(
            &device,
            &mut stream,
            3,
            vec![ResidentShard { vectors, rows: 1 }],
        );
        if kind == 0 || kind == 5 {
            let corpus = result?;
            assert_eq!(
                corpus
                    .shards()
                    .next()
                    .unwrap()
                    .vectors()
                    .owner()
                    .device_ptr()?,
                pointer
            );
            assert_eq!(corpus.searcher()?.search(&[1.0, 0.0, 0.0], 1)?[0].id, 0);
        } else {
            assert!(result.is_err(), "kind={kind}");
        }
    }
    let small = stream.allocate(128 * 2)?;
    assert!(
        Corpus::from_device(
            &device,
            &mut stream,
            3,
            vec![ResidentShard {
                vectors: small,
                rows: 1
            }]
        )
        .is_err()
    );
    let empty = Corpus::from_device(&device, &mut stream, 3, vec![])?;
    assert!(empty.is_empty());
    assert_eq!(empty.memory_usage().total(), 0);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn empty_device_shapes_preserve_validity_and_clear_results() -> hrxdb::Result<()> {
    use hrxdb::DeviceQueries;
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
    let mut worker = corpus.searcher()?;
    let values = worker.stream().allocate(24)?;
    worker.stream().upload(
        values.binding(),
        &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    for batch in [0, 1, 2] {
        let mut output = DeviceNeighbors::new(worker.stream(), batch, 1024)?;
        let _done = worker.search_device(
            DeviceQueries::new(values.binding(), batch, 3, 3)?,
            None,
            &mut output,
        )?;
        if batch == 2 {
            assert!(output.read(worker.stream()).is_err());
        } else {
            assert_eq!(output.read(worker.stream())?, vec![vec![]; batch]);
        }
        // Standalone empty selection resets query status and counts in the same output.
        let mut topk = TopK::new(worker.stream())?;
        let _done = topk.select(
            worker.stream(),
            ScoreBatch::new(values.binding(), batch, 0)?,
            &mut output,
        )?;
        assert_eq!(output.read(worker.stream())?, vec![vec![]; batch]);
    }
    Ok(())
}
