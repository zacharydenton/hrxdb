//! Named-row gather contracts and device composition.
use half::f16;
use hrxdb::{Corpus, Device, DeviceNeighbors, DeviceQueries, MAX_BATCH, MAX_K, ResidentShard};

#[test]
#[ignore = "requires gfx1151"]
fn gather_preserves_order_values_and_output_bounds() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    for dimensions in [1usize, 3, 127, 128, 129, 384, 769, 16_384] {
        let rows: Vec<Vec<f32>> = (0..11)
            .map(|r| {
                (0..dimensions)
                    .map(|c| match r {
                        0 => f16::from_bits(1).to_f32(),
                        1 => f16::MAX.to_f32(),
                        _ => f16::from_f32(((r * 17 + c * 13) % 61) as f32 - 29.25).to_f32(),
                    })
                    .collect()
            })
            .collect();
        let raw: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect()
            })
            .collect();
        let corpus = Corpus::build_fp16(&device, dimensions, &raw)?;
        let mut stream = corpus.stream()?;
        let count = if dimensions == 3 {
            MAX_K + 17
        } else {
            MAX_BATCH + 5
        };
        let ids: Vec<_> = (0..count).map(|i| [10, 0, 10, 3, 1, 2, 9][i % 7]).collect();
        let bytes = ids.len() * dimensions * 4;
        let buffer = stream.allocate(bytes + 36)?;
        stream.fill(buffer.binding(), 0xa5)?;
        let event = corpus.gather_into(&mut stream, &ids, buffer.try_slice(16, bytes + 20)?)?;
        let mut consumer = corpus.stream()?;
        consumer.wait_event(&event)?;
        let mut actual = vec![0; bytes + 36];
        consumer.read_blocking(buffer.binding(), &mut actual)?;
        assert_eq!(&actual[..16], &[0xa5; 16]);
        assert_eq!(&actual[16 + bytes..], &[0xa5; 20]);
        for (gathered, &id) in actual[16..16 + bytes]
            .chunks_exact(dimensions * 4)
            .zip(&ids)
        {
            let original = &rows[id as usize];
            let inverse = (1.0
                / original
                    .iter()
                    .map(|&v| (v as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()) as f32;
            for (value, &expected) in gathered.as_chunks::<4>().0.iter().zip(original) {
                assert_eq!(
                    f32::from_le_bytes(*value).to_bits(),
                    (expected * inverse).to_bits(),
                    "d={dimensions}, id={id}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn gather_routes_unaligned_shards_and_reports_capacity_extents() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let mut stream = device.stream()?;
    let mut resident = Vec::new();
    // The first shard's slack extends past every later shard's readable end.
    for (logical, capacity, sign) in [(1, 512, 1.0), (3, 256, -1.0)] {
        let buffer = stream.allocate(capacity * 128 * 2)?;
        let mut bytes = vec![0; capacity * 128 * 2];
        for row in 0..logical {
            bytes[row * 256..row * 256 + 2].copy_from_slice(&f16::from_f32(sign).to_le_bytes());
        }
        stream.upload(buffer.binding(), &bytes)?;
        resident.push(ResidentShard {
            vectors: buffer,
            rows: logical,
        });
    }
    let corpus = Corpus::from_device(&device, &mut stream, 3, resident)?;
    let ranges: Vec<_> = corpus
        .shards()
        .map(|s| (s.row_range(), s.capacity_range()))
        .collect();
    assert_eq!(ranges, [(0..1, 0..512), (1..4, 1..257)]);
    assert_eq!(corpus.capacity_range(), 0..512);
    let output = stream.allocate(6 * 3 * 4)?;
    let _done = corpus.gather_into(&mut stream, &[3, 0, 1, 0, 2, 3], output.binding())?;
    let mut bytes = vec![0; 6 * 3 * 4];
    stream.read_blocking(output.binding(), &mut bytes)?;
    let actual: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    assert_eq!(
        actual,
        [
            -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
            0.0, 0.0
        ]
    );
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn gather_validates_before_writing_and_handles_empty_requests() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 3, [[1.0, 0.0, 0.0]])?;
    let mut stream = corpus.stream()?;
    let output = stream.allocate(32)?;
    stream.fill(output.binding(), 0xa5)?;
    assert!(
        corpus
            .gather_into(&mut stream, &[0, 1], output.binding())
            .is_err()
    );
    assert!(
        corpus
            .gather_into(&mut stream, &[u32::MAX], output.binding())
            .is_err()
    );
    assert!(
        corpus
            .gather_into(&mut stream, &[0], output.try_slice(0, 11)?)
            .is_err()
    );
    assert!(
        corpus
            .gather_into(&mut stream, &[0], output.try_slice(1, 12)?)
            .is_err()
    );
    let _done = corpus.gather_into(&mut stream, &[], output.binding())?;
    let empty = Corpus::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
    assert_eq!(empty.capacity_range(), 0..0);
    let _done = empty.gather_into(&mut stream, &[], output.binding())?;
    assert!(
        empty
            .gather_into(&mut stream, &[0], output.binding())
            .is_err()
    );
    let mut bytes = [0; 32];
    stream.read_blocking(output.binding(), &mut bytes)?;
    assert_eq!(bytes, [0xa5; 32]);
    for shard in corpus.shards() {
        assert!(
            corpus
                .gather_into(&mut stream, &[0], shard.vectors())
                .is_err()
        );
        assert!(
            corpus
                .gather_into(&mut stream, &[0], shard.inverse_norms())
                .is_err()
        );
    }
    // A rejected write into borrowed corpus storage must not corrupt search.
    assert_eq!(corpus.searcher()?.search(&[1.0, 0.0, 0.0], 1)?[0].id, 0);
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn gather_composes_with_device_queries_and_reuses_ordered_scratch() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )?;
    let mut producer = corpus.stream()?;
    let mut worker = corpus.searcher()?;
    worker.reserve_device(2, 1)?;
    let first = producer.allocate(24)?;
    let second = producer.allocate(24)?;
    // The second call reuses routing scratch before the first completes.
    let _a = corpus.gather_into(&mut producer, &[2, 0], first.binding())?;
    let b = corpus.gather_into(&mut producer, &[1, 2], second.binding())?;
    worker.stream().wait_event(&b)?;
    let mut result = DeviceNeighbors::new(worker.stream(), 2, 1)?;
    let _done = worker.search_device(
        DeviceQueries::new(first.binding(), 2, 3, 3)?,
        None,
        &mut result,
    )?;
    assert_eq!(
        result
            .read(worker.stream())?
            .iter()
            .map(|r| r[0].id)
            .collect::<Vec<_>>(),
        [2, 0]
    );
    let _done = worker.search_device(
        DeviceQueries::new(second.binding(), 2, 3, 3)?,
        None,
        &mut result,
    )?;
    assert_eq!(
        result
            .read(worker.stream())?
            .iter()
            .map(|r| r[0].id)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn cloned_corpora_gather_on_independent_threads() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(&device, 1, [[1.0], [-1.0]])?;
    let handles: Vec<_> = (0..2)
        .map(|id| {
            let corpus = corpus.clone();
            std::thread::spawn(move || -> hrxdb::Result<()> {
                let mut stream = corpus.stream()?;
                let out = stream.allocate(4)?;
                let _done = corpus.gather_into(&mut stream, &[id], out.binding())?;
                let mut bytes = [0; 4];
                stream.read_blocking(out.binding(), &mut bytes)?;
                assert_eq!(f32::from_le_bytes(bytes), if id == 0 { 1.0 } else { -1.0 });
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
