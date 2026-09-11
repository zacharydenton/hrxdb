//! Queue searches and update visited IDs on the GPU, then read only the final result.
use hrxdb::{Corpus, Device, DeviceExclusions, DeviceNeighbors, DeviceQueries};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )?;
    let mut worker = corpus.searcher()?;
    worker.reserve_device(1, 1)?;
    let query = worker.stream().allocate(12)?;
    worker.stream().upload(
        query.binding(),
        &[1.0f32, 0.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let mut visited = DeviceExclusions::new(worker.stream(), corpus.len())?;
    let mut result = DeviceNeighbors::new(worker.stream(), 1, 1)?;
    for _ in 0..2 {
        let _done = worker.search_device(
            DeviceQueries::new(query.binding(), 1, 3, 3)?,
            Some(visited.binding()),
            &mut result,
        )?;
        visited.insert_device(worker.stream(), result.ids(), 1)?;
    }
    let done = worker.search_device(
        DeviceQueries::new(query.binding(), 1, 3, 3)?,
        Some(visited.binding()),
        &mut result,
    )?;
    let mut consumer = corpus.stream()?;
    consumer.wait_event(&done)?;
    let matches = result.read(&mut consumer)?;
    assert_eq!(matches[0][0].id, 2);
    println!("third unvisited match: {:?}", matches[0][0]);
    Ok(())
}
