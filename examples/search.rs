//! Map insertion positions to application IDs and move an index to a worker.
use hrxdb::{Device, FlatIndex};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let mut index = FlatIndex::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0]],
    )?;
    let external_ids = ["document-a", "document-b", "document-c"];
    std::thread::spawn(move || -> hrxdb::Result<()> {
        for neighbor in index.search(&[1.0, 0.0, 0.0], 2)? {
            println!(
                "{}: {:.6}",
                external_ids[neighbor.id as usize], neighbor.similarity
            );
        }
        Ok(())
    })
    .join()
    .expect("query worker panicked")
}
