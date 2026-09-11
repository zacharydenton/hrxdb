//! Map insertion positions to application IDs and share one corpus between independent workers.
use hrxdb::{Corpus, Device};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0]],
    )?;
    let external_ids = ["document-a", "document-b", "document-c"];
    let mut index = corpus.searcher()?;
    drop(device);
    // The worker and this snapshot share vector allocations.
    let snapshot = corpus.clone();
    let worker = std::thread::spawn(move || -> hrxdb::Result<()> {
        for neighbor in index.search(&[1.0, 0.0, 0.0], 2)? {
            println!(
                "{}: {:.6}",
                external_ids[neighbor.id as usize], neighbor.similarity
            );
        }
        Ok(())
    });
    assert_eq!(snapshot.len(), 3);
    worker.join().expect("query worker panicked")
}
