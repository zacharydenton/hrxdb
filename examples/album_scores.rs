//! Run an application-owned Loom kernel over hrxdb's borrowed corpus.
#[path = "support/album_scores.rs"]
mod album_scores;
use hrxdb::{Device, FlatIndex};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let index = FlatIndex::build(
        &device,
        3,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )?;
    drop(device); // Later callers need only the index, not its original Device.
    // Albums are application metadata, independent of hrxdb's insertion IDs.
    let ordinals = [0, 1, 0, 2];
    let queries = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let scores = album_scores::best_by_album(index.corpus(), &ordinals, 4, &queries)?;
    assert_eq!(
        scores,
        vec![
            vec![1.0, 0.0, 0.0, f32::NEG_INFINITY],
            vec![0.0, 1.0, -1.0, f32::NEG_INFINITY]
        ]
    );
    for (row, albums) in scores.iter().enumerate() {
        println!("query {row}: {albums:?}");
    }
    Ok(())
}
