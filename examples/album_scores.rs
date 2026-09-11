//! Custom album scoring followed by GPU top-k; read back only the winners.
#[path = "support/album_scores.rs"]
mod album_scores;
use hrxdb::{Corpus, Device, DeviceNeighbors, ScoreBatch, TopK};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
    )?;
    drop(device); // Later callers need only the corpus, not its original Device.
    // Albums are application metadata, independent of hrxdb's insertion IDs.
    let ordinals = [0, 1, 0, 2];
    let queries = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut stream = corpus.stream()?;
    let scores = album_scores::score_by_album(&mut stream, &corpus, &ordinals, 4, &queries)?;
    let mut topk = TopK::new(&stream)?;
    let mut output = DeviceNeighbors::new(&stream, 2, 2)?;
    let _done = topk.select(
        &mut stream,
        ScoreBatch::new(scores.binding(), 2, 4)?,
        &mut output,
    )?;
    let winners = output.read(&mut stream)?;
    assert_eq!(winners[0].iter().map(|n| n.id).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(winners[1].iter().map(|n| n.id).collect::<Vec<_>>(), [1, 0]);
    for (row, albums) in winners.iter().enumerate() {
        println!("query {row}: {albums:?}"); // IDs are album columns, not corpus rows.
    }
    Ok(())
}
