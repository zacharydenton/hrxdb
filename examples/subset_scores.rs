//! Gather two named subsets, compare on the GPU, and read only the winners.
use hrx::loom::{Compiler, Specialization};
use hrxdb::{Corpus, Device, DeviceNeighbors, ScoreBatch, TopK};

fn main() -> hrxdb::Result<()> {
    let device = Device::open(0)?;
    let corpus = Corpus::build(
        &device,
        3,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )?;
    let left_ids = [0, 2];
    let right_ids = [2, 1, 0];
    let mut stream = corpus.stream()?;
    let left = stream.allocate(left_ids.len() * corpus.dimensions() * 4)?;
    let right = stream.allocate(right_ids.len() * corpus.dimensions() * 4)?;
    let scores = stream.allocate(left_ids.len() * right_ids.len() * 4)?;
    let compiler = Compiler::for_stream(None, &stream)?;
    let mut spec = Specialization::new("subset_scores");
    for (key, value) in [
        ("subset.dimensions", corpus.dimensions()),
        ("subset.left", left_ids.len()),
        ("subset.right", right_ids.len()),
    ] {
        spec.set_config(key, value.to_string());
    }
    let artifact = compiler
        .module(include_str!("subset_scores.loom"))
        .compile(&spec)?;
    // SAFETY: authored example source, specialized for this device and shape.
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    let mut topk = TopK::new(&stream)?;
    topk.reserve(&stream, left_ids.len(), right_ids.len(), 1)?;
    let mut output = DeviceNeighbors::new(&stream, left_ids.len(), 1)?;

    let _left_done = corpus.gather_into(&mut stream, &left_ids, left.binding())?;
    let _right_done = corpus.gather_into(&mut stream, &right_ids, right.binding())?;
    // SAFETY: both gathered blocks are compact, normalized FP32 matrices of
    // the specialized shape. The same stream orders gathers before their use.
    unsafe {
        stream.dispatch(
            &kernel,
            [right_ids.len() as u32, left_ids.len() as u32, 1],
            [128, 1, 1],
            &hrx::Constants::new(),
            &[left.binding(), right.binding(), scores.binding()],
        )?;
    }
    let _done = topk.select(
        &mut stream,
        ScoreBatch::new(scores.binding(), left_ids.len(), right_ids.len())?,
        &mut output,
    )?;
    for (left_id, matches) in left_ids.iter().zip(output.read(&mut stream)?) {
        let neighbor = matches[0];
        // Selection IDs index right_ids, not the original corpus.
        let right_id = right_ids[neighbor.id as usize];
        assert_eq!(*left_id, right_id);
        assert_eq!(neighbor.similarity, 1.0);
        println!(
            "row {left_id}: best candidate row {right_id}, cosine {}",
            neighbor.similarity
        );
    }
    Ok(())
}
