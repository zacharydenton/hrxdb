//! Reuse an encoder's context for corpus loading and resident query tensors.
use hrx::{
    Access,
    execution::{GpuAccess, RuntimeOptions},
    inference::ModelContext,
    residency::ResidencyManager,
    tensor::{DType, Layout, TensorDesc},
};
use hrxdb::Corpus;

fn main() -> hrxdb::Result<()> {
    let manager = ResidencyManager::new(64 * 1024 * 1024)?;
    let context = ModelContext::new(RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    let rows = [[0, 0x3c, 0, 0, 0, 0], [0, 0, 0, 0x3c, 0, 0]];
    let corpus = Corpus::load_resident_fp16_in(&context, "example-v1", 3, rows)?;
    let search = corpus.prepare_search(&context, 1, 2, 1)?;

    // An encoder loaded in `context` can supply its output tensor here directly.
    let query = context.upload(
        TensorDesc::new(DType::F32, vec![1, 3])?.with_layout(Layout::Rows)?,
        &[1f32, 0., 0.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let results = search.submit(&query)?.read()?;
    assert_eq!(results[0][0].id, 0);
    println!("{results:?}");

    // A native kernel can read corpus buffers and write a context tensor in
    // one dispatch. Here gather's kernel produces a query for resident search.
    let desc = TensorDesc::new(DType::F32, vec![1, 3])?.with_layout(Layout::Rows)?;
    let output = context.allocate(desc.clone())?;
    let binding = output.binding().unwrap();
    let retained = (*corpus).clone();
    let mut stream = retained.stream()?;
    let mut graph = context.runtime().graph();
    // SAFETY: the callback retains its stream and immutable corpus, writes the
    // entire declared output, and drains the stream before returning. No scoped
    // view escapes. Other corpus users only read the shared native storage.
    unsafe {
        graph.gpu_scoped(
            &[GpuAccess {
                view: binding.clone(),
                access: Access::Write,
            }],
            move |views| {
                let result = retained.gather_into(&mut stream, &[1], views[0]);
                stream.synchronize()?;
                result?;
                Ok(())
            },
        )?;
    }
    let producer = graph.prepare()?.submit()?;
    let gathered = context.tensor(desc, binding, producer)?;
    assert_eq!(search.submit(&gathered)?.read()?[0][0].id, 1);
    Ok(())
}
