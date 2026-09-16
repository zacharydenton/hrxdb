//! Shared-context search integration on real GPU storage.
use hrx::{
    execution::RuntimeOptions,
    inference::ModelContext,
    tensor::{DType, Layout, TensorDesc},
};

#[test]
#[ignore = "requires gfx1151"]
fn resident_queries_retain_slots_and_match_search() -> hrxdb::Result<()> {
    let context = ModelContext::new(RuntimeOptions::default())?;
    let corpus = hrxdb::Corpus::build(
        &hrxdb::Device::open(0)?,
        3,
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    )?;
    let search = corpus.prepare_search(&context, 1, 2, 1)?;
    let bytes = [1f32, 0., 0.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let query = context.upload(
        TensorDesc::new(DType::F32, vec![1, 3])?.with_layout(Layout::Rows)?,
        &bytes,
    )?;
    let result = search.submit(&query)?;
    let ids = result.ids().clone();
    assert_eq!(result.read()?[0][0].id, 0);
    assert!(matches!(search.submit(&query), Err(hrx::Error::Busy(_))));
    assert_eq!(
        u32::from_le_bytes(context.download(&ids)?.wait()?[..4].try_into().unwrap()),
        0
    );
    drop(ids);
    assert_eq!(search.submit(&query)?.read()?[0][0].similarity, 1.);
    let bad = context.upload(
        TensorDesc::new(DType::F32, vec![1, 3])?.with_layout(Layout::Rows)?,
        &[0; 12],
    )?;
    assert!(search.submit(&bad)?.read().is_err());
    Ok(())
}
