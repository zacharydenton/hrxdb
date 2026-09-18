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
    let corpus = hrxdb::Corpus::build_in(&context, 3, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])?;
    assert!(
        corpus
            .context()
            .unwrap()
            .runtime()
            .same_domain(context.runtime())
    );
    drop(context);
    let context = corpus.context().unwrap().clone();
    let other = ModelContext::new(RuntimeOptions::default())?;
    assert!(
        matches!(corpus.prepare_search(&other, 1, 2, 1), Err(hrx::Error::Message(message)) if message.contains("another runtime"))
    );
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

#[test]
#[ignore = "requires gfx1151"]
fn context_fp16_storage_and_tensors_share_the_budget() -> hrxdb::Result<()> {
    let memory = hrxdb::Corpus::build_memory(3, 1)?;
    let manager = hrx::residency::ResidencyManager::new(memory.peak_bytes)?;
    let context = ModelContext::new(RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    let rows = [[0, 0x3c, 0, 0, 0, 0]];
    let corpus = hrxdb::Corpus::build_fp16_in(&context, 3, rows)?;
    assert_eq!(manager.statistics().reserved_bytes, memory.resident_bytes);
    // A tensor competes with corpus storage under the exact same ceiling.
    let desc = TensorDesc::new(DType::F32, vec![131])?;
    assert!(context.allocate(desc.clone()).is_err());
    let retained = corpus.clone();
    drop(corpus);
    assert_eq!(manager.statistics().reserved_bytes, memory.resident_bytes);
    let mut stream = retained.stream()?;
    assert!(stream.memory_budget().is_some());
    let mut bytes = [0; 6];
    stream.read_blocking(
        retained.shards().next().unwrap().vectors().slice(0, 6)?,
        &mut bytes,
    )?;
    assert_eq!(bytes, rows[0]);
    drop(stream);
    drop(retained);
    assert_eq!(manager.statistics().reserved_bytes, 0);
    let tensor = context.allocate(desc)?;
    assert_eq!(manager.statistics().reserved_bytes, 524);
    drop(tensor);
    assert!(hrxdb::Corpus::build_fp16_in(&context, 3, [[0; 6]]).is_err());
    assert_eq!(manager.statistics().reserved_bytes, 0);
    Ok(())
}
