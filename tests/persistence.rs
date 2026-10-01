//! Native-file round trips, including segmented snapshots and upload chunks.
use hrx::{execution::RuntimeOptions, inference::ModelContext};
use hrxdb::{Corpus, Result};

struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "hrxdb-persistence-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn bindings(corpus: &Corpus) -> Result<Vec<Vec<u8>>> {
    let mut stream = corpus.stream()?;
    corpus
        .shards()
        .map(|shard| {
            let rows = shard.row_range().len();
            let mut bytes = vec![0; rows * (corpus.padded_dimensions() * 2 + 4)];
            let (vectors, norms) = bytes.split_at_mut(rows * corpus.padded_dimensions() * 2);
            stream.read_blocking(shard.vectors().slice(0, vectors.len())?, vectors)?;
            stream.read_blocking(shard.inverse_norms().slice(0, norms.len())?, norms)?;
            Ok(bytes)
        })
        .collect()
}

#[test]
#[ignore = "requires gfx1151"]
fn native_round_trip_preserves_snapshots_and_search() -> Result<()> {
    let context = ModelContext::new(RuntimeOptions::default())?;
    let path = Temp::new();
    // More than one upload chunk, nontrivial dimension padding, and a split
    // caused by copy-on-write followed by an appended tail with reserve.
    let base = Corpus::build_in(&context, 3, (0..600_001).map(|i| [1., i as f32 + 1., -0.5]))?;
    let changed = base
        .clone()
        .update(&[300_000], [[0., 1., 0.]])?
        .append(std::iter::repeat_n([0., 0., 1.], 300))?;
    assert!(changed.shards().len() > 1);
    let empty = Corpus::build_in(&context, 3, std::iter::empty::<[f32; 3]>())?;
    for corpus in [&base, &changed, &empty] {
        corpus.save(&path.0)?;
        let loaded = Corpus::load_in(&context, &path.0)?;
        assert_eq!(loaded.len(), corpus.len());
        assert_eq!(loaded.dimensions(), corpus.dimensions());
        assert_eq!(loaded.capacity_range(), corpus.capacity_range());
        assert_eq!(bindings(&loaded)?, bindings(corpus)?);
        assert!(
            loaded
                .context()
                .unwrap()
                .runtime()
                .same_domain(context.runtime())
        );
        if !corpus.is_empty() {
            assert_eq!(
                loaded.searcher()?.search(&[0., 0., 1.], 10)?,
                corpus.searcher()?.search(&[0., 0., 1.], 10)?
            );
            let appended = loaded.append([[1., 0., 0.]])?;
            assert_eq!(appended.len(), corpus.len() + 1);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151"]
fn load_charges_and_releases_the_context_budget() -> Result<()> {
    let path = Temp::new();
    let context = ModelContext::new(RuntimeOptions::default())?;
    // Non-unit FP16 rows must keep their original inverse norms.
    let corpus = Corpus::build_fp16_in(&context, 3, [[0, 0x40, 0, 0x42, 0, 0x44]])?;
    corpus.save(&path.0)?;
    let resident = 256 * (128 * 2 + 4);
    let manager = hrx::residency::ResidencyManager::new(resident + 512)?;
    let budgeted = ModelContext::new(RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })?;
    let loaded = Corpus::load_in(&budgeted, &path.0)?;
    assert_eq!(manager.statistics().reserved_bytes, resident);
    assert_eq!(bindings(&loaded)?, bindings(&corpus)?);
    drop(loaded);
    assert_eq!(manager.statistics().reserved_bytes, 0);
    let small = hrx::residency::ResidencyManager::new(resident - 1)?;
    let budgeted = ModelContext::new(RuntimeOptions {
        memory_budget: Some(small.budget()),
        ..Default::default()
    })?;
    assert!(Corpus::load_in(&budgeted, &path.0).is_err());
    assert_eq!(small.statistics().reserved_bytes, 0);
    Ok(())
}
