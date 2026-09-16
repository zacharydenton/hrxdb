//! Opt-in budgeted corpus loading. Search and application workspaces are separate.
use crate::*;
use hrx::residency::{ModelLease, ResidencyManager};
use std::sync::Arc;

/// Logical allocation extents used to reserve a corpus before loading it.
/// Excludes compiler/driver memory, allocation granularity and caller-owned
/// input mappings. Peak includes host conversion and native upload staging.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CorpusBuildMemory {
    /// Shared FP16 vectors and FP32 inverse norms, including padding and slack.
    pub resident_bytes: usize,
    /// Resident storage plus both bounded conversion and upload staging buffers.
    pub peak_bytes: usize,
}
impl Corpus {
    /// Estimate allocation extents without opening a device or consuming rows.
    /// Empty corpora report zero bytes; budgeted loading requires nonempty rows.
    pub fn build_memory(dimensions: usize, count: usize) -> Result<CorpusBuildMemory> {
        let (padded, _) = layout(dimensions, count)?;
        if count == 0 {
            return Ok(CorpusBuildMemory {
                resident_bytes: 0,
                peak_bytes: 0,
            });
        }
        let per_shard = shard_rows(padded, MAX_ELEMENTS);
        let capacity = (0..count)
            .step_by(per_shard)
            .map(|start| shard_capacity((count - start).min(per_shard)))
            .sum::<usize>();
        let resident_bytes = capacity
            .checked_mul(padded * 2 + 4)
            .ok_or_else(|| invalid("corpus residency size overflow"))?;
        let chunk_rows = (CHUNK_BYTES / (padded * 2)).max(1).min(count);
        let peak_bytes = resident_bytes
            .checked_add(2 * chunk_rows * (padded * 2 + 4))
            .ok_or_else(|| invalid("corpus loader reservation overflow"))?;
        Ok(CorpusBuildMemory {
            resident_bytes,
            peak_bytes,
        })
    }

    /// Reserve peak loader memory, then load/reuse one immutable FP16 corpus.
    /// `artifact` must identify immutable row content (e.g. a content digest or
    /// immutable cache inode/version). Shape and native device identity are added
    /// automatically. Reusing a key does not consume or validate the iterator.
    ///
    /// A lease or persistent `lease.pin()` prevents eviction. Exported Corpus
    /// clones, Searchers and prepared searches also keep it non-idle after the
    /// lease disappears, so live native bindings cannot escape budget accounting.
    /// Register separate reservations for growing search/application workspaces.
    pub fn load_resident_fp16<I, R>(
        manager: &ResidencyManager,
        artifact: &str,
        device: &Device,
        dimensions: usize,
        rows: I,
    ) -> Result<ModelLease<Self>>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        let rows = rows.into_iter();
        let memory = Self::build_memory(dimensions, rows.len())?;
        if memory.resident_bytes == 0 {
            return Err(invalid("budgeted corpus must be nonempty"));
        }
        let device_id = device.stream()?.device_id();
        let key = format!(
            "hrxdb:fp16-v1:{device_id}:{dimensions}:{}:{artifact}",
            rows.len()
        );
        manager.load(
            key,
            memory.peak_bytes,
            |corpus: &Corpus| Arc::strong_count(&corpus.inner) == 1,
            || {
                let corpus = Self::build_fp16(device, dimensions, rows)?;
                let bytes = corpus.memory_usage().total();
                if bytes != memory.resident_bytes {
                    return Err(invalid("corpus allocation estimate mismatch"));
                }
                Ok((corpus, bytes))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires gfx1151"]
    fn workspace_growth_is_budgeted_and_failed_growth_preserves_search() -> Result<()> {
        let device = Device::open(0)?;
        let manager = ResidencyManager::new(120_000)?;
        let rows = [[0x00, 0x3c, 0, 0, 0, 0]];
        let lease = Corpus::load_resident_fp16(&manager, "workspace", &device, 3, rows)?;
        let corpus_bytes = manager.statistics().reserved_bytes;
        let corpus = (*lease).clone().with_workspace_budget(manager.budget());
        let mut search = corpus.searcher()?;
        let initial = manager.statistics().reserved_bytes;
        assert!(initial > corpus_bytes);
        assert!(
            search
                .reserve_batch(crate::MAX_BATCH, crate::MAX_K)
                .is_err()
        );
        assert!(manager.statistics().reserved_bytes <= 120_000);
        assert_eq!(search.search(&[1., 0., 0.], 1)?[0].id, 0);
        drop(search);
        assert_eq!(manager.statistics().reserved_bytes, corpus_bytes);
        Ok(())
    }
    #[test]
    fn estimates_include_padding_slack_and_loader_staging() -> Result<()> {
        assert_eq!(
            Corpus::build_memory(3, 1)?,
            CorpusBuildMemory {
                resident_bytes: 256 * 260,
                peak_bytes: 258 * 260
            }
        );
        assert_eq!(Corpus::build_memory(3, 257)?.resident_bytes, 512 * 260);
        assert_eq!(Corpus::build_memory(384, 0)?.peak_bytes, 0);
        assert!(Corpus::build_memory(0, 1).is_err());
        assert!(Corpus::build_memory(MAX_DIMENSIONS + 1, 1).is_err());
        assert!(Corpus::build_memory(3, MAX_ROWS + 1).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires gfx1151"]
    fn resident_corpus_reserves_rolls_back_and_protects_exported_clones() -> Result<()> {
        let device = Device::open(0)?;
        let rows = [[0x00, 0x3c, 0, 0, 0, 0]];
        let memory = Corpus::build_memory(3, 1)?;
        let manager = ResidencyManager::new(memory.peak_bytes)?;
        let lease = Corpus::load_resident_fp16(&manager, "one", &device, 3, rows)?;
        assert_eq!(lease.bytes(), memory.resident_bytes);
        assert_eq!(manager.statistics().reserved_bytes, memory.resident_bytes);
        let pin = lease.pin();
        let exported = (*lease).clone();
        let reused = Corpus::load_resident_fp16(&manager, "one", &device, 3, rows)?;
        assert!(Arc::ptr_eq(&lease.inner, &reused.inner));
        drop(reused);
        drop(lease);
        assert!(matches!(
            Corpus::load_resident_fp16(&manager, "two", &device, 3, rows),
            Err(Error::Busy(_))
        ));
        drop(pin);
        assert!(matches!(
            Corpus::load_resident_fp16(&manager, "two", &device, 3, rows),
            Err(Error::Busy(_))
        ));
        let mut search = exported.searcher()?;
        drop(exported);
        assert!(matches!(
            Corpus::load_resident_fp16(&manager, "two", &device, 3, rows),
            Err(Error::Busy(_))
        ));
        assert_eq!(search.search(&[1., 0., 0.], 1)?[0].id, 0);
        drop(search);
        assert!(Corpus::load_resident_fp16(&manager, "invalid", &device, 3, [[0; 6]]).is_err());
        assert_eq!(manager.statistics().reserved_bytes, 0);
        let loaded = Corpus::load_resident_fp16(&manager, "two", &device, 3, rows)?;
        assert_eq!(loaded.memory_usage().total(), memory.resident_bytes);
        assert_eq!(manager.statistics().evictions, 1);
        Ok(())
    }
}
