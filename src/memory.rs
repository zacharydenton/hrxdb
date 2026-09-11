//! Allocation accounting. Shared storage is reported separately from each worker.
use crate::*;
/// Bytes in one shared corpus allocation set; cloned handles do not duplicate it.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct CorpusMemory {
    /// Logical FP16 components, excluding dimension padding.
    pub vectors: usize,
    /// Zero padding within logical vector rows.
    pub dimension_padding: usize,
    /// Readable vector rows beyond logical shard lengths.
    pub vector_slack: usize,
    /// FP32 inverse norms for logical rows.
    pub norms: usize,
    /// Readable inverse norms beyond logical shard lengths.
    pub norm_slack: usize,
}
impl CorpusMemory {
    /// Total resident bytes. Count a shared corpus only once across searchers.
    pub fn total(&self) -> usize {
        self.vectors + self.dimension_padding + self.vector_slack + self.norms + self.norm_slack
    }
}
/// Private buffer bytes owned by one searcher, excluding shared storage and
/// caller-owned device outputs. Host-imported buffers include page rounding.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct WorkspaceMemory {
    /// Single-query input storage.
    pub queries: usize,
    /// Single-query full scores.
    pub scores: usize,
    /// Single-query selection scratch.
    pub selection: usize,
    /// Single-query result staging.
    pub readback: usize,
    /// Shared host-input exclusion bitmap.
    pub exclusions: usize,
    /// Reserved batch buffers, including query, selection, and readback.
    pub batch: usize,
    /// Device-query preparation scratch.
    pub device_queries: usize,
}
impl WorkspaceMemory {
    /// Total worker-owned buffer bytes, excluding runtime/compiler bookkeeping.
    pub fn total(&self) -> usize {
        self.queries
            + self.scores
            + self.selection
            + self.readback
            + self.exclusions
            + self.batch
            + self.device_queries
    }
}
/// Shared corpus bytes and private workspace bytes for one searcher.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct SearcherMemory {
    /// Count these bytes once for all searchers sharing the same corpus.
    pub corpus: CorpusMemory,
    /// Additional storage allocated by this searcher.
    pub workspace: WorkspaceMemory,
}
impl Corpus {
    /// Report storage bytes without initializing a searcher or querying hardware.
    pub fn memory_usage(&self) -> CorpusMemory {
        let capacity: usize = self.inner.shards.iter().map(|s| s.capacity).sum();
        CorpusMemory {
            vectors: self.len() * self.dimensions() * 2,
            dimension_padding: self.len() * (self.padded_dimensions() - self.dimensions()) * 2,
            vector_slack: (capacity - self.len()) * self.padded_dimensions() * 2,
            norms: self.len() * 4,
            norm_slack: (capacity - self.len()) * 4,
        }
    }
}
impl Searcher {
    /// Separate shared corpus storage from this searcher's reserved workspace.
    /// Excludes native allocation granularity, compiler code, runtime staging,
    /// and caller-owned inputs/outputs. No synchronization or GPU work occurs.
    pub fn memory_usage(&self) -> SearcherMemory {
        SearcherMemory {
            corpus: self.corpus.memory_usage(),
            workspace: WorkspaceMemory {
                queries: self.query.buffer().bytes(),
                scores: self.scores.bytes(),
                selection: self
                    .candidates
                    .iter()
                    .map(|(s, i)| s.bytes() + i.bytes())
                    .sum(),
                readback: self.readback.buffer().bytes(),
                exclusions: self.exclusions.buffer().bytes(),
                batch: self.batch_workspace_bytes(),
                device_queries: self.device_query_bytes(),
            },
        }
    }
}
