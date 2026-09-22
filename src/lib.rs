//! A GPU-powered vector database for unified-memory systems, built on HRX and Loom.
//!
//! HRX manages GPU buffers, streams, and execution; Loom kernels provide
//! exhaustive cosine search and composable GPU scoring on AMD Strix Halo.
//! The Rust API exposes FP16 corpus storage and FP32 scores.
//!
//! A shared [`Corpus`] owns immutable storage. Each [`Searcher`] owns its stream
//! and reusable workspace. [`TopK`] selects from application-defined GPU scores.
//! Scores describe the quantized corpus, not the original FP32 rows.
//!
//! Execution requires Linux x86_64, a gfx1151 AMD GPU, and the native runtime
//! prerequisites in the [hrx-rs documentation](https://docs.rs/hrx-rs/0.5.0/hrx/).
//! Building and generating documentation do not initialize GPU hardware.
//!
//! ```no_run
//! use hrxdb::{Corpus, Device};
//!
//! # fn main() -> hrxdb::Result<()> {
//! let device = Device::open(0)?;
//! let corpus = Corpus::build(&device, 3, [
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//! ])?;
//! let mut index = corpus.searcher()?;
//! let external_ids = ["document-a", "document-b"];
//! for neighbor in index.search(&[1.0, 0.0, 0.0], 2)? {
//!     println!("{}: {}", external_ids[neighbor.id as usize], neighbor.similarity);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! IDs are insertion positions, not application IDs. The index does not persist
//! vectors, map external IDs, or support updates, metadata filters, or approximate
//! search. Query-time exclusions use insertion IDs.
//! [`Searcher::corpus`] provides borrowed storage bindings for application-owned
//! GPU kernels, side arrays, and reductions; see [`Corpus`] for the contract.
mod batch;
mod corpus;
mod host;
mod kernels;
mod storage;
pub use storage::ResidentShard;
mod device;
mod exclusions;
mod gather;
mod inference;
mod memory;
mod residency;
pub use residency::CorpusBuildMemory;
mod queries;
pub use inference::{PreparedSearch, SearchInference};
mod selection;
pub use device::{DeviceNeighbors, ScoreBatch, TopK};
pub use exclusions::DeviceExclusions;
use host::HostBuffer;
pub use memory::{CorpusMemory, SearcherMemory, WorkspaceMemory};
pub use queries::DeviceQueries;

// Compile application example code unchanged inside the corpus hardware tests.
#[cfg(test)]
extern crate self as hrxdb;

use half::f16;
use hrx::{Buffer, Constants, Kernel, Stream};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use corpus::CorpusStorage;
pub use corpus::{Corpus, CorpusShardView};
pub use hrx::{Device, Error, Result};

const MAX_ROWS: usize = 1 << 30;
const MAX_ELEMENTS: usize = 1 << 32;
const SHARD_ROW_ALIGNMENT: usize = 256;
/// Largest supported top-k result count for search and standalone selection.
pub const MAX_K: usize = 1024;
/// Largest supported query count for batched search and standalone selection.
/// This does not limit the number of rows requested by [`Corpus::gather_into`].
pub const MAX_BATCH: usize = 64;
const MAX_DIMENSIONS: usize = 16_384;
const CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// Scan parameters. The benchmark can sweep all 16 supported configurations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScanConfig {
    /// Threads per workgroup: 128 or 256.
    pub threads: usize,
    /// Independent rows per wave: 1, 2, 4, or 8.
    pub rows_per_wave: usize,
    /// Adjacent FP16 components per lane per load: 2 or 4.
    pub load_width: usize,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            threads: 128,
            rows_per_wave: 2,
            load_width: 4,
        }
    }
}

impl ScanConfig {
    fn validate(self) -> Result<()> {
        if ![128, 256].contains(&self.threads)
            || ![1, 2, 4, 8].contains(&self.rows_per_wave)
            || ![2, 4].contains(&self.load_width)
        {
            return Err(invalid("unsupported scan configuration"));
        }
        Ok(())
    }

    /// Enumerate all 16 supported schedules in a stable order.
    pub fn configurations() -> impl Iterator<Item = Self> {
        [128, 256].into_iter().flat_map(|threads| {
            [1, 2, 4, 8].into_iter().flat_map(move |rows_per_wave| {
                [2, 4].into_iter().map(move |load_width| Self {
                    threads,
                    rows_per_wave,
                    load_width,
                })
            })
        })
    }
}

/// A ranked match, ordered by descending score then ascending ID.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Neighbor {
    /// Zero-based corpus insertion position, or score column for standalone [`TopK`].
    pub id: u32,
    /// Unclamped FP32 cosine score, or the application score supplied to [`TopK`].
    pub similarity: f32,
}

/// Host-completion measurements; these are not GPU timestamps.
#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    /// Scoring plus host submission/completion, in milliseconds.
    pub scan_ms: f64,
    /// Complete search, including query preparation and result readback, in milliseconds.
    pub search_ms: f64,
    /// Matching checksum kernel plus host submission/completion, in milliseconds.
    pub read_control_ms: f64,
    /// Matches returned by the measured full-search invocation.
    pub neighbors: Vec<Neighbor>,
}

/// Compiler artifacts and resource reports for reproducible tuning.
#[derive(Clone, Debug, Serialize)]
pub struct Compilation {
    /// Exported kernel symbol.
    pub symbol: String,
    /// Local artifact path. This may contain a user-specific cache directory.
    pub artifact: String,
    /// Optional resource/interface manifest produced by Loom.
    pub report: Option<serde_json::Value>,
    /// Compiler messages retained from successful compilation.
    pub diagnostics: Vec<String>,
}

/// Independent search worker over a shared immutable [`Corpus`].
///
/// Owns an ordered stream, compiled kernels, and reusable query/result workspace.
/// A searcher is [`Send`] but not [`Sync`]; operations require `&mut self`. Create
/// multiple workers from the same corpus to submit queries independently.
pub struct Searcher {
    corpus: Corpus,
    stream: Stream,
    compiler: hrx::loom::Compiler,
    scans: Vec<(Kernel, Kernel)>,
    query: HostBuffer,
    readback: HostBuffer,
    exclusions: HostBuffer,
    scores: Buffer,
    candidates: [(Buffer, Buffer); 2],
    first_select: Kernel,
    merge_select: Kernel,
    sort_select: Kernel,
    sorted_merge: Kernel,
    mask_scores: Kernel,
    selection_capacity: usize,
    dimensions: usize,
    padded: usize,
    count: usize,
    config: ScanConfig,
    reports: Vec<Compilation>,
    batch: Option<batch::BatchScratch>,
    device_queries: Option<queries::QueryWorkspace>,
}

struct Shard {
    data: Buffer,
    norms: Buffer,
    start: usize,
    count: usize,
    capacity: usize,
}

fn invalid(message: &str) -> Error {
    Error::Message(message.into())
}

fn layout(dimensions: usize, count: usize) -> Result<(usize, usize)> {
    if dimensions == 0 || dimensions > MAX_DIMENSIONS {
        return Err(invalid("dimensions must be in 1..=16384"));
    }
    if count > MAX_ROWS {
        return Err(invalid("row count exceeds 2^30"));
    }
    let padded = dimensions.div_ceil(128) * 128;
    let bytes = count
        .checked_mul(padded)
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("corpus allocation size overflow"))?;
    Ok((padded, bytes))
}

fn shard_rows(padded: usize, element_limit: usize) -> usize {
    // Reserve room for whole 256-row tiles without exceeding Loom's 32-bit
    // element limit. Smaller logical limits allow tests to force unaligned
    // interior shards; their padded allocations still obey the real limit.
    let maximum = (MAX_ELEMENTS / padded) / SHARD_ROW_ALIGNMENT * SHARD_ROW_ALIGNMENT;
    (element_limit / padded).max(1).min(maximum)
}

fn shard_capacity(rows: usize) -> usize {
    rows.div_ceil(SHARD_ROW_ALIGNMENT) * SHARD_ROW_ALIGNMENT
}

fn norm(row: &[f32], dimensions: usize) -> Result<f64> {
    if row.len() != dimensions {
        return Err(invalid("vector dimension mismatch"));
    }
    if row.iter().any(|x| !x.is_finite()) {
        return Err(invalid("vectors must be finite"));
    }
    let sum: f64 = row.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if sum == 0.0 {
        return Err(invalid("zero-norm vectors are not supported"));
    }
    Ok(sum.sqrt())
}

fn encode(row: &[f32], dimensions: usize, padded: usize, bytes: &mut Vec<u8>) -> Result<f32> {
    let length = norm(row, dimensions)?;
    let mut square_sum = 0.0f64;
    for &x in row {
        let h = f16::from_f64(x as f64 / length);
        let value = h.to_f64();
        square_sum += value * value;
        bytes.extend_from_slice(&h.to_le_bytes());
    }
    bytes.resize(bytes.len() + (padded - dimensions) * 2, 0);
    Ok((1.0 / square_sum.sqrt()) as f32)
}

fn encode_fp16(row: &[u8], dimensions: usize, padded: usize, bytes: &mut Vec<u8>) -> Result<f32> {
    if row.len() != dimensions * 2 {
        return Err(invalid(
            "FP16 row must contain exactly dimensions * 2 bytes",
        ));
    }
    let mut square_sum = 0.0f64;
    for component in row.as_chunks::<2>().0 {
        let value = f16::from_le_bytes(*component).to_f64();
        if !value.is_finite() {
            return Err(invalid("vectors must be finite"));
        }
        square_sum += value * value;
    }
    if square_sum == 0.0 {
        return Err(invalid("zero-norm vectors are not supported"));
    }
    bytes.extend_from_slice(row);
    bytes.resize(bytes.len() + (padded - dimensions) * 2, 0);
    Ok((1.0 / square_sum.sqrt()) as f32)
}

fn compile(
    compiler: &hrx::loom::Compiler,
    stream: &Stream,
    source: &str,
    spec: hrx::loom::Specialization,
) -> Result<(Kernel, Compilation)> {
    let artifact = compiler.module(source).compile(&spec)?;
    let report = Compilation {
        symbol: spec.symbol().to_owned(),
        artifact: artifact.path().display().to_string(),
        report: artifact.report().map(|report| report.json().clone()),
        diagnostics: artifact
            .diagnostics()
            .iter()
            .map(|d| d.message.clone())
            .collect(),
    };
    // SAFETY: source is authored in this crate and specialized for this device.
    let kernel = unsafe { stream.load_artifact(&artifact)? };
    Ok((kernel, report))
}

fn allocate_candidates(stream: &Stream, count: usize, k: usize) -> Result<[(Buffer, Buffer); 2]> {
    let bytes = count.div_ceil(1024).max(1) * k * 4;
    Ok([
        (stream.allocate(bytes)?, stream.allocate(bytes)?),
        (stream.allocate(bytes)?, stream.allocate(bytes)?),
    ])
}

impl Searcher {
    /// Build from a sized stream of rows without retaining a host corpus copy.
    /// A row is accepted as any `AsRef<[f32]>`, including borrowed slices.
    ///
    /// Rows are normalized and quantized to FP16. Dimensions must be 1–16,384;
    /// the iterator must accurately report at most 2^30 rows. Larger corpora are
    /// split internally into allocations of at most 2^32 FP16 elements, keeping
    /// scan addresses within the compiler's element-index limit. IDs and query
    /// results span the entire corpus. Vector and inverse-norm allocations
    /// reserve whole 256-row tiles; extra rows are readable slack, not indexed
    /// data. See [`CorpusShardView::capacity_rows`].
    /// The default scan schedule is tuned for 10M × 384 vectors. Construction
    /// allocates device storage, uploads the corpus, and compiles the kernels
    /// synchronously.
    ///
    /// # Errors
    /// Returns an error for an unsupported device, invalid dimensions/row count,
    /// nonfinite or zero-norm rows, mismatched row lengths, or runtime/compiler
    /// failures. Allocation and compiler limits depend on the system and shape.
    pub fn build<I, R>(device: &Device, dimensions: usize, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        Self::build_with_config(device, dimensions, rows, ScanConfig::default())
    }

    /// Build from rows of little-endian IEEE 754 binary16 bytes.
    ///
    /// Each row must contain exactly `dimensions * 2` bytes, without padding.
    /// Values are copied unchanged, zero padding is added, and inverse norms
    /// are computed from the supplied FP16 values. Rows need not be normalized.
    /// This avoids an FP32 row allocation and re-quantization during ingestion.
    /// Scores describe the supplied representation, which may differ from
    /// normalizing FP32 values before rounding in [`Self::build`].
    ///
    /// ```no_run
    /// # use hrxdb::{Device, Searcher};
    /// # fn main() -> hrxdb::Result<()> {
    /// let device = Device::open(0)?;
    /// // Two 2D rows: [1, 0] and [0, 1], stored as little-endian FP16.
    /// let bytes = [0x00, 0x3c, 0, 0, 0, 0, 0x00, 0x3c];
    /// let mut db = Searcher::build_fp16(&device, 2, bytes.as_chunks::<4>().0)?;
    /// let mut scores = vec![0.0; db.len()];
    /// db.scores_into(&[1.0, 0.0], &mut scores)?;
    /// assert_eq!(scores, [1.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// The shape, iterator, device, and runtime requirements are the same as
    /// [`Self::build`]. Incorrect byte lengths, nonfinite values, and zero-norm
    /// rows are rejected.
    pub fn build_fp16<I, R>(device: &Device, dimensions: usize, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        Self::build_fp16_with_config(device, dimensions, rows, ScanConfig::default())
    }

    /// Build from FP16 byte rows with an explicit scan schedule.
    ///
    /// Has the same input requirements and errors as [`Self::build_fp16`], and
    /// also rejects unsupported [`ScanConfig`] field values.
    pub fn build_fp16_with_config<I, R>(
        device: &Device,
        dimensions: usize,
        rows: I,
        config: ScanConfig,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        Self::build_encoded(
            device,
            dimensions,
            rows,
            config,
            MAX_ELEMENTS,
            |row, d, p, bytes| encode_fp16(row.as_ref(), d, p, bytes),
        )
    }

    /// Build an index with an explicit scan schedule.
    ///
    /// Has the same input requirements and errors as [`Self::build`], and also
    /// rejects unsupported [`ScanConfig`] field values.
    pub fn build_with_config<I, R>(
        device: &Device,
        dimensions: usize,
        rows: I,
        config: ScanConfig,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        Self::build_encoded(
            device,
            dimensions,
            rows,
            config,
            MAX_ELEMENTS,
            |row, d, p, bytes| encode(row.as_ref(), d, p, bytes),
        )
    }

    fn build_encoded<I, R>(
        device: &Device,
        dimensions: usize,
        rows: I,
        config: ScanConfig,
        element_limit: usize,
        encode_row: impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
    {
        config.validate()?;
        Self::with_config(
            Corpus::build_encoded(device, dimensions, rows, element_limit, encode_row)?,
            config,
        )
    }

    /// Create an independent searcher over shared corpus storage.
    pub fn new(corpus: Corpus) -> Result<Self> {
        Self::with_config(corpus, ScanConfig::default())
    }

    /// Create a searcher with a chosen single-query scan schedule.
    pub fn with_config(corpus: Corpus, config: ScanConfig) -> Result<Self> {
        let stream = corpus.stream()?;
        Self::on_stream(corpus, stream, config)
    }

    /// Take ownership of a caller's stream to compose custom kernels and search.
    /// The stream must belong to the corpus's device. Pending work stays ordered.
    pub fn on_stream(corpus: Corpus, stream: Stream, config: ScanConfig) -> Result<Self> {
        config.validate()?;
        if stream.device_id() != corpus.inner.device_id {
            return Err(invalid("stream belongs to another device"));
        }
        let count = corpus.len();
        let dimensions = corpus.dimensions();
        let padded = corpus.padded_dimensions();
        let compiler = hrx::loom::Compiler::for_stream(None, &stream)?;
        let mut scans = Vec::new();
        let mut reports = Vec::new();
        for shard in &corpus.inner.shards {
            let (scan, sr) = compile(
                &compiler,
                &stream,
                kernels::SCAN,
                kernels::scan_spec(shard.count, padded, config, false),
            )?;
            let (control, cr) = compile(
                &compiler,
                &stream,
                kernels::SCAN,
                kernels::scan_spec(shard.count, padded, config, true),
            )?;
            scans.push((scan, control));
            reports.extend([sr, cr]);
        }
        let query = HostBuffer::new(&stream, padded * 4)?;
        let readback = HostBuffer::new(&stream, MAX_K * 8)?;
        let exclusions = HostBuffer::new(&stream, count.div_ceil(32) * 4)?;
        let scores = stream.allocate(count * 4)?;
        let candidates = allocate_candidates(&stream, count, 32)?;
        let (first_select, fr) = compile(
            &compiler,
            &stream,
            kernels::SELECT,
            kernels::select_spec(true),
        )?;
        let (merge_select, mr) = compile(
            &compiler,
            &stream,
            kernels::SELECT,
            kernels::select_spec(false),
        )?;
        let (sort_select, ssr) = compile(
            &compiler,
            &stream,
            kernels::SORT,
            kernels::named_spec("sort_select"),
        )?;
        let (sorted_merge, smr) = compile(
            &compiler,
            &stream,
            kernels::SORT,
            kernels::named_spec("sorted_merge"),
        )?;
        let (mask_scores, er) = compile(
            &compiler,
            &stream,
            kernels::MASK,
            kernels::named_spec("mask_scores"),
        )?;
        reports.extend([fr, mr, ssr, smr, er]);
        Ok(Self {
            corpus,
            stream,
            compiler,
            scans,
            query,
            readback,
            exclusions,
            scores,
            candidates,
            first_select,
            merge_select,
            sort_select,
            sorted_merge,
            mask_scores,
            selection_capacity: 32,
            dimensions,
            padded,
            count,
            config,
            reports,
            batch: None,
            device_queries: None,
        })
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.count
    }
    /// Whether the index has no vectors.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Logical vector length supplied at construction.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    /// Stored vector length, rounded up to a multiple of 128 with zeros.
    pub fn padded_dimensions(&self) -> usize {
        self.padded
    }
    /// Number of independently addressed corpus allocations (zero when empty).
    pub fn shard_count(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.corpus.inner.shards.len()
        }
    }
    /// Currently selected scan schedule.
    pub fn config(&self) -> ScanConfig {
        self.config
    }
    /// Scoring/read-control report pairs in shard order, followed by small-k
    /// selection/merge, large-k sorting/merge, and exclusion masking reports.
    /// Batch kernel reports are appended as query widths are prepared.
    /// Artifact paths refer to the local HRX cache.
    pub fn compilation_reports(&self) -> &[Compilation] {
        &self.reports
    }

    /// Compile a new scan configuration outside query timing; retains the corpus.
    /// Returns an error for an invalid schedule or synchronization/compilation
    /// failure. The previous schedule is retained if compilation fails.
    /// This configures single-query scans; the batch matrix kernel uses its own
    /// schedule, independent of [`ScanConfig`].
    pub fn configure(&mut self, config: ScanConfig) -> Result<()> {
        config.validate()?;
        self.stream.synchronize()?;
        let mut compiled = Vec::with_capacity(self.corpus.inner.shards.len());
        for shard in &self.corpus.inner.shards {
            let (scan, sr) = compile(
                &self.compiler,
                &self.stream,
                kernels::SCAN,
                kernels::scan_spec(shard.count.max(1), self.padded, config, false),
            )?;
            let (control, cr) = compile(
                &self.compiler,
                &self.stream,
                kernels::SCAN,
                kernels::scan_spec(shard.count.max(1), self.padded, config, true),
            )?;
            compiled.push((scan, sr, control, cr));
        }
        for (i, (shard, (scan, sr, control, cr))) in self.scans.iter_mut().zip(compiled).enumerate()
        {
            *shard = (scan, control);
            self.reports[i * 2] = sr;
            self.reports[i * 2 + 1] = cr;
        }
        self.config = config;
        Ok(())
    }

    fn prepare_query(&mut self, query: &[f32], k: usize) -> Result<usize> {
        if !(1..=MAX_K).contains(&k) {
            return Err(invalid("k must be in 1..=1024"));
        }
        let length = norm(query, self.dimensions)?;
        self.stream.synchronize()?;
        // SAFETY: the preceding synchronization completed all reads of the query.
        let target = unsafe { self.query.bytes_mut() };
        for (&x, bytes) in query.iter().zip(target.as_chunks_mut::<4>().0) {
            bytes.copy_from_slice(&((x as f64 / length) as f32).to_le_bytes());
        }
        Ok(k.min(self.count))
    }

    fn dispatch_scan(&self, control: bool) -> Result<()> {
        self.dispatch_scan_from(control, self.query.buffer().binding())
    }

    fn dispatch_scan_from(&self, control: bool, query: hrx::View<'_>) -> Result<()> {
        if self.count == 0 {
            return Ok(());
        }
        for (shard, (scan, control_kernel)) in self.corpus.inner.shards.iter().zip(&self.scans) {
            let groups = shard
                .count
                .div_ceil(self.config.threads / 32 * self.config.rows_per_wave);
            // SAFETY: each data shard fits the 32-bit element address limit.
            // Norms are shard-local; scores use the global row range. The
            // specialized kernel guards tail rows. One stream orders accesses.
            unsafe {
                self.stream.dispatch(
                    if control { control_kernel } else { scan },
                    [groups as u32, 1, 1],
                    [self.config.threads as u32, 1, 1],
                    &Constants::new(),
                    &[
                        shard.data.binding(),
                        query,
                        shard.norms.binding(),
                        self.scores.try_slice(shard.start * 4, shard.count * 4)?,
                    ],
                )?;
            }
        }
        Ok(())
    }

    fn select(&self, k: usize) -> Result<usize> {
        if k > 32 {
            return self.select_sorted(k);
        }
        let mut count = self.count;
        let mut output = 0;
        let mut first = true;
        loop {
            let groups = count.div_ceil(1024);
            let mut constants = Constants::new();
            constants.push(count as u32)?;
            constants.push(k as u32)?;
            constants.push(0u32)?;
            let (input_scores, input_ids) = if first {
                (&self.scores, &self.scores)
            } else {
                (
                    &self.candidates[1 - output].0,
                    &self.candidates[1 - output].1,
                )
            };
            let (out_scores, out_ids) = &self.candidates[output];
            // SAFETY: each group accesses at most 1024 guarded input candidates
            // and writes k outputs. Both scratch pairs reserve the maximum first
            // level size at k=32. Later levels shrink; ping-pong avoids aliasing.
            unsafe {
                self.stream.dispatch(
                    if first {
                        &self.first_select
                    } else {
                        &self.merge_select
                    },
                    [groups as u32, 1, 1],
                    [256, 1, 1],
                    &constants,
                    &[
                        input_scores.binding(),
                        input_ids.binding(),
                        out_scores.binding(),
                        out_ids.binding(),
                    ],
                )?;
            }
            if groups == 1 {
                return Ok(output);
            }
            count = groups * k;
            output = 1 - output;
            first = false;
        }
    }

    /// Reserve selection scratch for queries returning up to `k` neighbors.
    ///
    /// Construction reserves k=32. Searches grow this capacity automatically;
    /// call this during setup to keep allocation out of the first larger query.
    /// Scratch is retained for subsequent queries. No kernels are compiled.
    ///
    /// # Errors
    /// `k` must be 1–1,024. Runtime allocation or synchronization errors propagate.
    pub fn reserve_search(&mut self, k: usize) -> Result<()> {
        if !(1..=MAX_K).contains(&k) {
            return Err(invalid("k must be in 1..=1024"));
        }
        let capacity = k.min(self.count).next_power_of_two().min(MAX_K);
        if capacity > self.selection_capacity {
            self.stream.synchronize()?;
            let candidates = allocate_candidates(&self.stream, self.count, capacity)?;
            self.candidates = candidates;
            self.selection_capacity = capacity;
        }
        Ok(())
    }

    fn select_sorted(&self, k: usize) -> Result<usize> {
        let mut groups = self.count.div_ceil(1024);
        let mut constants = Constants::new();
        constants.push(self.count as u32)?;
        constants.push(k as u32)?;
        constants.push(0u32)?;
        // SAFETY: each group sorts 1024 guarded scores in workgroup memory and
        // writes k candidates; reserve_search sizes both pairs for this level.
        unsafe {
            self.stream.dispatch(
                &self.sort_select,
                [groups as u32, 1, 1],
                [256, 1, 1],
                &constants,
                &[
                    self.scores.binding(),
                    self.candidates[0].0.binding(),
                    self.candidates[0].1.binding(),
                ],
            )?;
        }
        let mut output = 0;
        while groups > 1 {
            let next = 1 - output;
            let mut constants = Constants::new();
            constants.push(groups as u32)?;
            constants.push(k as u32)?;
            // SAFETY: merge reads pairs of sorted k-lists, guarding an odd last
            // group, and writes their top k to disjoint, smaller output lists.
            // The buffers ping-pong so reads never alias writes.
            unsafe {
                self.stream.dispatch(
                    &self.sorted_merge,
                    [groups.div_ceil(2) as u32, 1, 1],
                    [256, 1, 1],
                    &constants,
                    &[
                        self.candidates[output].0.binding(),
                        self.candidates[output].1.binding(),
                        self.candidates[next].0.binding(),
                        self.candidates[next].1.binding(),
                    ],
                )?;
            }
            groups = groups.div_ceil(2);
            output = next;
        }
        Ok(output)
    }

    fn apply_exclusions(&self) -> Result<()> {
        self.apply_exclusions_from(self.exclusions.buffer().binding())
    }

    fn apply_exclusions_from(&self, bitmap: hrx::View<'_>) -> Result<()> {
        let mut constants = Constants::new();
        constants.push(self.count as u32)?;
        // SAFETY: mask_scores guards every row, reads ceil(count/32) bitmap
        // words, and writes only the corresponding owned score allocation.
        unsafe {
            self.stream.dispatch(
                &self.mask_scores,
                [self.count.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                &constants,
                &[bitmap, self.scores.binding()],
            )
        }
    }

    fn read_neighbors_into(
        &mut self,
        output: usize,
        k: usize,
        neighbors: &mut Vec<Neighbor>,
    ) -> Result<()> {
        self.stream.copy(
            self.readback.buffer().try_slice(0, k * 4)?,
            self.candidates[output].0.try_slice(0, k * 4)?,
        )?;
        self.stream.copy(
            self.readback.buffer().try_slice(MAX_K * 4, k * 4)?,
            self.candidates[output].1.try_slice(0, k * 4)?,
        )?;
        self.stream.synchronize()?;
        // SAFETY: the copies into readback completed on this stream.
        let bytes = unsafe { self.readback.bytes() };
        neighbors.clear();
        neighbors.reserve(k);
        for i in 0..k {
            let neighbor = Neighbor {
                id: u32::from_le_bytes(
                    bytes[MAX_K * 4 + i * 4..MAX_K * 4 + i * 4 + 4]
                        .try_into()
                        .unwrap(),
                ),
                similarity: f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            };
            if neighbor.similarity != f32::NEG_INFINITY {
                neighbors.push(neighbor);
            }
        }
        Ok(())
    }

    /// Search with device-side top-k, without compiling kernels.
    ///
    /// Returns `min(k, self.len())` matches ordered by descending computed cosine
    /// score, then ascending insertion ID. The query is normalized to FP32.
    /// FP16 storage and FP32 arithmetic can change rankings from the original
    /// vectors. This method blocks until results are readable on the host.
    /// Selection scratch grows on the first larger-k query and is reused;
    /// [`Self::reserve_search`] can reserve it before serving queries.
    ///
    /// # Errors
    /// The query must have [`Self::dimensions`] finite components and nonzero
    /// norm; `k` must be 1–1,024, including for an empty index. Runtime failures
    /// propagate as errors.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_excluding(query, k, &[])
    }

    /// Search while excluding insertion IDs, with device-side selection.
    ///
    /// Returns up to `k` remaining rows, ordered as in [`Self::search`]. Excluded
    /// IDs may be unsorted and repeated. Exclusions apply only to this query;
    /// pass the growing visited set on each step of a similarity walk. Only
    /// the selected neighbors are read back, including when k exceeds 32.
    ///
    /// ```no_run
    /// # fn step(db: &mut hrxdb::Searcher, query: &[f32], visited: &mut Vec<u32>) -> hrxdb::Result<()> {
    /// if let Some(next) = db.search_excluding(query, 1, visited)?.first() {
    ///     visited.push(next.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// In addition to [`Self::search`]'s requirements, all excluded IDs must be
    /// less than [`Self::len`]. Invalid input leaves the index usable.
    pub fn search_excluding(
        &mut self,
        query: &[f32],
        k: usize,
        excluded: &[u32],
    ) -> Result<Vec<Neighbor>> {
        let mut output = Vec::new();
        self.search_excluding_into(query, k, excluded, &mut output)?;
        Ok(output)
    }

    /// Reuse a host output vector; successful calls replace its contents while
    /// retaining capacity. Invalid inputs leave it unchanged; runtime errors may
    /// leave partial output. This waits for GPU completion.
    pub fn search_into(
        &mut self,
        query: &[f32],
        k: usize,
        output: &mut Vec<Neighbor>,
    ) -> Result<()> {
        self.search_excluding_into(query, k, &[], output)
    }

    /// Search with exclusions into reusable host output. See `search_into`.
    pub fn search_excluding_into(
        &mut self,
        query: &[f32],
        k: usize,
        excluded: &[u32],
        output: &mut Vec<Neighbor>,
    ) -> Result<()> {
        if excluded.iter().any(|&id| id as usize >= self.count) {
            return Err(invalid("excluded ID is outside the index"));
        }
        let mut k = self.prepare_query(query, k)?;
        if k == 0 {
            self.stream.synchronize()?;
            output.clear();
            return Ok(());
        }
        if !excluded.is_empty() {
            // SAFETY: prepare_query completed the stream before this host
            // write. The index owns the bitmap and retains it through dispatch.
            let bitmap = unsafe { self.exclusions.bytes_mut() };
            bitmap.fill(0);
            let mut remaining = self.count;
            for &id in excluded {
                let byte = &mut bitmap[id as usize / 8];
                let bit = 1 << (id % 8);
                if *byte & bit == 0 {
                    remaining -= 1;
                    *byte |= bit;
                }
            }
            k = k.min(remaining);
            if k == 0 {
                output.clear();
                return Ok(());
            }
        }
        self.reserve_search(k)?;
        self.dispatch_scan(false)?;
        if !excluded.is_empty() {
            self.apply_exclusions()?;
        }
        let selected = self.select(k)?;
        self.read_neighbors_into(selected, k, output)
    }

    /// Measure separate, completed control/scan/full-search invocations.
    ///
    /// Uses host wall time rather than GPU timestamps. Control and scan exclude
    /// query preparation; full search includes it. Inputs and errors are the
    /// same as [`Self::search`]. No warmups are performed by this method.
    pub fn measure(&mut self, query: &[f32], k: usize) -> Result<Measurement> {
        self.measure_excluding(query, k, &[])
    }

    /// Measure a query with exclusions, using separate control/scan/search runs.
    ///
    /// Timing has the same semantics as [`Self::measure`]; the complete search
    /// includes bitmap preparation and device masking. Call [`Self::reserve_search`]
    /// first to exclude scratch growth from the measurement.
    /// Inputs and errors are the same as [`Self::search_excluding`].
    pub fn measure_excluding(
        &mut self,
        query: &[f32],
        k: usize,
        excluded: &[u32],
    ) -> Result<Measurement> {
        if excluded.iter().any(|&id| id as usize >= self.count) {
            return Err(invalid("excluded ID is outside the index"));
        }
        self.prepare_query(query, k)?;
        self.stream.synchronize()?;
        let start = Instant::now();
        self.dispatch_scan(true)?;
        self.stream.synchronize()?;
        let read_control_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        self.dispatch_scan(false)?;
        self.stream.synchronize()?;
        let scan_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let neighbors = self.search_excluding(query, k, excluded)?;
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(Measurement {
            scan_ms,
            search_ms,
            read_control_ms,
            neighbors,
        })
    }

    /// Return every cosine score in insertion-ID order for host-side selection.
    ///
    /// Returns one FP32 score per insertion ID and allocates a full host score
    /// array. Use [`Self::scores_into`] to reuse a caller-owned buffer across
    /// queries. Both methods support larger result sets, exclusions, and grouped
    /// results by letting the caller select from the scores on the host.
    /// The query has the same validation requirements as [`Self::search`].
    pub fn scores(&mut self, query: &[f32]) -> Result<Vec<f32>> {
        let mut scores = vec![0.0; self.count];
        self.scores_into(query, &mut scores)?;
        Ok(scores)
    }

    /// Write every cosine score in insertion-ID order into a reusable buffer.
    ///
    /// Transfers `self.len() * 4` bytes from the device and blocks until the
    /// output is readable. No host score array or GPU buffer is allocated by
    /// this method. Selection, exclusions, and grouping are left to the caller;
    /// [`Self::search`] uses device selection for its supported k range.
    ///
    /// # Errors
    /// `output.len()` must equal [`Self::len`], including for an empty index.
    /// The query has the same validation requirements as [`Self::search`].
    /// Invalid inputs leave the output unchanged. Runtime failures propagate
    /// as errors and may leave the output partially written.
    pub fn scores_into(&mut self, query: &[f32], output: &mut [f32]) -> Result<()> {
        if output.len() != self.count {
            return Err(invalid(
                "score output length must equal the index row count",
            ));
        }
        self.prepare_query(query, 1)?;
        if self.is_empty() {
            return Ok(());
        }
        self.dispatch_scan(false)?;
        // SAFETY: output is an exclusively borrowed, initialized f32 slice;
        // all bit patterns are valid f32 values. The byte slice covers exactly
        // that allocation, and read_blocking completes before the borrow ends.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(output),
            )
        };
        self.stream
            .read_blocking(self.scores.try_slice(0, bytes.len())?, bytes)?;
        for value in output {
            *value = f32::from_bits(u32::from_le(value.to_bits()));
        }
        Ok(())
    }
}

impl Drop for Searcher {
    fn drop(&mut self) {
        if self.stream.synchronize().is_err() {
            self.query.abandon();
            self.readback.abandon();
            self.exclusions.abandon();
            if let Some(batch) = &mut self.batch {
                batch.abandon();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_can_move_between_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<Searcher>();
    }
    #[test]
    fn layout_and_shards_cover_large_corpora() {
        for dimensions in (128..=MAX_DIMENSIONS).step_by(128) {
            let padded = dimensions.div_ceil(128) * 128;
            let maximum = shard_rows(padded, MAX_ELEMENTS);
            assert!(maximum * padded <= MAX_ELEMENTS);
            assert!(maximum.is_multiple_of(SHARD_ROW_ALIGNMENT));
            assert!((maximum + SHARD_ROW_ALIGNMENT) * padded > MAX_ELEMENTS);
            for count in [0, maximum - 1, maximum, maximum + 1, MAX_ROWS] {
                assert_eq!(
                    layout(dimensions, count).unwrap(),
                    (padded, count * padded * 2)
                );
                let mut covered = 0;
                for start in (0..count).step_by(maximum) {
                    let size = (count - start).min(maximum);
                    assert_eq!(start, covered);
                    assert!(size * padded <= MAX_ELEMENTS);
                    let capacity = shard_capacity(size);
                    assert!(capacity >= size);
                    assert!(capacity - size < SHARD_ROW_ALIGNMENT);
                    assert!(capacity.is_multiple_of(SHARD_ROW_ALIGNMENT));
                    assert!(capacity * padded <= MAX_ELEMENTS);
                    covered += size;
                }
                assert_eq!(covered, count);
            }
        }
        assert_eq!(shard_rows(512, MAX_ELEMENTS), 8_388_608);
        assert_eq!(shard_rows(384, MAX_ELEMENTS), 11_184_640);
        assert_eq!(shard_capacity(0), 0);
        assert_eq!(layout(512, 8_942_135).unwrap(), (512, 9_156_746_240));
    }

    #[test]
    fn fp16_encoding_preserves_bytes_and_corrects_norms() {
        // Include negative zero, the smallest subnormal, and the largest finite
        // value. None should be normalized or quantized again.
        let bits = [0x4200u16, 0xc400, 0x8000, 0x0001, 0x7bff];
        let row: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut bytes = vec![42];
        let inverse = encode_fp16(&row, bits.len(), 128, &mut bytes).unwrap();
        assert_eq!(&bytes[1..1 + row.len()], row);
        assert_eq!(bytes.len(), 257);
        assert!(bytes[1 + row.len()..].iter().all(|&b| b == 0));
        let square_sum: f64 = bits
            .iter()
            .map(|&b| f16::from_bits(b).to_f64().powi(2))
            .sum();
        assert!((square_sum.sqrt() * inverse as f64 - 1.0).abs() < 1e-7);
        let mut tiny = Vec::new();
        assert_eq!(
            encode_fp16(&1u16.to_le_bytes(), 1, 128, &mut tiny).unwrap(),
            16_777_216.0
        );
    }

    #[test]
    fn fp16_encoding_rejects_invalid_rows() {
        for row in [
            vec![],
            vec![0],
            vec![0, 0, 0],
            vec![0; 4],
            0u16.to_le_bytes().to_vec(),
            0x8000u16.to_le_bytes().to_vec(),
            0x7c00u16.to_le_bytes().to_vec(),
            0xfc00u16.to_le_bytes().to_vec(),
            0x7e00u16.to_le_bytes().to_vec(),
        ] {
            let mut bytes = vec![42];
            assert!(encode_fp16(&row, 1, 128, &mut bytes).is_err());
            assert_eq!(bytes, [42]);
        }
    }
    #[test]
    fn layout_and_encoding() {
        assert_eq!(layout(384, 10_000_000).unwrap(), (384, 7_680_000_000));
        assert_eq!(layout(129, 5).unwrap(), (256, 2560));
        assert!(layout(0, 1).is_err());
        assert!(layout(usize::MAX, 1).is_err());
        assert!(layout(384, MAX_ROWS + 1).is_err());
        let mut bytes = Vec::new();
        let inverse = encode(&[3.0, 4.0], 2, 128, &mut bytes).unwrap();
        let sum: f64 = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_le_bytes(*b).to_f64().powi(2))
            .sum();
        assert!((sum.sqrt() * inverse as f64 - 1.0).abs() < 1e-7);
        assert!(bytes[4..].iter().all(|&b| b == 0));
        for row in [[0.0, 0.0], [f32::NAN, 1.0], [f32::INFINITY, 1.0]] {
            assert!(encode(&row, 2, 128, &mut Vec::new()).is_err());
        }
        assert!(norm(&[1.0], 2).is_err());
        assert!(norm(&[f32::MAX, f32::MAX], 2).unwrap().is_finite());
        assert!(norm(&[f32::from_bits(1)], 1).unwrap() > 0.0);
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn every_k_matches_exact_scores_with_masked_and_padded_tails() -> Result<()> {
        let device = Device::open(0)?;
        let count = 2057;
        let mut db = Searcher::build(&device, 1, (0..count).map(|_| [1.0]))?;
        db.reserve_search(MAX_K)?;
        let values: Vec<f32> = (0..count)
            .map(|i| match i % 5 {
                0 => f32::NEG_INFINITY,
                1 => -0.0,
                _ => ((i * 747_796_405usize % 1009) as f32 - 500.0) / 1000.0,
            })
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        db.stream.upload_blocking(db.scores.binding(), &bytes)?;
        let mut expected: Vec<_> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(id, &similarity)| Neighbor {
                id: id as u32,
                similarity,
            })
            .collect();
        expected.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap()
                .then(a.id.cmp(&b.id))
        });
        for k in 1..=MAX_K {
            let output = db.select(k)?;
            let mut actual = Vec::new();
            db.read_neighbors_into(output, k, &mut actual)?;
            assert_eq!(actual, expected[..k], "k={k}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn sharded_queries_preserve_ids_scores_and_configuration() -> Result<()> {
        let device = Device::open(0)?;
        let rows: Vec<_> = (0..4099)
            .map(|i| [((i * 37) % 101) as f32 - 50.0, 1.0, (i % 7) as f32])
            .collect();
        let mut single = Searcher::build(&device, 3, &rows)?;
        let mut sharded = Searcher::build_encoded(
            &device,
            3,
            &rows,
            ScanConfig::default(),
            128 * 1025,
            |row, d, p, bytes| encode(*row, d, p, bytes),
        )?;
        assert_eq!(sharded.shard_count(), 4);
        let excluded = [0, 1, 1024, 1025, 2049, 2050, 3074, 3075, 4098, 1025];
        for config in [
            ScanConfig::default(),
            ScanConfig {
                threads: 256,
                rows_per_wave: 8,
                load_width: 2,
            },
        ] {
            sharded.configure(config)?;
            single.configure(config)?;
            for query in [[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]] {
                assert_eq!(sharded.scores(&query)?, single.scores(&query)?);
                for k in [1, 32, 33, 1024] {
                    assert_eq!(sharded.search(&query, k)?, single.search(&query, k)?);
                    assert_eq!(
                        sharded.search_excluding(&query, k, &excluded)?,
                        single.search_excluding(&query, k, &excluded)?
                    );
                }
            }
        }
        let mut encoded = Vec::new();
        for row in &rows {
            encode(row, 3, 3, &mut encoded)?;
        }
        let mut fp16 = Searcher::build_encoded(
            &device,
            3,
            encoded.as_chunks::<6>().0,
            single.config(),
            128 * 1025,
            |row, d, p, bytes| encode_fp16(*row, d, p, bytes),
        )?;
        assert_eq!(fp16.shard_count(), 4);
        assert_eq!(
            fp16.scores(&[1.0, 2.0, 3.0])?,
            single.scores(&[1.0, 2.0, 3.0])?
        );
        assert_eq!(
            fp16.search_excluding(&[1.0, 2.0, 3.0], 1024, &excluded)?,
            single.search_excluding(&[1.0, 2.0, 3.0], 1024, &excluded)?
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn selection_matches_exact_scores_across_three_levels() -> Result<()> {
        let device = Device::open(0)?;
        let count = 65_537;
        let mut db = Searcher::build(&device, 1, (0..count).map(|_| [1.0]))?;
        let values: Vec<f32> = (0..count)
            .map(|i| match i % 9 {
                0 => -f32::MAX,
                1 => -0.0,
                2 => 0.0,
                3 => f32::from_bits(1.0f32.to_bits() + 1),
                4 => 1.0,
                _ => -(i % 17) as f32,
            })
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        db.stream.upload_blocking(db.scores.binding(), &bytes)?;
        let mut expected: Vec<_> = values.iter().enumerate().collect();
        expected.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap().then(a.0.cmp(&b.0)));
        for k in (1..=32).chain([33, 63, 64, 65, 127, 255, 511, 1000, 1023, 1024]) {
            db.reserve_search(k)?;
            let output = db.select(k)?;
            let mut actual = Vec::new();
            db.read_neighbors_into(output, k, &mut actual)?;
            for (got, &(id, &value)) in actual.iter().zip(&expected) {
                assert_eq!(got.id, id as u32);
                assert_eq!(got.similarity, value);
            }
        }
        Ok(())
    }
}
