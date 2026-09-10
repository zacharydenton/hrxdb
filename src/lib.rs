//! Exhaustive GPU cosine search with FP16 storage and FP32 accumulation.
//!
//! An index owns an immutable corpus and serializes queries through `&mut self`.
//! Scores describe the quantized corpus, not the original FP32 rows.
//!
//! Execution requires Linux x86_64, a gfx1151 AMD GPU, and the native runtime
//! prerequisites in the [hrx-rs documentation](https://docs.rs/hrx-rs/0.4.0/hrx/).
//! Building and generating documentation do not initialize GPU hardware.
//!
//! ```no_run
//! use hrxdb::{Device, FlatIndex};
//!
//! # fn main() -> hrxdb::Result<()> {
//! let device = Device::open(0)?;
//! let mut index = FlatIndex::build(&device, 3, [
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//! ])?;
//! let external_ids = ["document-a", "document-b"];
//! for neighbor in index.search(&[1.0, 0.0, 0.0], 2)? {
//!     println!("{}: {}", external_ids[neighbor.id as usize], neighbor.similarity);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! IDs are insertion positions, not application IDs. The index does not persist
//! vectors, map external IDs, or support updates, filtering, or approximate search.
mod host;
mod kernels;
use host::HostBuffer;

use half::f16;
use hrx::{Buffer, Constants, Kernel, Stream};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub use hrx::{Device, Error, Result};

const MAX_ROWS: usize = 1 << 30;
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

/// A cosine match, sorted by descending similarity then ascending insertion ID.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Neighbor {
    /// Zero-based insertion position. Use this to index a separate external-ID array.
    pub id: u32,
    /// Unclamped cosine score over the quantized row using FP32 arithmetic.
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

/// Immutable GPU-resident vector corpus. Query and result scratch are reused.
///
/// The index is [`Send`]: ownership may move to another thread. Query operations
/// require `&mut self`; the index is not [`Sync`]. A mutex can serialize shared
/// ownership when needed.
pub struct FlatIndex {
    stream: Stream,
    compiler: hrx::loom::Compiler,
    data: Buffer,
    norms: Buffer,
    query: HostBuffer,
    readback: HostBuffer,
    scores: Buffer,
    candidates: [(Buffer, Buffer); 2],
    scan: Kernel,
    control: Kernel,
    first_select: Kernel,
    merge_select: Kernel,
    dimensions: usize,
    padded: usize,
    count: usize,
    config: ScanConfig,
    reports: Vec<Compilation>,
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

fn compile(
    compiler: &hrx::loom::Compiler,
    stream: &Stream,
    source: &str,
    spec: hrx::loom::Specialization,
) -> Result<(Kernel, Compilation)> {
    let artifact = compiler.module(source).compile(&spec)?;
    let report = Compilation {
        symbol: spec.symbol.clone(),
        artifact: artifact.path().display().to_string(),
        report: artifact.report().cloned(),
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

impl FlatIndex {
    /// Build from a sized stream of rows without retaining a host corpus copy.
    /// A row is accepted as any `AsRef<[f32]>`, including borrowed slices.
    ///
    /// Rows are normalized and quantized to FP16. Dimensions must be 1–16,384;
    /// the iterator must accurately report at most 2^30 rows. The default scan
    /// schedule is tuned for 10M × 384 vectors. Construction allocates device
    /// storage, uploads the corpus, and compiles the kernels synchronously.
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
        config.validate()?;
        if device.target().as_str() != "gfx1151" {
            return Err(invalid("hrxdb currently targets gfx1151"));
        }
        let mut rows = rows.into_iter();
        let count = rows.len();
        let (padded, bytes) = layout(dimensions, count)?;
        let mut stream = device.stream()?;
        let data = stream.allocate(bytes)?;
        let norms = stream.allocate(count * 4)?;
        let chunk_rows = (CHUNK_BYTES / (padded * 2)).max(1);
        let mut chunk = Vec::with_capacity(chunk_rows * padded * 2);
        let mut norm_chunk = Vec::with_capacity(chunk_rows * 4);
        for start in (0..count).step_by(chunk_rows) {
            chunk.clear();
            norm_chunk.clear();
            for _ in start..(start + chunk_rows).min(count) {
                let row = rows
                    .next()
                    .ok_or_else(|| invalid("row iterator returned fewer rows than declared"))?;
                let inverse = encode(row.as_ref(), dimensions, padded, &mut chunk)?;
                norm_chunk.extend_from_slice(&inverse.to_le_bytes());
            }
            stream.upload(data.try_slice(start * padded * 2, chunk.len())?, &chunk)?;
            stream.upload(norms.try_slice(start * 4, norm_chunk.len())?, &norm_chunk)?;
            // Bound HRX's owned staging as well as our host conversion buffers.
            stream.synchronize()?;
        }
        if rows.next().is_some() {
            return Err(invalid("row iterator returned more rows than declared"));
        }
        let query = HostBuffer::new(&stream, padded * 4)?;
        let readback = HostBuffer::new(&stream, 256)?;
        let scores = stream.allocate(count * 4)?;
        let scratch_bytes = count.div_ceil(1024).max(1) * 32 * 4;
        let candidates = [
            (
                stream.allocate(scratch_bytes)?,
                stream.allocate(scratch_bytes)?,
            ),
            (
                stream.allocate(scratch_bytes)?,
                stream.allocate(scratch_bytes)?,
            ),
        ];
        let compiler = hrx::loom::Compiler::with_options(
            None,
            hrx::loom::CompilerOptions {
                target: device.target().clone(),
                ..Default::default()
            },
        )?;
        let (scan, sr) = compile(
            &compiler,
            &stream,
            kernels::SCAN,
            kernels::scan_spec(count.max(1), padded, config, false),
        )?;
        let (control, cr) = compile(
            &compiler,
            &stream,
            kernels::SCAN,
            kernels::scan_spec(count.max(1), padded, config, true),
        )?;
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
        Ok(Self {
            stream,
            compiler,
            data,
            norms,
            query,
            readback,
            scores,
            candidates,
            scan,
            control,
            first_select,
            merge_select,
            dimensions,
            padded,
            count,
            config,
            reports: vec![sr, cr, fr, mr],
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
    /// Currently selected scan schedule.
    pub fn config(&self) -> ScanConfig {
        self.config
    }
    /// Reports for scoring, read control, initial selection, and candidate merge,
    /// in that order. Artifact paths refer to the local HRX cache.
    pub fn compilation_reports(&self) -> &[Compilation] {
        &self.reports
    }

    /// Compile a new scan configuration outside query timing; retains the corpus.
    /// Returns an error for an invalid schedule or synchronization/compilation
    /// failure. The previous schedule is retained if compilation fails.
    pub fn configure(&mut self, config: ScanConfig) -> Result<()> {
        config.validate()?;
        self.stream.synchronize()?;
        let (scan, sr) = compile(
            &self.compiler,
            &self.stream,
            kernels::SCAN,
            kernels::scan_spec(self.count.max(1), self.padded, config, false),
        )?;
        let (control, cr) = compile(
            &self.compiler,
            &self.stream,
            kernels::SCAN,
            kernels::scan_spec(self.count.max(1), self.padded, config, true),
        )?;
        self.scan = scan;
        self.control = control;
        self.config = config;
        self.reports[0] = sr;
        self.reports[1] = cr;
        Ok(())
    }

    fn prepare_query(&mut self, query: &[f32], k: usize) -> Result<usize> {
        if !(1..=32).contains(&k) {
            return Err(invalid("k must be in 1..=32"));
        }
        let length = norm(query, self.dimensions)?;
        self.stream.synchronize()?;
        // SAFETY: the preceding synchronization completed all reads of the query.
        let target = unsafe { self.query.bytes_mut() };
        for (&x, bytes) in query.iter().zip(target.chunks_exact_mut(4)) {
            bytes.copy_from_slice(&((x as f64 / length) as f32).to_le_bytes());
        }
        Ok(k.min(self.count))
    }

    fn dispatch_scan(&self, control: bool) -> Result<()> {
        if self.count == 0 {
            return Ok(());
        }
        let groups = self
            .count
            .div_ceil(self.config.threads / 32 * self.config.rows_per_wave);
        // SAFETY: dimensions and wave-uniform tail masks are generated from this
        // allocation's shape; byte addressing uses offset (64 bits). Bindings
        // are owned here and all accesses are ordered on the same stream.
        unsafe {
            self.stream.dispatch(
                if control { &self.control } else { &self.scan },
                [groups as u32, 1, 1],
                [self.config.threads as u32, 1, 1],
                &Constants::new(),
                &[
                    self.data.binding(),
                    self.query.buffer().binding(),
                    self.norms.binding(),
                    self.scores.binding(),
                ],
            )
        }
    }

    fn select(&self, k: usize) -> Result<usize> {
        let mut count = self.count;
        let mut output = 0;
        let mut first = true;
        loop {
            let groups = count.div_ceil(1024);
            let mut constants = Constants::new();
            constants.push(count as u32)?;
            constants.push(k as u32)?;
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

    fn read_neighbors(&mut self, output: usize, k: usize) -> Result<Vec<Neighbor>> {
        self.stream.copy(
            self.readback.buffer().try_slice(0, k * 4)?,
            self.candidates[output].0.try_slice(0, k * 4)?,
        )?;
        self.stream.copy(
            self.readback.buffer().try_slice(128, k * 4)?,
            self.candidates[output].1.try_slice(0, k * 4)?,
        )?;
        self.stream.synchronize()?;
        // SAFETY: the copies into readback completed on this stream.
        let bytes = unsafe { self.readback.bytes() };
        Ok((0..k)
            .map(|i| Neighbor {
                id: u32::from_le_bytes(bytes[128 + i * 4..132 + i * 4].try_into().unwrap()),
                similarity: f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            })
            .collect())
    }

    /// Search without compiling kernels or allocating corpus/selection scratch.
    ///
    /// Returns `min(k, self.len())` matches ordered by descending computed cosine
    /// score, then ascending insertion ID. The query is normalized to FP32.
    /// FP16 storage and FP32 arithmetic can change rankings from the original
    /// vectors. This method blocks until results are readable on the host.
    ///
    /// # Errors
    /// The query must have [`Self::dimensions`] finite components and nonzero
    /// norm; `k` must be 1–32, including for an empty index. Runtime failures
    /// propagate as errors.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        let k = self.prepare_query(query, k)?;
        if k == 0 {
            self.stream.synchronize()?;
            return Ok(Vec::new());
        }
        self.dispatch_scan(false)?;
        let output = self.select(k)?;
        self.read_neighbors(output, k)
    }

    /// Measure separate, completed control/scan/full-search invocations.
    ///
    /// Uses host wall time rather than GPU timestamps. Control and scan exclude
    /// query preparation; full search includes it. Inputs and errors are the
    /// same as [`Self::search`]. No warmups are performed by this method.
    pub fn measure(&mut self, query: &[f32], k: usize) -> Result<Measurement> {
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
        let neighbors = self.search(query, k)?;
        let search_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(Measurement {
            scan_ms,
            search_ms,
            read_control_ms,
            neighbors,
        })
    }

    /// Diagnostic score readback for numerical validation, not the query hot path.
    ///
    /// Returns one FP32 score per insertion ID and allocates a full host score
    /// array. The query has the same validation requirements as [`Self::search`].
    pub fn scores(&mut self, query: &[f32]) -> Result<Vec<f32>> {
        if self.is_empty() {
            self.prepare_query(query, 1)?;
            return Ok(Vec::new());
        }
        self.prepare_query(query, 1)?;
        self.dispatch_scan(false)?;
        let mut bytes = vec![0; self.count * 4];
        self.stream
            .read_blocking(self.scores.try_slice(0, bytes.len())?, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}

impl Drop for FlatIndex {
    fn drop(&mut self) {
        if self.stream.synchronize().is_err() {
            self.query.abandon();
            self.readback.abandon();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_can_move_between_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<FlatIndex>();
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
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f64().powi(2))
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
    fn selection_matches_exact_scores_across_three_levels() -> Result<()> {
        let device = Device::open(0)?;
        let count = 65_537;
        let mut db = FlatIndex::build(&device, 1, (0..count).map(|_| [1.0]))?;
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
        for k in 1..=32 {
            let output = db.select(k)?;
            let actual = db.read_neighbors(output, k)?;
            for (got, &(id, &value)) in actual.iter().zip(&expected) {
                assert_eq!(got.id, id as u32);
                assert_eq!(got.similarity, value);
            }
        }
        Ok(())
    }
}
