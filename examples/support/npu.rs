//! Experimental NPU-only retrieval. Deliberately separate from the public backend.
use half::f16;
use hrx::{
    Error, Result,
    execution::{
        Access, BindingContract, Buffer, ExecutableGraph, KernelContract, MemoryPlacement,
        NpuDevice, Runtime, RuntimeOptions,
    },
    loom::{Compiler, Specialization},
    npu::NpuKernel,
    residency::MemoryBudget,
};
use hrxdb::Neighbor;
use serde::Serialize;
use std::time::Instant;

const CHUNK: usize = 1024;
fn invalid(message: &str) -> Error {
    Error::Message(message.into())
}

pub fn compile(
    device: &NpuDevice,
    source: &str,
    symbol: &str,
    config: &[(&str, usize)],
    bindings: &[(usize, Access)],
) -> Result<NpuKernel> {
    let mut spec = Specialization::new(symbol);
    for &(name, value) in config {
        spec.set_config(name, value.to_string());
    }
    let artifact = Compiler::for_target(None, device.target())?
        .module(source)
        .compile(&spec)?;
    let contract = KernelContract {
        bindings: bindings
            .iter()
            .map(|&(bytes, access)| BindingContract {
                bytes,
                alignment: 64,
                access,
                layout: "contiguous rows; repeated query DMA".into(),
            })
            .collect(),
        constants: vec![],
    };
    // SAFETY: this helper only loads the authored probe kernels. Each caller
    // supplies their specialized extents, access modes and disjoint bindings.
    // Scan has three input DMA channels and occupies two columns; the other
    // authored pipelines have at most two inputs and occupy one column.
    unsafe { device.load_artifact(&artifact, if symbol == "scan" { 2 } else { 1 }, contract) }
}

pub fn allocate(runtime: &Runtime, device: &NpuDevice, bytes: usize) -> Result<Buffer> {
    runtime.allocate(bytes, MemoryPlacement::NpuLocal(device.clone()))
}

pub struct StoredRows {
    pub dimensions: usize,
    pub padded: usize,
    pub count: usize,
    pub vectors: Vec<u8>,
    pub inverses: Vec<f32>,
}
impl StoredRows {
    pub fn build<I, R>(dimensions: usize, rows: I, direct_fp16: bool) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        if !(1..=1024).contains(&dimensions) {
            return Err(invalid("probe dimensions must be in 1..=1024"));
        }
        let rows = rows.into_iter();
        let count = rows.len();
        if count > (1 << 30) {
            return Err(invalid("too many rows"));
        }
        let padded = dimensions.div_ceil(128) * 128;
        let mut vectors = vec![0; count.div_ceil(8) * 8 * padded * 2];
        let mut inverses = Vec::with_capacity(count);
        for (i, row) in rows.enumerate() {
            if i >= count {
                return Err(invalid("row iterator length mismatch"));
            }
            let row = row.as_ref();
            let length = norm(row, dimensions)?;
            let mut square_sum = 0.0f64;
            for (col, &value) in row.iter().enumerate() {
                let half = if direct_fp16 {
                    f16::from_f32(value)
                } else {
                    f16::from_f64(value as f64 / length)
                };
                if !half.is_finite() {
                    return Err(invalid("FP16 row is not finite"));
                }
                square_sum += half.to_f64() * half.to_f64();
                let at = (i * padded + col) * 2;
                vectors[at..at + 2].copy_from_slice(&half.to_le_bytes());
            }
            if square_sum == 0.0 {
                return Err(invalid("FP16 row has zero norm"));
            }
            inverses.push((1.0 / square_sum.sqrt()) as f32);
        }
        if inverses.len() != count {
            return Err(invalid("row iterator length mismatch"));
        }
        Ok(Self {
            dimensions,
            padded,
            count,
            vectors,
            inverses,
        })
    }
    pub fn reference(&self, query: &[f32]) -> Result<Vec<f32>> {
        let length = norm(query, self.dimensions)?;
        Ok((0..self.count)
            .map(|row| {
                let mut sum = 0.0f64;
                for (col, &q) in query.iter().enumerate() {
                    let at = (row * self.padded + col) * 2;
                    let h = f16::from_le_bytes(self.vectors[at..at + 2].try_into().unwrap());
                    sum += h.to_f64() * ((q as f64 / length) as f32) as f64;
                }
                (sum * self.inverses[row] as f64) as f32
            })
            .collect())
    }
}
fn norm(row: &[f32], dimensions: usize) -> Result<f64> {
    if row.len() != dimensions || row.iter().any(|v| !v.is_finite()) {
        return Err(invalid("invalid row dimensions or nonfinite value"));
    }
    let square: f64 = row.iter().map(|&v| (v as f64).powi(2)).sum();
    if square == 0.0 {
        return Err(invalid("zero-norm row"));
    }
    Ok(square.sqrt())
}
struct Chunk {
    scan: ExecutableGraph,
    select: ExecutableGraph,
    start: usize,
    rows: usize,
}
#[derive(Default, Serialize)]
pub struct Timings {
    pub scan_ms: f64,
    pub selection_ms: f64,
    pub total_ms: f64,
}
pub struct SearchOutput {
    pub neighbors: Vec<Vec<Neighbor>>,
    pub scores: Option<Vec<Vec<f32>>>,
    pub timings: Timings,
}
pub struct Search {
    pub runtime: Runtime,
    /// Optional exploration limit, checked only between completed chunks.
    pub deadline: Option<Instant>,
    dimensions: usize,
    count: usize,
    k: usize,
    query: Buffer,
    metadata: Vec<Buffer>,
    pairs: Buffer,
    running: [Buffer; 2],
    chunks: Vec<Chunk>,
    inverses: Vec<f32>,
    excluded: Vec<u32>,
}
impl Search {
    pub fn new(rows: &StoredRows, k: usize, budget: Option<MemoryBudget>) -> Result<Self> {
        if !(1..=1024).contains(&k) {
            return Err(invalid("k must be in 1..=1024"));
        }
        let runtime = Runtime::with_options(RuntimeOptions {
            memory_budget: budget,
            ..Default::default()
        })?;
        let device = runtime.npu(0)?;
        let query = allocate(&runtime, &device, rows.padded * 4)?;
        let mut metadata = Vec::new();
        let pairs = allocate(&runtime, &device, CHUNK * 8)?;
        // Pipeline DMA extents must be at least one 64-byte record. Extra
        // candidate slots are sentinels, and never become logical results.
        let capacity = k.div_ceil(8) * 8;
        let current = allocate(&runtime, &device, capacity * 8)?;
        let running = [
            allocate(&runtime, &device, capacity * 8)?,
            allocate(&runtime, &device, capacity * 8)?,
        ];
        let empty = allocate(&runtime, &device, capacity * 8)?;
        for pair in empty.map_write()?.chunks_exact_mut(8) {
            pair[..4].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
            pair[4..].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        let merge = compile(
            &device,
            include_str!("../../kernels/npu_merge.loom"),
            "merge",
            &[("merge.k", capacity)],
            &[
                (capacity * 8, Access::Read),
                (capacity * 8, Access::Read),
                (capacity * 8, Access::Write),
            ],
        )?;
        let mut chunks = Vec::new();
        // Cache the full-chunk kernels and compile at most one tail variant.
        let mut plans = std::collections::HashMap::new();
        for start in (0..rows.count).step_by(CHUNK) {
            let count = (rows.count - start).min(CHUNK).div_ceil(8) * 8;
            // Published XDNA images require binding offset zero. Give each
            // corpus chunk independent backing instead of binding subviews.
            let data = allocate(&runtime, &device, count * rows.padded * 2)?;
            data.map_write()?.copy_from_slice(
                &rows.vectors[start * rows.padded * 2..(start + count) * rows.padded * 2],
            );
            let meta = allocate(&runtime, &device, count * 8)?;
            {
                let mut bytes = meta.map_write()?;
                for local in 0..count {
                    let i = start + local;
                    let at = (local / 8 * 16 + local % 8) * 4;
                    bytes[at..at + 4].copy_from_slice(
                        &rows.inverses.get(i).copied().unwrap_or(-1.0).to_le_bytes(),
                    );
                    bytes[at + 32..at + 36].copy_from_slice(&(i as u32).to_le_bytes());
                }
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = plans.entry(count) {
                let scan = compile(
                    &device,
                    include_str!("../../kernels/npu_scan.loom"),
                    "scan",
                    &[
                        ("scan.records", count / 8),
                        ("scan.dimensions", rows.padded),
                    ],
                    &[
                        (count * rows.padded * 2, Access::Read),
                        (rows.padded * 4, Access::Read),
                        (count * 8, Access::Read),
                        (count * 8, Access::Write),
                    ],
                )?;
                let select = compile(
                    &device,
                    include_str!("../../kernels/npu_select.loom"),
                    "select",
                    &[("select.items", count), ("select.k", capacity)],
                    &[(count * 8, Access::Read), (capacity * 8, Access::Write)],
                )?;
                entry.insert((scan, select));
            }
            let (scan_kernel, select_kernel) = &plans[&count];
            let mut scan = runtime.graph();
            scan.npu(
                scan_kernel,
                &[
                    data.view(),
                    query.view(),
                    meta.view(),
                    pairs.slice(0..count * 8)?,
                ],
            )?;
            let index = chunks.len();
            let mut select = runtime.graph();
            select.npu(select_kernel, &[pairs.slice(0..count * 8)?, current.view()])?;
            select.npu(
                &merge,
                &[
                    if index == 0 {
                        empty.view()
                    } else {
                        running[(index - 1) % 2].view()
                    },
                    current.view(),
                    running[index % 2].view(),
                ],
            )?;
            let scan = scan.prepare().map_err(|error| {
                invalid(&format!(
                    "NPU scan preparation at row {start}, after {} prepared chunks: {error}",
                    chunks.len()
                ))
            })?;
            let select = select.prepare().map_err(|error| {
                invalid(&format!(
                    "NPU selection preparation at row {start}, after {} prepared chunks: {error}",
                    chunks.len()
                ))
            })?;
            chunks.push(Chunk {
                scan,
                select,
                start,
                rows: count,
            });
            metadata.push(meta);
        }
        Ok(Self {
            runtime,
            deadline: None,
            dimensions: rows.dimensions,
            count: rows.count,
            k,
            query,
            metadata,
            pairs,
            running,
            chunks,
            inverses: rows.inverses.clone(),
            excluded: Vec::new(),
        })
    }
    pub fn search(
        &mut self,
        queries: &[f32],
        excluded: &[u32],
        diagnostic: bool,
    ) -> Result<SearchOutput> {
        if !queries.len().is_multiple_of(self.dimensions) || queries.len() / self.dimensions > 64 {
            return Err(invalid("query batch must have 0..=64 complete rows"));
        }
        let lengths = queries
            .chunks_exact(self.dimensions)
            .map(|q| norm(q, self.dimensions))
            .collect::<Result<Vec<_>>>()?;
        if excluded.iter().any(|&id| id as usize >= self.count) {
            return Err(invalid("excluded ID out of range"));
        }
        let start = Instant::now();
        if excluded != self.excluded {
            for (&id, inverse) in self
                .excluded
                .iter()
                .map(|id| (id, self.inverses[*id as usize]))
                .chain(excluded.iter().map(|id| (id, -1.0)))
            {
                let id = id as usize;
                let mut bytes = self.metadata[id / CHUNK].map_write()?;
                let local = id % CHUNK;
                let at = (local / 8 * 16 + local % 8) * 4;
                bytes[at..at + 4].copy_from_slice(&inverse.to_le_bytes());
            }
            self.excluded = excluded.to_vec();
        }
        let mut output = SearchOutput {
            neighbors: Vec::new(),
            scores: diagnostic.then(Vec::new),
            timings: Timings::default(),
        };
        for (query, length) in queries.chunks_exact(self.dimensions).zip(lengths) {
            {
                let mut bytes = self.query.map_write()?;
                bytes.fill(0);
                for (word, &q) in bytes.chunks_exact_mut(4).zip(query) {
                    word.copy_from_slice(&((q as f64 / length) as f32).to_le_bytes());
                }
            }
            let mut scores = diagnostic.then(|| Vec::with_capacity(self.count));
            for chunk in &self.chunks {
                if self
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline)
                {
                    return Err(invalid("probe time limit reached between completed chunks"));
                }
                let t = Instant::now();
                chunk.scan.submit()?.wait()?;
                output.timings.scan_ms += t.elapsed().as_secs_f64() * 1000.0;
                if let Some(scores) = &mut scores {
                    let bytes = self.pairs.map_read()?;
                    for pair in bytes[..chunk.rows * 8]
                        .chunks_exact(8)
                        .take((self.count - chunk.start).min(chunk.rows))
                    {
                        scores.push(f32::from_le_bytes(pair[..4].try_into().unwrap()));
                    }
                }
                let t = Instant::now();
                chunk.select.submit()?.wait()?;
                output.timings.selection_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
            let mut neighbors = Vec::new();
            if !self.chunks.is_empty() {
                let bytes = self.running[(self.chunks.len() - 1) % 2].map_read()?;
                for pair in bytes.chunks_exact(8).take(self.k) {
                    let id = u32::from_le_bytes(pair[4..].try_into().unwrap());
                    if id != u32::MAX {
                        neighbors.push(Neighbor {
                            id,
                            similarity: f32::from_le_bytes(pair[..4].try_into().unwrap()),
                        });
                    }
                }
            }
            output.neighbors.push(neighbors);
            if let Some(all) = &mut output.scores {
                all.push(scores.unwrap());
            }
        }
        output.timings.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(output)
    }
}

pub fn generated_row(id: usize, dimensions: usize) -> Vec<f32> {
    let mut x = (id as u32).wrapping_mul(747796405).wrapping_add(2891336453);
    (0..dimensions)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as f32 / 8388608.0 - 1.0
        })
        .collect()
}

pub fn validate(
    stored: &StoredRows,
    queries: &[f32],
    output: &SearchOutput,
    excluded: &[u32],
    k: usize,
) -> Result<f32> {
    let scores = output
        .scores
        .as_ref()
        .ok_or_else(|| invalid("diagnostic scores required"))?;
    let batch = queries.len() / stored.dimensions;
    if !queries.len().is_multiple_of(stored.dimensions)
        || output.neighbors.len() != batch
        || scores.len() != batch
        || scores.iter().any(|row| row.len() != stored.count)
    {
        return Err(invalid("diagnostic output shape mismatch"));
    }
    let mut error = 0.0f32;
    for ((query, scores), actual) in queries
        .chunks_exact(stored.dimensions)
        .zip(scores)
        .zip(&output.neighbors)
    {
        let reference = stored.reference(query)?;
        let mut expected = Vec::new();
        for (id, (&score, &want)) in scores.iter().zip(&reference).enumerate() {
            if excluded.contains(&(id as u32)) {
                if score != f32::NEG_INFINITY {
                    return Err(invalid("excluded row was scored"));
                }
            } else {
                if !score.is_finite() {
                    return Err(invalid("nonfinite NPU score"));
                }
                error = error.max((score - want).abs());
                expected.push(Neighbor {
                    id: id as u32,
                    similarity: score,
                });
            }
        }
        expected.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap()
                .then(a.id.cmp(&b.id))
        });
        expected.truncate(k);
        if &expected != actual {
            return Err(invalid("NPU top-k differs from CPU sort of NPU scores"));
        }
    }
    if error > 3e-6 {
        return Err(invalid(&format!("NPU score error {error} exceeds 3e-6")));
    }
    Ok(error)
}

/// Exhaust every finite half encoding and replay changed FP32 operands.
pub fn primitives() -> Result<serde_json::Value> {
    primitives_with_offset(0)
}

/// Reproduce native binding-offset admission using the same validated kernel.
pub fn primitives_with_offset(offset: usize) -> Result<serde_json::Value> {
    if !offset.is_multiple_of(64) || offset > 4096 {
        return Err(invalid(
            "probe binding offset must be a multiple of 64 in 0..=4096",
        ));
    }
    let runtime = Runtime::new()?;
    let device = runtime.npu(0)?;
    let halves: Vec<_> = (0..=u16::MAX)
        .map(f16::from_bits)
        .filter(|h| h.is_finite())
        .collect();
    let records = halves.len() / 128;
    let kernel = compile(
        &device,
        include_str!("../../kernels/npu_primitives.loom"),
        "primitives",
        &[("probe.records", records)],
        &[
            (halves.len() * 2, Access::Read),
            (halves.len() * 4, Access::Read),
            (halves.len() * 16, Access::Write),
        ],
    )?;
    let input = allocate(&runtime, &device, halves.len() * 2 + offset)?;
    let query = allocate(&runtime, &device, halves.len() * 4)?;
    let output = allocate(&runtime, &device, halves.len() * 16)?;
    for (word, half) in input.map_write()?[offset..]
        .chunks_exact_mut(2)
        .zip(&halves)
    {
        word.copy_from_slice(&half.to_le_bytes());
    }
    let mut graph = runtime.graph();
    graph.npu(
        &kernel,
        &[
            input.slice(offset..offset + halves.len() * 2)?,
            query.view(),
            output.view(),
        ],
    )?;
    let graph = graph.prepare()?;
    let start = Instant::now();
    for pass in 0..2 {
        let values = [
            0.25f32,
            -0.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            0.0,
            -0.0,
            1e-30,
            -1e-30,
        ];
        for (i, word) in query.map_write()?.chunks_exact_mut(4).enumerate() {
            word.copy_from_slice(&values[(i + pass) % values.len()].to_le_bytes());
        }
        graph.submit()?.wait()?;
        for (i, word) in output.map_read()?.chunks_exact(4).enumerate() {
            let row = i / 512 * 128 + i % 128;
            let value = halves[row].to_f32();
            let q = values[(row + pass) % values.len()];
            let expected = match i % 512 / 128 {
                0 => value,
                1 => value + q,
                2 => value * q,
                _ => {
                    if value > q {
                        value
                    } else {
                        q
                    }
                }
            };
            let actual = f32::from_le_bytes(word.try_into().unwrap());
            if actual.to_bits() != expected.to_bits() {
                return Err(invalid(&format!(
                    "primitive {} half {:04x}: {actual:?} != {expected:?}",
                    i % 512 / 128,
                    halves[row].to_bits()
                )));
            }
        }
    }
    Ok(
        serde_json::json!({"target": device.target().as_str(), "finite_half_encodings": halves.len(), "replays": 2,
        "bitwise_conversion_arithmetic_and_comparison": true, "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0,
        "statistics": runtime.statistics()}),
    )
}
