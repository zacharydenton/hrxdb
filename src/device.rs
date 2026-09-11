//! Device input/output contracts shared by search and standalone selection.
use crate::*;
use hrx::{Event, View};

/// A borrowed, contiguous row-major FP32 score matrix. Rows are independent
/// queries; columns are candidate IDs. NaN and negative infinity are excluded.
#[derive(Clone, Copy, Debug)]
pub struct ScoreBatch<'a> {
    pub(crate) values: View<'a>,
    pub(crate) batch: usize,
    pub(crate) columns: usize,
}
impl<'a> ScoreBatch<'a> {
    /// Validate shape and binding extent without reading GPU data. At most 64
    /// rows and 2^30 columns are supported, including empty matrices.
    pub fn new(values: View<'a>, batch: usize, columns: usize) -> Result<Self> {
        if batch > 64 || columns > MAX_ROWS {
            return Err(invalid("score matrix exceeds supported shape"));
        }
        Ok(Self {
            values: values.slice(0, batch * columns * 4)?,
            batch,
            columns,
        })
    }
}

/// Caller-owned fixed-stride device results, reusable across submissions.
///
/// Scores and IDs have shape `[batch, k]`. Only each row's `counts` entries are
/// valid; remaining slots are -infinity/u32::MAX after submission. `status` is
/// zero on success, one for an invalid device query. Invalid rows have count 0.
/// Creating this object clears counts/status; slots are unspecified until used.
/// All four regions share one allocation and one host readback transfer.
/// Bindings may be consumed on the submitting stream, or after waiting on the
/// returned HRX event on another stream. Bindings are read-only to consumers.
pub struct DeviceNeighbors {
    storage: Buffer,
    pub(crate) batch: usize,
    pub(crate) k: usize,
    pub(crate) device: usize,
    readback: Vec<u8>,
}
impl DeviceNeighbors {
    /// Allocate reusable output for 0..=64 queries and k=1..=1024 on this device.
    pub fn new(stream: &Stream, batch: usize, k: usize) -> Result<Self> {
        if batch > 64 || !(1..=MAX_K).contains(&k) {
            return Err(invalid("invalid device result shape"));
        }
        let out = Self {
            storage: stream.allocate(batch * (k * 8 + 8))?,
            batch,
            k,
            device: stream.device_id(),
            readback: Vec::new(),
        };
        if batch > 0 {
            stream.fill(out.storage.try_slice(0, batch * 8)?, 0)?;
        }
        Ok(out)
    }
    /// Number of query rows.
    pub fn batch_size(&self) -> usize {
        self.batch
    }
    /// Maximum results per row (the output stride).
    pub fn k(&self) -> usize {
        self.k
    }
    /// FP32 scores in row-major `[batch, k]` order.
    pub fn scores(&self) -> View<'_> {
        self.storage.slice(self.batch * 8, self.batch * self.k * 4)
    }
    /// u32 candidate IDs in row-major `[batch, k]` order.
    pub fn ids(&self) -> View<'_> {
        self.storage
            .slice(self.batch * (8 + self.k * 4), self.batch * self.k * 4)
    }
    /// u32 valid-result counts, one per query.
    pub fn counts(&self) -> View<'_> {
        self.storage.slice(self.batch * 4, self.batch * 4)
    }
    /// u32 status, one per query: 0 success, 1 invalid query.
    pub fn status(&self) -> View<'_> {
        self.storage.slice(0, self.batch * 4)
    }
    /// Owned device buffer bytes, excluding runtime allocation granularity.
    pub fn memory_usage(&self) -> usize {
        self.storage.bytes()
    }
    pub(crate) fn validate(&self, stream: &Stream, batch: usize) -> Result<()> {
        if self.device != stream.device_id() {
            return Err(invalid("output belongs to another device"));
        }
        if batch != self.batch {
            return Err(invalid("output query count does not match input"));
        }
        Ok(())
    }
    /// Wait on this stream and read results. For cross-stream consumption, first
    /// queue `stream.wait_event(&completion)`. Invalid query status returns an error.
    pub fn read(&mut self, stream: &mut Stream) -> Result<Vec<Vec<Neighbor>>> {
        let mut out = Vec::new();
        self.read_into(stream, &mut out)?;
        Ok(out)
    }
    /// Read into reusable host rows. Invalid query status leaves output unchanged.
    /// Readback staging and host row capacities are retained for reuse; device-only
    /// users allocate no host staging until their first read.
    pub fn read_into(&mut self, stream: &mut Stream, out: &mut Vec<Vec<Neighbor>>) -> Result<()> {
        self.validate(stream, self.batch)?;
        self.readback.resize(self.batch * (self.k * 8 + 8), 0);
        if self.batch > 0 {
            stream.read_blocking(self.storage.binding(), &mut self.readback)?;
        }
        let (status, rest) = self.readback.split_at_mut(self.batch * 4);
        let (counts, rest) = rest.split_at_mut(self.batch * 4);
        let (scores, ids) = rest.split_at_mut(self.batch * self.k * 4);
        if let Some(q) = status
            .as_chunks::<4>()
            .0
            .iter()
            .position(|s| u32::from_le_bytes(*s) != 0)
        {
            return Err(invalid(&format!(
                "device query {q} is nonfinite or has zero norm"
            )));
        }
        out.resize_with(self.batch, Vec::new);
        for (q, row) in out.iter_mut().enumerate() {
            let n = u32::from_le_bytes(counts[q * 4..q * 4 + 4].try_into().unwrap()) as usize;
            if n > self.k {
                return Err(invalid(&format!(
                    "invalid device result count {n} for row {q}, k={} (check stream ordering)",
                    self.k
                )));
            }
            row.clear();
            row.reserve(n);
            for r in 0..n {
                let at = (q * self.k + r) * 4;
                row.push(Neighbor {
                    id: u32::from_le_bytes(ids[at..at + 4].try_into().unwrap()),
                    similarity: f32::from_le_bytes(scores[at..at + 4].try_into().unwrap()),
                });
            }
        }
        Ok(())
    }
}
pub(crate) fn finish(
    stream: &Stream,
    kernel: &Kernel,
    out: &DeviceNeighbors,
    selected: Option<(View<'_>, View<'_>, usize)>,
) -> Result<()> {
    if out.batch == 0 {
        return Ok(());
    }
    let (scores, ids, m) = selected.unwrap_or((out.scores(), out.ids(), 0));
    let mut c = Constants::new();
    c.push(out.batch as u32)?;
    c.push(m as u32)?;
    c.push(out.k as u32)?;
    // SAFETY: selected lists have batch*m entries; outputs have batch*k and
    // per-query counts/status. For m=0 the fallback reads stay inside output
    // allocations and their values are masked; no logical candidate is read.
    unsafe {
        stream.dispatch(
            kernel,
            [out.batch as u32, 1, 1],
            [256, 1, 1],
            &c,
            &[
                scores,
                ids,
                out.status(),
                out.scores(),
                out.ids(),
                out.counts(),
            ],
        )
    }
}

/// Reusable GPU top-k over application-defined score matrices, independent of a
/// corpus. Queues work on caller-owned streams of one device without host readback.
/// Ties prefer lower column IDs. NaN and -infinity are absent; +infinity is valid.
/// Large matrices are selected in bounded 262,144-column tiles with running GPU
/// top-k. Scores are read in place, including when a matrix exceeds 4 GiB.
pub struct TopK {
    device: usize,
    compiler: hrx::loom::Compiler,
    plans: std::collections::HashMap<usize, selection::SelectionPlan>,
    candidates: Option<[(Buffer, Buffer); 2]>,
    capacity: usize,
    finish: Kernel,
    merge: Kernel,
    lists: Option<[(Buffer, Buffer); 2]>,
    list_capacity: usize,
}
impl TopK {
    /// Prepare selection for this device. No corpus is allocated or required.
    pub fn new(stream: &Stream) -> Result<Self> {
        let compiler = hrx::loom::Compiler::with_options(
            None,
            hrx::loom::CompilerOptions {
                target: stream.target().clone(),
                ..Default::default()
            },
        )?;
        let (finish, _) = compile(
            &compiler,
            stream,
            include_str!("../kernels/finish.loom"),
            kernels::named_spec("finish"),
        )?;
        let mut spec = kernels::named_spec("sorted_merge");
        spec.config.insert("db.merge.planar".into(), "1".into());
        let (merge, _) = compile(&compiler, stream, kernels::SORT, spec)?;
        Ok(Self {
            device: stream.device_id(),
            compiler,
            plans: Default::default(),
            candidates: None,
            capacity: 0,
            finish,
            merge,
            lists: None,
            list_capacity: 0,
        })
    }
    /// Prepare scratch and kernels before serving a shape. Buffers grow and are
    /// reused; allocation replacement is safe for already queued HRX device work.
    pub fn reserve(
        &mut self,
        stream: &Stream,
        batch: usize,
        columns: usize,
        k: usize,
    ) -> Result<()> {
        if stream.device_id() != self.device {
            return Err(invalid("top-k belongs to another device"));
        }
        if batch > 64 || columns > MAX_ROWS || !(1..=MAX_K).contains(&k) {
            return Err(invalid("invalid top-k shape"));
        }
        if batch == 0 || columns == 0 {
            return Ok(());
        }
        const TILE: usize = 262_144;
        let width = if columns > TILE { 1 } else { batch };
        if !self.plans.contains_key(&width) {
            let (plan, _) = selection::SelectionPlan::new(&self.compiler, stream, width, TILE)?;
            self.plans.insert(width, plan);
        }
        let bytes = width * columns.min(TILE).div_ceil(1024) * k.min(columns) * 4;
        if bytes > self.capacity {
            let pair = || Ok::<_, Error>((stream.allocate(bytes)?, stream.allocate(bytes)?));
            self.candidates = Some([pair()?, pair()?]);
            self.capacity = bytes;
        }
        if columns > TILE && batch * k * 4 > self.list_capacity {
            let bytes = batch * k * 4;
            self.lists = Some([
                (stream.allocate(bytes)?, stream.allocate(bytes)?),
                (stream.allocate(bytes * 2)?, stream.allocate(bytes * 2)?),
            ]);
            self.list_capacity = bytes;
        }
        Ok(())
    }
    /// Submit selection into caller-owned output and return a completion event.
    /// Scores must be ready on this stream, or ordered with an HRX event first.
    /// Reuse one TopK on one ordered stream at a time; synchronize before using
    /// its workspace on another stream of the same device.
    pub fn select(
        &mut self,
        stream: &mut Stream,
        scores: ScoreBatch<'_>,
        out: &mut DeviceNeighbors,
    ) -> Result<Event> {
        out.validate(stream, scores.batch)?;
        self.reserve(stream, scores.batch, scores.columns, out.k)?;
        if out.batch > 0 {
            stream.fill(out.status(), 0)?;
        }
        if scores.batch > 0 && scores.columns > 0 {
            let pairs = self.candidates.as_ref().unwrap();
            let k = out.k.min(scores.columns);
            const TILE: usize = 262_144;
            if scores.columns <= TILE {
                let selected = selection::select_scores(
                    stream,
                    pairs,
                    &self.plans[&scores.batch],
                    selection::SelectionInput {
                        scores: scores.values,
                        rows: scores.columns,
                        start: 0,
                        k,
                        batch: scores.batch,
                    },
                )?;
                finish(
                    stream,
                    &self.finish,
                    out,
                    Some((pairs[selected].0.binding(), pairs[selected].1.binding(), k)),
                )?;
            } else {
                let lists = self.lists.as_ref().unwrap();
                let running = &lists[0];
                let joined = &lists[1];
                let bytes = scores.batch * k * 4;
                for start in (0..scores.columns).step_by(TILE) {
                    let rows = (scores.columns - start).min(TILE);
                    for q in 0..scores.batch {
                        // Slice each row directly: no full-score packing/copy and
                        // no compiler address expression spanning multiple GB.
                        let values = scores
                            .values
                            .slice((q * scores.columns + start) * 4, rows * 4)?;
                        let selected = selection::select_scores(
                            stream,
                            pairs,
                            &self.plans[&1],
                            selection::SelectionInput {
                                scores: values,
                                rows,
                                start,
                                k,
                                batch: 1,
                            },
                        )?;
                        stream.copy(
                            joined.0.try_slice(bytes + q * k * 4, k * 4)?,
                            pairs[selected].0.try_slice(0, k * 4)?,
                        )?;
                        stream.copy(
                            joined.1.try_slice(bytes + q * k * 4, k * 4)?,
                            pairs[selected].1.try_slice(0, k * 4)?,
                        )?;
                    }
                    if start == 0 {
                        stream.copy(
                            running.0.try_slice(0, bytes)?,
                            joined.0.try_slice(bytes, bytes)?,
                        )?;
                        stream.copy(
                            running.1.try_slice(0, bytes)?,
                            joined.1.try_slice(bytes, bytes)?,
                        )?;
                    } else {
                        stream.copy(
                            joined.0.try_slice(0, bytes)?,
                            running.0.try_slice(0, bytes)?,
                        )?;
                        stream.copy(
                            joined.1.try_slice(0, bytes)?,
                            running.1.try_slice(0, bytes)?,
                        )?;
                        let mut c = Constants::new();
                        c.push((scores.batch * 2) as u32)?;
                        c.push(k as u32)?;
                        // SAFETY: planar pairs each query's sorted running and
                        // tile lists. Outputs are disjoint, batch*k allocations.
                        unsafe {
                            stream.dispatch(
                                &self.merge,
                                [scores.batch as u32, 1, 1],
                                [256, 1, 1],
                                &c,
                                &[
                                    joined.0.binding(),
                                    joined.1.binding(),
                                    running.0.binding(),
                                    running.1.binding(),
                                ],
                            )?;
                        }
                    }
                }
                finish(
                    stream,
                    &self.finish,
                    out,
                    Some((running.0.binding(), running.1.binding(), k)),
                )?;
            }
        } else {
            finish(stream, &self.finish, out, None)?;
        }
        stream.record_event()
    }
    /// Private selection buffer bytes; caller-owned matrices/results are excluded.
    pub fn memory_usage(&self) -> usize {
        self.capacity * 4 + self.list_capacity * 6
    }
}
