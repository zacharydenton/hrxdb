//! Asynchronous, validated device-input search.
use crate::*;
use hrx::{Event, View};

/// Borrowed FP32 queries in row-major device storage. Values are normalized on
/// the GPU; nonfinite or zero-norm rows produce status=1 and zero matches.
/// Stride is in FP32 elements. All accessed rows must be ready on the searcher's
/// stream (use an event for a producer on another stream).
#[derive(Clone, Copy, Debug)]
pub struct DeviceQueries<'a> {
    values: View<'a>,
    batch: usize,
    dimensions: usize,
    stride: usize,
}
impl<'a> DeviceQueries<'a> {
    /// Validate a 0..=64 query batch, dimensions 1..=16384, and row stride between
    /// dimensions and 16384. The binding must cover batch*stride FP32 elements.
    pub fn new(values: View<'a>, batch: usize, dimensions: usize, stride: usize) -> Result<Self> {
        if batch > MAX_BATCH
            || !(1..=MAX_DIMENSIONS).contains(&dimensions)
            || stride < dimensions
            || stride > MAX_DIMENSIONS
        {
            return Err(invalid("invalid device query shape or stride"));
        }
        Ok(Self {
            values: values.slice(0, batch * stride * 4)?,
            batch,
            dimensions,
            stride,
        })
    }
}
pub(crate) struct QueryWorkspace {
    query: Buffer,
    width: usize,
    plans: std::collections::HashMap<usize, Kernel>,
    finish: Kernel,
}
impl Searcher {
    /// Prepare device normalization, scan, and selection for repeated submissions.
    /// First use can allocate/compile and wait while replacing host-imported
    /// batch workspace. Fully reserved device submissions perform no host wait.
    pub fn reserve_device(&mut self, batch: usize, k: usize) -> Result<()> {
        if batch > MAX_BATCH || !(1..=MAX_K).contains(&k) {
            return Err(invalid("invalid device search shape"));
        }
        if batch == 0 {
            return Ok(());
        }
        if batch == 1 {
            self.reserve_search(k)?;
        } else {
            self.reserve_batch(batch, k)?;
        }
        let width = if batch == 1 {
            1
        } else {
            batch.next_power_of_two().max(8)
        };
        if self.device_queries.is_none() {
            let (finish, _) = compile(
                &self.compiler,
                &self.stream,
                include_str!("../kernels/finish.loom"),
                kernels::named_spec("finish"),
            )?;
            self.device_queries = Some(QueryWorkspace {
                query: self.stream.allocate(width * self.padded * 4)?,
                width,
                plans: Default::default(),
                finish,
            });
        }
        let workspace = self.device_queries.as_mut().unwrap();
        if workspace.width < width {
            workspace.query = self.stream.allocate(width * self.padded * 4)?;
            workspace.width = width;
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = workspace.plans.entry(width) {
            let mut spec = kernels::named_spec("prepare_queries");
            for (key, value) in [
                ("query.dimensions", self.dimensions),
                ("query.padded", self.padded),
                ("query.width", width),
            ] {
                spec.set_config(key, value.to_string());
            }
            let (plan, report) = compile(
                &self.compiler,
                &self.stream,
                include_str!("../kernels/prepare_queries.loom"),
                spec,
            )?;
            entry.insert(plan);
            self.reports.push(report);
        }
        Ok(())
    }

    /// Queue GPU normalization, exhaustive cosine search, and top-k into owned
    /// device outputs. Return a completion event without reading results back.
    ///
    /// `excluded` is an optional shared bitmap (u32 words, least-significant bit
    /// first) over insertion IDs, at least ceil(corpus.len()/32)*4 bytes. It is
    /// read in place, allowing device producers to update it incrementally.
    /// Output batch count must match input. `output.k()` selects k. Invalid
    /// queries set output status=1 and count=0; `DeviceNeighbors::read` reports
    /// these as errors. Inputs/bitmap must be ready on this searcher's stream.
    /// Keep buffers read-only until completion. Slots beyond counts are invalid.
    ///
    /// Scaled FP32 norm accumulation handles extreme finite magnitudes without
    /// overflow. Its rounding can differ from host FP64 normalization; very tiny
    /// relative components may underflow. Scores use FP32 accumulation.
    pub fn search_device(
        &mut self,
        queries: DeviceQueries<'_>,
        excluded: Option<View<'_>>,
        output: &mut DeviceNeighbors,
    ) -> Result<Event> {
        if queries.dimensions != self.dimensions {
            return Err(invalid("device query dimension mismatch"));
        }
        output.validate(&self.stream, queries.batch)?;
        let excluded = excluded
            .map(|v| v.slice(0, self.count.div_ceil(32) * 4))
            .transpose()?;
        self.reserve_device(queries.batch, output.k)?;
        if queries.batch == 0 {
            return self.stream.record_event();
        }
        let width = if queries.batch == 1 {
            1
        } else {
            queries.batch.next_power_of_two().max(8)
        };
        let workspace = self.device_queries.as_ref().unwrap();
        self.stream.fill(workspace.query.binding(), 0)?;
        let mut c = Constants::new();
        c.push(queries.batch as u32)?;
        c.push(queries.stride as u32)?;
        // SAFETY: query shape/stride and output extents are validated. GPU
        // normalization checks every logical row before producing packed values.
        unsafe {
            self.stream.dispatch(
                &workspace.plans[&width],
                [queries.batch as u32, 1, 1],
                [128, 1, 1],
                &c,
                &[queries.values, workspace.query.binding(), output.status()],
            )?;
        }
        let k = output.k.min(self.count);
        if k == 0 {
            device::finish(&self.stream, &workspace.finish, output, None)?;
        } else if queries.batch == 1 {
            self.dispatch_scan_from(false, workspace.query.binding())?;
            if let Some(bitmap) = excluded {
                self.apply_exclusions_from(bitmap)?;
            }
            let selected = self.select(k)?;
            device::finish(
                &self.stream,
                &workspace.finish,
                output,
                Some((
                    self.candidates[selected].0.binding(),
                    self.candidates[selected].1.binding(),
                    k,
                )),
            )?;
        } else {
            let scratch = self.batch.as_ref().unwrap();
            scratch.scan(
                &self.stream,
                &self.corpus,
                workspace.query.binding(),
                excluded,
                queries.batch,
                k,
            )?;
            device::finish(
                &self.stream,
                &workspace.finish,
                output,
                Some((scratch.running.0.binding(), scratch.running.1.binding(), k)),
            )?;
        }
        self.stream.record_event()
    }
    pub(crate) fn device_query_bytes(&self) -> usize {
        self.device_queries.as_ref().map_or(0, |w| w.query.bytes())
    }
}
