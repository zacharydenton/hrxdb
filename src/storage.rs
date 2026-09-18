//! Immutable shared vector storage, independent of search execution.
use crate::*;
use hrx::inference::ModelContext;
impl Corpus {
    /// Build FP32 rows using the caller's GPU, allocation ceiling and context.
    /// Normalization and input requirements match [`Self::build`]. The context
    /// is retained by corpus clones and workers. Native storage, upload staging
    /// and subsequent workspace allocations use its memory budget, if present.
    pub fn build_in<I, R>(context: &ModelContext, dimensions: usize, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        Self::build_encoded_in(context, dimensions, rows, |r, d, p, b| {
            encode(r.as_ref(), d, p, b)
        })
    }

    /// Build little-endian FP16 rows in the caller's retained model context.
    /// Values and input requirements match [`Self::build_fp16`]. Native corpus
    /// buffers and upload staging share the context's GPU and allocation
    /// ceiling; they are not exported as coordinated `BufferView`s. Searchers
    /// inherit the ceiling, and prepared searches require this same runtime.
    ///
    /// ```no_run
    /// # fn example(context: &hrx::inference::ModelContext) -> hrxdb::Result<()> {
    /// let corpus = hrxdb::Corpus::build_fp16_in(context, 3, [[0, 0x3c, 0, 0, 0, 0]])?;
    /// let search = corpus.prepare_search(context, 1, 1, 1)?;
    /// // Submit an encoder's DeviceTensor directly to search.submit(&tensor).
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_fp16_in<I, R>(context: &ModelContext, dimensions: usize, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        Self::build_encoded_in(context, dimensions, rows, |r, d, p, b| {
            encode_fp16(r.as_ref(), d, p, b)
        })
    }

    fn build_encoded_in<I, R>(
        context: &ModelContext,
        dimensions: usize,
        rows: I,
        encode_row: impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
    {
        let device = Device::open(context.runtime().gpu()?.index())?;
        Self::build_encoded_with_context(
            &device,
            Some(context),
            dimensions,
            rows,
            MAX_ELEMENTS,
            encode_row,
        )
    }

    /// Build shared, immutable storage from FP32 rows, normalizing before FP16 conversion.
    /// Dimensions must be 1..=16384; rows must be finite and nonzero.
    pub fn build<I, R>(device: &Device, dimensions: usize, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        Self::build_encoded(device, dimensions, rows, MAX_ELEMENTS, |r, d, p, b| {
            encode(r.as_ref(), d, p, b)
        })
    }
    /// Build shared storage from little-endian FP16 rows, preserving their values.
    /// Logical rows must be finite and nonzero. Dimension padding and inverse norms are added.
    pub fn build_fp16<I, R>(device: &Device, dimensions: usize, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        Self::build_encoded(device, dimensions, rows, MAX_ELEMENTS, |r, d, p, b| {
            encode_fp16(r.as_ref(), d, p, b)
        })
    }
    pub(crate) fn build_encoded<I, R>(
        device: &Device,
        dimensions: usize,
        rows: I,
        element_limit: usize,
        encode_row: impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
    {
        Self::build_encoded_with_context(device, None, dimensions, rows, element_limit, encode_row)
    }

    fn build_encoded_with_context<I, R>(
        device: &Device,
        context: Option<&ModelContext>,
        dimensions: usize,
        rows: I,
        element_limit: usize,
        mut encode_row: impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
    {
        if device.target().as_str() != "gfx1151" {
            return Err(invalid("hrxdb currently targets gfx1151"));
        }
        let mut rows = rows.into_iter();
        let count = rows.len();
        let (padded, _) = layout(dimensions, count)?;
        let mut stream = device.stream()?;
        let budget = context.and_then(|context| context.runtime().memory_budget());
        if let Some(budget) = budget {
            stream = stream.with_memory_budget(budget.clone());
        }
        let chunk_rows = (CHUNK_BYTES / (padded * 2)).max(1).min(count);
        let _conversion = budget
            .map(|budget| budget.reserve(chunk_rows * (padded * 2 + 4)))
            .transpose()?;
        let mut chunk = Vec::with_capacity(chunk_rows * padded * 2);
        let mut norm_chunk = Vec::with_capacity(chunk_rows * 4);
        let mut shards = Vec::new();
        let per_shard = shard_rows(padded, element_limit);
        for start in (0..count).step_by(per_shard) {
            let shard_count = (count - start).min(per_shard);
            let capacity = shard_capacity(shard_count);
            let data = stream.allocate(capacity * padded * 2)?;
            let norms = stream.allocate(capacity * 4)?;
            if capacity > shard_count {
                // Initialize only the slack; these are not indexed zero rows.
                // Contents are deliberately not part of the public contract.
                // The ingestion synchronization below also completes the fills.
                stream.fill(
                    data.try_slice(
                        shard_count * padded * 2,
                        (capacity - shard_count) * padded * 2,
                    )?,
                    0,
                )?;
                stream.fill(
                    norms.try_slice(shard_count * 4, (capacity - shard_count) * 4)?,
                    0,
                )?;
            }
            for local in (0..shard_count).step_by(chunk_rows) {
                chunk.clear();
                norm_chunk.clear();
                for _ in local..(local + chunk_rows).min(shard_count) {
                    let row = rows
                        .next()
                        .ok_or_else(|| invalid("row iterator returned fewer rows than declared"))?;
                    let inverse = encode_row(&row, dimensions, padded, &mut chunk)?;
                    norm_chunk.extend_from_slice(&inverse.to_le_bytes());
                }
                stream.upload(data.try_slice(local * padded * 2, chunk.len())?, &chunk)?;
                stream.upload(norms.try_slice(local * 4, norm_chunk.len())?, &norm_chunk)?;
                // Bound HRX's owned staging as well as the conversion buffers.
                stream.synchronize()?;
            }
            shards.push(Shard {
                data,
                norms,
                start,
                count: shard_count,
                capacity,
            });
        }
        if rows.next().is_some() {
            return Err(invalid("row iterator returned more rows than declared"));
        }
        Ok(Self {
            workspace_budget: budget.cloned(),
            inner: std::sync::Arc::new(CorpusStorage {
                context: context.map(crate::corpus::CorpusContext::shared),
                device: device.clone(),
                device_id: stream.device_id(),
                shards,
                dimensions,
                padded,
                count,
                gather: Default::default(),
            }),
        })
    }
}

/// One owned resident FP16 allocation to adopt without copying its vectors.
/// Rows use the corpus's padded stride; capacity must be a multiple of 256.
/// Logical rows must be finite, nonzero, and have zero dimension padding.
pub struct ResidentShard {
    /// Owned GPU allocation, including readable row slack.
    pub vectors: Buffer,
    /// Logical row count (slack is excluded).
    pub rows: usize,
}
impl Corpus {
    /// Adopt FP16 allocations on the producer's ordered stream, validating on
    /// the GPU and computing inverse norms with FP32 accumulation.
    ///
    /// FP32 norm reduction can round differently from host ingestion.
    /// No vector data is transferred to the host or copied to another allocation.
    /// The only readback is a four-byte validation result. This call completes
    /// producer work and validation before publishing the immutable corpus.
    /// Wait on a producer event first if its writes used another stream.
    ///
    /// Buffers use a row stride of `ceil(dimensions/128)*128` FP16 elements.
    /// Capacity must cover the logical rows, be divisible by 256, and fit the
    /// 2^32-element per-shard limit. Shard order defines insertion IDs. Provided
    /// allocations are consumed even on validation failure; no unchecked path
    /// accepts inconsistent norms or invalid rows.
    pub fn from_device(
        device: &Device,
        stream: &mut Stream,
        dimensions: usize,
        resident: Vec<ResidentShard>,
    ) -> Result<Self> {
        let (padded, _) = layout(dimensions, 0)?;
        if device.target().as_str() != "gfx1151" {
            return Err(invalid("hrxdb currently targets gfx1151"));
        }
        let check = device.stream()?;
        if check.device_id() != stream.device_id() {
            return Err(invalid("producer stream belongs to another device"));
        }
        let mut count = 0usize;
        for shard in &resident {
            let row_bytes = padded * 2;
            let capacity = shard.vectors.bytes() / row_bytes;
            if shard.rows == 0
                || shard.rows > capacity
                || !shard.vectors.bytes().is_multiple_of(row_bytes)
                || !capacity.is_multiple_of(SHARD_ROW_ALIGNMENT)
                || capacity > MAX_ELEMENTS / padded
            {
                return Err(invalid(
                    "resident shard has invalid row count, stride, capacity, or element extent",
                ));
            }
            count = count
                .checked_add(shard.rows)
                .filter(|n| *n <= MAX_ROWS)
                .ok_or_else(|| invalid("resident corpus exceeds 2^30 rows"))?;
        }
        let compiler = hrx::loom::Compiler::for_stream(None, stream)?;
        let status = stream.allocate_zeroed(4)?;
        let mut shards = Vec::with_capacity(resident.len());
        let mut start = 0;
        for shard in resident {
            let capacity = shard.vectors.bytes() / (padded * 2);
            let norms = stream.allocate_zeroed(capacity * 4)?;
            let mut spec = kernels::named_spec("import_corpus");
            for (key, value) in [
                ("import.rows", shard.rows),
                ("import.dimensions", dimensions),
                ("import.padded", padded),
            ] {
                spec.set_config(key, value.to_string());
            }
            let (kernel, _) = compile(
                &compiler,
                stream,
                include_str!("../kernels/import_corpus.loom"),
                spec,
            )?;
            // SAFETY: validated extents cover every logical padded row. The
            // kernel only reads vectors and writes owned norms/status; invalid
            // values are counted on device, never used for addressing.
            unsafe {
                stream.dispatch(
                    &kernel,
                    [shard.rows as u32, 1, 1],
                    [128, 1, 1],
                    &Constants::new(),
                    &[shard.vectors.binding(), norms.binding(), status.binding()],
                )?;
            }
            shards.push(Shard {
                data: shard.vectors,
                norms,
                start,
                count: shard.rows,
                capacity,
            });
            start += shard.rows;
        }
        let mut errors = [0; 4];
        stream.read_blocking(status.binding(), &mut errors)?;
        if u32::from_le_bytes(errors) != 0 {
            return Err(invalid(
                "resident corpus contains nonfinite, zero-norm, or nonzero-padding rows",
            ));
        }
        Ok(Self {
            workspace_budget: None,
            inner: std::sync::Arc::new(CorpusStorage {
                context: None,
                device: device.clone(),
                device_id: stream.device_id(),
                shards,
                dimensions,
                padded,
                count,
                gather: Default::default(),
            }),
        })
    }
}
