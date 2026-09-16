//! Gather named corpus rows into dense, normalized FP32 device storage.
use crate::*;
use hrx::{Event, View};

impl Corpus {
    /// Gather insertion IDs into contiguous row-major FP32 device storage.
    ///
    /// Output row `i` corresponds to `rows[i]`; arbitrary order and duplicates
    /// are preserved. Each component is the stored FP16 value expanded to FP32
    /// and multiplied by its stored inverse norm. Rows have exactly
    /// [`Self::dimensions`] components, with no dimension or row padding.
    /// This normalization permits cosine comparisons by dot product, subject
    /// to FP32 rounding; pairwise dots need not be bitwise equal to search.
    ///
    /// `output` must be four-byte aligned and cover `rows.len()*dimensions*4`
    /// bytes. A larger binding's suffix is untouched. All IDs and extents are
    /// checked before submission; invalid arguments leave output unchanged.
    /// An empty request records an event without writing output. At most 2^30
    /// gathered rows and 2^32 output components are supported; [`MAX_BATCH`] and
    /// [`MAX_K`] do not restrict this operation.
    ///
    /// The stream and output must belong to this corpus's device. Output may
    /// not alias corpus storage. Callers must order prior output uses on this
    /// stream, or wait on their events first. Returned completion can be waited
    /// on by another stream; same-stream consumers may be queued immediately.
    /// No vector readback or host completion wait occurs. First use compiles a
    /// kernel cached across corpus clones; ID routing uses stream-owned scratch
    /// that can be reused on subsequent calls. Runtime failures may leave partial
    /// output and do not cancel commands already submitted.
    ///
    /// ```no_run
    /// # fn example(corpus: &hrxdb::Corpus) -> hrxdb::Result<()> {
    /// let mut stream = corpus.stream()?;
    /// let ids = [7, 2, 7];
    /// let block = stream.allocate(ids.len() * corpus.dimensions() * 4)?;
    /// let done = corpus.gather_into(&mut stream, &ids, block.binding())?;
    /// // Bind block to a custom kernel here, or use it as DeviceQueries.
    /// let mut consumer = corpus.stream()?;
    /// consumer.wait_event(&done)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn gather_into(
        &self,
        stream: &mut Stream,
        rows: &[u32],
        output: View<'_>,
    ) -> Result<Event> {
        if stream.device_id() != self.inner.device_id {
            return Err(invalid("gather stream belongs to another device"));
        }
        let bytes = gather_bytes(rows.len(), self.dimensions())?;
        if !output.offset().is_multiple_of(4) {
            return Err(invalid("gather output must be four-byte aligned"));
        }
        let output = output.slice(0, bytes)?;
        if rows.iter().any(|&id| id as usize >= self.len()) {
            return Err(invalid("gather ID is outside the corpus"));
        }
        if rows.is_empty() {
            return stream.record_event();
        }
        let shards = &self.inner.shards;
        if shards.iter().any(|s| {
            std::ptr::eq(output.owner(), &s.data) || std::ptr::eq(output.owner(), &s.norms)
        }) {
            return Err(invalid("gather output must not alias corpus storage"));
        }

        // Route only the requested IDs. Shader work is O(rows*dimensions), not
        // O(corpus size) or O(rows*shards). Destinations restore requested order.
        let mut routes = vec![Vec::<[u32; 2]>::new(); shards.len()];
        for (destination, &id) in rows.iter().enumerate() {
            let shard = shards.partition_point(|s| s.start <= id as usize) - 1;
            routes[shard].push([
                (id as usize - shards[shard].start) as u32,
                destination as u32,
            ]);
        }
        let mut payload = Vec::with_capacity(rows.len() * 8);
        for pair in routes.iter().flatten() {
            payload.extend_from_slice(&pair[0].to_le_bytes());
            payload.extend_from_slice(&pair[1].to_le_bytes());
        }
        let kernel = self.gather_kernel(stream)?;
        let routing = stream.scratch(payload.len())?;
        stream.upload(routing.try_slice(0, payload.len())?, &payload)?;
        let mut offset = 0;
        for (shard, requests) in shards.iter().zip(&routes) {
            if requests.is_empty() {
                continue;
            }
            let mut constants = Constants::new();
            constants.push(requests.len() as u32)?;
            constants.push(shard.count as u32)?;
            constants.push(rows.len() as u32)?;
            // SAFETY: CPU routing validates source IDs and assigns unique,
            // in-bounds destinations. Specialized strides/limits match both
            // buffers. Every requested component is written exactly once.
            unsafe {
                stream.dispatch(
                    &kernel,
                    [requests.len() as u32, 1, 1],
                    [128, 1, 1],
                    &constants,
                    &[
                        shard.data.binding(),
                        shard.norms.binding(),
                        routing.try_slice(offset, requests.len() * 8)?,
                        output,
                    ],
                )?;
            }
            offset += requests.len() * 8;
        }
        stream.recycle(routing)?;
        stream.record_event()
    }

    fn gather_kernel(&self, stream: &Stream) -> Result<Kernel> {
        let mut cached = self
            .inner
            .gather
            .lock()
            .map_err(|_| invalid("gather cache poisoned"))?;
        if let Some(kernel) = &*cached {
            return Ok(kernel.clone());
        }
        let compiler = hrx::loom::Compiler::for_stream(None, stream)?;
        let mut spec = kernels::named_spec("gather");
        for (key, value) in [
            ("gather.dimensions", self.dimensions()),
            ("gather.padded", self.padded_dimensions()),
            (
                "gather.input_limit",
                (MAX_ELEMENTS / self.padded_dimensions()).min(MAX_ROWS),
            ),
            (
                "gather.output_limit",
                (MAX_ELEMENTS / self.dimensions()).min(MAX_ROWS),
            ),
        ] {
            spec.set_config(key, value.to_string());
        }
        let (kernel, _) = compile(
            &compiler,
            stream,
            include_str!("../kernels/gather.loom"),
            spec,
        )?;
        *cached = Some(kernel.clone());
        Ok(kernel)
    }
}

fn gather_bytes(rows: usize, dimensions: usize) -> Result<usize> {
    let elements = rows.checked_mul(dimensions).filter(|&n| n <= MAX_ELEMENTS);
    if rows > MAX_ROWS || elements.is_none() {
        return Err(invalid("gather exceeds supported output extent"));
    }
    Ok(elements.unwrap() * 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_extents_respect_element_address_limits() {
        assert_eq!(gather_bytes(0, 3).unwrap(), 0);
        for d in [1, 3, 384, 16_384] {
            let rows = (MAX_ELEMENTS / d).min(MAX_ROWS);
            assert_eq!(gather_bytes(rows, d).unwrap(), rows * d * 4);
            assert!(gather_bytes(rows + 1, d).is_err());
        }
        assert!(gather_bytes(usize::MAX, 16_384).is_err());
    }
}
