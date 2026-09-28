//! Tiled batch queries with bounded device score storage and running top-k.
use super::*;

const TILE_ROWS: usize = 262_144;

fn selection_width(queries: usize, k: usize) -> usize {
    if k > 32 {
        queries
    } else {
        queries.next_power_of_two().max(8)
    }
}

pub(crate) struct BatchScratch {
    width: usize,
    k: usize,
    tile_rows: usize,
    query: HostBuffer,
    readback: HostBuffer,
    scores: Buffer,
    candidates: [(Buffer, Buffer); 2],
    pub(crate) running: (Buffer, Buffer),
    merge_scratch: (Buffer, Buffer),
    plans: Vec<BatchPlan>,
    // Only the most recent exact query count, effective k, and mask mode.
    search_graph: Option<(usize, usize, bool, hrx::GraphExec)>,
    #[cfg(test)]
    immediate: bool,
    #[cfg(test)]
    staged_merge: Option<bench::StagedMerge>,
}

struct BatchPlan {
    width: usize,
    scan: Kernel,
    selections: std::collections::HashMap<usize, crate::selection::SelectionPlan>,
    running_merge: Kernel,
}

impl BatchScratch {
    fn new(
        stream: &Stream,
        width: usize,
        k: usize,
        tile_rows: usize,
        padded: usize,
    ) -> Result<Self> {
        let pair = |bytes| -> Result<_> { Ok((stream.allocate(bytes)?, stream.allocate(bytes)?)) };
        let scratch = width * tile_rows.div_ceil(1024) * k * 4;
        Ok(Self {
            width,
            k,
            tile_rows,
            query: HostBuffer::new(stream, width * padded * 4)?,
            readback: HostBuffer::new(stream, width * k * 8)?,
            scores: stream.allocate(width * tile_rows * 4)?,
            candidates: [pair(scratch)?, pair(scratch)?],
            running: pair(width * k * 4)?,
            merge_scratch: pair(width * k * 4)?,
            plans: Vec::new(),
            search_graph: None,
            #[cfg(test)]
            immediate: false,
            #[cfg(test)]
            staged_merge: None,
        })
    }

    /// Resize tile storage for a corpus whose capacity range ends at `rows`,
    /// keeping compiled plans. Scratch of any tile size serves any corpus, so
    /// failure leaves it valid.
    pub(crate) fn retile(&mut self, stream: &Stream, rows: usize, padded: usize) -> Result<()> {
        // set_corpus also calls this when capacity is unchanged. Recorded
        // bindings and tile counts still belong to the previous snapshot.
        self.search_graph = None;
        let tile_rows = rows.min(TILE_ROWS);
        if tile_rows == self.tile_rows {
            return Ok(());
        }
        let mut fresh = Self::new(stream, self.width, self.k, tile_rows, padded)?;
        fresh.plans = std::mem::take(&mut self.plans);
        *self = fresh;
        Ok(())
    }

    pub(crate) fn abandon(&mut self) {
        self.query.abandon();
        self.readback.abandon();
    }
}

fn batch_size(dimensions: usize, values: usize, k: usize) -> Result<usize> {
    if !(1..=MAX_K).contains(&k) {
        return Err(invalid("k must be in 1..=1024"));
    }
    if !values.is_multiple_of(dimensions) {
        return Err(invalid(
            "batch queries must contain complete dimension-sized rows",
        ));
    }
    let count = values / dimensions;
    if count > MAX_BATCH {
        return Err(invalid(
            "a query batch supports at most 64 rows; split larger batches",
        ));
    }
    Ok(count)
}

impl Searcher {
    /// Bytes reserved in batch query, score, selection, and readback buffers.
    ///
    /// This is additional to the corpus and single-query workspace. Returns
    /// zero before batch workspace is reserved; excludes compiler/runtime
    /// bookkeeping and returned host neighbor vectors.
    pub fn batch_workspace_bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |s| {
            s.query.buffer().binding().len()
                + s.readback.buffer().binding().len()
                + s.scores.binding().len()
                + [
                    &s.candidates[0],
                    &s.candidates[1],
                    &s.running,
                    &s.merge_scratch,
                ]
                .iter()
                .map(|(a, b)| a.binding().len() + b.binding().len())
                .sum::<usize>()
        })
    }

    /// Reserve and compile workspace for a batch of `query_count` queries.
    ///
    /// `search_batch` calls this automatically. Call it during setup to exclude
    /// allocation and compilation from the first batch. Scan widths are rounded
    /// to 8, 16, 32, or 64; large-k selection omits padded query rows.
    /// Both are cached and workspace is retained.
    /// Score storage covers at most 262,144 corpus rows, independent of index size.
    /// At width 64 this uses about 66 MiB for k=5 or 322 MiB for k=1,024, plus
    /// query storage (96 KiB at 384 dimensions). A one-query batch uses `search`.
    ///
    /// # Errors
    /// `query_count` must be 1–64 and `k` must be 1–1,024. Allocation,
    /// synchronization, and compiler failures propagate as errors.
    pub fn reserve_batch(&mut self, query_count: usize, k: usize) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&query_count) {
            return Err(invalid("query_count must be in 1..=64"));
        }
        if !(1..=MAX_K).contains(&k) {
            return Err(invalid("k must be in 1..=1024"));
        }
        if query_count == 1 || self.is_empty() {
            return self.reserve_search(k);
        }
        let width = query_count.next_power_of_two().max(8);
        let query_count = selection_width(query_count, k.min(self.count));
        let capacity = k.min(self.count).next_power_of_two();
        // Sized by capacity, so appends within the reserve keep this scratch.
        let tile_rows = self.corpus.capacity_range().end.min(TILE_ROWS);
        if self.batch.as_ref().is_some_and(|s| {
            s.width >= width
                && s.k >= capacity
                && s.plans
                    .iter()
                    .any(|p| p.width == width && p.selections.contains_key(&query_count))
        }) {
            return Ok(());
        }
        self.stream.synchronize()?;
        if self
            .batch
            .as_ref()
            .is_none_or(|s| s.width < width || s.k < capacity)
        {
            let old_width = self.batch.as_ref().map_or(0, |s| s.width);
            let old_k = self.batch.as_ref().map_or(0, |s| s.k);
            let mut scratch = BatchScratch::new(
                &self.stream,
                width.max(old_width),
                capacity.max(old_k),
                tile_rows,
                self.padded,
            )?;
            if let Some(old) = self.batch.take() {
                scratch.plans = old.plans;
            }
            self.batch = Some(scratch);
        }
        if let Some(plan) = self
            .batch
            .as_mut()
            .unwrap()
            .plans
            .iter_mut()
            .find(|p| p.width == width)
        {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                plan.selections.entry(query_count)
            {
                let (selection, reports) = crate::selection::SelectionPlan::new(
                    &self.compiler,
                    &self.stream,
                    query_count,
                    TILE_ROWS,
                )?;
                entry.insert(selection);
                self.reports.extend(reports);
            }
            return Ok(());
        }
        let mut reports = Vec::new();
        let mut build = |source, mut spec: hrx::loom::Specialization, queries: usize| {
            spec.set_config("db.batch", queries.to_string());
            if spec.symbol() != "batch_scan" {
                spec.set_config("db.select.limit", TILE_ROWS.to_string());
            }
            let (kernel, report) = compile(&self.compiler, &self.stream, source, spec)?;
            reports.push(report);
            Ok::<_, Error>(kernel)
        };
        let mut scan_spec = kernels::named_spec("batch_scan");
        scan_spec.set_config("db.scan.dimensions", self.padded.to_string());
        let scan = build(kernels::BATCH_SCAN, scan_spec, width)?;
        let running_merge = build(kernels::BATCH_MERGE, kernels::named_spec("batch_merge"), 1)?;
        let (selection, selection_reports) = crate::selection::SelectionPlan::new(
            &self.compiler,
            &self.stream,
            query_count,
            TILE_ROWS,
        )?;
        reports.extend(selection_reports);
        self.batch.as_mut().unwrap().plans.push(BatchPlan {
            width,
            scan,
            selections: std::collections::HashMap::from([(query_count, selection)]),
            running_merge,
        });
        self.reports.extend(reports);
        Ok(())
    }

    /// Search up to 64 row-major FP32 queries, sharing corpus reads across them.
    ///
    /// `queries` contains `batch_size * self.dimensions()` components. Results
    /// retain query order and use the same score/ID ordering as [`Self::search`].
    /// A tiled FP16 × FP32 matrix kernel reuses each corpus tile across the batch;
    /// device selection maintains one running top-k per query. No full corpus ×
    /// batch score matrix or host-side selection is used. Queries are normalized
    /// to FP32 without additional quantization. Accumulation order differs from
    /// `search`, so rounding can change rankings of near-ties.
    ///
    /// An empty batch returns an empty vector. First use of a query width may
    /// allocate and compile; [`Self::reserve_batch`] can prepare it in advance.
    /// The latest query count, effective k and exclusion mode reuse a recorded
    /// command graph. The first search for a shape records it; workspace growth
    /// and [`Self::set_corpus`] invalidate it. Query and exclusion contents can
    /// change between replays without rebuilding the graph.
    ///
    /// ```no_run
    /// # use hrxdb::{Device, Searcher};
    /// # fn main() -> hrxdb::Result<()> {
    /// let device = Device::open(0)?;
    /// let mut db = Searcher::build(&device, 3, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])?;
    /// db.reserve_batch(2, 5)?;
    /// let queries = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    /// let results = db.search_batch(&queries, 5)?;
    /// assert_eq!(results[0][0].id, 0);
    /// assert_eq!(results[1][0].id, 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Rejects incomplete rows, more than 64 queries, k outside 1–1,024, and
    /// nonfinite or zero-norm queries. All queries are validated before execution.
    /// Runtime and compilation failures propagate as errors.
    pub fn search_batch(&mut self, queries: &[f32], k: usize) -> Result<Vec<Vec<Neighbor>>> {
        self.search_batch_excluding(queries, k, &[])
    }

    /// Search a batch with one shared set of excluded insertion IDs.
    ///
    /// Exclusions may be unsorted and repeated, apply only to this call, and
    /// compose with sharding and all supported k values. Each query returns
    /// `min(k, remaining_rows)` matches. Only final neighbors are read back.
    /// Input requirements and errors match [`Self::search_batch`]; out-of-range
    /// excluded IDs are also rejected.
    pub fn search_batch_excluding(
        &mut self,
        queries: &[f32],
        k: usize,
        excluded: &[u32],
    ) -> Result<Vec<Vec<Neighbor>>> {
        let mut output = Vec::new();
        self.search_batch_excluding_into(queries, k, excluded, &mut output)?;
        Ok(output)
    }

    /// Replace a reusable batch output, retaining the capacity of surviving rows.
    /// Invalid input leaves output unchanged; runtime failures may partially write it.
    pub fn search_batch_into(
        &mut self,
        queries: &[f32],
        k: usize,
        output: &mut Vec<Vec<Neighbor>>,
    ) -> Result<()> {
        self.search_batch_excluding_into(queries, k, &[], output)
    }

    /// Batch search with shared exclusions into reusable host output.
    pub fn search_batch_excluding_into(
        &mut self,
        queries: &[f32],
        k: usize,
        excluded: &[u32],
        result: &mut Vec<Vec<Neighbor>>,
    ) -> Result<()> {
        self.search_batch_impl(queries, k, excluded, result, None)
    }

    pub(crate) fn search_batch_impl(
        &mut self,
        queries: &[f32],
        k: usize,
        excluded: &[u32],
        result: &mut Vec<Vec<Neighbor>>,
        profile: Option<&mut Option<ExecutionProfile>>,
    ) -> Result<()> {
        let count = batch_size(self.dimensions, queries.len(), k)?;
        if excluded.iter().any(|&id| id as usize >= self.count) {
            return Err(invalid("excluded ID is outside the index"));
        }
        let mut lengths = [0.0; MAX_BATCH];
        for (i, row) in queries.chunks_exact(self.dimensions).enumerate() {
            lengths[i] = norm(row, self.dimensions)?;
        }
        result.resize_with(count, Vec::new);
        for row in result.iter_mut() {
            row.clear();
        }
        if count == 0 {
            return Ok(());
        }
        if self.is_empty() {
            return Ok(());
        }
        if count == 1 {
            return self.search_impl(queries, k, excluded, &mut result[0], profile);
        }
        self.stream.synchronize()?;
        let mut remaining = self.count;
        if !excluded.is_empty() {
            // SAFETY: the stream is complete and the bitmap is exclusively owned.
            let bitmap = unsafe { self.exclusions.bytes_mut() };
            bitmap.fill(0);
            for &id in excluded {
                let byte = &mut bitmap[id as usize / 8];
                let bit = 1 << (id % 8);
                if *byte & bit == 0 {
                    *byte |= bit;
                    remaining -= 1;
                }
            }
        }
        if !excluded.is_empty() {
            // SAFETY: no device work has been submitted since synchronization.
            unsafe { self.exclusions.publish()? };
        }
        let k = k.min(remaining);
        if k == 0 {
            return Ok(());
        }
        self.reserve_batch(count, k)?;
        let width = count.next_power_of_two().max(8);
        let scratch = self.batch.as_mut().unwrap();
        // SAFETY: this call synchronized before preparing exclusions/reserving
        // workspace; no GPU work has been submitted since that synchronization.
        let bytes = unsafe { scratch.query.bytes_mut() };
        bytes.fill(0);
        for (q, row) in queries.chunks_exact(self.dimensions).enumerate() {
            for (dim, &value) in row.iter().enumerate() {
                let at = (dim * width + q) * 4;
                bytes[at..at + 4]
                    .copy_from_slice(&((value as f64 / lengths[q]) as f32).to_le_bytes());
            }
        }
        // SAFETY: the query remains exclusively host-owned here.
        unsafe { scratch.query.publish()? };
        let bytes = count * k * 4;
        let exclusions = (!excluded.is_empty()).then(|| self.exclusions.buffer().binding());
        if let Some(profile) = profile {
            let start = Instant::now();
            let mut commands = Commands::profile(&self.stream)?;
            scratch.search_commands(&self.corpus, exclusions, count, k, &mut commands)?;
            let (graph, commands) = commands.finish_profiled()?;
            *profile = Some(profiling::replay(
                &mut self.stream,
                graph,
                commands,
                start.elapsed().as_secs_f64() * 1000.0,
            )?);
        } else {
            scratch.replay(&mut self.stream, &self.corpus, exclusions, count, k)?;
        }
        // SAFETY: the stream completed both copies to this owned readback buffer.
        let readback = unsafe { scratch.readback.bytes()? };
        for (q, neighbors) in result.iter_mut().enumerate() {
            neighbors.reserve(k);
            for rank in 0..k {
                let at = (q * k + rank) * 4;
                let similarity = f32::from_le_bytes(readback[at..at + 4].try_into().unwrap());
                let id =
                    u32::from_le_bytes(readback[bytes + at..bytes + at + 4].try_into().unwrap());
                if similarity != f32::NEG_INFINITY {
                    neighbors.push(Neighbor { id, similarity });
                }
            }
        }
        Ok(())
    }
}

fn select_tile<'a>(
    commands: &mut Commands<'a>,
    scratch: &'a BatchScratch,
    plan: &'a BatchPlan,
    rows: usize,
    start: usize,
    k: usize,
    queries: usize,
) -> Result<usize> {
    crate::selection::select_score_commands(
        commands,
        &scratch.candidates,
        &plan.selections[&queries],
        crate::selection::SelectionInput {
            scores: scratch.scores.binding(),
            rows,
            start,
            k,
            batch: queries,
        },
    )
}

impl BatchScratch {
    fn replay(
        &mut self,
        stream: &mut Stream,
        corpus: &Corpus,
        exclusions: Option<hrx::View<'_>>,
        count: usize,
        k: usize,
    ) -> Result<()> {
        #[cfg(test)]
        if self.immediate {
            self.search_commands(
                corpus,
                exclusions,
                count,
                k,
                &mut Commands::immediate(stream),
            )?;
            return stream.synchronize();
        }
        let masked = exclusions.is_some();
        if !self
            .search_graph
            .as_ref()
            .is_some_and(|(queries, cached_k, mask, _)| {
                (*queries, *cached_k, *mask) == (count, k, masked)
            })
        {
            self.search_graph = None;
            let mut commands = Commands::record(stream)?;
            self.search_commands(corpus, exclusions, count, k, &mut commands)?;
            self.search_graph = Some((count, k, masked, commands.finish()?));
        }
        stream.launch(&mut self.search_graph.as_mut().unwrap().3)?;
        stream.synchronize()?;
        Ok(())
    }

    fn search_commands<'a>(
        &'a self,
        corpus: &'a Corpus,
        exclusions: Option<hrx::View<'a>>,
        count: usize,
        k: usize,
        commands: &mut Commands<'a>,
    ) -> Result<()> {
        self.scan_commands(
            corpus,
            self.query.buffer().binding(),
            exclusions,
            count,
            k,
            commands,
        )?;
        let bytes = count * k * 4;
        commands.copy(
            self.readback.buffer().try_slice(0, bytes)?,
            self.running.0.try_slice(0, bytes)?,
        )?;
        commands.copy(
            self.readback.buffer().try_slice(bytes, bytes)?,
            self.running.1.try_slice(0, bytes)?,
        )?;
        Ok(())
    }

    pub(crate) fn scan(
        &self,
        stream: &Stream,
        corpus: &Corpus,
        query: hrx::View<'_>,
        exclusions: Option<hrx::View<'_>>,
        queries: usize,
        k: usize,
    ) -> Result<()> {
        self.scan_commands(
            corpus,
            query,
            exclusions,
            queries,
            k,
            &mut Commands::immediate(stream),
        )
    }

    fn scan_commands<'a>(
        &'a self,
        corpus: &'a Corpus,
        query: hrx::View<'a>,
        exclusions: Option<hrx::View<'a>>,
        queries: usize,
        k: usize,
        commands: &mut Commands<'a>,
    ) -> Result<()> {
        let width = queries.next_power_of_two().max(8);
        let queries = selection_width(queries, k);
        let plan = self.plans.iter().find(|p| p.width == width).unwrap();
        let tiles: usize = corpus
            .inner
            .shards
            .iter()
            .map(|shard| shard.count.div_ceil(self.tile_rows))
            .sum();
        // Alternate destinations so no merge reads and writes the same list.
        // Choose the first destination so the final tile always lands in running,
        // including when shard boundaries add extra short tiles.
        let start_in_scratch = tiles.is_multiple_of(2);
        #[cfg(test)]
        let start_in_scratch = start_in_scratch && self.staged_merge.is_none();
        let (mut running, mut spare) = if start_in_scratch {
            (&self.merge_scratch, &self.running)
        } else {
            (&self.running, &self.merge_scratch)
        };
        let mut started = false;
        for shard in &corpus.inner.shards {
            for local in (0..shard.count).step_by(self.tile_rows) {
                let rows = (shard.count - local).min(self.tile_rows);
                let start = shard.start + local;
                let mut constants = Constants::new();
                constants.push(rows as u32)?;
                constants.push(u32::from(exclusions.is_some()))?;
                constants.push(start as u32)?;
                constants.push(corpus.len() as u32)?;
                // SAFETY: the tile belongs to this shard, has at most TILE_ROWS
                // rows, and its dimensions match the compiled kernel. The query
                // and score buffers reserve width rows. Norms use local row
                // offsets; the exclusion bitmap uses global insertion IDs.
                unsafe {
                    commands.dispatch(
                        &plan.scan,
                        [rows.div_ceil(64) as u32, 1, 1],
                        [(width * 4) as u32, 1, 1],
                        &constants,
                        &[
                            shard.vectors(corpus.padded_dimensions(), local, rows)?,
                            query,
                            shard.inverse_norms(local, rows)?,
                            exclusions.unwrap_or(self.scores.binding()),
                            self.scores.binding(),
                        ],
                    )?;
                }
                let output = select_tile(commands, self, plan, rows, start, k, queries)?;
                let bytes = queries * k * 4;
                if !started {
                    commands.copy(
                        running.0.try_slice(0, bytes)?,
                        self.candidates[output].0.try_slice(0, bytes)?,
                    )?;
                    commands.copy(
                        running.1.try_slice(0, bytes)?,
                        self.candidates[output].1.try_slice(0, bytes)?,
                    )?;
                    started = true;
                } else {
                    #[cfg(test)]
                    if let Some(staged) = &self.staged_merge {
                        staged.record(commands, running, &self.candidates[output], queries, k)?;
                        continue;
                    }
                    let mut constants = Constants::new();
                    constants.push(queries as u32)?;
                    constants.push(k as u32)?;
                    constants.push(k.ilog2() + 1)?;
                    // SAFETY: both inputs contain queries*k sorted entries.
                    // The output is the other running pair, never either input.
                    // k is in 1..=1024; its bit length bounds binary search.
                    unsafe {
                        commands.dispatch(
                            &plan.running_merge,
                            [queries as u32, 1, 1],
                            [256, 1, 1],
                            &constants,
                            &[
                                running.0.binding(),
                                running.1.binding(),
                                self.candidates[output].0.binding(),
                                self.candidates[output].1.binding(),
                                spare.0.binding(),
                                spare.1.binding(),
                            ],
                        )?;
                    }
                    std::mem::swap(&mut running, &mut spare);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires gfx1151"]
    fn cached_batch_replays_follow_queries_shapes_masks_and_snapshots() -> Result<()> {
        let device = Device::open(0)?;
        let mut corpus = Corpus::build(
            &device,
            3,
            (0..2057).map(|i| [1.0, (i % 17) as f32 - 8.0, (i % 11) as f32 - 5.0]),
        )?;
        let mut cached = corpus.searcher()?;
        let mut immediate = corpus.searcher()?;
        for snapshot in 0..3 {
            if snapshot == 1 {
                // Same dimensions, count, and capacity; bindings still change.
                corpus = corpus.update(&[0, 1024, 2056], [[0.0, 1.0, 0.0]; 3])?;
            } else if snapshot == 2 {
                corpus = corpus.append([[0.0, 0.0, 1.0]; 31])?;
            }
            cached.set_corpus(corpus.clone())?;
            immediate.set_corpus(corpus.clone())?;
            assert!(
                cached
                    .batch
                    .as_ref()
                    .is_none_or(|b| b.search_graph.is_none())
            );
            for (iteration, (count, k, masked)) in [
                (3, 5, false),
                (3, 5, false),
                (4, 5, false),
                (4, 5, true),
                (4, 5, true),
                (60, 33, true),
                (3, 1024, false),
                (3, 10, true),
                (3, 10, false),
                (1, 5, false),
                (0, 5, false),
                (3, 5, false),
            ]
            .into_iter()
            .enumerate()
            {
                let queries: Vec<_> = (0..count)
                    .flat_map(|q| {
                        [
                            1.0,
                            ((q + iteration) % 7) as f32 - 3.0,
                            (iteration % 5) as f32 - 2.0,
                        ]
                    })
                    .collect();
                let exclusions = if masked {
                    vec![iteration as u32, 1024, 2056, iteration as u32]
                } else {
                    Vec::new()
                };
                if count > 1 {
                    immediate.reserve_batch(count, k)?;
                    immediate.batch.as_mut().unwrap().immediate = true;
                }
                let expected = immediate.search_batch_excluding(&queries, k, &exclusions)?;
                let actual = cached.search_batch_excluding(&queries, k, &exclusions)?;
                assert_eq!(
                    actual, expected,
                    "snapshot={snapshot} iteration={iteration}"
                );
                for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                    assert_eq!(a.similarity.to_bits(), b.similarity.to_bits());
                }
                if count > 1 {
                    let graph = cached
                        .batch
                        .as_ref()
                        .unwrap()
                        .search_graph
                        .as_ref()
                        .unwrap();
                    assert_eq!((graph.0, graph.1, graph.2), (count, k, masked));
                }
            }
            let excluded: Vec<_> = (0..corpus.len() as u32).collect();
            assert!(
                cached
                    .search_batch_excluding(&[1.0; 9], 10, &excluded)?
                    .iter()
                    .all(Vec::is_empty)
            );
            // The same masked shape can have a different effective k.
            let excluded: Vec<_> = (2..corpus.len() as u32).collect();
            assert_eq!(
                cached.search_batch_excluding(&[1.0; 9], 10, &excluded)?,
                immediate.search_batch_excluding(&[1.0; 9], 10, &excluded)?
            );
            assert_eq!(
                cached
                    .batch
                    .as_ref()
                    .unwrap()
                    .search_graph
                    .as_ref()
                    .unwrap()
                    .1,
                2
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn shorter_final_dispatch_has_correct_scores_and_ids() -> Result<()> {
        let device = Device::open(0)?;
        for tail in [1usize, 31, 65] {
            let count = TILE_ROWS + tail;
            let row = |r: usize| -> [f32; 3] {
                if r < TILE_ROWS {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, ((r - TILE_ROWS) % 7) as f32 / 8.0]
                }
            };
            let encoded = (0..count).map(|r| {
                row(r)
                    .into_iter()
                    .flat_map(|x| f16::from_f32(x).to_le_bytes())
                    .collect::<Vec<_>>()
            });
            let mut db = Searcher::build_fp16(&device, 3, encoded)?;
            // Exercise cached wide -> narrow -> wide query plans as well as
            // a final corpus tile smaller than all preceding dispatches.
            for batch in [60usize, 2, 3, 4, 8, 17, 33, 64, 9, 3] {
                let queries: Vec<f32> = (0..batch)
                    .flat_map(|q| {
                        [
                            0.0,
                            if q % 2 == 0 { 1.0 } else { -1.0 },
                            (q % 5) as f32 / 8.0,
                        ]
                    })
                    .collect();
                let results = db.search_batch(&queries, 5)?;
                let mut bytes = vec![0u8; batch * tail * 4];
                db.stream.read_blocking(
                    db.batch
                        .as_ref()
                        .unwrap()
                        .scores
                        .try_slice(0, bytes.len())?,
                    &mut bytes,
                )?;
                for (q, query) in queries.as_chunks::<3>().0.iter().enumerate() {
                    let qnorm = norm(query, 3)?;
                    for r in 0..tail {
                        let x = row(TILE_ROWS + r);
                        let inv = (1.0 / norm(&x, 3)?) as f32;
                        let expected = x
                            .iter()
                            .zip(query)
                            .map(|(&x, &y)| x as f64 * ((y as f64 / qnorm) as f32) as f64)
                            .sum::<f64>()
                            * inv as f64;
                        let at = (q * tail + r) * 4;
                        let actual = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                        assert!(
                            actual.is_finite() && (actual as f64 - expected).abs() < 3e-6,
                            "tail={tail}, batch={batch}, query={q}, row={r}: {actual} vs {expected}"
                        );
                    }
                    if q % 2 == 0 {
                        assert!(
                            results[q]
                                .iter()
                                .take(tail.min(5))
                                .all(|n| n.id as usize >= TILE_ROWS)
                        );
                    } else {
                        assert_eq!(
                            results[q].iter().map(|n| n.id).collect::<Vec<_>>(),
                            vec![0, 1, 2, 3, 4]
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn validates_batch_shape_without_gpu() {
        assert_eq!(batch_size(3, 0, 5).unwrap(), 0);
        assert_eq!(batch_size(3, 180, 5).unwrap(), 60);
        assert_eq!(batch_size(3, 192, 1024).unwrap(), 64);
        for (values, k) in [(1, 5), (181, 5), (195, 5), (0, 0), (3, 1025)] {
            assert!(batch_size(3, values, k).is_err());
        }
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn tiled_batches_merge_global_ids_and_shared_exclusions() -> Result<()> {
        let device = Device::open(0)?;
        let rows: Vec<_> = (0..2065)
            .map(|i| {
                let mut row = [0.0; 3];
                row[i % 3] = if i % 7 == 0 { -1.0 } else { 1.0 };
                row
            })
            .collect();
        let mut db = Searcher::build_encoded(
            &device,
            3,
            &rows,
            ScanConfig::default(),
            1031 * 128,
            |row, d, p, bytes| encode(*row, d, p, bytes),
        )?;
        assert_eq!(db.shard_count(), 3);
        let queries: Vec<_> = (0..60).flat_map(|i| rows[i]).collect();
        db.reserve_batch(60, 1024)?;
        // Force several short, unaligned tiles in each unaligned shard. This
        // exercises masks crossing word boundaries and k larger than a tile.
        db.batch.as_mut().unwrap().tile_rows = 257;
        let all: Vec<_> = (0..db.len() as u32).collect();
        let some = [0, 7, 8, 31, 32, 33, 256, 257, 1030, 1031, 1032, 2064, 1031];
        for excluded in [&[][..], &some, &all[2..], &all] {
            for k in [1, 5, 32, 33, 1024] {
                for count in [3usize, 60, 4] {
                    let queries = &queries[..count * 3];
                    let batch = db.search_batch_excluding(queries, k, excluded)?;
                    for (actual, query) in batch.iter().zip(queries.as_chunks::<3>().0) {
                        assert_eq!(*actual, db.search_excluding(query, k, excluded)?);
                    }
                }
            }
        }
        assert_eq!(
            db.search_batch(&queries, 5)?[0],
            db.search(&queries[..3], 5)?
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn batch_validates_all_queries_and_reuses_workspace() -> Result<()> {
        let device = Device::open(0)?;
        let mut db = Searcher::build(&device, 3, [[1.0, 0.0, 0.0]; 67])?;
        assert_eq!(db.batch_workspace_bytes(), 0);
        let valid = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for bad in [
            vec![1.0],
            vec![1.0; 65 * 3],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, f32::NAN, 0.0, 0.0],
            vec![f32::INFINITY; 6],
        ] {
            assert!(db.search_batch(&bad, 5).is_err());
        }
        assert_eq!(db.batch_workspace_bytes(), 0);
        assert!(db.search_batch(&[], 0).is_err());
        assert!(db.search_batch(&valid, 1025).is_err());
        assert!(db.search_batch_excluding(&valid, 5, &[67]).is_err());
        assert!(db.search_batch(&[], 5)?.is_empty());
        for size in [0, 65] {
            assert!(db.reserve_batch(size, 5).is_err());
        }
        assert!(db.reserve_batch(2, 0).is_err());
        db.reserve_batch(60, 1024)?;
        let bytes = db.batch_workspace_bytes();
        let first = db.search_batch(&valid, 5)?;
        let reports = db.compilation_reports().len();
        assert_eq!(db.search_batch(&valid, 5)?, first);
        assert_eq!(db.compilation_reports().len(), reports);
        assert_eq!(db.batch_workspace_bytes(), bytes);
        assert!(first[0].iter().all(|n| n.similarity == 1.0));
        assert!(first[1].iter().all(|n| n.similarity == 0.0));
        let mut empty = Searcher::build(&device, 3, std::iter::empty::<[f32; 3]>())?;
        assert_eq!(empty.search_batch(&valid, 1024)?, vec![vec![], vec![]]);
        assert!(empty.search_batch(&[0.0; 6], 5).is_err());
        assert!(empty.search_batch_excluding(&valid, 5, &[0]).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn batch_scores_match_fp16_cpu_reference() -> Result<()> {
        let device = Device::open(0)?;
        for d in [1usize, 3, 127, 769] {
            let rows: Vec<Vec<f32>> = (0..67)
                .map(|i| {
                    (0..d)
                        .map(|j| {
                            if i == 0 {
                                f16::from_bits(1).to_f32()
                            } else if i == 1 {
                                65504.0
                            } else {
                                f16::from_f32(((i * 37 + j * 17) % 101) as f32 - 49.25).to_f32()
                            }
                        })
                        .collect()
                })
                .collect();
            let raw: Vec<Vec<u8>> = rows
                .iter()
                .map(|r| {
                    r.iter()
                        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                        .collect()
                })
                .collect();
            let mut db = Searcher::build_fp16(&device, d, &raw)?;
            for batch in [2usize, 3, 4, 8, 9, 17, 33, 64, 3] {
                let queries: Vec<_> = (0..batch)
                    .flat_map(|q| (0..d).map(move |j| ((q * 43 + j * 11) % 97) as f32 - 47.25))
                    .collect();
                db.search_batch(&queries, 5)?;
                let scratch = db.batch.as_ref().unwrap();
                let mut bytes = vec![0; batch * rows.len() * 4];
                db.stream
                    .read_blocking(scratch.scores.try_slice(0, bytes.len())?, &mut bytes)?;
                for (q, query) in queries.chunks_exact(d).enumerate() {
                    let length = norm(query, d)?;
                    for (r, row) in rows.iter().enumerate() {
                        let inverse = (1.0 / norm(row, d)?) as f32;
                        let expected: f64 = row
                            .iter()
                            .zip(query)
                            .map(|(&x, &y)| x as f64 * ((y as f64 / length) as f32) as f64)
                            .sum::<f64>()
                            * inverse as f64;
                        let at = (q * rows.len() + r) * 4;
                        let actual = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                        assert!(
                            (actual as f64 - expected).abs() < 3e-6,
                            "d={d} batch={batch} q={q} r={r}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "batch_bench.rs"]
mod bench;

#[cfg(test)]
#[path = "batch_merge_tests.rs"]
mod merge_tests;
