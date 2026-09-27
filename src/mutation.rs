//! New snapshots from existing ones: appended rows, replaced rows, compaction.
//!
//! A snapshot is an ordered list of segments, each a run of insertion IDs held
//! by one shared [`Allocation`]. Deriving a snapshot never writes a row that an
//! existing snapshot can read as a logical row:
//!
//! - Appends write rows past an allocation's `claimed` mark, which no snapshot
//!   references, after advancing the mark with a compare-and-swap. The rows an
//!   older snapshot sees as slack may change while it runs. That is within the
//!   slack contract: hrxdb's scans guard or specialize on each segment's
//!   extent and never select slack, and caller kernels must mask it. Slack only
//!   ever holds zero or finite rows, and any byte-wise mixture of zero with a
//!   finite IEEE value is finite, so a concurrent reader cannot observe NaN.
//! - Updates copy the touched pages to new allocations unless no other handle
//!   can observe the rows (see [`Corpus::update`]).
use crate::*;
use hrx::View;
use std::sync::atomic::Ordering;

/// Rows reserved by the first append into a new tail allocation.
const TAIL_ROWS: usize = 4_096;

/// Untouched pages between two touched ones that an update copies rather than
/// keeping as a separate segment: 16 pages are one 262,144-row batch tile.
/// Batched search pays about 0.25 ms of selection and merging per tile
/// (60 queries, 3M x 768), so pieces shorter than a tile cost more search time
/// than their copy. With this bound, 100 random updates over 3M rows produce
/// one copy instead of 175 segments that slowed batched search by 68%.
const MERGE_PAGES: usize = 16;

/// Destination of consecutive appended rows: allocation and first row in it.
struct Placement {
    storage: Arc<Allocation>,
    row: usize,
    rows: usize,
}

/// Stream and kernels shared by the snapshots derived from one build.
pub(crate) struct Lineage {
    stream: Stream,
    scatter: Option<Kernel>,
}

/// A mutation's stream, and every buffer its queued work uses.
///
/// Streams retain neither staging nor destination buffers for queued copies,
/// so they are kept here until the work completes, including on error paths.
struct Writer<'a> {
    stream: &'a mut Stream,
    scatter: &'a mut Option<Kernel>,
    padded: usize,
    allocations: Vec<Arc<Allocation>>,
    staging: Vec<Arc<Buffer>>,
}

impl Writer<'_> {
    fn allocate(
        &mut self,
        capacity: usize,
        rows: usize,
        appended: bool,
    ) -> Result<Arc<Allocation>> {
        let storage = Allocation::new(self.stream, self.padded, capacity, rows, appended)?;
        self.allocations.push(storage.clone());
        Ok(storage)
    }

    /// Host bytes in new GPU-coherent staging storage. A mapped write of this
    /// kind costs microseconds where a first `Stream::upload` costs milliseconds.
    fn stage(&mut self, bytes: &[u8]) -> Result<Arc<Buffer>> {
        let staging = self.stream.allocate_shared(bytes.len())?;
        // SAFETY: the allocation is new, so no device work uses it, and it holds
        // at least `bytes.len()` bytes. Shared allocations are coherent and need
        // no cache maintenance before the device reads them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                staging.device_ptr()?.cast::<u8>(),
                bytes.len(),
            );
        }
        let staging = Arc::new(staging);
        self.staging.push(staging.clone());
        Ok(staging)
    }

    /// Copy between equal ranges, in pieces within the 4 GiB native limit.
    fn copy(&self, destination: View<'_>, source: View<'_>) -> Result<()> {
        const PIECE: usize = 1 << 30;
        for at in (0..source.len()).step_by(PIECE) {
            let bytes = PIECE.min(source.len() - at);
            self.stream
                .copy(destination.slice(at, bytes)?, source.slice(at, bytes)?)?;
        }
        Ok(())
    }

    /// Write staged rows to destination rows in one dispatch. `routes` pairs a
    /// staged row with a distinct row of `vectors`/`norms`, which hold `rows`
    /// rows of this corpus's stride.
    fn scatter(
        &mut self,
        staged: [&Buffer; 2],
        routes: &[[u32; 2]],
        vectors: View<'_>,
        norms: View<'_>,
        rows: usize,
    ) -> Result<()> {
        if self.scatter.is_none() {
            let compiler = hrx::loom::Compiler::for_stream(None, self.stream)?;
            let mut spec = kernels::named_spec("scatter");
            spec.set_config("scatter.words", (self.padded / 2).to_string());
            spec.set_config("scatter.limit", (MAX_ELEMENTS / self.padded).to_string());
            let source = include_str!("../kernels/scatter.loom");
            *self.scatter = Some(compile(&compiler, self.stream, source, spec)?.0);
        }
        let bytes: Vec<u8> = routes
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let routing = self.stage(&bytes)?;
        let mut constants = Constants::new();
        constants.push(routes.len() as u32)?;
        constants.push((staged[1].bytes() / 4) as u32)?;
        constants.push(rows as u32)?;
        // SAFETY: callers route validated staged rows to distinct destination
        // rows below `rows`; both views hold `rows` rows of the specialized
        // stride. Each workgroup writes only its own destination row.
        unsafe {
            self.stream.dispatch(
                self.scatter.as_ref().expect("scatter kernel"),
                [routes.len() as u32, 1, 1],
                [128, 1, 1],
                &constants,
                &[
                    staged[0].binding(),
                    staged[1].binding(),
                    routing.binding(),
                    vectors,
                    norms,
                ],
            )
        }
    }

    /// Complete queued work, after which the buffers it used may be released.
    fn synchronize(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.allocations.clear();
        self.staging.clear();
        Ok(())
    }
}

impl Corpus {
    /// Run `mutate` on the stream shared by every snapshot derived from the
    /// same build, and complete its work before any buffer it used can be
    /// released. A new stream costs about 2 ms and its first copy about 8 ms
    /// (transfer kernels load per stream), so one is kept for the lineage.
    /// Mutations of one lineage are serialized by it.
    fn write<T>(&self, mutate: impl FnOnce(&mut Writer<'_>) -> Result<T>) -> Result<T> {
        let mut slot = self
            .inner
            .writer
            .lock()
            .map_err(|_| invalid("corpus writer poisoned"))?;
        let mut lineage = match slot.take() {
            Some(lineage) => lineage,
            None => Lineage {
                stream: self.stream()?,
                scatter: None,
            },
        };
        // Budget policy belongs to the calling handle, not the lineage. All
        // previous work is complete, so a budgeted call can rebind the stream
        // without losing its cached kernels. HRX has no budget-detach API;
        // returning to an unbudgeted sibling therefore needs a fresh stream.
        match &self.workspace_budget {
            Some(budget) => lineage.stream = lineage.stream.with_memory_budget(budget.clone()),
            None if lineage.stream.memory_budget().is_some() => {
                lineage = Lineage {
                    stream: self.stream()?,
                    scatter: None,
                };
            }
            None => {}
        }
        let mut writer = Writer {
            stream: &mut lineage.stream,
            scatter: &mut lineage.scatter,
            padded: self.padded_dimensions(),
            allocations: Vec::new(),
            staging: Vec::new(),
        };
        let result = mutate(&mut writer);
        if let Err(error) = writer.synchronize() {
            // Completion is unknown: never release memory the GPU may use,
            // and start the next mutation on a new stream.
            std::mem::forget(std::mem::take(&mut writer.allocations));
            std::mem::forget(std::mem::take(&mut writer.staging));
            return Err(error);
        }
        *slot = Some(lineage);
        result
    }

    /// Return a snapshot with FP32 `rows` appended at insertion IDs
    /// `len()..len() + n`, normalized and converted exactly as [`Self::build`]
    /// does.
    ///
    /// Existing rows are shared, not copied, and this snapshot, its clones and
    /// its searchers keep seeing exactly their rows. Rows go into the reserve
    /// of the last allocation when this snapshot is its most recent extension;
    /// otherwise the last appended segment moves to an allocation of twice its
    /// capacity (4,096 rows at first), so repeated appends leave one tail
    /// segment and copy each row an amortized constant number of times. Cost
    /// is one host conversion and staged copy of the new rows, and the
    /// occasional device copy of the tail when it grows. Work completes before
    /// this returns; allocations use [`Self::stream`]'s workspace budget.
    ///
    /// Searchers stay bound to their snapshot; serve the new rows with
    /// [`Searcher::set_corpus`] or a new searcher. Side arrays sized by
    /// [`Self::capacity_range`] must be rebuilt for the new snapshot.
    ///
    /// ```no_run
    /// # fn example(
    /// #     corpus: &mut hrxdb::Corpus,
    /// #     searcher: &mut hrxdb::Searcher,
    /// #     fresh: &[Vec<f32>],
    /// # ) -> hrxdb::Result<()> {
    /// let first = corpus.len(); // The new rows take IDs first..first + fresh.len().
    /// *corpus = corpus.append(fresh)?;
    /// searcher.set_corpus(corpus.clone())?; // Waits for its queued work.
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Rows must be finite and nonzero with [`Self::dimensions`] components,
    /// and the total must stay within 2^30 rows. On error no snapshot changes;
    /// reserve rows written before the error are not reused.
    pub fn append<I, R>(&self, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        self.append_encoded(rows, |r, d, p, b| encode(r.as_ref(), d, p, b))
    }

    /// Append little-endian FP16 rows, preserving their values as
    /// [`Self::build_fp16`] does. Placement, cost and errors match
    /// [`Self::append`].
    pub fn append_fp16<I, R>(&self, rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        self.append_encoded(rows, |r, d, p, b| encode_fp16(r.as_ref(), d, p, b))
    }

    fn append_encoded<I, R>(
        &self,
        rows: I,
        mut encode_row: impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut input = rows.into_iter();
        let n = input.len();
        if n == 0 {
            return Ok(self.clone());
        }
        let count = self
            .len()
            .checked_add(n)
            .filter(|&c| c <= MAX_ROWS)
            .ok_or_else(|| invalid("row count exceeds 2^30"))?;
        let padded = self.padded_dimensions();
        let limit = shard_rows(padded, MAX_ELEMENTS);
        let shards = self.write(|w| {
            let mut shards = self.inner.shards.clone();
            let mut placements = Vec::new();
            let mut remaining = n;
            if let Some(last) = shards.last_mut() {
                let end = last.offset + last.count;
                let reserve = last.storage.capacity - end;
                let movable = last.storage.appended && last.count + n <= limit;
                if reserve >= n || (reserve > 0 && !movable) {
                    let rows = reserve.min(n);
                    // Fails when another snapshot already extended these rows.
                    if last
                        .storage
                        .claimed
                        .compare_exchange(end, end + rows, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        placements.push(Placement {
                            storage: last.storage.clone(),
                            row: end,
                            rows,
                        });
                        last.count += rows;
                        remaining -= rows;
                    }
                }
                if remaining == n && movable {
                    // Move the tail: copy its rows into twice the capacity.
                    let rows = last.count + n;
                    let capacity = shard_capacity(rows)
                        .max(2 * last.capacity)
                        .max(TAIL_ROWS)
                        .min(limit);
                    let storage = w.allocate(capacity, rows, true)?;
                    w.copy(
                        storage.data.try_slice(0, last.count * padded * 2)?,
                        last.vectors(padded, 0, last.count)?,
                    )?;
                    w.copy(
                        storage.norms.try_slice(0, last.count * 4)?,
                        last.inverse_norms(0, last.count)?,
                    )?;
                    placements.push(Placement {
                        storage: storage.clone(),
                        row: last.count,
                        rows: n,
                    });
                    *last = Shard {
                        storage,
                        offset: 0,
                        start: last.start,
                        count: rows,
                        capacity,
                    };
                    remaining = 0;
                }
            }
            while remaining > 0 {
                let rows = remaining.min(limit);
                let capacity = shard_capacity(rows).max(TAIL_ROWS).min(limit);
                let storage = w.allocate(capacity, rows, true)?;
                placements.push(Placement {
                    storage: storage.clone(),
                    row: 0,
                    rows,
                });
                shards.push(Shard {
                    storage,
                    offset: 0,
                    start: count - remaining,
                    count: rows,
                    capacity,
                });
                remaining -= rows;
            }
            self.write_rows(w, &placements, &mut input, &mut encode_row)?;
            Ok(shards)
        })?;
        if input.next().is_some() {
            return Err(invalid("row iterator returned more rows than declared"));
        }
        Ok(self.derive(shards, count))
    }

    /// Encode rows into their placements in bounded chunks, as build does.
    fn write_rows<R>(
        &self,
        w: &mut Writer<'_>,
        placements: &[Placement],
        rows: &mut impl Iterator<Item = R>,
        encode_row: &mut impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<()> {
        let (dimensions, padded) = (self.dimensions(), self.padded_dimensions());
        let total: usize = placements.iter().map(|p| p.rows).sum();
        let chunk_rows = (CHUNK_BYTES / (padded * 2)).max(1).min(total);
        let _conversion = self
            .workspace_budget
            .as_ref()
            .map(|budget| budget.reserve(chunk_rows * (padded * 2 + 4)))
            .transpose()?;
        let mut chunk = Vec::with_capacity(chunk_rows * padded * 2);
        let mut norms = Vec::with_capacity(chunk_rows * 4);
        for placement in placements {
            for local in (0..placement.rows).step_by(chunk_rows) {
                chunk.clear();
                norms.clear();
                for _ in local..(local + chunk_rows).min(placement.rows) {
                    let row = rows
                        .next()
                        .ok_or_else(|| invalid("row iterator returned fewer rows than declared"))?;
                    let inverse = encode_row(&row, dimensions, padded, &mut chunk)?;
                    norms.extend_from_slice(&inverse.to_le_bytes());
                }
                let at = placement.row + local;
                let staged = w.stage(&chunk)?;
                w.copy(
                    placement
                        .storage
                        .data
                        .try_slice(at * padded * 2, chunk.len())?,
                    staged.binding(),
                )?;
                let staged = w.stage(&norms)?;
                w.copy(
                    placement.storage.norms.try_slice(at * 4, norms.len())?,
                    staged.binding(),
                )?;
                // Bound staging as well as the conversion buffers.
                w.synchronize()?;
            }
        }
        Ok(())
    }
    /// Return a snapshot with FP32 rows replacing the vectors at insertion
    /// IDs `ids`, normalized and converted exactly as [`Self::build`] does.
    /// `rows` yields one row per ID, in the order of `ids`.
    ///
    /// Other handles never observe the change. When this handle is the only
    /// one to its storage and no other snapshot shares any of its allocations
    /// (no clones, searchers, prepared searches or residency cache entries),
    /// rows are rewritten in place by one scatter dispatch per shard: the cost
    /// is the staged copy of the new rows, under a millisecond for 100 rows.
    /// Complete custom kernels that read this corpus first, as before dropping
    /// it. Release searchers with [`Searcher::set_corpus`] or by dropping them.
    ///
    /// Otherwise the update copies on write. Every 16,384-row page of an
    /// allocation that holds an updated row is copied to a new allocation on
    /// the device, together with any untouched gap shorter than 16 pages
    /// (262,144 rows, one batch-search tile) between two such pages, and the
    /// rows are written there. The cost is that many rows times
    /// `padded_dimensions * 2 + 4` bytes of device copy, and as much new memory
    /// while older snapshots share the original allocations: 100 consecutive
    /// IDs copy one or two pages (25 MB at 768 dimensions); 100 random IDs over
    /// millions of rows copy nearly everything. Each copied run can split a
    /// shard in three, and runs are at least a tile apart, so fragmentation
    /// stays within a few shards per 262,144 rows. [`Self::compact`] restores
    /// the build layout.
    ///
    /// # Errors
    /// IDs must be distinct and less than [`Self::len`], and at most
    /// 2^32 / [`Self::padded_dimensions`] per call; rows have the requirements
    /// of [`Self::append`]. Invalid input is rejected before any GPU work. The
    /// handle is consumed either way; pass a clone to keep it.
    pub fn update<I, R>(self, ids: &[u32], rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[f32]>,
    {
        self.update_encoded(ids, rows, |r, d, p, b| encode(r.as_ref(), d, p, b))
    }

    /// Replace rows with little-endian FP16 values, preserving them as
    /// [`Self::build_fp16`] does. Placement, cost and errors match
    /// [`Self::update`].
    pub fn update_fp16<I, R>(self, ids: &[u32], rows: I) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
        R: AsRef<[u8]>,
    {
        self.update_encoded(ids, rows, |r, d, p, b| encode_fp16(r.as_ref(), d, p, b))
    }

    fn update_encoded<I, R>(
        mut self,
        ids: &[u32],
        rows: I,
        mut encode_row: impl FnMut(&R, usize, usize, &mut Vec<u8>) -> Result<f32>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = R>,
        I::IntoIter: ExactSizeIterator,
    {
        let rows = rows.into_iter();
        if rows.len() != ids.len() {
            return Err(invalid("update needs exactly one row per ID"));
        }
        if ids.iter().any(|&id| id as usize >= self.len()) {
            return Err(invalid("updated ID is outside the corpus"));
        }
        let n = ids.len();
        if n == 0 {
            return Ok(self);
        }
        let (dimensions, padded) = (self.dimensions(), self.padded_dimensions());
        if n > MAX_ELEMENTS / padded {
            return Err(invalid(
                "an update replaces at most 2^32 / padded_dimensions rows",
            ));
        }
        let stride = padded * 2;
        // Retain the charge while conversion and GPU staging coexist. Reserve
        // before allocating or consuming rows, as build and append do.
        let _conversion = self
            .workspace_budget
            .as_ref()
            .map(|budget| budget.reserve(n * (stride + 4) + stride))
            .transpose()?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by_key(|&i| ids[i]);
        if order.windows(2).any(|w| ids[w[0]] == ids[w[1]]) {
            return Err(invalid("updated IDs must be distinct"));
        }
        // Stage rows in ascending ID order, so each shard's rows are one run
        // of the staging buffer, routed in a single scatter.
        let mut rank = vec![0; n];
        for (position, &i) in order.iter().enumerate() {
            rank[i] = position;
        }
        let mut vectors = vec![0; n * stride];
        let mut norms = vec![0; n * 4];
        let mut row_bytes = Vec::with_capacity(stride);
        let mut count = 0;
        for row in rows {
            let i = *rank
                .get(count)
                .ok_or_else(|| invalid("row iterator returned more rows than declared"))?;
            row_bytes.clear();
            let inverse = encode_row(&row, dimensions, padded, &mut row_bytes)?;
            vectors[i * stride..(i + 1) * stride].copy_from_slice(&row_bytes);
            norms[i * 4..i * 4 + 4].copy_from_slice(&inverse.to_le_bytes());
            count += 1;
        }
        if count != n {
            return Err(invalid("row iterator returned fewer rows than declared"));
        }
        let sorted: Vec<usize> = order.iter().map(|&i| ids[i] as usize).collect();
        // Nothing else can reach an exclusive handle's allocations: once this
        // holds, no other handle can come to share them while `self` is owned.
        let exclusive = Arc::get_mut(&mut self.inner).is_some_and(|s| s.exclusive());
        let shards = self.write(|w| {
            let staged_vectors = w.stage(&vectors)?;
            let staged_norms = w.stage(&norms)?;
            let staged = [&*staged_vectors, &*staged_norms];
            if exclusive {
                let mut at = 0;
                for shard in &self.inner.shards {
                    let end = shard.start + shard.count;
                    let mut routes = Vec::new();
                    while at < n && sorted[at] < end {
                        routes.push([at as u32, (sorted[at] - shard.start) as u32]);
                        at += 1;
                    }
                    if !routes.is_empty() {
                        w.scatter(
                            staged,
                            &routes,
                            shard.vectors(padded, 0, shard.capacity)?,
                            shard.inverse_norms(0, shard.capacity)?,
                            shard.capacity,
                        )?;
                    }
                }
                return Ok(None);
            }
            let last = self.inner.shards.len() - 1;
            let mut shards = Vec::with_capacity(self.inner.shards.len() + 2);
            let mut at = 0;
            for (index, shard) in self.inner.shards.iter().enumerate() {
                let end = shard.start + shard.count;
                let first = at;
                while at < n && sorted[at] < end {
                    at += 1;
                }
                if first == at {
                    shards.push(shard.clone());
                    continue;
                }
                // Allocation-relative bounds of this segment and its pieces.
                let (low, high) = (shard.offset, shard.offset + shard.count);
                let page_of = |id: usize| (id - shard.start + low) / PAGE_ROWS;
                let piece = |from: usize, to: usize| Shard {
                    storage: shard.storage.clone(),
                    offset: from,
                    start: shard.start + from - low,
                    count: to - from,
                    capacity: if to == high {
                        shard.offset + shard.capacity - from
                    } else {
                        to - from
                    },
                };
                let mut cursor = low;
                let mut i = first;
                while i < at {
                    // Coalesce touched pages separated by less than a tile.
                    let page = page_of(sorted[i]);
                    let mut last_page = page;
                    let mut j = i;
                    while j < at && page_of(sorted[j]) <= last_page + MERGE_PAGES {
                        last_page = page_of(sorted[j]);
                        j += 1;
                    }
                    let from = (page * PAGE_ROWS).max(low);
                    let to = ((last_page + 1) * PAGE_ROWS).min(high);
                    if from > cursor {
                        shards.push(piece(cursor, from));
                    }
                    let rows = to - from;
                    // The end of the snapshot keeps its append reserve.
                    let capacity = if index == last && to == high {
                        shard.offset + shard.capacity - from
                    } else {
                        shard_capacity(rows)
                    };
                    let storage = w.allocate(capacity, rows, shard.storage.appended)?;
                    let local = from - low;
                    w.copy(
                        storage.data.try_slice(0, rows * stride)?,
                        shard.vectors(padded, local, rows)?,
                    )?;
                    w.copy(
                        storage.norms.try_slice(0, rows * 4)?,
                        shard.inverse_norms(local, rows)?,
                    )?;
                    let start = shard.start + local;
                    let routes: Vec<_> = (i..j)
                        .map(|k| [k as u32, (sorted[k] - start) as u32])
                        .collect();
                    w.scatter(
                        staged,
                        &routes,
                        storage.data.binding(),
                        storage.norms.binding(),
                        capacity,
                    )?;
                    shards.push(Shard {
                        storage,
                        offset: 0,
                        start,
                        count: rows,
                        capacity,
                    });
                    cursor = to;
                    i = j;
                }
                if cursor < high {
                    shards.push(piece(cursor, high));
                }
            }
            Ok(Some(shards))
        })?;
        Ok(match shards {
            Some(shards) => self.derive(shards, self.len()),
            None => self,
        })
    }

    /// Return a snapshot with the build layout: the fewest shards, no rows
    /// superseded by updates, and no append reserve beyond 256-row tiles.
    ///
    /// Rows are copied on the device; nothing is read back or converted, so
    /// vectors, inverse norms and scores are bitwise unchanged. This snapshot
    /// keeps its allocations, so peak memory holds both until it is dropped.
    /// Use it after many scattered updates, or to release allocations that
    /// only superseded rows keep alive.
    pub fn compact(&self) -> Result<Self> {
        let padded = self.padded_dimensions();
        let count = self.len();
        let per_shard = shard_rows(padded, MAX_ELEMENTS);
        let shards = self.write(|w| {
            let mut shards = Vec::new();
            let mut source = self.inner.shards.iter().peekable();
            let mut consumed = 0; // Rows of the current source segment already copied.
            for start in (0..count).step_by(per_shard) {
                let rows = (count - start).min(per_shard);
                let capacity = shard_capacity(rows);
                let storage = w.allocate(capacity, rows, false)?;
                let mut filled = 0;
                while filled < rows {
                    let segment = source.peek().expect("segments cover the corpus");
                    let take = (segment.count - consumed).min(rows - filled);
                    w.copy(
                        storage
                            .data
                            .try_slice(filled * padded * 2, take * padded * 2)?,
                        segment.vectors(padded, consumed, take)?,
                    )?;
                    w.copy(
                        storage.norms.try_slice(filled * 4, take * 4)?,
                        segment.inverse_norms(consumed, take)?,
                    )?;
                    filled += take;
                    consumed += take;
                    if consumed == segment.count {
                        source.next();
                        consumed = 0;
                    }
                }
                shards.push(Shard {
                    storage,
                    offset: 0,
                    start,
                    count: rows,
                    capacity,
                });
            }
            Ok(shards)
        })?;
        Ok(self.derive(shards, count))
    }

    /// A snapshot of `shards` that shares this one's device, context and policy.
    fn derive(&self, shards: Vec<Shard>, count: usize) -> Self {
        let gather = self
            .inner
            .gather
            .lock()
            .map(|kernel| kernel.clone())
            .unwrap_or_default();
        Self {
            workspace_budget: self.workspace_budget.clone(),
            inner: Arc::new(CorpusStorage {
                context: self.inner.context.clone(),
                device: self.inner.device.clone(),
                device_id: self.inner.device_id,
                shards,
                dimensions: self.inner.dimensions,
                padded: self.inner.padded,
                count,
                gather: std::sync::Mutex::new(gather),
                writer: self.inner.writer.clone(),
            }),
        }
    }
}

impl CorpusStorage {
    /// Whether no other snapshot holds any of these allocations.
    pub(crate) fn exclusive(&self) -> bool {
        self.shards.iter().all(|shard| {
            let own = self
                .shards
                .iter()
                .filter(|s| Arc::ptr_eq(&s.storage, &shard.storage))
                .count();
            Arc::strong_count(&shard.storage) == own
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(i: usize, d: usize) -> Vec<f32> {
        (0..d)
            .map(|j| ((i * 37 + j * 13) % 103) as f32 - 49.25)
            .collect()
    }

    /// Application kernels over every shard view agree with host grouping.
    fn assert_albums(corpus: &Corpus, d: usize) -> Result<()> {
        let ordinals: Vec<u32> = (0..corpus.len()).map(|i| (i % 7) as u32).collect();
        let queries: Vec<f32> = (0..3).flat_map(|q| row(5_000 + q, d)).collect();
        let actual = crate::corpus::album_example::best_by_album(corpus, &ordinals, 7, &queries)?;
        let mut searcher = corpus.searcher()?;
        for (q, query) in queries.chunks_exact(d).enumerate() {
            let mut expected = [f32::NEG_INFINITY; 7];
            for (score, &album) in searcher.scores(query)?.iter().zip(&ordinals) {
                expected[album as usize] = expected[album as usize].max(*score);
            }
            for (&got, &want) in actual[q].iter().zip(&expected) {
                assert!((got - want).abs() < 3e-6, "query {q}: {got} vs {want}");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires gfx1151"]
    fn snapshots_keep_unaligned_shards_and_custom_kernels_consistent() -> Result<()> {
        let device = Device::open(0)?;
        let d = 129;
        let padded = 256;
        let mut expected: Vec<_> = (0..1_000).map(|i| row(i, d)).collect();
        // Shards of 300 rows: interior slack overlaps the next shard's rows.
        let build = |rows: &[Vec<f32>]| {
            Corpus::build_encoded(&device, d, rows, 300 * padded, |r, d, p, b| {
                encode(r, d, p, b)
            })
        };
        let base = build(&expected)?;
        assert_eq!(base.inner.shards.len(), 4);
        assert_albums(&base, d)?;
        let fresh: Vec<_> = (1_000..1_500).map(|i| row(i, d)).collect();
        let appended = base.append(&fresh)?;
        expected.extend(fresh);
        // The last shard's 156-row reserve is filled in place; the rest is a tail.
        let shards = &appended.inner.shards;
        assert!(Arc::ptr_eq(
            &shards[3].storage,
            &base.inner.shards[3].storage
        ));
        assert_eq!((shards[3].count, shards.len()), (256, 5));
        assert_albums(&appended, d)?;
        // The base's custom kernels still mask the slack an append wrote into.
        assert_albums(&base, d)?;

        let ids = [299u32, 5, 300, 1_200];
        let replacements: Vec<_> = ids.iter().map(|&i| row(i as usize + 9_000, d)).collect();
        let updated = appended.clone().update(&ids, &replacements)?;
        for (&id, replacement) in ids.iter().zip(&replacements) {
            expected[id as usize] = replacement.clone();
        }
        let reference = build(&expected)?;
        let (mut a, mut b) = (updated.searcher()?, reference.searcher()?);
        for q in 0..3 {
            let query = row(7_000 + q, d);
            assert_eq!(a.scores(&query)?, b.scores(&query)?);
        }
        assert_eq!(updated.inner.shards.len(), 5);
        assert!(!Arc::ptr_eq(
            &updated.inner.shards[0].storage,
            &shards[0].storage
        ));
        assert!(Arc::ptr_eq(
            &updated.inner.shards[2].storage,
            &shards[2].storage
        ));
        assert_albums(&updated, d)?;
        assert_albums(&appended, d)?;
        let compact = updated.compact()?;
        assert_eq!(compact.inner.shards.len(), 1);
        assert_albums(&compact, d)?;
        Ok(())
    }
}
