//! Persistent device exclusion state for iterative retrieval.
use crate::*;
use hrx::View;
/// Reusable device bitmap over insertion IDs. Updates are incremental and may
/// consume top-k IDs directly, keeping a similarity walk entirely on the GPU.
/// Use one ordered stream, or establish event dependencies between streams.
pub struct DeviceExclusions {
    bitmap: Buffer,
    rows: usize,
    device: usize,
    update: Kernel,
}
impl DeviceExclusions {
    /// Allocate an initially empty set over 0..=2^30 rows.
    pub fn new(stream: &Stream, rows: usize) -> Result<Self> {
        if rows > MAX_ROWS {
            return Err(invalid("exclusion set exceeds 2^30 rows"));
        }
        let compiler = hrx::loom::Compiler::with_options(
            None,
            hrx::loom::CompilerOptions {
                target: stream.target().clone(),
                ..Default::default()
            },
        )?;
        let (update, _) = compile(
            &compiler,
            stream,
            include_str!("../kernels/exclusions.loom"),
            kernels::named_spec("update_exclusions"),
        )?;
        let bitmap = stream.allocate(rows.div_ceil(32) * 4)?;
        stream.fill(bitmap.binding(), 0)?;
        Ok(Self {
            bitmap,
            rows,
            device: stream.device_id(),
            update,
        })
    }
    /// Borrow the u32 bitmap (least-significant bit first) for device search.
    pub fn binding(&self) -> View<'_> {
        self.bitmap.binding()
    }
    /// Number of insertion IDs represented.
    pub fn len(&self) -> usize {
        self.rows
    }
    /// Whether the set's ID domain is empty; does not count current exclusions.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
    /// Allocated bitmap bytes.
    pub fn memory_usage(&self) -> usize {
        self.bitmap.bytes()
    }
    fn check(&self, stream: &Stream) -> Result<()> {
        if stream.device_id() != self.device {
            return Err(invalid("exclusion set belongs to another device"));
        }
        Ok(())
    }
    /// Queue removal of all exclusions.
    pub fn clear(&mut self, stream: &Stream) -> Result<()> {
        self.check(stream)?;
        stream.fill(self.bitmap.binding(), 0)
    }
    /// Add host IDs, validating every ID before changing the set. Only these IDs
    /// are uploaded; duplicate updates are harmless.
    pub fn insert(&mut self, stream: &mut Stream, ids: &[u32]) -> Result<()> {
        self.update_host(stream, ids, false)
    }
    /// Remove host IDs, preserving all other bits.
    pub fn remove(&mut self, stream: &mut Stream, ids: &[u32]) -> Result<()> {
        self.update_host(stream, ids, true)
    }
    fn update_host(&mut self, stream: &mut Stream, ids: &[u32], remove: bool) -> Result<()> {
        self.check(stream)?;
        if ids.len() > MAX_ROWS || ids.iter().any(|&id| id as usize >= self.rows) {
            return Err(invalid("excluded ID is outside the corpus"));
        }
        if ids.is_empty() {
            return Ok(());
        }
        let buffer = stream.allocate(ids.len() * 4)?;
        stream.upload(
            buffer.binding(),
            &ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.update_device(stream, buffer.binding(), ids.len(), remove)
    }
    /// Add device-generated IDs without host readback. Values outside the ID
    /// domain, including the u32::MAX result sentinel, are ignored. Input has
    /// at least count contiguous u32 entries and must be ready on this stream.
    pub fn insert_device(&mut self, stream: &Stream, ids: View<'_>, count: usize) -> Result<()> {
        self.update_device(stream, ids, count, false)
    }
    /// Remove device-generated IDs; out-of-range IDs are ignored.
    pub fn remove_device(&mut self, stream: &Stream, ids: View<'_>, count: usize) -> Result<()> {
        self.update_device(stream, ids, count, true)
    }
    fn update_device(
        &mut self,
        stream: &Stream,
        ids: View<'_>,
        count: usize,
        remove: bool,
    ) -> Result<()> {
        self.check(stream)?;
        if count > MAX_ROWS {
            return Err(invalid("too many exclusion updates"));
        }
        let ids = ids.slice(0, count * 4)?;
        if count == 0 || self.rows == 0 {
            return Ok(());
        }
        let mut c = Constants::new();
        c.push(count as u32)?;
        c.push(self.rows as u32)?;
        c.push(u32::from(remove))?;
        // SAFETY: input and bitmap spans are checked; the kernel guards every
        // ID before bit addressing and atomically combines duplicate updates.
        unsafe {
            stream.dispatch(
                &self.update,
                [count.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                &c,
                &[ids, self.bitmap.binding()],
            )
        }
    }
}
