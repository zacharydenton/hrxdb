//! Owned native host-visible storage shared with the GPU.
use hrx::{Buffer, Result, Stream};
use std::ptr::NonNull;

pub(crate) struct HostBuffer {
    buffer: Option<Buffer>,
    pointer: NonNull<u8>,
    bytes: usize,
}

// SAFETY: the native Buffer owns the stable mapping. Moving this wrapper does
// not access it. Searcher synchronizes before host access and destruction;
// exclusive borrowing prevents concurrent host writes. This is not Sync.
unsafe impl Send for HostBuffer where Buffer: Send {}

impl HostBuffer {
    pub(crate) fn new(stream: &Stream, bytes: usize) -> Result<Self> {
        let bytes = bytes.max(1);
        let buffer = stream.allocate_zeroed(bytes)?;
        let pointer = NonNull::new(buffer.device_ptr()?.cast())
            .ok_or_else(|| crate::invalid("native buffer has no host mapping"))?;
        Ok(Self {
            buffer: Some(buffer),
            pointer,
            bytes,
        })
    }
    pub(crate) fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().unwrap()
    }
    /// Caller must establish completion of device work touching this allocation.
    pub(crate) unsafe fn bytes(&self) -> Result<&[u8]> {
        // SAFETY: the caller establishes device completion and excludes host writes.
        unsafe { self.buffer().cache_control(false, 0, self.bytes)? };
        // SAFETY: native backing remains owned; caller establishes completion.
        Ok(unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.bytes) })
    }
    /// Caller must establish completion of device work touching this allocation.
    /// Publish the modified range before submitting device work.
    pub(crate) unsafe fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: exclusive object borrow and caller's device completion proof.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.bytes) }
    }
    /// Caller must exclude device access until the host writes are published.
    pub(crate) unsafe fn publish(&self) -> Result<()> {
        // SAFETY: the caller excludes device use until this publication completes.
        unsafe { self.buffer().cache_control(true, 0, self.bytes) }
    }
    pub(crate) fn abandon(&mut self) {
        // Uncertain completion retains the entire native mapping/queue owner.
        if let Some(buffer) = self.buffer.take() {
            std::mem::forget(buffer);
        }
    }
}
