//! Owned, page-aligned host storage imported once into HRX.
use hrx::{Buffer, Result, Stream};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ptr::NonNull,
};

pub(crate) struct HostBuffer {
    buffer: Option<Buffer>,
    pointer: NonNull<u8>,
    layout: Layout,
    leaked: bool,
}

// SAFETY: this wrapper uniquely owns its allocation and driver registration.
// Moving it transfers ownership without moving the backing pages or accessing
// them. Buffer itself is Send. Host accesses require exclusive Searcher access
// and completion of its stream; Searcher also synchronizes before destruction.
// This does not implement Sync: sharing host access would need a separate proof.
unsafe impl Send for HostBuffer where Buffer: Send {}

impl HostBuffer {
    pub(crate) fn new(stream: &Stream, bytes: usize) -> Result<Self> {
        // SAFETY: sysconf has no pointer arguments or side effects.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return Err(crate::invalid("cannot determine host page size"));
        }
        let page = page as usize;
        let size = bytes.max(1).div_ceil(page) * page;
        let layout = Layout::from_size_align(size, page)
            .map_err(|_| crate::invalid("invalid host allocation layout"))?;
        // SAFETY: layout is nonzero and valid; this object owns the allocation.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| crate::invalid("host allocation failed"))?;
        // SAFETY: page-aligned owned memory is held until the imported Buffer is
        // destroyed. Searcher synchronizes before every host access and on drop.
        let imported = unsafe { stream.import_host(pointer.as_ptr().cast(), size) };
        match imported {
            Ok(buffer) => Ok(Self {
                buffer: Some(buffer),
                pointer,
                layout,
                leaked: false,
            }),
            Err(error) => {
                // SAFETY: import failed, so the driver retains no allocation.
                unsafe {
                    dealloc(pointer.as_ptr(), layout);
                }
                Err(error)
            }
        }
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        self.buffer.as_ref().unwrap()
    }

    /// Caller must establish completion of device work touching this allocation.
    pub(crate) unsafe fn bytes(&self) -> &[u8] {
        // SAFETY: guaranteed by this method's caller and the allocation owner.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
    }

    /// Caller must establish completion of device work touching this allocation.
    pub(crate) unsafe fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: guaranteed by this method's caller and exclusive object borrow.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }

    pub(crate) fn abandon(&mut self) {
        // Completion failed: retain both the driver's registration and its backing
        // memory rather than freeing pages that the device may still access.
        if let Some(buffer) = self.buffer.take() {
            std::mem::forget(buffer);
        }
        self.leaked = true;
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        drop(self.buffer.take());
        if !self.leaked {
            // SAFETY: owner has synchronized; registration was released above.
            unsafe {
                dealloc(self.pointer.as_ptr(), self.layout);
            }
        }
    }
}
