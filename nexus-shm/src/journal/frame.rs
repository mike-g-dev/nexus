use std::sync::atomic::AtomicU32;

pub(crate) const FRAME_HEADER: usize = 8;
pub(crate) const ALIGN: usize = 8;

pub(crate) const TYPE_DATA: u16 = 0;
pub(crate) const TYPE_PAD: u16 = 1;

pub(crate) const fn align_up(n: usize) -> usize {
    (n + ALIGN - 1) & !(ALIGN - 1)
}

pub(crate) const fn footprint(body: usize) -> usize {
    FRAME_HEADER + align_up(body)
}

pub(crate) unsafe fn commit_len<'a>(ptr: *mut u8) -> &'a AtomicU32 {
    // SAFETY: caller guarantees ptr is 8-byte-aligned, initialized, and live for
    // 'a; AtomicU32 shares u32's layout.
    unsafe { AtomicU32::from_ptr(ptr.cast()) }
}

pub(crate) unsafe fn write_kind(ptr: *mut u8, frame_type: u16) {
    // SAFETY: caller guarantees the 8-byte frame header at ptr is within a live
    // mapping reserved for this record.
    unsafe {
        std::ptr::write_unaligned(ptr.add(4).cast::<u16>(), frame_type);
        std::ptr::write_unaligned(ptr.add(6).cast::<u16>(), 0);
    }
}

pub(crate) unsafe fn read_kind(ptr: *mut u8) -> u16 {
    // SAFETY: caller guarantees the frame header is published (read after an
    // Acquire load of commit_len) and within a live mapping.
    unsafe { std::ptr::read_unaligned(ptr.add(4).cast::<u16>()) }
}
