use std::fs::{File, OpenOptions};
use std::num::NonZeroUsize;
use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;
use std::ptr::NonNull;

use nix::sys::mman::{MapFlags, ProtFlags, mmap, munmap};

use crate::error::ShmError;

#[derive(Clone, Copy, Default)]
pub struct MapOptions {
    pub populate: bool,
    pub huge_pages: bool,
}

pub(crate) struct Mapping {
    ptr: NonNull<u8>,
    len: NonZeroUsize,
    file: File,
}

impl Mapping {
    pub(crate) fn create(
        path: &Path,
        len: NonZeroUsize,
        opts: MapOptions,
    ) -> Result<Self, ShmError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.set_len(len.get() as u64)?;
        Self::map(file, len, opts)
    }

    pub(crate) fn open(path: &Path, opts: MapOptions) -> Result<Self, ShmError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len =
            NonZeroUsize::new(file.metadata()?.len() as usize).ok_or(ShmError::EmptySegment)?;
        Self::map(file, len, opts)
    }

    fn map(file: File, len: NonZeroUsize, opts: MapOptions) -> Result<Self, ShmError> {
        let mut flags = MapFlags::MAP_SHARED;
        if opts.populate {
            flags |= MapFlags::MAP_POPULATE;
        }
        if opts.huge_pages {
            flags |= MapFlags::MAP_HUGETLB;
        }

        let ptr = unsafe {
            mmap(
                None,
                len,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                flags,
                file.as_fd(),
                0,
            )
        }
        .map_err(|e| {
            let io = std::io::Error::from_raw_os_error(e as i32);
            if opts.huge_pages {
                ShmError::HugePagesUnavailable(io)
            } else {
                ShmError::Os(io)
            }
        })?;

        Ok(Self {
            ptr: ptr.cast(),
            len,
            file,
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.ptr.cast(), self.len.get());
        }
    }
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}
