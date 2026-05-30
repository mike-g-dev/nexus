use std::os::fd::BorrowedFd;

use nix::errno::Errno;
use nix::fcntl::{FcntlArg, fcntl};
use nix::libc;

use crate::error::ShmError;

const OWNER_OFFSET: i64 = 0;
const OWNER_LEN: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    Alive,
    Dead,
    Unknown,
}

fn owner_flock(l_type: libc::c_short) -> libc::flock {
    let mut lk: libc::flock = unsafe { std::mem::zeroed() };
    lk.l_type = l_type;
    lk.l_whence = libc::SEEK_SET as libc::c_short;
    lk.l_start = OWNER_OFFSET;
    lk.l_len = OWNER_LEN;
    lk
}

pub(crate) fn acquire_owner(fd: BorrowedFd<'_>) -> Result<bool, ShmError> {
    let lk = owner_flock(libc::F_WRLCK as libc::c_short);
    match fcntl(fd, FcntlArg::F_OFD_SETLK(&lk)) {
        Ok(_) => Ok(true),
        Err(Errno::EACCES | Errno::EAGAIN) => Ok(false),
        Err(e) => Err(ShmError::Os(std::io::Error::from_raw_os_error(e as i32))),
    }
}

pub(crate) fn owner_liveness(fd: BorrowedFd<'_>) -> Liveness {
    let mut lk = owner_flock(libc::F_WRLCK as libc::c_short);
    match fcntl(fd, FcntlArg::F_OFD_GETLK(&mut lk)) {
        Ok(_) if lk.l_type == libc::F_UNLCK as libc::c_short => Liveness::Dead,
        Ok(_) => Liveness::Alive,
        Err(_) => Liveness::Unknown,
    }
}
