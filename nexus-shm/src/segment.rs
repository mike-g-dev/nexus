use std::path::Path;

use crate::control::{ControlBlock, status};
use crate::error::ShmError;
use crate::lock::{self, Liveness};
use crate::region::{MapOptions, Mapping};

const HEADER: usize = size_of::<ControlBlock>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerStatus {
    Uninit,
    Alive,
    Dead,
}

pub struct Segment {
    mapping: Mapping,
    creator: bool,
}

impl Segment {
    pub fn create(path: &Path, data_len: usize, opts: MapOptions) -> Result<Self, ShmError> {
        let total = (HEADER + data_len)
            .try_into()
            .map_err(|_| ShmError::EmptySegment)?;
        let mapping = Mapping::create(path, total, opts)?;

        if !lock::acquire_owner(mapping.as_fd())? {
            return Err(ShmError::OwnerActive);
        }

        let cb = unsafe { &mut *mapping.as_ptr().cast::<ControlBlock>() };
        let generation = cb.generation().wrapping_add(1);
        cb.write_header(flags(opts), generation, std::process::id(), data_len as u64);

        Ok(Self {
            mapping,
            creator: true,
        })
    }

    pub fn attach(path: &Path, opts: MapOptions) -> Result<Self, ShmError> {
        let mapping = Mapping::open(path, opts)?;
        Self::control_of(&mapping).validate()?;
        Ok(Self {
            mapping,
            creator: false,
        })
    }

    pub fn peer_status(&self) -> PeerStatus {
        match self.control().status() {
            s if s == status::ALIVE => PeerStatus::Alive,
            s if s == status::DEAD => PeerStatus::Dead,
            _ => PeerStatus::Uninit,
        }
    }

    pub fn peer_liveness(&self) -> Liveness {
        lock::owner_liveness(self.mapping.as_fd())
    }

    pub fn data(&self) -> *mut u8 {
        unsafe { self.mapping.as_ptr().add(HEADER) }
    }

    pub fn data_len(&self) -> usize {
        self.control().data_len() as usize
    }

    fn control(&self) -> &ControlBlock {
        Self::control_of(&self.mapping)
    }

    fn control_of(mapping: &Mapping) -> &ControlBlock {
        unsafe { &*mapping.as_ptr().cast::<ControlBlock>() }
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        if self.creator {
            self.control().mark_dead();
        }
    }
}

fn flags(opts: MapOptions) -> u16 {
    u16::from(opts.populate) | (u16::from(opts.huge_pages) << 1)
}

#[cfg(test)]
mod tests {
    use super::{PeerStatus, Segment};
    use crate::lock::Liveness;
    use crate::region::MapOptions;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nexus-shm-{}-{}", std::process::id(), name))
    }

    #[test]
    fn create_attach_roundtrip() {
        let path = temp_path("roundtrip");
        let _ = std::fs::remove_file(&path);

        let seg = Segment::create(&path, 4096, MapOptions::default()).unwrap();
        assert_eq!(seg.data_len(), 4096);
        assert_eq!(seg.peer_status(), PeerStatus::Alive);

        unsafe { seg.data().write(0xAB) };

        let peer = Segment::attach(&path, MapOptions::default()).unwrap();
        assert_eq!(peer.data_len(), 4096);
        assert_eq!(peer.peer_status(), PeerStatus::Alive);
        assert_eq!(unsafe { peer.data().read() }, 0xAB);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn creator_drop_marks_dead() {
        let path = temp_path("dead");
        let _ = std::fs::remove_file(&path);

        let seg = Segment::create(&path, 64, MapOptions::default()).unwrap();
        drop(seg);

        let peer = Segment::attach(&path, MapOptions::default()).unwrap();
        assert_eq!(peer.peer_status(), PeerStatus::Dead);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn kernel_liveness_tracks_owner() {
        let path = temp_path("liveness");
        let _ = std::fs::remove_file(&path);

        let owner = Segment::create(&path, 64, MapOptions::default()).unwrap();
        let peer = Segment::attach(&path, MapOptions::default()).unwrap();
        assert_eq!(peer.peer_liveness(), Liveness::Alive);

        drop(owner);
        assert_eq!(peer.peer_liveness(), Liveness::Dead);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_foreign_file() {
        let path = temp_path("foreign");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();

        assert!(Segment::attach(&path, MapOptions::default()).is_err());

        std::fs::remove_file(&path).unwrap();
    }
}
