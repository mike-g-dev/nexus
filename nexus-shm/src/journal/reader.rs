use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::sync::atomic::Ordering;

use crate::error::ShmError;
use crate::segment::Segment;

use super::error::JournalError;
use super::frame::{FRAME_HEADER, TYPE_PAD, align_up, commit_len, footprint, read_kind};
use super::header::{RecordHeader, SeqHeader};

/// Read half of a journal: walks committed records across segments in order.
pub struct Reader<H: RecordHeader> {
    pub(super) base: std::path::PathBuf,
    pub(super) segment_size: usize,
    pub(super) map: crate::region::MapOptions,
    pub(super) segments: Vec<Segment>,
    pub(super) seg_idx: usize,
    pub(super) cursor: usize,
    pub(super) _marker: PhantomData<H>,
}

impl<H: RecordHeader> Reader<H> {
    /// Yield the next committed record, or `Ok(None)` once caught up to the
    /// write tail. Real I/O failures while opening a rolled segment surface as
    /// `Err` rather than being mistaken for end-of-log.
    pub fn next_record(&mut self) -> Result<Option<ReadRecord<'_, H>>, JournalError> {
        loop {
            if self.cursor + FRAME_HEADER > self.segment_size {
                if self.advance_segment()? {
                    continue;
                }
                return Ok(None);
            }
            let data = self.segments[self.seg_idx].data();
            // SAFETY: cursor is an 8-aligned offset within the mapped data.
            let cl = unsafe { commit_len(data.add(self.cursor)) }.load(Ordering::Acquire);
            if cl == 0 {
                return Ok(None);
            }
            // SAFETY: cl > 0 was Acquire-loaded, so the frame header is published.
            if unsafe { read_kind(data.add(self.cursor)) } == TYPE_PAD {
                self.cursor += align_up(cl as usize);
                if self.cursor + FRAME_HEADER > self.segment_size && !self.advance_segment()? {
                    return Ok(None);
                }
                continue;
            }
            let body = cl as usize;
            let hsize = size_of::<H>();
            if body < hsize {
                return Ok(None);
            }
            let off = self.cursor;
            // SAFETY: the committed frame holds `H` at `off + FRAME_HEADER`;
            // `H: Pod`, so an unaligned read is valid.
            let header =
                unsafe { std::ptr::read_unaligned(data.add(off + FRAME_HEADER).cast::<H>()) };
            // SAFETY: the payload lies within the committed frame and the mapping
            // outlives the borrow held through `&mut self`.
            let payload = unsafe {
                std::slice::from_raw_parts(data.add(off + FRAME_HEADER + hsize), body - hsize)
            };
            self.cursor = off + footprint(body);
            return Ok(Some(ReadRecord { header, payload }));
        }
    }

    fn advance_segment(&mut self) -> Result<bool, JournalError> {
        if self.seg_idx + 1 >= self.segments.len() && !self.load_next()? {
            return Ok(false);
        }
        self.seg_idx += 1;
        self.cursor = 0;
        Ok(true)
    }

    fn load_next(&mut self) -> Result<bool, JournalError> {
        let next = self.segments.len() as u64;
        let path = super::segment_path(&self.base, next);
        match Segment::attach(&path, self.map) {
            Ok(seg) => {
                self.segments.push(seg);
                Ok(true)
            }
            Err(ShmError::Os(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Iterate committed records whose header sequence falls in `range`.
    pub fn read_range<R>(&mut self, range: R) -> Result<ReadRange<'_, H>, JournalError>
    where
        H: SeqHeader,
        R: RangeBounds<u64>,
    {
        while self.load_next()? {}
        let lo = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n.saturating_add(1),
            std::ops::Bound::Unbounded => 0,
        };
        let hi = match range.end_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n.saturating_sub(1),
            std::ops::Bound::Unbounded => u64::MAX,
        };
        Ok(ReadRange {
            segments: &self.segments,
            segment_size: self.segment_size,
            seg_idx: 0,
            cursor: 0,
            lo,
            hi,
            _marker: PhantomData,
        })
    }
}

/// A committed record: a copy of the header and a zero-copy view of the payload.
pub struct ReadRecord<'a, H: RecordHeader> {
    header: H,
    payload: &'a [u8],
}

impl<H: RecordHeader> ReadRecord<'_, H> {
    pub fn header(&self) -> H {
        self.header
    }

    pub fn payload(&self) -> &[u8] {
        self.payload
    }
}

/// Borrowing iterator over a sequence range, returned by [`Reader::read_range`].
pub struct ReadRange<'a, H: SeqHeader> {
    segments: &'a [Segment],
    segment_size: usize,
    seg_idx: usize,
    cursor: usize,
    lo: u64,
    hi: u64,
    _marker: PhantomData<H>,
}

impl<'a, H: SeqHeader> Iterator for ReadRange<'a, H> {
    type Item = ReadRecord<'a, H>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.seg_idx >= self.segments.len() {
                return None;
            }
            if self.cursor + FRAME_HEADER > self.segment_size {
                self.seg_idx += 1;
                self.cursor = 0;
                continue;
            }
            let data = self.segments[self.seg_idx].data();
            // SAFETY: cursor is an 8-aligned offset within the mapped data.
            let cl = unsafe { commit_len(data.add(self.cursor)) }.load(Ordering::Acquire);
            if cl == 0 {
                return None;
            }
            // SAFETY: cl > 0 was Acquire-loaded, so the frame header is published.
            if unsafe { read_kind(data.add(self.cursor)) } == TYPE_PAD {
                self.cursor += align_up(cl as usize);
                continue;
            }
            let body = cl as usize;
            let hsize = size_of::<H>();
            if body < hsize {
                return None;
            }
            let off = self.cursor;
            self.cursor = off + footprint(body);
            // SAFETY: the committed frame holds `H` at `off + FRAME_HEADER`; `H: Pod`.
            let header =
                unsafe { std::ptr::read_unaligned(data.add(off + FRAME_HEADER).cast::<H>()) };
            if header.seq() < self.lo || header.seq() > self.hi {
                continue;
            }
            // SAFETY: the payload lies within the committed frame and `segments`
            // outlives `'a`.
            let payload = unsafe {
                std::slice::from_raw_parts(data.add(off + FRAME_HEADER + hsize), body - hsize)
            };
            return Some(ReadRecord { header, payload });
        }
    }
}
