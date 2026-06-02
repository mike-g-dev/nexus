/// Apply or remove XOR mask in-place.
///
/// The mask is a 4-byte key applied cyclically. This function is
/// symmetric — applying it twice with the same key restores the original.
///
/// Dispatches to SIMD when available (SSE2/AVX2 on x86_64).
#[inline]
pub fn apply_mask(buf: &mut [u8], mask: [u8; 4]) {
    if mask == [0; 4] {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Compile-time AVX2 detection (when built with -C target-cpu=native
        // or -C target-feature=+avx2). Zero runtime cost.
        #[cfg(target_feature = "avx2")]
        {
            // SAFETY: AVX2 confirmed at compile time.
            unsafe { apply_mask_avx2(buf, mask) };
            return;
        }
        // Runtime AVX2 detection (cached global atomic, ~3 cycles).
        #[cfg(not(target_feature = "avx2"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: checked avx2 support
                unsafe { apply_mask_avx2(buf, mask) };
                return;
            }
        }
        // SSE2 is baseline on x86_64, always available
        // SAFETY: SSE2 is guaranteed on x86_64
        unsafe { apply_mask_sse2(buf, mask) };
        return;
    }

    #[allow(unreachable_code)]
    apply_mask_scalar(buf, mask);
}

/// Scalar fallback — processes 8 bytes at a time via u64 XOR,
/// then handles the tail byte-by-byte.
fn apply_mask_scalar(buf: &mut [u8], mask: [u8; 4]) {
    let mask_u32 = u32::from_ne_bytes(mask);
    let mask_u64 = u64::from(mask_u32) | (u64::from(mask_u32) << 32);

    // SAFETY: u64 has no invalid bit patterns, and align_to_mut splits the
    // buffer at naturally aligned boundaries. The prefix/suffix handle any
    // unaligned bytes at the edges.
    let (prefix, middle, suffix) = unsafe { buf.align_to_mut::<u64>() };

    // Handle unaligned prefix
    for (i, byte) in prefix.iter_mut().enumerate() {
        *byte ^= mask[i & 3];
    }

    // Bulk XOR 8 bytes at a time
    // Rotate mask to align with prefix offset (no allocation)
    let offset = prefix.len() & 3;
    let aligned_mask = if offset == 0 {
        mask_u64
    } else {
        let rotated: [u8; 8] = [
            mask[offset & 3],
            mask[(offset + 1) & 3],
            mask[(offset + 2) & 3],
            mask[(offset + 3) & 3],
            mask[offset & 3],
            mask[(offset + 1) & 3],
            mask[(offset + 2) & 3],
            mask[(offset + 3) & 3],
        ];
        u64::from_ne_bytes(rotated)
    };
    for word in middle.iter_mut() {
        *word ^= aligned_mask;
    }

    // Handle unaligned suffix
    let suffix_offset = (prefix.len() + middle.len() * 8) & 3;
    for (i, byte) in suffix.iter_mut().enumerate() {
        *byte ^= mask[(suffix_offset + i) & 3];
    }
}

/// SSE2 WebSocket mask XOR — processes 16 bytes per iteration.
///
/// # Safety
///
/// Caller must ensure the CPU supports SSE2 (baseline on x86_64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn apply_mask_sse2(buf: &mut [u8], mask: [u8; 4]) {
    use std::arch::x86_64::*;

    let len = buf.len();
    if len < 16 {
        apply_mask_scalar(buf, mask);
        return;
    }

    let mask_u32 = u32::from_ne_bytes(mask);
    let mask_vec = _mm_set1_epi32(mask_u32 as i32);

    let ptr = buf.as_mut_ptr();
    let mut i = 0usize;

    while i + 16 <= len {
        // SAFETY: i + 16 <= len, ptr is valid for buf's length
        unsafe {
            let data = _mm_loadu_si128(ptr.add(i) as *const __m128i);
            let masked = _mm_xor_si128(data, mask_vec);
            _mm_storeu_si128(ptr.add(i) as *mut __m128i, masked);
        }
        i += 16;
    }

    while i < len {
        // SAFETY: i < len
        unsafe { *buf.get_unchecked_mut(i) ^= mask[i & 3] };
        i += 1;
    }
}

/// AVX2 WebSocket mask XOR — processes 32 bytes per iteration.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_mask_avx2(buf: &mut [u8], mask: [u8; 4]) {
    use std::arch::x86_64::*;

    let len = buf.len();
    if len < 32 {
        // SAFETY: SSE2 is baseline on x86_64
        unsafe { apply_mask_sse2(buf, mask) };
        return;
    }

    let mask_u32 = u32::from_ne_bytes(mask);
    let mask_vec = _mm256_set1_epi32(mask_u32 as i32);

    let ptr = buf.as_mut_ptr();
    let mut i = 0usize;

    while i + 32 <= len {
        // SAFETY: i + 32 <= len
        unsafe {
            let data = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let masked = _mm256_xor_si256(data, mask_vec);
            _mm256_storeu_si256(ptr.add(i) as *mut __m256i, masked);
        }
        i += 32;
    }

    if i + 16 <= len {
        // SAFETY: i + 16 <= len, SSE2 available
        unsafe {
            let mask_128 = _mm_set1_epi32(mask_u32 as i32);
            let data = _mm_loadu_si128(ptr.add(i) as *const __m128i);
            let masked = _mm_xor_si128(data, mask_128);
            _mm_storeu_si128(ptr.add(i) as *mut __m128i, masked);
        }
        i += 16;
    }

    while i < len {
        // SAFETY: i < len
        unsafe { *buf.get_unchecked_mut(i) ^= mask[i & 3] };
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let original = b"Hello, WebSocket!".to_vec();
        let mask = [0x37, 0xFA, 0x21, 0x3D];

        let mut buf = original.clone();
        apply_mask(&mut buf, mask);
        assert_ne!(&buf, &original);
        apply_mask(&mut buf, mask);
        assert_eq!(&buf, &original);
    }

    #[test]
    fn known_answer() {
        // RFC 6455 doesn't specify test vectors, but we can verify manually:
        // payload: [0x48, 0x65, 0x6C, 0x6C, 0x6F] = "Hello"
        // mask:    [0x37, 0xFA, 0x21, 0x3D]
        // XOR:     [0x7F, 0x9F, 0x4D, 0x51, 0x58]
        let mask = [0x37, 0xFA, 0x21, 0x3D];
        let mut buf = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F];
        apply_mask(&mut buf, mask);
        assert_eq!(buf, vec![0x7F, 0x9F, 0x4D, 0x51, 0x58]);
    }

    #[test]
    fn empty_payload() {
        let mask = [0x37, 0xFA, 0x21, 0x3D];
        let mut buf = vec![];
        apply_mask(&mut buf, mask);
        assert!(buf.is_empty());
    }

    #[test]
    fn one_byte() {
        let mask = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut buf = vec![0x55];
        apply_mask(&mut buf, mask);
        assert_eq!(buf, vec![0x55 ^ 0xAA]);
    }

    #[test]
    fn two_bytes() {
        let mask = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut buf = vec![0x11, 0x22];
        apply_mask(&mut buf, mask);
        assert_eq!(buf, vec![0x11 ^ 0xAA, 0x22 ^ 0xBB]);
    }

    #[test]
    fn three_bytes() {
        let mask = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut buf = vec![0x11, 0x22, 0x33];
        apply_mask(&mut buf, mask);
        assert_eq!(buf, vec![0x11 ^ 0xAA, 0x22 ^ 0xBB, 0x33 ^ 0xCC]);
    }

    #[test]
    fn exactly_four_bytes() {
        let mask = [0xAA, 0xBB, 0xCC, 0xDD];
        let original = vec![0x11, 0x22, 0x33, 0x44];
        let mut buf = original.clone();
        apply_mask(&mut buf, mask);
        assert_eq!(
            buf,
            vec![0x11 ^ 0xAA, 0x22 ^ 0xBB, 0x33 ^ 0xCC, 0x44 ^ 0xDD]
        );
        apply_mask(&mut buf, mask);
        assert_eq!(buf, original);
    }

    #[test]
    fn large_payload_round_trip() {
        let mask = [0xDE, 0xAD, 0xBE, 0xEF];
        let original: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
        let mut buf = original.clone();
        apply_mask(&mut buf, mask);
        assert_ne!(&buf, &original);
        apply_mask(&mut buf, mask);
        assert_eq!(&buf, &original);
    }

    #[test]
    fn zero_mask_is_noop() {
        let original = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F];
        let mut buf = original.clone();
        apply_mask(&mut buf, [0, 0, 0, 0]);
        assert_eq!(buf, original);
    }

    #[test]
    fn simd_matches_scalar() {
        let mask = [0x12, 0x34, 0x56, 0x78];
        let original: Vec<u8> = (0..257).map(|i| (i & 0xFF) as u8).collect();

        let mut scalar = original.clone();
        apply_mask_scalar(&mut scalar, mask);

        let mut dispatch = original;
        apply_mask(&mut dispatch, mask);

        assert_eq!(scalar, dispatch, "SIMD path must match scalar");
    }
}
