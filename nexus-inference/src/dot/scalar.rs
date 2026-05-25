#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let n4 = a.len() & !3;
    let (a_bulk, a_tail) = a.split_at(n4);
    let (b_bulk, b_tail) = b.split_at(n4);
    for (ac, bc) in a_bulk.chunks_exact(4).zip(b_bulk.chunks_exact(4)) {
        s0 += ac[0] * bc[0];
        s1 += ac[1] * bc[1];
        s2 += ac[2] * bc[2];
        s3 += ac[3] * bc[3];
    }
    for (&a_val, &b_val) in a_tail.iter().zip(b_tail) {
        s0 += a_val * b_val;
    }
    (s0 + s2) + (s1 + s3)
}

#[inline]
pub fn dot4_f32(rows: &[f32], input: &[f32]) -> [f32; 4] {
    let n = input.len();
    [
        dot_f32(&rows[..n], input),
        dot_f32(&rows[n..2 * n], input),
        dot_f32(&rows[2 * n..3 * n], input),
        dot_f32(&rows[3 * n..4 * n], input),
    ]
}
