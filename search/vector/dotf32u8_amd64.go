//go:build amd64

package vector

// dotF32U8AVX2 sums q[i]*float32(b[i]) over the first n elements using AVX2
// (zero-extend byte -> int32 -> float32, multiply, accumulate). n must be a
// positive multiple of 8. Implemented in dotf32u8_amd64.s.
//
//go:noescape
func dotF32U8AVX2(q *float32, b *byte, n int) float32

// dotF32U8 is the float32-times-unsigned-byte dot used by scalar8Scorer: AVX2 for
// the 8-wide body, portable loop for the tail and non-AVX2 targets.
func dotF32U8(q []float32, b []byte) float32 {
	n := len(q)
	if useAVX2 && n >= 8 {
		n8 := n &^ 7
		s := dotF32U8AVX2(&q[0], &b[0], n8)
		for i := n8; i < n; i++ {
			s += q[i] * float32(b[i])
		}
		return s
	}
	return dotF32U8Generic(q, b)
}
