//go:build amd64

package vector

// dotF32U8AVX2 (VMULPS+VADDPS) and dotF32U8FMA (VFMADD231PS, 4 accumulators) sum
// q[i]*float32(b[i]) over the first n elements (zero-extend byte -> int32 ->
// float32). n must be a positive multiple of 8. Implemented in dotf32u8_amd64.s.
//
//go:noescape
func dotF32U8AVX2(q *float32, b *byte, n int) float32

//go:noescape
func dotF32U8FMA(q *float32, b *byte, n int) float32

// dotF32U8 is the float32-times-unsigned-byte dot used by scalar8Scorer: the
// fastest available 8-wide kernel for the body (FMA, else AVX2), portable loop
// for the tail and non-AVX2 targets.
func dotF32U8(q []float32, b []byte) float32 {
	n := len(q)
	if n >= 8 {
		n8 := n &^ 7
		var s float32
		switch {
		case useFMA:
			s = dotF32U8FMA(&q[0], &b[0], n8)
		case useAVX2:
			s = dotF32U8AVX2(&q[0], &b[0], n8)
		default:
			return dotF32U8Generic(q, b)
		}
		for i := n8; i < n; i++ {
			s += q[i] * float32(b[i])
		}
		return s
	}
	return dotF32U8Generic(q, b)
}
