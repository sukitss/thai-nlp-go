//go:build amd64

package vector

// dotFloat32AVX2 (VMULPS+VADDPS) and dotFloat32FMA (VFMADD231PS) both sum the
// pairwise products of the first n float32s of a and b, eight lanes at a time. n
// must be a positive multiple of 8. Implemented in dotf32_amd64.s.
//
//go:noescape
func dotFloat32AVX2(a, b *float32, n int) float32

//go:noescape
func dotFloat32FMA(a, b *float32, n int) float32

// dot uses the fastest available 8-wide kernel for the body (FMA, else AVX2),
// finishing the remainder (and the whole thing without AVX2) with the portable
// loop. useAVX2/useFMA are shared with the int8 path (dotint8_amd64.go).
func dot(a, b []float32) float32 {
	n := len(a)
	if n >= 8 {
		n8 := n &^ 7
		var s float32
		switch {
		case useFMA:
			s = dotFloat32FMA(&a[0], &b[0], n8)
		case useAVX2:
			s = dotFloat32AVX2(&a[0], &b[0], n8)
		default:
			return dotFloat32Generic(a, b)
		}
		for i := n8; i < n; i++ {
			s += a[i] * b[i]
		}
		return s
	}
	return dotFloat32Generic(a, b)
}
