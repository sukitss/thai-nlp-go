//go:build amd64

package vector

// dotFloat32AVX2 sums the pairwise products of the first n float32s of a and b
// using AVX2 (eight lanes). n must be a positive multiple of 8. Implemented in
// dotf32_amd64.s.
//
//go:noescape
func dotFloat32AVX2(a, b *float32, n int) float32

// dot uses the AVX2 kernel for the 8-wide body when available, finishing the
// remainder (and the whole thing without AVX2) with the portable loop. useAVX2
// is shared with the int8 path (dotint8_amd64.go).
func dot(a, b []float32) float32 {
	n := len(a)
	if useAVX2 && n >= 8 {
		n8 := n &^ 7
		s := dotFloat32AVX2(&a[0], &b[0], n8)
		for i := n8; i < n; i++ {
			s += a[i] * b[i]
		}
		return s
	}
	return dotFloat32Generic(a, b)
}
