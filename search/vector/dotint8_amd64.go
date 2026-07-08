//go:build amd64

package vector

import "golang.org/x/sys/cpu"

// useAVX2 gates the assembly kernel. GOAMD64=v1 does not guarantee AVX2, so it is
// detected once at startup and the portable loop is used when it is absent.
var useAVX2 = cpu.X86.HasAVX2

// dotInt8AVX2 sums the pairwise int8 products of the first n bytes of a and b
// using AVX2, returning the result in an int32. n must be a positive multiple of
// 16. Implemented in dotint8_amd64.s.
//
//go:noescape
func dotInt8AVX2(a, b *byte, n int) int32

// dotInt8 is the int8 dot product used by [Scalar8Sym]: the AVX2 kernel handles
// the 16-wide body and the portable loop finishes the remainder (and does the
// whole thing when AVX2 is absent).
func dotInt8(a, b []byte) int32 {
	n := len(a)
	if useAVX2 && n >= 16 {
		n16 := n &^ 15
		s := dotInt8AVX2(&a[0], &b[0], n16)
		for i := n16; i < n; i++ {
			s += int32(int8(a[i])) * int32(int8(b[i]))
		}
		return s
	}
	return dotInt8Generic(a, b)
}
