package vector

// dotInt8Generic is the portable int8 dot product: it interprets each code byte
// as a signed int8 and sums the pairwise products in an int32 accumulator. a and
// b must have equal length. It is the fallback where the AVX2 kernel is
// unavailable, and computes the tail the kernel leaves.
func dotInt8Generic(a, b []byte) int32 {
	var s int32
	for i := range a {
		s += int32(int8(a[i])) * int32(int8(b[i]))
	}
	return s
}
