package vector

// dotF32U8Generic dots a float32 slice q against a byte slice interpreted as
// UNSIGNED bytes (0..255), as scalar8 scoring needs. Portable fallback + tail.
func dotF32U8Generic(q []float32, b []byte) float32 {
	var s float32
	for i, x := range q {
		s += x * float32(b[i])
	}
	return s
}
