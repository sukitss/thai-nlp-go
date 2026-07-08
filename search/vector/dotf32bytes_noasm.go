//go:build !amd64

package vector

import (
	"encoding/binary"
	"math"
)

// dotF32Bytes decodes each little-endian float32 from code and dots it with q.
// The portable path makes no endianness or alignment assumptions.
func dotF32Bytes(q []float32, code []byte) float32 {
	var s float32
	for i, x := range q {
		s += x * math.Float32frombits(binary.LittleEndian.Uint32(code[i*4:]))
	}
	return s
}
