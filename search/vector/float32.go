package vector

import (
	"encoding/binary"
	"math"
)

// Float32 is the exact, uncompressed baseline quantizer: it stores each vector
// normalized to unit length, so the score is the true cosine similarity and
// [Flat] returns the exact top-k. It costs Dim*4 bytes per vector and exists to
// be the ground truth every other quantizer's recall is measured against.
type Float32 struct {
	dim int
}

// NewFloat32 returns a Float32 quantizer for dim-dimensional vectors.
func NewFloat32(dim int) *Float32 {
	if dim <= 0 {
		panic("vector: NewFloat32 dim must be > 0")
	}
	return &Float32{dim: dim}
}

func (q *Float32) Dim() int     { return q.dim }
func (q *Float32) CodeLen() int { return q.dim * 4 }
func (q *Float32) Name() string { return "float32" }

// Encode stores the L2-normalized vector as little-endian float32 bytes.
func (q *Float32) Encode(vec []float32) []byte {
	if len(vec) != q.dim {
		panic("vector: Float32.Encode dim mismatch")
	}
	nv := normalized(vec)
	code := make([]byte, q.dim*4)
	for i, x := range nv {
		binary.LittleEndian.PutUint32(code[i*4:], math.Float32bits(x))
	}
	return code
}

// Query returns a scorer holding the normalized query; Score reads a code back
// into floats and returns their dot product (cosine similarity).
func (q *Float32) Query(query []float32) Scorer {
	if len(query) != q.dim {
		panic("vector: Float32.Query dim mismatch")
	}
	return &float32Scorer{q: normalized(query)}
}

type float32Scorer struct{ q []float32 }

// Score dots the query against the stored float32 code. dotF32Bytes reinterprets
// the little-endian code as float32 and uses the AVX2 dot kernel on amd64.
func (s *float32Scorer) Score(code []byte) float32 {
	return dotF32Bytes(s.q, code)
}
