package vector

import (
	"encoding/binary"
	"math"
)

// Scalar8 quantizes each (normalized) vector to one uint8 per dimension using a
// per-vector affine scale: value = min + code*scale, with min and scale stored
// as a small header. At ~1 byte per dimension it is ~4x smaller than [Float32]
// and markedly more accurate than [Binary]. Scoring is asymmetric: the query
// stays float and is dotted against each dequantized code, approximating cosine
// similarity.
//
// Code layout: min float32 | scale float32 | dim bytes. CodeLen = 8 + dim.
type Scalar8 struct {
	dim int
}

// NewScalar8 returns a Scalar8 quantizer for dim-dimensional vectors.
func NewScalar8(dim int) *Scalar8 {
	if dim <= 0 {
		panic("vector: NewScalar8 dim must be > 0")
	}
	return &Scalar8{dim: dim}
}

func (q *Scalar8) Dim() int     { return q.dim }
func (q *Scalar8) CodeLen() int { return 8 + q.dim }
func (q *Scalar8) Name() string { return "scalar8" }

// Encode normalizes vec, then maps its per-vector [min,max] range onto 0..255.
func (q *Scalar8) Encode(vec []float32) []byte {
	if len(vec) != q.dim {
		panic("vector: Scalar8.Encode dim mismatch")
	}
	nv := normalized(vec)
	min, max := nv[0], nv[0]
	for _, x := range nv[1:] {
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}
	scale := (max - min) / 255
	code := make([]byte, 8+q.dim)
	binary.LittleEndian.PutUint32(code[0:], math.Float32bits(min))
	binary.LittleEndian.PutUint32(code[4:], math.Float32bits(scale))
	if scale == 0 { // constant vector: all codes 0
		return code
	}
	inv := 1 / scale
	for i, x := range nv {
		v := (x - min) * inv
		r := int32(v + 0.5) // round
		if r < 0 {
			r = 0
		} else if r > 255 {
			r = 255
		}
		code[8+i] = byte(r)
	}
	return code
}

// Query holds the normalized query and its dimension sum; Score dequantizes
// each code on the fly and returns the dot product.
func (q *Scalar8) Query(query []float32) Scorer {
	if len(query) != q.dim {
		panic("vector: Scalar8.Query dim mismatch")
	}
	nq := normalized(query)
	var qsum float32
	for _, x := range nq {
		qsum += x
	}
	return &scalar8Scorer{q: nq, qsum: qsum}
}

type scalar8Scorer struct {
	q    []float32
	qsum float32
}

func (s *scalar8Scorer) Score(code []byte) float32 {
	min := math.Float32frombits(binary.LittleEndian.Uint32(code[0:]))
	scale := math.Float32frombits(binary.LittleEndian.Uint32(code[4:]))
	// dot(q, min + scale*byte) = min*sum(q) + scale*sum(q[i]*byte[i]); the second
	// sum is a float32-times-unsigned-byte dot with an AVX2 kernel on amd64.
	acc := dotF32U8(s.q, code[8:8+len(s.q)])
	return min*s.qsum + scale*acc
}
