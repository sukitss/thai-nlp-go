package vector

import (
	"errors"
	"math/rand"
)

// PQ is a product quantizer. It splits each vector into m contiguous
// subvectors and, per subspace, learns a k-means codebook of 2^nbits
// centroids; a vector is then stored as one centroid id per subspace. Memory
// is m bytes per vector (nbits <= 8), independent of dimensionality, so it is
// the most tunable memory/accuracy trade-off here — e.g. a 1024-dim vector at
// m=64 costs 64 bytes (64x smaller) plus a shared codebook.
//
// Scoring is asymmetric (ADC): the query stays in full float precision and a
// per-query table of query-subvector-to-centroid dot products is precomputed
// once, so Score is m table lookups. Because centroids approximate the
// normalized corpus subvectors, the summed score approximates cosine
// similarity (larger = more similar).
//
// A PQ must be trained on representative vectors with [TrainPQ] before use;
// training is deterministic for a fixed seed.
type PQ struct {
	dim   int
	m     int // number of subspaces
	dsub  int // dimensions per subspace (dim/m)
	ksub  int // centroids per subspace (1<<nbits)
	nbits int
	// codebook[s] holds ksub centroids of dsub floats, flattened:
	// centroid c is codebook[s][c*dsub : (c+1)*dsub].
	codebook [][]float32
}

// TrainPQ learns a product-quantization codebook from vecs (each length dim).
// dim must be divisible by m, nbits must be in 1..8, and there must be at least
// 2^nbits training vectors. Training is deterministic for a given seed.
func TrainPQ(vecs [][]float32, m, nbits int, seed int64) (*PQ, error) {
	if len(vecs) == 0 {
		return nil, errors.New("vector: TrainPQ needs at least one vector")
	}
	dim := len(vecs[0])
	if dim <= 0 || dim%m != 0 {
		return nil, errors.New("vector: TrainPQ dim must be a positive multiple of m")
	}
	if nbits < 1 || nbits > 8 {
		return nil, errors.New("vector: TrainPQ nbits must be in 1..8")
	}
	ksub := 1 << nbits
	if len(vecs) < ksub {
		return nil, errors.New("vector: TrainPQ needs at least 2^nbits vectors")
	}
	dsub := dim / m
	q := &PQ{dim: dim, m: m, dsub: dsub, ksub: ksub, nbits: nbits, codebook: make([][]float32, m)}

	// Normalize the training set once (scoring works on unit vectors).
	norm := make([][]float32, len(vecs))
	for i, v := range vecs {
		if len(v) != dim {
			return nil, errors.New("vector: TrainPQ vectors have inconsistent dim")
		}
		norm[i] = normalized(v)
	}

	rng := rand.New(rand.NewSource(seed))
	sub := make([]float32, len(norm)*dsub) // reused per subspace
	for s := 0; s < m; s++ {
		off := s * dsub
		for i, v := range norm {
			copy(sub[i*dsub:], v[off:off+dsub])
		}
		q.codebook[s] = kmeans(sub, len(norm), dsub, ksub, 25, rng)
	}
	return q, nil
}

func (q *PQ) Dim() int     { return q.dim }
func (q *PQ) CodeLen() int { return q.m }
func (q *PQ) Name() string { return "pq" }

// Encode assigns each subvector to its nearest centroid.
func (q *PQ) Encode(vec []float32) []byte {
	if len(vec) != q.dim {
		panic("vector: PQ.Encode dim mismatch")
	}
	nv := normalized(vec)
	code := make([]byte, q.m)
	for s := 0; s < q.m; s++ {
		off := s * q.dsub
		code[s] = byte(nearest(nv[off:off+q.dsub], q.codebook[s], q.dsub, q.ksub))
	}
	return code
}

// Query precomputes the asymmetric distance table: for each subspace, the dot
// product of the query subvector with every centroid.
func (q *PQ) Query(query []float32) Scorer {
	if len(query) != q.dim {
		panic("vector: PQ.Query dim mismatch")
	}
	nq := normalized(query)
	table := make([]float32, q.m*q.ksub)
	for s := 0; s < q.m; s++ {
		off := s * q.dsub
		qsub := nq[off : off+q.dsub]
		book := q.codebook[s]
		base := s * q.ksub
		for c := 0; c < q.ksub; c++ {
			table[base+c] = dot(qsub, book[c*q.dsub:c*q.dsub+q.dsub])
		}
	}
	return &pqScorer{m: q.m, ksub: q.ksub, table: table}
}

type pqScorer struct {
	m     int
	ksub  int
	table []float32
}

func (s *pqScorer) Score(code []byte) float32 {
	var sum float32
	for sub := 0; sub < s.m; sub++ {
		sum += s.table[sub*s.ksub+int(code[sub])]
	}
	return sum
}

// nearest returns the index of the centroid in book (ksub centroids of dsub
// floats) closest to x under squared L2 distance.
func nearest(x, book []float32, dsub, ksub int) int {
	best, bestD := 0, float32(1e30)
	for c := 0; c < ksub; c++ {
		cen := book[c*dsub : c*dsub+dsub]
		var d float32
		for i, xi := range x {
			diff := xi - cen[i]
			d += diff * diff
		}
		if d < bestD {
			bestD, best = d, c
		}
	}
	return best
}

// kmeans runs Lloyd's algorithm on n points of dsub dims (flattened in data)
// and returns ksub centroids flattened as ksub*dsub floats. Empty clusters are
// reseeded to a random point. Deterministic for a given rng.
func kmeans(data []float32, n, dsub, ksub, iters int, rng *rand.Rand) []float32 {
	cent := make([]float32, ksub*dsub)
	// Init: pick ksub distinct random points.
	perm := rng.Perm(n)
	for c := 0; c < ksub; c++ {
		copy(cent[c*dsub:], data[perm[c]*dsub:perm[c]*dsub+dsub])
	}
	assign := make([]int, n)
	sum := make([]float32, ksub*dsub)
	count := make([]int, ksub)
	for it := 0; it < iters; it++ {
		changed := false
		for i := 0; i < n; i++ {
			a := nearest(data[i*dsub:i*dsub+dsub], cent, dsub, ksub)
			if a != assign[i] {
				assign[i] = a
				changed = true
			}
		}
		if it > 0 && !changed {
			break
		}
		for i := range sum {
			sum[i] = 0
		}
		for c := range count {
			count[c] = 0
		}
		for i := 0; i < n; i++ {
			a := assign[i]
			count[a]++
			base := a * dsub
			for j, x := range data[i*dsub : i*dsub+dsub] {
				sum[base+j] += x
			}
		}
		for c := 0; c < ksub; c++ {
			if count[c] == 0 {
				// Reseed an empty cluster to a random point.
				p := rng.Intn(n)
				copy(cent[c*dsub:], data[p*dsub:p*dsub+dsub])
				continue
			}
			inv := 1 / float32(count[c])
			base := c * dsub
			for j := 0; j < dsub; j++ {
				cent[base+j] = sum[base+j] * inv
			}
		}
	}
	return cent
}
