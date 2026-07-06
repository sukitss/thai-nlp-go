package charseg

// Averaged-perceptron training for the hashed character-level boundary
// classifier. Dependency-free (same family as crf.Train). Only candidate
// positions (isCandidate) are scored/updated, so training and inference use the
// identical decision set and the classes are naturally balanced.

import "math/rand"

// Example is one training document: its runes and, per rune, whether a sentence
// boundary follows (true only at candidate positions).
type Example struct {
	Runes []rune
	Bnd   []bool
}

// TrainConfig configures training.
type TrainConfig struct {
	Bits  int // weight-table size = 1<<Bits
	Iters int // perceptron passes
	Seed  int64
}

func (c TrainConfig) withDefaults() TrainConfig {
	if c.Bits == 0 {
		c.Bits = 20
	}
	if c.Iters == 0 {
		c.Iters = 10
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
	return c
}

// Train fits a Model with an averaged perceptron. Deterministic given the same
// input and config.
func Train(examples []Example, cfg TrainConfig) *Model {
	cfg = cfg.withDefaults()
	size := 1 << cfg.Bits
	mask := uint64(size - 1)

	w := make([]float64, size)   // current weights
	sum := make([]float64, size) // running sum for averaging
	last := make([]int64, size)  // timestamp of last update (lazy averaging)
	var c int64                  // global step counter

	// flush brings sum[k] up to date before reading/updating w[k].
	flush := func(k uint32) {
		sum[k] += w[k] * float64(c-last[k])
		last[k] = c
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	order := make([]int, len(examples))
	for i := range order {
		order[i] = i
	}

	var buf [maxFeat]uint64
	for it := 0; it < cfg.Iters; it++ {
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		for _, ei := range order {
			r := examples[ei].Runes
			bnd := examples[ei].Bnd
			for i := 0; i < len(r); i++ {
				if i != len(r)-1 && !isCandidate(r, i) {
					continue // non-candidate: never a decision point
				}
				c++
				n := features(r, i, &buf)
				// score
				var s float64
				for k := 0; k < n; k++ {
					idx := uint32(buf[k] & mask)
					s += w[idx]
				}
				pred := s > 0
				gold := bnd[i]
				if pred == gold {
					continue
				}
				var delta float64 = 1
				if pred { // predicted boundary but gold says no
					delta = -1
				}
				for k := 0; k < n; k++ {
					idx := uint32(buf[k] & mask)
					flush(idx)
					w[idx] += delta
				}
			}
		}
	}

	// finalize averaged weights
	out := make([]float32, size)
	for k := 0; k < size; k++ {
		flush(uint32(k))
		if c > 0 {
			out[k] = float32(sum[k] / float64(c))
		} else {
			out[k] = float32(w[k])
		}
	}
	return NewModel(out)
}
