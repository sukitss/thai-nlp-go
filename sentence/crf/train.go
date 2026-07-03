package crf

import (
	"math/rand"
	"strconv"
	"strings"
)

// Example is one training/eval document: word tokens (keep whitespace, as from
// tokenize.Segment) with a boundary label per token ('I' inside, 'E' end of
// sentence).
type Example struct {
	Tokens []string
	Labels []byte
}

// acc is one weight with lazy averaging (Collins averaged perceptron): sum
// accumulates w over "time" so the final averaged weight is sum/c.
type acc struct {
	w, sum float64
	t      int
}

func (a *acc) update(c int, d float64) {
	a.sum += a.w * float64(c-a.t)
	a.t = c
	a.w += d
}

func (a *acc) avg(c int) float64 {
	if c == 0 {
		return a.w
	}
	return (a.sum + a.w*float64(c-a.t)) / float64(c)
}

// Train fits a Model on labelled examples with an averaged structured
// perceptron (Viterbi decode + perceptron updates). Deterministic given the
// same input and iters. The result plugs into the same fast inference as the
// embedded model. crfcut is highly domain-dependent, so train on data matching
// your target domain.
func Train(examples []Example, iters int) *Model {
	load() // enders/starters needed by feature extraction
	fw := map[string]*[2]acc{}
	get := func(k string) *[2]acc {
		p := fw[k]
		if p == nil {
			p = &[2]acc{}
			fw[k] = p
		}
		return p
	}
	var tr [2][2]acc
	c := 0

	// precompute feature keys per example once
	keysByEx := make([][][]string, len(examples))
	for i, ex := range examples {
		keysByEx[i] = featureKeys(ex.Tokens)
	}

	rng := rand.New(rand.NewSource(1))
	order := make([]int, len(examples))
	for i := range order {
		order[i] = i
	}
	for it := 0; it < iters; it++ {
		rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		for _, ei := range order {
			keys := keysByEx[ei]
			gold := examples[ei].Labels
			if len(keys) == 0 {
				continue
			}
			c++
			// score with current weights
			scores := make([][2]float64, len(keys))
			for t, ks := range keys {
				var si, se float64
				for _, k := range ks {
					p := fw[k]
					if p != nil {
						si += p[0].w
						se += p[1].w
					}
				}
				scores[t] = [2]float64{si, se}
			}
			pred := viterbiInts(scores, &tr)
			// perceptron updates where prediction differs
			for t := range keys {
				g := lab(gold[t])
				p := pred[t]
				if g == p {
					continue
				}
				for _, k := range keys[t] {
					wk := get(k)
					wk[g].update(c, +1)
					wk[p].update(c, -1)
				}
			}
			for t := 1; t < len(keys); t++ {
				g0, g1 := lab(gold[t-1]), lab(gold[t])
				p0, p1 := pred[t-1], pred[t]
				if g0 == p0 && g1 == p1 {
					continue
				}
				tr[g0][g1].update(c, +1)
				tr[p0][p1].update(c, -1)
			}
		}
	}

	// finalize averaged weights
	m := &Model{feats: make(map[string]weights, len(fw))}
	for k, p := range fw {
		m.feats[k] = weights{i: p[0].avg(c), e: p[1].avg(c)}
	}
	m.tII, m.tIE = tr[0][0].avg(c), tr[0][1].avg(c)
	m.tEI, m.tEE = tr[1][0].avg(c), tr[1][1].avg(c)
	return m
}

func lab(b byte) int {
	if b == 'E' {
		return 1
	}
	return 0
}

// viterbiInts decodes best label path (0/1) given state scores and transitions.
func viterbiInts(scores [][2]float64, tr *[2][2]acc) []int {
	n := len(scores)
	trans := [2][2]float64{{tr[0][0].w, tr[0][1].w}, {tr[1][0].w, tr[1][1].w}}
	prev := scores[0]
	bp := make([][2]int, n)
	for i := 1; i < n; i++ {
		var cur [2]float64
		for l := 0; l < 2; l++ {
			best, bestp := prev[0]+trans[0][l], 0
			if v := prev[1] + trans[1][l]; v > best {
				best, bestp = v, 1
			}
			cur[l] = best + scores[i][l]
			bp[i][l] = bestp
		}
		prev = cur
	}
	out := make([]int, n)
	last := 0
	if prev[1] > prev[0] {
		last = 1
	}
	out[n-1] = last
	for i := n - 1; i > 0; i-- {
		last = bp[i][last]
		out[i-1] = last
	}
	return out
}

// featureKeys returns the active feature keys per token (same keys as the
// scoring path); used for training and to introspect features.
func featureKeys(toks []string) [][]string {
	if len(toks) == 0 {
		return nil
	}
	pad := make([]string, 0, len(toks)+2*window)
	for k := 0; k < window; k++ {
		pad = append(pad, "xxpad")
	}
	pad = append(pad, toks...)
	for k := 0; k < window; k++ {
		pad = append(pad, "xxpad")
	}
	ender := make([]string, len(pad))
	starter := make([]string, len(pad))
	for i, t := range pad {
		if enders[t] {
			ender[i] = "ender"
		} else {
			ender[i] = "normal"
		}
		if starters[t] {
			starter[i] = "starter"
		} else {
			starter[i] = "normal"
		}
	}
	out := make([][]string, len(toks))
	for i := window; i < len(pad)-window; i++ {
		keys := make([]string, 0, 40)
		keys = append(keys, "bias")
		for ng := 1; ng <= 3; ng++ {
			for j := i - window; j < i+window+2-ng; j++ {
				fp := strconv.Itoa(ng) + "_" + strconv.Itoa(j-i) + "_" + strconv.Itoa(j-i+ng)
				keys = append(keys, "word_"+fp+"="+strings.Join(pad[j:j+ng], "|"))
				keys = append(keys, "ender_"+fp+"="+strings.Join(ender[j:j+ng], "|"))
				keys = append(keys, "starter_"+fp+"="+strings.Join(starter[j:j+ng], "|"))
			}
		}
		out[i-window] = keys
	}
	return out
}
