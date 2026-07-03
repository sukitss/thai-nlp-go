// Package crf provides CRF-based Thai sentence segmentation — a faithful port of
// PyThaiNLP's crfcut (a linear-chain CRF over word tokens, trained on TED
// subtitles). It is far more accurate than whitespace splitting yet still fast
// and CPU-only (Viterbi over two labels): suitable for large batch workloads,
// unlike LLM/neural segmenters.
//
// The default model (Default) is the embedded crfcut model. The CRF is also
// retrainable: use Train to fit a Model on your own in-domain labelled data
// (crfcut is highly domain-dependent), Eval to measure it, and Model.Save /
// LoadModel to persist it. See cmd/crftrain.
//
// This is a separate, opt-in package: it pulls in the tokenizer and embeds the
// ~2 MB model, so the parent `sentence` package stays dependency-light. The
// model is CC-BY-4.0 (PyThaiNLP); see NOTICE.
package crf

import (
	"bufio"
	_ "embed"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/sukitss/thai-nlp-go/tokenize"
)

// Baseline linear-chain CRF transition weights (labels: 0=I "inside", 1=E
// "end"), from the crfcut model dump.
const (
	tII = 1.071558
	tIE = -0.604471
	tEI = -0.710948
	tEE = -2.615432
)

//go:embed data/crfcut_model.tsv
var modelData string

//go:embed data/crfcut_enders.txt
var endersData string

//go:embed data/crfcut_starters.txt
var startersData string

type weights struct{ i, e float64 }

// Model is a CRF sentence segmenter: state-feature weights plus the 2x2
// transition weights. Safe for concurrent reads.
type Model struct {
	feats              map[string]weights
	tII, tIE, tEI, tEE float64
}

var (
	once         sync.Once
	defaultModel *Model
	enders       map[string]bool // feature inputs (shared across models)
	starters     map[string]bool
	seg          *tokenize.Segmenter
)

func load() {
	once.Do(func() {
		feats := make(map[string]weights, 40000)
		sc := bufio.NewScanner(strings.NewReader(modelData))
		sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
		for sc.Scan() {
			line := sc.Text()
			a := strings.IndexByte(line, '\t')
			if a < 0 {
				continue
			}
			b := strings.IndexByte(line[a+1:], '\t')
			if b < 0 {
				continue
			}
			b += a + 1
			wi, _ := strconv.ParseFloat(line[a+1:b], 64)
			we, _ := strconv.ParseFloat(line[b+1:], 64)
			feats[line[:a]] = weights{wi, we}
		}
		defaultModel = &Model{feats: feats, tII: tII, tIE: tIE, tEI: tEI, tEE: tEE}
		enders = parseSet(endersData)
		starters = parseSet(startersData)
		seg, _ = tokenize.NewDefault()
	})
}

// Default returns the embedded crfcut model (the baseline).
func Default() *Model {
	load()
	return defaultModel
}

func parseSet(data string) map[string]bool {
	m := map[string]bool{}
	sc := bufio.NewScanner(strings.NewReader(data))
	for sc.Scan() {
		if w := strings.TrimSpace(sc.Text()); w != "" {
			m[w] = true
		}
	}
	return m
}

// Split segments text into sentences with the default model. Returns nil for
// empty input.
func Split(text string) []string { return Default().Split(text) }

// Split segments text into sentences using this model.
func (m *Model) Split(text string) []string {
	load()
	toks := seg.Segment(text) // keep whitespace (word_tokenize default)
	if len(toks) == 0 {
		return nil
	}
	labs := m.tag(m.stateScores(toks))
	labs[len(labs)-1] = 'E' // always cut the last sentence

	// Terminal-punctuation and whitespace overrides (mirror crfcut).
	for idx, tok := range toks {
		st := strings.TrimSpace(tok)
		if strings.HasSuffix(st, "!") || strings.HasSuffix(st, ".") || strings.HasSuffix(st, "?") {
			labs[idx] = 'E'
		} else if (idx == 0 || labs[idx-1] == 'E') && st == "" {
			labs[idx] = 'I'
		}
	}

	var out []string
	var sb strings.Builder
	for i, w := range toks {
		sb.WriteString(w)
		if labs[i] == 'E' && sb.Len() > 0 {
			out = append(out, sb.String())
			sb.Reset()
		}
	}
	return out
}

const window = 2

// tags returns raw I/E labels for the tokens (no punctuation/space overrides);
// used by training and evaluation.
func (m *Model) tags(toks []string) []byte {
	if len(toks) == 0 {
		return nil
	}
	return m.tag(m.stateScores(toks))
}

// stateScores computes, per token, the summed CRF state weights {I, E}, building
// each feature key into a reused byte buffer and looking it up via
// m.feats[string(buf)] (the compiler special-cases map[string(bytes)] to avoid
// allocating the key). Ports crfcut._extract_features (window 2, n-grams 1..3).
func (m *Model) stateScores(toks []string) [][2]float64 {
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

	out := make([][2]float64, len(toks))
	buf := make([]byte, 0, 64)
	add := func(si, se *float64, prefix string, ng, d0 int, parts []string) {
		buf = buf[:0]
		buf = append(buf, prefix...)
		buf = strconv.AppendInt(buf, int64(ng), 10)
		buf = append(buf, '_')
		buf = strconv.AppendInt(buf, int64(d0), 10)
		buf = append(buf, '_')
		buf = strconv.AppendInt(buf, int64(d0+ng), 10)
		buf = append(buf, '=')
		for k, p := range parts {
			if k > 0 {
				buf = append(buf, '|')
			}
			buf = append(buf, p...)
		}
		w := m.feats[string(buf)]
		*si += w.i
		*se += w.e
	}

	for i := window; i < len(pad)-window; i++ {
		si, se := m.feats["bias"].i, m.feats["bias"].e
		for ng := 1; ng <= 3; ng++ {
			for j := i - window; j < i+window+2-ng; j++ {
				add(&si, &se, "word_", ng, j-i, pad[j:j+ng])
				add(&si, &se, "ender_", ng, j-i, ender[j:j+ng])
				add(&si, &se, "starter_", ng, j-i, starter[j:j+ng])
			}
		}
		out[i-window] = [2]float64{si, se}
	}
	return out
}

// tag runs Viterbi over the two labels and returns 'I'/'E' per token.
func (m *Model) tag(scores [][2]float64) []byte {
	n := len(scores)
	trans := [2][2]float64{{m.tII, m.tIE}, {m.tEI, m.tEE}}
	prev := scores[0]
	bp := make([][2]int, n)
	for i := 1; i < n; i++ {
		var cur [2]float64
		for l := 0; l < 2; l++ {
			best := math.Inf(-1)
			bestp := 0
			for p := 0; p < 2; p++ {
				if v := prev[p] + trans[p][l]; v > best {
					best, bestp = v, p
				}
			}
			cur[l] = best + scores[i][l]
			bp[i][l] = bestp
		}
		prev = cur
	}
	labs := make([]byte, n)
	last := 0
	if prev[1] > prev[0] {
		last = 1
	}
	labs[n-1] = label(last)
	for i := n - 1; i > 0; i-- {
		last = bp[i][last]
		labs[i-1] = label(last)
	}
	return labs
}

func label(l int) byte {
	if l == 1 {
		return 'E'
	}
	return 'I'
}
