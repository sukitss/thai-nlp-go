// Package crf provides CRF-based Thai sentence segmentation — a faithful port of
// PyThaiNLP's crfcut (a linear-chain CRF over word tokens, trained on TED
// subtitles). It is far more accurate than whitespace splitting yet still fast
// and CPU-only (Viterbi over two labels): suitable for large batch workloads,
// unlike LLM/neural segmenters.
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

// Linear-chain CRF transition weights (labels: 0=I "inside", 1=E "end").
// From the crfcut model dump.
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

var (
	once     sync.Once
	features map[string]weights
	enders   map[string]bool
	starters map[string]bool
	seg      *tokenize.Segmenter
)

func load() {
	once.Do(func() {
		features = make(map[string]weights, 40000)
		sc := bufio.NewScanner(strings.NewReader(modelData))
		sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
		for sc.Scan() {
			line := sc.Text()
			// attr \t wI \t wE  (attr may contain spaces but not tabs)
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
			features[line[:a]] = weights{wi, we}
		}
		enders = parseSet(endersData)
		starters = parseSet(startersData)
		seg, _ = tokenize.NewDefault()
	})
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

// Split segments text into sentences using the CRF model. It returns nil for
// empty input.
func Split(text string) []string {
	load()
	toks := seg.Segment(text) // keep whitespace (word_tokenize default)
	if len(toks) == 0 {
		return nil
	}
	labs := tag(extractFeatures(toks))
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

// extractFeatures ports crfcut._extract_features (window 2, n-grams 1..3).
func extractFeatures(toks []string) [][]string {
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

	out := make([][]string, 0, len(toks))
	for i := window; i < len(pad)-window; i++ {
		feats := make([]string, 0, 40)
		feats = append(feats, "bias")
		for ng := 1; ng <= 3; ng++ {
			for j := i - window; j < i+window+2-ng; j++ {
				fp := strconv.Itoa(ng) + "_" + strconv.Itoa(j-i) + "_" + strconv.Itoa(j-i+ng)
				feats = append(feats, "word_"+fp+"="+strings.Join(pad[j:j+ng], "|"))
				feats = append(feats, "ender_"+fp+"="+strings.Join(ender[j:j+ng], "|"))
				feats = append(feats, "starter_"+fp+"="+strings.Join(starter[j:j+ng], "|"))
			}
		}
		out = append(out, feats)
	}
	return out
}

func stateScore(attrs []string, label int) float64 {
	var s float64
	if label == 0 {
		for _, a := range attrs {
			s += features[a].i
		}
	} else {
		for _, a := range attrs {
			s += features[a].e
		}
	}
	return s
}

// tag runs Viterbi over the two labels and returns 'I'/'E' per token.
func tag(feats [][]string) []byte {
	n := len(feats)
	trans := [2][2]float64{{tII, tIE}, {tEI, tEE}}
	prev := [2]float64{stateScore(feats[0], 0), stateScore(feats[0], 1)}
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
			cur[l] = best + stateScore(feats[i], l)
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
