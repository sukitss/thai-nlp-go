package crf

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Metrics reports sentence-boundary (label 'E') precision/recall/F1 and
// per-token accuracy, following crfcut's evaluation (E is the informative class).
type Metrics struct {
	EP, ER, EF1 float64
	Accuracy    float64
	TP, FP, FN  int
	Tokens      int
}

// Labeler returns an 'I'/'E' label per token.
type Labeler func(tokens []string) []byte

// Eval scores a labeler against gold examples.
func Eval(pred Labeler, gold []Example) Metrics {
	var m Metrics
	correct := 0
	for _, ex := range gold {
		if len(ex.Tokens) == 0 {
			continue
		}
		p := pred(ex.Tokens)
		for t := range ex.Tokens {
			g := ex.Labels[t]
			pt := p[t]
			m.Tokens++
			if pt == g {
				correct++
			}
			switch {
			case pt == 'E' && g == 'E':
				m.TP++
			case pt == 'E' && g != 'E':
				m.FP++
			case pt != 'E' && g == 'E':
				m.FN++
			}
		}
	}
	if m.TP+m.FP > 0 {
		m.EP = float64(m.TP) / float64(m.TP+m.FP)
	}
	if m.TP+m.FN > 0 {
		m.ER = float64(m.TP) / float64(m.TP+m.FN)
	}
	if m.EP+m.ER > 0 {
		m.EF1 = 2 * m.EP * m.ER / (m.EP + m.ER)
	}
	if m.Tokens > 0 {
		m.Accuracy = float64(correct) / float64(m.Tokens)
	}
	return m
}

func (m Metrics) String() string {
	return fmt.Sprintf("E-P=%.4f E-R=%.4f E-F1=%.4f acc=%.4f (TP=%d FP=%d FN=%d)",
		m.EP, m.ER, m.EF1, m.Accuracy, m.TP, m.FP, m.FN)
}

// Labels returns the model's raw I/E labels for tokens (no punctuation/space
// overrides) — the model's own prediction, for evaluation.
func (m *Model) Labels(tokens []string) []byte { return m.tags(tokens) }

// WhitespaceBaseline predicts a sentence end wherever a space token follows (and
// at the last token) — i.e. naive split-on-space, the thing a CRF should beat.
func WhitespaceBaseline(tokens []string) []byte {
	out := make([]byte, len(tokens))
	for i := range tokens {
		out[i] = 'I'
		if i+1 < len(tokens) && strings.TrimSpace(tokens[i+1]) == "" {
			out[i] = 'E'
		}
	}
	if len(out) > 0 {
		out[len(out)-1] = 'E'
	}
	return out
}

// Save writes the model (transitions + state-feature weights) in a simple text
// format loadable by LoadModel.
func (m *Model) Save(w io.Writer) error {
	bw := bufio.NewWriter(w)
	if _, err := fmt.Fprintf(bw, "#trans\t%g\t%g\t%g\t%g\n", m.tII, m.tIE, m.tEI, m.tEE); err != nil {
		return err
	}
	for k, v := range m.feats {
		if _, err := fmt.Fprintf(bw, "%s\t%g\t%g\n", k, v.i, v.e); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// LoadModel reads a model written by Save.
func LoadModel(r io.Reader) (*Model, error) {
	m := &Model{feats: map[string]weights{}, tII: tII, tIE: tIE, tEI: tEI, tEE: tEE}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<16), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#trans\t") {
			f := strings.Split(line, "\t")
			if len(f) == 5 {
				m.tII, _ = strconv.ParseFloat(f[1], 64)
				m.tIE, _ = strconv.ParseFloat(f[2], 64)
				m.tEI, _ = strconv.ParseFloat(f[3], 64)
				m.tEE, _ = strconv.ParseFloat(f[4], 64)
			}
			continue
		}
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
		m.feats[line[:a]] = weights{wi, we}
	}
	return m, sc.Err()
}
