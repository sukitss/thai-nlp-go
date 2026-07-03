package crf

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"
)

// buildSynthetic makes documents whose sentence boundaries are marked by a
// content token (no spaces at all), so whitespace splitting cannot find them —
// a learnable "hard" case (the ~23% of Thai boundaries with no space).
func buildSynthetic(rng *rand.Rand, nDocs int) []Example {
	fillers := []string{"ก", "ข", "ค", "ง", "จ", "ฉ", "ช"}
	const marker = "จบ" // sentence-ending marker
	docs := make([]Example, 0, nDocs)
	for d := 0; d < nDocs; d++ {
		var toks []string
		var labs []byte
		nSent := rng.Intn(4) + 2
		for s := 0; s < nSent; s++ {
			n := rng.Intn(4) + 1
			for i := 0; i < n; i++ {
				toks = append(toks, fillers[rng.Intn(len(fillers))])
				labs = append(labs, 'I')
			}
			toks = append(toks, marker)
			labs = append(labs, 'E') // boundary at the marker, no space
		}
		docs = append(docs, Example{Tokens: toks, Labels: labs})
	}
	return docs
}

// TestTrainLearns: the trained model should learn the no-space boundary pattern,
// beating the whitespace baseline decisively on held-out data.
func TestTrainLearns(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	train := buildSynthetic(rng, 300)
	test := buildSynthetic(rng, 100)

	m := Train(train, 15)
	trained := Eval(m.Labels, test)
	base := Eval(WhitespaceBaseline, test)

	t.Logf("trained: %s", trained)
	t.Logf("whitespace: %s", base)
	if trained.EF1 < 0.95 {
		t.Errorf("trained E-F1 = %.4f, want >= 0.95", trained.EF1)
	}
	if trained.EF1 <= base.EF1 {
		t.Errorf("trained (%.4f) should beat whitespace (%.4f)", trained.EF1, base.EF1)
	}
}

// TestSaveLoadRoundTrip: a saved+loaded model produces identical predictions.
func TestSaveLoadRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	m := Train(buildSynthetic(rng, 100), 8)

	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	m2, err := LoadModel(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for _, ex := range buildSynthetic(rng, 20) {
		a := string(m.Labels(ex.Tokens))
		b := string(m2.Labels(ex.Tokens))
		if a != b {
			t.Fatalf("save/load changed prediction:\n %q\n %q", a, b)
		}
	}
}

// TestWhitespaceBaseline sanity: predicts E before space tokens + at end.
func TestWhitespaceBaseline(t *testing.T) {
	got := string(WhitespaceBaseline([]string{"ก", " ", "ข", "ค"}))
	if !strings.HasSuffix(got, "E") { // last token is always E
		t.Errorf("last should be E: %q", got)
	}
	if got[0] != 'E' { // "ก" precedes a space -> E
		t.Errorf("token before space should be E: %q", got)
	}
}
