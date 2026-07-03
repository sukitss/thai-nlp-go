package chunk

import (
	"bufio"
	"os"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/sentence/crf"
)

func corpusLines(t testing.TB) []string {
	t.Helper()
	f, err := os.Open("../sentence/crf/testdata/crfcut.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	return lines
}

// TestCRFConcatPreserving backs the package-doc claim that crf.Split can be
// passed as Options.Sentences directly: on trimmed single-line paragraphs
// (exactly what Split hands to Sentences) its segments concatenate back to
// the input. (On raw text it can drop trailing whitespace after terminal
// punctuation, and whitespace-only input yields nothing — neither case
// reaches Sentences.)
func TestCRFConcatPreserving(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the CRF model")
	}
	for _, line := range corpusLines(t) {
		par := strings.TrimSpace(line)
		if par == "" {
			continue
		}
		if got := strings.Join(crf.Split(par), ""); got != par {
			t.Fatalf("crf.Split not concatenation-preserving:\n in %q\nout %q", par, got)
		}
	}
}

// TestSplitWithCRFSentences runs the full pipeline with the CRF splitter
// over the real corpus and re-asserts every chunk invariant.
func TestSplitWithCRFSentences(t *testing.T) {
	if testing.Short() {
		t.Skip("loads the CRF model")
	}
	text := strings.Join(corpusLines(t), "\n")
	for _, o := range []Options{
		{MaxUnits: 40, Sentences: crf.Split},
		{MaxUnits: 120, OverlapUnits: 30, Sentences: crf.Split},
	} {
		cs, err := Split(text, o)
		if err != nil {
			t.Fatal(err)
		}
		checkInvariants(t, text, cs, o)
	}
}
