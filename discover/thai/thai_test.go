package thai_test

import (
	"bufio"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/discover"
	"github.com/sukitss/thai-nlp-go/discover/thai"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

// wanted is every term the corpus in testdata/corpus.txt actually contains:
// four coined personal names, and one coined creature name that is written
// only ever as "คาลูก้าหุ้มเกราะ".
//
// The corpus is invented for this test, so the answer is known exactly and
// precision and recall are both measurable — see TestCorpusPrecisionRecall,
// which reports them.
var wanted = []string{"ก็อดวิน", "ซานคว่า", "ตูเรีย", "มิราเบล", "คาลูก้า"}

func corpus(t *testing.T) [][]string {
	t.Helper()
	f, err := os.Open("testdata/corpus.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	seg, err := tokenize.NewDefault()
	if err != nil {
		t.Fatal(err)
	}
	var docs [][]string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			docs = append(docs, seg.SegmentNoWS(line))
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return docs
}

// TestCorpusPrecisionRecall is the package's measurement: against a corpus whose
// vocabulary is known, how much of what it should find does it find, and how
// much of what it finds is real.
//
// Both are asserted at 1.0 rather than at some tolerance. This corpus is small
// and clean by construction; anything less than perfect on it is a regression,
// and the numbers that matter on real prose are recorded in the package doc.
func TestCorpusPrecisionRecall(t *testing.T) {
	terms := discover.TermsFromSlices(corpus(t), thai.Options())

	found := make(map[string]bool, len(terms))
	for _, tm := range terms {
		found[tm.Text] = true
	}
	want := make(map[string]bool, len(wanted))
	for _, w := range wanted {
		want[w] = true
	}

	var missing, spurious []string
	for _, w := range wanted {
		if !found[w] {
			missing = append(missing, w)
		}
	}
	for _, tm := range terms {
		if !want[tm.Text] {
			spurious = append(spurious, tm.Text)
		}
	}
	sort.Strings(spurious)

	hits := len(wanted) - len(missing)
	recall := float64(hits) / float64(len(wanted))
	precision := 1.0
	if len(terms) > 0 {
		precision = float64(hits) / float64(len(terms))
	}
	t.Logf("terms=%d precision=%.2f recall=%.2f", len(terms), precision, recall)

	if len(missing) > 0 {
		t.Errorf("did not find %v", missing)
	}
	if len(spurious) > 0 {
		t.Errorf("reported %d terms that are not in the corpus's vocabulary: %v", len(spurious), spurious)
	}
}

// The creature is written as "คาลูก้าหุ้มเกราะ" every single time, and the name
// is the coined half — "หุ้มเกราะ" is an ordinary word that also appears on
// soldiers and coats elsewhere in the corpus. Reporting the whole phrase would
// be worse than useless for a tokenizer dictionary: longest-match would then
// swallow the name and a search for it would find nothing.
func TestFixedPhraseYieldsTheNameNotThePhrase(t *testing.T) {
	terms := discover.TermsFromSlices(corpus(t), thai.Options())
	var got []string
	for _, tm := range terms {
		if strings.Contains(tm.Text, "คาลูก้า") {
			got = append(got, tm.Text)
		}
	}
	if len(got) != 1 || got[0] != "คาลูก้า" {
		t.Errorf("want exactly [คาลูก้า], got %v", got)
	}
}

func TestPronounceable(t *testing.T) {
	cases := []struct {
		text string
		want bool
		why  string
	}{
		{"ทานูกิ", true, "a loanword, three vowelled clusters"},
		{"ก็อบลิน", true, "two bare consonants in a row is still readable"},
		{"โคโบลด์", true, "a loanword with a silent mark"},
		{"ดผม", false, "three bare consonants cannot be read aloud"},
		{"ดคน", false, "segmentation debris"},
		{"เหรอ?", false, "a question mark rode along, so the run crossed a boundary"},
		{"กล่าวว่า“", false, "so did an opening quote"},
		{"ๆก็", false, "ๆ repeats the word before it and cannot open a term"},
		{"สุดๆ", false, "a term ending in ๆ is the same word twice"},
		{"machine", false, "not Thai script"},
		{"", false, "nothing at all"},
	}
	for _, c := range cases {
		if got := thai.Pronounceable(c.text); got != c.want {
			t.Errorf("Pronounceable(%q) = %v, want %v — %s", c.text, got, c.want, c.why)
		}
	}
}

// Options is usable more than once and does not reload the dictionary into a
// different state each time.
func TestOptionsAreStable(t *testing.T) {
	a, b := thai.Options(), thai.Options()
	if a.Known == nil || b.Known == nil {
		t.Fatal("the dictionary should be wired as the known-word test")
	}
	for _, w := range []string{"บ้าน", "เดิน"} {
		if !a.Known(w) || !b.Known(w) {
			t.Errorf("%q should be a known word", w)
		}
	}
	if a.Affix == nil || a.Affix("ก็") {
		t.Error("a two-character particle must not be peelable as an affix")
	}
	if !a.Affix("หุ้ม") {
		t.Error("an ordinary content word should be peelable")
	}
}

func BenchmarkTerms(b *testing.B) {
	f, err := os.Open("testdata/corpus.txt")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	seg, err := tokenize.NewDefault()
	if err != nil {
		b.Fatal(err)
	}
	var docs [][]string
	var nbytes int64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		nbytes += int64(len(line)) + 1
		docs = append(docs, seg.SegmentNoWS(line))
	}
	opts := thai.Options() // outside the timer: it loads the dictionary
	b.SetBytes(nbytes)
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		sink += len(discover.TermsFromSlices(docs, opts))
	}
	_ = sink
}
