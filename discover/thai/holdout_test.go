package thai_test

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/discover"
	"github.com/sukitss/thai-nlp-go/discover/thai"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

// held-out evaluation: take words the dictionary already knows, remove them
// from it, and see whether the corpus alone can put them back.
//
// The problem this package solves has no labelled data — a corpus of coined
// names is coined precisely because nobody wrote them down. Holding out real
// words simulates it exactly: the tokenizer then shreds them the way it shreds
// a real unknown word, and the answer is known in advance, so recall is a
// number rather than an impression. The corpus is the repo's own tokenizer
// stress corpus, so this measures the same Thai every other package is
// measured on.
//
// Read the number with one caveat: this is HARDER than the case the package
// exists for. A held-out dictionary word usually breaks into other dictionary
// words — "มหาวิทยาลัย" becomes มหา + วิทยาลัย, both real — so it must clear
// the cohesion bar with no help from the dictionary's silence. A coined name
// breaks into debris that is not words at all, which is exactly the evidence
// this package leans on. Recall here is therefore a floor on what a corpus of
// genuine unknowns would score.
const (
	holdoutCount   = 40 // words to remove
	holdoutMinFreq = 30 // …chosen from words at least this common in the corpus
	holdoutMinRune = 3  // …and long enough that losing them is not a lost cause
)

func holdoutCorpus(tb testing.TB) []string {
	tb.Helper()
	f, err := os.Open("../../tokenize/testdata/stress.txt")
	if err != nil {
		tb.Skipf("stress corpus unavailable: %v", err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			lines = append(lines, line)
		}
	}
	if err := sc.Err(); err != nil {
		tb.Fatal(err)
	}
	return lines
}

// pickHoldout returns the words to remove, chosen deterministically: the most
// common qualifying words in the corpus, ties broken by spelling.
func pickHoldout(tb testing.TB, lines []string, base *dict.FlatTrie) []string {
	tb.Helper()
	seg := tokenize.New(base)
	freq := map[string]int{}
	for _, l := range lines {
		for _, tk := range seg.SegmentNoWS(l) {
			freq[tk]++
		}
	}
	var cands []string
	for w, n := range freq {
		if n < holdoutMinFreq || len([]rune(w)) < holdoutMinRune || !thai.Pronounceable(w) {
			continue
		}
		cands = append(cands, w)
	}
	sort.Slice(cands, func(i, j int) bool {
		if freq[cands[i]] != freq[cands[j]] {
			return freq[cands[i]] > freq[cands[j]]
		}
		return cands[i] < cands[j]
	})
	if len(cands) > holdoutCount {
		cands = cands[:holdoutCount]
	}
	return cands
}

// reducedDict copies the dictionary minus the held-out words.
func reducedDict(tb testing.TB, base *dict.FlatTrie, remove []string) *dict.Trie {
	tb.Helper()
	gone := make(map[string]bool, len(remove))
	for _, w := range remove {
		gone[w] = true
	}
	out := dict.NewTrie()
	base.WalkPrefix("", func(word string, _ int32) bool {
		if !gone[word] {
			out.Add(word)
		}
		return true
	})
	if out.Len() == 0 {
		tb.Fatal("reduced dictionary is empty")
	}
	return out
}

func TestHoldoutRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("holdout evaluation reads the full stress corpus")
	}
	base, err := dict.Default()
	if err != nil {
		t.Fatal(err)
	}
	lines := holdoutCorpus(t)
	held := pickHoldout(t, lines, base)
	if len(held) < holdoutCount {
		t.Skipf("corpus yielded only %d qualifying words", len(held))
	}

	reduced := reducedDict(t, base, held)
	seg := tokenize.New(reduced)
	docs := make([][]string, 0, len(lines))
	for _, l := range lines {
		docs = append(docs, seg.SegmentNoWS(l))
	}

	opts := discover.Options{
		Known: reduced.Contains,
		Valid: thai.Pronounceable,
		Affix: func(token string) bool {
			return len([]rune(token)) >= 3 && reduced.Contains(token)
		},
	}
	terms := discover.TermsFromSlices(docs, opts)

	found := make(map[string]bool, len(terms))
	for _, tm := range terms {
		found[tm.Text] = true
	}
	var missing []string
	for _, w := range held {
		if !found[w] {
			missing = append(missing, w)
		}
	}
	recall := float64(len(held)-len(missing)) / float64(len(held))
	t.Logf("docs=%d held-out=%d proposals=%d recall=%.2f", len(docs), len(held), len(terms), recall)

	// Split by length, because the two halves fail for different reasons and
	// only one of them is what this package is for. A short common word has no
	// edges to find — "ที่" stands next to everything, so nothing about its
	// surroundings says where it begins. The words worth recovering are
	// content words: names, jargon, borrowings.
	byLen := map[bool][2]int{} // long? → {recovered, total}
	for _, w := range held {
		long := len([]rune(w)) >= 4
		c := byLen[long]
		c[1]++
		if found[w] {
			c[0]++
		}
		byLen[long] = c
	}
	for _, long := range []bool{true, false} {
		c := byLen[long]
		if c[1] == 0 {
			continue
		}
		label := "short (2-3 chars)"
		if long {
			label = "long (4+ chars)"
		}
		t.Logf("  %-18s %d/%d = %.2f", label, c[0], c[1], float64(c[0])/float64(c[1]))
	}
	if len(missing) > 0 {
		t.Logf("not recovered: %s", strings.Join(missing, " "))
	}

	// The bar is deliberately below what the run scores. This is a floor
	// against regression, not a target.
	//
	// Half the held-out words are two or three characters long, and those are
	// not what this package is for: a short common word stands next to
	// everything, so nothing about its surroundings says where it begins, and
	// the fragment test that keeps debris like "หนิงเอ๋อร์" out of the results
	// costs exactly those words. That trade was made on purpose — see the note
	// in discover/fragments.go — and the split below is here so a future change
	// shows which half it moved.
	const floor = 0.4
	if recall < floor {
		t.Errorf("recall %.2f is below the floor of %.2f", recall, floor)
	}
}

func BenchmarkHoldout(b *testing.B) {
	base, err := dict.Default()
	if err != nil {
		b.Fatal(err)
	}
	lines := holdoutCorpus(b)
	held := pickHoldout(b, lines, base)
	reduced := reducedDict(b, base, held)
	seg := tokenize.New(reduced)
	var docs [][]string
	var nbytes int64
	for _, l := range lines {
		nbytes += int64(len(l)) + 1
		docs = append(docs, seg.SegmentNoWS(l))
	}
	opts := discover.Options{Known: reduced.Contains, Valid: thai.Pronounceable}
	b.SetBytes(nbytes)
	b.ReportAllocs()
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		sink += len(discover.TermsFromSlices(docs, opts))
	}
	_ = sink
	fmt.Fprintf(os.Stderr, "")
}
