package chunk

import (
	"math/rand/v2"
	"os"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/sentence"
)

// checkInvariants asserts, for any Split output, the package contract:
//
//  1. Text == input[Start:End] for every chunk;
//  2. chunks appear in order, [Start,End) advance strictly, and every gap
//     between covered spans (and before/after them) is whitespace-only, so
//     no non-whitespace content is lost or duplicated outside overlaps;
//  3. Measure(Text) <= MaxUnits except documented unsplittable hard-cut
//     pieces (single rune, or base rune + Thai combining marks);
//  4. no boundary lands mid-rune, starts/ends with whitespace, or splits a
//     Thai combining mark from a non-whitespace base;
//  5. any [Start,End) overlap between consecutive chunks measures at most
//     OverlapUnits.
func checkInvariants(t *testing.T, text string, cs []Chunk, o Options) {
	t.Helper()
	measure := o.Measure
	if measure == nil {
		measure = utf8.RuneCountInString
	}
	// Rune boundaries as Go decoding sees them (each invalid byte is its own
	// 1-byte RuneError rune, matching the chunker's traversal).
	boundary := make([]bool, len(text)+1)
	for i := range text {
		boundary[i] = true
	}
	boundary[len(text)] = true
	pos := 0
	for i, c := range cs {
		if c.Start < 0 || c.End > len(text) || c.Start >= c.End {
			t.Fatalf("chunk %d: bad range [%d,%d) in len %d", i, c.Start, c.End, len(text))
		}
		if text[c.Start:c.End] != c.Text {
			t.Fatalf("chunk %d: Text %q != input[%d:%d] %q", i, c.Text, c.Start, c.End, text[c.Start:c.End])
		}
		if !boundary[c.Start] || !boundary[c.End] {
			t.Fatalf("chunk %d: boundary mid-rune [%d,%d)", i, c.Start, c.End)
		}
		checkMarkBoundary(t, text, i, c.Start)
		checkMarkBoundary(t, text, i, c.End)
		if r, _ := utf8.DecodeRuneInString(c.Text); unicode.IsSpace(r) {
			t.Fatalf("chunk %d starts with whitespace: %q", i, c.Text)
		}
		if r, _ := utf8.DecodeLastRuneInString(c.Text); unicode.IsSpace(r) {
			t.Fatalf("chunk %d ends with whitespace: %q", i, c.Text)
		}
		if m := measure(c.Text); m > o.MaxUnits && !unsplittable(c.Text) {
			t.Fatalf("chunk %d: measure %d > MaxUnits %d: %q", i, m, o.MaxUnits, c.Text)
		}
		if i > 0 {
			p := cs[i-1]
			if c.Start <= p.Start || c.End <= p.End {
				t.Fatalf("chunk %d: no forward progress: [%d,%d) after [%d,%d)", i, c.Start, c.End, p.Start, p.End)
			}
			if c.Start < p.End {
				if o.OverlapUnits == 0 {
					t.Fatalf("chunk %d: overlap with OverlapUnits=0", i)
				}
				if m := measure(text[c.Start:p.End]); m > o.OverlapUnits {
					t.Fatalf("chunk %d: overlap %q measures %d > OverlapUnits %d", i, text[c.Start:p.End], m, o.OverlapUnits)
				}
			}
		}
		if c.Start > pos && strings.TrimSpace(text[pos:c.Start]) != "" {
			t.Fatalf("chunk %d: non-whitespace lost in gap %q", i, text[pos:c.Start])
		}
		if c.End > pos {
			pos = c.End
		}
	}
	if strings.TrimSpace(text[pos:]) != "" {
		t.Fatalf("non-whitespace lost after last chunk: %q", text[pos:])
	}
}

// checkMarkBoundary: a Thai combining mark right after a boundary is legal
// only when the rune before the boundary is whitespace (orphan mark, no base
// was split). Holds for the built-in splitter and any tokenizer-derived one.
func checkMarkBoundary(t *testing.T, text string, i, b int) {
	t.Helper()
	if b == 0 || b >= len(text) {
		return
	}
	r, _ := utf8.DecodeRuneInString(text[b:])
	if !isThaiMark(r) {
		return
	}
	prev, _ := utf8.DecodeLastRuneInString(text[:b])
	if !unicode.IsSpace(prev) {
		t.Fatalf("chunk %d: boundary %d splits combining mark %q from base %q", i, b, r, prev)
	}
}

func unsplittable(s string) bool {
	first := true
	for _, r := range s {
		if first {
			first = false
			continue
		}
		if !isThaiMark(r) {
			return false
		}
	}
	return true
}

func mustSplit(t *testing.T, text string, o Options) []Chunk {
	t.Helper()
	cs, err := Split(text, o)
	if err != nil {
		t.Fatalf("Split(%q): %v", text, err)
	}
	checkInvariants(t, text, cs, o)
	return cs
}

func texts(cs []Chunk) []string {
	out := make([]string, len(cs))
	for i, c := range cs {
		out[i] = c.Text
	}
	return out
}

func eqStrings(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d chunks %q, want %d %q", len(got), got, len(want), want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("chunk %d: got %q, want %q", i, got[i], want[i])
		}
	}
}

func TestErrors(t *testing.T) {
	for _, o := range []Options{
		{MaxUnits: 0},
		{MaxUnits: -3},
	} {
		if _, err := Split("x", o); err != ErrMaxUnits {
			t.Errorf("MaxUnits=%d: err = %v, want ErrMaxUnits", o.MaxUnits, err)
		}
	}
	for _, o := range []Options{
		{MaxUnits: 5, OverlapUnits: 5},
		{MaxUnits: 5, OverlapUnits: 9},
		{MaxUnits: 5, OverlapUnits: -1},
	} {
		if _, err := Split("x", o); err != ErrOverlap {
			t.Errorf("OverlapUnits=%d: err = %v, want ErrOverlap", o.OverlapUnits, err)
		}
	}
}

func TestEmptyAndWhitespaceOnly(t *testing.T) {
	for _, in := range []string{"", " ", "\n", "\r\n\r\n", " \t \n  \r ", "\n\n\n"} {
		cs, err := Split(in, Options{MaxUnits: 10})
		if err != nil || len(cs) != 0 {
			t.Errorf("Split(%q) = %v, %v; want empty, nil", in, cs, err)
		}
	}
}

func TestTiny(t *testing.T) {
	cs := mustSplit(t, "ก", Options{MaxUnits: 5})
	eqStrings(t, texts(cs), []string{"ก"})
	if cs[0].Start != 0 || cs[0].End != len("ก") {
		t.Fatalf("offsets [%d,%d)", cs[0].Start, cs[0].End)
	}

	cs = mustSplit(t, "  hi  ", Options{MaxUnits: 5})
	eqStrings(t, texts(cs), []string{"hi"})
	if cs[0].Start != 2 || cs[0].End != 4 {
		t.Fatalf("offsets [%d,%d), want [2,4)", cs[0].Start, cs[0].End)
	}
}

func TestParagraphPerLineProseStyle(t *testing.T) {
	lines := []string{
		"บทที่ ๑ การเดินทางเริ่มต้น",
		"เขาออกเดินทางตั้งแต่เช้าตรู่ อากาศยังเย็นอยู่",
		"“ไปไหนกัน” เสียงหนึ่งดังขึ้นจากข้างหลัง",
		"เธอหันไปมองโดยไม่ตอบอะไร",
	}
	text := strings.Join(lines, "\n")
	cs := mustSplit(t, text, Options{MaxUnits: 200})
	eqStrings(t, texts(cs), lines)
	for i, c := range cs {
		if want := strings.Index(text, lines[i]); c.Start != want {
			t.Fatalf("chunk %d Start = %d, want %d", i, c.Start, want)
		}
	}
}

func TestCRLFAndBlankLines(t *testing.T) {
	text := "บรรทัดหนึ่ง\r\nบรรทัดสอง\r\n\r\n  บรรทัดสาม  \r\n"
	cs := mustSplit(t, text, Options{MaxUnits: 100})
	eqStrings(t, texts(cs), []string{"บรรทัดหนึ่ง", "บรรทัดสอง", "บรรทัดสาม"})
	for i, c := range cs {
		if strings.Index(text, c.Text) != c.Start {
			t.Fatalf("chunk %d Start = %d, want %d", i, c.Start, strings.Index(text, c.Text))
		}
	}
}

func TestGreedyPacking(t *testing.T) {
	cs := mustSplit(t, "aa bb cc dd", Options{MaxUnits: 6})
	eqStrings(t, texts(cs), []string{"aa bb", "cc dd"})

	// หนึ่ง=5 runes(+space 6), สอง=3(+space 4), สาม=3.
	cs = mustSplit(t, "หนึ่ง สอง สาม", Options{MaxUnits: 10})
	eqStrings(t, texts(cs), []string{"หนึ่ง สอง", "สาม"})
}

func TestOverlap(t *testing.T) {
	cs := mustSplit(t, "aa bb cc dd", Options{MaxUnits: 6, OverlapUnits: 3})
	eqStrings(t, texts(cs), []string{"aa bb", "bb cc", "cc dd"})
	if cs[1].Start != 3 || cs[2].Start != 6 {
		t.Fatalf("overlap starts %d,%d, want 3,6", cs[1].Start, cs[2].Start)
	}
}

func TestOverlapSkippedWhenLastSentenceTooBig(t *testing.T) {
	cs := mustSplit(t, "aaaa bbbb", Options{MaxUnits: 8, OverlapUnits: 4})
	eqStrings(t, texts(cs), []string{"aaaa", "bbbb"})
	if cs[1].Start < cs[0].End {
		t.Fatalf("unexpected overlap: [%d,%d) then [%d,%d)", cs[0].Start, cs[0].End, cs[1].Start, cs[1].End)
	}
}

func TestOverlapShrinksToFitNextSentence(t *testing.T) {
	// Units measure 3,3,3 ("aa "=3...). MaxUnits 7, OverlapUnits 6: a full
	// two-sentence overlap (6) cannot fit the next sentence (9 > 7), so it
	// shrinks to one sentence.
	cs := mustSplit(t, "aa bb cc", Options{MaxUnits: 7, OverlapUnits: 6})
	eqStrings(t, texts(cs), []string{"aa bb", "bb cc"})
}

func TestOverlapNeverCrossesParagraphs(t *testing.T) {
	text := "aa bb cc dd\nee ff gg hh"
	cs := mustSplit(t, text, Options{MaxUnits: 6, OverlapUnits: 3})
	nl := strings.IndexByte(text, '\n')
	for _, c := range cs {
		if c.Start < nl && c.End > nl {
			t.Fatalf("chunk [%d,%d) %q spans the paragraph break", c.Start, c.End, c.Text)
		}
	}
	eqStrings(t, texts(cs), []string{"aa bb", "bb cc", "cc dd", "ee ff", "ff gg", "gg hh"})
}

func TestHugeSingleSentence(t *testing.T) {
	text := strings.Repeat("ก", 500)
	o := Options{MaxUnits: 64}
	cs := mustSplit(t, text, o)
	if want := (500 + 63) / 64; len(cs) != want {
		t.Fatalf("got %d chunks, want %d", len(cs), want)
	}
	var sb strings.Builder
	for _, c := range cs {
		sb.WriteString(c.Text)
	}
	if sb.String() != text {
		t.Fatal("hard-cut pieces do not reassemble the sentence")
	}
}

func TestMaxUnitsOne(t *testing.T) {
	cs := mustSplit(t, "กา", Options{MaxUnits: 1})
	eqStrings(t, texts(cs), []string{"ก", "า"})

	cs = mustSplit(t, "ก ข", Options{MaxUnits: 1})
	eqStrings(t, texts(cs), []string{"ก", "ข"})
}

func TestHardCutAvoidsCombiningMarks(t *testing.T) {
	// Alternating base+mark: an odd MaxUnits would naively cut before "ั".
	text := strings.Repeat("กั", 100)
	cs := mustSplit(t, text, Options{MaxUnits: 5})
	for i, c := range cs {
		if n := utf8.RuneCountInString(c.Text); n != 4 && i != len(cs)-1 {
			t.Fatalf("chunk %d: %d runes, want 4 (cut shifted off the mark)", i, n)
		}
	}
}

func TestUnsplittableMarkRunExceedsMaxUnits(t *testing.T) {
	text := "ก" + strings.Repeat("ั", 10)
	cs := mustSplit(t, text, Options{MaxUnits: 5})
	if len(cs) != 1 || cs[0].Text != text {
		t.Fatalf("got %q, want the whole unsplittable run as one chunk", texts(cs))
	}
}

func TestCustomMeasureWordCount(t *testing.T) {
	words := func(s string) int { return len(strings.Fields(s)) }
	text := "one two three four five six"

	cs := mustSplit(t, text, Options{MaxUnits: 2, Measure: words})
	eqStrings(t, texts(cs), []string{"one two", "three four", "five six"})

	cs = mustSplit(t, text, Options{MaxUnits: 2, OverlapUnits: 1, Measure: words})
	eqStrings(t, texts(cs), []string{"one two", "two three", "three four", "four five", "five six"})
}

func TestCustomSentences(t *testing.T) {
	after := func(par string) []string { return strings.SplitAfter(par, ".") }
	cs := mustSplit(t, "A.B.C", Options{MaxUnits: 2, Sentences: after})
	eqStrings(t, texts(cs), []string{"A.", "B.", "C"})
}

func TestCustomSentencesReceivesTrimmedParagraphs(t *testing.T) {
	var seen []string
	spy := func(par string) []string {
		seen = append(seen, par)
		return []string{par}
	}
	text := "  ยาวเกินหนึ่งชิ้นแน่นอน  \r\nสั้น\n\n  อีกย่อหน้าที่ยาวเกินพอ  "
	mustSplit(t, text, Options{MaxUnits: 10, Sentences: spy})
	if len(seen) == 0 {
		t.Fatal("Sentences never invoked")
	}
	for _, par := range seen {
		if strings.ContainsAny(par, lineBreaks) {
			t.Fatalf("paragraph contains line break: %q", par)
		}
		if strings.TrimSpace(par) != par {
			t.Fatalf("paragraph not trimmed: %q", par)
		}
		if utf8.RuneCountInString(par) <= 10 {
			t.Fatalf("Sentences invoked for a paragraph that fits: %q", par)
		}
	}
}

// Invalid Sentences output (dropped bytes, wrong text, empty, mid-rune cut)
// must fall back to the built-in splitter, byte-for-byte.
func TestCustomSentencesFallback(t *testing.T) {
	text := "คำหนึ่ง คำสอง คำสาม คำสี่ คำห้า"
	o := Options{MaxUnits: 8}
	want, err := Split(text, o)
	if err != nil {
		t.Fatal(err)
	}
	for name, bad := range map[string]func(string) []string{
		"drops whitespace": strings.Fields,
		"wrong text":       func(par string) []string { return []string{"x", par[1:]} },
		"nil":              func(par string) []string { return nil },
		"lost suffix":      func(par string) []string { return []string{par[:len(par)-3]} },
		"mid-rune":         func(par string) []string { return []string{par[:1], par[1:]} },
	} {
		o := Options{MaxUnits: 8, Sentences: bad}
		got := mustSplit(t, text, o)
		if len(got) != len(want) {
			t.Fatalf("%s: got %q, want builtin fallback %q", name, texts(got), texts(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: chunk %d = %+v, want %+v", name, i, got[i], want[i])
			}
		}
	}
}

// The built-in splitter must use the same boundaries as sentence.Split
// (whitespace+newline), just offset-preserving instead of dropping spaces.
func TestBuiltinMatchesSentenceSplit(t *testing.T) {
	data, err := os.ReadFile("../sentence/testdata/sent.txt")
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(1, 2))
	pars := strings.Split(string(data), "\n")
	for i := 0; i < 200; i++ {
		pars = append(pars, genParagraph(r))
	}
	for _, raw := range pars {
		par := strings.TrimSpace(strings.ReplaceAll(raw, "\r", " "))
		if par == "" {
			continue
		}
		var got []string
		prev := 0
		for _, cut := range builtinCuts(par, nil) {
			if seg := strings.TrimSpace(par[prev:cut]); seg != "" {
				got = append(got, seg)
			}
			prev = cut
		}
		eqStrings(t, got, sentence.Split(par))
	}
}

var propOptions = []Options{
	{MaxUnits: 1},
	{MaxUnits: 2, OverlapUnits: 1},
	{MaxUnits: 5, OverlapUnits: 2},
	{MaxUnits: 8},
	{MaxUnits: 17, OverlapUnits: 16},
	{MaxUnits: 64, OverlapUnits: 20},
	{MaxUnits: 9, Measure: func(s string) int { return len(strings.Fields(s)) }},
	{MaxUnits: 4, OverlapUnits: 2, Measure: func(s string) int { return len(strings.Fields(s)) }},
}

func TestPropertyRandomDocs(t *testing.T) {
	for seed := range uint64(40) {
		r := rand.New(rand.NewPCG(seed, 42))
		text := genDoc(r)
		for _, o := range propOptions {
			cs, err := Split(text, o)
			if err != nil {
				t.Fatalf("seed %d: %v", seed, err)
			}
			checkInvariants(t, text, cs, o)
		}
	}
}

func TestPropertyRealCorpus(t *testing.T) {
	data, err := os.ReadFile("../sentence/crf/testdata/crfcut.txt")
	if err != nil {
		t.Fatal(err)
	}
	text := string(data)
	for _, o := range propOptions {
		cs, err := Split(text, o)
		if err != nil {
			t.Fatal(err)
		}
		checkInvariants(t, text, cs, o)
	}
}

// Random Thai/mixed document generator for property tests and fuzz seeds.

const (
	genConsonants = "กขคงจฉชซญดตถทนบปผพฟมยรลวสหอฮ"
	genMarks      = "ัิีึืุู่้๊๋็์"
	genFollow     = "ะาำ"
	genLead       = "เแโใไ"
	genLatin      = "abcdefghijklmnopqrstuvwxyz"
)

var (
	genConsonantsR = []rune(genConsonants)
	genMarksR      = []rune(genMarks)
	genFollowR     = []rune(genFollow)
	genLeadR       = []rune(genLead)
	genSeps        = []string{" ", "  ", "\t", " \t "}
	genBreaks      = []string{"\n", "\r\n", "\n\n", "\r", "\n  \n", "  \n"}
)

func genWord(r *rand.Rand, sb *strings.Builder) {
	if r.IntN(5) == 0 {
		for range 1 + r.IntN(7) {
			sb.WriteByte(genLatin[r.IntN(len(genLatin))])
		}
		return
	}
	for range 1 + r.IntN(4) { // syllables
		if r.IntN(4) == 0 {
			sb.WriteRune(genLeadR[r.IntN(len(genLeadR))])
		}
		sb.WriteRune(genConsonantsR[r.IntN(len(genConsonantsR))])
		for r.IntN(3) == 0 {
			sb.WriteRune(genMarksR[r.IntN(len(genMarksR))])
		}
		if r.IntN(4) == 0 {
			sb.WriteRune(genFollowR[r.IntN(len(genFollowR))])
		}
	}
}

func genParagraph(r *rand.Rand) string {
	var sb strings.Builder
	for p := 1 + r.IntN(6); p > 0; p-- {
		for w := 1 + r.IntN(5); w > 0; w-- {
			genWord(r, &sb)
		}
		if p > 1 {
			sb.WriteString(genSeps[r.IntN(len(genSeps))])
		}
	}
	return sb.String()
}

func genDoc(r *rand.Rand) string {
	var sb strings.Builder
	for p := 1 + r.IntN(20); p > 0; p-- {
		if r.IntN(6) == 0 {
			sb.WriteString("   ")
		}
		sb.WriteString(genParagraph(r))
		if p > 1 || r.IntN(2) == 0 {
			sb.WriteString(genBreaks[r.IntN(len(genBreaks))])
		}
	}
	return sb.String()
}
