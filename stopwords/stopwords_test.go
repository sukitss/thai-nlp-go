package stopwords

import (
	"bufio"
	"os"
	"strings"
	"testing"
)

func readLines(t testing.TB, path string) []string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if w := strings.TrimSpace(sc.Text()); w != "" {
			lines = append(lines, w)
		}
	}
	return lines
}

// TestThaiGolden checks the embedded Thai set matches the cleaned golden
// (normalized + deduped PyThaiNLP thai_stopwords, 1,027 words).
func TestThaiGolden(t *testing.T) {
	gold := readLines(t, "testdata/stopwords_th.golden")
	d := Default()
	if d.Len() != len(gold) {
		t.Fatalf("count mismatch: Default=%d golden=%d", d.Len(), len(gold))
	}
	got := strings.Join(d.Words(), "\n")
	want := strings.Join(gold, "\n") // golden file is already sorted
	if got != want {
		t.Errorf("Default() set differs from golden")
	}
	t.Logf("✅ Thai set = cleaned PyThaiNLP: %d words", d.Len())
}

// TestThaiSetIsClean guards the quality fixes: no BOM/zero-width chars, and no
// decomposed tone-vowel encodings (every word is normalized).
func TestThaiSetIsClean(t *testing.T) {
	const decomposedAM = "ํา" // nikhahit + sara aa (should be composed U+0E33)
	for _, w := range Default().Words() {
		if strings.ContainsRune(w, '\uFEFF') || strings.ContainsRune(w, '\u200b') {
			t.Errorf("stopword contains BOM/zero-width: %q", w)
		}
		if strings.Contains(w, decomposedAM) {
			t.Errorf("stopword has decomposed encoding (should be normalized): %q", w)
		}
	}
}

func TestIsStopword(t *testing.T) {
	d := Default()
	if !d.IsStopword("และ") { // common Thai stopword
		t.Error("และ should be a Thai stopword")
	}
	if d.IsStopword("ภาษาไทย") { // content word
		t.Error("ภาษาไทย should not be a stopword")
	}
	if d.IsStopword("") {
		t.Error("empty string should not be a stopword")
	}
}

// TestAcronymCaseSensitive verifies "it" is a stopword but "IT" (acronym) is not.
func TestAcronymCaseSensitive(t *testing.T) {
	en := English()
	if !en.IsStopword("it") {
		t.Error(`"it" should be an English stopword`)
	}
	if en.IsStopword("IT") {
		t.Error(`"IT" (acronym) should NOT match the stopword "it"`)
	}
}

func TestFilter(t *testing.T) {
	set := Union(Default(), English())
	in := []string{"ผม", "และ", "รัก", "the", "cat", "ภาษาไทย"}
	got := set.Filter(in)
	want := []string{"ผม", "รัก", "cat", "ภาษาไทย"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("Filter = %v, want %v", got, want)
	}
}

// TestFilterUnchanged: Filter returns the input slice when nothing is dropped.
func TestFilterUnchanged(t *testing.T) {
	d := Default()
	in := []string{"ภาษาไทย", "แมว", "หนังสือ"}
	got := d.Filter(in)
	if &got[0] != &in[0] {
		t.Error("Filter should return the input slice unchanged when nothing is dropped")
	}
}

func TestUnionAndNew(t *testing.T) {
	custom := New("อาริน", "  ", "เวธกา")
	if custom.Len() != 2 { // blank ignored
		t.Fatalf("New len = %d, want 2", custom.Len())
	}
	u := Union(Default(), custom)
	if !u.IsStopword("และ") || !u.IsStopword("อาริน") {
		t.Error("Union should contain words from both sets")
	}
}

func TestBuilderDocFreq(t *testing.T) {
	b := NewBuilder()
	// "และ"/"ที่" appear in every doc → stopwords; content words rare
	b.AddDoc([]string{"และ", "ที่", "แมว", "วิ่ง"})
	b.AddDoc([]string{"และ", "ที่", "หมา", "เห่า"})
	b.AddDoc([]string{"และ", "ที่", "นก", "บิน"})
	b.AddDoc([]string{"และ", "ปลา", "ว่าย"}) // "ที่" absent here → 3/4
	if b.Docs() != 4 {
		t.Fatalf("Docs=%d want 4", b.Docs())
	}
	// threshold 1.0 → only words in ALL docs (และ)
	all := b.Build(1.0)
	if !all.IsStopword("และ") || all.IsStopword("ที่") || all.IsStopword("แมว") {
		t.Errorf("Build(1.0) = %v, want only และ", all.Words())
	}
	// threshold 0.7 → และ(4/4) + ที่(3/4)
	most := b.Build(0.7)
	if !most.IsStopword("และ") || !most.IsStopword("ที่") || most.IsStopword("แมว") {
		t.Errorf("Build(0.7) = %v, want และ+ที่", most.Words())
	}
	// dedup within a doc: repeated token counts once
	b2 := NewBuilder()
	b2.AddDoc([]string{"x", "x", "x"})
	b2.AddDoc([]string{"y"})
	if got := b2.Build(1.0); got.IsStopword("x") { // x in 1/2 docs, not all
		t.Errorf("x should not be stopword at 1.0: %v", got.Words())
	}
}

func TestBuilderEmpty(t *testing.T) {
	if NewBuilder().Build(0.5).Len() != 0 {
		t.Error("empty builder should give empty set")
	}
}

// TestBuilderDF: DF exposes per-term document frequency (dups in one doc
// count once), and DocCount matches Docs.
func TestBuilderDF(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"และ", "แมว", "แมว", "แมว"}) // แมว dup → counts once
	b.AddDoc([]string{"และ", "หมา"})
	b.AddDoc([]string{"และ"})
	if got := b.DF("และ"); got != 3 {
		t.Errorf("DF(และ) = %d, want 3", got)
	}
	if got := b.DF("แมว"); got != 1 {
		t.Errorf("DF(แมว) = %d, want 1 (dups in one doc count once)", got)
	}
	if got := b.DF("ไม่มี"); got != 0 {
		t.Errorf("DF(unseen) = %d, want 0", got)
	}
	if b.DocCount() != 3 || b.DocCount() != b.Docs() {
		t.Errorf("DocCount = %d, Docs = %d, want both 3", b.DocCount(), b.Docs())
	}
	if NewBuilder().DocCount() != 0 {
		t.Error("empty builder DocCount should be 0")
	}
}

// TestBuilderTerms: Terms visits every counted term exactly once with its DF,
// and stops early when fn returns false.
func TestBuilderTerms(t *testing.T) {
	b := NewBuilder()
	b.AddDoc([]string{"และ", "แมว"})
	b.AddDoc([]string{"และ", "หมา"})
	got := map[string]int{}
	b.Terms(func(term string, df int) bool {
		if _, dup := got[term]; dup {
			t.Errorf("Terms visited %q twice", term)
		}
		got[term] = df
		return true
	})
	want := map[string]int{"และ": 2, "แมว": 1, "หมา": 1}
	if len(got) != len(want) {
		t.Fatalf("Terms visited %d terms, want %d: %v", len(got), len(want), got)
	}
	for term, df := range want {
		if got[term] != df {
			t.Errorf("Terms df(%q) = %d, want %d", term, got[term], df)
		}
	}
	// early stop
	n := 0
	b.Terms(func(string, int) bool { n++; return false })
	if n != 1 {
		t.Errorf("early stop visited %d terms, want 1", n)
	}
	// empty builder: fn never called
	NewBuilder().Terms(func(string, int) bool {
		t.Error("Terms called fn on empty builder")
		return true
	})
}

func TestCJKKStopwords(t *testing.T) {
	cases := []struct {
		name string
		set  *Set
		stop []string // must be stopwords
		keep []string // must NOT be stopwords (content)
	}{
		{"zh", Chinese(), []string{"的", "了", "是", "我"}, []string{"北京", "大学"}},
		{"ja", Japanese(), []string{"は", "が", "を", "です"}, []string{"日本語", "東京"}},
		{"ko", Korean(), []string{"은", "는", "이", "하다"}, []string{"한국어", "서울"}},
	}
	for _, c := range cases {
		for _, w := range c.stop {
			if !c.set.IsStopword(w) {
				t.Errorf("%s: %q should be a stopword", c.name, w)
			}
		}
		for _, w := range c.keep {
			if c.set.IsStopword(w) {
				t.Errorf("%s: content word %q must NOT be a stopword", c.name, w)
			}
		}
	}
	// header comment must not leak in as a token
	if Chinese().IsStopword("#") || Chinese().IsStopword("# curated function words") {
		t.Error("comment line leaked into set")
	}
}

func TestMultilingualUnion(t *testing.T) {
	m := Multilingual()
	for _, w := range []string{"และ" /*th*/, "the" /*en*/, "的" /*zh*/, "は" /*ja*/, "은" /*ko*/} {
		if !m.IsStopword(w) {
			t.Errorf("Multilingual missing %q", w)
		}
	}
}
