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
