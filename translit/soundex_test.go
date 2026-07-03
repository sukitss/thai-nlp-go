package translit

import (
	"strings"
	"testing"
)

func soundexGolden(t *testing.T, goldFile string, fn func(string) string) {
	in := readLines(t, "testdata/soundex.txt")
	gold := readLines(t, goldFile)
	if len(in) != len(gold) {
		t.Fatalf("count mismatch: in=%d gold=%d", len(in), len(gold))
	}
	diffs := 0
	for i := range in {
		if got := fn(in[i]); got != gold[i] {
			diffs++
			if diffs <= 10 {
				t.Errorf("%q -> %q, want %q", in[i], got, gold[i])
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ %s: %d words match PyThaiNLP", goldFile, len(in))
	}
}

func TestUdom83Golden(t *testing.T) { soundexGolden(t, "testdata/udom83.golden", Udom83) }
func TestLK82Golden(t *testing.T)   { soundexGolden(t, "testdata/lk82.golden", LK82) }

// TestSoundexVariantsAgree: for obvious spelling variants at least one phonetic
// key should collapse them (the keys have different strengths).
func TestSoundexVariantsAgree(t *testing.T) {
	pairs := [][2]string{{"ทองดี", "ทองดา"}, {"บ้าน", "บาน"}, {"รัก", "ลัก"}}
	for _, p := range pairs {
		agree := Udom83(p[0]) == Udom83(p[1]) ||
			LK82(p[0]) == LK82(p[1]) ||
			Key(p[0]) == Key(p[1]) // Metasound
		if !agree {
			t.Errorf("no key collapses variant %q/%q", p[0], p[1])
		}
	}
}

func TestSoundexEmpty(t *testing.T) {
	if Udom83("") != "" || LK82("") != "" {
		t.Error("empty input should give empty code")
	}
	// codes are at most 7 (udom83) / 5 (lk82) runes
	for _, w := range []string{"ก", "ประเทศไทย", "สวัสดีครับ"} {
		if n := len([]rune(Udom83(w))); n > 7 {
			t.Errorf("Udom83(%q) len %d > 7", w, n)
		}
		if n := len([]rune(LK82(w))); n > 5 {
			t.Errorf("LK82(%q) len %d > 5", w, n)
		}
	}
	_ = strings.TrimSpace
}
