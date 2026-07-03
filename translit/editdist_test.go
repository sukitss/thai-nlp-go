package translit

import "testing"

func TestLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"ก", "", 1},
		{"", "กข", 2},
		{"กา", "กา", 0},
		{"กิตติ", "กิติ", 1}, // one deletion
		{"อุจิวะ", "อุจิฮะ", 1}, // one substitution (ว/ฮ)
		{"abc", "abd", 1},
	}
	for _, c := range cases {
		if got := Levenshtein(c.a, c.b); got != c.want {
			t.Errorf("Levenshtein(%q,%q)=%d, want %d", c.a, c.b, got, c.want)
		}
		if got := Levenshtein(c.b, c.a); got != c.want { // symmetric
			t.Errorf("asymmetric for %q/%q", c.a, c.b)
		}
	}
}

func TestSimilarity(t *testing.T) {
	if Similarity("", "") != 1 {
		t.Error("empty/empty should be 1")
	}
	if Similarity("กา", "กา") != 1 {
		t.Error("identical should be 1")
	}
	// อุจิวะ/อุจิฮะ: 1 edit over 6 runes -> ~0.833
	if s := Similarity("อุจิวะ", "อุจิฮะ"); s < 0.8 || s > 0.9 {
		t.Errorf("Similarity(อุจิวะ,อุจิฮะ)=%.3f, want ~0.83", s)
	}
}
