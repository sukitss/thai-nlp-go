package normalize

import (
	"testing"

	"golang.org/x/text/unicode/norm"
)

func TestFoldWidth(t *testing.T) {
	cases := map[string]string{
		"Ａ":   "A", // full-width Latin → half
		"３":   "3", // full-width digit → half
		"　":   " ", // ideographic space → space
		"ＡＢＣ": "ABC",
		"ｶ":   "カ", // half-width kana → full
		"A":   "A", // already narrow: unchanged
	}
	for in, want := range cases {
		if got := FoldWidth(in); got != want {
			t.Errorf("FoldWidth(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNFCComposesKorean(t *testing.T) {
	composed := "가"
	decomposed := norm.NFD.String(composed)
	if decomposed == composed {
		t.Skip("no NFD form to test")
	}
	if NFC(decomposed) != composed {
		t.Errorf("NFC(NFD(가)) = %q, want %q", NFC(decomposed), composed)
	}
}

// The measurable win: variant forms that used to miss now unify. Each pair is a
// term as it might be indexed vs as it might be queried.
func TestFoldForIndexUnifiesVariants(t *testing.T) {
	pairs := [][2]string{
		{"ＡＰＩ", "API"},                  // full-width vs ASCII
		{"２０２４", "2024"},                // full-width digits
		{"ｶﾀｶﾅ", "カタカナ"},                // half-width kana vs full
		{norm.NFD.String("한국어"), "한국어"}, // NFD vs NFC Korean
		{"ＡＢ　ＣＤ", "AB CD"},              // width + ideographic space
	}
	unified := 0
	for _, p := range pairs {
		a, b := FoldForIndex(p[0]), FoldForIndex(p[1])
		if a == b {
			unified++
		} else {
			t.Errorf("FoldForIndex mismatch: %q→%q vs %q→%q", p[0], a, p[1], b)
		}
	}
	t.Logf("unified %d/%d variant pairs (were distinct before fold)", unified, len(pairs))
}

// simplified↔traditional and hiragana↔katakana are deliberately NOT folded.
func TestFoldForIndexKeepsMeaningfulDistinctions(t *testing.T) {
	if FoldForIndex("繁體") == FoldForIndex("简体") {
		t.Error("simplified/traditional must stay distinct (needs OpenCC table, out of scope)")
	}
	if FoldForIndex("が") == FoldForIndex("ガ") {
		t.Error("hiragana/katakana must stay distinct (different meaning)")
	}
}

// Latin/ASCII text must pass through untouched (fast path).
func TestFoldForIndexASCIIUnchanged(t *testing.T) {
	for _, s := range []string{"hello world", "BM25 index", "user@example.com", ""} {
		if got := FoldForIndex(s); got != s {
			t.Errorf("FoldForIndex(%q) = %q, want unchanged", s, got)
		}
	}
}

func BenchmarkFoldForIndexASCII(b *testing.B) {
	s := "the quick brown fox BM25"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FoldForIndex(s)
	}
}

func BenchmarkFoldForIndexCJK(b *testing.B) {
	s := "ＡＰＩ　２０２４ ｶﾀｶﾅ 北京大学"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = FoldForIndex(s)
	}
}
