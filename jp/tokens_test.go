package jp

import (
	"testing"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/token"
)

// TestTokensExactOffsets pins byte offsets on hand-computed cases (kanji/kana
// are 3 bytes, ASCII 1).
func TestTokensExactOffsets(t *testing.T) {
	cases := []struct {
		in   string
		want []token.Token
	}{
		{"コンピュータ", []token.Token{
			{Text: "コンピュータ", Start: 0, End: 18},
		}},
		{"水を飲む", []token.Token{
			{Text: "水", Start: 0, End: 3},
			{Text: "を", Start: 3, End: 6},
			{Text: "飲む", Start: 6, End: 12},
		}},
		{"ABC 123 と", []token.Token{ // spaces skipped, offsets exact
			{Text: "ABC", Start: 0, End: 3},
			{Text: "123", Start: 4, End: 7},
			{Text: "と", Start: 8, End: 11},
		}},
		{"", nil},
	}
	for _, c := range cases {
		got := Tokens(c.in)
		if len(got) != len(c.want) {
			t.Errorf("Tokens(%q) = %v, want %v", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("Tokens(%q)[%d] = %+v, want %+v", c.in, i, got[i], c.want[i])
			}
		}
	}
}

// checkTokensAgainst asserts toks carry exactly want's strings in order and
// that every offset slices back to its token, ascending and non-overlapping.
func checkTokensAgainst(t *testing.T, in string, toks []token.Token, want []string, label string) {
	t.Helper()
	if len(toks) != len(want) {
		t.Fatalf("%s: len %d != %d (in=%q)", label, len(toks), len(want), in)
	}
	prev := 0
	for i := range toks {
		if toks[i].Text != want[i] {
			t.Fatalf("%s[%d].Text = %q, want %q (in=%q)", label, i, toks[i].Text, want[i], in)
		}
		if toks[i].Start < prev || toks[i].End > len(in) || toks[i].Start > toks[i].End {
			t.Fatalf("%s[%d] bad range [%d,%d) (in=%q)", label, i, toks[i].Start, toks[i].End, in)
		}
		if in[toks[i].Start:toks[i].End] != toks[i].Text {
			t.Fatalf("%s[%d]: in[%d:%d] != Text %q (in=%q)", label, i, toks[i].Start, toks[i].End, toks[i].Text, in)
		}
		prev = toks[i].End
	}
}

// TestTokensMatchCut: on the reference corpus and mixed cases, Tokens must
// equal Cut and TokensDP must equal CutDP, string for string, with exact
// offsets.
func TestTokensMatchCut(t *testing.T) {
	inputs := readLines(t, "testdata/ja.txt")
	inputs = append(inputs, "日本語を勉強します", "機械学習と深層学習 ABC 123", "半角ｶﾀｶﾅもーOK", "混ぜThaiไทย🎉", "")
	for _, in := range inputs {
		checkTokensAgainst(t, in, Tokens(in), Cut(in), "Tokens")
		checkTokensAgainst(t, in, TokensDP(in), CutDP(in), "TokensDP")
	}
}

// TestRobustTokens: no panic on adversarial input (incl. invalid UTF-8) and
// the slice invariant text[Start:End]==Text always holds — on invalid UTF-8
// Token.Text keeps the raw bytes (the documented difference from Cut, which
// substitutes U+FFFD).
func TestRobustTokens(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			for _, toks := range [][]token.Token{Tokens(c.in), TokensDP(c.in)} {
				prev := 0
				for i, tk := range toks {
					if tk.Start < prev || tk.End > len(c.in) || tk.Start > tk.End {
						t.Fatalf("token %d bad range [%d,%d)", i, tk.Start, tk.End)
					}
					if c.in[tk.Start:tk.End] != tk.Text {
						t.Fatalf("token %d: in[%d:%d] != Text %q", i, tk.Start, tk.End, tk.Text)
					}
					prev = tk.End
				}
			}
			// same boundaries as Cut: rune counts per token must match
			toks, cut := Tokens(c.in), Cut(c.in)
			if len(toks) != len(cut) {
				t.Fatalf("Tokens len %d != Cut len %d", len(toks), len(cut))
			}
			for i := range toks {
				if utf8.RuneCountInString(toks[i].Text) != utf8.RuneCountInString(cut[i]) {
					t.Errorf("token %d rune count differs: %q vs %q", i, toks[i].Text, cut[i])
				}
			}
		})
	}
}

func BenchmarkTokens(b *testing.B) {
	const s = "日本語を勉強します 機械学習と深層学習 コンピュータ ABC 123"
	Tokens("warm")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Tokens(s))
	}
	_ = n
}
