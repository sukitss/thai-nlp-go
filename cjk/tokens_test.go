package cjk

import (
	"testing"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/token"
)

// TestTokensExactOffsets pins byte offsets on hand-computed cases (Han runes
// are 3 bytes, ASCII 1).
func TestTokensExactOffsets(t *testing.T) {
	cases := []struct {
		in   string
		want []token.Token
	}{
		{"北京大学", []token.Token{
			{Text: "北京大学", Start: 0, End: 12},
		}},
		{"AT&T很酷", []token.Token{
			{Text: "AT", Start: 0, End: 2},
			{Text: "&", Start: 2, End: 3},
			{Text: "T", Start: 3, End: 4},
			{Text: "很酷", Start: 4, End: 10},
		}},
		{"深度学习 和 机器学习", []token.Token{ // spaces skipped, offsets exact
			{Text: "深度", Start: 0, End: 6},
			{Text: "学习", Start: 6, End: 12},
			{Text: "和", Start: 13, End: 16},
			{Text: "机器", Start: 17, End: 23},
			{Text: "学习", Start: 23, End: 29},
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
// that every offset slices back to its token, in ascending non-overlapping
// order. Used for Tokens↔Cut and TokensDP↔CutDP equivalence on valid UTF-8.
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
	inputs := readLines(t, "testdata/zh.txt")
	inputs = append(inputs, "我爱自然语言处理", "AT&T很酷 深度学习", "北京大学生前来应聘", "混ぜabc123 🎉ผสม", "")
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
			// same boundaries as Cut/CutDP: rune counts per token must match
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

// lensOnly hides PrefixWeights so the DP path must take its greedy fallback
// (a Prefixer-only custom dictionary).
type lensOnly struct{ d dict.Prefixer }

func (l lensOnly) PrefixLens(text []rune, start int, out []int) []int {
	return l.d.PrefixLens(text, start, out)
}

// TestTokensCustomDicts: with a weight-less Prefixer TokensDP falls back to
// greedy within Han runs; with a plain (zero-weight) Trie it takes the DP
// path — both must equal CutDP and stay offset-exact. Tokens likewise.
func TestTokensCustomDicts(t *testing.T) {
	tr := dict.NewTrie()
	for _, w := range []string{"北京", "大学", "北京大学"} {
		tr.Add(w)
	}
	for _, s := range []*Segmenter{New(tr), New(lensOnly{tr})} {
		for _, in := range []string{"北京大学生", "我爱北京大学 abc"} {
			checkTokensAgainst(t, in, s.TokensDP(in), s.CutDP(in), "TokensDP(custom)")
			checkTokensAgainst(t, in, s.Tokens(in), s.Cut(in), "Tokens(custom)")
		}
	}
}

func BenchmarkTokens(b *testing.B) {
	const s = "我爱自然语言处理 北京大学生前来应聘 AT&T很酷 深度学习和机器学习"
	Tokens("warm")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Tokens(s))
	}
	_ = n
}
