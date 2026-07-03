package en

import (
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/token"
)

// TestTokensExactOffsets pins byte offsets on hand-computed cases, including
// multi-byte letters (é = 2 bytes, Thai = 3, emoji = 4 and non-word).
func TestTokensExactOffsets(t *testing.T) {
	cases := []struct {
		in   string
		want []token.Token
	}{
		{"hello world", []token.Token{
			{Text: "hello", Start: 0, End: 5},
			{Text: "world", Start: 6, End: 11},
		}},
		{"don't stop", []token.Token{
			{Text: "don't", Start: 0, End: 5},
			{Text: "stop", Start: 6, End: 10},
		}},
		{"café ก1", []token.Token{
			{Text: "café", Start: 0, End: 5}, // é is 2 bytes
			{Text: "ก1", Start: 6, End: 10},  // Thai letter (3 bytes) + digit
		}},
		{"🎉Hi!", []token.Token{
			{Text: "Hi", Start: 4, End: 6}, // emoji (4 bytes) is not a word char
		}},
		{"-lead trail-", []token.Token{
			{Text: "lead", Start: 1, End: 5},
			{Text: "trail", Start: 6, End: 11},
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

// TestTokensMatchCut: on every adversarial input (incl. invalid UTF-8) Tokens
// must yield exactly Cut's strings in order — original casing, NOT CutLower's
// folded form — and offsets that slice back to each token.
func TestTokensMatchCut(t *testing.T) {
	inputs := []string{"The Quick-Brown FOX doesn't Jump", "MixedไทยEnglish你好 123"}
	for _, c := range adversarialInputs {
		inputs = append(inputs, c.in)
	}
	for _, in := range inputs {
		toks := Tokens(in)
		cut := Cut(in)
		if len(toks) != len(cut) {
			t.Fatalf("Tokens len %d != Cut len %d (in=%q)", len(toks), len(cut), in)
		}
		prev := 0
		for i := range toks {
			if toks[i].Text != cut[i] {
				t.Errorf("Tokens[%d].Text = %q, want Cut's %q (in=%q)", i, toks[i].Text, cut[i], in)
			}
			if toks[i].Start < prev || toks[i].End > len(in) || toks[i].Start > toks[i].End {
				t.Errorf("Tokens[%d] bad range [%d,%d) (in=%q)", i, toks[i].Start, toks[i].End, in)
			}
			if in[toks[i].Start:toks[i].End] != toks[i].Text {
				t.Errorf("Tokens[%d]: in[%d:%d] != Text %q (in=%q)", i, toks[i].Start, toks[i].End, toks[i].Text, in)
			}
			prev = toks[i].End
		}
	}
}

// TestTokensOriginalCasing: the documented CutLower exception — Tokens keeps
// raw casing; lowercasing is the caller's job.
func TestTokensOriginalCasing(t *testing.T) {
	toks := Tokens("Hello WORLD")
	if toks[0].Text != "Hello" || toks[1].Text != "WORLD" {
		t.Fatalf("Tokens folded case: %v", toks)
	}
	lower := CutLower("Hello WORLD")
	for i := range toks {
		if strings.ToLower(toks[i].Text) != lower[i] {
			t.Errorf("ToLower(Tokens[%d].Text) = %q, want %q", i, strings.ToLower(toks[i].Text), lower[i])
		}
	}
}

func BenchmarkTokens(b *testing.B) {
	const s = "The quick brown-fox doesn't jump over 42 lazy dogs near the river-bank at dawn."
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Tokens(s))
	}
	_ = n
}
