package kr

import (
	"testing"

	"github.com/sukitss/thai-nlp-go/token"
)

// TestTokensExactOffsets pins byte offsets on hand-computed cases (Hangul
// syllables are 3 bytes, ASCII 1). Stems cover exactly their own bytes — a
// byte-prefix of the surface eojeol.
func TestTokensExactOffsets(t *testing.T) {
	cases := []struct {
		in   string
		want []token.Token
	}{
		{"서울에서 부산까지", []token.Token{
			{Text: "서울에서", Start: 0, End: 12},
			{Text: "서울", Start: 0, End: 6}, // stem = prefix of 서울에서
			{Text: "부산까지", Start: 13, End: 25},
			{Text: "부산", Start: 13, End: 19},
		}},
		{"집으로 갔다", []token.Token{
			{Text: "집으로", Start: 0, End: 9},
			{Text: "집", Start: 0, End: 3},
			{Text: "갔다", Start: 10, End: 16},
		}},
		{"AI 기술", []token.Token{
			{Text: "AI", Start: 0, End: 2},
			{Text: "기술", Start: 3, End: 9},
		}},
		// de-dup: 서울 appears as its own eojeol first; the later stem of
		// 서울에서 is dropped, keeping the FIRST occurrence's offsets.
		{"서울 서울에서", []token.Token{
			{Text: "서울", Start: 0, End: 6},
			{Text: "서울에서", Start: 7, End: 19},
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

// TestTokensMatchCut: on adversarial input too (incl. invalid UTF-8), Tokens
// must yield exactly Cut's strings in order, every offset must slice back to
// its token (stems included — the byte-prefix guarantee), and End must never
// exceed the input length.
func TestTokensMatchCut(t *testing.T) {
	inputs := []string{
		"서울에서 부산까지 기차로 갑니다",
		"친구에게 학교부터 나라보다",
		"한국어를 배우다 ABC 123 ผสมไทย🎉",
	}
	for _, c := range adversarialInputs {
		inputs = append(inputs, c.in)
	}
	for _, in := range inputs {
		toks := Tokens(in)
		cut := Cut(in)
		if len(toks) != len(cut) {
			t.Fatalf("Tokens len %d != Cut len %d (in=%q)", len(toks), len(cut), in)
		}
		for i := range toks {
			if toks[i].Text != cut[i] {
				t.Errorf("Tokens[%d].Text = %q, want Cut's %q (in=%q)", i, toks[i].Text, cut[i], in)
			}
			if toks[i].Start < 0 || toks[i].End > len(in) || toks[i].Start > toks[i].End {
				t.Errorf("Tokens[%d] bad range [%d,%d) (in=%q)", i, toks[i].Start, toks[i].End, in)
			}
			if in[toks[i].Start:toks[i].End] != toks[i].Text {
				t.Errorf("Tokens[%d]: in[%d:%d] != Text %q (in=%q)", i, toks[i].Start, toks[i].End, toks[i].Text, in)
			}
		}
	}
}

// TestStemIsPrefix verifies the assumption the stem offsets rely on: stripJosa
// only ever removes a byte-suffix, so the stem is a byte-prefix of its eojeol.
func TestStemIsPrefix(t *testing.T) {
	for _, f := range []string{"서울에서", "부산까지", "집으로", "친구에게", "학교부터", "나라보다", "그것조차", "이것마저"} {
		stem := stripJosa(f)
		if stem == "" {
			t.Errorf("stripJosa(%q) found no stem", f)
			continue
		}
		if f[:len(stem)] != stem {
			t.Errorf("stem %q is not a byte-prefix of %q", stem, f)
		}
	}
}

func BenchmarkTokens(b *testing.B) {
	const s = "서울에서 부산까지 기차로 갑니다 ABC 123"
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(Tokens(s))
	}
	_ = n
}
