package jp

import (
	"strings"
	"testing"
	"unicode"
)

// adversarialInputs are hostile/degenerate inputs Cut/CutDP must survive.
var adversarialInputs = []struct {
	name string
	in   string
}{
	{"empty", ""},
	{"invalid-utf8", "\xff\xfe"},
	{"lone-surrogate", "\xed\xa0\x80"},
	{"truncated-thai", "\xe0\xb8"},
	{"nul-bytes", "水\x00を a\x00b"},
	{"all-whitespace", "   \t\n　  "},
	{"mark-flood", strings.Repeat("่", 10000)}, // 10k combining tone marks
	{"invalid-inside-jp", "水を\xff飲む"},
	{"mixed-scripts", "ผมรัก 中文と日本語 한국어 English 123 ๑๒๓ 🎉"},
}

// stripWS drops all Unicode whitespace after UTF-8 sanitization — Cut/CutDP
// drop whitespace and replace invalid bytes with U+FFFD (via []rune), so their
// joined output must equal this exactly.
func stripWS(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, string([]rune(s)))
}

// TestRobustCut: no panic on adversarial input, and the coverage invariant
// holds — concatenating the tokens reproduces the input minus whitespace (no
// lost or invented runes), extending TestCutDPValid to hostile input.
func TestRobustCut(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			want := stripWS(c.in)
			if got := strings.Join(Cut(c.in), ""); got != want {
				t.Errorf("Cut concat = %q, want %q", got, want)
			}
			if got := strings.Join(CutDP(c.in), ""); got != want {
				t.Errorf("CutDP concat = %q, want %q", got, want)
			}
		})
	}
}

// TestRobustAppendBytes: the []byte path must equal the []string path on
// adversarial input too.
func TestRobustAppendBytes(t *testing.T) {
	seg, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			want := strings.Join(seg.Cut(c.in), " ")
			if got := string(seg.AppendBytes(nil, c.in, ' ')); got != want {
				t.Errorf("AppendBytes = %q, want %q", got, want)
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB segments without panic and keeps coverage.
func TestRobustLargeInput(t *testing.T) {
	big := strings.Repeat("日本語を勉強します 東京都に住む 機械学習と自然言語処理 ", 12000) // ~1 MB
	want := stripWS(big)
	if got := strings.Join(Cut(big), ""); got != want {
		t.Error("Cut concat does not reconstruct 1MB input")
	}
	if got := strings.Join(CutDP(big), ""); got != want {
		t.Error("CutDP concat does not reconstruct 1MB input")
	}
}
