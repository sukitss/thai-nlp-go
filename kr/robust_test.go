package kr

import (
	"strings"
	"testing"
	"unicode"
)

// adversarialInputs are hostile/degenerate inputs Cut must survive.
var adversarialInputs = []struct {
	name string
	in   string
}{
	{"empty", ""},
	{"invalid-utf8", "\xff\xfe"},
	{"lone-surrogate", "\xed\xa0\x80"},
	{"truncated-thai", "\xe0\xb8"},
	{"nul-bytes", "서울\x00에서 a\x00b"},
	{"all-whitespace", "   \t\n　  "},
	{"mark-flood", strings.Repeat("่", 10000)}, // 10k combining tone marks
	{"particle-only", "에서 으로부터 까지"},
	{"mixed-scripts", "ผมรัก 中文と日本語 한국어 English 123 ๑๒๓ 🎉"},
}

// TestRobustCut: no panic, and every token is non-empty, whitespace-free and
// unique (Cut documents de-duplication within a result).
func TestRobustCut(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			seen := map[string]bool{}
			for _, tok := range Cut(c.in) {
				if tok == "" {
					t.Error("Cut emitted an empty token")
				}
				if strings.IndexFunc(tok, unicode.IsSpace) >= 0 {
					t.Errorf("Cut token contains whitespace: %q", tok)
				}
				if seen[tok] {
					t.Errorf("Cut emitted duplicate token %q", tok)
				}
				seen[tok] = true
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB tokenizes without panic.
func TestRobustLargeInput(t *testing.T) {
	big := strings.Repeat("서울에서 부산까지 기차로 갑니다 ABC 123 ", 22000) // ~1.1 MB
	if len(Cut(big)) == 0 {
		t.Error("Cut returned no tokens for 1MB input")
	}
}
