package en

import (
	"strings"
	"testing"
	"unicode"
)

// adversarialInputs are hostile/degenerate inputs Cut/CutLower must survive.
var adversarialInputs = []struct {
	name string
	in   string
}{
	{"empty", ""},
	{"invalid-utf8", "\xff\xfe"},
	{"lone-surrogate", "\xed\xa0\x80"},
	{"truncated-thai", "\xe0\xb8"},
	{"nul-bytes", "a\x00b ก\x00ข"},
	{"all-whitespace", "   \t\n　  "},
	{"mark-flood", strings.Repeat("่", 10000)}, // 10k combining tone marks
	{"punct-only", "!!! --- ''' ..."},
	{"dangling-connectors", "-a- 'b' c- -d '-'"},
	{"mixed-scripts", "ผมรัก 中文と日本語 한국어 English 123 ๑๒๓ 🎉"},
}

// TestRobustCut: no panic, and every token is non-empty with no whitespace
// inside (the unigram indexing contract).
func TestRobustCut(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			for _, tok := range Cut(c.in) {
				if tok == "" {
					t.Error("Cut emitted an empty token")
				}
				if strings.IndexFunc(tok, unicode.IsSpace) >= 0 {
					t.Errorf("Cut token contains whitespace: %q", tok)
				}
			}
		})
	}
}

// TestRobustCutLower: CutLower must be exactly Cut with tokens lowercased.
func TestRobustCutLower(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			cut := Cut(c.in)
			lower := CutLower(c.in)
			if len(cut) != len(lower) {
				t.Fatalf("CutLower len %d != Cut len %d", len(lower), len(cut))
			}
			for i := range cut {
				if lower[i] != strings.ToLower(cut[i]) {
					t.Errorf("CutLower[%d] = %q, want %q", i, lower[i], strings.ToLower(cut[i]))
				}
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB tokenizes without panic.
func TestRobustLargeInput(t *testing.T) {
	big := strings.Repeat("The quick brown-fox doesn't jump over 42 lazy dogs. ", 21000) // ~1.1 MB
	if len(Cut(big)) == 0 {
		t.Error("Cut returned no tokens for 1MB input")
	}
}
