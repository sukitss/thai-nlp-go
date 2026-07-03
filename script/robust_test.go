package script

import (
	"strings"
	"testing"
)

// adversarialInputs are hostile/degenerate inputs SplitByScript must survive.
var adversarialInputs = []struct {
	name string
	in   string
}{
	{"empty", ""},
	{"invalid-utf8", "\xff\xfe"},
	{"lone-surrogate", "\xed\xa0\x80"},
	{"truncated-thai", "\xe0\xb8"},
	{"nul-bytes", "ก\x00ข a\x00b"},
	{"all-whitespace", "   \t\n　  "},
	{"mark-flood", strings.Repeat("่", 10000)}, // 10k combining tone marks
	{"invalid-inside-thai", "ฉันรัก\xffภาษา"},
	{"mixed-scripts", "ผมรัก 中文と日本語 한국어 English 123 ๑๒๓ 🎉"},
	{"boundary-per-rune", "กa我あ한กa我あ한"},
}

// TestRobustSplitByScript: no panic, no empty runs, and no text lost —
// concatenating the runs reproduces the input after UTF-8 sanitization
// (invalid bytes become U+FFFD, i.e. string([]rune(in)); for valid UTF-8
// that is the input byte-for-byte, as TestNoScriptLost already pins).
func TestRobustSplitByScript(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			var b strings.Builder
			for _, r := range SplitByScript(c.in) {
				if r.Text == "" {
					t.Error("SplitByScript emitted an empty run")
				}
				b.WriteString(r.Text)
			}
			if want := string([]rune(c.in)); b.String() != want {
				t.Errorf("runs lost text: got %q, want %q", b.String(), want)
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB splits without panic and loses nothing.
func TestRobustLargeInput(t *testing.T) {
	big := strings.Repeat("ตัวอย่างไทยenglish你好안녕あいう 123! ", 15000) // ~1.1 MB
	var b strings.Builder
	for _, r := range SplitByScript(big) {
		b.WriteString(r.Text)
	}
	if b.String() != big {
		t.Error("runs lost text on 1MB input")
	}
}
