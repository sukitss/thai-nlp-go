package multi

import (
	"strings"
	"testing"
)

// adversarialInputs are hostile/degenerate inputs the router must survive:
// empty, invalid UTF-8, NUL bytes, all-whitespace, a combining-mark flood and
// script boundaries in every direction.
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
	{"invalid-inside-han", "我爱\xff自然"},
	{"mixed-scripts", "ผมรัก 中文と日本語 한국어 English 123 ๑๒๓ 🎉"},
	{"all-boundaries", "กa我あ한กa我あ한"},
}

// TestRobustSegment: no panic on adversarial input, and no empty tokens (an
// empty token would corrupt a downstream index).
func TestRobustSegment(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			for _, tok := range Segment(c.in) {
				if tok == "" {
					t.Error("Segment emitted an empty token")
				}
			}
		})
	}
}

// TestRobustAppendBytes: the []byte path must equal the []string path on
// adversarial input too (the basic test only covers well-formed text).
func TestRobustAppendBytes(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			want := strings.Join(Segment(c.in), " ")
			got := string(AppendBytes(nil, c.in, ' '))
			if got != want {
				t.Errorf("AppendBytes = %q, want %q", got, want)
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB of mixed text routes and segments without panic,
// and the two output paths stay in agreement.
func TestRobustLargeInput(t *testing.T) {
	big := strings.Repeat("ผมอ่าน三国志と日本語 hello 한국어 ๑๒๓ ", 15000) // ~1.1 MB
	toks := Segment(big)
	if len(toks) == 0 {
		t.Fatal("Segment returned no tokens for 1MB input")
	}
	want := strings.Join(toks, " ")
	if got := string(AppendBytes(nil, big, ' ')); got != want {
		t.Error("AppendBytes mismatch on 1MB input")
	}
}
