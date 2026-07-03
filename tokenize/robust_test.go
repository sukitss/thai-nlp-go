package tokenize

import (
	"strings"
	"testing"
)

// adversarialInputs are hostile/degenerate inputs every public entry point must
// survive: empty, invalid UTF-8 (stray bytes, lone surrogate, truncated Thai
// sequence), NUL bytes, all-whitespace, a combining-mark flood and mixed scripts.
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
}

// TestRobustSegment: no panic on adversarial input, and the reconstruction
// invariant holds — concatenating Segment (whitespace kept) reproduces the
// input after UTF-8 sanitization (invalid bytes become U+FFFD, exactly
// string([]rune(in))); for valid UTF-8 that is the input byte-for-byte.
func TestRobustSegment(t *testing.T) {
	seg := defaultSeg(t)
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			got := strings.Join(seg.Segment(c.in), "")
			if want := string([]rune(c.in)); got != want {
				t.Errorf("Segment concat = %q, want %q", got, want)
			}
		})
	}
}

// TestRobustSegmentNoWS: no panic, and no empty/space-only tokens on
// adversarial input. (PyThaiNLP's keep_whitespace=False strips ASCII spaces
// only — a token like "\n" or "　" is faithful upstream behavior — so the
// invariant is strip(" "), not full Unicode TrimSpace.)
func TestRobustSegmentNoWS(t *testing.T) {
	seg := defaultSeg(t)
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			for _, tok := range seg.SegmentNoWS(c.in) {
				if strings.Trim(tok, " ") == "" {
					t.Errorf("SegmentNoWS emitted blank token %q", tok)
				}
			}
		})
	}
}

// TestRobustAppendBytes: the []byte path must equal the []string path on
// adversarial input too (the corpus test only covers well-formed text).
func TestRobustAppendBytes(t *testing.T) {
	seg := defaultSeg(t)
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			want := strings.Join(seg.SegmentNoWS(c.in), " ")
			got := string(seg.AppendBytes(nil, c.in, ' '))
			if got != want {
				t.Errorf("AppendBytes = %q, want %q", got, want)
			}
		})
	}
}

// TestRobustSession: an overlay Session must uphold the same invariants as the
// base segmenter on adversarial input.
func TestRobustSession(t *testing.T) {
	sess := defaultSeg(t).Session([]string{"อาริน", "เวธกา"})
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			got := strings.Join(sess.Segment(c.in), "")
			if want := string([]rune(c.in)); got != want {
				t.Errorf("Session Segment concat = %q, want %q", got, want)
			}
			want := strings.Join(sess.SegmentNoWS(c.in), " ")
			if got := string(sess.AppendBytes(nil, c.in, ' ')); got != want {
				t.Errorf("Session AppendBytes = %q, want %q", got, want)
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB of mixed text segments without panic and keeps
// the reconstruction invariant (guards against super-linear blowups: this test
// completes in well under a minute or times out CI).
func TestRobustLargeInput(t *testing.T) {
	seg := defaultSeg(t)
	big := strings.Repeat("ฉันรักภาษาไทยมาก abc 123 ๑๒๓ ", 20000) // ~1.1 MB
	if got := strings.Join(seg.Segment(big), ""); got != big {
		t.Error("Segment concat does not reconstruct 1MB input")
	}
	want := strings.Join(seg.SegmentNoWS(big), " ")
	if got := string(seg.AppendBytes(nil, big, ' ')); got != want {
		t.Error("AppendBytes mismatch on 1MB input")
	}
}
