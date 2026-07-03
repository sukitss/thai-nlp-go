package normalize

import (
	"strings"
	"testing"
)

// adversarialInputs are hostile/degenerate inputs the normalizer must survive:
// empty, invalid UTF-8, NUL bytes, all-whitespace, a combining-mark flood and
// mixed scripts.
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

// TestRobustNormalize: no panic on adversarial input, and Normalize is
// idempotent on these pinned inputs. (One-pass idempotence is NOT universal —
// see FuzzNormalize for the dangling-mark-after-space counterexample — but it
// holds on all of these, so a regression here must be reviewed.)
func TestRobustNormalize(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			n1 := Normalize(c.in)
			if n2 := Normalize(n1); n2 != n1 {
				t.Errorf("Normalize not idempotent: %q -> %q -> %q", c.in, n1, n2)
			}
		})
	}
}

// TestRobustCanonical: same for Canonical and Reorder.
func TestRobustCanonical(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			c1 := Canonical(c.in)
			if c2 := Canonical(c1); c2 != c1 {
				t.Errorf("Canonical not idempotent: %q -> %q -> %q", c.in, c1, c2)
			}
			r1 := Reorder(c.in)
			if r2 := Reorder(r1); r2 != r1 {
				t.Errorf("Reorder not idempotent: %q -> %q -> %q", c.in, r1, r2)
			}
		})
	}
}

// TestRobustLargeInput: ~1 MB normalizes without panic and stays idempotent.
func TestRobustLargeInput(t *testing.T) {
	big := strings.Repeat("เเปลก นานาาา ก    ข ฉันรักภาษาไทย abc 123 ", 10000) // ~1 MB
	n1 := Normalize(big)
	if n2 := Normalize(n1); n2 != n1 {
		t.Error("Normalize not idempotent on 1MB input")
	}
	c1 := Canonical(big)
	if c2 := Canonical(c1); c2 != c1 {
		t.Error("Canonical not idempotent on 1MB input")
	}
}
