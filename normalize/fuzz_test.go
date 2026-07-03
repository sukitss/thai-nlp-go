package normalize

import "testing"

// FuzzNormalize is a native Go fuzz target (go test -fuzz=FuzzNormalize) that
// guards Normalize, Canonical and Reorder against panics on arbitrary input.
//
// It deliberately does NOT assert idempotence: Normalize is not idempotent, as
// a faithful consequence of PyThaiNLP's rule order (verified byte-for-byte by
// TestGolden). RemoveDangling runs after RemoveDupSpaces (which trims), so
// dropping a dangling mark can expose whitespace only the next pass cleans up:
//
//	Normalize("a ้")     = "a "  -> "a"        (one extra pass)
//	Normalize("ํ\rํ\rํ")  peels one dangling mark per pass ("\r" is not
//	                     space-collapsed), so convergence is unbounded.
//
// Both counterexamples were found by this fuzzer and are kept as seeds in
// testdata/fuzz/FuzzNormalize. Idempotence on realistic inputs is pinned by
// TestRobustNormalize/TestRobustCanonical; byte-parity with PyThaiNLP — the
// package's actual contract — is pinned by TestGolden.
func FuzzNormalize(f *testing.F) {
	for _, s := range []string{
		"", "ฉันรักภาษาไทย", "เเปลก", "นานาาา", "ก    ข",
		"เกา่", "แก้ว้้", "ผมรัก中文abc한국", "๑๒๓ abc",
		"a ้", "ๆ ่", "ํ\rํ\rํ", "แ\xe0 \xe0\xb9 ้", // dangling-mark edge cases
		"\xff\xfe", "\xed\xa0\x80", "\xe0\xb8", "ก\x00ข",
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		_ = Normalize(s)
		_ = Canonical(s)
		_ = Reorder(s)
	})
}
