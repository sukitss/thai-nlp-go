package tokenize

import (
	"math/rand"
	"strings"
	"testing"
)

// runes spanning Thai consonants/vowels/tones, punctuation, latin, digits
// (incl. Thai digits, category Nd), whitespace and a couple of CJK/Hangul.
var fuzzRunes = []rune(
	"กขคงจฉชซญฎฏดตถทธนบปผพฟภมยรลวศษสหฬอฮ" +
		"ะัาำิีึืุูเแโใไ็่้๊๋์ๆฯ" +
		"abcXYZ-0123456789,. \t\n@#/" +
		"๐๑๒๓๔๕๖๗๘๙" +
		"日한")

func randThai(rng *rand.Rand, maxLen int) string {
	n := rng.Intn(maxLen) + 1
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteRune(fuzzRunes[rng.Intn(len(fuzzRunes))])
	}
	return b.String()
}

// TestFastEqualsOracle asserts the hand-coded fast path equals the regexp2
// oracle (which is verified against PyThaiNLP) over many random strings.
func TestFastEqualsOracle(t *testing.T) {
	fast := defaultSeg(t)
	oracle := newOracle(fast.d)

	n := 200000
	if testing.Short() {
		n = 20000
	}
	rng := rand.New(rand.NewSource(12345))
	diffs := 0
	for i := 0; i < n; i++ {
		s := randThai(rng, 40)
		a := strings.Join(fast.SegmentNoWS(s), "|")
		b := strings.Join(oracle.SegmentNoWS(s), "|")
		if a != b {
			diffs++
			if diffs <= 10 {
				t.Errorf("DIFF input=%q\n  fast  =%s\n  oracle=%s", s, a, b)
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ %d random strings: fast == oracle", n)
	} else {
		t.Fatalf("%d/%d diffs", diffs, n)
	}
}

// FuzzSegment is a native Go fuzz target (go test -fuzz=FuzzSegment) that keeps
// the fast path and the oracle in agreement on arbitrary input.
func FuzzSegment(f *testing.F) {
	seg, err := NewDefault()
	if err != nil {
		f.Fatal(err)
	}
	oracle := newOracle(seg.d)
	for _, s := range []string{"", "ฉันรักภาษาไทย", "abc123 ทดสอบ๑๒๓", "กขคง"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		a := strings.Join(seg.SegmentNoWS(s), "|")
		b := strings.Join(oracle.SegmentNoWS(s), "|")
		if a != b {
			t.Fatalf("mismatch on %q\n  fast  =%s\n  oracle=%s", s, a, b)
		}
	})
}
