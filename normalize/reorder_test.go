package normalize

import (
	"math/rand"
	"strings"
	"testing"

	"golang.org/x/text/unicode/norm"
)

// runes whose combining class Reorder knows (base + all reorderable/barrier
// marks). Thai has no canonical composition, so NFC over these is pure
// reordering — a perfect oracle for Reorder.
var reorderRunes = []rune{
	0x0E01,                                 // ก base consonant
	0x0E31, 0x0E34, 0x0E35, 0x0E36, 0x0E37, // class 0 above vowels (barriers)
	0x0E47, 0x0E4C, 0x0E4D, 0x0E4E, // class 0 above signs (barriers)
	0x0E3A,         // class 9 phinthu
	0x0E38, 0x0E39, // class 103 below vowels
	0x0E48, 0x0E49, 0x0E4A, 0x0E4B, // class 107 tone marks
	0x0331, 0x0303, // generic below/above (220/230)
}

func seq(rs ...rune) string { return string(rs) }

// TestReorderVsNFC: Reorder must equal Unicode NFC over random Thai mark
// sequences (the authoritative canonical ordering).
func TestReorderVsNFC(t *testing.T) {
	rng := rand.New(rand.NewSource(20260703))
	const N = 200000
	diffs := 0
	var b strings.Builder
	for i := 0; i < N; i++ {
		b.Reset()
		n := rng.Intn(6) + 1
		for k := 0; k < n; k++ {
			b.WriteRune(reorderRunes[rng.Intn(len(reorderRunes))])
		}
		s := b.String()
		if got, want := Reorder(s), norm.NFC.String(s); got != want {
			diffs++
			if diffs <= 10 {
				t.Errorf("Reorder(%q)=%q, NFC=%q", s, got, want)
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ Reorder == NFC over %d random Thai mark sequences", N)
	}
}

// TestReorderUTCCases uses the equivalence/non-equivalence cases from UTC
// L2/18-216 (byte order made explicit via codepoints).
func TestReorderUTCCases(t *testing.T) {
	const ko = 0x0E01 // ก base
	// Case 1: differently-ordered sequences that ARE canonically equivalent
	// (non-zero classes only) must fold to the same bytes.
	equiv := [][2]string{
		{seq(ko, 0x0E3A, 0x0E48), seq(ko, 0x0E48, 0x0E3A)}, // phinthu(9)+tone(107)
		{seq(ko, 0x0E38, 0x0E48), seq(ko, 0x0E48, 0x0E38)}, // below vowel(103)+tone(107)
		{seq(ko, 0x0E3A, 0x0303), seq(ko, 0x0303, 0x0E3A)}, // phinthu(9)+tilde(230)
		{seq(ko, 0x0331, 0x0E48), seq(ko, 0x0E48, 0x0331)}, // macron(220)+tone(107)
	}
	for _, p := range equiv {
		if Reorder(p[0]) != Reorder(p[1]) {
			t.Errorf("expected equivalent: %q / %q -> %q / %q", p[0], p[1], Reorder(p[0]), Reorder(p[1]))
		}
	}
	// Case 2: a class-0 mark makes sequences NON-equivalent (opaque boundary).
	nonEquiv := [][2]string{
		{seq(ko, 0x0E3A, 0x0E34), seq(ko, 0x0E34, 0x0E3A)}, // phinthu(9) + sara i (class 0)
		{seq(ko, 0x0E38, 0x0E34), seq(ko, 0x0E34, 0x0E38)}, // below vowel(103) + sara i (class 0)
	}
	for _, p := range nonEquiv {
		if Reorder(p[0]) == Reorder(p[1]) {
			t.Errorf("expected NON-equivalent (class-0 barrier): %q / %q", p[0], p[1])
		}
	}
}

func TestReorderIdempotent(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	var b strings.Builder
	for i := 0; i < 20000; i++ {
		b.Reset()
		for k := 0; k < rng.Intn(6)+1; k++ {
			b.WriteRune(reorderRunes[rng.Intn(len(reorderRunes))])
		}
		once := Reorder(b.String())
		if once != Reorder(once) {
			t.Fatalf("not idempotent: %q -> %q", b.String(), once)
		}
	}
}

// TestCanonical folds tone-before-vowel with vowel-before-tone (something plain
// Normalize does not guarantee for all encodings).
func TestCanonical(t *testing.T) {
	const ko = 0x0E01
	a := Canonical(seq(ko, 0x0E38, 0x0E48)) // vowel then tone
	b := Canonical(seq(ko, 0x0E48, 0x0E38)) // tone then vowel
	if a != b {
		t.Errorf("Canonical should fold tone/vowel order: %q vs %q", a, b)
	}
}
