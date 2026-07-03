package translit

// Robustness regression tests. LK82 used to panic (slice bounds out of range)
// on any input whose cleaned text starts with a non-consonant rune (Thai
// vowels, digits, Latin, spaces, emoji, ...), and the panic reached
// PhoneticKeys, SharesPhoneticKey, SoundsLike and SoundIndex Add/Lookup —
// all of which take untrusted text. These sweeps lock in graceful handling of
// arbitrary input, plus the MaxSoundLen CPU cap on the fuzzy matchers.

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func mustNotPanic(t *testing.T, label string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s panicked: %v", label, r)
		}
	}()
	fn()
}

// TestArbitraryInputNoPanic sweeps every single rune in U+0020–U+0FFF (covers
// ASCII, Latin, all Thai vowels/tones/digits, Lao) plus emoji/astral runes and
// non-consonant-final combos through every text-taking entry point.
func TestArbitraryInputNoPanic(t *testing.T) {
	var inputs []string
	for r := rune(0x0020); r <= 0x0fff; r++ {
		inputs = append(inputs, string(r))
	}
	inputs = append(inputs,
		"😀", "🇹🇭", "𝄞", "🀄", string(rune(0x10ffff)), // emoji / astral plane
		"กา5", "5า", " ๆ", "12า34", "ๆๆ", "ex", // combos ending/starting off-alphabet
	)
	funcs := []struct {
		name string
		fn   func(string)
	}{
		{"LK82", func(s string) { LK82(s) }},
		{"Udom83", func(s string) { Udom83(s) }},
		{"Metasound", func(s string) { Metasound(s, 4) }},
		{"CompleteSoundex", func(s string) { CompleteSoundex(s) }},
		{"PhoneticKeys", func(s string) { PhoneticKeys(s) }},
		{"SoundsLike", func(s string) { SoundsLike(s, "ทดสอบ", 0.8) }},
	}
	fails := 0
	for _, s := range inputs {
		for _, f := range funcs {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s(%q) panicked: %v", f.name, s, r)
						fails++
					}
				}()
				f.fn(s)
			}()
			if fails > 10 {
				t.Fatal("too many panics, aborting sweep")
			}
		}
	}
}

// TestLK82NonConsonantStart locks the (deterministic) outputs for inputs that
// previously panicked — the natural continuation of the algorithm, identical
// to what Python slicing semantics (word[2:] on a short string) would give.
func TestLK82NonConsonantStart(t *testing.T) {
	cases := []struct{ in, want string }{
		{"า", "90000"}, // lone vowel: t2['า']='9'
		{"5", "50000"}, // digit passes through unmapped
		{" ", " 0000"}, // space passes through unmapped
		{"ๆ", ""},      // sign-only input cleans to empty
		{"กา5", "ก9500"},
		{"5า", "า5000"}, // else-branch consumes both runes
		{" ๆ", " 0000"},
	}
	for _, c := range cases {
		if got := LK82(c.in); got != c.want {
			t.Errorf("LK82(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestSoundIndexAdversarial: Add/Lookup with vowel-only, digit, space and
// empty strings must not panic and must stay self-consistent; an over-cap
// query must return nothing, fast.
func TestSoundIndexAdversarial(t *testing.T) {
	idx := NewSoundIndex()
	adversarial := []string{"า", "ู", "5", " ", "", "๑๒๓"}
	for i, s := range adversarial {
		i, s := i, s
		mustNotPanic(t, fmt.Sprintf("Add(%q)", s), func() { idx.Add(fmt.Sprintf("adv%d", i), s) })
	}
	idx.Add("hero", "ทดสอบ")
	for _, s := range adversarial {
		s := s
		mustNotPanic(t, fmt.Sprintf("Lookup(%q)", s), func() { idx.Lookup(s) })
	}
	// a vowel-only spelling is still findable via its LK82 key
	if got := idx.Lookup("า"); len(got) == 0 || got[0] != "adv0" {
		t.Errorf(`Lookup("า") = %v, want [adv0 ...]`, got)
	}
	// normal lookups still work alongside the junk entries
	if got := idx.Lookup("ทดสอบ"); len(got) == 0 || got[0] != "hero" {
		t.Errorf(`Lookup("ทดสอบ") = %v, want [hero ...]`, got)
	}
	// over-cap query: no matches, and the cap must short-circuit the O(n·m) work
	long := strings.Repeat("ก", 10000)
	start := time.Now()
	got := idx.Lookup(long)
	elapsed := time.Since(start)
	if got != nil {
		t.Errorf("Lookup(10k runes) = %v, want nil (over MaxSoundLen)", got)
	}
	if elapsed > 250*time.Millisecond {
		t.Errorf("Lookup(10k runes) took %v, cap should make it near-instant", elapsed)
	}
}

// TestMaxSoundLenBoundary: exactly at the cap still matches; one over never does.
func TestMaxSoundLenBoundary(t *testing.T) {
	at := strings.Repeat("ก", MaxSoundLen)
	over := at + "ก"
	if !SoundsLike(at, at, 0.8) {
		t.Error("SoundsLike at exactly MaxSoundLen runes should still match")
	}
	if SoundsLike(over, over, 0.8) {
		t.Error("SoundsLike over MaxSoundLen runes must return false, even for identical inputs")
	}
	if SoundsLike("ทดสอบ", over, 0.8) {
		t.Error("SoundsLike with one over-cap side must return false")
	}
}

// TestEmptyStringAllExported routes "" through every exported entry point.
func TestEmptyStringAllExported(t *testing.T) {
	mustNotPanic(t, "empty-string sweep", func() {
		_ = CompleteSoundex("")
		_ = CompleteSoundexSimilarity("", "")
		_ = CompleteSoundexSyllables(nil)
		_ = CompleteSoundexSyllables([]string{""})
		_ = Key("")
		_ = LK82("")
		_ = Levenshtein("", "")
		_ = Metasound("", 4)
		_ = PhoneticKeys("")
		_ = SharesPhoneticKey("", "")
		_ = Similarity("", "")
		_ = SoundsLike("", "", 0.8)
		_ = Udom83("")
		idx := NewSoundIndex()
		idx.Add("id", "")
		idx.Alias("", "id")
		_ = idx.Lookup("")
	})
}
