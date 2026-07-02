package dict

import "testing"

// TestDefaultIsShared verifies Default returns the same instance every time —
// the whole point is loading the dictionary once and sharing it (perf/mem).
func TestDefaultIsShared(t *testing.T) {
	a, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	b, err := Default()
	if err != nil {
		t.Fatalf("Default (2nd): %v", err)
	}
	if a != b {
		t.Fatalf("Default returned different instances: %p vs %p", a, b)
	}
}

// TestDefaultLookup sanity-checks that a known word is found.
func TestDefaultLookup(t *testing.T) {
	d, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	text := []rune("ภาษาไทย")
	lens := d.PrefixLens(text, 0, nil)
	if len(lens) == 0 {
		t.Fatalf("expected some dictionary prefix for %q, got none", string(text))
	}
}

// TestFlatMatchesPointer checks the flat trie and pointer trie agree.
func TestFlatMatchesPointer(t *testing.T) {
	pt := NewTrie()
	for _, w := range []string{"ก", "กา", "กาแฟ", "แฟน"} {
		pt.Add(w)
	}
	d, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	// Different dictionaries, but both must satisfy Prefixer and be usable.
	var _ Prefixer = pt
	var _ Prefixer = d
	text := []rune("กาแฟ")
	got := pt.PrefixLens(text, 0, nil)
	// expect prefixes: "ก"(1), "กา"(2), "กาแฟ"(4)
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 4 {
		t.Fatalf("pointer PrefixLens = %v, want [1 2 4]", got)
	}
}
