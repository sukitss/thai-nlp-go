package tokenize

import (
	"strings"
	"testing"
)

func TestNGramSplit(t *testing.T) {
	g := NewNGram(2)
	got := g.Split("กขค")
	want := []string{"กข", "ขค"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("Split = %v, want %v", got, want)
	}
}

func TestNGramWhitespaceSeparates(t *testing.T) {
	g := NewNGram(2)
	// n-grams never span the space; "ก" run is shorter than n → emitted whole.
	got := g.Split("กขค ก")
	want := []string{"กข", "ขค", "ก"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("Split = %v, want %v", got, want)
	}
}

func TestNGramAppendBytesMatchesSplit(t *testing.T) {
	g := NewNGram(3)
	for _, s := range []string{"ฉันรักภาษาไทย", "abc123 ทดสอบ", "ก"} {
		want := strings.Join(g.Split(s), " ")
		got := string(g.AppendBytes(nil, s, ' '))
		if got != want {
			t.Fatalf("AppendBytes(%q)=%q, want %q", s, got, want)
		}
	}
}

func TestNGramShortRun(t *testing.T) {
	g := NewNGram(5)
	got := g.Split("กขค") // shorter than n → whole
	if len(got) != 1 || got[0] != "กขค" {
		t.Fatalf("short run = %v, want [กขค]", got)
	}
}
