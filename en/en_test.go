package en

import (
	"strings"
	"testing"
)

func TestCut(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"hello world", []string{"hello", "world"}},
		{"don't stop", []string{"don't", "stop"}},         // internal apostrophe kept
		{"world-wide web", []string{"world-wide", "web"}}, // internal hyphen kept
		{"foo, bar. baz!", []string{"foo", "bar", "baz"}}, // punctuation dropped
		{"abc123 x2", []string{"abc123", "x2"}},           // digits are word chars
		{"  spaced   out  ", []string{"spaced", "out"}},
		{"-lead trail-", []string{"lead", "trail"}}, // edge hyphens dropped
		{"", nil},
	}
	for _, c := range cases {
		got := Cut(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("Cut(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestCutLower(t *testing.T) {
	got := CutLower("Hello WORLD Don'T")
	want := []string{"hello", "world", "don't"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("CutLower = %v, want %v", got, want)
	}
}
