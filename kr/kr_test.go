package kr

import (
	"strings"
	"testing"
)

func TestCut(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		// multi-syllable particles stripped → eojeol + stem
		{"서울에서 부산까지", []string{"서울에서", "서울", "부산까지", "부산"}},
		{"집으로 갔다", []string{"집으로", "집", "갔다"}},
		{"", nil},
		// single-syllable particles left attached (ambiguous without a dict)
		{"한국어를 배우다", []string{"한국어를", "배우다"}},
		// non-Korean passes through
		{"AI 기술", []string{"AI", "기술"}},
	}
	for _, c := range cases {
		got := Cut(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("Cut(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestStemRecall: content stem recoverable for multi-syllable particle forms.
func TestStemRecall(t *testing.T) {
	pairs := [][2]string{
		{"서울에서", "서울"}, {"부산까지", "부산"}, {"집으로", "집"},
		{"친구에게", "친구"}, {"학교부터", "학교"}, {"나라보다", "나라"},
	}
	for _, p := range pairs {
		toks := Cut(p[0])
		found := false
		for _, tk := range toks {
			if tk == p[1] {
				found = true
			}
		}
		if !found {
			t.Errorf("Cut(%q)=%v missing stem %q", p[0], toks, p[1])
		}
	}
}

// TestNoOverStrip: content words are never mangled (multi-syllable-only design).
func TestNoOverStrip(t *testing.T) {
	// 사랑/사과/결과 end in a syllable that is a single-syllable particle, but we
	// don't strip those → the content word stays whole.
	for _, w := range []string{"사랑", "사과", "결과"} {
		if got := Cut(w); len(got) != 1 || got[0] != w {
			t.Errorf("Cut(%q) = %v, want [%q]", w, got, w)
		}
	}
}
