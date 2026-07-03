package multi

import (
	"strings"
	"testing"
)

// FuzzMulti is a native Go fuzz target (go test -fuzz=FuzzMulti) that guards
// the whole normalize→detect→route pipeline on arbitrary input: no panic, no
// empty tokens, and AppendBytes always equals Segment joined.
func FuzzMulti(f *testing.F) {
	for _, s := range []string{
		"", "ผมรักabc中文日本語한국", "ฉันรักภาษาไทย", "日本語を勉強",
		"自然语言处理", "한국어 test", "เเปลก 123 ๑๒๓",
		"\xff\xfe", "\xed\xa0\x80", "\xe0\xb8", "ก\x00ข", "我爱\xff自然",
	} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		toks := Segment(s)
		for _, tok := range toks {
			if tok == "" {
				t.Fatalf("Segment(%q) emitted an empty token: %q", s, toks)
			}
		}
		want := strings.Join(toks, " ")
		if got := string(AppendBytes(nil, s, ' ')); got != want {
			t.Fatalf("AppendBytes mismatch on %q: %q vs %q", s, got, want)
		}
	})
}
