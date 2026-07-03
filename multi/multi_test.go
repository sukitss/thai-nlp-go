package multi

import (
	"strings"
	"testing"
)

func TestSegmentRouting(t *testing.T) {
	// Thai + Latin + Chinese + Japanese(kana) + Korean, one call
	got := Segment("ผมรักabc中文日本語한국")
	joined := strings.Join(got, "|")
	// spot-check that each language produced sensible tokens
	for _, want := range []string{"ผม", "รัก", "abc", "한국"} {
		if !strings.Contains("|"+joined+"|", "|"+want+"|") {
			t.Errorf("Segment missing %q in %v", want, got)
		}
	}
}

func TestSegmentThai(t *testing.T) {
	got := Segment("ฉันรักภาษาไทยมาก")
	want := []string{"ฉัน", "รัก", "ภาษาไทย", "มาก"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("Segment(thai) = %v, want %v", got, want)
	}
}

func TestJapaneseVsChineseRouting(t *testing.T) {
	// kana present → Japanese tokenizer (私 は 日本語 ...)
	jpToks := Segment("日本語を勉強")
	if !contains(jpToks, "を") { // を is a Japanese particle the jp path keeps
		t.Errorf("expected Japanese routing for kana text, got %v", jpToks)
	}
	// pure Han, no kana → Chinese by default
	cnToks := Segment("自然语言处理")
	if len(cnToks) == 0 {
		t.Error("pure-Han should route to Chinese and produce tokens")
	}
}

func TestAppendBytesMatchesSegment(t *testing.T) {
	for _, s := range []string{"ผมรักabc中文", "日本語テスト", "한국어 test", ""} {
		want := strings.Join(Segment(s), " ")
		got := string(AppendBytes(nil, s, ' '))
		if got != want {
			t.Errorf("AppendBytes(%q)=%q want %q", s, got, want)
		}
	}
}

func TestEmpty(t *testing.T) {
	if Segment("") != nil {
		t.Error("Segment(\"\") should be nil")
	}
}

func contains(ss []string, x string) bool {
	for _, s := range ss {
		if s == x {
			return true
		}
	}
	return false
}
