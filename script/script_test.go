package script

import (
	"strings"
	"testing"
)

func joined(runs []Run) string {
	var parts []string
	for _, r := range runs {
		parts = append(parts, r.Script.String()+":"+r.Text)
	}
	return strings.Join(parts, " | ")
}

func TestSplitByScript(t *testing.T) {
	cases := []struct{ in, want string }{
		{"สวัสดีHello你好", "Thai:สวัสดี | Latin:Hello | Han:你好"},
		{"ตัวอย่างenglishwording", "Thai:ตัวอย่าง | Latin:englishwording"},
		{"abc 123 def", "Latin:abc 123 def"},                     // common (space/digit) stays in the run
		{"ราคา 1,234 บาท", "Thai:ราคา 1,234 บาท"},                // digits/space attach to Thai
		{"เนี่ยลี่ (Nie Li)", "Thai:เนี่ยลี่ ( | Latin:Nie Li)"}, // common attaches to preceding (UAX#24)
		{"한국어test", "Hangul:한국어 | Latin:test"},
		{"", ""},
		{"   ", "Other:   "}, // only common -> one Other run
	}
	for _, c := range cases {
		got := joined(SplitByScript(c.in))
		if got != c.want {
			t.Errorf("SplitByScript(%q)\n  got  %q\n  want %q", c.in, got, c.want)
		}
	}
}

func TestScriptOf(t *testing.T) {
	checks := []struct {
		r rune
		s Script
	}{
		{'ก', Thai}, {'A', Latin}, {'z', Latin}, {'你', Han},
		{'한', Hangul}, {'あ', Kana}, {'ร', Thai},
		{' ', Common}, {'1', Common}, {'!', Common},
	}
	for _, c := range checks {
		if got := ScriptOf(c.r); got != c.s {
			t.Errorf("ScriptOf(%q) = %v, want %v", c.r, got, c.s)
		}
	}
}

// TestNoScriptLost: concatenating run texts reproduces the input.
func TestNoScriptLost(t *testing.T) {
	for _, in := range []string{"สวัสดีHello你好안녕123!", "mixไทยenglish你", "ก"} {
		var b strings.Builder
		for _, r := range SplitByScript(in) {
			b.WriteString(r.Text)
		}
		if b.String() != in {
			t.Errorf("runs lost text: %q -> %q", in, b.String())
		}
	}
}

func BenchmarkSplitByScript(b *testing.B) {
	const s = "ตัวอย่างข้อความภาษาไทยที่ปนenglishwordingและ中文以及한국어 123 บาท"
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(SplitByScript(s))
	}
	_ = n
}
