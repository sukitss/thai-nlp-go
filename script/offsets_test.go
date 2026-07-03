package script

import (
	"testing"
	"unicode/utf8"
)

// TestRunStartExact pins Run.Start byte offsets on hand-computed cases (Thai
// 3 bytes/rune, Han/Hangul 3, emoji 4, ASCII 1).
func TestRunStartExact(t *testing.T) {
	cases := []struct {
		in     string
		starts []int
	}{
		{"สวัสดีHello你好", []int{0, 18, 23}}, // 6 Thai runes = 18 bytes, "Hello" = 5
		{"เนี่ยลี่ (Nie Li)", []int{0, 26}}, // 8 Thai runes = 24 bytes + " (" → Latin starts at N
		{"한국어test", []int{0, 9}},            // 3 Hangul runes = 9 bytes
		{"ราคา 1,234 บาท", []int{0}},        // one Thai run (common attaches)
		{"   ", []int{0}},    // only-common → one Other run
		{"🎉กa", []int{0, 7}}, // leading emoji (common) attaches to Thai; Latin at 4+3
		{"", nil},            //
	}
	for _, c := range cases {
		runs := SplitByScript(c.in)
		if len(runs) != len(c.starts) {
			t.Errorf("SplitByScript(%q) = %d runs, want %d (%v)", c.in, len(runs), len(c.starts), runs)
			continue
		}
		for i, r := range runs {
			if r.Start != c.starts[i] {
				t.Errorf("SplitByScript(%q) run %d Start = %d, want %d", c.in, i, r.Start, c.starts[i])
			}
		}
	}
}

// TestRunStartSlicesBack: for valid UTF-8 input every Run.Start is correct —
// input[Start:Start+len(Text)] == Text — and runs are contiguous, covering the
// whole input.
func TestRunStartSlicesBack(t *testing.T) {
	inputs := []string{
		"สวัสดีHello你好안녕123!",
		"mixไทยenglish你",
		"ก",
		"ผมรัก 中文と日本語 한국어 English 123 ๑๒๓ 🎉",
		"   \t\n　  ",
		"กa我あ한กa我あ한",
		"",
	}
	for _, in := range inputs {
		prev := 0
		for i, r := range SplitByScript(in) {
			if r.Start != prev {
				t.Errorf("in=%q run %d: Start=%d, want contiguous %d", in, i, r.Start, prev)
			}
			end := r.Start + len(r.Text)
			if end > len(in) || in[r.Start:end] != r.Text {
				t.Errorf("in=%q run %d: in[%d:%d] != Text %q", in, i, r.Start, end, r.Text)
			}
			prev = end
		}
		if prev != len(in) {
			t.Errorf("in=%q: runs cover %d bytes, want %d", in, prev, len(in))
		}
	}
}

// TestRunStartInvalidUTF8: on invalid UTF-8 the documented exception applies —
// Text is sanitized (U+FFFD) so len(Text) may differ from the source bytes,
// but Start must still point at the original byte of each run's first rune.
func TestRunStartInvalidUTF8(t *testing.T) {
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			prev := -1
			for i, r := range SplitByScript(c.in) {
				if r.Start < 0 || r.Start >= len(c.in) && len(c.in) > 0 {
					t.Errorf("run %d: Start=%d out of range (len %d)", i, r.Start, len(c.in))
				}
				if r.Start <= prev {
					t.Errorf("run %d: Start=%d not increasing (prev %d)", i, r.Start, prev)
				}
				prev = r.Start
			}
			// contiguity in decoded runes: each run's Start advances by the
			// encoded length of the previous run's source bytes; verify via a
			// decode walk of the input.
			runs := SplitByScript(c.in)
			ri, b := 0, 0
			for ri < len(runs) && b < len(c.in) {
				if runs[ri].Start != b {
					t.Fatalf("run %d: Start=%d, want %d (decode walk)", ri, runs[ri].Start, b)
				}
				n := utf8.RuneCountInString(runs[ri].Text)
				for k := 0; k < n; k++ {
					_, sz := utf8.DecodeRuneInString(c.in[b:])
					b += sz
				}
				ri++
			}
			if ri != len(runs) || b != len(c.in) {
				t.Fatalf("decode walk ended at run %d/%d byte %d/%d", ri, len(runs), b, len(c.in))
			}
		})
	}
}
