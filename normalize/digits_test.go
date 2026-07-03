package normalize

import (
	"strings"
	"testing"
)

func TestDigitsToArabic(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"all-ten-digits", "๐๑๒๓๔๕๖๗๘๙", "0123456789"},
		{"empty", "", ""},
		{"ascii-only", "chapter 5, page 10", "chapter 5, page 10"},
		{"thai-no-digits", "บทที่ห้า หน้าสิบ", "บทที่ห้า หน้าสิบ"},
		{"mixed", "บทที่ ๕ หน้า ๑๐/25", "บทที่ 5 หน้า 10/25"},
		{"digit-at-start", "๙ ชีวิต", "9 ชีวิต"},
		{"digit-at-end", "ระดับ ๗", "ระดับ 7"},
		{"adjacent-digits-in-word", "ปี๒๕๖๗พอดี", "ปี2567พอดี"},
		{"arabic-untouched", "๑2๓4", "1234"},
		{"cjk-untouched", "第๑章 中文と日本語", "第1章 中文と日本語"},
		// U+0E4F ๏ and U+0E5A ๚ flank the digit block; they must not fold.
		{"block-neighbors", "๏๐๙๚", "๏09๚"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DigitsToArabic(c.in); got != c.want {
				t.Errorf("DigitsToArabic(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestDigitsToArabicIdempotent: folding twice equals folding once (the output
// never contains a Thai digit).
func TestDigitsToArabicIdempotent(t *testing.T) {
	for _, s := range []string{"๐๑๒๓๔๕๖๗๘๙", "บทที่ ๕", "abc", "ก\x00ข๕"} {
		once := DigitsToArabic(s)
		if twice := DigitsToArabic(once); twice != once {
			t.Errorf("not idempotent on %q: %q -> %q", s, once, twice)
		}
	}
}

// TestDigitsToArabicZeroAlloc: input without a Thai digit is returned as-is
// with zero allocation — the fast path an index pipeline hits on most text.
func TestDigitsToArabicZeroAlloc(t *testing.T) {
	inputs := []string{
		"plain ascii text 0123456789",
		"ฉันรักภาษาไทยมาก ไม่มีเลขไทย",
		"中文と日本語 한국어 mixed",
		strings.Repeat("ทดสอบ abc ", 100),
	}
	for _, s := range inputs {
		s := s
		if got := DigitsToArabic(s); got != s {
			t.Fatalf("DigitsToArabic changed digit-free input %q -> %q", s, got)
		}
		if n := testing.AllocsPerRun(100, func() { _ = DigitsToArabic(s) }); n != 0 {
			t.Errorf("DigitsToArabic(%q...) allocates %v times on the no-digit path, want 0", s[:10], n)
		}
	}
}

// TestDigitsToArabicInvalidUTF8: hostile bytes must not panic, non-digit bytes
// (including invalid sequences) pass through verbatim, and Thai digits still
// fold next to them. "\xe0\xb9" is the nastiest case: a truncated prefix of a
// Thai digit's own encoding (E0 B9 9x).
func TestDigitsToArabicInvalidUTF8(t *testing.T) {
	cases := []struct{ in, want string }{
		{"\xff\xfe", "\xff\xfe"},
		{"\xed\xa0\x80", "\xed\xa0\x80"},
		{"\xe0\xb9", "\xe0\xb9"},
		{"\xe0\xb9๕", "\xe0\xb95"},
		{"ก\x00ข๕", "ก\x00ข5"},
		{"๕\xff๖", "5\xff6"},
	}
	for _, c := range cases {
		if got := DigitsToArabic(c.in); got != c.want {
			t.Errorf("DigitsToArabic(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
