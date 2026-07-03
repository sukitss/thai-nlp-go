package translit

import (
	"math"
	"testing"
)

// TestCompleteSoundexGolden checks CompleteSoundex matches PyThaiNLP's
// complete_soundex() byte-for-byte on single-syllable Thai words (words the
// PyThaiNLP syllable tokenizer keeps as one syllable, so no CRF tokenization
// is involved — the encoding path is identical).
func TestCompleteSoundexGolden(t *testing.T) {
	in := readLines(t, "testdata/complete_soundex.txt")
	gold := readLines(t, "testdata/complete_soundex.golden")
	if len(in) != len(gold) {
		t.Fatalf("count mismatch: in=%d gold=%d", len(in), len(gold))
	}
	diffs := 0
	for i := range in {
		if got := CompleteSoundex(in[i]); got != gold[i] {
			diffs++
			if diffs <= 10 {
				t.Errorf("CompleteSoundex(%q) = %q, want %q", in[i], got, gold[i])
			}
		}
	}
	if diffs == 0 {
		t.Logf("✅ complete_soundex: %d single-syllable words match PyThaiNLP", len(in))
	}
}

// TestCompleteSoundexSyllables checks the multi-syllable join path. Syllable
// splits were taken from PyThaiNLP's syllable_tokenize(clean_text(word)) so both
// sides encode identical syllables; the expected code is PyThaiNLP's
// complete_soundex(word).
func TestCompleteSoundexSyllables(t *testing.T) {
	cases := []struct {
		sylls []string
		want  string
	}{
		{[]string{"ปุญ", "ญา"}, "ปป4G0น-ยย1B0--*"},                            // ปุญญา
		{[]string{"ปัน", "นา"}, "ปป1A0น-นน1B0--"},                             // ปันนา
		{[]string{"กรุง", "เทพ"}, "กก4Gง0รทท5Jบ0-"},                           // กรุงเทพ
		{[]string{"ประ", "เทศ"}, "ปป1A-0รทท5Jด0-"},                            // ประเทศ
		{[]string{"มะ", "ละ", "กอ"}, "มม1A-0-รร1A-0-กก1A-0-ออ1A-0-"},          // มะละกอ
		{[]string{"ทำ", "งาน"}, "ทท1Aม0-งง1Bน0-"},                             // ทำงาน
		{[]string{"นัก", "เรียน"}, "นน1A0ก-รร2Dน0-"},                          // นักเรียน
		{[]string{"โรง", "เรียน"}, "รร7Nง0-รร2Dน0-"},                          // โรงเรียน
		{[]string{"สวัส", "ดี"}, "ซศ1Aด0วดด2D-0-"},                            // สวัสดี
		{[]string{"ความ", "รัก"}, "คค1Bม0-รร1Aก0-"},                           // ความรัก
		{[]string{"ประ", "เทศ", "ไทย"}, "ปป1A-0รทท5Jด0-ทท1Aย0-"},              // ประเทศไทย
		{[]string{"อา", "หาร"}, "ออ1B-0-ฮห1Bน0-"},                             // อาหาร
		{[]string{"โทร", "ศัพ"}, "ซซ7N-0-ซศ1Aบ0-"},                            // โทรศัพท์
		{[]string{"คอม", "พิว", "เตอ"}, "คค1A-0-ออ7Mม0-พพ2Cว0-ตต9R-0-"},       // คอมพิวเตอร์
		{[]string{"มหา", "วิท", "ยา", "ลัย"}, "มม1B-0-วว2Cด0-ยย1B0--รร1Aย0-"}, // มหาวิทยาลัย
	}
	for _, c := range cases {
		if got := CompleteSoundexSyllables(c.sylls); got != c.want {
			t.Errorf("CompleteSoundexSyllables(%q) = %q, want %q", c.sylls, got, c.want)
		}
	}
}

// TestCompleteSoundexSimilarity checks CompleteSoundexSimilarity against
// PyThaiNLP's complete_soundex_similarity() on pre-encoded codes.
func TestCompleteSoundexSimilarity(t *testing.T) {
	cases := []struct {
		a, b string
		want float64
	}{
		{"กก1Bน2-", "กก1Bน2-", 1.0},                                      // ก้าน vs ก้าน
		{"คข1A-0-มม1A-0-คข3Fน0-", "คข7Mม1-คข3Fน0-", 0.23809523809523808}, // ขมขืน vs ข่มขืน
		{"รร1Aก0-", "รร1Aก0-", 1.0},                                      // รัก vs ลัก
		{"ซซ1Bย0-", "นน1A2ม-", 0.2857142857142857},                       // ทราย vs น้ำ
		{"บบ1Bน2-", "บบ1Bน0-", 0.8571428571428571},                       // บ้าน vs บาน
		{"ดด2D-0-", "ตต2D-0-", 0.7142857142857143},                       // ดี vs ตี
		{"", "", 1.0},
		{"กก1Bน2-", "", 0.0},
	}
	for _, c := range cases {
		got := CompleteSoundexSimilarity(c.a, c.b)
		if math.Abs(got-c.want) > 1e-12 {
			t.Errorf("CompleteSoundexSimilarity(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestCompleteSoundexEmpty(t *testing.T) {
	if CompleteSoundex("") != "" {
		t.Error("CompleteSoundex(\"\") should be empty")
	}
	if CompleteSoundexSyllables(nil) != "" {
		t.Error("CompleteSoundexSyllables(nil) should be empty")
	}
	// a word that cleans to empty (consonant + thanthakhat) -> empty code
	if got := CompleteSoundex("ก์"); got != "" {
		t.Errorf("CompleteSoundex(karan word) = %q, want empty", got)
	}
}
