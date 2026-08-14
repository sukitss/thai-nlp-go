package auto_test

import (
	"strings"
	"testing"
	"unicode"

	"github.com/sukitss/thai-nlp-go/discover"
	"github.com/sukitss/thai-nlp-go/discover/auto"
)

func has(terms []discover.Term, text string) bool {
	for _, t := range terms {
		if t.Text == text {
			return true
		}
	}
	return false
}

func texts(terms []discover.Term) []string {
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		out = append(out, t.Text)
	}
	return out
}

// The corpus a real caller has is not in one language. A Thai coined name and an
// English collocation must both come out of the same call, each judged by its
// own script's rules.
func TestMixedThaiAndEnglish(t *testing.T) {
	// Thai prose written the way prose is: one name in varied frames, and
	// enough ordinary sentences that ordinary words are the common ones.
	thaiFrames := []string{
		"มิราเบลเดินเข้ามาในห้องแล้วมองไปที่หน้าต่าง",
		"เมื่อวานนี้มิราเบลบอกกับเราว่าจะกลับมาก่อนค่ำ",
		"ทุกคนรู้ว่ามิราเบลไม่เคยผิดสัญญาแม้แต่ครั้งเดียว",
		"เขาถามมิราเบลถึงเรื่องที่เกิดขึ้นเมื่อคืนก่อน",
		"มิราเบลยิ้มเล็กน้อยแล้วตอบว่าไม่มีอะไรต้องกังวล",
		"พวกเขาเห็นมิราเบลยืนอยู่ตรงประตูใหญ่ของอาคารหลังนั้น",
	}
	thaiProse := []string{
		"วันนี้อากาศดีมากและทุกคนก็ออกไปเดินเล่นกันที่สวนสาธารณะ",
		"เขาบอกว่าจะกลับมาอีกครั้งเมื่อทุกอย่างเรียบร้อยดีแล้ว",
		"ในห้องประชุมมีคนนั่งอยู่หลายคนและไม่มีใครพูดอะไรเลย",
		"เธอมองไปที่หน้าต่างแล้วคิดถึงเรื่องที่เกิดขึ้นเมื่อวาน",
		"ทุกคนรู้ดีว่าการเดินทางครั้งนี้ไม่ใช่เรื่องง่ายเลยสักนิด",
		"เขาเดินออกจากอาคารหลังนั้นไปโดยไม่หันกลับมามองอีกครั้ง",
	}
	var docs []string
	for i := 0; i < 400; i++ {
		docs = append(docs,
			"the man at the desk of the school read the letter of the day",
			"a woman in the city of the north sent word to the board again",
			"they left the top of the hour to the team at the lab today")
		if i%40 == 0 {
			docs = append(docs,
				"we study machine learning daily in the lab of the city",
				"the report on machine learning went to the board of the school",
				"her machine learning course starts at the top of the hour")
		}
	}
	for i := 0; i < 12; i++ {
		docs = append(docs, thaiProse...)
		if i%3 == 0 {
			docs = append(docs, thaiFrames...)
		}
	}
	terms := auto.Terms(docs, auto.Options{})

	if !has(terms, "machine learning") {
		t.Errorf("the English collocation should be found, got %v", texts(terms))
	}
	if !has(terms, "มิราเบล") {
		t.Errorf("the Thai name should be found, got %v", texts(terms))
	}
	if has(terms, "of the") {
		t.Error("a frequent function-word pair is not a term in any script")
	}
	// The failure this routing exists to prevent: joining English tokens the
	// way Thai joins them.
	for _, tm := range terms {
		if strings.Contains(tm.Text, "machinelearning") {
			t.Errorf("English tokens must be rejoined with spaces, got %q", tm.Text)
		}
	}
}

// Latin script has no dictionary here, so nothing in it may claim the
// unknown-word concession — otherwise every phrase in it would qualify.
func TestNoDictionaryMeansNoConcession(t *testing.T) {
	var docs []string
	for i := 0; i < 60; i++ {
		docs = append(docs, "he walked to the door and then he walked back again")
	}
	for _, tm := range auto.Terms(docs, auto.Options{}) {
		if tm.Unknown {
			t.Errorf("%q was treated as an unknown word", tm.Text)
		}
	}
}

// Edge cases where Thai and Latin meet. Every one of these is something a real
// corpus contains, and each has a right answer that differs from the other
// script's right answer — which is what the routing is for.
func TestThaiEnglishEdges(t *testing.T) {
	// A corpus big enough for the statistics to mean something, carrying one
	// example of each edge case often enough to be judged.
	background := []string{
		"the man at the desk of the school read the letter of the day",
		"a woman in the city of the north sent word to the board again",
		"วันนี้อากาศดีมากและทุกคนก็ออกไปเดินเล่นกันที่สวนสาธารณะ",
		"ในห้องประชุมมีคนนั่งอยู่หลายคนและไม่มีใครพูดอะไรเลยสักคน",
	}
	edges := []string{
		// Latin words with no space before the Thai that follows them.
		"ระบบAIของเราทำงานได้ดีมากในตอนนี้และยังพัฒนาต่อไปเรื่อย",
		"เขาบอกว่าระบบAIตัวใหม่จะมาถึงในเดือนหน้าอย่างแน่นอนแล้ว",
		"ทีมงานเชื่อว่าระบบAIจะช่วยงานได้มากกว่าที่เคยคิดกันไว้",
		// A number against a unit.
		"ไฟล์ขนาด 5GB ถูกอัปโหลดขึ้นไปเรียบร้อยแล้วตั้งแต่เมื่อคืน",
		"เขาส่งไฟล์ขนาด 5GB มาให้ดูก่อนที่จะเริ่มประชุมกันในวันนี้",
		"ระบบแจ้งว่าไฟล์ขนาด 5GB นั้นใหญ่เกินกว่าที่กำหนดไว้แล้ว",
	}
	var docs []string
	for i := 0; i < 200; i++ {
		docs = append(docs, background...)
		if i%10 == 0 {
			docs = append(docs, edges...)
		}
	}
	terms := auto.Terms(docs, auto.Options{})

	for _, tm := range terms {
		if thaiAndLatin(tm.Text) {
			t.Errorf("a term must not straddle a script boundary, got %q — "+
				"a dictionary entry for it would hide both halves", tm.Text)
		}
	}
	if has(terms, "5GB") {
		t.Error("a number and its unit are two tokens, not a term to look up")
	}
}

func thaiAndLatin(text string) bool {
	th, latin := false, false
	for _, r := range text {
		switch {
		case r >= 0x0E00 && r <= 0x0E7F:
			th = true
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			latin = true
		}
	}
	return th && latin
}

// Case is not folded: "Machine Learning" in a title and "machine learning" in
// prose are counted apart. A caller that wants them together lowercases before
// calling, which is a decision only the caller can make — folding case is wrong
// for acronyms and for names.
func TestCaseIsTheCallersDecision(t *testing.T) {
	var docs []string
	for i := 0; i < 400; i++ {
		docs = append(docs,
			"the man at the desk of the school read the letter of the day",
			"a woman in the city of the north sent word to the board again",
			"they left the top of the hour to the team at the lab today",
			"he wrote to the board of the city about the work of the team")
		if i%40 == 0 {
			docs = append(docs,
				"we study machine learning daily in the lab of the city",
				"the report on machine learning went to the board of the school",
				"her machine learning course starts at the top of the hour")
		}
	}
	terms := auto.Terms(docs, auto.Options{})
	if !has(terms, "machine learning") {
		t.Fatalf("precondition failed: %v", texts(terms))
	}

	lower := make([]string, len(docs))
	for i, d := range docs {
		lower[i] = strings.ToUpper(d)
	}
	upper := auto.Terms(lower, auto.Options{})
	if !has(upper, "MACHINE LEARNING") {
		t.Errorf("the same corpus in upper case should yield the upper-case term, got %v", texts(upper))
	}
	if has(upper, "machine learning") {
		t.Error("case must not be folded behind the caller's back")
	}
}

// Thai digits are Thai script and Arabic digits are not; neither is a word, and
// a run of them must not be proposed as one.
func TestDigitsAreNotWords(t *testing.T) {
	var docs []string
	for i := 0; i < 200; i++ {
		docs = append(docs,
			"วันนี้อากาศดีมากและทุกคนก็ออกไปเดินเล่นกันที่สวนสาธารณะ",
			"เขาเขียนเลข ๑๒๓ ไว้ที่กระดานแล้วเดินออกไปจากห้องเรียน",
			"ราคาสินค้าอยู่ที่ 1,000 บาทซึ่งถูกกว่าที่คิดไว้มากนัก")
	}
	for _, tm := range auto.Terms(docs, auto.Options{}) {
		digits := true
		for _, r := range tm.Text {
			if !unicode.IsDigit(r) && r != ',' && r != ' ' && r != '.' {
				digits = false
				break
			}
		}
		if digits {
			t.Errorf("a run of digits is not a term, got %q", tm.Text)
		}
	}
}

// An empty corpus, blank documents and punctuation-only documents must not
// panic and must not propose anything.
func TestEmptyAndPunctuationOnly(t *testing.T) {
	if got := auto.Terms(nil, auto.Options{}); len(got) != 0 {
		t.Errorf("nil corpus → no terms, got %v", texts(got))
	}
	var docs []string
	for i := 0; i < 40; i++ {
		docs = append(docs, "", "   ", "…", "“”", "!!!", "。。。", "\t\n")
	}
	if got := auto.Terms(docs, auto.Options{}); len(got) != 0 {
		t.Errorf("nothing to read → no terms, got %v", texts(got))
	}
}
