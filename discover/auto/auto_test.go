package auto_test

import (
	"strings"
	"testing"

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
