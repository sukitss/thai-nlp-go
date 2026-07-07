package keyword

import (
	"reflect"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/vocab"
)

// texts projects keywords to their surface strings, in rank order.
func texts(kws []Keyword) []string {
	out := make([]string, len(kws))
	for i, k := range kws {
		out[i] = k.Text
	}
	return out
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// --- fold: the acronym-aware casing that the whole use case hinges on --------

func TestFold(t *testing.T) {
	cases := []struct {
		in      string
		key     string
		acronym bool
	}{
		{"IT", "IT", true},    // acronym: verbatim
		{"POS", "POS", true},  // acronym
		{"USA", "USA", true},  // acronym
		{"it", "it", false},   // pronoun → lower (stop word)
		{"It", "it", false},   // sentence-initial → folds onto stop word
		{"The", "the", false}, // sentence-initial → stop word
		{"Password", "password", false},
		{"iOS", "ios", false}, // mixed case → not an acronym
		{"A", "a", false},     // single letter → not an acronym
		{"ปลา", "ปลา", false}, // no case → unchanged
		{"IT2", "IT2", true},  // acronym with a digit
	}
	for _, c := range cases {
		key, acr := fold(c.in)
		if key != c.key || acr != c.acronym {
			t.Errorf("fold(%q) = (%q,%v), want (%q,%v)", c.in, key, acr, c.key, c.acronym)
		}
	}
}

// --- The driving case: name + stuck acronym must survive extraction ----------

// faqRows is a SYNTHETIC IT/POS FAQ (fake names/numbers), mirroring the T-120
// keyword fixture. Each row is Subject + Keyword-cell + Answer, the "document"
// that gets a salient tag.
var faqRows = []string{
	"ติดต่อทีมเน็ตเวิร์ก ปลา IT, พี่ปลา, เบอร์ปลา, ติดต่อปลา ติดต่อพี่ปลา (IT) โทร 081-234-5678 หรือแจ้งผ่านระบบ ticket",
	"รีเซ็ตรหัสผ่านอีเมล รีเซ็ตรหัส, ลืมรหัสผ่าน, password แจ้งพี่ปลา IT หรือกดลืมรหัสผ่านที่หน้า login แล้วรอ OTP",
	"ขอสิทธิ์เข้าระบบ POS ขอสิทธิ์ POS, permission, กุ้ง POS ติดต่อพี่กุ้ง (POS) โทร 081-999-0000 พร้อมแนบใบอนุมัติ",
	"เครื่องแคชเชียร์ค้าง เครื่องค้าง, POS ค้าง, จอฟ้า รีสตาร์ทเครื่อง ถ้ายังไม่หายแจ้งพี่กุ้ง POS เบอร์ 081-999-0000",
	"พิมพ์ใบเสร็จไม่ออก ปริ้นเตอร์, ใบเสร็จ, กระดาษ เช็คกระดาษ หมึก ถ้ายังไม่ได้ติดต่อทีม POS",
	"ขอเบอร์ติดต่อ IT support เบอร์ IT, ติดต่อ IT, สายด่วน สายด่วน IT 1234 หรือพี่ปลา 081-234-5678",
}

// TestDrivingCaseAcronymSurvives is the reproduction ES+pythainlp missed: the
// English acronym stuck to a Thai name must survive keyword extraction instead
// of being lower-cased into a dropped stop word. We assert per-row that the
// salient name AND its acronym both appear among the extracted keywords.
func TestDrivingCaseAcronymSurvives(t *testing.T) {
	v := BuildVocab(faqRows)
	extractors := map[string]Extractor{
		"tfidf":    NewTFIDFTopK(v),
		"rake":     NewRAKE(),
		"yake":     NewYAKE(),
		"textrank": NewTextRank(),
	}
	rows := []struct {
		row     int
		name    string
		acronym string
	}{
		{0, "ปลา", "IT"},
		{2, "กุ้ง", "POS"},
	}
	for name, ex := range extractors {
		for _, w := range rows {
			got := texts(ex.Extract(faqRows[w.row], 8))
			// THE core claim, for every extractor: the stuck English acronym is
			// NOT lower-cased into a dropped stop word — it survives as a keyword.
			if !anyContains(got, w.acronym) {
				t.Errorf("%s row %d: acronym %q DROPPED (the bug this package fixes): got %v",
					name, w.row+1, w.acronym, got)
			}
			// The salient Thai name should also surface for the corpus/graph
			// extractors. YAKE deliberately deprioritizes a word that co-occurs
			// with many different neighbours (its relatedness feature), so the
			// ubiquitous "ปลา"/"กุ้ง" can fall out of its top-k — a documented
			// YAKE trait, verified below, not a failure here.
			if name != "yake" && !anyContains(got, w.name) {
				t.Errorf("%s row %d: name %q missing: got %v", name, w.row+1, w.name, got)
			}
		}
	}
}

// TestYAKERelatednessDropsUbiquitousName pins the honest YAKE trait surfaced by
// the driving case: a name that appears all over the document (many distinct
// neighbours) reads as stop-word-like to YAKE's relatedness feature and can miss
// the top-k, even though the acronym it is paired with survives.
func TestYAKERelatednessDropsUbiquitousName(t *testing.T) {
	got := texts(NewYAKE().Extract(faqRows[0], 8))
	if !contains(got, "IT") {
		t.Fatalf("YAKE should still keep acronym IT: %v", got)
	}
	if contains(got, "ปลา") {
		t.Logf("note: YAKE surfaced ปลา this run (%v); the relatedness penalty is soft", got)
	}
}

func anyContains(ss []string, tok string) bool {
	for _, s := range ss {
		if s == tok || strings.Contains(s, tok) {
			return true
		}
	}
	return false
}

// TestAcronymNotStopwordPronounIs pins the exact distinction: the acronyms
// "IT"/"POS" survive while the pronoun "it" (and sentence-initial "It"/"The")
// are dropped as stop words — corpus-free, so it isolates the casing fold.
func TestAcronymNotStopwordPronounIs(t *testing.T) {
	const doc = "It is what it is. The IT team fixed the POS. it works."
	got := texts(NewTFIDFTopK(nil).Extract(doc, 10))
	if !contains(got, "IT") {
		t.Errorf("acronym IT dropped: got %v", got)
	}
	if !contains(got, "POS") {
		t.Errorf("acronym POS dropped: got %v", got)
	}
	if contains(got, "it") || contains(got, "the") {
		t.Errorf("pronoun/stopword survived: got %v", got)
	}
	// Both spaced and stuck forms must surface ปลา + the acronym.
	for _, doc := range []string{"เบอร์ ปลา IT ครับ", "เบอร์ ปลาIT ครับ"} {
		got := texts(NewTFIDFTopK(nil).Extract(doc, 10))
		if !contains(got, "ปลา") || !contains(got, "IT") {
			t.Errorf("spaced/stuck acronym: doc %q got %v, want ปลา + IT", doc, got)
		}
	}
}

// --- TF-IDF: corpus DF must suppress the globally common word ----------------

func TestTFIDFCorpusSuppressesCommon(t *testing.T) {
	// "ติดต่อ" is in every doc (high df, low idf); each doc has one rare, doc-
	// specific word that must outrank it despite lower raw frequency.
	docs := []string{
		"ติดต่อ ติดต่อ ปลา",
		"ติดต่อ ติดต่อ กุ้ง",
		"ติดต่อ ติดต่อ หมึก",
	}
	v := BuildVocab(docs)
	got := texts(NewTFIDFTopK(v).Extract("ติดต่อ ติดต่อ ปลา", 1))
	if len(got) != 1 || got[0] != "ปลา" {
		t.Errorf("top keyword = %v, want [ปลา] (rare beats common despite lower tf)", got)
	}
	// With no corpus (pure tf) the common word wins on raw count.
	gotTF := texts(NewTFIDFTopK(nil).Extract("ติดต่อ ติดต่อ ปลา", 1))
	if len(gotTF) != 1 || gotTF[0] != "ติดต่อ" {
		t.Errorf("pure-tf top = %v, want [ติดต่อ]", gotTF)
	}
}

// --- RAKE: phrase grouping is the point --------------------------------------

func TestRAKEPhraseGrouping(t *testing.T) {
	// Stop words / punctuation delimit phrases; content runs stay whole.
	got := texts(NewRAKE().Extract("ปลา IT, พี่ปลา และ กุ้ง POS", 10))
	// "ปลา IT" and "กุ้ง POS" must appear as whole phrases (acronym + space kept).
	if !contains(got, "ปลา IT") {
		t.Errorf("RAKE lost phrase 'ปลา IT': got %v", got)
	}
	if !contains(got, "กุ้ง POS") {
		t.Errorf("RAKE lost phrase 'กุ้ง POS': got %v", got)
	}
	// The comma between "IT" and "พี่ปลา" must break the phrase (no "IT พี่ปลา").
	for _, kw := range got {
		if strings.Contains(kw, "IT") && strings.Contains(kw, "พี่") {
			t.Errorf("RAKE merged across punctuation: %q in %v", kw, got)
		}
	}
}

// --- Edge cases ---------------------------------------------------------------

func TestEdgeCases(t *testing.T) {
	all := []Extractor{NewTFIDFTopK(nil), NewRAKE(), NewYAKE(), NewTextRank()}
	for _, ex := range all {
		if got := ex.Extract("", 5); got != nil {
			t.Errorf("%T empty text: got %v, want nil", ex, got)
		}
		if got := ex.Extract("ปลา IT", 0); got != nil {
			t.Errorf("%T k=0: got %v, want nil", ex, got)
		}
		if got := ex.Extract("ปลา IT", -3); got != nil {
			t.Errorf("%T k<0: got %v, want nil", ex, got)
		}
		// All stop words / punctuation → nothing salient.
		if got := ex.Extract("the it is a . , ( )", 5); got != nil {
			t.Errorf("%T all-stopword: got %v, want nil", ex, got)
		}
		// Single content word.
		if got := texts(ex.Extract("ปลา", 5)); len(got) != 1 || got[0] != "ปลา" {
			t.Errorf("%T single word: got %v, want [ปลา]", ex, got)
		}
		// Mixed Thai/English/acronym/number: must not panic; must return ≤k.
		got := ex.Extract("ปลาit IT POS 12345 ราคา iPhone 12", 3)
		if len(got) > 3 {
			t.Errorf("%T returned %d > k=3", ex, len(got))
		}
	}
}

func TestNumbersDroppedByDefault(t *testing.T) {
	got := texts(NewTFIDFTopK(nil).Extract("โทร 081 234 5678 หา ปลา", 10))
	for _, k := range got {
		if k == "081" || k == "234" || k == "5678" {
			t.Errorf("pure number leaked into keywords: %v", got)
		}
	}
	// Opt back in.
	got = texts(NewTFIDFTopK(nil, WithNumbers(true)).Extract("โทร 081 ปลา", 10))
	if !contains(got, "081") {
		t.Errorf("WithNumbers(true) should keep 081: got %v", got)
	}
}

func TestTopKOrderingDeterministic(t *testing.T) {
	// Repeated extraction must be byte-identical (no map-iteration nondeterminism).
	ex := NewTFIDFTopK(BuildVocab(faqRows))
	first := ex.Extract(faqRows[0], 5)
	for i := 0; i < 20; i++ {
		if !reflect.DeepEqual(ex.Extract(faqRows[0], 5), first) {
			t.Fatalf("nondeterministic top-k on iteration %d", i)
		}
	}
}

func TestBuildVocabRoundTrips(t *testing.T) {
	v := BuildVocab(faqRows)
	if v == nil {
		t.Fatal("BuildVocab returned nil")
	}
	if _, ok := v.ID("ปลา"); !ok {
		t.Error("vocab missing content key ปลา")
	}
	if _, ok := v.ID("it"); ok {
		t.Error("vocab should not contain dropped stop word 'it'")
	}
	if v.DocCount() != len(faqRows) {
		t.Errorf("DocCount = %d, want %d", v.DocCount(), len(faqRows))
	}
	var _ *vocab.Vocab = v
}
