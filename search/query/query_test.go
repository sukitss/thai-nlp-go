package query

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/multi"
	"github.com/sukitss/thai-nlp-go/search/keyword"
	"github.com/sukitss/thai-nlp-go/stopwords"
)

func has(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

// --- Parse: filler stripped, entities + acronyms kept --------------------------

func TestParseStripsFillerKeepsEntities(t *testing.T) {
	p := New()
	// The long conversational case: intent + entities buried in filler.
	got := p.Parse("ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ")
	for _, keep := range []string{"SOP", "จัดซื้อจัดจ้าง", "แผนก", "IT", "PC"} {
		if !has(got, keep) {
			t.Errorf("content term %q dropped: got %v", keep, got)
		}
	}
	for _, drop := range []string{"ช่วย", "สรุป", "หน่อย", "เอา", "นะ", "ที่", "การ"} {
		if has(got, drop) {
			t.Errorf("filler %q survived: got %v", drop, got)
		}
	}
}

// TestAcronymRetention is the crux: a bare, lower-case acronym the stop list
// would eat is retained as the upper-case form the index holds — including the
// stuck form split at the script boundary.
func TestAcronymRetention(t *testing.T) {
	p := New()
	cases := []struct {
		text string
		want []string
	}{
		{"it", []string{"IT"}},
		{"pc", []string{"PC"}},
		{"hr", []string{"HR"}},
		{"กุ้ง it", []string{"กุ้ง", "IT"}},
		{"ขอเบอร์ปลาit หน่อย", []string{"เบอร์", "ปลา", "IT"}}, // stuck ปลาit → ปลา + IT
		{"กุ้งpos อยู่แผนกไหน", []string{"กุ้ง", "POS", "แผนก"}},
	}
	for _, c := range cases {
		got := p.Parse(c.text)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Parse(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

// A real acronym typed upper-case is kept verbatim; a genuine English stop word
// that is NOT a known acronym is still dropped.
func TestParseKeepsAcronymDropsStopword(t *testing.T) {
	p := New()
	got := p.Parse("the IT team fixed it and the POS")
	if !has(got, "IT") || !has(got, "POS") {
		t.Errorf("acronyms dropped: got %v", got)
	}
	// "it" is a known acronym here (seed), so unlike a pure stop word it is
	// retained as IT — the pronoun/acronym ambiguity resolves toward the acronym
	// on purpose for queries. "the"/"team"/"fixed"/"and" must be gone.
	for _, drop := range []string{"the", "and"} {
		if has(got, drop) {
			t.Errorf("stop word %q survived: got %v", drop, got)
		}
	}
}

// --- Expand: bidirectional acronym/abbreviation equivalence --------------------

func TestExpandBidirectional(t *testing.T) {
	p := New()
	cases := []struct {
		term string
		want []string // membership (order: term first, then group)
	}{
		{"IT", []string{"IT", "ไอที", "เทคโนโลยีสารสนเทศ"}},
		{"ไอที", []string{"ไอที", "IT", "เทคโนโลยีสารสนเทศ"}},
		{"เทคโนโลยีสารสนเทศ", []string{"เทคโนโลยีสารสนเทศ", "IT", "ไอที"}},
		{"pc", []string{"pc", "PC", "พีซี", "คอมพิวเตอร์"}}, // lower term + upper case variant + group
		{"HR", []string{"HR", "เอชอาร์", "ทรัพยากรบุคคล", "ฝ่ายบุคคล"}},
	}
	for _, c := range cases {
		got := p.Expand(c.term)
		for _, w := range c.want {
			if !has(got, w) {
				t.Errorf("Expand(%q) missing %q: got %v", c.term, w, got)
			}
		}
		if got[0] != c.term {
			t.Errorf("Expand(%q)[0] = %q, want the term itself", c.term, got[0])
		}
	}
	// Unknown term: itself plus its upper-case case-variant, nothing invented.
	if got := p.Expand("xyz"); !reflect.DeepEqual(got, []string{"xyz", "XYZ"}) {
		t.Errorf("Expand(unknown) = %v, want [xyz XYZ]", got)
	}
	if got := p.Expand("ปลา"); !reflect.DeepEqual(got, []string{"ปลา"}) {
		t.Errorf("Expand(non-acronym Thai) = %v, want [ปลา]", got)
	}
}

// --- Aliases: per-tenant groups, and merging into a seed group -----------------

func TestAliasesInjectAndMerge(t *testing.T) {
	p := New(WithAliases(map[string][]string{
		"ENG": {"วิศวกรรม", "ช่าง"}, // brand-new group
		"IT":  {"สารสนเทศ"},         // shares "IT" with the seed → merges in
	}))
	// New group is expandable and retainable.
	if got := p.Expand("eng"); !has(got, "วิศวกรรม") || !has(got, "ช่าง") {
		t.Errorf("alias group not expanded: %v", got)
	}
	if got := p.Parse("eng"); !reflect.DeepEqual(got, []string{"ENG"}) {
		t.Errorf("alias acronym not retained: Parse(eng) = %v", got)
	}
	// Merge: the seed IT group now also yields the injected synonym, and the seed
	// members remain.
	got := p.Expand("IT")
	for _, w := range []string{"ไอที", "เทคโนโลยีสารสนเทศ", "สารสนเทศ"} {
		if !has(got, w) {
			t.Errorf("merged IT group missing %q: got %v", w, got)
		}
	}
}

// --- ParseExpand: parse then expand, de-duplicated -----------------------------

func TestParseExpand(t *testing.T) {
	p := New()
	got := p.ParseExpand("กุ้ง it")
	for _, w := range []string{"กุ้ง", "IT", "ไอที", "เทคโนโลยีสารสนเทศ"} {
		if !has(got, w) {
			t.Errorf("ParseExpand missing %q: got %v", w, got)
		}
	}
	// No duplicates.
	seen := map[string]bool{}
	for _, s := range got {
		if seen[s] {
			t.Errorf("ParseExpand duplicate %q in %v", s, got)
		}
		seen[s] = true
	}
}

// --- Salient: trimmed head-entity subset --------------------------------------

func TestSalient(t *testing.T) {
	p := New()
	full := p.Parse("ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ")
	sal := p.Salient("ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ", 2)
	if len(sal) != 2 {
		t.Fatalf("Salient(k=2) len = %d, want 2", len(sal))
	}
	for _, s := range sal {
		if !has(full, s) {
			t.Errorf("Salient term %q not in Parse output %v", s, full)
		}
	}
	// k>=len returns all parse terms unchanged.
	if got := p.Salient("it", 10); !reflect.DeepEqual(got, []string{"IT"}) {
		t.Errorf("Salient(k>=len) = %v, want [IT]", got)
	}
}

// --- Options: analyzer overlay & custom stopwords ------------------------------

func TestWithAnalyzerOverlay(t *testing.T) {
	// Base + filler stop set still applies through a custom analyzer.
	a := &multi.Analyzer{}
	p := New(WithAnalyzer(a))
	if got := p.Parse("ช่วย IT หน่อย"); !reflect.DeepEqual(got, []string{"IT"}) {
		t.Errorf("Parse with custom analyzer = %v, want [IT]", got)
	}
}

func TestWithStopwordsUnionsFiller(t *testing.T) {
	// A minimal base set must still get the filler union: "ช่วย"/"หน่อย" dropped.
	p := New(WithStopwords(stopwords.New("foo")))
	got := p.Parse("ช่วย ปลา หน่อย")
	if !reflect.DeepEqual(got, []string{"ปลา"}) {
		t.Errorf("filler not unioned with base stopwords: got %v", got)
	}
}

// --- Edge cases ---------------------------------------------------------------

func TestEdgeCases(t *testing.T) {
	p := New()
	if got := p.Parse(""); got != nil {
		t.Errorf("Parse(empty) = %v, want nil", got)
	}
	if got := p.ParseExpand(""); got != nil {
		t.Errorf("ParseExpand(empty) = %v, want nil", got)
	}
	// All filler / punctuation → nothing.
	if got := p.Parse("ช่วย ขอ หน่อย นะ ครับ ??? , ."); got != nil {
		t.Errorf("Parse(all-filler) = %v, want nil", got)
	}
	// Numbers dropped by default; kept with WithNumbers.
	if got := p.Parse("โทร 081 234 หา ปลา"); has(got, "081") || has(got, "234") {
		t.Errorf("numbers leaked: %v", got)
	}
	if got := New(WithNumbers(true)).Parse("โทร 081 ปลา"); !has(got, "081") {
		t.Errorf("WithNumbers(true) dropped 081: %v", got)
	}
}

func TestDeterministic(t *testing.T) {
	p := New()
	const q = "ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ"
	first := p.ParseExpand(q)
	for i := 0; i < 50; i++ {
		if !reflect.DeepEqual(p.ParseExpand(q), first) {
			t.Fatalf("nondeterministic ParseExpand at iteration %d", i)
		}
	}
}

// TestConcurrent exercises the read-only guarantee under -race.
func TestConcurrent(t *testing.T) {
	p := New()
	qs := []string{"it", "ขอเบอร์ปลาit หน่อย", "hr", "กุ้งpos อยู่แผนกไหน"}
	done := make(chan struct{})
	for i := 0; i < 8; i++ {
		go func(i int) {
			for j := 0; j < 200; j++ {
				_ = p.ParseExpand(qs[(i+j)%len(qs)])
			}
			done <- struct{}{}
		}(i)
	}
	for i := 0; i < 8; i++ {
		<-done
	}
}

// =============================================================================
// Retrieval eval: does query understanding actually recover indexed rows?
//
// One acronym-aware index (search/keyword over the fixture rows). We vary ONLY
// the query analysis and measure hit@k against ground-truth relevant rows:
//
//	naive  — ES/pythainlp style: lower-case everything, drop stop words, no
//	         acronym awareness (the pipeline the user is stuck with today).
//	raw    — the raw query tokenized exactly like the index (acronym-aware fold),
//	         no filler stripping, no expansion.
//	parse  — Parser.Parse: filler stripped + bare acronym retained.
//	expand — Parser.ParseExpand: parse + acronym/script expansion.
// =============================================================================

// faqCorpus is a SYNTHETIC IT/POS/HR/SOP FAQ (fake names/numbers), extending the
// T-120 keyword fixture. Rows r5 (SOP/PC, Thai only) and r6 (HR, Thai only)
// deliberately contain NO Latin acronym, so only expansion can reach them.
var faqCorpus = []string{
	/* r0 */ "ติดต่อทีมเน็ตเวิร์ก ปลา IT พี่ปลา เบอร์ปลา ติดต่อพี่ปลา IT โทร 081-234-5678",
	/* r1 */ "รีเซ็ตรหัสผ่านอีเมล ลืมรหัสผ่าน password แจ้งพี่ปลา IT กดลืมรหัสผ่านที่หน้า login",
	/* r2 */ "ขอสิทธิ์เข้าระบบ POS permission กุ้ง POS ติดต่อพี่กุ้ง POS โทร 081-999-0000",
	/* r3 */ "เครื่องแคชเชียร์ค้าง POS ค้าง จอฟ้า รีสตาร์ทเครื่อง แจ้งพี่กุ้ง POS เบอร์ 081-999-0000",
	/* r4 */ "คู่มือ SOP การจัดซื้อจัดจ้าง สำหรับแผนก IT ขั้นตอนการขออนุมัติงบประมาณ",
	/* r5 */ "ระเบียบขั้นตอนปฏิบัติงานการจัดซื้อจัดจ้างอุปกรณ์คอมพิวเตอร์ของแผนกพีซี",
	/* r6 */ "ติดต่อฝ่ายบุคคล ทรัพยากรบุคคล เรื่องเงินเดือนและสวัสดิการ",
	/* r7 */ "ขอเบอร์ติดต่อ IT support สายด่วน IT 1234 หรือพี่ปลา",
	/* r8 */ "จอคอมพิวเตอร์ PC เปิดไม่ติด แจ้งทีม IT ตรวจเช็คสายไฟ",
	/* r9 */ "การตลาด MKT ทำโปรโมชั่นและแคมเปญ ติดต่อทีมมาร์เก็ตติ้ง",
}

type evalCase struct {
	query string
	rel   map[int]bool // ground-truth relevant row ids
	short bool         // part of the short + bare-acronym slice
}

func evalCases() []evalCase {
	rel := func(ids ...int) map[int]bool {
		m := map[int]bool{}
		for _, id := range ids {
			m[id] = true
		}
		return m
	}
	return []evalCase{
		{"ขอเบอร์ปลาit หน่อย", rel(0, 7), false},
		{"อยากติดต่อพี่ปลาทำยังไง", rel(0, 7), false},
		{"กุ้งpos อยู่แผนกไหน", rel(2, 3), false},
		{"ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ", rel(4, 5), false},
		{"it", rel(0, 1, 4, 7, 8), true},
		{"pc", rel(5, 8), true},
		{"hr", rel(6), true},
		{"mkt", rel(9), true},
	}
}

// bm25Index is a minimal in-test BM25 retriever over acronym-aware index terms.
type bm25Index struct {
	docs  [][]string
	tf    []map[string]int
	dlen  []float64
	avgdl float64
	idf   func(string) float64
}

func newBM25(rows []string) *bm25Index {
	v := keyword.BuildVocab(rows)
	ix := &bm25Index{idf: v.IDF}
	var total float64
	for _, r := range rows {
		terms := keyword.Keys(r)
		tf := map[string]int{}
		for _, t := range terms {
			tf[t]++
		}
		ix.docs = append(ix.docs, terms)
		ix.tf = append(ix.tf, tf)
		ix.dlen = append(ix.dlen, float64(len(terms)))
		total += float64(len(terms))
	}
	ix.avgdl = total / float64(len(rows))
	return ix
}

// search returns doc ids with positive score, best first (ties by id). A method
// that yields no matchable term retrieves nothing — hit@k is then 0 at every k.
func (ix *bm25Index) search(queryTerms []string) []int {
	const k1, b = 1.2, 0.75
	uniq := map[string]bool{}
	var qt []string
	for _, t := range queryTerms {
		if !uniq[t] {
			uniq[t] = true
			qt = append(qt, t)
		}
	}
	type hit struct {
		id    int
		score float64
	}
	var hits []hit
	for i := range ix.docs {
		var s float64
		for _, t := range qt {
			f := float64(ix.tf[i][t])
			if f == 0 {
				continue
			}
			s += ix.idf(t) * (f * (k1 + 1)) / (f + k1*(1-b+b*ix.dlen[i]/ix.avgdl))
		}
		if s > 0 {
			hits = append(hits, hit{i, s})
		}
	}
	sort.SliceStable(hits, func(a, c int) bool {
		if hits[a].score != hits[c].score {
			return hits[a].score > hits[c].score
		}
		return hits[a].id < hits[c].id
	})
	out := make([]int, len(hits))
	for i, h := range hits {
		out[i] = h.id
	}
	return out
}

func hitAtK(ranked []int, rel map[int]bool, k int) float64 {
	for i := 0; i < k && i < len(ranked); i++ {
		if rel[ranked[i]] {
			return 1
		}
	}
	return 0
}

// naive analyzer: the ES/pythainlp baseline — lower-case Latin, drop stop words,
// NO acronym awareness ("IT"→"it"→stopword→dropped; "PC"→"pc"≠index "PC").
func naiveTerms(text string) []string {
	a := &multi.Analyzer{LowerLatin: true, Stop: stopwords.Multilingual()}
	return a.Terms(text)
}

// TestRetrievalHitAtK measures and prints the method × hit@{1,5} table plus the
// short+acronym slice, and asserts the qualitative claims the package exists to
// make. Run `go test -run RetrievalHitAtK -v ./search/query/` to see the table.
func TestRetrievalHitAtK(t *testing.T) {
	ix := newBM25(faqCorpus)
	p := New()
	cases := evalCases()

	methods := []struct {
		name string
		fn   func(string) []int
	}{
		{"naive", func(q string) []int { return ix.search(naiveTerms(q)) }},
		{"raw", func(q string) []int { return ix.search(keyword.Keys(q)) }},
		{"parse", func(q string) []int { return ix.search(keyword.Keys(strings.Join(p.Parse(q), " "))) }},
		{"parse-expand", func(q string) []int { return ix.search(keyword.Keys(strings.Join(p.ParseExpand(q), " "))) }},
	}

	type agg struct{ h1all, h5all, h1short, h5short, nAll, nShort float64 }
	res := map[string]*agg{}
	perQuery := map[string]map[string]float64{} // query → method → hit@5

	for _, m := range methods {
		res[m.name] = &agg{}
	}
	for _, c := range cases {
		perQuery[c.query] = map[string]float64{}
		for _, m := range methods {
			ranked := m.fn(c.query)
			h1 := hitAtK(ranked, c.rel, 1)
			h5 := hitAtK(ranked, c.rel, 5)
			a := res[m.name]
			a.h1all += h1
			a.h5all += h5
			a.nAll++
			if c.short {
				a.h1short += h1
				a.h5short += h5
				a.nShort++
			}
			perQuery[c.query][m.name] = h5
		}
	}

	t.Logf("Retrieval hit@k over %d fixture rows, %d queries (acronym-aware BM25 index)", len(faqCorpus), len(cases))
	t.Logf("%-13s | all hit@1 | all hit@5 | short hit@1 | short hit@5", "method")
	t.Logf("%s", strings.Repeat("-", 68))
	for _, m := range methods {
		a := res[m.name]
		t.Logf("%-13s |   %.3f   |   %.3f   |    %.3f    |    %.3f",
			m.name, a.h1all/a.nAll, a.h5all/a.nAll, a.h1short/a.nShort, a.h5short/a.nShort)
	}
	t.Logf("%s", strings.Repeat("-", 68))
	t.Logf("per-query hit@5 (isolating retention vs expansion):")
	for _, c := range cases {
		tag := ""
		if c.short {
			tag = " [short+acronym]"
		}
		t.Logf("  %-52s naive=%.0f raw=%.0f parse=%.0f expand=%.0f%s",
			c.query, perQuery[c.query]["naive"], perQuery[c.query]["raw"],
			perQuery[c.query]["parse"], perQuery[c.query]["parse-expand"], tag)
	}

	// --- Assertions: the honest, load-bearing claims -------------------------
	naive, raw, parse, exp := res["naive"], res["raw"], res["parse"], res["parse-expand"]

	// 1. On the short + bare-acronym slice, naive and raw retrieve NOTHING
	//    (the acronym is dropped or case-mismatched); parse/expand recover.
	if naive.h5short/naive.nShort != 0 || raw.h5short/raw.nShort != 0 {
		t.Errorf("expected naive/raw to score 0 on the bare-acronym slice, got naive=%.3f raw=%.3f",
			naive.h5short/naive.nShort, raw.h5short/raw.nShort)
	}
	if parse.h5short/parse.nShort <= raw.h5short/raw.nShort {
		t.Errorf("parse did not beat raw on the acronym slice: parse=%.3f raw=%.3f",
			parse.h5short/parse.nShort, raw.h5short/raw.nShort)
	}
	if exp.h5short/exp.nShort < parse.h5short/parse.nShort {
		t.Errorf("expand regressed vs parse on the acronym slice: expand=%.3f parse=%.3f",
			exp.h5short/exp.nShort, parse.h5short/parse.nShort)
	}

	// 2. Aggregate: query understanding strictly improves recall of the answer.
	if parse.h1all <= naive.h1all {
		t.Errorf("parse hit@1 did not beat naive: parse=%.3f naive=%.3f", parse.h1all/parse.nAll, naive.h1all/naive.nAll)
	}
	if exp.h5all < parse.h5all {
		t.Errorf("expand hit@5 regressed vs parse: expand=%.3f parse=%.3f", exp.h5all/exp.nAll, parse.h5all/parse.nAll)
	}

	// 3. Expansion is UNIQUELY needed for the synonym-only HR row: "hr" reaches
	//    r6 only after expanding to ทรัพยากรบุคคล/ฝ่ายบุคคล.
	if perQuery["hr"]["parse"] != 0 {
		t.Errorf("expected bare 'hr' to miss under parse (no literal HR indexed), got hit")
	}
	if perQuery["hr"]["parse-expand"] != 1 {
		t.Errorf("expected 'hr' to hit the HR row after expansion, got miss")
	}
}
