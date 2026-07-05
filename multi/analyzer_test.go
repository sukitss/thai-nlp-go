package multi

import (
	"strings"
	"sync"
	"testing"

	"github.com/sukitss/thai-nlp-go/cjk"
	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/en"
	"github.com/sukitss/thai-nlp-go/jp"
	"github.com/sukitss/thai-nlp-go/kr"
	"github.com/sukitss/thai-nlp-go/normalize"
	"github.com/sukitss/thai-nlp-go/stopwords"
)

// analyzerCorpus is a mixed Thai/EN/CN/JP/KR/digit corpus for pipeline and
// property tests (the golden mixedPassage inputs plus extra digit/case lines).
func analyzerCorpus() []string {
	var lines []string
	for _, c := range mixedPassage {
		lines = append(lines, c.in)
	}
	return append(lines,
		"The Quick BROWN Fox กระโดดข้ามรั้ว 一二三",
		"บทที่ ๕ ราคา ๑,๒๓๔ บาท กับ 25 USD",
		"北京大学生活で学生会に入った",
		"三国志を読む 그리고 한국어도 공부한다",
		"IT department กับ it ตัวเล็ก",
		"  spaced   out\t\ttext  ๑๒๓  ",
	)
}

// refTerms is an independent reimplementation of the Terms pipeline for
// cross-checking: it goes through the ORIGINAL rune-based forEachRun walker
// (not the Analyzer's byte-offset walker) and composes the public per-language
// Cut functions and post-filters directly.
func refTerms(a *Analyzer, text string) []string {
	norm := normalize.Normalize(text)
	if a.FoldThaiDigits {
		norm = normalize.DigitsToArabic(norm)
	}
	if norm == "" {
		return nil
	}
	th := thai_()
	if a.Overlay != nil {
		th = th.SessionWithDict(a.Overlay)
	}
	var out []string
	forEachRun(norm, func(l lang, run string, hasKana bool) {
		var toks []string
		switch l {
		case thai:
			toks = th.SegmentNoWS(run)
		case cjkHan:
			jpMode := hasKana
			if !jpMode {
				if a.Han != "" {
					jpMode = a.Han == HanJapanese
				} else {
					jpMode = DefaultHan == "jp"
				}
			}
			switch {
			case jpMode && a.UseDP:
				toks = jp.CutDP(run)
			case jpMode:
				toks = jp.Cut(run)
			case a.UseDP:
				toks = cjk.CutDP(run)
			default:
				toks = cjk.Cut(run)
			}
		case hangul:
			toks = kr.Cut(run)
		default:
			toks = en.Cut(run)
		}
		for _, t := range toks {
			if a.LowerLatin && (l == latin || l == other) {
				t = strings.ToLower(t)
			}
			if a.Stop != nil && a.Stop.IsStopword(t) {
				continue
			}
			out = append(out, t)
		}
	})
	return out
}

func glossary(words ...string) *dict.Trie {
	ov := dict.NewTrie()
	for _, w := range words {
		ov.Add(w)
	}
	return ov
}

// TestAnalyzerPipeline: over the mixed corpus and a spread of configurations,
// Terms matches the independent reimplementation, AppendTerms is exactly
// Terms joined, and Tokens carries the same token texts (unfiltered).
func TestAnalyzerPipeline(t *testing.T) {
	configs := []struct {
		name string
		a    *Analyzer
	}{
		{"zero", &Analyzer{}},
		{"lower", &Analyzer{LowerLatin: true}},
		{"stop-thai", &Analyzer{Stop: stopwords.Default()}},
		{"lower+stop", &Analyzer{LowerLatin: true, Stop: stopwords.Union(stopwords.Default(), stopwords.English())}},
		{"fold-digits", &Analyzer{FoldThaiDigits: true}},
		{"han-jp", &Analyzer{Han: HanJapanese}},
		{"dp", &Analyzer{UseDP: true, Han: HanChinese}},
		{"overlay+all", &Analyzer{
			Overlay:        glossary("เนี่ยหลี่", "กลอรี่"),
			Stop:           stopwords.Default(),
			LowerLatin:     true,
			Han:            HanChinese,
			UseDP:          true,
			FoldThaiDigits: true,
		}},
	}
	corpus := analyzerCorpus()
	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			for _, line := range corpus {
				got := cfg.a.Terms(line)
				want := refTerms(cfg.a, line)
				if strings.Join(got, "|") != strings.Join(want, "|") {
					t.Errorf("Terms(%q)\n  got  %q\n  want %q", line, got, want)
				}
				joined := strings.Join(got, " ")
				if ab := string(cfg.a.AppendTerms(nil, line, ' ')); ab != joined {
					t.Errorf("AppendTerms(%q) = %q, want %q", line, ab, joined)
				}
				// Tokens is the same pipeline with no post-filter: with the
				// filters off, its texts must equal Terms exactly.
				if cfg.a.Stop == nil && !cfg.a.LowerLatin {
					_, toks := cfg.a.Tokens(line)
					texts := make([]string, len(toks))
					for i, tk := range toks {
						texts[i] = tk.Text
					}
					if strings.Join(texts, "|") != strings.Join(got, "|") {
						t.Errorf("Tokens(%q) texts\n  got  %q\n  want %q", line, texts, got)
					}
				}
			}
		})
	}
}

// TestAnalyzerZeroMatchesSegment: the zero-value Analyzer is Segment — same
// normalize, same routing, same tokenizers, no filters.
func TestAnalyzerZeroMatchesSegment(t *testing.T) {
	var a Analyzer
	for _, line := range analyzerCorpus() {
		got := a.Terms(line)
		want := Segment(line)
		if strings.Join(got, "|") != strings.Join(want, "|") {
			t.Errorf("zero Analyzer.Terms(%q) = %q, want Segment output %q", line, got, want)
		}
	}
}

// TestAnalyzerOverlay: a glossary word segments as one term with the overlay
// set and splits without it, and Tokens sees the overlay too.
func TestAnalyzerOverlay(t *testing.T) {
	const text = "เนี่ยหลี่เปิดตาขึ้น"
	base := &Analyzer{}
	if got := base.Terms(text); strings.Join(got, "|") != "เนี่ย|ห|ลี่|เปิด|ตา|ขึ้น" {
		t.Fatalf("no-overlay Terms = %q (dictionary changed? re-pin this test)", got)
	}
	ov := &Analyzer{Overlay: glossary("เนี่ยหลี่")}
	if got := ov.Terms(text); strings.Join(got, "|") != "เนี่ยหลี่|เปิด|ตา|ขึ้น" {
		t.Errorf("overlay Terms = %q, want glossary word whole", got)
	}
	norm, toks := ov.Tokens(text)
	if len(toks) == 0 || toks[0].Text != "เนี่ยหลี่" {
		t.Fatalf("overlay Tokens = %v, want first token เนี่ยหลี่", toks)
	}
	if norm[toks[0].Start:toks[0].End] != toks[0].Text {
		t.Errorf("overlay token offsets broken: %q", norm[toks[0].Start:toks[0].End])
	}
}

// TestAnalyzerOverlayConcurrent: two tenants with different glossaries share
// the process; 16 goroutines under -race must each get their own tenant's
// segmentation with no cross-tenant leakage and no data race.
func TestAnalyzerOverlayConcurrent(t *testing.T) {
	tenantA := &Analyzer{Overlay: glossary("เนี่ยหลี่")}
	tenantB := &Analyzer{Overlay: glossary("กลอรี่")}
	const (
		textNie   = "เนี่ยหลี่เปิดตาขึ้น"
		textGlory = "เมืองกลอรี่สวยงาม"
		wantANie  = "เนี่ยหลี่|เปิด|ตา|ขึ้น"
		wantAGlo  = "เมือง|กล|อ|รี่|สวยงาม" // A's overlay must NOT know B's word
		wantBNie  = "เนี่ย|ห|ลี่|เปิด|ตา|ขึ้น"
		wantBGlo  = "เมือง|กลอรี่|สวยงาม"
	)
	var wg sync.WaitGroup
	errs := make(chan string, 16)
	for g := 0; g < 16; g++ {
		a, wantNie, wantGlo := tenantA, wantANie, wantAGlo
		if g%2 == 1 {
			a, wantNie, wantGlo = tenantB, wantBNie, wantBGlo
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				if got := strings.Join(a.Terms(textNie), "|"); got != wantNie {
					errs <- "Terms(nie) = " + got + ", want " + wantNie
					return
				}
				if got := strings.Join(a.Terms(textGlory), "|"); got != wantGlo {
					errs <- "Terms(glory) = " + got + ", want " + wantGlo
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error("cross-tenant leakage: " + e)
	}
}

// TestAnalyzerStopwords: stop words vanish from Terms but stay in Tokens (the
// highlighting path must see everything); nil Stop keeps all.
func TestAnalyzerStopwords(t *testing.T) {
	const text = "ความทรงจำของเขา" // ของ is a Thai stop word
	if !stopwords.Default().IsStopword("ของ") {
		t.Fatal("precondition: ของ should be in the default Thai stop-word set")
	}
	withStop := &Analyzer{Stop: stopwords.Default()}
	if got := withStop.Terms(text); contains(got, "ของ") {
		t.Errorf("Terms kept stop word ของ: %q", got)
	}
	_, toks := withStop.Tokens(text)
	found := false
	for _, tk := range toks {
		if tk.Text == "ของ" {
			found = true
		}
	}
	if !found {
		t.Errorf("Tokens must NOT filter stop words, got %v", toks)
	}
	var noStop Analyzer // nil Stop keeps all
	if got := noStop.Terms(text); !contains(got, "ของ") {
		t.Errorf("nil Stop must keep all terms, got %q", got)
	}
	// ๆ segments as its own token and is droppable via a custom set.
	custom := &Analyzer{Stop: stopwords.New("ๆ")}
	if got := custom.Terms("วิ่งเร็วๆ"); contains(got, "ๆ") {
		t.Errorf("custom set failed to drop ๆ: %q", got)
	}
	if got := noStop.Terms("วิ่งเร็วๆ"); !contains(got, "ๆ") {
		t.Fatalf("precondition: ๆ should appear unfiltered, got %q", got)
	}
}

// TestAnalyzerLowerLatinStop: folding happens BEFORE the stop check, so a
// lowercase set catches capitalized forms; with folding off the check stays
// case-sensitive (the acronym-preserving default).
func TestAnalyzerLowerLatinStop(t *testing.T) {
	stop := stopwords.New("the")
	const text = "The quick fox"
	folded := &Analyzer{LowerLatin: true, Stop: stop}
	if got := folded.Terms(text); strings.Join(got, "|") != "quick|fox" {
		t.Errorf("LowerLatin+Stop Terms = %q, want [quick fox]", got)
	}
	// Tokens keeps original casing regardless.
	_, toks := folded.Tokens(text)
	if len(toks) == 0 || toks[0].Text != "The" {
		t.Errorf("Tokens must keep original case, got %v", toks)
	}
	cased := &Analyzer{Stop: stop} // no folding: "The" != "the"
	if got := cased.Terms(text); strings.Join(got, "|") != "The|quick|fox" {
		t.Errorf("case-sensitive Terms = %q, want [The quick fox]", got)
	}
	lowerOnly := &Analyzer{LowerLatin: true}
	if got := lowerOnly.Terms(text); strings.Join(got, "|") != "the|quick|fox" {
		t.Errorf("LowerLatin Terms = %q, want [the quick fox]", got)
	}
}

// TestAnalyzerFoldThaiDigits: with folding on, Thai digits become ASCII in
// both terms and the normalized string; off by default they stay Thai.
func TestAnalyzerFoldThaiDigits(t *testing.T) {
	const text = "บทที่ ๕"
	folded := &Analyzer{FoldThaiDigits: true}
	if got := folded.Terms(text); !contains(got, "5") || contains(got, "๕") {
		t.Errorf("folded Terms = %q, want term 5 and no ๕", got)
	}
	norm, _ := folded.Tokens(text)
	if !strings.Contains(norm, "5") || strings.Contains(norm, "๕") {
		t.Errorf("folded norm = %q, want 5 and no ๕", norm)
	}
	var plain Analyzer
	if got := plain.Terms(text); !contains(got, "๕") {
		t.Errorf("default Terms = %q, want ๕ kept", got)
	}
}

// TestAnalyzerHanMode: an ambiguous all-kanji run routes per a.Han, and
// DefaultHan is NOT consulted when Han is set — proven by flipping the global
// to the opposite value and asserting the output does not move. The zero value
// keeps following DefaultHan (documented backward-compat fallback).
func TestAnalyzerHanMode(t *testing.T) {
	saved := DefaultHan
	defer func() { DefaultHan = saved }()
	const text = "三国志" // pinned: cn keeps it whole, jp splits 三国|志
	wantCN, wantJP := "三国志", "三国|志"

	DefaultHan = "jp" // opposite of the Analyzer's explicit choice
	cn := &Analyzer{Han: HanChinese}
	if got := strings.Join(cn.Terms(text), "|"); got != wantCN {
		t.Errorf("Han=cn Terms = %q, want %q (DefaultHan leaked in)", got, wantCN)
	}
	DefaultHan = "cn"
	jpA := &Analyzer{Han: HanJapanese}
	if got := strings.Join(jpA.Terms(text), "|"); got != wantJP {
		t.Errorf("Han=jp Terms = %q, want %q (DefaultHan leaked in)", got, wantJP)
	}
	// Tokens follows the same routing.
	if _, toks := jpA.Tokens(text); len(toks) != 2 || toks[0].Text != "三国" {
		t.Errorf("Han=jp Tokens = %v, want 三国|志", toks)
	}
	// Zero value: follows the global (backward compat).
	var zero Analyzer
	DefaultHan = "jp"
	if got := strings.Join(zero.Terms(text), "|"); got != wantJP {
		t.Errorf("zero-Han Terms with DefaultHan=jp = %q, want %q", got, wantJP)
	}
	DefaultHan = "cn"
	if got := strings.Join(zero.Terms(text), "|"); got != wantCN {
		t.Errorf("zero-Han Terms with DefaultHan=cn = %q, want %q", got, wantCN)
	}
}

// TestAnalyzerUseDP: 北京大学生 is the classic case where greedy longest-match
// (北京大学|生) and DP max-probability (北京|大学生) disagree.
func TestAnalyzerUseDP(t *testing.T) {
	const text = "北京大学生"
	greedy := &Analyzer{Han: HanChinese}
	if got := strings.Join(greedy.Terms(text), "|"); got != "北京大学|生" {
		t.Fatalf("greedy Terms = %q (dictionary changed? re-pin this test)", got)
	}
	dp := &Analyzer{Han: HanChinese, UseDP: true}
	if got := strings.Join(dp.Terms(text), "|"); got != "北京|大学生" {
		t.Errorf("DP Terms = %q, want 北京|大学生", got)
	}
}

// checkTokenOffsets asserts the Tokens contract on one input: every token is
// non-empty, in bounds, and norm[Start:End] == Text.
func checkTokenOffsets(t *testing.T, a *Analyzer, in string) {
	t.Helper()
	norm, toks := a.Tokens(in)
	for i, tk := range toks {
		if tk.Text == "" {
			t.Fatalf("Tokens(%.40q) token %d is empty", in, i)
		}
		if tk.Start < 0 || tk.End < tk.Start || tk.End > len(norm) {
			t.Fatalf("Tokens(%.40q) token %d out of bounds: [%d,%d) of %d", in, i, tk.Start, tk.End, len(norm))
		}
		if norm[tk.Start:tk.End] != tk.Text {
			t.Fatalf("Tokens(%.40q) token %d: norm[%d:%d]=%q != Text %q",
				in, i, tk.Start, tk.End, norm[tk.Start:tk.End], tk.Text)
		}
	}
}

// TestAnalyzerTokensOffsets: the norm[Start:End]==Text property over the mixed
// corpus, every adversarial input (invalid UTF-8, NUL, mark floods), all
// configuration spreads, and ~1 MB of input — no panics anywhere.
func TestAnalyzerTokensOffsets(t *testing.T) {
	configs := []*Analyzer{
		{},
		{FoldThaiDigits: true, LowerLatin: true, Stop: stopwords.Default()},
		{Overlay: glossary("เนี่ยหลี่"), Han: HanJapanese, UseDP: true},
	}
	inputs := analyzerCorpus()
	for _, c := range adversarialInputs {
		inputs = append(inputs, c.in)
	}
	inputs = append(inputs, strings.Repeat("ผมอ่าน三国志と日本語 hello 한국어 ๑๒๓ ", 15000)) // ~1.1 MB
	for _, a := range configs {
		for _, in := range inputs {
			checkTokenOffsets(t, a, in)
		}
	}
}

// FuzzAnalyzer: no config × input combination may panic, emit an empty term,
// break the AppendTerms==Terms join equality, or violate the offsets contract.
func FuzzAnalyzer(f *testing.F) {
	for _, s := range []string{
		"", "ผมรักabc中文日本語한국", "บทที่ ๕ The Quick", "北京大学生活",
		"เนี่ยหลี่เปิดตา", "\xff\xfe", "\xe0\xb9", "ก\x00ข", "我爱\xff自然",
	} {
		f.Add(s, uint8(0))
		f.Add(s, uint8(0xFF))
	}
	stop := stopwords.New("the", "ของ", "ๆ")
	overlay := glossary("เนี่ยหลี่")
	f.Fuzz(func(t *testing.T, s string, cfg uint8) {
		a := &Analyzer{
			LowerLatin:     cfg&1 != 0,
			UseDP:          cfg&2 != 0,
			FoldThaiDigits: cfg&4 != 0,
		}
		switch {
		case cfg&8 != 0:
			a.Han = HanChinese
		case cfg&16 != 0:
			a.Han = HanJapanese
		}
		if cfg&32 != 0 {
			a.Stop = stop
		}
		if cfg&64 != 0 {
			a.Overlay = overlay
		}
		terms := a.Terms(s)
		for _, tm := range terms {
			if tm == "" {
				t.Fatalf("Terms(%q, cfg=%08b) emitted an empty term: %q", s, cfg, terms)
			}
		}
		want := strings.Join(terms, " ")
		if got := string(a.AppendTerms(nil, s, ' ')); got != want {
			t.Fatalf("AppendTerms mismatch on (%q, cfg=%08b): %q vs %q", s, cfg, got, want)
		}
		norm, toks := a.Tokens(s)
		for i, tk := range toks {
			if tk.Start < 0 || tk.End < tk.Start || tk.End > len(norm) || norm[tk.Start:tk.End] != tk.Text {
				t.Fatalf("Tokens(%q, cfg=%08b) token %d violates offsets: %+v in norm %q", s, cfg, i, tk, norm)
			}
		}
	})
}

// ---------- benchmarks: Analyzer overhead vs plain Segment ----------

const analyzerBenchText = "ผมอ่านนิยายแปลจีน三国志และมังงะ日本語ที่แปลเป็นไทยกับ한국webtoon"

func BenchmarkAnalyzerTerms(b *testing.B) {
	a := &Analyzer{Han: HanChinese}
	a.Terms("warm")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(a.Terms(analyzerBenchText))
	}
	_ = n
}

// BenchmarkAnalyzerTermsFull: the realistic per-tenant configuration (overlay
// session per call + stop filter + folds) — the price of the whole pipeline.
func BenchmarkAnalyzerTermsFull(b *testing.B) {
	a := &Analyzer{
		Overlay:        glossary("เนี่ยหลี่", "กลอรี่"),
		Stop:           stopwords.Default(),
		LowerLatin:     true,
		Han:            HanChinese,
		FoldThaiDigits: true,
	}
	a.Terms("warm")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		n += len(a.Terms(analyzerBenchText))
	}
	_ = n
}

// BenchmarkAnalyzerAppendTerms: dst is reused; remaining allocations are the
// per-run tokenizer []string results (same trade-off as AppendBytes) and, with
// an overlay, the per-call session.
func BenchmarkAnalyzerAppendTerms(b *testing.B) {
	a := &Analyzer{Han: HanChinese}
	buf := make([]byte, 0, 256)
	buf = a.AppendTerms(buf, "warm", ' ')
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = a.AppendTerms(buf[:0], analyzerBenchText, ' ')
	}
	_ = buf
}

func BenchmarkAnalyzerTokens(b *testing.B) {
	a := &Analyzer{Han: HanChinese}
	a.Tokens("warm")
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		_, toks := a.Tokens(analyzerBenchText)
		n += len(toks)
	}
	_ = n
}

func TestAnalyzerFoldWidth(t *testing.T) {
	plain := Analyzer{}
	folded := Analyzer{FoldWidth: true}
	// full-width query must match half-width indexed form only when FoldWidth on
	if got := folded.Terms("ＡＰＩ"); len(got) != 1 || got[0] != "api" && got[0] != "API" {
		// note: LowerLatin off → expect "API"
		if len(got) == 0 || got[0] != "API" {
			t.Errorf("FoldWidth Terms(ＡＰＩ) = %v, want [API]", got)
		}
	}
	// without fold, full-width stays full-width (distinct from ASCII)
	p := plain.Terms("ＡＰＩ")
	f := folded.Terms("ＡＰＩ")
	if len(p) > 0 && len(f) > 0 && p[0] == f[0] {
		t.Errorf("FoldWidth should change output: plain=%v folded=%v", p, f)
	}
}

func TestAnalyzerSubwords(t *testing.T) {
	base := Analyzer{}
	sw := Analyzer{Subwords: true}
	// Thai compound: subwords must add the parts, keep the whole.
	got := sw.Terms("ภาษาไทย")
	joined := " " + strings.Join(got, " ") + " "
	if !strings.Contains(joined, " ภาษาไทย ") {
		t.Errorf("coarse token missing: %v", got)
	}
	// at least one proper subword present (ภาษา or ไทย), and more tokens than base
	if len(got) <= len(base.Terms("ภาษาไทย")) {
		t.Errorf("subwords should add tokens: base=%v sw=%v", base.Terms("ภาษาไทย"), got)
	}
	// measurable: a sub-query matches the subword-expanded index
	hasPart := strings.Contains(joined, " ภาษา ") || strings.Contains(joined, " ไทย ")
	if !hasPart {
		t.Errorf("no dictionary subword emitted for ภาษาไทย: %v", got)
	}
	t.Logf("ภาษาไทย → %v", got)
}

func TestAnalyzerSubwordsCJK(t *testing.T) {
	sw := Analyzer{Subwords: true}
	got := sw.Terms("北京大学")
	joined := " " + strings.Join(got, " ") + " "
	t.Logf("北京大学 → %v", got)
	if !strings.Contains(joined, " 北京大学 ") && !strings.Contains(joined, " 大学 ") {
		t.Errorf("expected coarse or subword for 北京大学: %v", got)
	}
}

func TestAnalyzerOverlayCJK(t *testing.T) {
	// A made-up character name that the base dict would split. Without overlay
	// it splits; with overlay it stays whole.
	name := "聂力" // Nie Li — likely split into 聂 / 力 by base
	base := Analyzer{}
	ov := dict.NewTrie()
	ov.Add(name)
	withOv := Analyzer{Overlay: ov}

	got := withOv.Terms(name)
	joined := " " + strings.Join(got, " ") + " "
	if !strings.Contains(joined, " "+name+" ") {
		t.Errorf("overlay CJK name not kept whole: base=%v overlay=%v", base.Terms(name), got)
	}
	t.Logf("聂力: base=%v overlay=%v", base.Terms(name), got)
}
