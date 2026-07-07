// Package query turns a user's search QUERY into terms an index can actually
// match. It is the query-side half of retrieval: the document side is tokenized
// and indexed once (multi/tokenize + vocab + search/keyword), but a real user
// does not type index terms — they type one of two things that both miss even
// when the answer is indexed:
//
//	LONG, conversational: "ช่วยสรุป SOP ที่เกี่ยวการจัดซื้อจัดจ้างหน่อย เอาแผนก it กับ pc นะ"
//	    — the intent (SOP, จัดซื้อจัดจ้าง) and the entities (IT, PC) are buried in
//	    filler (ช่วย/สรุป/หน่อย/นะ); embedding or indexing the raw sentence drifts
//	    away from the terse, formal document.
//	SHORT, bare acronym: "it", "pc", "hr", "กุ้ง it"
//	    — no context at all, and the lower-cased acronym looks exactly like an
//	    English stop word, so the naive pipeline drops it and searches for
//	    nothing.
//
// Two primitives fix this, both pure-Go and corpus-free:
//
//	Parse       — strip conversational filler and question particles and generic
//	              verbs of intent, KEEP the content entities and acronyms, and
//	              RETAIN a bare acronym ("it"→"IT") the stop list would otherwise
//	              drop. Query-tuned keyword extraction (see search/keyword).
//	Expand      — expand an acronym to its equivalent forms (case + script
//	              variants + a curated Thai-enterprise seed, bidirectional) so a
//	              one-word query "pc" also matches พีซี / คอมพิวเตอร์.
//	ParseExpand — Parse, then Expand every term: the full query-understanding
//	              path, ready to hand to a sparse or hybrid matcher.
//
// # Acronym awareness is the crux
//
// The stopwords package matches case-sensitively on purpose ("it" is a stop
// word, "IT" is not), and search/keyword folds a document token to an
// acronym-aware key. A query does the SAME fold, plus one extra step the
// document side does not need: a lower-case token that would be dropped is
// checked against the acronym seed first, so "it"/"pc"/"hr" survive as the
// upper-case acronyms "IT"/"PC"/"HR" that the index actually holds. A stuck form
// like "ปลาit" is split at the script boundary by the tokenizer into "ปลา" and
// "it", and the "it" is then retained — so both spaced "ปลา IT" and stuck
// "ปลาit" reach the same terms.
//
// # Data, not policy
//
// The filler list (data/query_filler_th.txt) and the acronym seed
// (data/acronyms_th.txt) are editable data files, and every org has its own
// abbreviations, so WithAliases injects a per-tenant map that merges into the
// seed. What the parser does NOT do is decide how the terms are matched, fused
// or ranked — that is the platform's retrieval policy.
//
// # Concurrency
//
// A Parser is immutable after New and safe for concurrent Parse/Expand calls
// (the analyzer, stop set and acronym table are all read-only).
package query

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/sukitss/thai-nlp-go/multi"
	"github.com/sukitss/thai-nlp-go/search/keyword"
	"github.com/sukitss/thai-nlp-go/stopwords"
)

// Parser holds a configured query-understanding pipeline: the tokenization
// front-end, the drop-set (base stop words unioned with the query filler list),
// the acronym table, and an optional salience extractor. Build one with New and
// share it.
type Parser struct {
	analyzer   *multi.Analyzer
	stop       *stopwords.Set // base ∪ filler; consulted after acronym retention
	acr        *acronyms
	extractor  keyword.Extractor
	minWordLen int
	dropNumber bool

	// resolved from options in New:
	baseStop *stopwords.Set      // WithStopwords; nil ⇒ Multilingual
	aliases  map[string][]string // WithAliases, merged into acr
}

// Option configures a Parser at construction.
type Option func(*Parser)

// WithAnalyzer sets the tokenization front-end. Use it to attach a per-tenant
// dictionary Overlay (character names, product codes) so those survive as whole
// tokens, or to change CJK routing. Leave the Analyzer's Stop and LowerLatin at
// their zero values: the parser does its OWN acronym-aware folding and stop-word
// handling, and LowerLatin would fold "IT" to the stop word "it" and defeat
// acronym retention. Default: the zero Analyzer (normalize + route + tokenize).
func WithAnalyzer(a *multi.Analyzer) Option { return func(p *Parser) { p.analyzer = a } }

// WithStopwords sets the BASE stop-word set that, unioned with the built-in
// query-filler list, forms the drop-set. Pass a domain set to add corpus stop
// words; the filler list is always included on top. Matching is acronym-aware,
// and acronym retention runs before the check, so a set containing "it" still
// never drops the acronym "IT". Default: stopwords.Multilingual().
func WithStopwords(s *stopwords.Set) Option { return func(p *Parser) { p.baseStop = s } }

// WithAliases injects per-tenant acronym/abbreviation groups on top of the seed:
// each key maps to its equivalent forms, merged bidirectionally (and merged INTO
// a seed group if they share a term). A Latin key also becomes retainable, so a
// house abbreviation typed in lower case survives parsing like the built-ins.
//
//	WithAliases(map[string][]string{"ENG": {"วิศวกรรม", "ช่าง"}})
func WithAliases(m map[string][]string) Option {
	return func(p *Parser) {
		if p.aliases == nil {
			p.aliases = map[string][]string{}
		}
		for k, v := range m {
			p.aliases[k] = v
		}
	}
}

// WithExtractor sets the salience extractor backing Salient (Parse itself is
// exhaustive and does not use it). Any search/keyword.Extractor works; it is
// re-pointed at the parser's drop-set so it strips the same filler. Default: a
// corpus-free RAKE extractor.
func WithExtractor(e keyword.Extractor) Option { return func(p *Parser) { p.extractor = e } }

// WithMinWordLen drops non-acronym content words shorter than n runes (acronyms
// are always kept). Default 2.
func WithMinWordLen(n int) Option { return func(p *Parser) { p.minWordLen = n } }

// WithNumbers keeps pure-number tokens as terms (default false: numbers are
// dropped, so phone numbers and ids do not pollute the query).
func WithNumbers(keep bool) Option { return func(p *Parser) { p.dropNumber = !keep } }

// New builds a Parser from the given options.
func New(opts ...Option) *Parser {
	p := &Parser{
		analyzer:   &multi.Analyzer{},
		acr:        newAcronyms(),
		minWordLen: 2,
		dropNumber: true,
	}
	for _, o := range opts {
		o(p)
	}
	if p.analyzer == nil {
		p.analyzer = &multi.Analyzer{}
	}
	if p.minWordLen < 1 {
		p.minWordLen = 1
	}
	// Merge user aliases into the acronym table.
	for k, v := range p.aliases {
		p.acr.addGroup(append([]string{k}, v...))
	}
	// Drop-set = base stop words ∪ query filler.
	base := p.baseStop
	if base == nil {
		base = stopwords.Multilingual()
	}
	p.stop = stopwords.Union(base, fillerSet())
	if p.extractor == nil {
		p.extractor = keyword.NewRAKE(keyword.WithStopwords(p.stop))
	}
	return p
}

// Parse turns text into clean, de-duplicated search terms in first-occurrence
// order: normalized and tokenized, filler and stop words and (by default)
// numbers dropped, ordinary words folded to lower case, acronyms kept verbatim,
// and bare would-be-dropped acronyms retained as their upper-case form. Returns
// nil when nothing survives.
func (p *Parser) Parse(text string) []string {
	_, toks := p.analyzer.Tokens(text)
	if len(toks) == 0 {
		return nil
	}
	out := make([]string, 0, len(toks))
	seen := make(map[string]struct{}, len(toks))
	for _, t := range toks {
		key, ok := p.classify(t.Text)
		if !ok {
			continue
		}
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// classify maps one token surface to its retained term key, or ok=false to drop
// it. The order is the contract: fold, then acronym RETENTION (so a bare "it"
// becomes "IT" before the stop check can drop it), then stop-word and length
// filters for ordinary words.
func (p *Parser) classify(surface string) (string, bool) {
	hasLetter, hasDigit := false, false
	for _, r := range surface {
		if unicode.IsLetter(r) {
			hasLetter = true
		} else if unicode.IsDigit(r) {
			hasDigit = true
		}
	}
	switch {
	case !hasLetter && !hasDigit:
		return "", false // punctuation / symbol
	case !hasLetter:
		if p.dropNumber {
			return "", false // pure number
		}
		return surface, true
	}
	key, acronym := fold(surface)
	if !acronym {
		if kept, ok := p.acr.retain(surface); ok {
			return kept, true // "it"→"IT": a known acronym the stop list would eat
		}
	}
	if acronym {
		return key, true // real acronym: exempt from stop/length filters
	}
	if p.stop.IsStopword(key) {
		return "", false
	}
	if utf8.RuneCountInString(key) < p.minWordLen {
		return "", false
	}
	return key, true
}

// Expand returns term followed by its equivalent acronym/abbreviation forms
// (case + script variants and the seed groups), de-duplicated, term first. A
// term with no known equivalents returns just itself (plus its upper-case form
// when Latin).
func (p *Parser) Expand(term string) []string { return p.acr.expand(term) }

// ParseExpand is Parse followed by Expand on every term: the full
// query-understanding output, de-duplicated across expansions in a stable order.
// Hand this to a sparse/hybrid matcher as the query term set.
func (p *Parser) ParseExpand(text string) []string {
	terms := p.Parse(text)
	if len(terms) == 0 {
		return nil
	}
	out := make([]string, 0, len(terms)*2)
	seen := make(map[string]struct{}, len(terms)*2)
	for _, t := range terms {
		for _, v := range p.Expand(t) {
			if _, ok := seen[v]; ok {
				continue
			}
			seen[v] = struct{}{}
			out = append(out, v)
		}
	}
	return out
}

// Salient returns the k most salient parsed terms — a trimmed subset for when a
// query is long and you want only its head entities (a tag, a facet, a short
// dense-embedding string) rather than every term. It ranks the Parse terms by
// the configured extractor's salience; with k<=0 or k>=len it returns all Parse
// terms unchanged. The retrieval path is Parse/ParseExpand, not this.
func (p *Parser) Salient(text string, k int) []string {
	terms := p.Parse(text)
	if k <= 0 || k >= len(terms) {
		return terms
	}
	// Score parsed terms by the extractor's salience over the same text.
	score := map[string]float64{}
	for _, kw := range p.extractor.Extract(text, len(terms)*2) {
		for _, key := range keyword.Keys(kw.Text) {
			if s, ok := score[key]; !ok || kw.Score > s {
				score[key] = kw.Score
			}
		}
	}
	ranked := make([]string, len(terms))
	copy(ranked, terms)
	// Stable selection sort by score desc, ties by original order (already stable
	// because we only swap on a strictly greater score).
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(ranked); j++ {
			if score[ranked[j]] > score[ranked[best]] {
				best = j
			}
		}
		if best != i {
			v := ranked[best]
			copy(ranked[i+1:best+1], ranked[i:best])
			ranked[i] = v
		}
	}
	return ranked[:k]
}

// fold returns a token's acronym-aware key and whether it is an acronym: an
// all-upper-case Latin token with at least two letters is an acronym, kept
// verbatim ("IT", "POS"); everything else is lower-cased ("It"→"it",
// "Password"→"password"), leaving script-less text (Thai, digits) unchanged.
// This matches search/keyword's fold so query keys line up with index keys.
func fold(tok string) (key string, acronym bool) {
	var upper, lower int
	for _, r := range tok {
		if !unicode.IsLetter(r) {
			continue
		}
		switch {
		case unicode.IsUpper(r):
			upper++
		case unicode.IsLower(r):
			lower++
		}
	}
	if upper >= 2 && lower == 0 {
		return tok, true
	}
	return strings.ToLower(tok), false
}
