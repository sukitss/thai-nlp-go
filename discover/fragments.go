package discover

import "sort"

// fragmentShare is how much of a candidate's occurrences may be explained by
// longer candidates before it is judged a fragment of them. A third leaves room
// for a term that genuinely stands alone sometimes and appears inside a longer
// name at other times.
const fragmentShare = 0.33

func sortByCount(in []Term) []Term {
	sort.Slice(in, func(i, j int) bool {
		if in[i].Count != in[j].Count {
			return in[i].Count > in[j].Count
		}
		return in[i].Text < in[j].Text
	})
	return in
}

// dropFragments removes candidates that exist only because a longer one does.
//
// The direction of this test is the whole point, and the obvious version has it
// backwards. Deleting the longer candidate whenever a shorter one is more
// frequent looks reasonable and is wrong: on a Thai corpus it deleted "ก็อบลิน"
// (goblin, 141 occurrences) in favour of "ก็อบ" (359) — and "ก็อบ" is not a word
// at all, it is the shared front of ก็อบลิน, ก็อบคิจิ and ก็อบมิ.
//
// So the question is not which one is bigger. It is whether a candidate is ever
// seen standing on its own: one whose occurrences are nearly all accounted for
// by longer candidates is a piece of them, however the counts fall.
//
// One longer candidate does not count as an explanation, though: the one that
// merely bolts ordinary words onto this one. A borrowing used in a fixed phrase
// ("ทานูกิ", always followed by "หุ้มเกราะ") is otherwise swallowed by that
// phrase, and the phrase is not the thing a reader looks up — the borrowing is.
// See Options.Affix for how the wrapping words are told apart from the parts of
// a name; a caller that leaves it nil keeps the plain rule.
func dropFragments(in []Term, unigram map[string]int, o Options) []Term {
	if len(in) < 2 {
		return in
	}
	out := in[:0]
	for i, t := range in {
		inside := 0
		for j, other := range in {
			if i == j {
				continue
			}
			at := indexTokens(other.Tokens, t.Tokens)
			if at < 0 {
				continue
			}
			if o.Affix != nil &&
				allAffix(other.Tokens[:at], unigram, other.Count, o) &&
				allAffix(other.Tokens[at+len(t.Tokens):], unigram, other.Count, o) {
				// `other` is this candidate wrapped in ordinary vocabulary. It
				// does not explain this one away; if anything, this one
				// explains it.
				continue
			}
			inside += other.Count
		}
		// This sum over-counts, and deliberately so. Longer candidates nest and
		// overlap — one occurrence of "ที่" inside "ในที่จะ" is counted by both
		// "ในที่" and "ที่จะ" — so `inside` is an upper bound on how much of
		// this candidate longer ones explain, not the exact figure. Counting
		// exactly would take a second pass over the corpus with positions.
		//
		// The bias runs the safe way. Over-counting makes this test stricter,
		// and the thing it is defending against is debris: "หนิงเอ๋อร์" (205)
		// left over from "เซียวหนิงเอ๋อร์" (201), which reads as a name, passes
		// every other test, and would poison a tokenizer dictionary. It was
		// tried the exact way, counting only outermost parents, and the debris
		// came straight back — a name has several extensions and the outermost
		// ones are individually small. What the strictness costs is short
		// common words with many extensions, which have no edges to find in the
		// first place (see the holdout benchmark in discover/thai).
		if inside > 0 && float64(t.Count-inside) < fragmentShare*float64(t.Count) {
			continue
		}
		out = append(out, t)
	}
	return dropWrappers(out, unigram, o)
}

// dropWrappers removes a term that is a core wrapped in ordinary words:
// "คุณเอลินา" around "เอลินา", "ทานูกิหุ้มเกราะ" around "ทานูกิ".
//
// Keeping both is worse than useless for the usual purpose of this package,
// which is to feed a tokenizer's dictionary: longest-match would take the
// longer entry every time, and a search for the name itself would find nothing.
// The shorter one is also the thing a reader would look up.
//
// Stripping stops when fewer than two tokens are left, so a term made entirely
// of ordinary words that earned its place by cohesion — "กระต่ายเขาแหลม" — is
// never taken apart.
func dropWrappers(in []Term, unigram map[string]int, o Options) []Term {
	if o.Affix == nil || len(in) < 2 {
		return in
	}
	have := make(map[string]bool, len(in))
	for _, t := range in {
		have[t.Text] = true
	}
	out := in[:0]
	for _, t := range in {
		core, _ := stripAffixes(t, unigram, o)
		if core == "" || !have[core] {
			out = append(out, t)
		}
	}
	return out
}

// stripAffixes peels ordinary words off both ends of a term and returns what is
// left, plus a signature of what was removed. It returns "" when nothing was
// removed or when too little is left to be a term.
func stripAffixes(t Term, unigram map[string]int, o Options) (core, wrap string) {
	lo, hi := 0, len(t.Tokens)
	for lo < hi && allAffix(t.Tokens[lo:lo+1], unigram, t.Count, o) {
		wrap += t.Tokens[lo] + "\x00"
		lo++
	}
	for hi > lo && allAffix(t.Tokens[hi-1:hi], unigram, t.Count, o) {
		wrap += "\x00" + t.Tokens[hi-1]
		hi--
	}
	if wrap == "" || hi-lo < 2 {
		return "", ""
	}
	return o.join()(t.Tokens[lo:hi]), wrap
}

// indexTokens returns where sub starts inside tokens as a contiguous run, or -1.
// Comparing tokens rather than joined text keeps "of the" from being found
// inside a word, which the string form would do in any language written without
// spaces — and Thai is one.
func indexTokens(tokens, sub []string) int {
	if len(sub) == 0 || len(sub) >= len(tokens) {
		return -1
	}
	for i := 0; i+len(sub) <= len(tokens); i++ {
		match := true
		for j, s := range sub {
			if tokens[i+j] != s {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

// allAffix reports whether every one of these tokens is ordinary vocabulary
// that also demonstrably lives outside this term.
func allAffix(tokens []string, unigram map[string]int, count int, o Options) bool {
	for _, t := range tokens {
		if !o.Affix(t) {
			return false
		}
		total := unigram[t]
		if total <= 0 || float64(total-count)/float64(total) < o.AffixFreeShare {
			return false
		}
	}
	return true
}
