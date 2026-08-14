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
	var parents []Term
	for i, t := range in {
		parents = parents[:0]
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
			parents = append(parents, other)
		}
		// Count each occurrence once. Longer candidates nest — every
		// occurrence of "ที่จะมา" is also an occurrence of "ที่จะ" — so adding
		// up every longer candidate double-counts, and for a short common word
		// with many extensions it can exceed the word's own count and condemn
		// it. Only the outermost ones describe distinct occurrences.
		inside := 0
		for a, p := range parents {
			nested := false
			for b, q := range parents {
				if a != b && indexTokens(q.Tokens, p.Tokens) >= 0 {
					nested = true
					break
				}
			}
			if !nested {
				inside += p.Count
			}
		}
		if inside > 0 && float64(t.Count-inside) < fragmentShare*float64(t.Count) {
			continue
		}
		out = append(out, t)
	}
	return dropWrappers(out, unigram, o)
}

// dropWrappers removes a term that is another surviving term plus ordinary
// words — "คุณเอลินา" once "เอลินา" is known, "ทานูกิหุ้มเกราะ" once "ทานูกิ"
// is.
//
// Keeping both is worse than useless for the usual purpose of this package,
// which is to feed a tokenizer's dictionary: a longest-match tokenizer would
// take the longer entry every time, and then a search for the name itself finds
// nothing. The shorter one is also the thing a reader would look up.
func dropWrappers(in []Term, unigram map[string]int, o Options) []Term {
	if o.Affix == nil || len(in) < 2 {
		return in
	}
	out := in[:0]
	for i, t := range in {
		wrapper := false
		for j, inner := range in {
			if i == j || len(inner.Tokens) >= len(t.Tokens) {
				continue
			}
			at := indexTokens(t.Tokens, inner.Tokens)
			if at < 0 {
				continue
			}
			if allAffix(t.Tokens[:at], unigram, t.Count, o) &&
				allAffix(t.Tokens[at+len(inner.Tokens):], unigram, t.Count, o) {
				wrapper = true
				break
			}
		}
		if !wrapper {
			out = append(out, t)
		}
	}
	return out
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
