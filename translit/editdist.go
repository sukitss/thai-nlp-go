package translit

// Edit-distance helpers for fuzzy name-variant matching. Pair them with the
// phonetic keys (Key/Udom83/LK82): use a phonetic key to bucket candidates, then
// edit distance to rank/threshold — this recovers variants the keys miss.

// Levenshtein returns the rune-level edit distance between a and b
// (insertions, deletions, substitutions each cost 1).
func Levenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	if len(ra) < len(rb) { // keep the shorter as the DP row
		ra, rb = rb, ra
	}
	if len(rb) == 0 {
		return len(ra)
	}
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := i
		prevDiag := prev[0]
		prev[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			ins := cur + 1
			del := prev[j] + 1
			sub := prevDiag + cost
			cur = min3(ins, del, sub)
			prevDiag = prev[j]
			prev[j] = cur
		}
	}
	return prev[len(rb)]
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

// Similarity returns 1 - dist/maxLen in [0,1] (1 = identical). Empty vs empty is 1.
func Similarity(a, b string) float64 {
	ra, rb := []rune(a), []rune(b)
	maxLen := len(ra)
	if len(rb) > maxLen {
		maxLen = len(rb)
	}
	if maxLen == 0 {
		return 1
	}
	return 1 - float64(Levenshtein(a, b))/float64(maxLen)
}
