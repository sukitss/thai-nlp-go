package dict

// Subwords returns a fine, non-overlapping segmentation of word into the
// smaller dictionary words it is built from — the fine field in a coarse/fine
// keyword scheme (as RAGFlow does). Indexing these alongside the full token
// restores recall for sub-queries: "ภาษาไทย" → "ภาษา","ไทย"; "北京大学" →
// "北京","大学". The full token is kept by the caller (this returns only the
// parts) and is never returned as its own subword.
//
// Segmentation is greedy longest-match, but the first match may not span the
// whole word (that would just be the token again) — so a compound is always
// broken at least once. Characters with no dictionary word are skipped. Returns
// nil when word has no proper sub-segmentation. max caps the parts (<=0 = no
// cap). This is a re-segmentation, not all-substrings, so no "ภา"/"ภาษ" noise.
func Subwords(d Prefixer, word string, max int) []string {
	rs := []rune(word)
	n := len(rs)
	if n < 2 || d == nil {
		return nil
	}
	var out []string
	var lens []int
	for i := 0; i < n; {
		lens = d.PrefixLens(rs, i, lens)
		best := 0
		for _, l := range lens {
			if l <= 0 {
				continue
			}
			if i == 0 && i+l == n {
				continue // the full token — force at least one split
			}
			if l > best {
				best = l
			}
		}
		if best == 0 {
			i++ // no dictionary word starts here; skip the character
			continue
		}
		out = append(out, string(rs[i:i+best]))
		if max > 0 && len(out) >= max {
			break
		}
		i += best
	}
	return out
}
