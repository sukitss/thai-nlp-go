package dict

// OverlayDict layers a small per-session word list on top of a shared read-only
// base dictionary. Lookups return the union of both, so session-specific words
// (custom names, product codes, a domain glossary) are recognized without
// rebuilding or copying the large base. Create one via Segmenter.Session.
//
// An OverlayDict is single-writer/single-reader: it keeps scratch buffers to
// avoid per-lookup allocation, so it is NOT safe for concurrent use. Give each
// goroutine its own Session (they all share the base safely).
type OverlayDict struct {
	base    Prefixer
	baseW   Weighter // non-nil when base also supports weighted lookups
	overlay *Trie
	ovbuf   []int   // scratch: overlay lengths
	merged  []int   // scratch: merged result
	baseLen []int   // scratch: base lengths for the Prefixer-only base fallback
	ovLen   []int32 // scratch: overlay weighted lengths
	ovW     []int32 // scratch: overlay weights
	mLen    []int32 // scratch: merged weighted lengths
	mW      []int32 // scratch: merged weights
}

// NewOverlayDict wraps base with an overlay trie of extra words.
func NewOverlayDict(base Prefixer, overlay *Trie) *OverlayDict {
	o := &OverlayDict{base: base, overlay: overlay}
	o.baseW, _ = base.(Weighter)
	return o
}

// PrefixLens returns the ascending, de-duplicated union of the base and overlay
// prefix lengths. When the overlay contributes nothing it returns the base
// result directly (the common, hot case).
func (o *OverlayDict) PrefixLens(text []rune, start int, out []int) []int {
	base := o.base.PrefixLens(text, start, out)
	o.ovbuf = o.overlay.PrefixLens(text, start, o.ovbuf[:0])
	if len(o.ovbuf) == 0 {
		return base
	}
	o.merged = mergeAscUnique(o.merged[:0], base, o.ovbuf)
	return o.merged
}

// PrefixWeights is the weighted form of PrefixLens (OverlayDict satisfies
// Weighter): the same ascending, de-duplicated union of lengths, with each
// length's weight alongside. Overlay-only words carry the overlay weight and
// base-only words the base weight. For a word found in both, the overlay's
// weight shadows the base's when the overlay is weighted (built with
// AddWeighted — a per-tenant glossary can re-weight a base word); an unweighted
// overlay carries no weight opinion, so the base weight passes through. A base
// that only implements Prefixer contributes its lengths with weight 0.
func (o *OverlayDict) PrefixWeights(text []rune, start int, outLen, outW []int32) ([]int32, []int32) {
	if o.baseW != nil {
		outLen, outW = o.baseW.PrefixWeights(text, start, outLen, outW)
	} else {
		o.baseLen = o.base.PrefixLens(text, start, o.baseLen[:0])
		outLen, outW = outLen[:0], outW[:0]
		for _, l := range o.baseLen {
			outLen = append(outLen, int32(l))
			outW = append(outW, 0)
		}
	}
	o.ovLen, o.ovW = o.overlay.PrefixWeights(text, start, o.ovLen[:0], o.ovW[:0])
	if len(o.ovLen) == 0 {
		return outLen, outW
	}
	o.mLen, o.mW = mergeWeightedAscUnique(o.mLen[:0], o.mW[:0], outLen, outW, o.ovLen, o.ovW, o.overlay.Weighted())
	return o.mLen, o.mW
}

// mergeWeightedAscUnique merges two ascending (length, weight) pair lists into
// dstL/dstW, dropping duplicate lengths. On a duplicate the a pair is kept,
// unless bWins is set — then the b (overlay) pair shadows it. dstL/dstW are
// expected to be reset (len 0) scratch slices.
func mergeWeightedAscUnique(dstL, dstW, aL, aW, bL, bW []int32, bWins bool) ([]int32, []int32) {
	i, j := 0, 0
	for i < len(aL) && j < len(bL) {
		switch {
		case aL[i] < bL[j]:
			dstL, dstW = append(dstL, aL[i]), append(dstW, aW[i])
			i++
		case aL[i] > bL[j]:
			dstL, dstW = append(dstL, bL[j]), append(dstW, bW[j])
			j++
		default:
			if bWins {
				dstL, dstW = append(dstL, bL[j]), append(dstW, bW[j])
			} else {
				dstL, dstW = append(dstL, aL[i]), append(dstW, aW[i])
			}
			i++
			j++
		}
	}
	for ; i < len(aL); i++ {
		dstL, dstW = append(dstL, aL[i]), append(dstW, aW[i])
	}
	for ; j < len(bL); j++ {
		dstL, dstW = append(dstL, bL[j]), append(dstW, bW[j])
	}
	return dstL, dstW
}

// mergeAscUnique merges two ascending, individually-unique int slices into dst,
// dropping duplicates. dst is expected to be a reset (len 0) scratch slice.
func mergeAscUnique(dst, a, b []int) []int {
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			dst = append(dst, a[i])
			i++
		case a[i] > b[j]:
			dst = append(dst, b[j])
			j++
		default:
			dst = append(dst, a[i])
			i++
			j++
		}
	}
	dst = append(dst, a[i:]...)
	dst = append(dst, b[j:]...)
	return dst
}
