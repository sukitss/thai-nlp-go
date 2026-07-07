package sketch

import "math"

// CountMin is a count-min sketch: it approximates how many times each key was
// added to a stream using a fixed depth*width table of counters instead of a
// per-key map. Each of the depth rows maps a key to one of width counters with
// an independent hash and adds there; Estimate returns the MINIMUM of a key's
// counters, which can only be inflated by collisions — so a count-min sketch
// NEVER under-estimates. With width = ceil(e/epsilon) and depth = ceil(ln(1/delta)),
// the overestimate is at most epsilon*Total() with probability at least
// 1-delta.
//
// It is the tool for approximate document-frequency / term-frequency when a full
// vocabulary map is too large to hold: memory is fixed at depth*width*8 bytes
// regardless of how many distinct keys are seen.
type CountMin struct {
	w, d   int
	seed   uint64
	counts []uint64 // d rows of w counters, row-major
	total  uint64
}

// NewCountMin returns a count-min sketch with the given width and depth (both
// >= 1) seeded by seed. Prefer [NewCountMinParams] when you want to specify the
// error/probability targets directly.
func NewCountMin(width, depth int, seed uint64) *CountMin {
	if width < 1 || depth < 1 {
		panic("sketch: NewCountMin width and depth must be >= 1")
	}
	return &CountMin{w: width, d: depth, seed: seed, counts: make([]uint64, width*depth)}
}

// NewCountMinParams returns a count-min sketch sized for a target additive error
// epsilon (as a fraction of the total count) and failure probability delta,
// using width = ceil(e/epsilon) and depth = ceil(ln(1/delta)). epsilon and delta
// must be in (0, 1).
func NewCountMinParams(epsilon, delta float64, seed uint64) *CountMin {
	if !(epsilon > 0 && epsilon < 1) || !(delta > 0 && delta < 1) {
		panic("sketch: NewCountMinParams epsilon and delta must be in (0,1)")
	}
	w := int(math.Ceil(math.E / epsilon))
	d := int(math.Ceil(math.Log(1 / delta)))
	if w < 1 {
		w = 1
	}
	if d < 1 {
		d = 1
	}
	return NewCountMin(w, d, seed)
}

// Width reports the number of counters per row.
func (c *CountMin) Width() int { return c.w }

// Depth reports the number of rows (hash functions).
func (c *CountMin) Depth() int { return c.d }

// Total reports the sum of all added counts.
func (c *CountMin) Total() uint64 { return c.total }

// SizeBytes reports the counter table size in bytes (depth*width*8).
func (c *CountMin) SizeBytes() int { return len(c.counts) * 8 }

// hashes derives the two base hashes for a key; the depth row hashes are the
// linear combination h1 + i*h2 (Kirsch-Mitzenmacher double hashing), which is
// sufficient for count-min's collision-probability bound. h2 is forced odd so it
// is coprime to the counter count when width is a power of two.
func (c *CountMin) hashes(h1, h2 uint64, row int) int {
	return int((h1 + uint64(row)*h2) % uint64(c.w))
}

// Add increases the count of key by n.
func (c *CountMin) Add(key []byte, n uint64) {
	h1 := hashBytes(key, c.seed)
	h2 := hashBytes(key, c.seed^0x9e3779b97f4a7c15) | 1
	for i := 0; i < c.d; i++ {
		c.counts[i*c.w+c.hashes(h1, h2, i)] += n
	}
	c.total += n
}

// AddString increases the count of key by n without allocating.
func (c *CountMin) AddString(key string, n uint64) {
	h1 := hashString(key, c.seed)
	h2 := hashString(key, c.seed^0x9e3779b97f4a7c15) | 1
	for i := 0; i < c.d; i++ {
		c.counts[i*c.w+c.hashes(h1, h2, i)] += n
	}
	c.total += n
}

// Estimate returns the approximate count of key: the minimum over its rows. It
// never under-estimates the true count and exceeds it by at most epsilon*Total()
// with probability >= 1-delta for a sketch built via [NewCountMinParams].
func (c *CountMin) Estimate(key []byte) uint64 {
	h1 := hashBytes(key, c.seed)
	h2 := hashBytes(key, c.seed^0x9e3779b97f4a7c15) | 1
	return c.estimate(h1, h2)
}

// EstimateString is Estimate for a string key.
func (c *CountMin) EstimateString(key string) uint64 {
	h1 := hashString(key, c.seed)
	h2 := hashString(key, c.seed^0x9e3779b97f4a7c15) | 1
	return c.estimate(h1, h2)
}

func (c *CountMin) estimate(h1, h2 uint64) uint64 {
	min := ^uint64(0)
	for i := 0; i < c.d; i++ {
		v := c.counts[i*c.w+c.hashes(h1, h2, i)]
		if v < min {
			min = v
		}
	}
	return min
}
