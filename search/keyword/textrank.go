package keyword

// TextRank ranks content words by centrality in a word co-occurrence graph
// (Mihalcea & Tarau, 2004). It is corpus-free: build an undirected graph whose
// nodes are the distinct content words and whose edges connect words that
// co-occur within a sliding window, then run weighted PageRank; the highest-
// scoring words are the keywords.
//
// Unlike RAKE, co-occurrence is not confined to a single phrase — the window
// slides across delimiters too — so a word links to its neighbours across the
// whole document, and globally connected words (recurring, well-connected)
// float to the top without any corpus. It returns single words; the
// [Keyword.Text] is the acronym-aware folded key, so "IT"/"POS" survive.
type TextRank struct {
	cfg    config
	window int
	damp   float64
	iters  int
}

// NewTextRank returns a TextRank extractor. Defaults: co-occurrence window 4,
// damping 0.85, 30 iterations — the common TextRank settings; tune with the
// With* options.
func NewTextRank(opts ...Option) *TextRank {
	return &TextRank{cfg: defaultConfig().apply(opts), window: 4, damp: 0.85, iters: 30}
}

// WithWindow sets the co-occurrence window (in content words). Larger windows
// connect more distant words. Must be ≥ 1; default 4.
func (e *TextRank) WithWindow(w int) *TextRank {
	if w >= 1 {
		e.window = w
	}
	return e
}

// Extract returns the k most central content words, best first.
func (e *TextRank) Extract(text string, k int) []Keyword {
	if k <= 0 {
		return nil
	}
	_, terms := e.cfg.analyze(text)
	words := contentKeys(terms)
	if len(words) == 0 {
		return nil
	}

	// Assign each distinct word a node id.
	id := map[string]int{}
	for _, w := range words {
		if _, ok := id[w]; !ok {
			id[w] = len(id)
		}
	}
	n := len(id)
	if n == 1 {
		return []Keyword{{Text: words[0], Score: 1}}
	}

	// Undirected weighted co-occurrence graph within the sliding window.
	adj := make([]map[int]float64, n)
	for i := range adj {
		adj[i] = map[int]float64{}
	}
	for i := 0; i < len(words); i++ {
		a := id[words[i]]
		for j := i + 1; j < len(words) && j <= i+e.window; j++ {
			b := id[words[j]]
			if a == b {
				continue
			}
			adj[a][b]++
			adj[b][a]++
		}
	}

	// Weighted PageRank. Nodes with no edges keep the base rank.
	outW := make([]float64, n)
	for i := 0; i < n; i++ {
		for _, w := range adj[i] {
			outW[i] += w
		}
	}
	score := make([]float64, n)
	next := make([]float64, n)
	for i := range score {
		score[i] = 1.0 / float64(n)
	}
	base := (1 - e.damp) / float64(n)
	for it := 0; it < e.iters; it++ {
		for i := 0; i < n; i++ {
			var sum float64
			for j, w := range adj[i] {
				if outW[j] > 0 {
					sum += w / outW[j] * score[j]
				}
			}
			next[i] = base + e.damp*sum
		}
		score, next = next, score
	}

	name := make([]string, n)
	for w, i := range id {
		name[i] = w
	}
	kws := make([]Keyword, n)
	for i := range score {
		kws[i] = Keyword{Text: name[i], Score: score[i]}
	}
	return topK(kws, k)
}
