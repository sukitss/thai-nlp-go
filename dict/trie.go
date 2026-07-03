package dict

// Pointer trie — ported from PyThaiNLP's pythainlp.util.trie.Trie.
// Used to build dictionaries and as the per-session overlay backend. For the
// large base dictionary prefer FlatTrie (flattrie.go), which loads far faster.

import (
	"bufio"
	"os"
	"slices"
	"strconv"
	"strings"
)

// walkBufCap is the initial capacity of the rune scratch buffers used by
// Words/WalkPrefix — enough for typical dictionary words without regrowth.
const walkBufCap = 32

type trieNode struct {
	end      bool
	weight   int32 // per-word weight (e.g. log-frequency), 0 if unweighted
	children map[rune]*trieNode
}

// Trie is a rune-keyed prefix tree of dictionary words.
//
// Trie has no internal locking by design (lookups must stay lock- and
// allocation-free): build it fully, then treat it as read-only — concurrent
// readers are safe. To change a dictionary that is already shared across
// goroutines, rebuild-and-swap: build a new Trie and atomically swap the
// pointer; never Add to a live one.
type Trie struct {
	root     *trieNode
	count    int
	weighted bool // true once any word was added with a weight
}

// NewTrie returns an empty Trie.
func NewTrie() *Trie { return &Trie{root: &trieNode{}} }

// Len reports how many distinct words the trie holds.
func (t *Trie) Len() int { return t.count }

// Add inserts a word (surrounding whitespace is trimmed; blanks are ignored).
func (t *Trie) Add(word string) {
	word = strings.TrimSpace(word)
	if word == "" {
		return
	}
	cur := t.root
	for _, ch := range word {
		if cur.children == nil {
			cur.children = make(map[rune]*trieNode)
		}
		child := cur.children[ch]
		if child == nil {
			child = &trieNode{}
			cur.children[ch] = child
		}
		cur = child
	}
	if !cur.end {
		cur.end = true
		t.count++
	}
}

// AddWeighted inserts a word carrying a weight (e.g. an integer log-frequency
// used by DAG maximum-probability segmentation). Otherwise like Add.
func (t *Trie) AddWeighted(word string, weight int32) {
	word = strings.TrimSpace(word)
	if word == "" {
		return
	}
	cur := t.root
	for _, ch := range word {
		if cur.children == nil {
			cur.children = make(map[rune]*trieNode)
		}
		child := cur.children[ch]
		if child == nil {
			child = &trieNode{}
			cur.children[ch] = child
		}
		cur = child
	}
	if !cur.end {
		cur.end = true
		t.count++
	}
	cur.weight = weight
	t.weighted = true
}

// PrefixLens appends, to out, the rune-lengths L such that text[start:start+L]
// is a word in the trie. Lengths come out in ascending order (short → long),
// matching PyThaiNLP's Trie.prefixes walk order. out is reset before use.
func (t *Trie) PrefixLens(text []rune, start int, out []int) []int {
	out = out[:0]
	cur := t.root
	n := len(text)
	for i := start; i < n; i++ {
		if cur.children == nil {
			break
		}
		node := cur.children[text[i]]
		if node == nil {
			break
		}
		if node.end {
			out = append(out, i+1-start)
		}
		cur = node
	}
	return out
}

// PrefixWeights appends, for each dictionary word that is a prefix of
// text[start:], a (rune-length, weight) pair — the weighted form of PrefixLens.
// outLen and outW are reset and kept in sync. Words added without a weight
// report 0. See FlatTrie.PrefixWeights for the contract — the two are
// interchangeable.
func (t *Trie) PrefixWeights(text []rune, start int, outLen, outW []int32) ([]int32, []int32) {
	outLen, outW = outLen[:0], outW[:0]
	cur := t.root
	n := len(text)
	for i := start; i < n; i++ {
		cur = cur.children[text[i]]
		if cur == nil {
			break
		}
		if cur.end {
			outLen = append(outLen, int32(i+1-start))
			outW = append(outW, cur.weight)
		}
	}
	return outLen, outW
}

// Weighted reports whether any word was added with AddWeighted. It is never
// cleared, even if every weighted word is later Removed.
func (t *Trie) Weighted() bool { return t.weighted }

// Weight returns the stored weight of an exact word and whether the word is in
// the trie. Words added without a weight report 0. See FlatTrie.Weight — the
// two are interchangeable.
func (t *Trie) Weight(word string) (int32, bool) {
	cur := t.root
	for _, r := range word {
		cur = cur.children[r]
		if cur == nil {
			return 0, false
		}
	}
	if !cur.end {
		return 0, false
	}
	return cur.weight, true
}

// Contains reports whether word is in the trie.
func (t *Trie) Contains(word string) bool {
	_, ok := t.Weight(word)
	return ok
}

// Remove deletes a word, reporting whether it was present (surrounding
// whitespace is trimmed, matching Add). The word's weight is cleared with it.
// Nodes left with no words below them are pruned — cheap here because the
// descent already collected the path — so a removed branch is fully reclaimed;
// a word that prefixes longer words only has its end flag cleared. Remove
// follows the same concurrency contract as Add: never mutate a trie that is
// shared across goroutines — rebuild-and-swap instead.
func (t *Trie) Remove(word string) bool {
	word = strings.TrimSpace(word)
	if word == "" {
		return false
	}
	runes := []rune(word)
	path := make([]*trieNode, len(runes)+1)
	path[0] = t.root
	cur := t.root
	for i, r := range runes {
		cur = cur.children[r]
		if cur == nil {
			return false
		}
		path[i+1] = cur
	}
	if !cur.end {
		return false
	}
	cur.end = false
	cur.weight = 0
	t.count--
	for i := len(runes); i >= 1; i-- {
		nd := path[i]
		if nd.end || len(nd.children) > 0 {
			break
		}
		parent := path[i-1]
		delete(parent.children, runes[i-1])
		if len(parent.children) == 0 {
			parent.children = nil
		}
	}
	return true
}

// Words calls fn for every word in the trie, in lexicographic rune order, until
// fn returns false. weight is the word's stored weight — 0 for words added
// without one (so unweighted tries always pass 0). Equivalent to
// WalkPrefix("", fn).
func (t *Trie) Words(fn func(word string, weight int32) bool) { t.WalkPrefix("", fn) }

// WalkPrefix calls fn for every dictionary word starting with prefix (including
// prefix itself if it is a word), in lexicographic rune order, until fn returns
// false — e.g. autocompleting glossary entries or expanding a prefix into its
// dictionary terms. It descends to the prefix node once, then walks only that
// subtree. Beyond the string handed to fn, allocation is limited to two
// amortized rune buffers (the current word and the per-level sorted child
// runes).
func (t *Trie) WalkPrefix(prefix string, fn func(word string, weight int32) bool) {
	cur := t.root
	word := make([]rune, 0, walkBufCap)
	for _, r := range prefix {
		cur = cur.children[r]
		if cur == nil {
			return
		}
		word = append(word, r)
	}
	keys := make([]rune, 0, walkBufCap)
	walkTrieNode(cur, &word, &keys, fn)
}

// walkTrieNode does a lexicographic DFS below nd. word holds the runes on the
// path down to nd; keys is a shared scratch where each recursion level appends
// (then truncates) its own sorted segment of child runes, so the whole walk
// reuses one buffer instead of allocating per node. Returns false as soon as fn
// does, unwinding the walk.
func walkTrieNode(nd *trieNode, word, keys *[]rune, fn func(string, int32) bool) bool {
	if nd.end && !fn(string(*word), nd.weight) {
		return false
	}
	base := len(*keys)
	for r := range nd.children {
		*keys = append(*keys, r)
	}
	slices.Sort((*keys)[base:])
	end := len(*keys)
	for i := base; i < end; i++ {
		r := (*keys)[i]
		*word = append(*word, r)
		ok := walkTrieNode(nd.children[r], word, keys, fn)
		*word = (*word)[:len(*word)-1]
		if !ok {
			*keys = (*keys)[:base]
			return false
		}
	}
	*keys = (*keys)[:base]
	return true
}

// LoadDict reads a dictionary text file (one word per line) into a Trie.
func LoadDict(path string) (*Trie, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	t := NewTrie()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		t.Add(sc.Text())
	}
	return t, sc.Err()
}

// LoadDictWeighted reads a dictionary file of "word<TAB>weight" lines (weight
// is a base-10 int32, e.g. a scaled log-frequency) into a weighted Trie. Lines
// without a tab are added unweighted; a line whose weight field does not parse
// as an int32 (garbage, overflow, empty) is skipped entirely — like blank
// lines, malformed lines never add a word or corrupt weights.
func LoadDictWeighted(path string) (*Trie, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	t := NewTrie()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if tab := strings.IndexByte(line, '\t'); tab >= 0 {
			w, err := strconv.ParseInt(strings.TrimSpace(line[tab+1:]), 10, 32)
			if err != nil {
				continue
			}
			t.AddWeighted(line[:tab], int32(w))
		} else {
			t.Add(line)
		}
	}
	return t, sc.Err()
}
