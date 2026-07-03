package dict

// Pointer trie — ported from PyThaiNLP's pythainlp.util.trie.Trie.
// Used to build dictionaries and as the per-session overlay backend. For the
// large base dictionary prefer FlatTrie (flattrie.go), which loads far faster.

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

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
