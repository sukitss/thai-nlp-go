package dict

// FlatTrie — a flat (CSR) serialization of the dictionary trie that loads in
// microseconds via mmap + zero-copy cast (vs tens of ms to rebuild a pointer
// trie). This is what makes cold-start cheap enough to spawn many workers.
//
// Build it once offline with BuildFlatFromTrie, then load with OpenFlat (mmap),
// ReadFlat (eager) or FromBytes (e.g. go:embed). Lookups are rune-native so the
// output is identical to the pointer Trie.

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"unsafe"

	mmap "github.com/blevesearch/mmap-go"
)

// File layout (little-endian, every section 4-byte aligned):
//
//	[0]  magic uint32 | numNodes uint32 | numEdges uint32 | reserved uint32
//	edgeStart [numNodes+1] uint32   (CSR offsets into the edge arrays)
//	endBits   [ceil(numNodes/32)] uint32  (bitset: node terminates a word)
//	edgeRune  [numEdges] int32      (edge rune, sorted per node)
//	edgeTgt   [numEdges] uint32     (target node)
//	node 0 = root
const (
	flatMagic     uint32 = 0x4E4D4654 // "NMFT"
	flatHeaderLen        = 16
	flatWordBytes        = 4 // every section is an array of 4-byte words

	flagWeighted uint32 = 1 // header reserved field: trailing weights section
)

// FlatTrie is a read-only, memory-mapped dictionary. Safe for concurrent reads.
type FlatTrie struct {
	data      mmap.MMap // mmap region (retained to prevent GC/unmap)
	keep      []uint64  // eager/embedded buffer (retained to prevent GC)
	edgeStart []uint32
	endBits   []uint32
	edgeRune  []int32
	edgeTgt   []uint32
	weights   []int32 // per-node word weight; nil if the dictionary is unweighted
}

func (f *FlatTrie) isEnd(node uint32) bool {
	return f.endBits[node>>5]&(1<<(node&31)) != 0
}

// Weighted reports whether the dictionary carries per-word weights (for DAG
// maximum-probability segmentation).
func (f *FlatTrie) Weighted() bool { return f.weights != nil }

// Weight returns the stored weight of an exact word (e.g. a scaled log-frequency
// for Chinese/Thai, or -cost for Japanese) and whether the word is in the
// dictionary. On an unweighted dictionary the weight is 0. Useful for IDF-style
// term weighting in a retriever.
func (f *FlatTrie) Weight(word string) (int32, bool) {
	var cur uint32
	for _, r := range word {
		lo, hi := f.edgeStart[cur], f.edgeStart[cur+1]
		rr := int32(r)
		found := false
		for lo < hi {
			mid := (lo + hi) >> 1
			v := f.edgeRune[mid]
			if v == rr {
				cur = f.edgeTgt[mid]
				found = true
				break
			} else if v < rr {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if !found {
			return 0, false
		}
	}
	if !f.isEnd(cur) {
		return 0, false
	}
	if f.weights != nil {
		return f.weights[cur], true
	}
	return 0, true
}

// Contains reports whether word is in the dictionary.
func (f *FlatTrie) Contains(word string) bool {
	_, ok := f.Weight(word)
	return ok
}

// PrefixWeights appends, for each dictionary word that is a prefix of
// text[start:], a (rune-length, weight) pair — the weighted form of PrefixLens,
// for DAG/DP segmentation. outLen and outW are reset and kept in sync. If the
// dictionary is unweighted, weights are 0.
func (f *FlatTrie) PrefixWeights(text []rune, start int, outLen, outW []int32) ([]int32, []int32) {
	outLen, outW = outLen[:0], outW[:0]
	var cur uint32
	n := len(text)
	for i := start; i < n; i++ {
		lo, hi := f.edgeStart[cur], f.edgeStart[cur+1]
		r := int32(text[i])
		found := false
		for lo < hi {
			mid := (lo + hi) >> 1
			v := f.edgeRune[mid]
			if v == r {
				cur = f.edgeTgt[mid]
				found = true
				break
			} else if v < r {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if !found {
			break
		}
		if f.isEnd(cur) {
			outLen = append(outLen, int32(i+1-start))
			if f.weights != nil {
				outW = append(outW, f.weights[cur])
			} else {
				outW = append(outW, 0)
			}
		}
	}
	return outLen, outW
}

// PrefixLens appends the rune-lengths of every dictionary word that is a prefix
// of text[start:], in ascending order. out is reset before use. See
// Trie.PrefixLens for the contract — the two are interchangeable.
func (f *FlatTrie) PrefixLens(text []rune, start int, out []int) []int {
	out = out[:0]
	var cur uint32 // root
	n := len(text)
	for i := start; i < n; i++ {
		lo, hi := f.edgeStart[cur], f.edgeStart[cur+1]
		r := int32(text[i])
		found := false
		for lo < hi { // binary search (children sorted by rune)
			mid := (lo + hi) >> 1
			v := f.edgeRune[mid]
			if v == r {
				cur = f.edgeTgt[mid]
				found = true
				break
			} else if v < r {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if !found {
			break
		}
		if f.isEnd(cur) {
			out = append(out, i+1-start)
		}
	}
	return out
}

// WalkPrefix calls fn for every dictionary word starting with prefix (including
// prefix itself if it is a word), in lexicographic rune order, until fn returns
// false — e.g. autocompleting glossary entries or expanding a prefix into its
// dictionary terms. Weights are 0 on an unweighted dictionary. It descends to
// the prefix node once (binary search per rune), then walks only that subtree;
// edges are stored rune-sorted, so ordering costs nothing. See Trie.WalkPrefix
// — the two are interchangeable.
func (f *FlatTrie) WalkPrefix(prefix string, fn func(word string, weight int32) bool) {
	var cur uint32
	word := make([]rune, 0, walkBufCap)
	for _, r := range prefix {
		lo, hi := f.edgeStart[cur], f.edgeStart[cur+1]
		rr := int32(r)
		found := false
		for lo < hi {
			mid := (lo + hi) >> 1
			v := f.edgeRune[mid]
			if v == rr {
				cur = f.edgeTgt[mid]
				found = true
				break
			} else if v < rr {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		if !found {
			return
		}
		word = append(word, r)
	}
	f.walkNode(cur, &word, fn)
}

// walkNode does a lexicographic DFS below node, reusing one rune buffer for the
// current word. Returns false as soon as fn does, unwinding the walk.
func (f *FlatTrie) walkNode(node uint32, word *[]rune, fn func(string, int32) bool) bool {
	if f.isEnd(node) {
		var w int32
		if f.weights != nil {
			w = f.weights[node]
		}
		if !fn(string(*word), w) {
			return false
		}
	}
	for i := f.edgeStart[node]; i < f.edgeStart[node+1]; i++ {
		*word = append(*word, rune(f.edgeRune[i]))
		ok := f.walkNode(f.edgeTgt[i], word, fn)
		*word = (*word)[:len(*word)-1]
		if !ok {
			return false
		}
	}
	return true
}

// Close releases the trie's backing memory (unmapping the mmap region, if any)
// and invalidates it: lookups after Close panic recoverably (nil-slice index)
// instead of touching unmapped memory and killing the process with SIGSEGV.
// Close is idempotent. The caller must ensure no goroutine can still be using
// the trie when Close runs — to replace a live dictionary, build the new trie,
// atomically swap the pointer, and Close the old one only after all readers
// are done (refcounting/quiescing is the caller's job).
func (f *FlatTrie) Close() error {
	f.edgeStart, f.endBits, f.edgeRune, f.edgeTgt, f.weights = nil, nil, nil, nil, nil
	f.keep = nil
	data := f.data
	f.data = nil
	if data != nil {
		return data.Unmap()
	}
	return nil
}

// ---------- build (offline, once) ----------

// flatSections is the CSR image of a pointer Trie, ready to serialize.
type flatSections struct {
	numNodes  int
	edgeStart []uint32
	endBits   []uint32
	edgeRune  []int32
	edgeTgt   []uint32
	weights   []int32 // nil when the trie is unweighted
}

// byteSize is the exact serialized size (header + all sections).
func (fl *flatSections) byteSize() int {
	words := len(fl.edgeStart) + len(fl.endBits) + len(fl.edgeRune) + len(fl.edgeTgt) + len(fl.weights)
	return flatHeaderLen + flatWordBytes*words
}

// flatten converts a pointer Trie into its flat CSR sections. Node indices are
// assigned by BFS over rune-sorted children, so the output is deterministic for
// a given word set.
func flatten(t *Trie) flatSections {
	idx := map[*trieNode]uint32{t.root: 0}
	order := []*trieNode{t.root}
	for i := 0; i < len(order); i++ {
		nd := order[i]
		for _, r := range sortedRunes(nd) {
			c := nd.children[r]
			if _, ok := idx[c]; !ok {
				idx[c] = uint32(len(order))
				order = append(order, c)
			}
		}
	}
	numNodes := len(order)
	fl := flatSections{
		numNodes:  numNodes,
		edgeStart: make([]uint32, numNodes+1),
		endBits:   make([]uint32, (numNodes+31)/32),
	}
	if t.weighted {
		fl.weights = make([]int32, numNodes)
	}
	for i, nd := range order {
		fl.edgeStart[i] = uint32(len(fl.edgeRune))
		if nd.end {
			fl.endBits[i>>5] |= 1 << (uint(i) & 31)
		}
		if fl.weights != nil {
			fl.weights[i] = nd.weight
		}
		for _, r := range sortedRunes(nd) {
			fl.edgeRune = append(fl.edgeRune, int32(r))
			fl.edgeTgt = append(fl.edgeTgt, idx[nd.children[r]])
		}
	}
	fl.edgeStart[numNodes] = uint32(len(fl.edgeRune))
	return fl
}

// BuildFlatFromTrie converts a pointer Trie into the flat CSR format and writes
// it to path. Run this whenever the dictionary text changes.
func BuildFlatFromTrie(t *Trie, path string) error {
	fl := flatten(t)
	// Write to a temp file in the same directory, then rename over the target:
	// atomic on the same filesystem, and flush/close errors are never dropped,
	// so a failed build cannot leave a truncated .fdt behind.
	f, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	if err := writeFlat(f, fl); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// WriteFlat serializes the trie to w in the flat CSR format — byte-identical to
// what BuildFlatFromTrie writes to disk — for persisting a dictionary somewhere
// other than a local file (object store, database blob). Load it back with
// FromBytes. Unlike BuildFlatFromTrie there is no atomicity: on error w may
// have received a partial stream, so write to a staging location and promote
// only on success.
func (t *Trie) WriteFlat(w io.Writer) error { return writeFlat(w, flatten(t)) }

// FlatBytes serializes the trie to flat-format bytes in memory (WriteFlat into
// an exactly-sized buffer). FromBytes(FlatBytes(t)) is lookup-equivalent to t.
func FlatBytes(t *Trie) ([]byte, error) {
	fl := flatten(t)
	var buf bytes.Buffer
	buf.Grow(fl.byteSize())
	if err := writeFlat(&buf, fl); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writeFlat(dst io.Writer, fl flatSections) error {
	// header reserved field (index 3) flags a trailing weights section, so old
	// unweighted files (reserved=0) still load unchanged.
	var flags uint32
	if fl.weights != nil {
		flags = flagWeighted
	}
	w := bufio.NewWriter(dst)
	bw := binary.Write
	if err := bw(w, binary.LittleEndian, []uint32{flatMagic, uint32(fl.numNodes), uint32(len(fl.edgeRune)), flags}); err != nil {
		return err
	}
	for _, section := range []any{fl.edgeStart, fl.endBits, fl.edgeRune, fl.edgeTgt} {
		if err := bw(w, binary.LittleEndian, section); err != nil {
			return err
		}
	}
	if fl.weights != nil {
		if err := bw(w, binary.LittleEndian, fl.weights); err != nil {
			return err
		}
	}
	return w.Flush()
}

func sortedRunes(nd *trieNode) []rune {
	runes := make([]rune, 0, len(nd.children))
	for r := range nd.children {
		runes = append(runes, r)
	}
	sort.Slice(runes, func(a, b int) bool { return runes[a] < runes[b] })
	return runes
}

// ---------- load ----------

// OpenFlat memory-maps a flat trie file. Fastest load; the returned FlatTrie
// holds the mmap open until Close is called. See Close for the lifetime
// contract: never Close while any goroutine may still be doing lookups.
func OpenFlat(path string) (*FlatTrie, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		return nil, err
	}
	ft, err := parseFlat(data)
	if err != nil {
		_ = data.Unmap()
		return nil, err
	}
	ft.data = data
	return ft, nil
}

// ReadFlat reads a flat trie file fully into memory (no mmap).
func ReadFlat(path string) (*FlatTrie, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return FromBytes(data)
}

// FromBytes builds a FlatTrie from raw bytes, e.g. a go:embed'd dictionary.
// The bytes are copied into an 8-byte-aligned buffer first, because the unsafe
// zero-copy casts require alignment that an arbitrary/embedded slice does not
// guarantee. The input may be reused or freed after the call returns.
func FromBytes(data []byte) (*FlatTrie, error) {
	n := len(data)
	if n == 0 {
		return nil, fmt.Errorf("newmm: empty flat trie data")
	}
	buf64 := make([]uint64, (n+7)/8) // allocating as []uint64 guarantees 8-byte alignment
	buf := unsafe.Slice((*byte)(unsafe.Pointer(&buf64[0])), n)
	copy(buf, data)
	ft, err := parseFlat(buf)
	if err != nil {
		return nil, err
	}
	ft.keep = buf64
	return ft, nil
}

// parseFlat maps the CSR sections onto data (must be 4-byte aligned) zero-copy.
// The header counts drive unsafe casts, so it first checks that len(data) is
// exactly what the header implies, then validates the CSR invariants (edgeStart
// monotone and closed over numEdges, every edgeTgt in range) in one O(nodes+
// edges) pass — a corrupt or truncated file returns an error at load time
// instead of panicking (or reading out of bounds) at first lookup. Lookup
// paths stay validation-free.
func parseFlat(data []byte) (*FlatTrie, error) {
	if len(data) < flatHeaderLen || u32(data, 0) != flatMagic {
		return nil, fmt.Errorf("newmm: bad flat trie file (magic mismatch)")
	}
	numNodes := int(u32(data, 4))
	numEdges := int(u32(data, 8))
	flags := u32(data, 12)
	if numNodes < 1 {
		return nil, fmt.Errorf("newmm: bad flat trie file (no root node)")
	}
	n64, e64 := int64(numNodes), int64(numEdges)
	nbits := (numNodes + 31) / 32
	words := (n64 + 1) + int64(nbits) + 2*e64
	if flags&flagWeighted != 0 {
		words += n64
	}
	if want := flatHeaderLen + flatWordBytes*words; int64(len(data)) != want {
		return nil, fmt.Errorf("newmm: bad flat trie file (%d bytes, header implies %d)", len(data), want)
	}
	off := flatHeaderLen
	ft := &FlatTrie{}
	ft.edgeStart = castU32(data, off, numNodes+1)
	off += flatWordBytes * (numNodes + 1)
	ft.endBits = castU32(data, off, nbits)
	off += flatWordBytes * nbits
	ft.edgeRune = castI32(data, off, numEdges)
	off += flatWordBytes * numEdges
	ft.edgeTgt = castU32(data, off, numEdges)
	off += flatWordBytes * numEdges
	if flags&flagWeighted != 0 { // trailing weights section (one int32 per node)
		ft.weights = castI32(data, off, numNodes)
	}
	es := ft.edgeStart
	if es[0] != 0 || es[numNodes] != uint32(numEdges) {
		return nil, fmt.Errorf("newmm: bad flat trie file (edgeStart bounds %d..%d, want 0..%d)", es[0], es[numNodes], numEdges)
	}
	prev := uint32(0)
	for i, v := range es[1:] {
		if v < prev {
			return nil, fmt.Errorf("newmm: bad flat trie file (edgeStart not monotone at node %d)", i+1)
		}
		prev = v
	}
	limit := uint32(numNodes)
	for i, tgt := range ft.edgeTgt {
		if tgt >= limit {
			return nil, fmt.Errorf("newmm: bad flat trie file (edge %d targets node %d of %d)", i, tgt, numNodes)
		}
	}
	return ft, nil
}

func u32(b []byte, off int) uint32 { return binary.LittleEndian.Uint32(b[off:]) }

func castU32(b []byte, off, n int) []uint32 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[off])), n)
}

func castI32(b []byte, off, n int) []int32 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(&b[off])), n)
}
