package dict

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// buildFlatBytes builds t into a flat file in a temp dir and returns its bytes.
func buildFlatBytes(t *testing.T, tr *Trie) []byte {
	t.Helper()
	path := filepath.Join(t.TempDir(), "t.fdt")
	if err := BuildFlatFromTrie(tr, path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// TestWeightedRoundTrip pins the weighted golden path: AddWeighted (weight 0,
// negative, extremes, duplicates, mixed with Add) → flat → identical lookups.
func TestWeightedRoundTrip(t *testing.T) {
	tr := NewTrie()
	want := map[string]int32{}
	add := func(w string, wt int32) {
		tr.AddWeighted(w, wt)
		want[w] = wt
	}
	add("กา", 0)
	add("กาแฟ", -7)
	add("แฟน", math.MaxInt32)
	add("ใจ", math.MinInt32)
	add("ใจ", 42) // duplicate: last weight wins
	want["ใจ"] = 42
	tr.Add("ดี") // mixed unweighted word keeps weight 0
	want["ดี"] = 0

	ft, err := FromBytes(buildFlatBytes(t, tr))
	if err != nil {
		t.Fatal(err)
	}
	if !ft.Weighted() {
		t.Fatal("Weighted() = false, want true")
	}
	for w, wt := range want {
		got, ok := ft.Weight(w)
		if !ok || got != wt {
			t.Errorf("Weight(%q) = (%d,%v), want (%d,true)", w, got, ok, wt)
		}
	}
	if _, ok := ft.Weight("กาแ"); ok {
		t.Error("internal node reported as word")
	}
	corpus := []rune("กาแฟใจดีแฟน")
	var outL, outW []int32
	for start := 0; start < len(corpus); start++ {
		outL, outW = ft.PrefixWeights(corpus, start, outL, outW)
		for k, l := range outL {
			word := string(corpus[start : start+int(l)])
			wt, ok := want[word]
			if !ok {
				t.Fatalf("PrefixWeights reported non-word %q", word)
			}
			if outW[k] != wt {
				t.Errorf("PrefixWeights(%q) weight = %d, want %d", word, outW[k], wt)
			}
		}
	}
}

// TestEmptyTrieRoundTrip covers the numEdges=0 layout.
func TestEmptyTrieRoundTrip(t *testing.T) {
	ft, err := FromBytes(buildFlatBytes(t, NewTrie()))
	if err != nil {
		t.Fatal(err)
	}
	if ft.Weighted() {
		t.Error("empty trie reports Weighted()=true")
	}
	if ft.Contains("ก") || ft.Contains("") {
		t.Error("empty trie contains a word")
	}
	if lens := ft.PrefixLens([]rune("กข"), 0, nil); len(lens) != 0 {
		t.Errorf("PrefixLens on empty trie = %v", lens)
	}
}

func TestFromBytesRejectsNilAndGarbage(t *testing.T) {
	for _, data := range [][]byte{nil, {}, []byte("not a trie"), make([]byte, flatHeaderLen)} {
		if _, err := FromBytes(data); err == nil {
			t.Errorf("FromBytes(%d bytes) = nil error", len(data))
		}
	}
}

// corrupt returns a copy of data with 4 bytes at off overwritten by v.
func corrupt(data []byte, off int, v uint32) []byte {
	c := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(c[off:], v)
	return c
}

// TestCorruptFlatRejected: truncated/corrupt files must error at load, never
// panic, and never read out of bounds.
func TestCorruptFlatRejected(t *testing.T) {
	tr := NewTrie()
	for _, w := range []string{"ก", "กา", "กาแฟ", "แฟน", "ab", "abc"} {
		tr.AddWeighted(w, int32(len(w)))
	}
	valid := buildFlatBytes(t, tr)
	if _, err := FromBytes(valid); err != nil {
		t.Fatalf("valid file rejected: %v", err)
	}
	numNodes := int(binary.LittleEndian.Uint32(valid[4:]))
	numEdges := int(binary.LittleEndian.Uint32(valid[8:]))
	esOff := flatHeaderLen
	nbits := (numNodes + 31) / 32
	runeOff := esOff + flatWordBytes*(numNodes+1) + flatWordBytes*nbits
	tgtOff := runeOff + flatWordBytes*numEdges

	cases := map[string][]byte{
		"header only":            valid[:flatHeaderLen],
		"cut in edgeStart":       valid[:esOff+flatWordBytes],
		"cut at endBits":         valid[:esOff+flatWordBytes*(numNodes+1)],
		"cut at edgeRune":        valid[:runeOff],
		"cut at edgeTgt":         valid[:tgtOff],
		"cut before weights":     valid[:tgtOff+flatWordBytes*numEdges],
		"one byte short":         valid[:len(valid)-1],
		"one byte extra":         append(append([]byte(nil), valid...), 0),
		"huge numNodes":          corrupt(valid, 4, 0xFFFFFFF0),
		"zero numNodes":          corrupt(valid, 4, 0),
		"inflated numEdges":      corrupt(valid, 8, uint32(numEdges+1)),
		"dangling edgeTgt":       corrupt(valid, tgtOff, uint32(numNodes)+999),
		"edgeStart[0] nonzero":   corrupt(valid, esOff, 1),
		"edgeStart non-monotone": corrupt(valid, esOff+flatWordBytes, uint32(numEdges)+1),
	}
	for i := 1; i < len(valid); i += len(valid)/7 + 1 { // random-ish truncations
		cases["truncated at "+strconv.Itoa(i)] = valid[:i]
	}
	for name, data := range cases {
		if _, err := FromBytes(data); err == nil {
			t.Errorf("%s: FromBytes accepted corrupt data", name)
		} else if !strings.Contains(err.Error(), "flat trie") {
			t.Errorf("%s: unexpected error %v", name, err)
		}
	}

	// same rejection via the file loaders
	path := filepath.Join(t.TempDir(), "corrupt.fdt")
	if err := os.WriteFile(path, valid[:len(valid)-flatWordBytes], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadFlat(path); err == nil {
		t.Error("ReadFlat accepted corrupt file")
	}
	if ft, err := OpenFlat(path); err == nil {
		ft.Close()
		t.Error("OpenFlat accepted corrupt file")
	}
}

// TestFlatLifecycle: open errors, idempotent Close, and use-after-Close being
// recoverable (nil-slice panic or empty result) instead of a SIGSEGV.
func TestFlatLifecycle(t *testing.T) {
	if _, err := OpenFlat(filepath.Join(t.TempDir(), "missing.fdt")); err == nil {
		t.Fatal("OpenFlat(nonexistent) = nil error")
	}

	tr := NewTrie()
	tr.Add("กาแฟ")
	path := filepath.Join(t.TempDir(), "life.fdt")
	if err := BuildFlatFromTrie(tr, path); err != nil {
		t.Fatal(err)
	}
	checkClosed := func(name string, ft *FlatTrie) {
		t.Helper()
		if err := ft.Close(); err != nil {
			t.Fatalf("%s: Close: %v", name, err)
		}
		if err := ft.Close(); err != nil {
			t.Fatalf("%s: second Close: %v", name, err)
		}
		func() {
			defer func() { recover() }() // panic (recoverable) or zero results — never SIGSEGV
			if ft.Contains("กาแฟ") {
				t.Errorf("%s: Contains true after Close", name)
			}
			if lens := ft.PrefixLens([]rune("กาแฟ"), 0, nil); len(lens) != 0 {
				t.Errorf("%s: PrefixLens non-empty after Close", name)
			}
		}()
	}
	mm, err := OpenFlat(path)
	if err != nil {
		t.Fatal(err)
	}
	checkClosed("OpenFlat", mm)
	eager, err := ReadFlat(path)
	if err != nil {
		t.Fatal(err)
	}
	checkClosed("ReadFlat", eager)
}

// TestBuildFlatErrors: unwritable destination must propagate an error and
// never leave a partial target file behind.
func TestBuildFlatErrors(t *testing.T) {
	tr := NewTrie()
	tr.Add("ก")
	path := filepath.Join(t.TempDir(), "no", "such", "dir", "x.fdt")
	if err := BuildFlatFromTrie(tr, path); err == nil {
		t.Fatal("BuildFlatFromTrie into missing dir = nil error")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("partial target left behind: %v", err)
	}
}

func TestEmbeddedBytesParseable(t *testing.T) {
	data := EmbeddedBytes()
	if len(data) == 0 {
		t.Fatal("EmbeddedBytes is empty")
	}
	ft, err := FromBytes(data)
	if err != nil {
		t.Fatalf("embedded dictionary does not parse: %v", err)
	}
	if !ft.Contains("ภาษาไทย") {
		t.Error("embedded dict missing ภาษาไทย")
	}
}

// FuzzFromBytes: arbitrary mutations of a valid flat file must parse-or-error,
// never panic, and a successful parse must survive lookups.
func FuzzFromBytes(f *testing.F) {
	tr := NewTrie()
	for _, w := range []string{"ก", "กา", "กาแฟ", "ab", "abc"} {
		tr.AddWeighted(w, 3)
	}
	path := filepath.Join(f.TempDir(), "seed.fdt")
	if err := BuildFlatFromTrie(tr, path); err != nil {
		f.Fatal(err)
	}
	seed, err := os.ReadFile(path)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(seed)
	f.Add(seed[:len(seed)/2])
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		ft, err := FromBytes(data)
		if err != nil {
			return
		}
		text := []rune("กาแฟ abc ๆ")
		var lens []int
		var outL, outW []int32
		for s := 0; s < len(text); s++ {
			lens = ft.PrefixLens(text, s, lens)
			outL, outW = ft.PrefixWeights(text, s, outL, outW)
		}
		ft.Contains("กา")
		ft.Weight("abc")
		_ = outW
	})
}

// uniqueSampleWords returns sampleWords deduplicated (the crafted words can
// collide with the sampled ones), preserving first-seen order.
func uniqueSampleWords(t *testing.T) []string {
	t.Helper()
	words := sampleWords(t)
	seen := make(map[string]bool, len(words))
	out := make([]string, 0, len(words))
	for _, w := range words {
		if !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	return out
}

// TestWriteFlatMatchesBuildFile: WriteFlat and FlatBytes must produce exactly
// the bytes BuildFlatFromTrie writes to disk, for empty, unweighted and
// weighted tries.
func TestWriteFlatMatchesBuildFile(t *testing.T) {
	unweighted := NewTrie()
	weighted := NewTrie()
	for i, w := range []string{"ก", "กา", "กาแฟ", "แฟน", "a", "ab", "1", "๑๒๓"} {
		unweighted.Add(w)
		weighted.AddWeighted(w, int32(i)-3)
	}
	for name, tr := range map[string]*Trie{"empty": NewTrie(), "unweighted": unweighted, "weighted": weighted} {
		file := buildFlatBytes(t, tr)
		var buf bytes.Buffer
		if err := tr.WriteFlat(&buf); err != nil {
			t.Fatalf("%s: WriteFlat: %v", name, err)
		}
		if !bytes.Equal(buf.Bytes(), file) {
			t.Errorf("%s: WriteFlat differs from BuildFlatFromTrie file (%d vs %d bytes)", name, buf.Len(), len(file))
		}
		fb, err := FlatBytes(tr)
		if err != nil {
			t.Fatalf("%s: FlatBytes: %v", name, err)
		}
		if !bytes.Equal(fb, file) {
			t.Errorf("%s: FlatBytes differs from BuildFlatFromTrie file (%d vs %d bytes)", name, len(fb), len(file))
		}
	}
}

// TestFlatBytesRoundTrip: FromBytes(FlatBytes(t)) must be lookup-equivalent to
// the pointer trie across the whole read API, on a diverse word set
// (Thai/Latin/digits/single-rune/long/shared-prefix — see sampleWords).
func TestFlatBytesRoundTrip(t *testing.T) {
	words := uniqueSampleWords(t)
	tr := NewTrie()
	for i, w := range words {
		if i%3 == 0 {
			tr.Add(w) // mixed unweighted words keep weight 0
		} else {
			tr.AddWeighted(w, int32(i)-50)
		}
	}
	data, err := FlatBytes(tr)
	if err != nil {
		t.Fatal(err)
	}
	ft, err := FromBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if ft.Weighted() != tr.Weighted() {
		t.Fatalf("Weighted() = %v, want %v", ft.Weighted(), tr.Weighted())
	}

	// every word the trie enumerates must round-trip with its weight...
	n := 0
	tr.Words(func(w string, wt int32) bool {
		n++
		got, ok := ft.Weight(w)
		if !ok || got != wt {
			t.Errorf("flat Weight(%q) = (%d,%v), want (%d,true)", w, got, ok, wt)
		}
		return true
	})
	if n != tr.Len() || n != len(words) {
		t.Fatalf("Words enumerated %d, Len = %d, added %d", n, tr.Len(), len(words))
	}
	// ...and the flat side must enumerate exactly the same set back
	m := 0
	ft.WalkPrefix("", func(w string, wt int32) bool {
		m++
		got, ok := tr.Weight(w)
		if !ok || got != wt {
			t.Errorf("trie Weight(%q) = (%d,%v), want (%d,true)", w, got, ok, wt)
		}
		return true
	})
	if m != n {
		t.Fatalf("flat enumerates %d words, trie %d", m, n)
	}

	for _, w := range []string{"", "z", "ไม่มีในพจนานุกรมแน่นอน", "abcdefgh"} {
		if tr.Contains(w) != ft.Contains(w) {
			t.Errorf("Contains(%q): trie %v, flat %v", w, tr.Contains(w), ft.Contains(w))
		}
	}

	// PrefixLens / PrefixWeights parity at every position of a mixed corpus
	corpus := []rune(strings.Join(words[:60], "") + "zzไม่มีqq")
	var wantL, gotL []int
	var wantL32, wantW, gotL32, gotW []int32
	for s := 0; s < len(corpus); s++ {
		wantL = tr.PrefixLens(corpus, s, wantL)
		gotL = ft.PrefixLens(corpus, s, gotL)
		if !slices.Equal(wantL, gotL) {
			t.Fatalf("PrefixLens at %d: flat %v, trie %v", s, gotL, wantL)
		}
		wantL32, wantW = tr.PrefixWeights(corpus, s, wantL32, wantW)
		gotL32, gotW = ft.PrefixWeights(corpus, s, gotL32, gotW)
		if !slices.Equal(wantL32, gotL32) || !slices.Equal(wantW, gotW) {
			t.Fatalf("PrefixWeights at %d: flat (%v,%v), trie (%v,%v)", s, gotL32, gotW, wantL32, wantW)
		}
	}
}

// TestWriteFlatPropagatesWriterError: a failing writer must surface its error.
func TestWriteFlatPropagatesWriterError(t *testing.T) {
	tr := NewTrie()
	tr.Add("กาแฟ")
	if err := tr.WriteFlat(failWriter{}); err == nil {
		t.Fatal("WriteFlat(failWriter) = nil error")
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errWriteFail }

var errWriteFail = errors.New("write failed")
