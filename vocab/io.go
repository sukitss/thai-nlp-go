package vocab

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Binary snapshot format (little-endian, varints are canonical unsigned
// LEB128 as produced by binary.PutUvarint):
//
//	magic    uint32  = 0x56474E54 ("TNGV" on disk)
//	version  byte    = 1
//	docCount uvarint
//	numTerms uvarint
//	numTerms × entry, in id order (id 0 first):
//	    termLen uvarint | term bytes (UTF-8, may be empty) | df uvarint
//
// The format is deterministic (terms are written in id order and uvarints are
// canonical), so Save of the same state is byte-identical. Load validates
// everything — magic, version, counts vs file size, df ≤ docCount, duplicate
// terms, trailing bytes — and returns an error for corrupt or truncated input
// instead of panicking.
const (
	vocabMagic   uint32 = 0x56474E54 // "TNGV" little-endian
	vocabVersion byte   = 1
	headerLen           = 5 // magic + version
)

func (t *table) save(w io.Writer) error {
	bw := bufio.NewWriter(w)
	var hdr [headerLen]byte
	binary.LittleEndian.PutUint32(hdr[:4], vocabMagic)
	hdr[4] = vocabVersion
	if _, err := bw.Write(hdr[:]); err != nil {
		return err
	}
	var buf [binary.MaxVarintLen64]byte
	uv := func(x uint64) error {
		_, err := bw.Write(buf[:binary.PutUvarint(buf[:], x)])
		return err
	}
	if err := uv(uint64(t.docs)); err != nil {
		return err
	}
	if err := uv(uint64(len(t.terms))); err != nil {
		return err
	}
	for id, term := range t.terms {
		if err := uv(uint64(len(term))); err != nil {
			return err
		}
		if _, err := bw.WriteString(term); err != nil {
			return err
		}
		if err := uv(uint64(t.df[id])); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// Load reads a vocabulary snapshot written by Save and returns an immutable
// Vocab. It reads r fully into memory; term strings then share that single
// backing buffer (one allocation for all term bytes). Corrupt or truncated
// input returns a descriptive error — Load never panics.
func Load(r io.Reader) (*Vocab, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("vocab: read: %w", err)
	}
	if len(raw) < headerLen {
		return nil, errors.New("vocab: truncated header")
	}
	if binary.LittleEndian.Uint32(raw) != vocabMagic {
		return nil, errors.New("vocab: bad magic (not a vocab file)")
	}
	if raw[4] != vocabVersion {
		return nil, fmt.Errorf("vocab: unsupported format version %d (want %d)", raw[4], vocabVersion)
	}
	s := string(raw) // terms below are zero-copy slices of this
	p := headerLen
	uv := func(what string) (uint64, error) {
		x, n := binary.Uvarint(raw[p:])
		if n <= 0 {
			return 0, fmt.Errorf("vocab: truncated or invalid %s at offset %d", what, p)
		}
		p += n
		return x, nil
	}
	docs, err := uv("doc count")
	if err != nil {
		return nil, err
	}
	if docs > math.MaxInt {
		return nil, fmt.Errorf("vocab: doc count %d overflows int", docs)
	}
	nTerms, err := uv("term count")
	if err != nil {
		return nil, err
	}
	// Every entry takes at least 2 bytes (termLen + df varints), so this both
	// rejects impossible counts and bounds the allocations below by len(raw).
	if nTerms > uint64(len(raw)-p)/2 || nTerms > 1<<32 {
		return nil, fmt.Errorf("vocab: term count %d exceeds file size", nTerms)
	}
	v := &Vocab{table{
		ids:   make(map[string]uint32, nTerms),
		terms: make([]string, 0, nTerms),
		df:    make([]int, 0, nTerms),
		docs:  int(docs),
	}}
	for i := uint64(0); i < nTerms; i++ {
		tl, err := uv("term length")
		if err != nil {
			return nil, err
		}
		if tl > uint64(len(raw)-p) {
			return nil, fmt.Errorf("vocab: term %d length %d exceeds remaining %d bytes", i, tl, len(raw)-p)
		}
		term := s[p : p+int(tl)]
		p += int(tl)
		df, err := uv("df")
		if err != nil {
			return nil, err
		}
		if df > docs {
			return nil, fmt.Errorf("vocab: term %q df %d exceeds doc count %d", term, df, docs)
		}
		if _, dup := v.ids[term]; dup {
			return nil, fmt.Errorf("vocab: duplicate term %q", term)
		}
		v.ids[term] = uint32(i)
		v.terms = append(v.terms, term)
		v.df = append(v.df, int(df))
	}
	if p != len(raw) {
		return nil, fmt.Errorf("vocab: %d trailing bytes after %d terms", len(raw)-p, nTerms)
	}
	return v, nil
}
