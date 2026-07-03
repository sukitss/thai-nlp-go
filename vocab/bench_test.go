package vocab

import (
	"bytes"
	"fmt"
	"testing"
)

// benchTerms returns 100k distinct mixed Thai/Latin terms.
func benchTerms() []string {
	terms := make([]string, 100_000)
	for i := range terms {
		if i%2 == 0 {
			terms[i] = fmt.Sprintf("คำศัพท์%d", i)
		} else {
			terms[i] = fmt.Sprintf("term%d", i)
		}
	}
	return terms
}

// BenchmarkGetOrAssignHit: lookups of already-assigned terms on a 100k-term
// builder (the hot indexing path) — should be pure map-lookup speed, 0 allocs.
func BenchmarkGetOrAssignHit(b *testing.B) {
	terms := benchTerms()
	bl := NewBuilder()
	for _, t := range terms {
		bl.GetOrAssign(t)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var sum uint32
	for i := 0; i < b.N; i++ {
		sum += bl.GetOrAssign(terms[i%len(terms)])
	}
	_ = sum
}

// BenchmarkGetOrAssignNew: cost of assigning brand-new terms (map insert +
// slice appends), amortized over 100k inserts per iteration.
func BenchmarkGetOrAssignNew(b *testing.B) {
	terms := benchTerms()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bl := NewBuilder()
		for _, t := range terms {
			bl.GetOrAssign(t)
		}
	}
}

// BenchmarkVocabID: read-side lookups on a loaded 100k-term Vocab.
func BenchmarkVocabID(b *testing.B) {
	terms := benchTerms()
	bl := NewBuilder()
	bl.AddDoc(terms)
	var buf bytes.Buffer
	if err := bl.Save(&buf); err != nil {
		b.Fatal(err)
	}
	v, err := Load(&buf)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var sum uint32
	for i := 0; i < b.N; i++ {
		id, _ := v.ID(terms[i%len(terms)])
		sum += id
	}
	_ = sum
}

// BenchmarkLoad100k: full snapshot parse of a 100k-term vocabulary.
func BenchmarkLoad100k(b *testing.B) {
	bl := NewBuilder()
	bl.AddDoc(benchTerms())
	var buf bytes.Buffer
	if err := bl.Save(&buf); err != nil {
		b.Fatal(err)
	}
	data := buf.Bytes()
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Load(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}
