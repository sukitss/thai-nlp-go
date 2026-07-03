package translit

import (
	"strings"
	"sync"
	"testing"
)

func buildIndex() *SoundIndex {
	x := NewSoundIndex()
	x.Add("nieli", "เนียลี่")   // Tale of Demons and Gods — different translators
	x.Add("nieli", "เนี่ยหลี่") // same entity, different spelling
	x.Add("nieli", "เนี่ยลี่")
	x.Alias("นี่หลี่", "nieli") // interpret-variant the keys miss
	x.Add("somchai", "สมชาย")
	return x
}

func TestSoundIndexLookup(t *testing.T) {
	x := buildIndex()
	// query in one spelling finds the entity across all spellings
	for _, q := range []string{"เนี่ยลี่", "เนียลี่", "เนี่ยหลี่"} {
		got := x.Lookup(q)
		if len(got) == 0 || got[0] != "nieli" {
			t.Errorf("Lookup(%q) = %v, want nieli first", q, got)
		}
	}
	// alias catches the interpret-variant
	if got := x.Lookup("นี่หลี่"); len(got) == 0 || got[0] != "nieli" {
		t.Errorf("Lookup(นี่หลี่) = %v, want nieli (alias)", got)
	}
	// unrelated query should not return nieli
	for _, id := range x.Lookup("แมวน้ำ") {
		if id == "nieli" {
			t.Errorf("Lookup(แมวน้ำ) unexpectedly returned nieli")
		}
	}
}

func TestSoundIndexEmpty(t *testing.T) {
	x := NewSoundIndex()
	if got := x.Lookup("อะไรก็ได้"); got != nil {
		t.Errorf("empty index Lookup = %v, want nil", got)
	}
	if got := x.Lookup(""); got != nil {
		t.Errorf("empty query = %v, want nil", got)
	}
}

// TestSoundIndexConcurrentLookup: after build, concurrent Lookups are safe.
func TestSoundIndexConcurrentLookup(t *testing.T) {
	x := buildIndex()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				if got := x.Lookup("เนี่ยลี่"); len(got) == 0 || got[0] != "nieli" {
					t.Error("concurrent lookup wrong")
					return
				}
			}
		}()
	}
	wg.Wait()
	_ = strings.TrimSpace
}
