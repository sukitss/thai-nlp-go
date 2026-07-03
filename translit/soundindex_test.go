package translit

import (
	"reflect"
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

// scoreOf returns the Score for id in ms, or -1 if absent.
func scoreOf(ms []Match, id string) float64 {
	for _, m := range ms {
		if m.ID == id {
			return m.Score
		}
	}
	return -1
}

func TestSoundIndexLookupScored(t *testing.T) {
	x := buildIndex()
	// exact alias match scores exactly 1.0
	got := x.LookupScored("นี่หลี่", 0)
	if len(got) == 0 || got[0].ID != "nieli" || got[0].Score != 1.0 {
		t.Fatalf("LookupScored(นี่หลี่, 0) = %v, want nieli with score 1.0", got)
	}
	// exact registered surface also scores 1.0
	if s := scoreOf(x.LookupScored("เนี่ยลี่", 0), "nieli"); s != 1.0 {
		t.Errorf("LookupScored(เนี่ยลี่).nieli = %v, want 1.0", s)
	}
	// unregistered close variant: found via phonetic bucket, 0 < score < 1
	closeSim := scoreOf(x.LookupScored("เนียลี", 0), "nieli")
	if closeSim <= 0 || closeSim >= 1 {
		t.Fatalf("LookupScored(เนียลี).nieli = %v, want in (0,1)", closeSim)
	}
	// scores are monotone: exact alias (1.0) >= close variant >= weak bucket-mate
	x.Add("somjai", "สมใจ") // shares a phonetic bucket with สมชาย, edit-distant
	weakSim := scoreOf(x.LookupScored("สมชาย", 0), "somjai")
	if weakSim <= 0 {
		t.Fatalf("LookupScored(สมชาย).somjai = %v, want > 0 (bucket-mate)", weakSim)
	}
	if !(1.0 >= closeSim && closeSim >= weakSim) {
		t.Errorf("monotonicity broken: 1.0 >= %v (close) >= %v (weak)", closeSim, weakSim)
	}
}

func TestSoundIndexLookupScoredMinSim(t *testing.T) {
	x := buildIndex()
	x.Add("somjai", "สมใจ")
	// minSim 0 returns all candidates: exact somchai + weak bucket-mate somjai
	all := x.LookupScored("สมชาย", 0)
	if len(all) != 2 || all[0].ID != "somchai" || all[0].Score != 1.0 || all[1].ID != "somjai" {
		t.Fatalf("LookupScored(สมชาย, 0) = %v, want [somchai 1.0, somjai <1]", all)
	}
	// minSim 0.99 keeps only the exact match
	strict := x.LookupScored("สมชาย", 0.99)
	if len(strict) != 1 || strict[0].ID != "somchai" || strict[0].Score != 1.0 {
		t.Errorf("LookupScored(สมชาย, 0.99) = %v, want only exact somchai", strict)
	}
	// minSim above every candidate's score returns nil
	if got := x.LookupScored("เนียลี", 0.99); got != nil {
		t.Errorf("LookupScored(เนียลี, 0.99) = %v, want nil (no exact match)", got)
	}
}

// TestSoundIndexLookupScoredTieBreak: equal scores order by ID ascending.
func TestSoundIndexLookupScoredTieBreak(t *testing.T) {
	x := NewSoundIndex()
	x.Add("zhao", "สมชาย") // two entities, same surface: scores tie at 1.0
	x.Add("chen", "สมชาย")
	got := x.LookupScored("สมชาย", 0)
	if len(got) != 2 || got[0].ID != "chen" || got[1].ID != "zhao" {
		t.Fatalf("LookupScored tie = %v, want [chen zhao] (ID ascending)", got)
	}
	if got[0].Score != 1.0 || got[1].Score != 1.0 {
		t.Errorf("tie scores = %v, want both 1.0", got)
	}
}

func TestSoundIndexLookupScoredDeterministic(t *testing.T) {
	x := buildIndex()
	x.Add("somjai", "สมใจ")
	for _, q := range []string{"เนี่ยลี่", "นี่หลี่", "สมชาย", "เนียลี"} {
		first := x.LookupScored(q, 0)
		for i := 0; i < 10; i++ {
			if got := x.LookupScored(q, 0); !reflect.DeepEqual(got, first) {
				t.Fatalf("LookupScored(%q) not deterministic: %v vs %v", q, got, first)
			}
		}
	}
}

// TestSoundIndexLookupScoredDedup: an entity registered under several
// spellings and reachable through several phonetic buckets appears once,
// carrying its best score.
func TestSoundIndexLookupScoredDedup(t *testing.T) {
	x := buildIndex()
	got := x.LookupScored("เนี่ยลี่", 0) // hits all three nieli spellings' buckets
	seen := map[string]int{}
	for _, m := range got {
		seen[m.ID]++
	}
	if seen["nieli"] != 1 {
		t.Fatalf("LookupScored(เนี่ยลี่) = %v, want nieli exactly once", got)
	}
	if s := scoreOf(got, "nieli"); s != 1.0 {
		t.Errorf("dedup kept score %v, want best (1.0, the exact spelling)", s)
	}
}

func TestSoundIndexLookupScoredEdgeCases(t *testing.T) {
	empty := NewSoundIndex()
	if got := empty.LookupScored("อะไรก็ได้", 0); got != nil {
		t.Errorf("empty index LookupScored = %v, want nil", got)
	}
	x := buildIndex()
	if got := x.LookupScored("", 0); got != nil {
		t.Errorf("empty query = %v, want nil", got)
	}
	over := strings.Repeat("ก", MaxSoundLen+1)
	if got := x.LookupScored(over, 0); got != nil {
		t.Errorf("over-MaxSoundLen query = %v, want nil", got)
	}
}

// TestSoundIndexLookupScoredMatchesLookup: with minSim 0, LookupScored returns
// exactly the same id set as Lookup — both generate candidates from the same
// phonetic buckets + exact alias and drop similarity-0 candidates the same way.
func TestSoundIndexLookupScoredMatchesLookup(t *testing.T) {
	x := buildIndex()
	x.Add("somjai", "สมใจ")
	for _, q := range []string{"เนี่ยลี่", "เนียลี่", "เนี่ยหลี่", "นี่หลี่", "เนียลี", "สมชาย", "สมใจ", "แมวน้ำ"} {
		want := x.Lookup(q)
		got := x.LookupScored(q, 0)
		if len(got) != len(want) {
			t.Fatalf("q=%q: LookupScored ids %v != Lookup %v", q, got, want)
		}
		wantSet := map[string]bool{}
		for _, id := range want {
			wantSet[id] = true
		}
		for _, m := range got {
			if !wantSet[m.ID] {
				t.Errorf("q=%q: LookupScored returned %q not in Lookup %v", q, m.ID, want)
			}
		}
	}
}

func TestSoundIndexLen(t *testing.T) {
	x := NewSoundIndex()
	if x.Len() != 0 {
		t.Errorf("empty Len = %d, want 0", x.Len())
	}
	x = buildIndex() // nieli (3 spellings + 1 alias) + somchai
	if x.Len() != 2 {
		t.Errorf("Len = %d, want 2 distinct entities", x.Len())
	}
	x.Add("somjai", "สมใจ")
	if x.Len() != 3 {
		t.Errorf("Len after Add = %d, want 3", x.Len())
	}
}

// TestSoundIndexConcurrentLookupScored: after build, concurrent scored
// lookups are safe (read-only), verified under -race.
func TestSoundIndexConcurrentLookupScored(t *testing.T) {
	x := buildIndex()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				got := x.LookupScored("เนี่ยลี่", 0.5)
				if len(got) == 0 || got[0].ID != "nieli" || got[0].Score != 1.0 {
					t.Error("concurrent scored lookup wrong")
					return
				}
			}
		}()
	}
	wg.Wait()
}
