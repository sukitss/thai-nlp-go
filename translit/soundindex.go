package translit

import "sort"

// SoundIndex maps many surface spellings of an entity (e.g. a character name
// transliterated differently by different translators) to a stable entity id,
// so a query in any spelling finds the entity. It layers three signals:
//
//  1. phonetic keys (MetaSound/Udom83/LK82) — buckets same-sounding spellings
//  2. edit distance — ranks candidates and catches near-misses
//  3. aliases — a manual map for interpret-level variants the keys miss
//     (e.g. เนี่ยลี่ vs นี่หลี่, where translators heard the sound differently)
//
// Usage: Add/Alias to build the index, then Lookup. Build fully first; once
// built it is read-only and safe for concurrent Lookup across goroutines
// (do not Add/Alias concurrently with Lookup). Lookup allocates only its result.
type SoundIndex struct {
	buckets map[string][]int // phonetic key (algo-prefixed) -> entry indices
	entries []siEntry
	aliases map[string]int // exact surface form -> entry index
}

type siEntry struct {
	id   string
	name string
}

// NewSoundIndex returns an empty index.
func NewSoundIndex() *SoundIndex {
	return &SoundIndex{buckets: map[string][]int{}, aliases: map[string]int{}}
}

// keyed prefixes keep the three algorithms' keyspaces disjoint.
func bucketKeys(name string) []string {
	ks := PhoneticKeys(name) // [metasound, udom83, lk82]
	out := make([]string, 0, 3)
	for i, k := range ks {
		if k != "" {
			out = append(out, string(rune('a'+i))+"\x00"+k)
		}
	}
	return out
}

// Add registers a surface spelling name for entity id.
func (x *SoundIndex) Add(id, name string) {
	if name == "" {
		return
	}
	e := len(x.entries)
	x.entries = append(x.entries, siEntry{id: id, name: name})
	for _, k := range bucketKeys(name) {
		x.buckets[k] = append(x.buckets[k], e)
	}
}

// Alias maps an exact surface form directly to entity id — for variants the
// phonetic keys don't catch.
func (x *SoundIndex) Alias(surface, id string) {
	if surface == "" {
		return
	}
	e := len(x.entries)
	x.entries = append(x.entries, siEntry{id: id, name: surface})
	x.aliases[surface] = e
}

// Lookup returns the entity ids whose spellings sound like query, most similar
// first (by edit-distance similarity to query). An exact alias match ranks top.
func (x *SoundIndex) Lookup(query string) []string {
	if query == "" {
		return nil
	}
	// best similarity per entry index among candidates
	best := map[int]float64{}
	if e, ok := x.aliases[query]; ok {
		best[e] = 2 // above any similarity, so aliases rank first
	}
	for _, k := range bucketKeys(query) {
		for _, e := range x.buckets[k] {
			s := Similarity(query, x.entries[e].name)
			if s > best[e] {
				best[e] = s
			}
		}
	}
	if len(best) == 0 {
		return nil
	}
	// collect best score per id
	idScore := map[string]float64{}
	for e, s := range best {
		id := x.entries[e].id
		if s > idScore[id] {
			idScore[id] = s
		}
	}
	ids := make([]string, 0, len(idScore))
	for id := range idScore {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if idScore[ids[i]] != idScore[ids[j]] {
			return idScore[ids[i]] > idScore[ids[j]] // higher similarity first
		}
		return ids[i] < ids[j] // stable
	})
	return ids
}
