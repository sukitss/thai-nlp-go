package query

import (
	"bufio"
	_ "embed"
	"strings"

	"github.com/sukitss/thai-nlp-go/normalize"
)

//go:embed data/acronyms_th.txt
var acronymData string

// acronyms holds the bidirectional acronym/abbreviation equivalence groups and
// the indexes derived from them. It is built once at Parser construction and
// treated as read-only afterwards, so it is safe for concurrent Expand calls.
type acronyms struct {
	groups [][]string     // group id → its member surface forms
	lookup map[string]int // canonical(member) → group id
	forms  map[string]struct{}
}

// canon folds a surface form to its lookup identity: Unicode-normalized and
// lower-cased, so "IT"/"It"/"it" all key the same group and Thai forms (which
// have no case) key by their normalized selves.
func canon(s string) string {
	return strings.ToLower(normalize.Normalize(strings.TrimSpace(s)))
}

// isLatin reports whether s is a non-empty run of ASCII letters/digits — the
// shape of an acronym token ("IT", "POS", "IT2") as opposed to a Thai word.
func isLatin(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			continue
		}
		return false
	}
	return true
}

// newAcronyms parses the embedded seed into equivalence groups.
func newAcronyms() *acronyms {
	a := &acronyms{lookup: map[string]int{}, forms: map[string]struct{}{}}
	sc := bufio.NewScanner(strings.NewReader(acronymData))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		a.addGroup(strings.Fields(line))
	}
	return a
}

// addGroup merges members into the table as one equivalence group. If any member
// already belongs to a group, the new members are folded into that existing
// group (so a user alias sharing a term with the seed extends it rather than
// forking a parallel group); otherwise a fresh group is created. Members are
// stored de-duplicated by canonical form, first spelling wins.
func (a *acronyms) addGroup(members []string) {
	// Normalize and drop blanks/dupes within the incoming member list.
	clean := make([]string, 0, len(members))
	seen := map[string]struct{}{}
	for _, m := range members {
		m = normalize.Normalize(strings.TrimSpace(m))
		if m == "" {
			continue
		}
		c := canon(m)
		if _, dup := seen[c]; dup {
			continue
		}
		seen[c] = struct{}{}
		clean = append(clean, m)
	}
	if len(clean) == 0 {
		return
	}
	// Find an existing group any member already lives in.
	gid := -1
	for _, m := range clean {
		if id, ok := a.lookup[canon(m)]; ok {
			gid = id
			break
		}
	}
	if gid < 0 {
		gid = len(a.groups)
		a.groups = append(a.groups, nil)
	}
	for _, m := range clean {
		c := canon(m)
		if _, ok := a.lookup[c]; ok {
			continue // already present in some group; keep its first home
		}
		a.lookup[c] = gid
		a.groups[gid] = append(a.groups[gid], m)
		if isLatin(m) {
			a.forms[strings.ToLower(m)] = struct{}{}
		}
	}
}

// expand returns term followed by every equivalent form from its group, with
// the case variant (upper-cased) of a Latin term included even when it is not in
// the seed. Order is deterministic (term, then group order); duplicates removed.
func (a *acronyms) expand(term string) []string {
	out := make([]string, 0, 4)
	seen := map[string]struct{}{}
	add := func(s string) {
		if s == "" {
			return
		}
		if _, ok := seen[s]; ok {
			return
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	add(term)
	if isLatin(term) {
		add(strings.ToUpper(term)) // case variant: match an index that kept the acronym upper-case
	}
	if gid, ok := a.lookup[canon(term)]; ok {
		for _, m := range a.groups[gid] {
			add(m)
		}
	}
	return out
}

// retain reports whether a bare, would-be-dropped surface form is a known
// acronym, and returns its canonical upper-case spelling if so. Only Latin forms
// registered in the seed (or via WithAliases) qualify, so "it"/"pc" are kept as
// "IT"/"PC" while ordinary stop words are still dropped.
func (a *acronyms) retain(surface string) (string, bool) {
	if !isLatin(surface) {
		return "", false
	}
	if _, ok := a.forms[strings.ToLower(surface)]; ok {
		return strings.ToUpper(surface), true
	}
	return "", false
}
