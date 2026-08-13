// Package thai wires discover to Thai: the shared dictionary decides what is
// already known, and Thai orthography decides what could be a word at all.
//
// It is a separate package so that discovering English or Japanese terms does
// not link the 2.8 MB Thai dictionary into the binary.
//
//	seg, _ := tokenize.NewDefault()
//	var docs [][]string
//	for _, line := range corpus {
//	    docs = append(docs, seg.SegmentNoWS(line))
//	}
//	terms := discover.TermsFromSlices(docs, thai.Options())
package thai

import (
	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/discover"
)

// maxBareConsonants is how many bare-consonant clusters may sit in a row before
// a candidate is judged unpronounceable.
//
// A Thai syllable needs a vowel. TCC groups the characters that cannot be
// separated, so a cluster that is a lone consonant carries no vowel of its own
// — and a run of them cannot be read aloud. Measured on a novel corpus: the
// debris a failed segmentation leaves behind scores 3 ("ดผม" = ด·ผ·ม, "ดคน" =
// ด·ค·น) while real loanwords score 0-2 ("ทานูกิ" = ทา·นู·กิ, "ก็อบลิน" =
// ก็·อ·บ·ลิ·น). Three is therefore the first value that is always wrong.
const maxBareConsonants = 3

// minAffixRunes is the shortest known word that may be peeled off a term as
// ordinary vocabulary.
//
// The tokenizer does not only meet real words at a term's edges; it also splits
// the spelling of an unknown name into pieces that happen to be words. Those
// pieces are short — "ก็อบลิน" tokenizes as ก็·อบ·ลิน, and "ก็" is a real Thai
// particle, so peeling it off would leave "อบลิน" as the goblin's name. The
// words that genuinely wrap a term are content words and longer: หุ้ม, เกราะ,
// คุณ. Three characters is where the two groups separate, and this is a floor,
// not a judgement — an affix must also prove it lives outside the term.
const minAffixRunes = 3

// Options returns discover options for Thai: the embedded dictionary as the
// known-word test, and pronounceability as the well-formedness test.
//
// It loads the shared dictionary (dict.Default), so it is cheap to call more
// than once but not free the first time.
func Options() discover.Options {
	o := discover.Options{Valid: Pronounceable}
	if d, err := dict.Default(); err == nil {
		o.Known = d.Contains
		o.Affix = func(token string) bool {
			return len([]rune(token)) >= minAffixRunes && d.Contains(token)
		}
	}
	return o
}

// Pronounceable reports whether text could be read as Thai: no run of bare
// consonant clusters long enough to leave a syllable without a vowel.
//
// This is orthography, not statistics — it needs no corpus, which is what makes
// it useful for terms that are real but rare, where frequency and entropy have
// nothing to say.
func Pronounceable(text string) bool {
	r := []rune(text)
	if len(r) == 0 {
		return false
	}
	// ๆ repeats the word before it and ฯ abbreviates the one before it, so
	// neither can open a term or stand as one; a term ending in ๆ is the same
	// word twice, not a new one. Everything must be Thai script or a mark that
	// belongs to it — a quote or a question mark riding along ("กล่าวว่า“")
	// means the run crossed a boundary that is not a word boundary.
	for _, c := range r {
		if !isThaiRune(c) {
			return false
		}
	}
	if r[0] == 'ๆ' || r[0] == 'ฯ' || r[len(r)-1] == 'ๆ' {
		return false
	}
	pos := sharedTCC().PosArray(r)
	run, start := 0, 0
	for i := 1; i <= len(r); i++ {
		if !pos[i] {
			continue
		}
		cluster := r[start:i]
		start = i
		if len(cluster) == 1 && isConsonant(cluster[0]) {
			run++
			if run >= maxBareConsonants {
				return false
			}
			continue
		}
		run = 0
	}
	return true
}

func isConsonant(r rune) bool { return r >= 0x0E01 && r <= 0x0E2E }

// isThaiRune covers the Thai block: consonants, vowels, tone marks and the
// repetition/abbreviation signs. Digits and Latin are deliberately excluded —
// a term mixing scripts is a routing failure upstream, not a discovery.
func isThaiRune(r rune) bool { return r >= 0x0E01 && r <= 0x0E5B }
