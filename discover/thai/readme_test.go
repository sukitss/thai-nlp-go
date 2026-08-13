package thai_test

import (
	"testing"

	"github.com/sukitss/thai-nlp-go/dict"
	"github.com/sukitss/thai-nlp-go/tokenize"
)

// TestReadmeOverlaySnippet compiles the snippet the README hands callers.
func TestReadmeOverlaySnippet(t *testing.T) {
	base, err := dict.Default()
	if err != nil {
		t.Fatal(err)
	}
	overlay := dict.NewTrie()
	overlay.Add("คาลูก้า")
	seg := tokenize.New(tokenize.NewOverlayDict(base, overlay))
	got := seg.SegmentNoWS("ฝูงคาลูก้าหุ้มเกราะเดินผ่านหุบเขา")
	found := false
	for _, tk := range got {
		if tk == "คาลูก้า" {
			found = true
		}
	}
	if !found {
		t.Errorf("the overlay should make the name a single token, got %q", got)
	}
}
