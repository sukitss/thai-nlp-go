package tokenize

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/sukitss/thai-nlp-go/token"
)

// checkTokenInvariants asserts the offset contract for Tokens: raw-slice Text,
// contiguous spans covering the whole input.
func checkTokenInvariants(t *testing.T, text string, toks []token.Token) {
	t.Helper()
	prev := 0
	for i, tk := range toks {
		if tk.Start != prev {
			t.Fatalf("token %d: Start=%d, want contiguous %d (in=%q)", i, tk.Start, prev, text)
		}
		if tk.End < tk.Start || tk.End > len(text) {
			t.Fatalf("token %d: bad End=%d (in=%q)", i, tk.End, text)
		}
		if text[tk.Start:tk.End] != tk.Text {
			t.Fatalf("token %d: text[%d:%d]=%q, want Text=%q (in=%q)",
				i, tk.Start, tk.End, text[tk.Start:tk.End], tk.Text, text)
		}
		prev = tk.End
	}
	if prev != len(text) {
		t.Fatalf("tokens cover %d bytes, want %d (in=%q)", prev, len(text), text)
	}
}

// TestTokensExactOffsets pins byte offsets on hand-computed cases (Thai chars
// are 3 bytes, the emoji 4, ASCII 1).
func TestTokensExactOffsets(t *testing.T) {
	seg := defaultSeg(t)
	cases := []struct {
		in   string
		want []token.Token
	}{
		{"abc 123", []token.Token{
			{Text: "abc", Start: 0, End: 3},
			{Text: " ", Start: 3, End: 4},
			{Text: "123", Start: 4, End: 7},
		}},
		{"ฉัน รัก", []token.Token{
			{Text: "ฉัน", Start: 0, End: 9},
			{Text: " ", Start: 9, End: 10},
			{Text: "รัก", Start: 10, End: 19},
		}},
		{"กิน🎉ข้าว", []token.Token{
			{Text: "กิน", Start: 0, End: 9},
			{Text: "🎉", Start: 9, End: 13},
			{Text: "ข้าว", Start: 13, End: 25},
		}},
		{"", nil},
	}
	for _, c := range cases {
		got := seg.Tokens(c.in)
		if len(got) != len(c.want) {
			t.Errorf("Tokens(%q) = %v, want %v", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("Tokens(%q)[%d] = %+v, want %+v", c.in, i, got[i], c.want[i])
			}
		}
	}
}

// TestTokensNoWSExactOffsets: whitespace tokens dropped, offsets of the kept
// tokens unchanged (still relative to the original input).
func TestTokensNoWSExactOffsets(t *testing.T) {
	seg := defaultSeg(t)
	got := seg.TokensNoWS("ฉัน รัก")
	want := []token.Token{
		{Text: "ฉัน", Start: 0, End: 9},
		{Text: "รัก", Start: 10, End: 19},
	}
	if len(got) != len(want) {
		t.Fatalf("TokensNoWS = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("TokensNoWS[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestTokensMatchSegment: on the golden corpus, the stress corpus and random
// strings, Tokens must produce exactly Segment's strings in order (and
// TokensNoWS exactly SegmentNoWS's), with offsets that slice back to each
// token and reassemble the input.
func TestTokensMatchSegment(t *testing.T) {
	seg := defaultSeg(t)
	var inputs []string
	inputs = append(inputs, readLines(t, "testdata/corpus.txt")...)
	inputs = append(inputs, readLines(t, "testdata/stress.txt")...)
	rng := rand.New(rand.NewSource(7331))
	for i := 0; i < 2000; i++ {
		inputs = append(inputs, randThai(rng, 40))
	}
	for _, in := range inputs {
		toks := seg.Tokens(in)
		checkTokenInvariants(t, in, toks)
		segs := seg.Segment(in)
		if len(toks) != len(segs) {
			t.Fatalf("Tokens len %d != Segment len %d (in=%q)", len(toks), len(segs), in)
		}
		for i := range toks {
			if toks[i].Text != segs[i] {
				t.Fatalf("Tokens[%d].Text=%q != Segment[%d]=%q (in=%q)", i, toks[i].Text, i, segs[i], in)
			}
		}
		nws := seg.TokensNoWS(in)
		segNWS := seg.SegmentNoWS(in)
		if len(nws) != len(segNWS) {
			t.Fatalf("TokensNoWS len %d != SegmentNoWS len %d (in=%q)", len(nws), len(segNWS), in)
		}
		for i := range nws {
			if nws[i].Text != segNWS[i] {
				t.Fatalf("TokensNoWS[%d].Text=%q != SegmentNoWS[%d]=%q (in=%q)", i, nws[i].Text, i, segNWS[i], in)
			}
			if in[nws[i].Start:nws[i].End] != nws[i].Text {
				t.Fatalf("TokensNoWS[%d] offsets don't slice back (in=%q)", i, in)
			}
		}
	}
}

// TestRobustTokens: no panic on adversarial input (incl. invalid UTF-8), and
// the raw-slice invariants hold — text[Start:End]==Text, contiguous full
// coverage — so concatenating Token.Text reproduces the input byte-for-byte
// (Tokens keeps original bytes where Segment substitutes U+FFFD).
func TestRobustTokens(t *testing.T) {
	seg := defaultSeg(t)
	for _, c := range adversarialInputs {
		t.Run(c.name, func(t *testing.T) {
			checkTokenInvariants(t, c.in, seg.Tokens(c.in))
			for i, tk := range seg.TokensNoWS(c.in) {
				if c.in[tk.Start:tk.End] != tk.Text {
					t.Errorf("TokensNoWS[%d] offsets don't slice back (in=%q)", i, c.in)
				}
				if strings.Trim(tk.Text, " ") == "" {
					t.Errorf("TokensNoWS emitted blank token %q", tk.Text)
				}
			}
		})
	}
}

// TestAppendTokensReuse: AppendTokens appends after existing elements and
// reuses capacity.
func TestAppendTokensReuse(t *testing.T) {
	seg := defaultSeg(t)
	dst := seg.AppendTokens(nil, "ฉันรัก")
	n := len(dst)
	if n == 0 {
		t.Fatal("no tokens")
	}
	dst = seg.AppendTokens(dst, "abc")
	if len(dst) <= n {
		t.Fatalf("AppendTokens did not append: len %d -> %d", n, len(dst))
	}
	if dst[n].Text != "abc" || dst[n].Start != 0 {
		t.Errorf("appended token = %+v, want {abc 0 3} (offsets relative to the new call's input)", dst[n])
	}
	// reuse: same backing array when capacity suffices
	buf := make([]token.Token, 0, 64)
	out := seg.AppendTokens(buf, "ฉัน รัก")
	if cap(out) != cap(buf) {
		t.Errorf("AppendTokens reallocated despite capacity (cap %d -> %d)", cap(buf), cap(out))
	}
	if got := seg.AppendTokens(nil, ""); got != nil {
		t.Errorf("AppendTokens(nil, \"\") = %v, want nil", got)
	}
}

// BenchmarkSerialTokens — Tokens vs BenchmarkSerial/SegmentNoWS-style loops:
// the offset variant should stay the same order of magnitude as Segment.
func BenchmarkSerialTokens(b *testing.B) {
	lines, nbytes := benchInput(b)
	seg := defaultSeg(b)
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			sink += len(seg.Tokens(ln))
		}
	}
	_ = sink
}

// BenchmarkSerialSegment — baseline for BenchmarkSerialTokens (same tokens as
// []string, no offsets).
func BenchmarkSerialSegment(b *testing.B) {
	lines, nbytes := benchInput(b)
	seg := defaultSeg(b)
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			sink += len(seg.Segment(ln))
		}
	}
	_ = sink
}

// BenchmarkSerialAppendTokens — the reuse path (dst[:0] across lines), the
// zero-steady-state-allocation way to consume offsets over a corpus.
func BenchmarkSerialAppendTokens(b *testing.B) {
	lines, nbytes := benchInput(b)
	seg := defaultSeg(b)
	buf := make([]token.Token, 0, 256)
	b.SetBytes(nbytes)
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		for _, ln := range lines {
			buf = seg.AppendTokens(buf[:0], ln)
			sink += len(buf)
		}
	}
	_ = sink
}
