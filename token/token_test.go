package token

import "testing"

// TestSliceInvariant pins the contract every producer upholds: Start/End are
// byte offsets, End exclusive, and input[Start:End] == Text — including for
// multi-byte scripts and emoji.
func TestSliceInvariant(t *testing.T) {
	const in = "กa🎉ข" // 3 + 1 + 4 + 3 bytes
	toks := []Token{
		{Text: "ก", Start: 0, End: 3},
		{Text: "a", Start: 3, End: 4},
		{Text: "🎉", Start: 4, End: 8},
		{Text: "ข", Start: 8, End: 11},
	}
	prev := 0
	for i, tk := range toks {
		if in[tk.Start:tk.End] != tk.Text {
			t.Errorf("tok %d: in[%d:%d] = %q, want %q", i, tk.Start, tk.End, in[tk.Start:tk.End], tk.Text)
		}
		if tk.Start != prev {
			t.Errorf("tok %d: Start = %d, want contiguous %d", i, tk.Start, prev)
		}
		prev = tk.End
	}
	if prev != len(in) {
		t.Errorf("last End = %d, want %d", prev, len(in))
	}
}

// TestZeroValue: the zero Token is an empty token at offset 0 — safe to slice
// with against any string.
func TestZeroValue(t *testing.T) {
	var tk Token
	if s := "anything"[tk.Start:tk.End]; s != tk.Text {
		t.Errorf("zero Token slice = %q, want %q", s, tk.Text)
	}
}
