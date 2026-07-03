package dict

import "testing"

// benchmark PrefixLens over every start position of a mixed corpus line,
// simulating a tokenizer's dictionary probing.
func benchPrefix(b *testing.B, ft *FlatTrie, text string) {
	rs := []rune(text)
	buf := make([]int, 0, 16)
	b.ReportAllocs()
	b.ResetTimer()
	var n int
	for i := 0; i < b.N; i++ {
		for s := 0; s < len(rs); s++ {
			buf = ft.PrefixLens(rs, s, buf)
			n += len(buf)
		}
	}
	_ = n
}

func BenchmarkPrefixLensThai(b *testing.B) {
	ft, err := Default()
	if err != nil {
		b.Fatal(err)
	}
	benchPrefix(b, ft, "ผมชอบกินข้าวผัดกับไข่ดาวในวันที่อากาศดีมากๆเลยครับ")
}
