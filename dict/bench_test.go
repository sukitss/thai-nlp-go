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

// BenchmarkOpenFlat measures the mmap load path (parse + validate, no copy).
func BenchmarkOpenFlat(b *testing.B) {
	const path = "data/words_th.fdt"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ft, err := OpenFlat(path)
		if err != nil {
			b.Fatal(err)
		}
		ft.Close()
	}
}

// BenchmarkFromBytesEmbedded measures the cold-load path Default() takes
// (copy + parse + validate of the embedded dictionary).
func BenchmarkFromBytesEmbedded(b *testing.B) {
	data := EmbeddedBytes()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ft, err := FromBytes(data)
		if err != nil {
			b.Fatal(err)
		}
		_ = ft
	}
}
