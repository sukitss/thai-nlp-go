package search_test

import (
	"fmt"
	"hash/fnv"

	"github.com/sukitss/thai-nlp-go/search/fusion"
	"github.com/sukitss/thai-nlp-go/search/invidx"
	"github.com/sukitss/thai-nlp-go/search/keyword"
	"github.com/sukitss/thai-nlp-go/search/query"
	"github.com/sukitss/thai-nlp-go/search/vector"
	"github.com/sukitss/thai-nlp-go/search/weight"
	"github.com/sukitss/thai-nlp-go/vocab"
)

// embedDim is the dimensionality of the toy bag-of-terms embedding below.
const embedDim = 64

// embed is a stand-in for a real dense embedding model (BGE-m3, etc.). A
// production pipeline would call an embedding server here; to keep this example
// self-contained, deterministic and dependency-free it builds a bag-of-terms
// vector: each term is hashed to a couple of dimensions and accumulated. Two
// texts sharing more terms get a higher cosine — enough to demonstrate the dense
// side of the pipeline and how its ranking composes with the sparse side.
func embed(terms []string) []float32 {
	v := make([]float32, embedDim)
	for _, t := range terms {
		h := fnv.New32a()
		h.Write([]byte(t))
		x := h.Sum32()
		v[x%embedDim] += 1
		v[(x/embedDim)%embedDim] += 1
	}
	return v
}

// Example wires the whole search layer end-to-end on a tiny Thai FAQ corpus:
// index once (vocab + invidx for sparse, a vector matcher for dense), then for a
// messy real-world query run query.ParseExpand -> sparse search + dense search
// -> fusion.RRF and print the top hit.
func Example() {
	// A tiny in-memory FAQ corpus. Doc 0 is the contact card for a person
	// nicknamed "ปลา" who works in IT — the answer to the query below.
	docs := []string{
		"เบอร์โทรของปลา IT คือ 081-234-5678 ติดต่อได้ในเวลาทำการ",
		"วิธีรีเซ็ตรหัสผ่าน wifi ของบริษัทสำหรับพนักงานใหม่",
		"การลาพักร้อนต้องแจ้งล่วงหน้ากี่วันและใครเป็นผู้อนุมัติ",
		"ติดต่อฝ่าย HR เรื่องเงินเดือนและสวัสดิการพนักงาน",
	}

	// --- Index the corpus once. Documents and queries are reduced to the SAME
	// acronym-aware keys (keyword.Keys on the doc side, query.Parse on the query
	// side both fold "it" -> "IT"), so the two sides line up.
	vb := vocab.NewBuilder()
	docKeys := make([][]string, len(docs))
	for i, d := range docs {
		docKeys[i] = keyword.Keys(d) // content keys: stop words/numbers dropped, "IT" kept
		vb.AddDoc(docKeys[i])
	}

	// Sparse side: an inverted index over the vocab term ids, scored with BM25.
	ib := invidx.NewBuilder()
	// Dense side: an exact float32 vector matcher over the toy embeddings.
	mat := vector.NewFlat(vector.NewFloat32(embedDim))
	for i, keys := range docKeys {
		ids := make([]uint32, len(keys))
		for j, k := range keys {
			ids[j] = vb.GetOrAssign(k)
		}
		terms, tfs := invidx.Count(ids)
		ib.Add(uint32(i), terms, tfs)
		mat.Add(uint32(i), embed(keys))
	}
	idx := ib.Build()

	// --- Query time. A messy, conversational Thai query with a stuck acronym
	// ("ปลาit") and filler words ("ช่วย", "หน่อย").
	qp := query.New()
	terms := qp.ParseExpand("ช่วยหาเบอร์ปลาit หน่อย")

	// Sparse search: map query terms to vocab ids (drop unknown), run WAND top-k.
	qids := make([]uint32, 0, len(terms))
	for _, t := range terms {
		if id, ok := vb.ID(t); ok {
			qids = append(qids, id)
		}
	}
	sparse := idx.Search(qids, weight.NewBM25(), 10)

	// Dense search: embed the same query terms, nearest neighbours by cosine.
	dense := mat.TopK(embed(terms), 10)

	// --- Fuse the two rankings with Reciprocal Rank Fusion (scale-free).
	fused := fusion.RRF([][]fusion.Hit{
		fusion.Adapt(sparse, func(h invidx.Hit) fusion.Hit { return fusion.Hit{ID: h.ID, Score: h.Score} }),
		fusion.Adapt(dense, func(h vector.Hit) fusion.Hit { return fusion.Hit{ID: h.ID, Score: float64(h.Score)} }),
	}, 0)

	fmt.Printf("query terms: %v\n", terms)
	fmt.Printf("top hit: [%d] %s\n", fused[0].ID, docs[fused[0].ID])

	// Output:
	// query terms: [เบอร์ ปลา IT ไอที เทคโนโลยีสารสนเทศ]
	// top hit: [0] เบอร์โทรของปลา IT คือ 081-234-5678 ติดต่อได้ในเวลาทำการ
}

// ExampleHNSW shows the dense matcher's two scaling knobs on the same corpus: the
// approximate graph index HNSW (sub-linear TopK) and, on top of it, quantization
// via WithQuantizer to shrink the graph. It contrasts three ways to find the same
// nearest document:
//
//   - Flat + Float32  — exact, the baseline every approximate result is judged against
//   - HNSW  (float32) — approximate graph, here matching the exact top-1
//   - HNSW  + BinarySym — a 32×-smaller graph over binary codes; coarse on its own,
//     so its candidates are re-ranked exactly with Flat.Rescore (the two-stage
//     pattern you use at scale: cheap graph narrows, exact scorer decides)
func ExampleHNSW() {
	docs := []string{
		"เบอร์โทรของปลา IT คือ 081-234-5678 ติดต่อได้ในเวลาทำการ",
		"วิธีรีเซ็ตรหัสผ่าน wifi ของบริษัทสำหรับพนักงานใหม่",
		"การลาพักร้อนต้องแจ้งล่วงหน้ากี่วันและใครเป็นผู้อนุมัติ",
		"ติดต่อฝ่าย HR เรื่องเงินเดือนและสวัสดิการพนักงาน",
	}

	exact := vector.NewFlat(vector.NewFloat32(embedDim))  // exact baseline
	graph := vector.NewHNSW(embedDim, vector.WithSeed(1)) // approximate, float32
	small := vector.NewHNSW(embedDim, vector.WithSeed(1), // approximate, 32× smaller
		vector.WithQuantizer(vector.NewBinarySym(embedDim)))
	for i, d := range docs {
		v := embed(keyword.Keys(d))
		exact.Add(uint32(i), v)
		graph.Add(uint32(i), v)
		small.Add(uint32(i), v)
	}

	q := embed(keyword.Keys("เบอร์ ปลา IT"))
	want := exact.TopK(q, 1)[0].ID

	// Binary graph is coarse: pull a few candidates, then rescore them exactly.
	coarse := small.TopK(q, 3)
	ids := make([]uint32, len(coarse))
	for i, h := range coarse {
		ids[i] = h.ID
	}
	reranked := exact.Rescore(q, ids, 1)[0].ID

	fmt.Printf("float32 HNSW == exact: %v\n", graph.TopK(q, 1)[0].ID == want)
	fmt.Printf("binary HNSW + rescore == exact: %v\n", reranked == want)
	fmt.Printf("answer: %s\n", docs[want])

	// Output:
	// float32 HNSW == exact: true
	// binary HNSW + rescore == exact: true
	// answer: เบอร์โทรของปลา IT คือ 081-234-5678 ติดต่อได้ในเวลาทำการ
}
