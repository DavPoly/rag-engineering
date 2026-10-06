# Decision Log

What was chosen, what was rejected, why, and what the numbers showed. This
becomes interview material — write entries as you go, not retroactively.

Format per entry:

```text
## YYYY-MM-DD — <short title>

**Decision:** what you chose
**Alternatives considered:** at least one simpler, one more complex
**Why:** the reasoning, tied to a number where possible
**Evidence:** the eval result that drove it (or "none yet — revisit")
```

---

## 2026-10-03 — Chunking strategy

**Decision:** One chunk per `## ` section of each policy document. The section header stays in the chunk text. Chunk IDs are `<doc-slug>#<section-slug>`, for example `returns-policy-v2#restocking-fee`. Text before the first header becomes an `intro` chunk. `###` sub-headers stay inside their parent section. No token windowing.

**Alternatives considered:** (1) Fixed token windows, using the existing `chunk_size_tokens=300` and `chunk_overlap_tokens=50` settings. This is the simplest option but ignores document structure, so it can split a clause from its heading. (2) Splitting at `###` or paragraph level. This gives finer granularity at the cost of more chunks and less context per chunk, and these short policies don't need it. (3) LLM-based or semantic chunking. This is the most complex and costs money per document.

**Why:** The policies already come as short, named sections, and the section is the unit a citation needs. The golden set's `expected_chunks` are written in this ID format, so section chunks make citation checks exact. Measured on the 20 documents: 176 chunks, median 43 words, maximum 531, 15 chunks under 30 words, and 1 chunk over 300 words. The 15 short chunks may lack context on their own, which is why the header line is kept in the text.

**Evidence:** none yet, to revisit. The only check so far is structural: all `expected_chunks` IDs in the 30-question golden set resolve to produced chunk IDs. Retrieval recall@5 has not been measured. Revisit when fixed windows and section chunks can be compared on the same golden set.

---

## TODO — Where to stop on the retrieval ladder

**Decision:** Ship Vector search alone as the primary retriever. Reject
Hybrid (RRF-fused) and both reranked variants for production use.

**Alternatives considered:**
- Hybrid (BM25 + Vector via RRF): rejected — underperformed Vector alone
  on both recall@5 (0.48 vs 0.80) and MRR (0.32 vs 0.49). Regression
  analysis showed BM25 misses the correct chunk entirely on most
  natural-language queries, and RRF rewards chunks both retrievers
  weakly agree on over chunks Vector alone ranks highly — "mutual
  mediocrity" beating one strong signal.
- Vector + reranker: recall@5 0.88, MRR 0.61 vs Vector alone's 0.80/0.49
  — a real quality gain, but at 310ms vs 36ms (8.6x latency). Rejected
  for the live /ask endpoint; would reconsider for an offline/batch
  review workflow where latency doesn't block a human in real time.
- Hybrid + reranker: performs ~identically to Vector + reranker
  (0.88/0.57/319ms), confirming BM25 adds no value here even after
  reranking cleans up its noise.

**Why:** The endpoint serves a live support-agent chat flow; latency
degrades the interaction more than the marginal recall gain improves it
at this stage. BM25's actual strength (exact product codes) isn't
reflected in this golden set's natural-language phrasing — see note
below on routing.

**Evidence:** Measured 2026-10-05 on the 25 answerable and multi-doc
golden questions, with the `liquid/lfm-2.5-embedding-350m:free` embedding
model. Recall@5 / MRR: BM25 0.28 / 0.21; vector 0.88 / 0.68; hybrid (RRF)
0.52 / 0.30. Vector alone is best on both metrics, so hybrid stays
rejected. Latency was not measured in this run, because the eval timed
precomputed queries. The vector misses are q04, q10, and q25, and all
three also degraded answer quality in the faithfulness run. The reranker
numbers above came from the MiniLM model and are not comparable to these.

Revisit when query volume on exact product codes grows meaningfully
(route those to BM25 specifically, not via RRF blending), or if the
/ask endpoint moves to an async/reviewed workflow where latency budget
loosens.
