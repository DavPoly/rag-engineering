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

**Decision:**
**Alternatives considered:**
**Why:**
**Evidence:**

---

## TODO — Handling the version-conflict policy documents

**Decision:**
**Alternatives considered:**
**Why:**
**Evidence:**

---

<!-- Add more entries as they come up — embedding model choice, the
     response schema shape, anything you reversed your mind on. -->