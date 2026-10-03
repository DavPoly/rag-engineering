"""
Tests for retrieval.py.

TODO once each rung is implemented:
- test_bm25_finds_exact_product_code  # the doc planted to favour keyword search
- test_vector_retriever_returns_top_k
- test_reciprocal_rank_fusion_combines_lists
- test_recall_at_5_on_golden_set (mark as slow/integration, not unit)

Mock embedding-model calls in true unit tests — reserve real model loads
for the eval scripts in evals/, run deliberately.
"""