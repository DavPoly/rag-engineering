from policy_answer_service.config import get_settings
from policy_answer_service.ingestion import chunk_document, load_documents


def test_loads_all_20_documents():
    settings = get_settings()
    docs = load_documents(settings.policy_docs_dir)
    assert len(docs) == 20

def test_chunk_ids_match_slug_section_format():
    settings = get_settings()
    docs = load_documents(settings.policy_docs_dir)
    doc = next(d for d in docs if "returns" in d["slug"])
    chunks = chunk_document(doc)
    assert all("#" in c.chunk_id for c in chunks)

def test_metadata_attached():
    settings = get_settings()
    docs = load_documents(settings.policy_docs_dir)
    chunks = chunk_document(docs[0])
    assert all(c.effective_date and c.version for c in chunks)

def test_section_slug_strips_punctuation():
    doc = {
        "slug": "x",
        "title": "X",
        "effective_date": "2025-01-01",
        "version": "1.0",
        "body": "## Step 5: Carrier Review\nText",
    }
    chunks = chunk_document(doc)
    assert chunks[0].chunk_id == "x#step-5-carrier-review"

def test_all_versions_are_strings():
    settings = get_settings()
    docs = load_documents(settings.policy_docs_dir)
    assert all(isinstance(d["version"], str) for d in docs)

def test_conflicting_versions_both_ingested():
    # Both v1 and v2 of the returns policy should produce chunks —
    # filtering by effective_date happens at retrieval/generation time,
    # not here.
    settings = get_settings()
    docs = load_documents(settings.policy_docs_dir)
    slugs = {d["slug"] for d in docs}
    assert "returns-policy-v1" in slugs
    assert "returns-policy-v2" in slugs