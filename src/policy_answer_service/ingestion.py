"""
Ingestion pipeline: load policy docs, chunk them, attach metadata, store them.

Fill in once the 20 synthetic policy docs exist in data/policies/.

TODO:
- load_documents(dir) -> list[RawDocument]
- chunk_document(doc, chunk_size_tokens, overlap_tokens) -> list[Chunk]
- attach metadata: title, section, effective_date, version
- store() — pick something simple first (JSON/SQLite); don't reach for a
  vector DB until the retrieval ladder says you need one
"""