import re
from pathlib import Path
from typing import Any

import frontmatter

from policy_answer_service.config import Settings
from policy_answer_service.models import Chunk


def load_documents(policy_docs_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(policy_docs_dir.glob("*.md")):
        post = frontmatter.load(path)
        docs.append(
            {
                "slug": path.stem,
                "title": post.get("title"),
                "effective_date": post.get("effective_date"),
                "version": post.get("version"),
                "body": post.content,
            }
        )
    return docs


def section_slug(section_name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s-]", "", section_name.lower())
    return re.sub(r"\s+", "-", cleaned.strip())


def chunk_document(doc: dict[str, Any]) -> list[Chunk]:
    sections = re.split(r"\n(?=## )", doc["body"])
    chunks: list[Chunk] = []
    for section in sections:
        if not section.strip():
            continue
        header_match = re.match(r"##\s+(.+)", section)
        section_name = header_match.group(1).strip() if header_match else "intro"
        chunks.append(
            Chunk(
                chunk_id=f"{doc['slug']}#{section_slug(section_name)}",
                document_title=doc["title"],
                section=section_name,
                text=section.strip(),
                effective_date=str(doc["effective_date"]),
                version=doc["version"],
            )
        )
    return chunks


def store_chunks(chunks: list[Chunk], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def run_ingestion(settings: Settings) -> list[Chunk]:
    docs = load_documents(settings.policy_docs_dir)
    all_chunks = [c for doc in docs for c in chunk_document(doc)]
    store_chunks(all_chunks, settings.data_dir / "chunks.jsonl")
    return all_chunks
