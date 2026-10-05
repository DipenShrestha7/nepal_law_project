import unicodedata
from typing import Any, Dict
from chatbot.models.model import LegalGraphState


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    """
    Formats retrieved documents (from Qdrant or Web Search) into a clean,
    numbered context string for the LLM prompt. Safely handles missing metadata.
    """
    docs = state.get("documents", [])
    if not docs:
        return {"context_str": ""}

    blocks = []
    for idx, doc in enumerate(docs, start=1):
        # 1. Normalize text content safely
        raw_text = doc.get("parent_content_text") or doc.get("content_text") or ""
        clean_text = unicodedata.normalize("NFC", str(raw_text)).strip()

        # 2. Extract core header fields using safe .get()
        act_title = (
            doc.get("act_title") or doc.get("source_file") or "Unknown Legal Source"
        )
        doc_type = doc.get("title") or doc.get("doc_type") or "Provision"

        header_fields = [
            f"Authority: {act_title}",
            f"Document: {doc_type}",
        ]

        # 3. Add statutory identifiers only if they exist and are valid
        art_num = doc.get("article_number")
        if art_num not in (None, "N/A", ""):
            header_fields.append(f"Article: {art_num}")

        sec_num = doc.get("section_number")
        if sec_num not in (None, "N/A", ""):
            header_fields.append(f"Section: {sec_num}")

        sched_num = doc.get("schedule_number")
        if sched_num not in (None, "N/A", ""):
            header_fields.append(f"Schedule: {sched_num}")

        chapter = doc.get("chapter")
        if chapter not in (None, "N/A", ""):
            header_fields.append(f"Chapter/Part: {chapter}")

        # 4. Include source URL for web search results
        source_url = doc.get("source_url")
        if source_url:
            header_fields.append(f"URL: {source_url}")

        # 5. Assemble final formatted block
        header_str = " | ".join(header_fields)
        block = f"[{idx}] {header_str}\nContent: {clean_text}"
        blocks.append(block)

    return {"context_str": "\n\n".join(blocks)}
