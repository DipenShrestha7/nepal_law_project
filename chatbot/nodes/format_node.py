import unicodedata
from typing import Any, Dict
from chatbot.models.model import LegalGraphState


def format_context_node(state: LegalGraphState) -> Dict[str, Any]:
    docs = state["documents"]
    if not docs:
        return {"context_str": ""}

    blocks = []
    for idx, doc in enumerate(docs, start=1):
        clean_text = unicodedata.normalize(
            "NFC", doc.get("parent_content_text") or doc.get("content_text") or ""
        )
        header_fields = [
            f"Authority: {doc['act_title']}",
            f"Document: {doc['title'] or doc['doc_type'] or 'Provision'}",
        ]
        if doc["article_number"] is not None:
            header_fields.append(f"Article: {doc['article_number']}")
        if doc["section_number"] != "N/A":
            header_fields.append(f"Section: {doc['section_number']}")
        if doc["schedule_number"] is not None:
            header_fields.append(f"Schedule: {doc['schedule_number']}")
        if doc["chapter"] != "N/A":
            header_fields.append(f"Chapter/Part: {doc['chapter']}")

        block = f"[{idx}] {' | '.join(header_fields)}\nContent: {clean_text}\n"
        blocks.append(block)

    return {"context_str": "\n\n".join(blocks)}
