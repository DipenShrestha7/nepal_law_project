import asyncio

from chatbot.tools.document.data_extractor import live_web_search_and_extract
from chatbot.models.model import LegalGraphState


def doc_search_node(state: LegalGraphState) -> dict:
    """
    Handles deep PDF search, RAM extraction, and translation when
    a specific legal act is missing from the local database.
    """
    question = state.get("question", "")
    target_act = (state.get("anchors") or {}).get("target_act") or question
    print(
        f"[Node: doc_search] Fetching & extracting official document for: '{target_act}'"
    )

    result = asyncio.run(
        live_web_search_and_extract(
            act_name=target_act,
            user_query=question,
        )
    )

    if result["success"]:
        return {
            "documents": [
                {
                    "act_title": target_act,
                    "title": "Official document excerpt",
                    "content_text": result["extracted_context"],
                    "source_url": result["source_url"],
                    "doc_type": "document_pdf",
                }
            ]
        }

    print(f"[Node: doc_search] {result.get('error', 'Document search failed.')}")
    return {"documents": []}
