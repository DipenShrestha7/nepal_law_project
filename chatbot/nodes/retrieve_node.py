from typing import Any, Dict, Set, Tuple
from qdrant_client import models
from chatbot.models.model import LegalGraphState
from chatbot.utils.encoding import is_english_chunk
from chatbot.utils.extraction import extract_statutory_anchors, extract_payload_metadata
from chatbot.services.retrieval import build_query_filter
from chatbot.services.clients import (
    qdrant,
    embedder,
    sparse_embedder,
    COLLECTIONS,
    MAX_SCHEDULE_CONTEXT_CHARS,
)


def retrieve_node(state: LegalGraphState) -> Dict[str, Any]:
    anchors = state.get("anchors") or extract_statutory_anchors(state["question"])
    search_queries = state.get("search_queries") or [state["question"]]
    collected_results = []
    seen_keys: Set[Tuple[str, Any, Any, Any]] = set()

    query_filter = build_query_filter(anchors)

    for collection_name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name):
            continue

        for search_text in search_queries:
            dense_vector = embedder.encode(search_text).tolist()
            sparse_vector = next(iter(sparse_embedder.embed([search_text])))
            points = qdrant.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=15),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vector.indices.tolist(),
                            values=sparse_vector.values.tolist(),
                        ),
                        using="sparse",
                        limit=15,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=15,
            ).points

            for point in points:
                collected_results.append(
                    (collection_name, point, getattr(point, "score", 1.0))
                )

    collected_results.sort(key=lambda item: item[2], reverse=True)
    final_docs = []
    seen_provisions: Dict[str, int] = {}

    for collection_name, point, score in collected_results:
        payload = point.payload or {}
        chunk_data = extract_payload_metadata(point)
        content_text = (
            chunk_data["content_text"] or chunk_data["parent_content_text"] or ""
        )

        if (
            chunk_data.get("doc_type") == "schedule"
            and len(content_text) > MAX_SCHEDULE_CONTEXT_CHARS
        ):
            print(
                f"Skipping oversized malformed schedule payload from {collection_name}; "
                "re-run constitution ingestion to refresh it."
            )
            continue

        if not is_english_chunk(content_text):
            continue

        if anchors.get("is_constitutional_schedule_query"):
            doc_type = str(
                chunk_data.get("doc_type") or payload.get("doc_type") or ""
            ).lower()
            act_title = str(
                chunk_data.get("act_title") or payload.get("act_title") or ""
            ).lower()
            if "constitution" not in act_title and (
                doc_type == "schedule" or chunk_data.get("schedule_number") is not None
            ):
                continue

        art_num = chunk_data.get("article_number")
        sched_num = chunk_data.get("schedule_number")
        prov_key = (
            f"art_{art_num}"
            if art_num is not None
            else (f"sched_{sched_num}" if sched_num is not None else "other")
        )

        if prov_key != "other" and seen_provisions.get(prov_key, 0) >= 2:
            continue

        doc_key = (
            collection_name,
            payload.get("file_name"),
            payload.get("section_number")
            or payload.get("article_number")
            or payload.get("schedule_number")
            or payload.get("title"),
            payload.get("doc_type"),
        )
        if doc_key in seen_keys:
            continue

        seen_keys.add(doc_key)
        seen_provisions[prov_key] = seen_provisions.get(prov_key, 0) + 1

        chunk_data["collection_name"] = collection_name
        final_docs.append(chunk_data)

        if len(final_docs) >= 8:
            break
    if not final_docs:
        print("[Retrieval] Qdrant returned 0 documents. Routing to Web Search...")

    return {"documents": final_docs}
