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

    # 1. PRE-COMPUTE VECTOR EMBEDDINGS ONCE (Saves 50%+ CPU latency)
    encoded_queries = []
    for search_text in search_queries:
        try:
            dense_vec = embedder.encode(search_text).tolist()
            sparse_vec = next(iter(sparse_embedder.embed([search_text])))
            encoded_queries.append((dense_vec, sparse_vec))
        except Exception as e:
            print(f"[Retrieval Error] Failed to encode query '{search_text}': {e}")

    # 2. QUERY QDRANT COLLECTIONS
    for collection_name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name):
            continue

        for dense_vector, sparse_vector in encoded_queries:
            try:
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
            except Exception as err:
                print(
                    f"[Retrieval Error] Query failed on collection '{collection_name}': {err}"
                )

    # 3. SORT & DEDUPLICATE RESULTS
    collected_results.sort(key=lambda item: item[2], reverse=True)
    final_docs = []
    seen_provisions: Dict[str, int] = {}

    for collection_name, point, score in collected_results:
        payload = point.payload or {}
        chunk_data = extract_payload_metadata(point)
        content_text = (
            chunk_data.get("content_text")
            or chunk_data.get("parent_content_text")
            or ""
        )

        # Skip oversized schedules
        if (
            chunk_data.get("doc_type") == "schedule"
            and len(content_text) > MAX_SCHEDULE_CONTEXT_CHARS
        ):
            print(f"Skipping oversized schedule payload from {collection_name}.")
            continue

        # Language quality check
        if not is_english_chunk(content_text):
            continue

        # Constitutional schedule filter rule
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

        # 4. PROVISION DIVERSITY (Tracks section, article, AND schedule)
        sec_num = chunk_data.get("section_number")
        art_num = chunk_data.get("article_number")
        sched_num = chunk_data.get("schedule_number")

        if sec_num is not None:
            prov_key = f"sec_{sec_num}"
        elif art_num is not None:
            prov_key = f"art_{art_num}"
        elif sched_num is not None:
            prov_key = f"sched_{sched_num}"
        else:
            prov_key = "other"

        # Cap max 2 chunks per provision to ensure context diversity
        if prov_key != "other" and seen_provisions.get(prov_key, 0) >= 2:
            continue

        # 5. DEDUPLICATION KEY (Uses normalized chunk_data)
        doc_key = (
            collection_name,
            payload.get("file_name"),
            sec_num or art_num or sched_num or chunk_data.get("title"),
            chunk_data.get("doc_type"),
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
