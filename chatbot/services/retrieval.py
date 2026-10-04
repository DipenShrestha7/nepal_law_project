from typing import Union, Dict, Any
from qdrant_client import models
from chatbot.models.model import StatutoryAnchors


def build_query_filter(
    anchors: Union[StatutoryAnchors, Dict[str, Any]],
) -> models.Filter:
    # 1. Normalize Pydantic models or dicts safely
    if hasattr(anchors, "model_dump"):
        data = anchors.model_dump()
    elif hasattr(anchors, "dict"):
        data = anchors.dict()
    elif isinstance(anchors, dict):
        data = anchors
    else:
        data = getattr(anchors, "__dict__", {})

    # Start with base required filters
    must_conditions = [
        models.FieldCondition(key="language", match=models.MatchValue(value="en"))
    ]

    # 2. Dynamic Act Title Matching (Pushed to MUST)
    target_act = data.get("target_act") or data.get("act_title")
    if target_act and str(target_act).strip():
        # Handle shorthand alias if needed, or use exact string
        act_name = (
            "Constitution of Nepal"
            if target_act == "Constitution"
            else target_act.strip()
        )
        must_conditions.append(
            models.FieldCondition(
                key="act_title",
                match=models.MatchValue(value=act_name),
            )
        )

    # 3. Dynamic Section / Article / Schedule Filtering (Pushed to MUST)
    for field_name, plural_name in (
        ("article_number", "article_numbers"),
        ("section_number", "section_numbers"),
        ("schedule_number", "schedule_numbers"),
    ):
        raw_values = data.get(plural_name) or []
        if not raw_values and data.get(field_name) is not None:
            raw_values = [data[field_name]]

        # Cast to integer safely for Qdrant schema match
        clean_values = []
        for val in raw_values:
            if val is not None:
                try:
                    clean_values.append(int(val))
                except (ValueError, TypeError):
                    pass

        if clean_values:
            if len(clean_values) == 1:
                # Single anchor lookup (e.g., Section 11)
                must_conditions.append(
                    models.FieldCondition(
                        key=field_name, match=models.MatchValue(value=clean_values[0])
                    )
                )
            else:
                # Multiple anchor lookup (e.g., Section 177 OR Section 180)
                must_conditions.append(
                    models.FieldCondition(
                        key=field_name, match=models.MatchAny(any=clean_values)
                    )
                )

    # Everything is strictly enforced in must
    return models.Filter(must=must_conditions)
