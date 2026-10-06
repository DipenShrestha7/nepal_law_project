from typing import Any, Dict, Union
from qdrant_client import models
from chatbot.models.model import StatutoryAnchors


def build_query_filter(
    anchors: Union[StatutoryAnchors, Dict[str, Any]],
) -> models.Filter:
    """
    Builds a dynamic Qdrant Filter based on extracted statutory anchors.
    Uses MatchText for flexible full-text token matching on act titles.
    """
    if hasattr(anchors, "model_dump"):
        data = anchors.model_dump()
    elif hasattr(anchors, "dict"):
        data = anchors.dict()
    elif isinstance(anchors, dict):
        data = anchors
    else:
        data = getattr(anchors, "__dict__", {})

    must_conditions = []

    # 1. Flexible Act Title Full-Text Matching (Requires TEXT payload index)
    target_act = data.get("target_act") or data.get("act_title")
    if target_act and str(target_act).strip():
        must_conditions.append(
            models.FieldCondition(
                key="act_title",
                match=models.MatchText(text=str(target_act).strip()),
            )
        )

    # 2. Dynamic Section / Article / Schedule Filtering (Dual Int & Str Matching)
    for field_name, plural_name in (
        ("article_number", "article_numbers"),
        ("section_number", "section_numbers"),
        ("schedule_number", "schedule_numbers"),
    ):
        raw_values = data.get(plural_name) or []
        if not raw_values and data.get(field_name) is not None:
            raw_values = [data[field_name]]

        should_conditions = []
        int_values = []
        str_values = []

        for val in raw_values:
            if val is not None and str(val).strip() != "":
                val_str = str(val).strip()
                str_values.append(val_str)
                try:
                    int_values.append(int(val_str))
                except (ValueError, TypeError):
                    pass

        if int_values:
            if len(int_values) == 1:
                should_conditions.append(
                    models.FieldCondition(
                        key=field_name, match=models.MatchValue(value=int_values[0])
                    )
                )
            else:
                should_conditions.append(
                    models.FieldCondition(
                        key=field_name, match=models.MatchAny(any=int_values)
                    )
                )

        if str_values:
            if len(str_values) == 1:
                should_conditions.append(
                    models.FieldCondition(
                        key=field_name, match=models.MatchValue(value=str_values[0])
                    )
                )
            else:
                should_conditions.append(
                    models.FieldCondition(
                        key=field_name, match=models.MatchAny(any=str_values)
                    )
                )

        if should_conditions:
            must_conditions.append(models.Filter(should=should_conditions))

    return models.Filter(must=must_conditions) if must_conditions else None
