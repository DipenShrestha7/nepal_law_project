from qdrant_client import models
from chatbot.models.model import StatutoryAnchors


def build_query_filter(anchors: StatutoryAnchors) -> models.Filter:
    required_conditions = [
        models.FieldCondition(key="language", match=models.MatchValue(value="en"))
    ]
    anchor_conditions = []

    if anchors.get("target_act") == "Constitution":
        required_conditions.append(
            models.FieldCondition(
                key="act_title",
                match=models.MatchValue(value="Constitution of Nepal"),
            )
        )

    for field_name, plural_name in (
        ("article_number", "article_numbers"),
        ("section_number", "section_numbers"),
        ("schedule_number", "schedule_numbers"),
    ):
        values = anchors.get(plural_name) or []
        if not values and anchors.get(field_name) is not None:
            values = [anchors[field_name]]
        if values:
            anchor_conditions.append(
                models.FieldCondition(key=field_name, match=models.MatchAny(any=values))
            )

    return models.Filter(
        must=required_conditions,
        should=anchor_conditions or None,
    )
