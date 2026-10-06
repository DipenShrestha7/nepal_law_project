from chatbot.services.clients import qdrant, COLLECTIONS
from qdrant_client import models

OLD_TITLE = "National Civil Procedure Code"
NEW_TITLE = "National Civil Code"

OLD_FILE = "national_civil_procedure_code.pdf"
NEW_FILE = "national_civil_code.pdf"

collection_name = COLLECTIONS[0]
offset = None
updated_count = 0

while True:
    records, next_offset = qdrant.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="act_title",
                    # Changed from MatchValue to MatchText to match the TEXT payload index
                    match=models.MatchText(text=OLD_TITLE),
                )
            ]
        ),
        with_payload=True,
        with_vectors=False,
    )

    if not records:
        break

    for record in records:
        payload = record.payload or {}

        # 1. Replace act title and source filename
        payload["act_title"] = NEW_TITLE
        payload["source_file"] = NEW_FILE

        # 2. Replace source filename in parent_id
        if payload.get("parent_id"):
            payload["parent_id"] = payload["parent_id"].replace(OLD_FILE, NEW_FILE)

        # 3. Replace title inside citation string
        if payload.get("citation"):
            payload["citation"] = payload["citation"].replace(OLD_TITLE, NEW_TITLE)

        # 4. Replace title in content_text if present
        if payload.get("content_text"):
            payload["content_text"] = payload["content_text"].replace(
                OLD_TITLE, NEW_TITLE
            )

        # Apply updated payload to the point in Qdrant
        qdrant.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=[record.id],
        )
        updated_count += 1

    offset = next_offset
    if offset is None:
        break

print(f"Successfully updated payload across {updated_count} points in Qdrant!")
