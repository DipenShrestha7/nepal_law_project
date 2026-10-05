import re
import unicodedata
from typing import Any, Dict
from chatbot.models.model import StatutoryAnchors
from chatbot.utils.retrieval_utils import metadata_header


def extract_statutory_anchors(query: str) -> StatutoryAnchors:
    text = unicodedata.normalize("NFC", query).strip()

    # 1. Extract Article Numbers (Handles single and multiple: "Art 18", "Articles 18, 19 and 20")
    article_matches = re.finditer(
        r"\b(?:articles?|art\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3}(?:\s*(?:/|,|and|-)\s*(?:articles?|art\.?)?\s*\d{1,3})*)\b",
        text,
        re.IGNORECASE,
    )
    article_numbers = [
        int(num) for m in article_matches for num in re.findall(r"\d{1,3}", m.group(1))
    ]
    article_num = article_numbers[0] if article_numbers else None

    # 2. Extract Section Numbers (Handles single and multiple: "Section 12", "Sec 12, 13 and 14")
    section_matches = re.finditer(
        r"\b(?:sections?|sec\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3}(?:\s*(?:/|,|and|-)\s*\d{1,3})*)\b",
        text,
        re.IGNORECASE,
    )
    section_numbers = [
        int(num) for m in section_matches for num in re.findall(r"\d{1,3}", m.group(1))
    ]
    section_num = section_numbers[0] if section_numbers else None

    # 3. Extract Schedule Numbers
    schedule_numbers = [
        int(m.group(1))
        for m in re.finditer(
            r"\b(?:schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*(\d{1,2})\b",
            text,
            re.IGNORECASE,
        )
    ]
    schedule_num = schedule_numbers[0] if schedule_numbers else None

    # 4. Constitutional Schedule Flag
    is_const_sched = bool(
        any(num in {5, 6, 7, 8, 9} for num in schedule_numbers)
        and re.search(
            r"\b(?:operational|implement|local government|act|enabl|power|function|duty)\b",
            text,
            re.IGNORECASE,
        )
    )

    # 5. DYNAMIC TARGET ACT EXTRACTION (100% Dynamic, Zero Hardcoding)
    # Case-insensitive match for patterns like: "X Act", "X Act, 2074", "X Code", "X Constitution"
    target_act = None
    act_match = re.search(
        r"\b([a-zA-Z0-9-–—\s]+?\b(?:Act|Code|Rules|Regulations|Procedure|Constitution))\b(?:\s*,?\s*\d{4})?",
        text,
        re.IGNORECASE,
    )
    if act_match:
        extracted = act_match.group(1).strip()
        # Clean up leading noise words like "the", "about the", "under"
        extracted_clean = re.sub(
            r"^(?:what\s+is\s+|about\s+|under\s+|the\s+|in\s+)+",
            "",
            extracted,
            flags=re.IGNORECASE,
        ).strip()
        if len(extracted_clean) > 3:
            target_act = extracted_clean.title()

    return {
        "article_number": article_num,
        "section_number": section_num,
        "schedule_number": schedule_num,
        "article_numbers": article_numbers,
        "section_numbers": section_numbers,
        "schedule_numbers": schedule_numbers,
        "target_act": target_act,
        "is_constitutional_schedule_query": is_const_sched,
    }


def extract_payload_metadata(point) -> Dict[str, Any]:
    payload = point.payload or {}

    act_title = (
        payload.get("act_title")
        or payload.get("title")
        or payload.get("file_name", "Unknown Law").replace(".pdf", "")
    )

    # Normalize section_number to clean integer or None (instead of string "N/A")
    raw_section = (
        payload.get("section_number")
        or payload.get("rule_number")
        or payload.get("clause_label")
        or payload.get("section_no")
    )
    section_num = None
    if raw_section is not None:
        try:
            section_num = int(raw_section)
        except (ValueError, TypeError):
            section_num = str(raw_section)

    chapter = (
        payload.get("chapter") or payload.get("part") or payload.get("category") or None
    )

    content_text = (
        payload.get("content_text")
        or payload.get("text")
        or payload.get("chunk_text")
        or ""
    )
    parent_content = payload.get("parent_content_text") or payload.get("parent_text")
    title = payload.get("title") or payload.get("doc_type") or ""
    article_number = payload.get("article_number")
    schedule_number = payload.get("schedule_number")
    doc_type = payload.get("doc_type")

    # Fallback schedule extraction from header text
    if schedule_number is None:
        header_sample = f"{title} {content_text[:350]}"
        sched_match = re.search(
            r"(?i)\b(?:SCHEDULE|Schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*(\d{1,2})\b",
            header_sample,
        )
        if sched_match:
            schedule_number = int(sched_match.group(1))
            if not doc_type or doc_type == "section":
                doc_type = "schedule"

    return {
        "act_title": act_title,
        "chapter": chapter,
        "section_number": section_num,
        "article_number": article_number,
        "schedule_number": schedule_number,
        "doc_type": doc_type,
        "title": title,
        "content_text": content_text,
        "parent_content_text": parent_content,
        "metadata_header": metadata_header(payload),
        "score": getattr(point, "score", 1.0),
    }
