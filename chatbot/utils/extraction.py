import re
import unicodedata
from typing import Any, Dict
from chatbot.models.model import StatutoryAnchors
from chatbot.utils.retrieval_utils import metadata_header


def extract_statutory_anchors(query: str) -> StatutoryAnchors:
    text = unicodedata.normalize("NFC", query).strip()

    article_numbers = [
        int(number)
        for match in re.finditer(
            r"\b(?:articles?|art\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3}(?:\s*(?:/|,|and)\s*(?:articles?|art\.?)?\s*\d{1,3})*)",
            text,
            re.IGNORECASE,
        )
        for number in re.findall(r"\d{1,3}", match.group(1))
    ]
    article_num = article_numbers[0] if article_numbers else None

    section_numbers = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:section|sec\.?)\s*(?:no\.?|number)?\s*[-:]?\s*(\d{1,3})\b",
            text,
            re.IGNORECASE,
        )
    ]
    section_num = section_numbers[0] if section_numbers else None

    schedule_numbers = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(?:schedule|sched\.?)\s*(?:[-–—:]|no\.?|number)?\s*(\d{1,2})\b",
            text,
            re.IGNORECASE,
        )
    ]
    schedule_num = schedule_numbers[0] if schedule_numbers else None

    is_const_sched = bool(
        any(number in [5, 6, 7, 8, 9] for number in schedule_numbers)
        and re.search(
            r"\b(?:operational|implement|local government|act|enabl|power|function|duty)\b",
            text,
            re.IGNORECASE,
        )
    )

    target_act = None
    if re.search(r"\b(?:electronic transactions?|cyber|et act)\b", text, re.IGNORECASE):
        target_act = "Electronic Transactions Act"
    elif re.search(
        r"\b(?:criminal code|penal code|muluki criminal)\b", text, re.IGNORECASE
    ):
        target_act = "Penal Code"
    elif re.search(r"\b(?:civil code|muluki civil)\b", text, re.IGNORECASE):
        target_act = "Civil Code"
    elif re.search(
        r"\b(?:local government|local governance|lgoa)\b", text, re.IGNORECASE
    ):
        target_act = "Local Government Operation Act"
    elif re.search(r"\b(?:constitution)\b", text, re.IGNORECASE):
        target_act = "Constitution"
    else:
        act_match = re.search(
            r"\b([A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*)*\s+Act)"
            r"(?:\s*,?\s*\d{4})?\b",
            text,
        )
        if act_match:
            target_act = re.sub(r"\s+", " ", act_match.group(1)).strip()

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
    section_num = (
        payload.get("section_number")
        or payload.get("rule_number")
        or payload.get("clause_label")
        or payload.get("section_no")
        or "N/A"
    )
    chapter = (
        payload.get("chapter")
        or payload.get("part")
        or payload.get("category")
        or "N/A"
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
