from typing import Any, Dict, List, Optional, TypedDict


class StatutoryAnchors(TypedDict):
    article_number: Optional[int]
    section_number: Optional[int]
    schedule_number: Optional[int]
    article_numbers: List[int]
    section_numbers: List[int]
    schedule_numbers: List[int]
    target_act: Optional[str]
    is_constitutional_schedule_query: bool


class LegalGraphState(TypedDict):
    question: str
    search_query: str
    search_queries: List[str]
    anchors: StatutoryAnchors
    documents: List[Dict[str, Any]]
    context_str: str
    answer: str
