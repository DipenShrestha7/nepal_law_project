import re
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from chatbot.models.model import LegalGraphState
from chatbot.utils.extraction import extract_statutory_anchors
from chatbot.services.clients import llm


class LegalSearchPlan(BaseModel):
    official_act_en: str = Field(
        description="Official English title of the governing Nepalese statute (e.g., 'National Civil (Code) Act, 2017')"
    )
    official_act_ne: str = Field(
        description="Official Nepali title in Devanagari (e.g., 'मुलुकी देवानी संहिता, २०७४')"
    )
    statutory_keywords: List[str] = Field(
        description="3-5 formal statutory terms or phrases likely to appear inside the exact section (e.g., ['conclude marriage', 'twenty years of age'])"
    )
    sub_queries: List[str] = Field(
        description="1-3 focused, keyword-dense search phrases for in-memory BM25 section matching"
    )


SYSTEM_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Nepalese legal research assistant and search query optimizer.
Your task is to analyze a user's question and build a structured search plan for retrieving the governing Nepalese statute PDF and matching the exact sections via in-memory lexical (BM25) search.

GUIDELINES:
1. Identify the Statute: Determine the exact official English and Devanagari Nepali titles of the governing Nepalese Act or Regulation—even if the user did not name any Act in their question.
   - Normalize colloquial names to official titles (e.g., "Muluki Civil Code" -> "National Civil (Code) Act, 2017" / "मुलुकी देवानी संहिता, २०७४").
   - For open questions without an Act name, infer the governing law (e.g., "minimum age to marry" -> "National Civil (Code) Act, 2017"; "murder" -> "National Penal (Code) Act, 2017").
2. Statutory Keywords: Provide 3-5 formal terms actually printed inside Nepalese statutory text that verify the right section was retrieved (e.g., for marriage age, use "twenty years of age" and "conclude marriage" rather than "minimum age").
3. Sub-Queries for BM25: Convert layperson phrasing into 1-3 keyword-dense queries using formal statutory vocabulary.
    For Example:
   - "haddamyad" -> "statute of limitation time limit appeal days"
   - "murder" -> "homicide intention of causing death life imprisonment"
   - "firing" -> "employment termination dismissal notice period"
   - If the question has multiple parts (e.g., registration AND cancellation), create separate sub-queries for each part.""",
        ),
        ("human", "{user_query}"),
    ]
)

# Bind structured Pydantic output directly to the LLM
query_planner_chain = SYSTEM_EXPANSION_PROMPT | llm.with_structured_output(
    LegalSearchPlan
)


def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    user_query = (state.get("question") or state.get("user_query") or "").strip()
    anchors = extract_statutory_anchors(user_query)

    official_act_en = ""
    official_act_ne = ""
    statutory_keywords: List[str] = []
    sub_queries: List[str] = []

    # 1. DYNAMIC LLM STATUTE RESOLUTION & QUERY EXPANSION
    try:
        plan: LegalSearchPlan = query_planner_chain.invoke({"user_query": user_query})
        official_act_en = plan.official_act_en.strip()
        official_act_ne = plan.official_act_ne.strip()
        statutory_keywords = [
            kw.strip() for kw in plan.statutory_keywords if kw.strip()
        ]
        sub_queries = [sq.strip() for sq in plan.sub_queries if sq.strip()]

    except Exception as e:
        print(f"[Query Node Warning] Structured query planning failed: {e}")
        # Fallback if structured output fails
        if isinstance(anchors, dict) and anchors.get("target_act"):
            official_act_en = anchors["target_act"]
        clean_words = re.findall(r"\b[A-Za-z]{4,}\b", user_query)
        if clean_words:
            statutory_keywords = clean_words[:5]
            sub_queries = [" ".join(clean_words)]

    # 2. INJECT EXPLICIT STATUTORY ANCHORS (Section / Article / Schedule numbers)
    if isinstance(anchors, dict):
        if anchors.get("target_act") and not official_act_en:
            official_act_en = anchors["target_act"]

        for key, label in [
            ("section_number", "Section"),
            ("article_number", "Article"),
            ("schedule_number", "Schedule"),
        ]:
            val = anchors.get(key)
            if val is not None:
                anchor_term = f"{label} {val}"
                sub_queries.insert(0, anchor_term)
                if anchor_term not in statutory_keywords:
                    statutory_keywords.insert(0, anchor_term)

    # 3. DEDUPLICATE & CAP QUERIES FOR FAST IN-MEMORY BM25
    if not sub_queries:
        sub_queries = [user_query]

    deduplicated_queries = list(dict.fromkeys(sub_queries))[:4]

    return {
        "official_act_en": official_act_en,
        "official_act_ne": official_act_ne,
        "statutory_keywords": statutory_keywords,
        "sub_queries": deduplicated_queries,
        # Kept for backward compatibility with existing graph nodes
        "search_queries": deduplicated_queries,
        "search_query": " | ".join(deduplicated_queries),
        "anchors": anchors,
    }
