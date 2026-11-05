import re
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chatbot.models.model import LegalGraphState
from chatbot.utils.extraction import extract_statutory_anchors
from chatbot.services.clients import llm

# Universal system prompt for dynamic legal vocabulary expansion
SYSTEM_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Nepalese legal search query optimizer. 
Your task is to convert a user's query into 2-3 distinct, keyword-dense legal search queries formatted for Hybrid Vector Search across Nepalese statutory laws, acts, and regulations.

GUIDELINES:
1. Translate layperson terms into formal statutory vocabulary used in Nepalese legal texts (e.g., convert "murder" -> "homicide prohibition causing death", "doxxing" -> "privacy data character record", "firing" -> "employment termination dismissal").
2. Map modern or specific scenarios to legal concepts, offenses, rights, duties, or statutory remedies.
3. Keep queries concise, factual, and focused on key legal terms.
4. Output ONLY the search queries, one per line. Do not include bullet points, numbers, or introductory text.""",
        ),
        ("human", "{user_query}"),
    ]
)

# Build an LCEL chain
query_expansion_chain = SYSTEM_EXPANSION_PROMPT | llm | StrOutputParser()


def prepare_query_node(state: LegalGraphState) -> Dict[str, Any]:
    user_query = (state.get("question") or "").strip()
    anchors = extract_statutory_anchors(user_query)

    search_queries = [user_query]

    # 1. DYNAMIC ANCHOR INJECTION (Zero Hardcoding)
    if isinstance(anchors, dict):
        target_act = anchors.get("target_act") or ""
        sec_num = anchors.get("section_number")
        art_num = anchors.get("article_number")
        sched_num = anchors.get("schedule_number")

        if sec_num is not None:
            anchor_str = f"Section {sec_num} {target_act}".strip()
            search_queries.append(anchor_str)

        if art_num is not None:
            anchor_str = f"Article {art_num} {target_act}".strip()
            search_queries.append(anchor_str)

        if sched_num is not None:
            anchor_str = f"Schedule {sched_num} {target_act}".strip()
            search_queries.append(anchor_str)

        # If user asked about an Act generally without section numbers
        if target_act and sec_num is None and art_num is None and sched_num is None:
            search_queries.append(f"{target_act} overview scope preamble section 1")

    # 2. DYNAMIC LLM QUERY EXPANSION
    try:
        response_text = query_expansion_chain.invoke({"user_query": user_query})

        generated_lines = [
            re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
            for line in response_text.splitlines()
            if line.strip()
        ]

        # Limit LLM expansion to top 2 variations to prevent vector search latency bloat
        for line in generated_lines[:2]:
            if line and line.lower() != user_query.lower():
                search_queries.append(line)

    except Exception as e:
        print(f"[Query Node Warning] LLM Query Expansion failed: {e}")
        # Universal Fallback: Extract meaningful keywords if LLM call fails
        clean_words = re.findall(r"\b[A-Za-z]{4,}\b", user_query)
        if clean_words:
            search_queries.append(" ".join(clean_words))

    # 3. Deduplicate while preserving order and limit to max 4 total queries
    deduplicated_queries = list(dict.fromkeys([q for q in search_queries if q]))[:4]

    return {
        "search_query": " | ".join(deduplicated_queries),
        "search_queries": deduplicated_queries,
        "anchors": anchors,
    }
