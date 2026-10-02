from typing import Any, Dict
import re
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
            Your task is to convert a user's query into 3-4 distinct, keyword-dense legal search queries formatted for Hybrid Vector Search across Nepalese Laws (Constitution, Criminal Code, Civil Code, Local Government Operation Act, Electronic Transactions Act, Labour Act, etc.).

            GUIDELINES:
            1. Translate layperson terms into formal statutory vocabulary used in Nepalese law (e.g., convert "murder" -> "homicide prohibition causing death", "doxxing" -> "privacy data character record", "firing" -> "employment termination dismissal").
            2. Map modern or specific scenarios to BOTH specific enabling acts AND general penal/civil codes (e.g., cyber harassment -> Electronic Transactions Act AND outraging modesty / criminal intimidation under Penal Code).
            3. Extract core legal elements, offenses, rights, or remedies.
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

    # Start with the original raw query
    search_queries = [user_query]

    # 1. Inject Extracted Statutory Anchors directly into the search queue
    if isinstance(anchors, dict):
        if anchors.get("article_number"):
            search_queries.append(
                f"Article {anchors['article_number']} Constitution of Nepal"
            )
        if anchors.get("section_number"):
            search_queries.append(
                f"Section {anchors['section_number']} National Penal Code Civil Code"
            )
        if anchors.get("schedule_number"):
            search_queries.append(
                f"Schedule {anchors['schedule_number']} local level powers"
            )

    # 2. Dynamic LLM Query Expansion via LangChain LCEL
    try:
        response_text = query_expansion_chain.invoke({"user_query": user_query})

        # Parse output lines cleanly
        generated_lines = [
            re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
            for line in response_text.splitlines()
            if line.strip()
        ]

        for line in generated_lines[:4]:
            if line and line.lower() != user_query.lower():
                search_queries.append(line)

    except Exception as e:
        # 3. Universal Fallback (Zero hardcoded topics)
        # Fallback to basic keyword extraction if the LLM call fails/times out
        clean_words = re.findall(r"\b\w{4,}\b", user_query)
        if clean_words:
            search_queries.append(" ".join(clean_words))

    # 4. Deduplicate while preserving order
    deduplicated_queries = list(dict.fromkeys([q for q in search_queries if q]))

    return {
        "search_query": " | ".join(deduplicated_queries),
        "search_queries": deduplicated_queries,
        "anchors": anchors,
    }
