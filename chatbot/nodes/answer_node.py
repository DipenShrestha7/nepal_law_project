import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from chatbot.models.model import LegalGraphState
from chatbot.prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT_EN
from chatbot.services.clients import llm


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
    question = state["question"]
    context = state["context_str"]

    if not state["documents"]:
        return {
            "answer": "The provided legal context does not contain information regarding this query."
        }

    print("\n=== RETRIEVED DOCUMENTS IN CONTEXT ===")
    for doc in state.get("documents", []):
        print(f"Act: {doc.get('act_title')} | Section: {doc.get('section_number')}")
    print("======================================\n")

    messages = [
        SystemMessage(content=LEGAL_SYSTEM_PROMPT_EN),
        HumanMessage(content=f"\n{context}\n\n\nUSER QUESTION: {question}"),
    ]
    response = llm.invoke(messages)
    content = str(response.content or "").strip()
    if not content and hasattr(response, "additional_kwargs"):
        content = str(
            response.additional_kwargs.get("reasoning", "")
            or response.additional_kwargs.get("thinking", "")
        ).strip()

    cleaned_answer = re.sub(
        r"^User Safety:\s*\w+\s*", "", content, flags=re.IGNORECASE
    ).strip()

    if not cleaned_answer:
        cleaned_answer = "Error: The model generated an empty response. Please check your API connection or select an explicit OpenRouter model."

    return {"answer": cleaned_answer}
