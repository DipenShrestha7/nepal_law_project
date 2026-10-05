import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from chatbot.models.model import LegalGraphState
from chatbot.prompt.legal_system_prompt import LEGAL_SYSTEM_PROMPT_EN
from chatbot.services.clients import llm


def generate_answer_node(state: LegalGraphState) -> Dict[str, Any]:
    """
    Final generation node that constructs the prompt, invokes the LLM,
    and handles output cleaning and fallback execution.
    """
    question = state.get("question", "").strip()
    context = state.get("context_str", "").strip()
    docs = state.get("documents", [])

    # 1. Early return if no context/documents were retrieved
    if not docs or not context:
        return {
            "answer": "The provided legal context does not contain information regarding this query."
        }

    # 2. Terminal debug log
    print("\n=== RETRIEVED DOCUMENTS IN CONTEXT ===")
    for doc in docs:
        act = doc.get("act_title") or doc.get("source_file") or "Web/External Source"
        sec = (
            doc.get("section_number")
            or doc.get("article_number")
            or doc.get("schedule_number")
            or "N/A"
        )
        print(f"Act: {act} | Provision: {sec}")
    print("======================================\n")

    # 3. Formulate structured messages
    formatted_human_prompt = (
        f"=== LEGAL CONTEXT ===\n"
        f"{context}\n"
        f"=====================\n\n"
        f"USER QUESTION: {question}"
    )

    messages = [
        SystemMessage(content=LEGAL_SYSTEM_PROMPT_EN),
        HumanMessage(content=formatted_human_prompt),
    ]

    try:
        # 4. Invoke LLM
        response = llm.invoke(messages)
        content = str(response.content or "").strip()

        # Fallback for OpenRouter / DeepSeek reasoning models using additional_kwargs
        if not content and hasattr(response, "additional_kwargs"):
            content = str(
                response.additional_kwargs.get("reasoning", "")
                or response.additional_kwargs.get("thinking", "")
            ).strip()

        # 5. Strip ... reasoning blocks if present
        content = re.sub(r".*?", "", content, flags=re.DOTALL).strip()

        # 6. Clean legacy system safety prefixes
        cleaned_answer = re.sub(
            r"^User Safety:\s*\w+\s*", "", content, flags=re.IGNORECASE
        ).strip()

        if not cleaned_answer:
            cleaned_answer = (
                "Error: The model generated an empty response. "
                "Please check your API key or OpenRouter model configuration."
            )

        return {"answer": cleaned_answer}

    except Exception as err:
        print(f"[Generate Node Error] Exception during LLM generation: {err}")
        return {
            "answer": (
                "An unexpected error occurred while generating the legal answer. "
                "Please try again or refine your query."
            )
        }
