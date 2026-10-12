from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_community.tools import TavilySearchResults
from chatbot.models.model import LegalGraphState

load_dotenv()

web_search_tool = TavilySearchResults(k=3)


def search_web(query: str):
    """Run Tavily and normalize results for the legal-document pipeline."""
    search_results = web_search_tool.invoke({"query": query})
    return [
        {
            "act_title": "Web Search Result",
            "title": result.get("title") or "Web Search Result",
            "section_number": "N/A",
            "article_number": None,
            "chapter": "N/A",
            "schedule_number": None,
            "content_text": result.get("content") or "",
            "source_url": result.get("url") or "",
            "doc_type": "web_search",
            "collection_name": "web",
        }
        for result in search_results
    ]


def web_search_node(state: LegalGraphState):
    """Executes web search when local vector retrieval fails or is insufficient."""
    question = state["question"]
    target_act = state.get("anchors", {}).get("target_act", "")

    # Formulate a clean legal search query
    search_query = f"{question} {target_act} Nepal legal overview law summary".strip()

    return {"documents": search_web(search_query)}


def route_after_retrieval(state: LegalGraphState):
    """Choose local context, full-document search, or general web search."""
    documents = state.get("documents", [])

    if documents:
        return "generate"

    question = state.get("question", "").lower()
    needs_full_document = any(
        keyword in question
        for keyword in (
            "act",
            "ain",
            "section",
            "dafa",
            "punishment",
            "fine",
            "provision",
            "law",
            "statute",
            "rule",
            "code",
            "ऐन",
            "दफा",
        )
    )

    return "doc_search" if needs_full_document else "web_search"
