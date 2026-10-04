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
    """Decides whether to go to Generator or Web Search Fallback."""
    documents = state.get("documents", [])

    # If Qdrant returned no documents, fallback to Web Search
    if not documents:
        return "web_search"

    return "generate"
