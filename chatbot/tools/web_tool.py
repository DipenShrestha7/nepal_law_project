from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_tavily import TavilySearch
from chatbot.models.model import LegalGraphState

load_dotenv()

web_search_tool = TavilySearch(k=3)


def search_web(query: str):
    """Run Tavily and normalize results for the legal-document pipeline."""
    blacklisted_domains = (
        "apnilaw.com",
        "slideshare.net",
        "coursehero.com",
        "studocu.com",
        "scribd.com",
        "academia.edu",
        "researchgate.net",
        "github.com",
        "drive.google.com",
        "wordpress.com",
        "blogspot.com",
        "medium.com",
    )
    search_results = web_search_tool.invoke({"query": query})
    if isinstance(search_results, dict):
        search_results = search_results.get("results", [])
    elif not isinstance(search_results, list):
        search_results = []

    documents = []
    for result in search_results:
        if isinstance(result, dict):
            url = result.get("url") or ""
            title = result.get("title") or "Web Search Result"
            content = result.get("content") or ""
        elif isinstance(result, str):
            url = result if result.startswith("http") else ""
            title = "Web Search Result"
            content = ""
        else:
            continue

        if not url:
            continue
        if any(blocked in url.lower() for blocked in blacklisted_domains):
            continue

        documents.append(
            {
                "act_title": "Web Search Result",
                "title": title,
                "section_number": "N/A",
                "article_number": None,
                "chapter": "N/A",
                "schedule_number": None,
                "content_text": content,
                "source_url": url,
                "doc_type": "web_search",
                "collection_name": "web",
            }
        )

    return documents


def web_search_node(state: LegalGraphState):
    """Executes web search when local vector retrieval fails or is insufficient."""
    question = state["question"]
    target_act = state.get("anchors", {}).get("target_act", "")

    # Formulate a clean legal search query
    search_query = f"{question} {target_act} Nepal legal overview law summary".strip()

    return {"documents": search_web(search_query)}


def route_after_retrieval(state: LegalGraphState):
    """Choose formatted document context or the general web-search fallback."""
    documents = state.get("documents", [])

    if documents:
        return "generate"

    return "web_search"
