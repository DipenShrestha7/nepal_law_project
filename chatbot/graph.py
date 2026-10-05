from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from chatbot.models.model import LegalGraphState
from chatbot.nodes.query_node import prepare_query_node
from chatbot.nodes.retrieve_node import retrieve_node
from chatbot.nodes.format_node import format_context_node
from chatbot.nodes.answer_node import generate_answer_node
from chatbot.tools.web_tool import web_search_node, route_after_retrieval


def build_english_rag_graph() -> CompiledStateGraph:
    """
    Constructs and compiles the master English Legal RAG LangGraph workflow.

    Execution Paths:
    1. Local Vector Hit:
       prepare_query -> retrieve -> format_context -> generate -> END

    2. Web Search Fallback:
       prepare_query -> retrieve -> web_search -> format_context -> generate -> END
    """
    workflow = StateGraph(LegalGraphState)

    # 1. Register all graph nodes
    workflow.add_node("prepare_query", prepare_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_answer_node)

    # 2. Set entry point and primary edge
    workflow.set_entry_point("prepare_query")
    workflow.add_edge("prepare_query", "retrieve")

    # 3. Add conditional fallback edge after vector retrieval
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "web_search": "web_search",
            "generate": "format_context",  # Redirects local retrieval output to format_context
        },
    )

    # 4. Connect fallback path into context formatting
    workflow.add_edge("web_search", "format_context")

    # 5. Connect formatting output to answer generation and termination
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    # Compile executable graph
    return workflow.compile()


# Export compiled application instance
app = build_english_rag_graph()
