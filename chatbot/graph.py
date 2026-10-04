from langgraph.graph import END, StateGraph
from chatbot.models.model import LegalGraphState
from chatbot.nodes.query_node import prepare_query_node
from chatbot.nodes.retrieve_node import retrieve_node
from chatbot.nodes.format_node import format_context_node
from chatbot.nodes.answer_node import generate_answer_node
from chatbot.tools.web_tool import web_search_node, route_after_retrieval


def build_english_rag_graph():
    workflow = StateGraph(LegalGraphState)

    # Add all nodes
    workflow.add_node("prepare_query", prepare_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_answer_node)

    # Set entry point
    workflow.set_entry_point("prepare_query")
    workflow.add_edge("prepare_query", "retrieve")

    # Conditional routing: Send to 'web_search' or 'format_context'
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "web_search": "web_search",
            "generate": "format_context",  # Maps 'generate' route target to format_context first
        },
    )

    # Web search MUST go to format_context to process web docs
    workflow.add_edge("web_search", "format_context")

    # Formatted context flows into answer generation
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
