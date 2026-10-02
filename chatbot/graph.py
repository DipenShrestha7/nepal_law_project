from langgraph.graph import END, StateGraph
from chatbot.models.model import LegalGraphState
from chatbot.nodes.query_node import prepare_query_node
from chatbot.nodes.retrieve_node import retrieve_node
from chatbot.nodes.format_node import format_context_node
from chatbot.nodes.answer_node import generate_answer_node


def build_english_rag_graph():
    workflow = StateGraph(LegalGraphState)
    workflow.add_node("prepare_query", prepare_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_answer_node)

    workflow.set_entry_point("prepare_query")
    workflow.add_edge("prepare_query", "retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
