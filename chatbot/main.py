from chatbot.utils.encoding import setup_encoding
from chatbot.graph import build_english_rag_graph


def main():
    setup_encoding()
    app = build_english_rag_graph()

    print("=========================")
    print(" Nepal Legal RAG Chatbot")
    print("=========================\n")

    while True:
        user_input = input("User Query: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue

        initial_state = {
            "question": user_input,
            "search_query": "",
            "search_queries": [],
            "anchors": {
                "article_number": None,
                "section_number": None,
                "schedule_number": None,
                "article_numbers": [],
                "section_numbers": [],
                "schedule_numbers": [],
                "target_act": None,
                "is_constitutional_schedule_query": False,
            },
            "documents": [],
            "context_str": "",
            "answer": "",
        }
        result = app.invoke(initial_state)
        print("\n--- RESPONSE ---")
        print(result["answer"])
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
