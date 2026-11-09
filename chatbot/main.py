import sys
from chatbot.utils.encoding import setup_encoding
from chatbot.graph import build_english_rag_graph


def main():
    # 1. Safely configure terminal UTF-8 encoding
    setup_encoding()

    print("Initializing Nepal Legal RAG Pipeline...")
    try:
        # 2. Compile master workflow graph
        app = build_english_rag_graph()
    except Exception as init_err:
        print(f"Failed to initialize graph pipeline: {init_err}")
        sys.exit(1)

    print("\n=========================")
    print(" Nepal Legal RAG Chatbot")
    print("=========================")
    print("Type 'exit' or 'quit' to terminate.\n")

    # 3. Interactive CLI Execution Loop
    while True:
        try:
            user_input = input("User Query: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting session. Goodbye!")
                break
            if not user_input:
                continue

            # 4. Minimal Initial State (LangGraph auto-manages other keys)
            initial_state = {"question": user_input}

            print("\n[Pipeline] Processing query...")

            # 5. Invoke LangGraph pipeline
            result = app.invoke(initial_state)

            print("\n--- RESPONSE ---")
            print(result.get("answer", "No response generated."))
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession interrupted by user. Goodbye!")
            break
        except Exception as err:
            print(
                f"\n[Execution Error] An error occurred while processing query: {err}\n"
            )


if __name__ == "__main__":
    main()
