import time
from chatbot.test.testHybridEn import build_english_rag_graph

# All 7 test prompts covering core RAG edge cases
TEST_PROMPTS = {
    "Test 2: Warrantless Seizure & 30h Illegal Detention": (
        "Police seized my phone without a warrant for photographing pollution and locked me in a cell for 30 hours "
        "without a lawyer or judge. What rights were violated and which court can issue a writ for my release?"
    ),
    "Test 4: Ward Office Refusing Right to Information": (
        "I submitted a written request to my Ward Office asking for a copy of the budget allocation and tender "
        "documents for a recent bridge construction project in our village. The Ward Chairperson flatly refused "
        "to give me the documents, claiming that public citizens have no right to inspect internal government "
        "financial records. Is the Ward Chairperson allowed to hide this information, and where can I appeal "
        "to force them to release the documents?"
    ),
    "Test 5: Forced Eviction & Illegal House Demolition": (
        "The local roads department arrived with bulldozers at 6 AM without any advance notice, written order, "
        "or offer of alternative shelter/compensation, and destroyed my family's residential home where we have "
        "lived for 15 years to widen a road. When we tried to stand in front of the bulldozers, municipal officers "
        "physically pushed us away. Is it legal for the government to demolish a residential dwelling without prior "
        "notice or compensation? What urgent court order can stop them and get us emergency relief?"
    ),
    "Test 6: Child Labor & Wage Exploitation": (
        "A private garment factory owner in Kathmandu is employing 13-year-old children to work 14-hour night shifts. "
        "He pays them half of the minimum wage, prohibits them from taking weekend breaks, and locks the exit doors "
        "during work hours so they cannot leave. When adult workers tried to form a union to protest these conditions, "
        "the owner fired the union organizers on the spot. Which of our constitutional labor rights are being broken, "
        "and what state authorities should handle the child labor and illegal firing?"
    ),
    "Test 7: Public School Religious & Language Discrimination": (
        "A government-funded public school in my municipality refused to admit my daughter unless she took an exam "
        "in a language she doesn't speak at home, and the principal explicitly told us that they don't accept "
        "students from our religious minority community. Is a public school allowed to deny admission based on religion "
        "or force language barriers on primary students? What court can compel the school to admit her?"
    ),
}

if __name__ == "__main__":
    app = build_english_rag_graph()

    for title, prompt in TEST_PROMPTS.items():
        print(f"\n=================== {title.upper()} ===================")
        print(f"PROMPT: {prompt}\n")

        initial_state = {
            "question": prompt,
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
        print("--- RESPONSE ---")
        print(result["answer"])
        print("-" * 65)

        # Prevent free-tier API rate limiting
        time.sleep(2)
