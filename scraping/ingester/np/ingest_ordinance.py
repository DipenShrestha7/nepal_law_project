import os
from pathlib import Path
from qdrant_client import QdrantClient
from .utils.qdrant_utils import build_point, push_to_qdrant
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
ORDINANCE_DIR = Path("data/raw_pdfs/np/Ordinance")


def process_ordinance_folder():
    points = []
    point_id = 300000  # Unique ID offset range for Ordinances

    for pdf_path in ORDINANCE_DIR.glob("*.pdf"):
        # Custom parsing logic for Ordinance PDFs
        # ... (Extract text, extract दफा numbers) ...

        payload = {
            "act_title": pdf_path.stem.replace("_", " "),
            "doc_type": "section",
            "section_number": 1,  # Extracted from text
            "language": "np",
        }

        point = build_point(point_id, "दफा १...", payload)
        points.append(point)
        point_id += 1

    push_to_qdrant(qdrant, "nepal_legal_corpus", points)


if __name__ == "__main__":
    process_ordinance_folder()
