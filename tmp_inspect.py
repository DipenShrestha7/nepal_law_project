from pathlib import Path
import pdfplumber, re, unicodedata

pdf_path = Path(
    "data/raw_pdfs/np/Act_Not_In_Volume/केही_नेपाल_ऐनको_व्यवस्था_जगाउने_ऐन,_२०६३.pdf"
)
print(
    "exists",
    pdf_path.exists(),
    "size",
    pdf_path.stat().st_size if pdf_path.exists() else "missing",
)

if pdf_path.exists():
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for i, p in enumerate(pdf.pages, 1):
            txt = p.extract_text()
            print(f"--- PAGE {i} START ---")
            print(txt[:3000] if txt else "")
            print(f"--- PAGE {i} END ---")
            pages.append(txt or "")

    text = "\n".join(pages)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"www\.lawcommission\.(?:com\.)?gov\.np", "", text, flags=re.I)
    text = re.sub(r"नेपाल\s*कानून\s*आयोग", "", text)
    text = re.sub(r"\n\s*[०-९\d]+\s*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    print("--- CLEANED TEXT START ---")
    print(text[:12000])
    print("--- CLEANED TEXT END ---")

    patterns = [
        r"प्रस्तावना\s*[:\n]\s*(.*?)(?=१\.)",
        r"(१\.\s*संक्षिप्त नाम र प्रारम्भः.*?)(?=२\.)",
        r"(२\.\s*संशोधन वा खारेज भएका ऐनको व्यवस्था जागेको मानिनेः.*)",
        r"२\.\s*.*?(?=३\.)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        print("PATTERN", pat, "MATCH", bool(m))
        if m:
            print(m.group(0)[:1000])
            print("---")
