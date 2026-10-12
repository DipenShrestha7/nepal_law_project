import asyncio

from chatbot.tools.document.web_document_fetcher import WebDocumentFetcher


async def main():
    fetcher = WebDocumentFetcher()

    # Test with an Act that has an official English version
    result = await fetcher.fetch_best_document("forest act")

    if result.success:
        print("\n--- FETCH SUCCESSFUL ---")
        print(f"Source URL: {result.source_url}")
        print(f"File Type: {result.file_type}")
        print(f"Is English: {result.is_english}")
        print(f"Target Collection: {result.target_collection}")
        if result.file_bytes:
            print(f"Downloaded PDF Size: {len(result.file_bytes)} bytes")
    else:
        print(f"Fetch Failed: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
