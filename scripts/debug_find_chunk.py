# scripts/debug_find_chunk.py
"""One-off debug script — find which ingested chunks contain a given figure/phrase.
Not part of the production pipeline; run manually when investigating data issues."""

from src.config import get_opensearch_client, OPENSEARCH_INDEX_NAME  # adjust import path to match your actual config.py


def find_chunks_containing(client, index_name, search_term, size=10):
    query = {
        "query": {"match_phrase": {"text": search_term}},
        "size": size,
    }
    results = client.search(index=index_name, body=query)
    return [hit["_source"].get("text", "") for hit in results["hits"]["hits"]]


if __name__ == "__main__":
    client = get_opensearch_client()  # adjust to whatever your actual function/variable is named
    chunks = find_chunks_containing(client, OPENSEARCH_INDEX_NAME, "30,658")

    for c in chunks:
        print(c[:500])
        print("---")