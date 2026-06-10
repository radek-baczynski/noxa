OPENAPI_DESCRIPTION = """
Self-hosted web search and query answering for AI agents.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /web_search` | Search the web via DuckDuckGo |
| `POST /web_fetch` | Fetch and extract a single page |
| `POST /web_crawl` | Deep-crawl from seed URLs |
| `POST /content_select` | Select relevant passages from documents |
| `POST /web_answer` | Full search → fetch → answer pipeline |

Use **Try it out** on each operation below to send requests from this page.

Modes for `/content_select` and `/web_answer`: `fast`, `default`, `quality`.
"""

OPENAPI_TAGS = [
    {
        "name": "search",
        "description": "Web search via DDGS.",
    },
    {
        "name": "fetch",
        "description": "Single-page fetch and markdown extraction (Crawl4AI).",
    },
    {
        "name": "crawl",
        "description": "Multi-page deep crawl from seed URLs.",
    },
    {
        "name": "retrieval",
        "description": "Hybrid BM25 + embedding retrieval with reranking.",
    },
    {
        "name": "answer",
        "description": (
            "End-to-end web search and grounded answer generation. "
            "Use `return_debug: true` on `/web_answer` to emit pipeline dumps "
            "under `dump/` (paths returned in `debug.dump_dir` / `debug.meta_md`)."
        ),
    },
    {
        "name": "meta",
        "description": "Service health and metadata.",
    },
]
