I drafted this as a full PRD for the self-hosted runtime web search + answer service we designed in the conversation.

PRD: Self-Hosted Runtime Web Search & Query Answering Service

1. Product Name

Working name: Noxa

Alternative names:

* OpenSearch Agent Tool
* Runtime Research API
* Self-Hosted Web Answer Gateway
* Agent Search Runtime

This PRD uses Noxa as the working name.

⸻

2. One-Line Summary

Noxa is a self-hosted, low-cost, high-concurrency web search and query-answering service for AI agents, using ddgs for discovery, fast runtime crawling, hybrid retrieval, reranking, and a tiny local LLM to answer from selected evidence with citations.

⸻

3. Product Context

AI agents need web access, but hosted search APIs such as Exa, Tavily, Brave, Google Search API, or SerpAPI can become expensive, rate-limited, or hard to control at scale.

The goal is not to build a global search engine. The goal is to build a runtime research pipeline:

query
  -> search web
  -> fetch top pages
  -> extract readable content
  -> select best passages
  -> answer from limited context

The system should be cheap enough to run inside a broader agent platform, including Hermes-style agents, coding agents, personal agents, and internal company agents.

The core design principle is:

Do not make the answer model read the web.
Make retrieval select the best evidence first.
Then ask a tiny model to answer only from that evidence.

⸻

4. Problem Statement

Agents often need to answer questions like:

* “Does this TV support kiosk mode?”
* “Can this library run in GitHub Actions?”
* “What is the current pricing?”
* “What does this docs page say about launch-on-boot?”
* “Find the relevant part of documentation and answer with sources.”
* “Read multiple pages quickly and give me a grounded answer.”

Today, the common options are:

1. Use hosted search APIs.
2. Use a browser agent.
3. Use an LLM with built-in browsing.
4. Build a custom web crawler/index.

Each option has problems.

Hosted search APIs are simple but can be expensive and hard to control. Browser agents are slow and unreliable. LLM browsing is not available everywhere and is hard to self-host. Building a full web index is unrealistic.

The product solves this with a lightweight self-hosted pipeline optimized for runtime search over a small number of pages, not global indexing.

⸻

5. Target Users

5.1 Primary User: Agent Platform Developer

A developer building autonomous agents who needs a reliable internal web search tool.

Needs:

* Cheap web search.
* Self-hostable stack.
* JSON API.
* Fast enough for agent runtime.
* Citations and source links.
* Predictable cost.
* Configurable quality/speed modes.

5.2 Secondary User: AI Agent

The actual caller is often an AI agent using tools.

The agent needs:

* web_search for raw search results.
* web_fetch for a single URL.
* web_crawl for bounded deep crawl from seed URLs.
* web_answer for search + answer.
* Stable structured output.
* Clear errors.
* Source citations.

5.3 Tertiary User: Internal Company Assistant

A company assistant using the service to research public docs, pricing pages, changelogs, GitHub repos, and product pages.

Needs:

* Grounded answers.
* No hallucinated claims.
* Source visibility.
* Reasonable latency.
* Ability to abstain when sources are insufficient.

⸻

6. Goals

6.1 Product Goals

The system should:

1. Provide a self-hosted alternative to hosted web-answer APIs.
2. Use free or low-cost discovery through ddgs.
3. Fetch and extract readable content from top search results.
4. Select the best content before calling the answer LLM.
5. Use small/tiny local models for high concurrency.
6. Support CPU and GPU deployment.
7. Return answers with citations.
8. Provide separate tools for search, fetch, crawl, content selection, and answer.
9. Be simple enough to run as a Python service.
10. Be extensible to paid providers later.

6.2 Technical Goals

The system should:

1. Use async runtime crawling.
2. Use Crawl4AI (headless browser) for fetch and markdown extraction in v0.
3. Use hybrid retrieval: BM25 + embeddings.
4. Use a neural reranker for final passage selection.
5. Keep answer model context small.
6. Support multiple runtime modes: fast, default, quality.
7. Cache fetched pages, extracted content, embeddings, and results.
8. Expose OpenAI/tool-call-friendly APIs.
9. Provide observability around source quality, latency, and answer faithfulness.
10. Be benchmarkable with an internal eval set.

⸻

7. Non-Goals

The system will not initially:

1. Build or maintain a full web index.
2. Replace Google-scale search quality.
3. Crawl the whole internet.
4. Handle deep crawling synchronously.
5. Guarantee access to JavaScript-heavy or paywalled pages.
6. Bypass robots.txt, logins, anti-bot systems, or access restrictions.
7. Be a browser automation agent.
8. Store all crawled web pages permanently by default.
9. Answer questions without sources.
10. Fine-tune models in v0.

⸻

8. Product Principles

8.1 Evidence First

The system should optimize for selecting the best evidence before generating an answer.

Bad:

Send 20 pages to LLM and hope it figures it out.

Good:

Select 4-8 high-value passages, then answer.

8.2 Small Models Win If Context Is Good

Tiny models can answer well if the context is clean and relevant.

The answer model should not be responsible for research. It should only synthesize from selected sources.

8.3 Hybrid Retrieval Beats Single Retrieval

BM25 handles exact identifiers and technical terms.

Embeddings handle semantic/paraphrased matches.

Rerankers handle final precision.

The default pipeline should combine all three.

8.4 Runtime Crawling Must Be Bounded

Every request must have strict limits:

* Max search results.
* Max fetched pages.
* Max depth.
* Max bytes per page.
* Per-page timeout.
* Global timeout.
* Max chunks.
* Max tokens to answer model.

8.5 Provider Abstraction

ddgs should be the default search provider, but the architecture should allow:

* Exa
* Brave
* Tavily
* SerpAPI
* Google Custom Search
* Internal docs search
* User-uploaded files
* Company knowledge base

⸻

9. High-Level Architecture

Client / Agent
   |
   v
API Gateway / FastAPI
   |
   +--> /web_search
   |       |
   |       v
   |     ddgs provider
   |
   +--> /web_fetch
   |       |
   |       v
   |     Crawl4AI AsyncWebCrawler -> markdown
   |
   +--> /web_crawl
   |       |
   |       v
   |     Crawl4AI BestFirstCrawlingStrategy
   |
   +--> /web_answer
           |
           v
        ddgs search
           |
           v
        async fetch/extract
           |
           v
        passage chunking
           |
           v
        BM25 retrieval
           |
           v
        embedding retrieval
           |
           v
        hybrid candidate merge
           |
           v
        neural reranker
           |
           v
        context packer
           |
           v
        tiny answer model
           |
           v
        answer + citations

⸻

10. Core User Flows

10.1 Raw Search Flow

User or agent asks:

Search web for "open source Exa alternative"

System:

1. Calls ddgs.
2. Returns top search results.
3. Does not fetch pages.
4. Does not answer.

Output:

{
  "query": "open source Exa alternative",
  "provider": "ddgs",
  "results": [
    {
      "title": "SearXNG",
      "url": "https://...",
      "snippet": "..."
    }
  ]
}

⸻

10.2 Fetch Single URL Flow

User or agent asks:

Fetch this URL and return clean markdown.

System:

1. Downloads page with Crawl4AI AsyncWebCrawler.
2. Renders page in headless Chromium (handles JS-heavy sites).
3. Returns clean markdown via Crawl4AI markdown generator.
4. No fallback extractors in v0.

Output:

{
  "url": "https://...",
  "title": "Page title",
  "text": "...clean markdown...",
  "extractor": "crawl4ai",
  "fetched_at": "2026-06-09T..."
}

⸻

10.3 Query Answer Flow

User or agent asks:

Does TCL 50V6C support launch browser on boot?

System:

1. Searches ddgs.
2. Fetches top pages.
3. Extracts readable content.
4. Splits content into passages.
5. Runs BM25 retrieval.
6. Runs embedding retrieval.
7. Merges candidates.
8. Reranks top candidates.
9. Packs top passages into context.
10. Calls tiny answer model.
11. Returns answer with citations.

Output:

{
  "query": "Does TCL 50V6C support launch browser on boot?",
  "answer": "The sources do not clearly confirm native browser launch-on-boot support...",
  "citations": [
    {
      "source_id": 1,
      "url": "https://...",
      "title": "..."
    }
  ],
  "sources": [
    {
      "id": 1,
      "url": "https://...",
      "title": "...",
      "selected_passages": [...]
    }
  ]
}

⸻

10.4 Deep Crawl Flow

User or agent asks:

Crawl docs.example.com starting from the homepage and find pages about authentication.

System:

1. Accepts one or more seed URLs.
2. Runs Crawl4AI BestFirstCrawlingStrategy with bounded max_pages and max_depth.
3. Scores discovered links; optional query drives KeywordRelevanceScorer.
4. Returns crawled pages sorted by relevance score.

Output:

{
  "seed_urls": ["https://docs.example.com"],
  "query": "authentication",
  "pages": [
    {
      "url": "https://docs.example.com/auth",
      "title": "Authentication",
      "text": "...markdown...",
      "depth": 1,
      "score": 0.92,
      "seed_url": "https://docs.example.com"
    }
  ],
  "failed_urls": [],
  "timing_ms": { "crawl": 8200 }
}

⸻

11. API Surface

11.1 POST /web_search

Searches the web and returns search results only.

Request

{
  "query": "string",
  "max_results": 10,
  "provider": "ddgs",
  "region": "wt-wt",
  "safe_search": "moderate"
}

Response

{
  "query": "string",
  "provider": "ddgs",
  "results": [
    {
      "rank": 1,
      "title": "string",
      "url": "string",
      "snippet": "string",
      "source_provider": "ddgs"
    }
  ],
  "timing_ms": {
    "search": 430
  }
}

⸻

11.2 POST /web_fetch

Fetches and extracts content from a URL.

Request

{
  "url": "https://example.com/page",
  "mode": "fast",
  "include_links": true,
  "include_tables": true,
  "max_chars": 80000
}

Response

{
  "url": "https://example.com/page",
  "final_url": "https://example.com/page",
  "title": "Example Page",
  "text": "clean extracted markdown",
  "metadata": {
    "author": null,
    "published_at": null,
    "language": "en"
  },
  "extractor": "crawl4ai",
  "content_hash": "sha256...",
  "timing_ms": {
    "fetch": 320,
    "extract": 80
  }
}

⸻

11.3 POST /web_crawl

Crawls the most significant linked pages starting from one or more seed URLs.

Request

{
  "urls": ["https://docs.example.com"],
  "query": "authentication setup",
  "max_pages": 10,
  "max_depth": 2,
  "include_external": false
}

Response

{
  "seed_urls": ["https://docs.example.com"],
  "query": "authentication setup",
  "pages": [
    {
      "url": "https://docs.example.com/auth",
      "title": "Authentication",
      "text": "clean extracted markdown",
      "depth": 1,
      "score": 0.92,
      "seed_url": "https://docs.example.com"
    }
  ],
  "failed_urls": [],
  "timing_ms": {
    "crawl": 8200
  }
}

⸻

11.4 POST /content_select

Selects the best passages from already provided documents/pages.

Useful when the agent already has fetched pages and only wants the context selector.

Request

{
  "query": "string",
  "documents": [
    {
      "id": "doc_1",
      "url": "https://...",
      "title": "string",
      "text": "full extracted text"
    }
  ],
  "mode": "default",
  "token_budget": 3500
}

Response

{
  "query": "string",
  "selected_passages": [
    {
      "source_id": "doc_1",
      "url": "https://...",
      "title": "string",
      "passage_id": "doc_1#12",
      "text": "selected text",
      "score": 0.92,
      "selection_reason": "high reranker score"
    }
  ],
  "debug": {
    "total_passages": 180,
    "bm25_candidates": 50,
    "embedding_candidates": 50,
    "merged_candidates": 76,
    "reranked_candidates": 76,
    "final_passages": 6
  }
}

⸻

11.5 POST /web_answer

Full pipeline: search, fetch, select, answer.

Request

{
  "query": "string",
  "mode": "default",
  "search_provider": "ddgs",
  "max_search_results": 8,
  "max_pages": 8,
  "max_depth": 0,
  "answer_format": "text_with_citations",
  "return_sources": true,
  "return_debug": false
}

Response

{
  "query": "string",
  "answer": "string",
  "confidence": "medium",
  "abstained": false,
  "citations": [
    {
      "source_id": 1,
      "url": "https://...",
      "title": "string",
      "supports": "claim text or passage id"
    }
  ],
  "sources": [
    {
      "id": 1,
      "url": "https://...",
      "title": "string",
      "selected_passages": [
        {
          "passage_id": "1#3",
          "text": "string",
          "score": 0.91
        }
      ]
    }
  ],
  "timing_ms": {
    "search": 400,
    "fetch": 2200,
    "extract": 500,
    "select": 800,
    "answer": 900,
    "total": 4800
  }
}

⸻

12. Runtime Modes

12.1 Fast Mode

Optimized for high concurrency and low latency.

mode: fast
search:
  max_results: 5
fetch:
  max_pages: 5
  max_depth: 0
  per_page_timeout_ms: 5000
  global_timeout_ms: 10000
chunking:
  passage_tokens: 180
  overlap_tokens: 30
retrieval:
  bm25_top_k: 30
  embedding_top_k: 30
  merged_top_k: 40
rerank:
  model: Qwen3-Reranker-0.6B
  final_top_k: 3
answer:
  model: OCC-RAG-0.6B
  max_context_tokens: 2500
  max_output_tokens: 200

Expected use:

* Simple factual questions.
* Cheap agent browsing.
* High traffic.
* First-pass answers.

⸻

12.2 Default Mode

Balanced quality and speed.

mode: default
search:
  max_results: 8
fetch:
  max_pages: 8
  max_depth: 0
  per_page_timeout_ms: 8000
  global_timeout_ms: 25000
chunking:
  passage_tokens: 220
  overlap_tokens: 40
retrieval:
  bm25_top_k: 50
  embedding_top_k: 50
  merged_top_k: 80
rerank:
  model: Qwen3-Reranker-0.6B
  final_top_k: 6
answer:
  model: OCC-RAG-1.7B
  max_context_tokens: 3500
  max_output_tokens: 350

Expected use:

* Most agent questions.
* Technical documentation questions.
* Product research.
* Web QA with citations.

⸻

12.3 Quality Mode

Better evidence selection and synthesis, higher latency.

mode: quality
search:
  max_results: 12
fetch:
  max_pages: 12
  max_depth: 1
  per_page_timeout_ms: 10000
  global_timeout_ms: 45000
chunking:
  passage_tokens: 250
  overlap_tokens: 50
retrieval:
  bm25_top_k: 100
  embedding_top_k: 100
  merged_top_k: 150
rerank:
  model: Qwen3-Reranker-0.6B or Qwen3-Reranker-4B
  final_top_k: 8
answer:
  model: OCC-RAG-1.7B or SmolLM3-3B
  max_context_tokens: 6000
  max_output_tokens: 600

Expected use:

* Harder research.
* Multiple sources.
* Conflicting sources.
* Developer documentation.
* More nuanced answers.

⸻

13. Search Provider

13.1 Default Provider: ddgs

ddgs is used for search discovery.

Responsibilities:

* Convert query into search results.
* Return title, URL, snippet, and rank.
* Provide cheap/free search discovery.
* Fail gracefully if rate-limited.

ddgs is not responsible for:

* Fetching pages.
* Extracting content.
* Ranking passages.
* Answer generation.

13.2 Provider Interface

class SearchProvider:
    def search(
        self,
        query: str,
        max_results: int,
        region: str | None = None,
        safe_search: str | None = None,
    ) -> list[SearchResult]:
        ...

13.3 Future Providers

The system should support adding:

exa
brave
tavily
serpapi
google_custom_search
internal_docs
github_search

⸻

13.4 Proxy Support

The system should support optional HTTP/SOCKS proxies for all outbound traffic.

Configuration:

proxies:
  - http://user:pass@host:port
  - socks5://host:port

Applied to:

* ddgs search (DDGS proxy parameter).
* Crawl4AI fetch and deep crawl (BrowserConfig proxy settings).

When multiple proxies are configured, use round-robin rotation across requests.

Proxy credentials must be redacted in logs.

No proxy configured means direct connections.

⸻

14. Fetching and Extraction

14.1 Fetching

Use Crawl4AI AsyncWebCrawler with headless Chromium (Playwright).

Requirements:

* Follow redirects.
* Enforce per-page timeout.
* Enforce max response size.
* Set a reasonable user agent.
* Respect robots and domain-level policy where applicable.
* Cache extracted markdown by URL in SQLite.
* Apply optional proxy from configuration.
* SSRF guard: reject private/loopback/link-local IPs before fetch.
* HTTP(S) only.

Multi-URL fetch uses arun_many() for concurrent crawling.

PDF support is not required in v0.

14.2 Extraction

Primary (and only) extractor in v0:

Crawl4AI markdown generator (result.markdown.raw_markdown)

No fallback extractors in v0.

Crawl4AI internal cache is bypassed; Noxa SQLite cache stores results.

14.3 Deep Crawl

For /web_crawl, use Crawl4AI BestFirstCrawlingStrategy:

* Accept one or more seed URLs.
* Optional query drives KeywordRelevanceScorer for link prioritization.
* Bounded by max_pages (default 10) and max_depth (default 2).
* Same-domain by default (include_external: false).
* FilterChain with DomainFilter and ContentTypeFilter.

14.4 Extraction Output

Each extracted page should include:

{
  "url": "string",
  "final_url": "string",
  "title": "string",
  "text": "string",
  "markdown": "string",
  "language": "string",
  "published_at": "string|null",
  "content_hash": "string",
  "extractor": "crawl4ai"
}

⸻

15. Chunking

15.1 Chunking Goal

Convert extracted pages into passages small enough for retrieval but large enough to preserve meaning.

Default:

150-300 tokens per passage
30-50 token overlap

15.2 Chunk Structure

{
  "id": "source_1#passage_12",
  "source_id": "source_1",
  "url": "https://...",
  "title": "string",
  "text": "passage text",
  "start_char": 1200,
  "end_char": 2600,
  "token_count": 230,
  "source_rank": 3
}

15.3 Chunking Requirements

The chunker should:

1. Prefer paragraph boundaries.
2. Avoid splitting code blocks where possible.
3. Preserve headings.
4. Preserve tables if extracted cleanly.
5. Include title/header context when useful.
6. Track original source positions.

⸻

16. Hybrid Retrieval

The system uses three-stage retrieval:

BM25 retrieval
+
embedding retrieval
+
neural reranking

16.1 Why Hybrid Retrieval

BM25 is good for:

exact names
model IDs
error codes
API endpoints
product SKUs
version numbers
technical keywords

Embeddings are good for:

semantic matches
paraphrases
synonyms
mixed language
fuzzy intent

Rerankers are good for:

final query-passage relevance
discarding noisy matches
choosing best evidence

⸻

17. BM25 Layer

17.1 Purpose

BM25 is a cheap recall layer.

It quickly filters all passages down to a smaller candidate set.

Example:

300 passages -> BM25 top 50

17.2 BM25 Inputs

* User query.
* All extracted passages.

17.3 BM25 Output

A ranked list of candidate passage IDs and scores.

[
  {
    "passage_id": "source_1#12",
    "bm25_score": 12.4
  }
]

17.4 BM25 Enhancements

The system should support:

1. Query expansion.
2. Title boost.
3. URL boost.
4. Source rank boost.
5. Exact phrase boost.
6. Domain boost.
7. Recency boost later.

Example query expansion:

{
  "boot": ["startup", "restart", "power on", "autostart"],
  "browser": ["chrome", "webview", "web browser"],
  "launch": ["start", "open", "run"],
  "kiosk": ["lockdown", "single app mode", "fullscreen"]
}

⸻

18. Embedding Layer

18.1 Purpose

Embedding retrieval improves semantic recall.

It catches relevant passages that do not share exact keywords with the query.

Example:

Query:
  launch browser on boot
Relevant passage:
  automatically start Chrome after device restart

18.2 Default Embedding Models

Fast runtime default:

intfloat/multilingual-e5-small

Quality mode:

BAAI/bge-m3

Benchmark candidate:

Qwen/Qwen3-Embedding-0.6B

18.3 Recommended Initial Choice

Use:

multilingual-e5-small

Reasons:

* Small.
* CPU-friendly.
* Multilingual.
* Good enough for runtime.
* 384-dimensional vectors.
* Fast to batch.

18.4 Embedding Prefixes

For E5-style models:

query: {query}
passage: {passage}

18.5 Embedding Output

[
  {
    "passage_id": "source_2#4",
    "embedding_score": 0.78
  }
]

18.6 Embedding Cache

Store passage and query embeddings in SQLite with vector support (sqlite-vec extension).

Cache key:

sha256(model_name + passage_text)

Query embedding cache key:

sha256(model_name + query)

Storage:

* vec0 virtual table in the same noxa.db file.
* 384-dimensional vectors for multilingual-e5-small.
* KNN retrieval via vec0 MATCH for embedding top-K within a request's passage set.
* TTL: 30 days (configurable).

⸻

19. Candidate Merge

19.1 Input

* BM25 top K.
* Embedding top K.

19.2 Output

Merged, deduplicated candidate list.

19.3 Merge Strategy

Use rank-based reciprocal scoring instead of raw scores.

Example:

combined_score =
  bm25_weight * 1 / bm25_rank
  +
  embedding_weight * 1 / embedding_rank

Default weights:

bm25_weight: 0.45
embedding_weight: 0.55

19.4 Candidate Limits

Default mode:

bm25_top_k: 50
embedding_top_k: 50
merged_top_k: 80

⸻

20. Reranking Layer

20.1 Purpose

The reranker chooses the best final passages to send to the LLM.

This is the most important precision layer.

20.2 Default Reranker

Qwen/Qwen3-Reranker-0.6B

Alternative:

BAAI/bge-reranker-v2-m3

Quality mode:

Qwen/Qwen3-Reranker-4B

20.3 Reranker Input

query + passage

20.4 Reranker Output

{
  "passage_id": "source_1#12",
  "rerank_score": 0.94
}

20.5 Final Selection Rules

The selector should:

1. Pick highest-scoring passages.
2. Limit max passages per URL.
3. Avoid near-duplicates.
4. Preserve source diversity.
5. Prefer official docs when available.
6. Prefer recent sources when dates are relevant.
7. Keep within answer context token budget.

Default:

final_top_k: 6
max_per_url: 2
max_context_tokens: 3500

⸻

21. Context Packing

21.1 Purpose

Prepare the final prompt context for the answer model.

21.2 Input

Selected reranked passages.

21.3 Output

Formatted sources:

[1] Page title
URL: https://...
Passage:
...
[2] Page title
URL: https://...
Passage:
...

21.4 Packing Rules

The packer should:

1. Respect token budget.
2. Include source title and URL.
3. Include only selected passages.
4. Keep passages short.
5. Avoid repeated text.
6. Include enough neighboring context when needed.
7. Preserve citation IDs.

21.5 Retrieval Window Strategy

Retrieve small passages, then optionally expand around selected passages.

Example:

retrieval passage:
  200 tokens
generation window:
  selected passage ± neighboring paragraph
  400-700 tokens

This improves answer quality without sending full pages.

⸻

22. Answer Model

22.1 Purpose

The answer model produces the final response from selected evidence.

It should not search, browse, or infer beyond the provided sources.

22.2 Default Models

Fast mode:

OCC-RAG-0.6B

Default mode:

OCC-RAG-1.7B

Alternative generic model:

Qwen2.5-1.5B-Instruct
Qwen3-1.7B

Small quality mode:

SmolLM3-3B

22.3 Why OCC-RAG

OCC-RAG is specialized for:

* Context-grounded QA.
* Citations.
* Abstention.
* Faithful answer generation.

This matches the product exactly.

22.4 Prompt Template

Answer the question using only the provided sources.
Rules:
- Use only facts from the sources.
- Cite claims with source numbers like [1], [2].
- If the sources do not contain enough information, say that the sources are insufficient.
- Do not use outside knowledge.
- Keep the answer concise.
Question:
{query}
Sources:
{formatted_sources}
Answer:

22.5 Output Requirements

The answer should include:

1. Direct answer.
2. Citations.
3. Abstention if sources are insufficient.
4. No unsupported claims.
5. No hidden reasoning.
6. No raw chain-of-thought.

⸻

23. Deployment Options

23.1 Simple In-Process Python

Best for v0.

Components:

FastAPI
ddgs
crawl4ai
rank-bm25
sentence-transformers
transformers
torch

One worker process loads one model.

Concurrency strategy:

one active generation per worker
multiple worker processes per machine

Example:

16 CPU cores
  -> 4 workers
  -> 4 threads per model

23.2 CPU Deployment

Recommended for:

* OCC-RAG-0.6B.
* OCC-RAG-1.7B.
* multilingual-e5-small embeddings.
* Low/medium traffic.
* Cheap horizontal scaling.

CPU architecture:

API workers
  -> async search/fetch
  -> CPU embedding
  -> CPU reranker or small GPU reranker
  -> CPU answer model

23.3 GPU Deployment

Recommended for:

* Higher concurrency.
* Reranker acceleration.
* OCC-RAG-1.7B default answering.
* Quality mode.

Example GPU:

RTX 3060 12GB

Estimated target caps:

OCC-RAG-0.6B:
  context: 4096
  active_generation_concurrency: 20-40
OCC-RAG-1.7B:
  context: 4096
  active_generation_concurrency: 8-16

These should be validated with benchmarks.

23.4 Sidecar Model Server

Later, model inference can move to:

llama-server
vLLM
SGLang
ONNX Runtime server

Initial v0 can stay in-process for simplicity.

⸻

24. Caching

24.1 Cache Backend

Use SQLite (single noxa.db file) for all caching in v0.

* Key-value TTL cache table for search results, fetched markdown, and answers.
* sqlite-vec vec0 table for passage and query embeddings with KNN search.

24.2 Cache Types

The system should cache:

1. Search results.
2. Extracted markdown (from Crawl4AI).
3. Passage chunks (optional).
4. Passage embeddings (sqlite-vec).
5. Query embeddings (sqlite-vec).
6. Reranker scores (optional).
7. Final answers.

24.3 Cache Keys

Search cache:

sha256(provider + query + region + max_results)

Fetch cache:

sha256(url)

Extraction cache:

sha256(final_url + content_hash + extractor_version)

Embedding cache (sqlite-vec):

sha256(model_name + passage_text)

Answer cache:

sha256(query + selected_passage_ids + model_name + prompt_version)

24.4 Cache TTLs

Suggested defaults:

search_results:
  ttl: 1h
extracted_content:
  ttl: 24h
embeddings:
  ttl: 30d
reranker_scores:
  ttl: 7d
answers:
  ttl: 1h

For highly dynamic topics, TTL should be shorter.

⸻

25. Observability

25.1 Required Logs

Each request should log:

{
  "request_id": "string",
  "query": "string",
  "mode": "default",
  "search_provider": "ddgs",
  "num_search_results": 8,
  "num_pages_fetched": 7,
  "num_pages_failed": 1,
  "num_passages": 160,
  "num_bm25_candidates": 50,
  "num_embedding_candidates": 50,
  "num_merged_candidates": 74,
  "num_final_passages": 6,
  "answer_model": "OCC-RAG-1.7B",
  "latency_ms": 4800,
  "abstained": false
}

25.2 Metrics

Track:

* Request count.
* Error rate.
* Timeout rate.
* Search latency.
* Fetch latency.
* Extraction latency.
* Embedding latency.
* Reranking latency.
* Answer model latency.
* Tokens in.
* Tokens out.
* Cache hit rate.
* Abstention rate.
* Citation count.
* Sources per answer.
* Pages fetched per answer.

25.3 Debug Mode

Debug mode should return:

* Search results.
* Failed URLs.
* Extractor used per page.
* BM25 candidate scores.
* Embedding candidate scores.
* Reranker scores.
* Final packed context.
* Answer prompt token count.

Debug mode should be disabled by default.

⸻

26. Evaluation

26.1 Evaluation Set

Create an internal eval set with 100-300 real queries.

Categories:

1. Product specs.
2. Software documentation.
3. Pricing questions.
4. GitHub/project questions.
5. Polish queries.
6. Mixed Polish-English queries.
7. Time-sensitive queries.
8. Questions where sources are insufficient.
9. Questions requiring exact identifiers.
10. Questions requiring semantic paraphrase.

26.2 Metrics

Retrieval Metrics

* Recall@K: Did selected passages contain the answer?
* MRR: How high was the first relevant passage?
* Source diversity.
* Official-source selection rate.
* Duplicate passage rate.

Answer Metrics

* Faithfulness.
* Citation correctness.
* Abstention correctness.
* Answer completeness.
* Unsupported claim rate.
* Latency.
* Cost per answer.
* Tokens per answer.

26.3 Human Review Labels

Each eval item should include:

{
  "query": "string",
  "expected_answer": "string",
  "must_include_sources": ["url"],
  "acceptable_sources": ["url"],
  "should_abstain": false,
  "notes": "string"
}

⸻

27. Fine-Tuning Strategy

27.1 Fine-Tuning Is Not Required for v0

The initial product should use off-the-shelf models.

Fine-tuning should start only after there are logs showing clear failures.

27.2 First Fine-Tune Target: Reranker

Fine-tune the content selector before the answer model.

Reason:

If the correct evidence is selected, tiny answer models work well.

Training examples:

{
  "query": "Does this TV launch browser on boot?",
  "positive": "Passage that answers autostart/browser boot behavior.",
  "negative": "Passage that only mentions Google TV specs."
}

27.3 Second Fine-Tune Target: Answer Model

Fine-tune answer model only if selected evidence is correct but the model:

* Ignores citations.
* Hallucinates.
* Fails to abstain.
* Produces wrong JSON.
* Writes too much.
* Mishandles Polish/English answers.

⸻

28. Functional Requirements

28.1 Search

The system must:

* Support ddgs search.
* Return title, URL, snippet, rank.
* Limit max results.
* Handle search provider errors.
* Support provider abstraction.

28.2 Fetch

The system must:

* Fetch pages asynchronously via Crawl4AI.
* Follow redirects.
* Enforce timeouts.
* Enforce max bytes.
* Return structured errors for failed pages.
* Support optional proxy configuration.

28.3 Extract

The system must:

* Extract readable main content via Crawl4AI markdown generator.
* Return markdown/plain text.
* Track extractor used (crawl4ai).
* No fallback extractors in v0.

28.4 Crawl

The system must:

* Accept one or more seed URLs for /web_crawl.
* Crawl most significant linked pages with bounded depth and page count.
* Score links by optional query relevance.
* Return pages with url, title, text, depth, score.

28.5 Chunk

The system must:

* Split extracted content into passages.
* Preserve source metadata.
* Track passage positions.
* Support configurable chunk size.

28.6 Retrieve

The system must:

* Run BM25 over candidate passages.
* Run embedding retrieval over candidate passages.
* Merge and dedupe candidates.
* Support hybrid weights.

28.7 Rerank

The system must:

* Score query-passage pairs.
* Return top relevant passages.
* Limit per-source passage count.
* Avoid duplicate context.

28.8 Answer

The system must:

* Build a source-grounded prompt.
* Call a tiny answer model.
* Return citations.
* Abstain if evidence is insufficient.
* Return source list.

28.9 Modes

The system must support:

fast
default
quality

Each mode must have separate limits and model choices.

⸻

29. Non-Functional Requirements

29.1 Latency Targets

Fast mode:

p50: < 3s
p95: < 8s

Default mode:

p50: < 6s
p95: < 15s

Quality mode:

p50: < 15s
p95: < 45s

These are initial targets and should be validated.

29.2 Availability

Target:

v0: 99%
v1: 99.5%

29.3 Scalability

The system should scale horizontally by:

* Adding API workers.
* Adding fetch workers.
* Adding model workers.
* Adding GPU/CPU inference replicas.
* Sharing cache across workers.

29.4 Cost

The system should optimize for:

* Free discovery via ddgs.
* CPU-based embedding in fast mode.
* Tiny answer models.
* Caching.
* Optional paid fallback providers only when needed.

29.5 Security

The system must:

* Block internal IP ranges by default.
* Prevent SSRF.
* Limit redirects.
* Limit response size.
* Limit protocols to HTTP/HTTPS.
* Sanitize extracted text.
* Avoid executing page JavaScript unless browser fallback is explicitly enabled.
* Log external fetches.

29.6 Abuse Prevention

The system should:

* Rate-limit by API key.
* Limit max pages per request.
* Limit global timeout.
* Limit browser fallback usage.
* Limit repeated failed-domain fetches.
* Support domain denylist.

⸻

30. Error Handling

30.1 Search Failure

If ddgs fails:

{
  "error": {
    "code": "SEARCH_PROVIDER_FAILED",
    "message": "Search provider failed or was rate-limited.",
    "retryable": true
  }
}

30.2 Fetch Failure

If some pages fail, continue with successful pages.

If all pages fail:

{
  "error": {
    "code": "NO_FETCHABLE_SOURCES",
    "message": "Search returned results, but no pages could be fetched."
  }
}

30.3 Insufficient Evidence

If sources do not answer the query:

{
  "answer": "I do not have enough information from the fetched sources to answer this.",
  "abstained": true,
  "sources": [...]
}

30.4 Model Failure

If local model fails:

* Retry once.
* Fall back to smaller/faster model if configured.
* Return structured error if no model works.

⸻

31. Data Models

31.1 SearchResult

class SearchResult:
    rank: int
    title: str | None
    url: str
    snippet: str | None
    provider: str

31.2 ExtractedPage

class ExtractedPage:
    source_id: str
    url: str
    final_url: str
    title: str | None
    text: str
    markdown: str | None
    language: str | None
    published_at: str | None
    content_hash: str
    extractor: str

31.3 Passage

class Passage:
    passage_id: str
    source_id: str
    url: str
    title: str | None
    text: str
    start_char: int
    end_char: int
    token_count: int
    source_rank: int

31.4 ScoredPassage

class ScoredPassage:
    passage: Passage
    bm25_score: float | None
    embedding_score: float | None
    merged_score: float | None
    rerank_score: float | None

31.5 WebAnswer

class WebAnswer:
    query: str
    answer: str
    abstained: bool
    confidence: str
    citations: list[Citation]
    sources: list[AnswerSource]
    timing_ms: dict

⸻

32. Initial Tech Stack

32.1 Python Libraries

Search:

ddgs

Fetch / Extract / Crawl:

crawl4ai

Chunking:

custom paragraph chunker

BM25:

rank-bm25

Embeddings:

sentence-transformers

Reranking:

transformers

Answer model:

transformers
torch

API:

FastAPI
Uvicorn
Pydantic
pydantic-settings

Cache / Storage:

SQLite
sqlite-vec

Storage later:

Postgres
S3-compatible storage

⸻

33. Suggested v0 Implementation Plan

Milestone 1: Search + Fetch + Extract

Build:

* /web_search
* /web_fetch
* ddgs provider with optional proxy.
* Crawl4AI AsyncWebCrawler fetcher.
* SQLite TTL cache.
* Proxy configuration.

Success criteria:

* Can search and fetch top pages.
* Can extract readable markdown from common docs/blog pages.
* Returns structured errors.

⸻

Milestone 1b: Deep Crawl

Build:

* /web_crawl
* BestFirstCrawlingStrategy with KeywordRelevanceScorer.
* Bounded max_pages and max_depth.

Success criteria:

* Given seed URLs, returns most relevant linked pages.
* Respects depth and page limits.

⸻

Milestone 2: Chunking + BM25

Build:

* Passage chunker.
* BM25 prefilter.
* Basic source ranking.
* Debug output.

Success criteria:

* Given extracted pages, system returns top keyword-relevant passages.
* Works for exact model names, product names, API names.

⸻

Milestone 3: Embedding Retrieval

Build:

* Runtime embedding with multilingual-e5-small.
* sqlite-vec embedding cache and KNN retrieval.
* Embedding top K.
* Hybrid merge.

Success criteria:

* Catches paraphrased passages missed by BM25.
* Reasonable latency for 100-200 passages.

⸻

Milestone 4: Reranker

Build:

* Qwen3-Reranker-0.6B integration.
* Candidate reranking.
* Diversity selection.
* Context packing.

Success criteria:

* Top 4-8 selected passages usually contain answer.
* Duplicate passages reduced.
* Source diversity improved.

⸻

Milestone 5: Answer Model

Build:

* OCC-RAG-0.6B fast mode.
* OCC-RAG-1.7B default mode.
* Prompt template.
* Citation formatting.
* Abstention behavior.

Success criteria:

* Produces useful answers from selected sources.
* Includes citations.
* Abstains when evidence is insufficient.

⸻

Milestone 6: Full /web_answer

Build:

* End-to-end orchestration.
* Runtime modes.
* Timing metrics.
* Debug mode.
* Config file.

Success criteria:

* Agent can call one endpoint and receive answer + citations.
* Default mode works within latency target on common queries.

⸻

Milestone 7: Evaluation Harness

Build:

* Eval dataset format.
* Automated test runner.
* Human-review UI or JSON output.
* Metrics for retrieval and answer quality.

Success criteria:

* Can compare BM25-only vs hybrid vs hybrid+reranker.
* Can compare answer models.
* Can detect regressions.

⸻

34. Suggested Initial Configuration

service:
  default_mode: default
  max_concurrent_requests: 100
proxies: []
search:
  provider: ddgs
  default_max_results: 8
fetch:
  max_pages: 8
  per_page_timeout_ms: 8000
  global_timeout_ms: 25000
  max_bytes_per_page: 3000000
  user_agent: "Noxa/0.1"
crawl:
  max_pages: 10
  max_depth: 2
  include_external: false
extract:
  provider: crawl4ai
chunk:
  passage_tokens: 220
  overlap_tokens: 40
bm25:
  enabled: true
  top_k: 50
  title_boost: 0.5
  url_boost: 0.3
  source_rank_boost: true
embedding:
  enabled: true
  model: intfloat/multilingual-e5-small
  dimensions: 384
  top_k: 50
  batch_size: 64
  normalize: true
hybrid:
  bm25_weight: 0.45
  embedding_weight: 0.55
  max_candidates: 80
rerank:
  enabled: true
  model: Qwen/Qwen3-Reranker-0.6B
  final_top_k: 6
  max_per_url: 2
answer:
  model_fast: occ-ai/OCC-RAG-0.6B
  model_default: occ-ai/OCC-RAG-1.7B
  max_context_tokens: 3500
  max_output_tokens: 350
  temperature: 0
cache:
  sqlite_path: noxa.db
  search_ttl_seconds: 3600
  fetch_ttl_seconds: 86400
  embedding_ttl_seconds: 2592000

⸻

35. Agent Tool Definitions

35.1 web_search

{
  "name": "web_search",
  "description": "Search the web and return URLs/snippets. Does not fetch pages.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "max_results": { "type": "integer", "default": 8 }
    },
    "required": ["query"]
  }
}

35.2 web_fetch

{
  "name": "web_fetch",
  "description": "Fetch one URL and return cleaned readable markdown via Crawl4AI.",
  "input_schema": {
    "type": "object",
    "properties": {
      "url": { "type": "string" },
      "max_chars": { "type": "integer", "default": 80000 }
    },
    "required": ["url"]
  }
}

35.3 web_crawl

{
  "name": "web_crawl",
  "description": "Deep crawl from seed URLs and return the most significant linked pages.",
  "input_schema": {
    "type": "object",
    "properties": {
      "urls": {
        "type": "array",
        "items": { "type": "string" }
      },
      "query": { "type": "string" },
      "max_pages": { "type": "integer", "default": 10 },
      "max_depth": { "type": "integer", "default": 2 },
      "include_external": { "type": "boolean", "default": false }
    },
    "required": ["urls"]
  }
}

35.4 web_answer

{
  "name": "web_answer",
  "description": "Search the web, select the best evidence, and answer with citations.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "mode": {
        "type": "string",
        "enum": ["fast", "default", "quality"],
        "default": "default"
      }
    },
    "required": ["query"]
  }
}

35.5 content_select

{
  "name": "content_select",
  "description": "Given documents and a query, select the best passages to send to an LLM.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "documents": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "url": { "type": "string" },
            "title": { "type": "string" },
            "text": { "type": "string" }
          },
          "required": ["text"]
        }
      },
      "token_budget": { "type": "integer", "default": 3500 }
    },
    "required": ["query", "documents"]
  }
}

⸻

36. Risks

36.1 ddgs Reliability

Risk:

* ddgs may be rate-limited or inconsistent.

Mitigation:

* Add provider abstraction.
* Add paid fallback provider.
* Cache search results.
* Retry with backoff.
* Expose clear provider errors.

36.2 Extraction Quality

Risk:

* Some pages extract poorly.

Mitigation:

* Multiple extractors.
* Browser fallback.
* Extraction quality score.
* Skip low-quality pages.

36.3 Hallucinations

Risk:

* Tiny answer models may hallucinate.

Mitigation:

* OCC-RAG-style grounded model.
* Strict prompt.
* Short context.
* Citation requirement.
* Abstention.
* Eval set.

36.4 Slow Runtime

Risk:

* Fetching pages dominates latency.

Mitigation:

* Async fetch.
* Cache.
* Small max pages.
* Per-page timeout.
* Fast mode.
* No browser by default.

36.5 Poor Retrieval

Risk:

* BM25 misses semantic matches.
* Embeddings miss exact IDs.
* Reranker selects wrong passages.

Mitigation:

* Hybrid retrieval.
* Query expansion.
* Reranker.
* Eval harness.
* Later reranker fine-tuning.

36.6 Cost Creep

Risk:

* Browser fallback, large models, or paid providers increase cost.

Mitigation:

* Strict budgets.
* Mode-based limits.
* Default to local models.
* Paid fallback opt-in only.

⸻

37. Open Questions

1. Crawl4AI is the v0 fetch/crawl engine (resolved: yes, in v0).
2. Should PDF extraction be included in v0?
3. Should answer model run in-process or behind sidecar server?
4. Should reranker run on CPU, GPU, or separate service?
5. What is the minimum acceptable answer quality for fast mode?
6. What latency target is acceptable for default mode?
7. Should the service store source logs permanently?
8. Should user/tenant-specific caches be isolated?
9. Should there be a paid-provider fallback in v0?
10. Should the service support internal/private docs in the same pipeline?

⸻

38. Recommended v0 Scope

Build only:

/web_search
/web_fetch
/web_crawl
/content_select
/web_answer

Use:

ddgs
crawl4ai
rank-bm25
multilingual-e5-small
sqlite-vec
Qwen3-Reranker-0.6B
OCC-RAG-0.6B / OCC-RAG-1.7B
FastAPI
SQLite cache

Do not build yet:

extractor fallbacks (trafilatura/selectolax)
PDF extraction
fine-tuning
full web index
unbounded deep crawl
paid provider fallback
complex UI

⸻

39. Success Criteria

The v0 is successful if:

1. It can answer common web/documentation questions with citations.
2. It returns useful answers in default mode under 15 seconds p95.
3. It can run locally/self-hosted.
4. It avoids sending full pages to the answer model.
5. It keeps answer context under 4k tokens by default.
6. It works with tiny answer models.
7. It exposes clean tool APIs for agents.
8. It can be evaluated and improved systematically.
9. It can gracefully abstain when evidence is insufficient.
10. It is cheaper than relying entirely on hosted search-answer APIs.

⸻

40. Final Recommended Architecture

Agent
  |
  v
/web_answer
  |
  v
DDGS search
  |
  v
Crawl4AI fetch (markdown)
  |
  v
Passage chunking
  |
  +--> BM25 top 50
  |
  +--> multilingual-e5-small embedding top 50 (sqlite-vec KNN)
  |
  v
Hybrid merge top 80
  |
  v
Qwen3-Reranker-0.6B top 6
  |
  v
Context packer, max ~3500 tokens
  |
  v
OCC-RAG-1.7B answer model
  |
  v
Answer + citations + selected sources

The key product bet:

A small local answer model becomes good enough when the evidence selector is excellent.

Therefore, the most important component is not the final LLM.

The most important component is:

hybrid retrieval + reranking + context packing