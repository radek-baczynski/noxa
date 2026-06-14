# Noxa

Self-hosted web search and query answering for AI agents — an open-source Exa alternative.

## Setup

```bash
uv sync
uv sync --extra ml       # llama-cpp-python for answer, embed, and rerank
crawl4ai-setup           # installs Playwright Chromium
```

**ml**: `llama-cpp-python` + `huggingface-hub` — one process, no sidecars. Answer, embedding (`create_embedding`), and reranking all run through llama.cpp.

**NVIDIA GPU**: rebuild llama.cpp with CUDA support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --force-reinstall --no-cache-dir llama-cpp-python
```

Models download to `.noxa_models/` (GGUF) on first boot.

Copy the example env and set your HuggingFace token (models download at startup):

```bash
cp .env.example .env
# edit HF_TOKEN=...
```

Boot fails if models cannot be downloaded or loaded (unless `NOXA_PRELOAD_MODELS=false`).

## Run

```bash
uv run noxa
# or
uv run uvicorn noxa.app:app --reload
```

## Endpoints

- `POST /web_search` — ddgs search
- `POST /web_fetch` — fetch single URL as markdown (Crawl4AI)
- `POST /web_crawl` — bounded deep crawl from seed URLs
- `POST /content_select` — hybrid retrieval + reranking over documents
- `POST /web_answer` — full search → fetch → answer pipeline

## Configuration

Put settings in `.env` (see `.env.example`). Most vars use the `NOXA_` prefix; `HF_TOKEN` is also read for model downloads.

### Presets

Copy one full block into `.env` to test. Replace `hf_...` with your token (or use `hf auth login`).

All models are Hugging Face **GGUF repos**. Noxa picks the `Q4_K_M` file from each repo. Leave `NOXA_EMBED_MODEL` / `NOXA_RERANK_MODEL` empty for built-in defaults (`nomic-embed-text-v1.5`, `Qwen3-Reranker-0.6B`).

---

### Runtime presets (where to run)

#### Mac + llama.cpp (recommended on Apple Silicon)

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Auto-detect (backends chosen per platform)

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=
NOXA_ANSWER_MODEL_DEFAULT=
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

Empty answer models → built-in defaults (`unsloth/Qwen3-0.6B-GGUF`, `unsloth/Qwen3-1.7B-GGUF`).

#### Cloud CPU (headless Linux, llama.cpp on CPU threads)

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=cloud-cpu
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Cloud GPU (NVIDIA; CUDA llama.cpp build)

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=cloud-gpu
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Search/fetch only (skip model download at boot)

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=false
```

---

### Model presets (what to run — copy to test)

All use `NOXA_RUNTIME_PROFILE=mac-local` + `llama_cpp` unless noted. Each repo was checked for a `Q4_K_M` `.gguf` file via the Hugging Face Hub API. Download counts from Hugging Face Hub (June 2026).

| Family | Preset | Fast model | Default model |
|--------|--------|------------|---------------|
| Qwen3 | balanced (default) | `unsloth/Qwen3-0.6B-GGUF` | `unsloth/Qwen3-1.7B-GGUF` |
| Qwen3 | 4B quality | `unsloth/Qwen3-0.6B-GGUF` | `unsloth/Qwen3-4B-Instruct-2507-GGUF` |
| Qwen3.5 | latest | `unsloth/Qwen3.5-0.8B-GGUF` | `unsloth/Qwen3.5-4B-GGUF` |
| Qwen2.5 | classic | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `Qwen/Qwen2.5-3B-Instruct-GGUF` |
| Phi | 4-mini | `unsloth/Phi-4-mini-instruct-GGUF` | `MaziyarPanahi/Phi-4-mini-instruct-GGUF` |
| Gemma 4 | efficient | `unsloth/gemma-4-E2B-it-GGUF` | `unsloth/gemma-4-E4B-it-GGUF` |
| Gemma 3 | compact | `MaziyarPanahi/gemma-3-1b-it-GGUF` | `unsloth/gemma-3-4b-it-GGUF` |
| Gemma 2 | tiny | `bartowski/gemma-2-2b-it-GGUF` | `bartowski/gemma-2-2b-it-GGUF` |
| SmolLM2 | ultra-fast | `unsloth/SmolLM2-135M-Instruct-GGUF` | `bartowski/SmolLM2-1.7B-Instruct-GGUF` |
| SmolLM3 | 3B | `unsloth/SmolLM3-3B-GGUF` | `unsloth/SmolLM3-3B-GGUF` |
| SmolLM3 | 3B 128K | `unsloth/SmolLM3-3B-GGUF` | `unsloth/SmolLM3-3B-128K-GGUF` |
| SmolLM3 | 2 fast + 3 default | `unsloth/SmolLM2-135M-Instruct-GGUF` | `unsloth/SmolLM3-3B-GGUF` |
| Granite 4.1 | 3B (RAG) | `unsloth/granite-4.1-3b-GGUF` | `unsloth/granite-4.1-3b-GGUF` |
| Granite 4.1 | 3B + 8B | `unsloth/granite-4.1-3b-GGUF` | `unsloth/granite-4.1-8b-GGUF` |
| Granite 4.0 | 350m + 4.1 3B | `unsloth/granite-4.0-350m-GGUF` | `unsloth/granite-4.1-3b-GGUF` |
| Granite 3.3 | 2B + 8B | `unsloth/granite-3.3-2b-instruct-GGUF` | `ibm-granite/granite-3.3-8b-instruct-GGUF` |
| Liquid LFM2.5 | 1.2B instruct | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` |
| Liquid LFM2.5 | 350m + 1.2B | `LiquidAI/LFM2.5-350M-GGUF` | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` |
| Liquid LFM2.5 | 1.2B + 8B MoE | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LiquidAI/LFM2.5-8B-A1B-GGUF` |
| Liquid LFM2.5 | thinking | `LiquidAI/LFM2.5-350M-GGUF` | `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` |
| Llama 3.2 | 1B + 3B | `unsloth/Llama-3.2-1B-Instruct-GGUF` | `unsloth/Llama-3.2-3B-Instruct-GGUF` |
| Llama 3.1 | 8B quality | `unsloth/Llama-3.2-1B-Instruct-GGUF` | `MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF` |
| Mistral | 7B | `unsloth/Llama-3.2-1B-Instruct-GGUF` | `MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF` |
| DeepSeek | R1 distill 7B | `unsloth/Qwen3-0.6B-GGUF` | `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` |

---

#### Qwen3 balanced (default)

Fast `unsloth/Qwen3-0.6B-GGUF` (~79k dl) + default `unsloth/Qwen3-1.7B-GGUF` (~28k dl). Auto embed/rerank.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Qwen3 + multilingual retrieval

Same answer models + explicit retrieval pair: `nomic-ai/nomic-embed-text-v1.5-GGUF` + `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp`.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
NOXA_RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Qwen3 4B quality

Fast `unsloth/Qwen3-0.6B-GGUF` + default `unsloth/Qwen3-4B-Instruct-2507-GGUF` (~64k dl). Use API `mode: quality` to exercise the 4B model. Auto embed/rerank.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-4B-Instruct-2507-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Qwen3 MaziyarPanahi 1.7B default

Fast `unsloth/Qwen3-0.6B-GGUF` + default `MaziyarPanahi/Qwen3-1.7B-GGUF` (~289k dl, most-downloaded 1.7B quant). Auto embed/rerank.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=MaziyarPanahi/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Qwen3.5 latest

Newer Qwen line. Fast `unsloth/Qwen3.5-0.8B-GGUF` (~315k dl) + default `unsloth/Qwen3.5-4B-GGUF` (~721k dl). Use `mode: quality` for the 4B model.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3.5-0.8B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3.5-4B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Qwen2.5 classic

Stable prior-gen Qwen. Fast `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (~254k dl) + default `Qwen/Qwen2.5-3B-Instruct-GGUF` (~262k dl). Good baseline vs Qwen3.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=Qwen/Qwen2.5-1.5B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=Qwen/Qwen2.5-3B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Phi-4 mini (Microsoft)

Fast `unsloth/Phi-4-mini-instruct-GGUF` (~60k dl) + default `MaziyarPanahi/Phi-4-mini-instruct-GGUF` (~175k dl). Strong small instruct model outside the Qwen/Llama families.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Phi-4-mini-instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=MaziyarPanahi/Phi-4-mini-instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Gemma 4 efficient (Google)

Fast `unsloth/gemma-4-E2B-it-GGUF` (~1.0M dl) + default `unsloth/gemma-4-E4B-it-GGUF` (~958k dl). New Gemma 4 line; use `mode: quality` for E4B.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/gemma-4-E2B-it-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/gemma-4-E4B-it-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Gemma 3 compact (Google)

Fast `MaziyarPanahi/gemma-3-1b-it-GGUF` (~170k dl) + default `unsloth/gemma-3-4b-it-GGUF` (~44k dl).

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=MaziyarPanahi/gemma-3-1b-it-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/gemma-3-4b-it-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Gemma 2 tiny (2B)

Both modes use `bartowski/gemma-2-2b-it-GGUF` (~327k dl). Smallest Gemma preset; good for latency experiments.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=bartowski/gemma-2-2b-it-GGUF
NOXA_ANSWER_MODEL_DEFAULT=bartowski/gemma-2-2b-it-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### SmolLM2 ultra-fast (Hugging Face)

Fast `unsloth/SmolLM2-135M-Instruct-GGUF` (~62k dl) + default `bartowski/SmolLM2-1.7B-Instruct-GGUF` (~49k dl). Fastest answer preset; useful for smoke tests.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/SmolLM2-135M-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=bartowski/SmolLM2-1.7B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### SmolLM3 3B (Hugging Face)

SmolLM3 is **3B only** (multilingual, dual-mode reasoning). Both modes use `unsloth/SmolLM3-3B-GGUF` (~6k dl). Base model: `HuggingFaceTB/SmolLM3-3B` (~519k dl).

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/SmolLM3-3B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/SmolLM3-3B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### SmolLM3 3B 128K (long context)

Fast `unsloth/SmolLM3-3B-GGUF` + default `unsloth/SmolLM3-3B-128K-GGUF` (~3k dl). Use `mode: quality` for the 128K variant when you need longer context.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/SmolLM3-3B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/SmolLM3-3B-128K-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### SmolLM2 fast + SmolLM3 default

Fast `unsloth/SmolLM2-135M-Instruct-GGUF` (~62k dl) + default `unsloth/SmolLM3-3B-GGUF` (~6k dl). Smallest fast path with SmolLM3 quality on `default` / `quality` modes.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/SmolLM2-135M-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/SmolLM3-3B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Granite 4.1 3B (IBM, RAG-focused)

Latest IBM Granite instruct line; **explicitly tuned for RAG**. Both modes use `unsloth/granite-4.1-3b-GGUF` (~10k dl). Base: `ibm-granite/granite-4.1-3b` (~231k dl). Multilingual, Apache 2.0.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/granite-4.1-3b-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/granite-4.1-3b-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Granite 4.1 3B + 8B

Fast `unsloth/granite-4.1-3b-GGUF` (~10k dl) + default `unsloth/granite-4.1-8b-GGUF` (~16k dl). Use `mode: quality` for the 8B model.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/granite-4.1-3b-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/granite-4.1-8b-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Granite 4.0 350m + 4.1 3B (nano fast)

Fast `unsloth/granite-4.0-350m-GGUF` (~4k dl) + default `unsloth/granite-4.1-3b-GGUF` (~10k dl). Smallest Granite preset; 350M nano instruct for `fast` mode.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/granite-4.0-350m-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/granite-4.1-3b-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Granite 3.3 2B + 8B (prior gen)

Fast `unsloth/granite-3.3-2b-instruct-GGUF` (~723 dl) + default `ibm-granite/granite-3.3-8b-instruct-GGUF` (~937 dl). Previous Granite 3.3 instruct line.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/granite-3.3-2b-instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=ibm-granite/granite-3.3-8b-instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Liquid LFM2.5 1.2B Instruct

[Liquid AI](https://huggingface.co/LiquidAI) **LFM2.5** edge hybrid models. Both modes use `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` (~51k dl). Base: `LiquidAI/LFM2.5-1.2B-Instruct` (~207k dl). Multilingual, runs under ~1GB RAM at Q4.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Liquid LFM2.5 350m + 1.2B

Fast `LiquidAI/LFM2.5-350M-GGUF` (~10k dl) + default `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` (~51k dl). Smallest Liquid preset.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=LiquidAI/LFM2.5-350M-GGUF
NOXA_ANSWER_MODEL_DEFAULT=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Liquid LFM2.5 1.2B + 8B MoE

Fast `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` + default `LiquidAI/LFM2.5-8B-A1B-GGUF` (~164k dl). 8B MoE (A1B active); use `mode: quality` for the larger model.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=LiquidAI/LFM2.5-8B-A1B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Liquid LFM2.5 1.2B Thinking

Fast `LiquidAI/LFM2.5-350M-GGUF` + default `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` (~12k dl). Reasoning-tuned variant for `default` / `quality` modes.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=LiquidAI/LFM2.5-350M-GGUF
NOXA_ANSWER_MODEL_DEFAULT=LiquidAI/LFM2.5-1.2B-Thinking-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Llama 3.2 1B + 3B

Fast `unsloth/Llama-3.2-1B-Instruct-GGUF` (~27k dl) + default `unsloth/Llama-3.2-3B-Instruct-GGUF` (~76k dl). Meta stack with a real fast/default split.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Llama-3.2-1B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Llama-3.2-3B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Llama 3.2 3B Instruct (single model)

Both modes use `unsloth/Llama-3.2-3B-Instruct-GGUF` (~76k dl). Good A/B vs Qwen3 when you want one model for all modes.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Llama-3.2-3B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Llama-3.2-3B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Llama 3.1 8B quality

Fast `unsloth/Llama-3.2-1B-Instruct-GGUF` + default `MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF` (~179k dl). Heavier; use `mode: quality` and enough RAM/VRAM.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Llama-3.2-1B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=MaziyarPanahi/Meta-Llama-3.1-8B-Instruct-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Mistral 7B v0.3

Fast `unsloth/Llama-3.2-1B-Instruct-GGUF` + default `MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF` (~187k dl). Classic Mistral instruct line.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Llama-3.2-1B-Instruct-GGUF
NOXA_ANSWER_MODEL_DEFAULT=MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### DeepSeek R1 distill Qwen 7B (reasoning)

Fast `unsloth/Qwen3-0.6B-GGUF` + default `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` (~34k dl). Reasoning-oriented quality model; slower than 3–4B instruct models.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=
NOXA_RERANK_MODEL=
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### English-fast retrieval (small models)

Answer: Qwen3 unsloth stack. Retrieval: `nomic-ai/nomic-embed-text-v1.5-GGUF` + `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp`.

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Smallest retrieval (MiniLM)

Answer: Qwen3 unsloth stack. Retrieval: `sentence-transformers/all-MiniLM-L6-v2` + `Xenova/ms-marco-MiniLM-L-6-v2` (fastest/smallest pair).

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=mac-local
NOXA_ANSWER_MODEL_FAST=unsloth/Qwen3-0.6B-GGUF
NOXA_ANSWER_MODEL_DEFAULT=unsloth/Qwen3-1.7B-GGUF
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
NOXA_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

#### Torch transformers (no GGUF)

Needs `uv sync --extra ml`. Uses Hugging Face **transformers** model ids (not GGUF repos).

**Qwen3**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=Qwen/Qwen3-0.6B
NOXA_ANSWER_MODEL_DEFAULT=Qwen/Qwen3-1.7B
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
NOXA_RERANK_MODEL=jinaai/jina-reranker-v2-base-multilingual
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**Phi-4 mini**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=microsoft/Phi-4-mini-instruct
NOXA_ANSWER_MODEL_DEFAULT=microsoft/Phi-4-mini-instruct
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**Gemma 3 4B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=google/gemma-3-1b-it
NOXA_ANSWER_MODEL_DEFAULT=google/gemma-3-4b-it
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**Llama 3.2 3B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=meta-llama/Llama-3.2-1B-Instruct
NOXA_ANSWER_MODEL_DEFAULT=meta-llama/Llama-3.2-3B-Instruct
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**SmolLM2 1.7B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=HuggingFaceTB/SmolLM2-360M-Instruct
NOXA_ANSWER_MODEL_DEFAULT=HuggingFaceTB/SmolLM2-1.7B-Instruct
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
NOXA_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**SmolLM3 3B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=HuggingFaceTB/SmolLM3-3B
NOXA_ANSWER_MODEL_DEFAULT=HuggingFaceTB/SmolLM3-3B
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**Granite 4.1 3B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=ibm-granite/granite-4.1-3b
NOXA_ANSWER_MODEL_DEFAULT=ibm-granite/granite-4.1-3b
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**Granite 4.0 350m + 4.1 3B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=ibm-granite/granite-4.0-350m
NOXA_ANSWER_MODEL_DEFAULT=ibm-granite/granite-4.1-3b
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
NOXA_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

**Liquid LFM2.5 350m + 1.2B**

```bash
HF_TOKEN=hf_...
NOXA_RUNTIME_PROFILE=auto
NOXA_ANSWER_MODEL_FAST=LiquidAI/LFM2.5-350M
NOXA_ANSWER_MODEL_DEFAULT=LiquidAI/LFM2.5-1.2B-Instruct
NOXA_ANSWER_GGUF_QUANT=Q4_K_M
NOXA_EMBED_MODEL=BAAI/bge-small-en-v1.5
NOXA_RERANK_MODEL=BAAI/bge-reranker-base
NOXA_MODEL_CACHE_DIR=.noxa_models
NOXA_PRELOAD_MODELS=true
```

### Runtime

| Variable | Values | Meaning |
|----------|--------|---------|
| `NOXA_RUNTIME_PROFILE` | `auto`, `mac-local`, `cloud-cpu`, `cloud-gpu` | Hardware profile for llama.cpp GPU offload defaults |

All inference (answer, embed, rerank) uses **llama-cpp-python**.

### Models

| Variable | Example | Notes |
|----------|---------|-------|
| `NOXA_ANSWER_MODEL_FAST` | `unsloth/Qwen3-0.6B-GGUF` | Fast mode GGUF repo |
| `NOXA_ANSWER_MODEL_DEFAULT` | `unsloth/Qwen3-1.7B-GGUF` | Default/quality mode GGUF repo |
| `NOXA_ANSWER_GGUF_QUANT` | `Q4_K_M` | Quantization tag to pick from each GGUF repo |
| `NOXA_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | Embedding GGUF repo (`create_embedding`) |
| `NOXA_RERANK_MODEL` | `Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp` | Reranker GGUF repo (yes/no logit scoring) |

Request `mode` (`fast` / `default` / `quality`) selects which answer model role is used — not a separate env var.

### Bootstrap & cache

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_TOKEN` | — | Hugging Face token for gated models / rate limits |
| `NOXA_PRELOAD_MODELS` | `true` | Download and warm models at startup |
| `NOXA_MODEL_CACHE_DIR` | `.noxa_models` | Local GGUF download directory |
| `NOXA_SQLITE_PATH` | `noxa.db` | Search/fetch/embedding cache |

### Server & fetch

| Variable | Default | Notes |
|----------|---------|-------|
| `NOXA_HOST` | `0.0.0.0` | Bind address |
| `NOXA_PORT` | `8000` | Listen port |
| `NOXA_DEFAULT_MODE` | `default` | Default pipeline mode when request omits `mode` |
| `NOXA_GLOBAL_TIMEOUT_MS` | `25000` | Fetch stage budget |
| `NOXA_PER_PAGE_TIMEOUT_MS` | `8000` | Single-page fetch timeout |
| `NOXA_MAX_CHARS_PER_PAGE` | `80000` | Truncate fetched markdown |
| `NOXA_PROXIES` | — | Comma-separated proxy URLs |
| `NOXA_DEBUG_DUMP_DIR` | `dump` | Debug output for `return_debug: true` |

### Benchmark

Benchmark answer backends:

```bash
uv run python scripts/benchmark_answer.py --fixture dump/web_answer_* --output benchmark.csv
```

See [prd.md](prd.md) for full specification.
