from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(StrEnum):
    FAST = "fast"
    DEFAULT = "default"
    QUALITY = "quality"


class SearchProvider(StrEnum):
    DDGS = "ddgs"


class SafeSearch(StrEnum):
    OFF = "off"
    MODERATE = "moderate"
    ON = "on"


class ModeConfig(BaseModel):
    max_search_results: int
    max_pages: int
    max_depth: int
    per_page_timeout_ms: int
    global_timeout_ms: int
    passage_tokens: int
    overlap_tokens: int
    bm25_top_k: int
    embedding_top_k: int
    merged_top_k: int
    rerank_final_top_k: int
    answer_model: str
    max_context_tokens: int
    max_output_tokens: int


MODE_PRESETS: dict[RuntimeMode, ModeConfig] = {
    RuntimeMode.FAST: ModeConfig(
        max_search_results=5,
        max_pages=3,
        max_depth=0,
        per_page_timeout_ms=5000,
        global_timeout_ms=10000,
        passage_tokens=180,
        overlap_tokens=30,
        bm25_top_k=30,
        embedding_top_k=30,
        merged_top_k=20,
        rerank_final_top_k=8,
        answer_model="answer_fast",
        max_context_tokens=2500,
        max_output_tokens=1024,
    ),
    RuntimeMode.DEFAULT: ModeConfig(
        max_search_results=6,
        max_pages=5,
        max_depth=0,
        per_page_timeout_ms=8000,
        global_timeout_ms=25000,
        passage_tokens=220,
        overlap_tokens=40,
        bm25_top_k=50,
        embedding_top_k=50,
        merged_top_k=35,
        rerank_final_top_k=8,
        answer_model="answer_default",
        max_context_tokens=3500,
        max_output_tokens=1536,
    ),
    RuntimeMode.QUALITY: ModeConfig(
        max_search_results=10,
        max_pages=8,
        max_depth=1,
        per_page_timeout_ms=10000,
        global_timeout_ms=45000,
        passage_tokens=250,
        overlap_tokens=50,
        bm25_top_k=100,
        embedding_top_k=100,
        merged_top_k=70,
        rerank_final_top_k=10,
        answer_model="answer_default",
        max_context_tokens=6000,
        max_output_tokens=2048,
    ),
}


def resolve_mode_limits(
    mode_cfg: ModeConfig,
    max_search_results: int | None = None,
    max_pages: int | None = None,
) -> tuple[int, int]:
    """Use mode preset limits unless the request explicitly overrides them."""
    return (
        mode_cfg.max_search_results
        if max_search_results is None
        else max_search_results,
        mode_cfg.max_pages if max_pages is None else max_pages,
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NOXA_", env_file=".env", extra="ignore")

    default_mode: RuntimeMode = RuntimeMode.DEFAULT
    max_concurrent_requests: int = 100

    # Search
    search_provider: str = "ddgs"
    default_max_results: int = 8

    # Fetch / crawl
    per_page_timeout_ms: int = 8000
    global_timeout_ms: int = 25000
    max_bytes_per_page: int = 3_000_000
    max_chars_per_page: int = 80_000
    user_agent: str = "Noxa/0.1"
    crawl_max_pages: int = 10
    crawl_max_depth: int = 2
    crawl_include_external: bool = False

    # Runtime backends (auto picks best available per platform)
    runtime_profile: str = "auto"
    answer_backend: str = "auto"
    embed_backend: str = "auto"
    rerank_backend: str = "auto"
    model_cache_dir: str = ".noxa_models"

    # Model ids (optional; unset = built-in defaults)
    # answer_model_* for llama_cpp: Hugging Face GGUF repo (e.g. unsloth/Qwen3-0.6B-GGUF)
    answer_model_fast: str | None = None
    answer_model_default: str | None = None
    answer_gguf_quant: str = "Q4_K_M"
    embed_model: str | None = None
    rerank_model: str | None = None

    embedding_dimensions: int = 384
    embedding_batch_size: int = 64
    rerank_max_per_url: int = 2
    preload_models: bool = True
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "NOXA_HF_TOKEN"),
    )

    # Hybrid retrieval
    bm25_weight: float = 0.45
    embedding_weight: float = 0.55
    bm25_title_boost: float = 0.5
    bm25_url_boost: float = 0.3

    # Cache
    sqlite_path: str = "noxa.db"
    search_ttl_seconds: int = 3600
    fetch_ttl_seconds: int = 86400
    embedding_ttl_seconds: int = 2_592_000

    # Proxies: comma-separated list in env NOXA_PROXIES
    proxies: list[str] = Field(default_factory=list)

    host: str = "0.0.0.0"
    port: int = 8000
    debug_dump_dir: str = "dump"

    @field_validator("proxies", mode="before")
    @classmethod
    def parse_proxies(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return list(value)  # type: ignore[arg-type]

    def get_mode(self, mode: RuntimeMode | str | None) -> ModeConfig:
        key = RuntimeMode(mode) if mode else self.default_mode
        return MODE_PRESETS[key]

    def answer_role_for_mode(self, mode: RuntimeMode | str | None) -> str:
        return self.get_mode(mode).answer_model


def get_settings() -> Settings:
    return Settings()
