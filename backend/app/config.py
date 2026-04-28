from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / '.env')


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv('APP_NAME', 'ClarityAI Production')
    app_version: str = os.getenv('APP_VERSION', '3.0.0')
    api_prefix: str = os.getenv('API_PREFIX', '/api')
    storage_dir: Path = Path(os.getenv('STORAGE_DIR', BACKEND_DIR / 'storage'))
    database_url: str = os.getenv('DATABASE_URL', f"sqlite:///{BACKEND_DIR / 'storage' / 'clarityai.db'}")
    llm_base_url: str = os.getenv('LLM_BASE_URL', '').rstrip('/')
    llm_api_key: str = os.getenv('LLM_API_KEY', '')
    chat_model: str = os.getenv('CHAT_MODEL', '')
    enable_remote_llm: bool = os.getenv('ENABLE_REMOTE_LLM', 'true').lower() in {'1', 'true', 'yes'}
    temperature: float = float(os.getenv('TEMPERATURE', '0.2'))
    request_timeout_seconds: int = int(os.getenv('REQUEST_TIMEOUT_SECONDS', '180'))
    max_history_messages: int = int(os.getenv('MAX_HISTORY_MESSAGES', '12'))
    history_char_budget: int = int(os.getenv('HISTORY_CHAR_BUDGET', '12000'))
    chunk_size: int = int(os.getenv('CHUNK_SIZE', '1400'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '220'))
    retrieval_top_k: int = int(os.getenv('RETRIEVAL_TOP_K', '6'))
    retrieval_candidate_pool: int = int(os.getenv('RETRIEVAL_CANDIDATES', '18'))
    low_confidence_threshold: float = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.14'))
    auto_research_threshold: float = float(os.getenv('AUTO_RESEARCH_THRESHOLD', '0.18'))
    max_upload_size_mb: int = int(os.getenv('MAX_UPLOAD_SIZE_MB', '30'))
    cors_origins: list[str] = field(default_factory=lambda: _split_csv(os.getenv('CORS_ORIGINS', 'http://localhost:5173')))
    upload_dir: Path = field(init=False)

    enable_dense_retrieval: bool = os.getenv('ENABLE_DENSE_RETRIEVAL', 'false').lower() in {'1', 'true', 'yes'}
    embedding_base_url: str = os.getenv('EMBEDDING_BASE_URL', '').rstrip('/')
    embedding_api_key: str = os.getenv('EMBEDDING_API_KEY', '')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', '')
    embedding_batch_size: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '64'))

    enable_web_research: bool = os.getenv('ENABLE_WEB_RESEARCH', 'false').lower() in {'1', 'true', 'yes'}
    tavily_api_key: str = os.getenv('TAVILY_API_KEY', '')
    research_max_results: int = int(os.getenv('RESEARCH_MAX_RESULTS', '5'))
    research_trust_allowlist: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv(
                'RESEARCH_TRUST_ALLOWLIST',
                'gov,edu,nih.gov,who.int,nasa.gov,docs.python.org,fastapi.tiangolo.com,developer.mozilla.org,openai.com'
            )
        )
    )

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.storage_dir / 'uploads'
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def remote_llm_configured(self) -> bool:
        return bool(self.enable_remote_llm and self.llm_base_url and self.chat_model)

    @property
    def embedding_provider_configured(self) -> bool:
        if not self.enable_dense_retrieval:
            return False
        base_url = self.embedding_base_url or self.llm_base_url
        model = self.embedding_model
        return bool(base_url and model)

    @property
    def web_research_configured(self) -> bool:
        return bool(self.enable_web_research and self.tavily_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
