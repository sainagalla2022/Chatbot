"""Shared test setup: every test gets its own throw-away data folder,
plus fake embeddings and a fake LLM so no model or Ollama is ever needed."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import config


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point DATA_DIR and SQLITE_PATH at a temp folder so tests never touch real data."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "SQLITE_PATH", tmp_path / "test.db")
    return tmp_path


class FakeLLM:
    """Stands in for TinyLlama: returns canned replies and remembers the prompts.

    Give `replies` to return different text on each call (the last one repeats).
    """

    def __init__(self, reply: str = "fake answer", replies: list[str] | None = None) -> None:
        self.replies = replies if replies else [reply]
        self.prompts: list[str] = []
        self.stops: list[list[str] | None] = []  # the stop sequences of each call

    def invoke(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        index = min(len(self.prompts), len(self.replies) - 1)
        self.prompts.append(prompt)
        self.stops.append(stop)
        return self.replies[index]


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Hash-based embeddings: same text -> same vector, no model download."""
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
