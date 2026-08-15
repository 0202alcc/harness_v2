# Harness/chunker.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict
from LLManager import LLManager


DEFAULT_CHUNK_SIZE = 512


class Chunk(TypedDict):
    index: int
    token_start: int
    token_end: int
    token_count: int
    token_ids: list[int]
    text: str


class ChunkResult(TypedDict):
    input_text: str
    model: str
    total_tokens: int
    chunks: list[Chunk]


class Chunker(ABC):
    """
    Interface for Harness chunking strategies.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
    ) -> ChunkResult:
        raise NotImplementedError


class FixedTokenChunker(Chunker):
    """
    Splits text into fixed-size model-token chunks.

    V1 policy:
        - tokenizer: current llama.cpp model tokenizer
        - chunk size: 512 source tokens
        - overlap: 0
        - semantic boundary adjustment: none
    """

    def __init__(
        self,
        *,
        llm: LLManager,
        model: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        self.llm = llm
        self.model = model
        self.chunk_size = chunk_size

    def chunk(
        self,
        text: str,
    ) -> ChunkResult:
        """
        Tokenize text with the active model tokenizer and split
        it into non-overlapping fixed-size chunks.
        """

        token_ids = self.llm.tokenize(
            text,
            model=self.model,
            add_special=False,
            parse_special=False,
        )

        chunks: list[Chunk] = []

        for start in range(
            0,
            len(token_ids),
            self.chunk_size,
        ):
            end = min(
                start + self.chunk_size,
                len(token_ids),
            )

            chunk_token_ids = token_ids[start:end]

            chunk_text = self.llm.detokenize(
                chunk_token_ids,
                model=self.model,
            )

            chunks.append(
                {
                    "index": len(chunks),
                    "token_start": start,
                    "token_end": end,
                    "token_count": len(chunk_token_ids),
                    "token_ids": chunk_token_ids,
                    "text": chunk_text,
                }
            )

        return {
            "input_text": text,
            "model": self.model,
            "total_tokens": len(token_ids),
            "chunks": chunks,
        }