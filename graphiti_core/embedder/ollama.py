"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
else:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            'openai is required for OllamaEmbedder. '
            'Install it with: pip install openai'
        ) from None

from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = 'nomic-embed-text'
DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434/v1'


class OllamaEmbedderConfig(EmbedderConfig):
    """Configuration for Ollama embedder using OpenAI-compatible interface"""
    embedding_model: str = Field(default=DEFAULT_OLLAMA_MODEL)
    base_url: str = Field(default=DEFAULT_OLLAMA_BASE_URL)
    api_key: str | None = Field(default=None)  # Ollama local doesn't require API key


class OllamaEmbedder(EmbedderClient):
    """
    Ollama Embedder Client using OpenAI-compatible interface

    This client uses Ollama's OpenAI-compatible embedding API.
    Ollama provides an OpenAI-compatible interface for local models.

    Configuration via environment variables:
        OLLAMA_BASE_URL: Base URL for Ollama API (default: http://localhost:11434/v1)
        OLLAMA_EMBEDDING_MODEL: Model name (default: nomic-embed-text)
        OLLAMA_API_KEY: Optional API key (default: None)
    """

    def __init__(
        self,
        config: OllamaEmbedderConfig | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """
        Initialize the OllamaEmbedder with the provided configuration and client.

        Args:
            config (OllamaEmbedderConfig | None): Configuration for the Ollama embedder
            client (AsyncOpenAI | None): Optional OpenAI client for Ollama's OpenAI-compatible API
        """
        import os

        if config is None:
            # Load from environment variables
            config = OllamaEmbedderConfig(
                embedding_model=os.getenv('OLLAMA_EMBEDDING_MODEL', DEFAULT_OLLAMA_MODEL),
                base_url=os.getenv('OLLAMA_BASE_URL', DEFAULT_OLLAMA_BASE_URL),
                api_key=os.getenv('OLLAMA_API_KEY'),  # None by default for local Ollama
            )

        self.config = config

        if client is not None:
            self.client = client
        else:
            # Use OpenAI-compatible client for Ollama
            self.client = AsyncOpenAI(
                api_key=config.api_key or 'ollama',  # Ollama accepts any key or 'ollama'
                base_url=config.base_url
            )

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data using Ollama's local models.

        Args:
            input_data: Input text(s) to embed

        Returns:
            A list of floats representing the embedding vector

        Raises:
            ValueError: If no embeddings are returned
            Exception: If API call fails
        """
        try:
            result = await self.client.embeddings.create(
                input=input_data,
                model=self.config.embedding_model
            )

            if not result.data or not result.data[0].embedding:
                raise ValueError('No embeddings returned from Ollama API')

            embedding = result.data[0].embedding

            # Truncate or pad to configured embedding dimension
            if len(embedding) > self.config.embedding_dim:
                embedding = embedding[:self.config.embedding_dim]
            elif len(embedding) < self.config.embedding_dim:
                # Pad with zeros if shorter than expected
                embedding = list(embedding) + [0.0] * (self.config.embedding_dim - len(embedding))

            return embedding

        except Exception as e:
            logger.error(f'Ollama embedding API error: {e}')
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of input data using Ollama.

        Args:
            input_data_list: List of input texts to embed

        Returns:
            List of embedding vectors

        Note:
            Ollama handles batching efficiently through its OpenAI-compatible interface
        """
        if not input_data_list:
            return []

        try:
            result = await self.client.embeddings.create(
                input=input_data_list,
                model=self.config.embedding_model
            )

            if not result.data:
                raise ValueError('No embeddings returned from Ollama API')

            all_embeddings = []
            for item in result.data:
                embedding = item.embedding

                # Truncate or pad to configured dimension
                if len(embedding) > self.config.embedding_dim:
                    embedding = embedding[:self.config.embedding_dim]
                elif len(embedding) < self.config.embedding_dim:
                    embedding = list(embedding) + [0.0] * (self.config.embedding_dim - len(embedding))

                all_embeddings.append(embedding)

            return all_embeddings

        except Exception as e:
            logger.error(f'Ollama batch embedding API error: {e}')
            # Fall back to individual processing
            logger.info('Falling back to individual embedding processing')
            all_embeddings = []
            for item in input_data_list:
                embedding = await self.create(item)
                all_embeddings.append(embedding)

            return all_embeddings
