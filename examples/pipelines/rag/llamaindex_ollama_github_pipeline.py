"""
title: Llama Index Ollama Github Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-github
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import asyncio


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.readers.github import GithubRepositoryReader, GithubClient

        # Зчитування даних з ENV, або вручну
        github_token = os.getenv("GITHUB_TOKEN", input("Enter GITHUB_TOKEN: "))
        model_name = os.getenv("LLM_MODEL", input("Enter model name (default: llama3): ") or "llama3")
        base_url = os.getenv("BASE_URL", input("Enter base_url (default: http://localhost:11434): ") or "http://localhost:11434")
        owner = os.getenv("GITHUB_OWNER", input("Enter GitHub owner (default: open-webui): ") or "open-webui")
        repo = os.getenv("GITHUB_REPO", input("Enter GitHub repo (default: plugin-server): ") or "plugin-server")
        branch = os.getenv("GITHUB_BRANCH", input("Enter GitHub branch (default: main): ") or "main")

        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url=base_url,
        )
        Settings.llm = Ollama(model=model_name)

        github_client = GithubClient(github_token=github_token, verbose=True)
        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=(
                [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                    ".ico",
                    ".json",
                    ".ipynb",
                ],
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        )

        loop = asyncio.new_event_loop()
        reader._loop = loop

        try:
            # Завантаження даних
            self.documents = await asyncio.to_thread(reader.load_data, branch=branch)
            self.index = VectorStoreIndex.from_documents(self.documents)
        finally:
            loop.close()

        print(self.documents)
        print(self.index)
