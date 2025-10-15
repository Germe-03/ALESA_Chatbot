# src/alesa_bot/retrieval/embeddings.py
from __future__ import annotations
import os
from typing import List
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

class EmbeddingEncoder:
    def __init__(self, project: str, location: str, model: str = "text-embedding-004"):
        aiplatform.init(project=project, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model)

    def encode(self, texts: List[str]) -> List[List[float]]:
        # batch freundlich halten
        out: List[List[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            vecs = self.model.get_embeddings(batch)
            out.extend([v.values for v in vecs])
        return out
