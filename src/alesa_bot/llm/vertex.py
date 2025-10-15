 #===================== FILE: src/alesa_bot/llm/vertex.py =====================
from __future__ import annotations
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Content, Part


class VertexLLM:
    def __init__(self, project: str, location: str, model_name: str, creds_path: str) -> None:
        if creds_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
        vertexai.init(project=project, location=location)
        self.model = GenerativeModel(model_name)
        self._chat = None

    def start(self) -> None:
        # Leere History; System-Prompt reichen wir im Prompt-Text mit
        self._chat = self.model.start_chat(history=[])

    def generate(self, prompt: str) -> str:
        if self._chat is None:
            self.start()
        resp = self._chat.send_message(prompt)
        return (getattr(resp, 'text', '') or '')






