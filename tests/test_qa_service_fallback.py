from pathlib import Path
import unittest

from src.alesa_bot.core.types import Hit
from src.alesa_bot.services.qa_service import QAService


class DummyRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None

    def search(self, query: str, top_k: int = 6):
        self.last_query = query
        return self.hits


class DummyLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.prompts = []
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class QAFallbackTests(unittest.TestCase):
    def test_with_hits_returns_citations(self):
        hits = [Hit(path=Path("doc.txt"), page=1, snippet="snippet text")]
        qa = QAService(
            retriever=DummyRetriever(hits),
            llm=DummyLLM("antwort"),
            system_prompt="sys",
            query_expand=False,
        )
        answer, cites = qa.ask("frage")

        self.assertEqual(answer, "antwort")
        self.assertEqual(len(cites), 1)
        self.assertIn("doc.txt", cites[0])

    def test_without_hits_fallback_no_citations(self):
        qa = QAService(
            retriever=DummyRetriever([]),
            llm=DummyLLM("fallback"),
            system_prompt="sys",
            query_expand=False,
        )
        answer, cites = qa.ask("frage ohne treffer")

        self.assertEqual(answer, "fallback")
        self.assertEqual(cites, [])
        # ensure LLM still called once even ohne Treffer
        self.assertEqual(len(qa.llm.prompts), 1)


if __name__ == "__main__":
    unittest.main()
