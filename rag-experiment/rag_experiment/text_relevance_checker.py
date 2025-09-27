from pydantic_ai import ModelSettings
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel
import json


class TextRelevanceCheckResult(BaseModel):
    is_relevant: bool
    reason: str

class TextRelevanceChecker:
    _agent: Agent[TextRelevanceCheckResult]

    def __init__(self):
        model = OpenAIChatModel(
            model_name='gpt-oss:20b',
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
            settings=ModelSettings(temperature=0.0)
        )
        self._agent = Agent(
            model=model,
            output_type=TextRelevanceCheckResult,
            system_prompt="""
            Given a question and a text, determine if text is relevant to a question or not.
            """.strip()
        )

    def check(self, question: str, text: str) -> TextRelevanceCheckResult:
        result = self._agent.run_sync(json.dumps({
            "question": question,
            "text": text
        }))
        return result.output
