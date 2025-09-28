from typing import List
from pydantic_ai import ModelSettings
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel
import json


class ContextBasedAnsweringAgentResponse(BaseModel):
    answer: str
    found_answer: bool

class ContextBasedAnsweringAgent:
    agent: Agent[ContextBasedAnsweringAgentResponse]

    def __init__(self):
        model = OpenAIChatModel(
            model_name='gpt-oss:20b',
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
            settings=ModelSettings(temperature=0.0)
        )
        self.agent = Agent(
            model=model,
            output_type=ContextBasedAnsweringAgentResponse,
            system_prompt="""
            Given a question and a context, provide a concise and accurate answer based on the context.

            Prefer answers explicitly stated in the context.
            If there are none, you should infer and draw conclusions, but only from what is in the context.
            If not enough info, say: "The context does not provide the answer."
            """.strip()
        )

    def respond(self, query: str, context: List[str]) -> ContextBasedAnsweringAgentResponse:
        result = self.agent.run_sync(json.dumps({
            "question": query,
            "context": context
        }))
        return result.output
