from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
import pytest


@pytest.mark.asyncio
async def test_agent():
    llm = Ollama(
        model="gpt-oss:20b",
        temperature=0.1,
        request_timeout=120.0
    )

    def multiply(a: float, b: float) -> float:
        """Useful for multiplying two numbers."""
        return a * b

    agent = FunctionAgent(
        tools=[multiply],
        llm=llm,
        system_prompt="You are a helpful assistant that can multiply two numbers.",
    )

    response = await agent.run("What is 1234 * 4567?")
    print(response)

    # TODO: assert
