from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

def test_hello_world():
    model = OpenAIChatModel(
        model_name='gpt-oss:20b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    agent = Agent(model=model)
    response = agent.run_sync("hello!")
    print(f"Agent says: {response.output}")
    assert response.output != ''
