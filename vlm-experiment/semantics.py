from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

model = OpenAIChatModel(
    model_name='gpt-oss:20b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    settings=ModelSettings(temperature=0.1)
)

class SemanticsTestRequest(BaseModel):
    statement: str
    substatement: str

class SemanticsTestResult(BaseModel):
    match: bool
    match_reason: str

agent = Agent(
    model=model,
    system_prompt=
    "Given a statement and a substatement, determine if the substatement is semantically a subset of the statement.",
    output_type=SemanticsTestResult
)

def match_semantics(statement: str, substatement: str) -> SemanticsTestResult:
    print(f"Statement: \"{statement}\", Substatement: \"{substatement}\"")
    output = agent.run_sync(SemanticsTestRequest(
        statement=statement,
        substatement=substatement
    ).model_dump_json()).output
    print(f"Match: \"{output.match}\", Reason: \"{output.match_reason}\"")
    return output
