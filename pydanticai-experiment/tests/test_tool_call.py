from pydantic_ai import ModelSettings
from pydantic_ai.agent import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
    ToolCallPart
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import pytest

@pytest.fixture
def agent():
    model = OpenAIChatModel(
        model_name='gpt-oss:20b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        settings=ModelSettings(temperature=0.0)
    )

    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(
        model=model,
        tools=[add_numbers],
        output_type=[int, str],
        system_prompt="""
        Use `add_numbers` to add two numbers. Return a number on success,
        or a string if you cannot perform the addition.""")

    return agent

@pytest.mark.asyncio
async def test_tool_call_success(agent):
    async with agent.iter("give me a sum of 2, 3 and 88") as agent_run:
        await print_run(agent_run)

    result = agent_run.result.output
    print(f"Agent says ({type(result)}): {result}")

    tool_call_count = 0
    for m in agent_run.result.all_messages():
        if m.parts:
            for p in m.parts:
                if isinstance(p, ToolCallPart):
                    print(f"Tool call: {p.tool_name}({p.args})")
                    tool_call_count += 1

    assert isinstance(result, int)
    assert result == 93
    assert tool_call_count >= 2

@pytest.mark.asyncio
async def test_tool_call_error(agent):
    async with agent.iter("give me a sandwich") as agent_run:
        await print_run(agent_run)

    result = agent_run.result.output
    print(f"Agent says ({type(result)}): {result}")

    assert isinstance(result, str)

async def print_run(run):
    async for node in run:
        if Agent.is_user_prompt_node(node):
            print(f'=== UserPromptNode: {node.user_prompt} ===')
        elif Agent.is_model_request_node(node):
            print('=== ModelRequestNode: streaming partial request tokens ===')
            async with node.stream(run.ctx) as request_stream:
                final_result_found = False
                async for event in request_stream:
                    if isinstance(event, PartStartEvent):
                        print(f'[Request] Starting part {event.index}: {event.part!r}')
                    elif isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            print(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
                        elif isinstance(event.delta, ThinkingPartDelta):
                            print(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
                        elif isinstance(event.delta, ToolCallPartDelta):
                            print(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
                    elif isinstance(event, FinalResultEvent):
                        print(f'[Result] The model started producing a final result (tool_name={event.tool_name})')
                        final_result_found = True
                        break
                if final_result_found:
                    output = None
                    async for o in request_stream.stream_text():
                        output = o
                    print(f"\r[Output] {output}")
        elif Agent.is_call_tools_node(node):
            print('=== CallToolsNode: streaming partial response & tool usage ===')
            async with node.stream(run.ctx) as handle_stream:
                async for event in handle_stream:
                    if isinstance(event, FunctionToolCallEvent):
                        print(f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})')
                    elif isinstance(event, FunctionToolResultEvent):
                        print(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
        elif Agent.is_end_node(node):
            assert run.result is not None
            assert run.result.output == node.data.output
            print(f'=== Final Agent Output: {run.result.output} ===')
