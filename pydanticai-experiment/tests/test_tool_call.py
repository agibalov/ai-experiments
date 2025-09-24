from pydantic_ai import RunContext
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
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

async def test_tool_call():
    model = OpenAIChatModel(
        model_name='gpt-oss:20b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
    )
    agent = Agent(model=model, system_prompt="Use `add_numbers` to add two numbers.")

    @agent.tool
    async def add_numbers(ctx: RunContext[int], a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with agent.iter("give me a sum of 2, 3 and 88") as agent_run:
        await print_run(agent_run)

    print(f"Agent says: {agent_run.result.output}")
    assert agent_run.result.output != ''

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
                    async for output in request_stream.stream_text():
                        print(f'[Output] {output}')
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
