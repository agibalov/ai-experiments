from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
import ollama

call_log = []

@tool(parse_docstring=True)
def get_weather(location: str) -> str:
    """Get the current weather in a given location.
    
    Args:
        location: The location to get the weather for.
    """
    global call_log
    call_log.append(("get_weather", location))
    return f"{location}: 72 F, clear"

@tool(parse_docstring=True)
def get_local_time(location: str) -> str:
    """Get the current local time in a given location.
    
    Args:
        location: The location to get the time for.
    """
    global call_log
    call_log.append(("get_local_time", location))
    return "12:34:56"

def test_llm_can_use_tools():
    tools = [get_weather, get_local_time]
    tools_by_name = {tool.name: tool for tool in tools}

    model = "llama3.2:3b" # or "gpt-oss:20b"
    ollama.pull(model)
    llm = ChatOllama(model=model).bind_tools(tools)

    messages = [
        HumanMessage("What is the weather in New Jersey? Also, what's the local time there.")
    ]

    global call_log
    call_log = []

    while True:
        response = llm.invoke(messages)
        print("\n***", response)

        messages.append(response)
        if len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                tool = tools_by_name[tool_call["name"]]
                tool_response = tool.invoke(tool_call["args"])
                tool_message = ToolMessage(content=tool_response, tool_call_id=tool_call["id"])
                print("\n***", tool_message)

                messages.append(tool_message)
        else:
            break

    assert len(call_log) == 2
    assert ("get_weather", "New Jersey") in call_log
    assert ("get_local_time", "New Jersey") in call_log

    print(f"\nFinal message:\n{messages[-1].content}")
