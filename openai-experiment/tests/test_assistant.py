from openai import OpenAI
from .fixtures import client
import pytest

@pytest.fixture
def assistant(client):
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o")
    yield assistant
    client.beta.assistants.delete(assistant.id)

def test_assistant(client, assistant):
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?")

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.")

    assert run.status == 'completed'

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages = list(reversed(list(messages)))

    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")
        print()

    assert len(messages) > 1

    assert messages[0].role == "user"
    assert messages[0].content[0].type == "text"
    assert "Can you help me?" in messages[0].content[0].text.value

    response = ""

    assert messages[1].role == "assistant"
    assert messages[1].content[0].type == "text"
    response += messages[1].content[0].text.value

    if len(messages) == 3:
        assert messages[2].role == "assistant"
        assert messages[2].content[0].type == "text"
        response += messages[2].content[0].text.value

    assert "x = 1" in response
