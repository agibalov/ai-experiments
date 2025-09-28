from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

ollama_model = OpenAIChatModel(
    model_name='gpt-oss:20b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
agent = Agent(ollama_model, instructions="Be fun!")

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Sandwiches",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    recent_messages = req.messages[-10:]

    conversation_history = []
    for message in recent_messages:
        if message.role == "user":
            conversation_history.append(f"User: {message.content}")
        elif message.role == "assistant":
            conversation_history.append(f"Assistant: {message.content}")
        elif message.role == "system":
            conversation_history.append(f"System: {message.content}")

    full_conversation = "\n".join(conversation_history)
    response = await agent.run(full_conversation)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(full_conversation.split()),
            "completion_tokens": len(response.output.split()),
            "total_tokens": len(full_conversation.split()) + len(response.output.split()),
        },
    }
