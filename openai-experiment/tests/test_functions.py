import json
from .fixtures import client, interpreter

def test_functions(client, interpreter):
    def call(messages):
        return client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "description": "Get the order status. Use this function when you need to get the order status",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The order Id"
                                }
                            },
                            "required": ["order_id"],
                            "additionalProperties": False
                        }
                    }
                }
            ]
        )
    
    messages = [
        {
            "role": "system", 
            "content": """
            You are an online store robot. Among other things, you can check order statuses.
            When the order status is "absolutely_lost", you should nicely apologize and
            explain that the customer won't receive what they paid for, we won't make a refund,
            and that they should move on.
            """
        },
        {
            "role": "user",
            "content": "I'd like to know the status of my order"
        }            
    ]
    
    message = call(messages).choices[0].message
    print(message)
    assert interpreter(message.content, "this text is asking someone to provide an order ID") == "yes"

    messages.append(message)
    messages.append({
        "role": "user",
        "content": "sure, it's qqq111"
    })
    message = call(messages).choices[0].message
    print(message)
    assert message.role == "assistant"
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].type == "function"
    assert message.tool_calls[0].function.name == "get_order_status"
    assert json.loads(message.tool_calls[0].function.arguments)["order_id"] == "qqq111"

    messages.append(message)
    messages.append({
        "role": "tool",
        "content": json.dumps({
            "order_id": "qqq111",
            "status": "absolutely_lost"
        }),
        "tool_call_id": message.tool_calls[0].id
    })

    message = call(messages).choices[0].message
    print(message)
    assert interpreter(message.content, "this text is telling someone they won't receive their package or money back")
