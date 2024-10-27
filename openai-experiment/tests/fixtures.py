import pytest
from openai import OpenAI

@pytest.fixture(scope="package")
def client():
    return OpenAI()

@pytest.fixture(scope="package")
def interpreter(client):
    def f(text, statement):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are a robot that can check if a statement about
                    a given text is valid or not. I give you a text and
                    a statement about that text, and you respond with one
                    word: "yes" or "no". Only lower case.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    The text is: "{text}". 
                    The statement is: "{statement}".
                    What say you?
                    """
                }
            ]
        )
        return completion.choices[0].message.content    
    return f
