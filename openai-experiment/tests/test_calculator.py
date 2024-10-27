import pytest
from .fixtures import client, interpreter

@pytest.fixture
def calculator(client):    
    def f(request):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are a calculator robot. The only thing you can do
                    is calculating arithmetic expressions. You should not
                    have conversations with the user on any topic at all.
                    When user gives you an expression, you respond with a 
                    single number - the final calculation result.
                    """
                },
                {
                    "role": "user",
                    "content": request
                }
            ]
        )
        return completion.choices[0].message.content
    return f

def test_can_calculate(calculator):
    assert calculator("2 * 3 + 1") == "7"

# unreliable
def test_division_by_zero(calculator, interpreter):
    response = calculator("1/0")
    assert interpreter(response, "it may represent a result of division by zero") == "yes"

def test_refuses_to_talk(calculator, interpreter):
    response = calculator("what kind of wifi router do I need?")
    assert interpreter(response, "it says something about arithmetic expressions") == "yes"
