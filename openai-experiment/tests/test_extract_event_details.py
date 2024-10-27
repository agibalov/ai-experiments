from openai import OpenAI
from pydantic import BaseModel
from typing import List

def test_extract_event_details():
    class CalendarEvent(BaseModel):
        name: str
        day: str
        participants: List[str]

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
        ],
        response_format=CalendarEvent
    )

    event: CalendarEvent = completion.choices[0].message.parsed
    assert type(event) == CalendarEvent
    assert event.name == "Science Fair"
    assert event.day == "Friday"
    assert len(event.participants) == 2
    assert "Alice" in event.participants
    assert "Bob" in event.participants
