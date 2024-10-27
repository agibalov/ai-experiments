from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List

def test_extract_event_details():
    class CalendarEvent(BaseModel):
        name: str
        day: str
        participants: list[str]

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

def test_named_entity_recognition():
    class NamedEntityType(str, Enum):
        person = "person"
        place = "place"
        date = "date"

    class NamedEntity(BaseModel):
        text: str
        type: NamedEntityType

    class NamedEntitiesResponse(BaseModel):
        entities: List[NamedEntity]

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """
                Extract all named entities from the text. The types of entities you
                should extract are:

                1. "person" - a person's name like "Robert" or "John Smith"
                2. "place" - a geographic location like "Australia" or "New York"
                3. "date" - a date like just year "1939", month and year "May 1945" or full date like "January 1, 2024"
                """
            },
            {
                "role": "user",
                "content": """
                William Henry Gates III (born October 28, 1955) is an American businessman and philanthropist best known
                for co-founding the software company Microsoft with his childhood friend Paul Allen. During his career at
                Microsoft, Gates held the positions of chairman, chief executive officer (CEO), president, and chief
                software architect, while also being its largest individual shareholder until May 2014.[a] He was a pioneer
                of the microcomputer revolution of the 1970s and 1980s.
                """
            },
        ],
        response_format=NamedEntitiesResponse
    )

    response: NamedEntitiesResponse = completion.choices[0].message.parsed
    print(response)
