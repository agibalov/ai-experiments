from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List

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

                Gates was born and raised in Seattle, Washington. In 1975, he and Allen founded Microsoft in Albuquerque,
                New Mexico. Gates led the company as its chairman and chief executive officer until stepping down as CEO in
                January 2000, succeeded by Steve Ballmer, but he remained chairman of the board of directors and became
                chief software architect. During the late 1990s, he was criticized for his business tactics, which were
                considered anti-competitive.
                """
            },
        ],
        response_format=NamedEntitiesResponse
    )

    response: NamedEntitiesResponse = completion.choices[0].message.parsed
    for entity in response.entities:
        print(entity)

    assert NamedEntity(text="William Henry Gates III", type=NamedEntityType.person) in response.entities
    assert NamedEntity(text="Paul Allen", type=NamedEntityType.person) in response.entities
    assert NamedEntity(text="Steve Ballmer", type=NamedEntityType.person) in response.entities
    assert NamedEntity(text="October 28, 1955", type=NamedEntityType.date) in response.entities
    assert NamedEntity(text="May 2014", type=NamedEntityType.date) in response.entities
    assert NamedEntity(text="1975", type=NamedEntityType.date) in response.entities
    assert NamedEntity(text="1970s", type=NamedEntityType.date) in response.entities
    assert NamedEntity(text="1980s", type=NamedEntityType.date) in response.entities
    assert NamedEntity(text="Seattle, Washington", type=NamedEntityType.place) in response.entities
    assert NamedEntity(text="Albuquerque, New Mexico", type=NamedEntityType.place) in response.entities