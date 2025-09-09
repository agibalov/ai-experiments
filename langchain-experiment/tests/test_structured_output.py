from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import json

class Person(BaseModel):
    first_name: str = Field(..., description="The person's first name")
    last_name: str = Field(..., description="The person's last name")
    email: str = Field(..., description="The person's email address")

def test_llm_responds_with_json():
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        device_map="auto",
        model_kwargs={"dtype": "auto"},
        pipeline_kwargs={
            "max_new_tokens": 8192,
            "return_full_text": False,
            "do_sample": False
        }
    )
    chat = ChatHuggingFace(llm=llm)

    parser = JsonOutputParser()
    fixing = OutputFixingParser.from_llm(parser=parser, llm=chat)

    schema_json = json.dumps(Person.model_json_schema(), indent=2)
    prompt = ChatPromptTemplate.from_template("""
Output ONLY valid JSON that matches this JSON Schema:
{schema}

Instruction:
{instruction}

Valid JSON only; no prose.
""")

    chain = prompt.partial(schema=schema_json) | chat | fixing
    response_json = chain.invoke({"instruction": "Make a person named John Smith"})
    person = Person.model_validate(response_json)
    print(person)

    assert person.first_name == "John"
    assert person.last_name == "Smith"
    assert person.email is not None
