from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

import json

class TextTestResult(BaseModel):
    is_true: bool = Field(..., description="Whether the claim is true based on the text")
    why: str = Field(..., description="A brief explanation of why the statement is true or false")

def assert_text_entails_claim(text: str, claim: str) -> None:
    model = init_chat_model("ollama:llama3.2:3b")
    parser = JsonOutputParser()
    fixing = OutputFixingParser.from_llm(parser=parser, llm=model)
    schema_json = json.dumps(TextTestResult.model_json_schema(), indent=2)

    prompt = ChatPromptTemplate.from_template(
"""
You are an expert evaluating claims against the text.
Claims can be about what is explicitly stated in the text. For example:

Text: "The sky is blue."
Claim: "Blue is the sky's color."
Result: true, because the statement is a paraphrase of the text.

Claims can be about the qualities of the text. For example:

Text: "The sky is blue."
Claim: "Mentions sky"
Result: true, because the text mentions the sky.

Text: "The sky is blue."
Claim: "Mentions bicycles"
Result: false, because the text does not mention bicycles.

Text: "The sky is blue."
Claim: "Does not mention bicycles"
Result: true, because the text does not mention bicycles.

Allow for broad, vague, inaccurate claims as long as they don't contradict the text.

Output ONLY valid JSON that matches this JSON Schema:
{schema}

Now evaluate:

Text:
{text}

Claim:
{claim}

JSON only; no extra keys, no prose, no code fences.
""")

    chain = prompt.partial(schema=schema_json) | model | fixing
    result_json = chain.invoke({"text": text, "claim": claim})
    result = TextTestResult.model_validate(result_json)

    if not result.is_true:
        raise AssertionError(f"Claim is false: {result.why}")
