# openai-experiment

An OpenAI hello world.

## Prerequisites

* pyenv
* poetry

## How to do things

* `poetry install` to install dependencies
* `export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY_HERE>` and then:
  * `poetry run app` to run the app
  * `poetry run test` to run all tests
  * `poetry run test tests/test_calculator.py` - AI acts as a calculator
  * `poetry run test tests/test_functions.py` - AI calls a function in my code
  * `poetry run test tests/test_extract_event_details.py` - AI extracts event details from text (model provides a structured response)
  * `poetry run test tests/test_named_entity_recognition.py` - AI extracts people, dates and places from text (model provides a structured response)
  * `poetry run test tests/test_sentiment_analysis.py` - AI performs sentiment analysis (model provides a structured response)
  * `poetry run test tests/test_assistant.py` - assistant hello world
