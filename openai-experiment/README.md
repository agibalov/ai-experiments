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
  * `poetry run test tests/test_calculator.py` to run tests where AI acts as a calculator
  * `poetry run test tests/test_functions.py` to run tests where AI calls a function in my code
  * `poetry run test tests/test_extract_event_details.py` extract event details from text (model provides a structured response)
  * `poetry run test tests/test_named_entity_recognition.py` extract people, dates and places from text (model provides a structured response)
