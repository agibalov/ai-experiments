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
  * `run test tests/test_calculator.py` to run tests where AI acts as a calculator
  * `run test tests/test_functions.py` to run tests where AI calls a function in my code
