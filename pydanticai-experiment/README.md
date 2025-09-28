# pydanticai-experiment

Giving [Pydantic AI](https://ai.pydantic.dev/) a try.

## Prerequisites

* `uv`
* Open WebUI - `uv tool install --python 3.12 open-webui`

## How to do things

* `uv run pytest` to run tests.
* `uv run uvicorn pydanticai_experiment.main:app --reload` to run OpenAI-compatible server at http://localhost:8000/v1
* `./open-webui.sh` to run Open WebUI at http://localhost:8080 configured to connect to http://localhost:8000/v1
