# langchain-experiment

Learning [LangChain](https://www.langchain.com/).

## How to do things

* `uv run pytest` to run all tests
* `uv run pytest -k 'test_agent_sql and model_ollama_llama3_2_3b and temp_0_2 and seed_1337'` to run all agent tests from `test_agent_sql.py` using the specified parameters.
* `uv run pytest -k 'test_count_persons_older_than_30 and model_ollama_llama3_2_3b and temp_0_2 and seed_1337'` to run `test_count_persons_older_than_30` from `test_agent_sql.py` using the specified parameters.
