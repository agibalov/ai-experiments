# langchain-experiment

Learning [LangChain](https://www.langchain.com/).

## How to do things

* `uv run pytest` to run all tests
* `uv run pytest -k 'test_agent_sql and model_ollama_llama3_2_3b and temp_0_2 and seed_1337'` to run all agent tests from `test_agent_sql.py` using the specified parameters.
* `uv run pytest -k 'test_find_top_3_customers and model_ollama_llama3_2_3b and temp_0_2 and seed_1337'` to run `test_find_top_3_customers` from `test_agent_sql.py` using the specified parameters.

## test_agent_sql.py

`tests/test_agent_sql.py` is an effort to build a toy agent that makes SQL queries against an in-memory sqlite database to answer user's questions.

|        filepath         |           function           |                    params                    | passed | failed | SUBTOTAL |
| ----------------------- | ---------------------------- | -------------------------------------------- | -----: | -----: | -------: |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_2-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_2-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_2-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_2-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_6-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_6-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_6-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_0_6-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_1-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_1-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_1-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_llama3_2_3b-temp_1-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_2-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_2-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_2-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_2-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_6-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_6-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_6-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_0_6-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_1-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_1-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_1-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customers         | model_ollama_gpt_oss_20b-temp_1-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_1-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_2-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_2-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_2-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_2-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_6-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_6-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_6-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_0_6-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_1-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_1-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_1-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_gpt_oss_20b-temp_1-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_2-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_2-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_2-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_2-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_6-seed_0     |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_6-seed_1337  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_6-seed_2302  |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_0_6-seed_31337 |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_1-seed_0       |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_1-seed_1337    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_1-seed_2302    |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_gpt_oss_20b-temp_1-seed_31337   |      1 |      0 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0-seed_0       |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0-seed_1337    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0-seed_2302    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0-seed_31337   |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_2-seed_0     |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_2-seed_1337  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_2-seed_2302  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_2-seed_31337 |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_6-seed_0     |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_6-seed_1337  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_6-seed_2302  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_0_6-seed_31337 |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_1-seed_0       |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_1-seed_1337    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_count_customer_invoices | model_ollama_llama3_2_3b-temp_1-seed_31337   |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0-seed_0       |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0-seed_1337    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0-seed_2302    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0-seed_31337   |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_2-seed_0     |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_2-seed_1337  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_2-seed_2302  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_2-seed_31337 |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_6-seed_0     |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_6-seed_1337  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_6-seed_2302  |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_0_6-seed_31337 |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_1-seed_0       |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_1-seed_1337    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_1-seed_2302    |      0 |      1 |        1 |
| tests/test_agent_sql.py | test_find_top_3_customers    | model_ollama_llama3_2_3b-temp_1-seed_31337   |      0 |      1 |        1 |
| TOTAL                   |                              |                                              |     65 |     31 |       96 |

*To generate this table run `uv run pytest -k 'test_agent_sql' --md-report --md-report-verbose=2`*
