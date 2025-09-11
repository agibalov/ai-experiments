from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
import sqlite3

from tabulate import tabulate

from expert import assert_text_entails_claim

SCHEMA = """
        CREATE TABLE persons (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER
        );
""" 

def make_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()
    cur.executescript(SCHEMA)
    cur.executemany("INSERT INTO persons(id, name, age) VALUES (?,?,?)",
                    [(1, "Alice", 25),
                    (2, "Bob", 35),
                    (3, "Charlie", 45)])
    return conn    

def test_sql_agent_works():
    conn = make_db()

    def run_sql(sql: str) -> str:
        print(f"Running SQL: {sql}")
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return tabulate(rows, headers=cols, tablefmt="github")
        except Exception as e:
            print(e)
            return f"Error: {e}"
    
    memory = MemorySaver()
    model = init_chat_model("ollama:llama3.2:3b")
    tools = [Tool.from_function(run_sql, name="run_sql", description="Run SQL query against the database")]
    agent = create_react_agent(
        model.bind_tools(tools), 
        tools, 
        checkpointer=memory, 
        prompt=f"""
        You are an agent providing a human interface to an SQL database.
        
        You can use the tool to run SQL queries against the database. You can use the tool multiple 
        times. When you use the tool, always use a SELECT statement to query the database. Never 
        use INSERT, UPDATE, DELETE, or other statements that modify the database. Only use the 
        columns that are present in the schema. In 'select' queries, always explicitly list the 
        columns you want to select. Never use '*'. Never use parameter binding. Always interpolate 
        the values directly into the query.
        
        Whenever possible, use SQL for analytics. 
        
        Correct example:
        Question: "How many apples in the basket?"
        SQL to run: "select count(*) from basket"

        Incorrect example:
        Question: "How many apples in the basket?"
        SQL to run: "select apple from basket"

        Answer only the question that user asked, do not extend it with extra information user did 
        not ask for.

        Database schema:
        {SCHEMA}
        """)

    out = agent.invoke(
        {"messages": [HumanMessage("How many persons are older than 30?")]},
        config={"configurable": {"thread_id": "andrey-1"}}
    )
    response = out["messages"][-1].content
    print(response)
    assert_text_entails_claim(response, "mentions there are 2 persons")
