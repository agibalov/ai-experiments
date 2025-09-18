from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from playwright.async_api import async_playwright
from debug_utils import pretty_agent_output
import pytest

@pytest.mark.asyncio
async def test_it_works():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        try:
            toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
            tools = toolkit.get_tools()

            model = init_chat_model("ollama:gpt-oss:20b", temperature=0, seed=31337, num_ctx=4096)
            agent = create_react_agent(
                model.bind_tools(tools), 
                tools)

            out = await agent.ainvoke({"messages": [
                #HumanMessage("What's the page title of https://example.com?")
                HumanMessage("In 2 sentences, what agibalov.io is about?")
                ]})
            pretty_agent_output(out)
            print(out["messages"][-1].content)
        
        finally:
            await ctx.close()
            await browser.close()
