import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

LLM_URL = os.environ.get("LLM_URL", "http://10.7.1.21/v1")
LLM_KEY = os.environ.get("LLM_KEY", "")

model = ChatOpenAI(
    model="qwen-35b",
    base_url=LLM_URL,
    api_key=LLM_KEY,
)

SEARCH_TOOL = DuckDuckGoSearchRun(name="web_search")

SYSTEM_PROMPT = """You are a helpful research assistant with web search capability.

Rules:
- Use `web_search` when you need current or factual information from the internet.
- Synthesize the search results into a clear, concise answer.
- Cite sources when possible.
- If the user's question is general knowledge you already know, you may answer directly without searching.
"""

agent = create_agent(
    model=model,
    tools=[SEARCH_TOOL],
    system_prompt=SYSTEM_PROMPT,
)


def ask(question: str) -> str:
    result = agent.invoke(
        {"messages": [("user", question)]},
        stream_mode="values",
    )
    last_message = result["messages"][-1]
    return last_message.content


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "How do you explain the double slit experiment?"
    print(ask(query))
