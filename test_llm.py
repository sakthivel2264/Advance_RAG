
import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0,
    max_tokens=200,
)

response = llm.invoke(
    "Explain what Retrieval-Augmented Generation is in three sentences."
)

print(response.content)
