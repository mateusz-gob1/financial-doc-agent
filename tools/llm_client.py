import os
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler


def get_llm(model: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        temperature=temperature,
    )


def get_langfuse_handler() -> CallbackHandler:
    # LangFuse v4: reads LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST from env
    return CallbackHandler()
