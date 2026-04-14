from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# OpenRouter pricing per 1M tokens (as of April 2026)
MODEL_PRICING = {
    "google/gemini-2.5-flash":          {"input": 0.15,  "output": 0.60},
    "google/gemini-2.5-flash-lite":     {"input": 0.075, "output": 0.30},
    "google/gemini-2.5-flash-preview":  {"input": 0.15,  "output": 0.60},
}
DEFAULT_PRICING = {"input": 0.15, "output": 0.60}


class CostTracker(BaseCallbackHandler):
    """
    LangChain callback that sums token usage across all LLM calls in a pipeline run.
    Attach to every llm.invoke() call via config={"callbacks": [tracker]}.
    """

    def __init__(self, model: str = ""):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        for generations in response.generations:
            for gen in generations:
                usage = getattr(gen.message, "usage_metadata", None) if hasattr(gen, "message") else None
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)
                    self.calls += 1

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "llm_calls": self.calls,
            "cost_usd": self.cost_usd,
        }
