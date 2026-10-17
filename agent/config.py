"""Which LLM the agents use. Set with environment variables:

    LLM_PROVIDER = gemini (default) | openrouter
    LLM_MODEL    = model name (default: gemini-3.5-flash-lite)
    LLM_RPM      = requests per minute allowed (default 4, free-tier safe)

    LLM_TEMPERATURE = only for models that support it (Gemini 3.x ignores it)

Keys: GOOGLE_API_KEY for gemini, OPENROUTER_API_KEY for openrouter.
"""

import os

from langchain_core.rate_limiters import InMemoryRateLimiter


def make_llm():
    rpm = float(os.environ.get("LLM_RPM", "4"))
    limiter = InMemoryRateLimiter(requests_per_second=rpm / 60,
                                  check_every_n_seconds=0.5, max_bucket_size=1)
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        model = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        return ChatOpenAI(model=model, base_url="https://openrouter.ai/api/v1",
                          api_key=os.environ["OPENROUTER_API_KEY"], temperature=0,
                          rate_limiter=limiter, max_retries=6,
                          extra_body={"models": [model, "openrouter/free"]})

    from langchain_google_genai import ChatGoogleGenerativeAI
    # No `temperature`: Gemini 3.x models use fixed sampling and ignore it
    # (LangChain prints a UserWarning on every call if it is passed).
    # Set LLM_TEMPERATURE only for an older model that supports it.
    extra = {}
    if os.environ.get("LLM_TEMPERATURE"):
        extra["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    return ChatGoogleGenerativeAI(
        model=os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite"),
        rate_limiter=limiter, max_retries=6, **extra)
