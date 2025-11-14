from openai import OpenAI
from .inference_auth_token import get_access_token

DEFAULT_MODEL = "openai/gpt-oss-120b"

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_access_token(),
            base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        )
    return _client


def call_llm(
    messages=None,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    # optional high-level interface:
    system_prompt: str | None = None,
    user_prompt: str | None = None,
):
    """
    Unified LLM call helper.

    Usage patterns supported:

      1) Low-level (explicit messages):
         call_llm(
             messages=[{"role": "system", "content": "..."},
                       {"role": "user", "content": "..."}],
             model="openai/gpt-oss-120b",
             temperature=0.2,
             max_tokens=2048,
         )

      2) High-level (system/user prompts – as used in codegen.py):
         call_llm(
             system_prompt="You are ...",
             user_prompt="Generate Python code ...",
             model="openai/gpt-oss-120b",
             temperature=0.2,
             max_tokens=2048,
         )
    """
    # If explicit messages are not provided, build them from system/user prompts.
    if messages is None:
        if system_prompt is None and user_prompt is None:
            raise ValueError(
                "call_llm requires either `messages` or (`system_prompt` and `user_prompt`)."
            )

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})

    client = _get_client()
    chosen_model = model or DEFAULT_MODEL

    resp = client.chat.completions.create(
        model=chosen_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content