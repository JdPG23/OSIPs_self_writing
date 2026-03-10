"""
Unified LLM client via OpenRouter (OpenAI-compatible API).
Also supports direct Anthropic/OpenAI as fallback.
"""

from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, int]:
    """Call an LLM via OpenRouter and return (response_text, tokens_used).

    Uses the OpenAI client pointed at OpenRouter's API.
    Model IDs follow OpenRouter format: "provider/model-name"
    (e.g., "anthropic/claude-sonnet-4.6", "deepseek/deepseek-v3.2")
    """
    if OPENROUTER_API_KEY:
        return _call_openrouter(system_prompt, user_prompt, model, temperature, max_tokens)
    else:
        # Fallback: try direct Anthropic if model starts with anthropic/
        if model.startswith("anthropic/"):
            return _call_anthropic_direct(system_prompt, user_prompt, model, temperature, max_tokens)
        else:
            raise ValueError(
                "OPENROUTER_API_KEY not set and model is not Anthropic. "
                "Set OPENROUTER_API_KEY in .env to use OpenRouter."
            )


def _call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int]:
    """Call LLM via OpenRouter (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    response = client.chat.completions.create(
        extra_headers={
            "X-OpenRouter-Title": "OSIP-Self-Writing",
        },
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    response_text = response.choices[0].message.content or ""
    tokens_used = response.usage.total_tokens if response.usage else 0

    return response_text, tokens_used


def _call_anthropic_direct(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int]:
    """Fallback: call Anthropic directly (if no OpenRouter key but have Anthropic key)."""
    import anthropic

    # Strip "anthropic/" prefix for direct API
    model_id = model.replace("anthropic/", "")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text
    tokens_used = message.usage.input_tokens + message.usage.output_tokens

    return response_text, tokens_used
