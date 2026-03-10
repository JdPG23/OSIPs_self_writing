"""
Unified LLM client abstraction.
Supports Anthropic (Claude) and OpenAI APIs.
"""

import os
from config import API_PROVIDER


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> tuple[str, int]:
    """Call an LLM and return (response_text, tokens_used).

    Dispatches to the appropriate provider based on config.API_PROVIDER.
    """
    provider = API_PROVIDER.lower()

    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, model, temperature, max_tokens)
    elif provider == "openai":
        return _call_openai(system_prompt, user_prompt, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown API provider: {provider}. Use 'anthropic' or 'openai'.")


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int]:
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    response_text = message.content[0].text
    tokens_used = message.usage.input_tokens + message.usage.output_tokens

    return response_text, tokens_used


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int]:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    response_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0

    return response_text, tokens_used
