"""Unified LLM client factory for selecting between Gemini, Anthropic, and DeepSeek."""

from typing import Union

from microstack.utils import config
from microstack.utils.logging import get_logger

logger = get_logger("llm.client")


def get_llm_client() -> Union:
    """
    Get LLM client based on configuration.

    Returns:
        GeminiClient, AnthropicClient, or DeepSeekClient depending on LLM_AGENT setting

    Raises:
        ValueError: If configured LLM agent is not recognized
        LLMConnectionError: If LLM client initialization fails
    """
    llm_agent = config.LLM_AGENT.lower()

    if llm_agent == "gemini":
        logger.info("Using Google Gemini LLM client")
        from microstack.llm.gemini import get_gemini_client

        return get_gemini_client()

    elif llm_agent == "anthropic":
        logger.info("Using Anthropic Claude LLM client")
        from microstack.llm.anthropic import get_anthropic_client

        return get_anthropic_client()

    elif llm_agent == "deepseek":
        logger.info("Using DeepSeek LLM client")
        from microstack.llm.deepseek import get_deepseek_client

        return get_deepseek_client()

    else:
        raise ValueError(
            f"Unknown LLM agent: {llm_agent}. "
            f"Supported options are 'gemini', 'anthropic', or 'deepseek'. "
            f"Set LLM_AGENT environment variable or config."
        )


def parse_query(user_query: str):
    """
    Parse a user query using the configured LLM client.

    Args:
        user_query: Natural language query from the user

    Returns:
        ParsedQuery object with extracted parameters
    """
    client = get_llm_client()
    return client.parse_query(user_query)
