"""Anthropic Claude LLM wrapper for natural language query parsing."""

from typing import Optional

from anthropic import Anthropic as AnthropicClient_SDK
from tenacity import retry, stop_after_attempt, wait_exponential

from microstack.utils.settings import settings
from microstack.llm.prompts import QUERY_PARSER_SYSTEM_PROMPT
from microstack.llm.models import ParsedQuery
from microstack.utils.exceptions import LLMConnectionError, QueryParsingError
from microstack.utils.logging import get_logger

logger = get_logger("llm.anthropic")


class AnthropicClient:
    """Client for Anthropic Claude LLM with structured output parsing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Anthropic Claude client.

        Args:
            api_key: Anthropic API key (uses settings if None)
            model: Model name (uses settings if None)
        """
        self.api_key = api_key or getattr(settings, "anthropic_api_key", None)
        self.model = model or getattr(
            settings, "claude_model", "claude-opus-4-20250805"
        )

        if not self.api_key:
            raise LLMConnectionError(
                "Anthropic",
                "API key not found. Please set ANTHROPIC_API_KEY in your environment.",
            )

        try:
            self.client = AnthropicClient_SDK(api_key=self.api_key)
            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except Exception as e:
            raise LLMConnectionError("Anthropic", str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def parse_query(self, user_query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured parameters using Claude.

        Args:
            user_query: Natural language query from the user

        Returns:
            ParsedQuery object with extracted parameters

        Raises:
            QueryParsingError: If query parsing fails
        """
        logger.info(f"Parsing query with Claude: {user_query}")

        try:
            import json

            # Create the prompt for Claude
            parsing_prompt = f"""{QUERY_PARSER_SYSTEM_PROMPT}

User query: {user_query}

Extract the structured parameters and return ONLY a valid JSON object (no markdown, no extra text).
Use null for any missing values."""

            # Call Claude API with JSON mode
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": parsing_prompt}],
            )

            # Extract the response text
            response_text = message.content[0].text.strip()

            # Try to parse JSON - handle markdown code blocks if present
            if response_text.startswith("```"):
                # Remove markdown code blocks
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            response_dict = json.loads(response_text)

            # Convert to ParsedQuery
            # Handle tuple conversions
            if response_dict.get("surface_miller_indices"):
                response_dict["surface_miller_indices"] = tuple(
                    response_dict["surface_miller_indices"]
                )
            if response_dict.get("scan_size"):
                response_dict["scan_size"] = tuple(response_dict["scan_size"])
            if response_dict.get("energy_range"):
                response_dict["energy_range"] = tuple(response_dict["energy_range"])

            # Set defaults for required fields - MUST be done before validation
            if response_dict.get("relax") is None or "relax" not in response_dict:
                response_dict["relax"] = True
            if (
                response_dict.get("use_scilink") is None
                or "use_scilink" not in response_dict
            ):
                response_dict["use_scilink"] = False

            # Set defaults for required fields
            if not response_dict.get("task_type"):
                # Infer task type from available parameters
                if response_dict.get("microscopy_type"):
                    response_dict["task_type"] = "Microscopy_Simulation"
                elif response_dict.get("supercell_x") or response_dict.get(
                    "material_formula"
                ):
                    response_dict["task_type"] = "SciLink_Structure_Generation"

            # Ensure confidence has a default
            if (
                "confidence" not in response_dict
                or response_dict.get("confidence") is None
            ):
                response_dict["confidence"] = 1.0

            # Fix ambiguities field if it's a string
            if response_dict.get("ambiguities") and isinstance(
                response_dict["ambiguities"], str
            ):
                ambig_str = response_dict["ambiguities"]
                # Convert to list with single element if it's a string
                response_dict["ambiguities"] = [ambig_str] if ambig_str else []

            # Fix missing_parameters field if it's a string
            if response_dict.get("missing_parameters") and isinstance(
                response_dict["missing_parameters"], str
            ):
                missing_str = response_dict["missing_parameters"]
                # Convert to list with single element if it's a string
                response_dict["missing_parameters"] = (
                    [missing_str] if missing_str else []
                )

            # Keyword-based fallback for microscopy type detection
            query_lower = user_query.lower()
            if not response_dict.get("microscopy_type"):
                if "iets" in query_lower:
                    response_dict["microscopy_type"] = "IETS"
                    response_dict["task_type"] = "Microscopy_Simulation"
                elif "stm" in query_lower:
                    response_dict["microscopy_type"] = "STM"
                    response_dict["task_type"] = "Microscopy_Simulation"
                elif "afm" in query_lower:
                    response_dict["microscopy_type"] = "AFM"
                    response_dict["task_type"] = "Microscopy_Simulation"

            result = ParsedQuery(**response_dict)

            logger.info(f"Parsed task type: {result.task_type}")
            if result.material_formula:
                logger.info(f"Parsed material: {result.material_formula}")
            if result.ambiguities:
                logger.warning(f"Query has ambiguities: {result.ambiguities}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            logger.debug(f"Response text: {response_text}")
            raise QueryParsingError(
                user_query, f"Invalid JSON response from Claude: {e}"
            )
        except Exception as e:
            logger.error(f"Query parsing failed: {e}", exc_info=True)
            raise QueryParsingError(user_query, str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def ask_clarification(
        self,
        context: str,
        options: list[str],
    ) -> str:
        """
        Ask Claude to help clarify ambiguous options.

        Args:
            context: Context about what needs clarification
            options: List of possible options

        Returns:
            Recommended option or explanation

        Raises:
            LLMConnectionError: If LLM call fails
        """
        prompt = f"""Context: {context}

Available options:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

Please recommend the most appropriate option and explain why briefly."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        except Exception as e:
            logger.error(f"Clarification request failed: {e}")
            raise LLMConnectionError("Anthropic", str(e))


# Global client instance
_client: Optional[AnthropicClient] = None


def get_anthropic_client() -> AnthropicClient:
    """
    Get or create the global Anthropic client instance.

    Returns:
        Anthropic client instance
    """
    global _client
    if _client is None:
        _client = AnthropicClient()
    return _client
