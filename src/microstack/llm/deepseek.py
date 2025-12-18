"""DeepSeek LLM wrapper for natural language query parsing."""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from tenacity import retry, stop_after_attempt, wait_exponential

from microstack.utils.settings import settings
from microstack.llm.prompts import QUERY_PARSER_SYSTEM_PROMPT
from microstack.llm.models import ParsedQuery
from microstack.utils.exceptions import LLMConnectionError, QueryParsingError
from microstack.utils.logging import get_logger
from microstack.relaxation.scilink_integration import SciLinkIntegration

logger = get_logger("llm.deepseek")


class DeepSeekClient:
    """Client for DeepSeek LLM with structured output parsing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (uses settings if None)
            model: Model name (uses settings if None)
        """
        self.api_key = api_key or settings.deepseek_api_key
        self.model = model or settings.deepseek_model

        if not self.api_key:
            raise LLMConnectionError(
                "DeepSeek",
                "API key not found. Please set DEEPSEEK_API_KEY in your environment.",
            )

        try:
            self.llm: BaseChatModel = ChatDeepSeek(
                model=self.model,
                api_key=self.api_key,
                timeout=settings.api_timeout,
                max_retries=settings.max_retries,
            )
            logger.info(f"Initialized DeepSeek client with model: {self.model}")
        except Exception as e:
            raise LLMConnectionError("DeepSeek", str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def parse_query(self, user_query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured parameters.

        Args:
            user_query: Natural language query from the user

        Returns:
            ParsedQuery object with extracted parameters

        Raises:
            QueryParsingError: If query parsing fails
        """
        logger.info(f"Parsing query: {user_query}")

        try:
            # Create structured output parser
            structured_llm = self.llm.with_structured_output(ParsedQuery)

            # Create messages with system prompt
            messages = [
                SystemMessage(content=QUERY_PARSER_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]

            # Parse the query
            result = structured_llm.invoke(messages)

            # Ensure list fields are properly initialized
            if result.ambiguities is None:
                result.ambiguities = []
            if result.missing_parameters is None:
                result.missing_parameters = []

            # Keyword-based fallback for microscopy type detection (including TEM)
            query_lower = user_query.lower()
            if not result.microscopy_type:
                detected_types = []
                for word in query_lower.split():
                    if word == "afm" and "afm" not in [t.lower() for t in detected_types]:
                        detected_types.append("AFM")
                    elif word == "stm" and "stm" not in [t.lower() for t in detected_types]:
                        detected_types.append("STM")
                    elif word == "iets" and "iets" not in [t.lower() for t in detected_types]:
                        detected_types.append("IETS")
                    elif word == "tem" and "tem" not in [t.lower() for t in detected_types]:
                        detected_types.append("TEM")

                if detected_types:
                    result.microscopy_type = (
                        detected_types[0] if len(detected_types) == 1 else detected_types
                    )

            # Keyword-based fallback for GPAW mode detection
            if not result.stm_gpaw_mode and result.microscopy_type:
                # Check if STM or IETS is requested
                micro_types = result.microscopy_type
                if isinstance(micro_types, str):
                    micro_types = [micro_types]
                if micro_types and ("STM" in micro_types or "IETS" in micro_types):
                    query_lower = user_query.lower()
                    # Look for GPAW mode keywords in the query
                    if "lcao" in query_lower:
                        result.stm_gpaw_mode = "lcao"
                        logger.info(f"Detected GPAW mode: lcao")
                    elif "pw" in query_lower or "plane-wave" in query_lower:
                        result.stm_gpaw_mode = "pw"
                        logger.info(f"Detected GPAW mode: pw")
                    elif "fd" in query_lower or "finite-difference" in query_lower:
                        result.stm_gpaw_mode = "fd"
                        logger.info(f"Detected GPAW mode: fd")
                    # If IETS is requested and no mode specified, default to lcao
                    elif "iets" in query_lower and "IETS" in micro_types:
                        result.stm_gpaw_mode = "lcao"
                        logger.info("Auto-selected GPAW mode: lcao (required for IETS)")

            logger.info(f"Parsed task type: {result.task_type}")
            if result.material_formula:
                logger.info(f"Parsed material: {result.material_formula}")
            if result.stm_gpaw_mode:
                logger.info(f"GPAW mode: {result.stm_gpaw_mode}")
            if result.ambiguities:
                logger.warning(f"Query has ambiguities: {result.ambiguities}")

            return result

        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
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
        Ask the LLM to help clarify ambiguous options.

        Args:
            context: Context about what needs clarification
            options: List of possible options

        Returns:
            Recommended option or explanation

        Raises:
            LLMConnectionError: If LLM call fails
        """
        prompt = f"""
        Context: {context}

        Available options:
        {chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

        Please recommend the most appropriate option and explain why briefly.
        """

        try:
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Clarification request failed: {e}")
            raise LLMConnectionError("DeepSeek", str(e))

        except Exception as e:
            logger.error(f"Clarification request failed: {e}")
            raise LLMConnectionError("DeepSeek", str(e))

    def generate_structure_with_scilink(self, parsed_query: ParsedQuery) -> dict:
        """
        Generates an atomic structure using SciLinkIntegration.

        Args:
            parsed_query: A ParsedQuery object with SciLink-specific parameters.

        Returns:
            A dictionary containing the status and path to the generated structure file.
        """
        logger.info(
            f"Attempting to generate structure with SciLink for query: {parsed_query.task_type}"
        )
        try:
            scilink_client = SciLinkIntegration(output_dir=str(settings.output_dir))
            result = scilink_client.generate_surface_structure(parsed_query)
            return result
        except LLMConnectionError as e:
            logger.error(f"SciLink LLM connection error: {e}")
            return {"status": "error", "message": f"SciLink LLM connection error: {e}"}
        except Exception as e:
            logger.error(
                f"Error during SciLink structure generation: {e}", exc_info=True
            )
            return {"status": "error", "message": f"SciLink generation failed: {e}"}


# Global client instance
_client: Optional[DeepSeekClient] = None


def get_deepseek_client() -> DeepSeekClient:
    """
    Get or create the global DeepSeek client instance.

    Returns:
        DeepSeek client instance
    """
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client
