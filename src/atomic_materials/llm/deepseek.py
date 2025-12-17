"""DeepSeek LLM wrapper for natural language query parsing."""

from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from atomic_materials.utils.settings import settings
from atomic_materials.llm.prompts import QUERY_PARSER_SYSTEM_PROMPT
from atomic_materials.utils.exceptions import LLMConnectionError, QueryParsingError
from atomic_materials.utils.logging import get_logger

logger = get_logger("llm.deepseek")


class ParsedQuery(BaseModel):
    """Structured representation of a parsed microscopy query."""

    microscopy_type: Literal["TEM", "AFM", "STM", "IETS", "TERS"] = Field(
        description="Type of microscopy simulation requested"
    )
    material_formula: Optional[str] = Field(
        default=None,
        description="Chemical formula or material identifier (e.g., 'Si', 'NaCl', 'graphene')",
    )
    material_id: Optional[str] = Field(
        default=None,
        description="Specific material database ID (e.g., 'mp-149', 'oqmd-12345')",
    )
    structure_source: Optional[Literal["materials_project", "oqmd", "local_file"]] = (
        Field(
            default=None,
            description="Source for structure data",
        )
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to local structure file (XYZ format)",
    )

    # TEM-specific parameters
    acceleration_voltage: Optional[float] = Field(
        default=None,
        description="Acceleration voltage in kV (TEM)",
    )
    defocus: Optional[float] = Field(
        default=None,
        description="Defocus value in nm (TEM)",
    )
    thickness: Optional[float] = Field(
        default=None,
        description="Sample thickness in nm (TEM)",
    )

    # AFM/STM-specific parameters
    tip_height: Optional[float] = Field(
        default=None,
        description="Tip height in Angstrom (AFM/STM)",
    )
    scan_size: Optional[tuple[float, float]] = Field(
        default=None,
        description="Scan size in nm (x, y) (AFM/STM/TERS)",
    )

    # STM-specific parameters
    bias_voltage: Optional[float] = Field(
        default=None,
        description="Bias voltage in V (STM)",
    )

    # IETS-specific parameters
    energy_range: Optional[tuple[float, float]] = Field(
        default=None,
        description="Energy range in meV (min, max) (IETS)",
    )

    # TERS-specific parameters
    laser_wavelength: Optional[float] = Field(
        default=None,
        description="Laser wavelength in nm (TERS)",
    )

    # General metadata
    confidence: float = Field(
        default=1.0,
        description="Confidence score for the parsing (0-1)",
        ge=0.0,
        le=1.0,
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="List of ambiguous elements in the query",
    )


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

            logger.info(f"Parsed microscopy type: {result.microscopy_type}")
            if result.material_formula:
                logger.info(f"Parsed material: {result.material_formula}")
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
