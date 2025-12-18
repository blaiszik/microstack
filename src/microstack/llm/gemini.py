"""Google Gemini LLM wrapper for natural language query parsing."""

from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from microstack.utils.settings import settings
from microstack.llm.prompts import QUERY_PARSER_SYSTEM_PROMPT
from microstack.llm.models import ParsedQuery
from microstack.utils.exceptions import LLMConnectionError, QueryParsingError
from microstack.utils.logging import get_logger

logger = get_logger("llm.gemini")


class GeminiClient:
    """Client for Google Gemini LLM with structured output parsing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Google Gemini client.

        Args:
            api_key: Google API key (uses settings if None)
            model: Model name (uses settings if None)
        """
        self.api_key = api_key or getattr(settings, "google_api_key", None)
        self.model = model or getattr(
            settings, "gemini_model", "gemini-3-flash-preview"
        )

        if not self.api_key:
            raise LLMConnectionError(
                "Gemini",
                "API key not found. Please set GOOGLE_API_KEY in your environment.",
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini client with model: {self.model}")
        except Exception as e:
            raise LLMConnectionError("Gemini", str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def parse_query(self, user_query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured parameters using Gemini.

        Args:
            user_query: Natural language query from the user

        Returns:
            ParsedQuery object with extracted parameters

        Raises:
            QueryParsingError: If query parsing fails
        """
        logger.info(f"Parsing query with Gemini: {user_query}")

        try:
            import json

            # Create the prompt for Gemini
            parsing_prompt = f"""{QUERY_PARSER_SYSTEM_PROMPT}

User query: {user_query}

Extract the structured parameters and return ONLY a valid JSON object (no markdown, no extra text).
Use null for any missing values."""

            # Call Gemini API
            response = self.client.generate_content(parsing_prompt)

            # Extract the response text
            response_text = response.text.strip()

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

            # Fix ambiguities field - ensure it's always a list
            if (
                response_dict.get("ambiguities") is None
                or "ambiguities" not in response_dict
            ):
                response_dict["ambiguities"] = []
            elif isinstance(response_dict["ambiguities"], str):
                # Convert to list with single element if it's a string
                ambig_str = response_dict["ambiguities"]
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
                # Detect multiple microscopy types in order of appearance
                detected_types = []
                # Check for each type in the order they appear in the query
                for word in query_lower.split():
                    if word == "afm" and "afm" not in [
                        t.lower() for t in detected_types
                    ]:
                        detected_types.append("AFM")
                    elif word == "stm" and "stm" not in [
                        t.lower() for t in detected_types
                    ]:
                        detected_types.append("STM")
                    elif word == "iets" and "iets" not in [
                        t.lower() for t in detected_types
                    ]:
                        detected_types.append("IETS")
                    elif word == "tem" and "tem" not in [
                        t.lower() for t in detected_types
                    ]:
                        detected_types.append("TEM")

                if detected_types:
                    # If single type, keep as string; if multiple, keep as list
                    response_dict["microscopy_type"] = (
                        detected_types[0]
                        if len(detected_types) == 1
                        else detected_types
                    )
                    response_dict["task_type"] = "Microscopy_Simulation"

            # Keyword-based fallback for GPAW mode detection
            if not response_dict.get("stm_gpaw_mode") and response_dict.get(
                "microscopy_type"
            ):
                # Check if STM or IETS is requested
                micro_types = response_dict["microscopy_type"]
                if isinstance(micro_types, str):
                    micro_types = [micro_types]
                if micro_types and ("STM" in micro_types or "IETS" in micro_types):
                    # Look for GPAW mode keywords in the query
                    if "lcao" in query_lower:
                        response_dict["stm_gpaw_mode"] = "lcao"
                        logger.info(f"Detected GPAW mode: lcao")
                    elif "pw" in query_lower or "plane-wave" in query_lower:
                        response_dict["stm_gpaw_mode"] = "pw"
                        logger.info(f"Detected GPAW mode: pw")
                    elif "fd" in query_lower or "finite-difference" in query_lower:
                        response_dict["stm_gpaw_mode"] = "fd"
                        logger.info(f"Detected GPAW mode: fd")
                    # If IETS is requested and no mode specified, default to lcao
                    elif "iets" in query_lower.lower() and "IETS" in micro_types:
                        response_dict["stm_gpaw_mode"] = "lcao"
                        logger.info("Auto-selected GPAW mode: lcao (required for IETS)")

            # Ensure microscopy_type is either string or list, not mixed
            if isinstance(response_dict.get("microscopy_type"), list):
                # Keep as list for multiple simulations
                if len(response_dict["microscopy_type"]) == 1:
                    response_dict["microscopy_type"] = response_dict["microscopy_type"][
                        0
                    ]

            result = ParsedQuery(**response_dict)

            logger.info(f"Parsed task type: {result.task_type}")
            if result.material_formula:
                logger.info(f"Parsed material: {result.material_formula}")
            if result.ambiguities:
                logger.warning(f"Query has ambiguities: {result.ambiguities}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Response text: {response_text}")
            raise QueryParsingError(
                user_query, f"Invalid JSON response from Gemini: {e}"
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
        Ask Gemini to help clarify ambiguous options.

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
            response = self.client.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Clarification request failed: {e}")
            raise LLMConnectionError("Gemini", str(e))


# Global client instance
_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """
    Get or create the global Gemini client instance.

    Returns:
        Gemini client instance
    """
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
