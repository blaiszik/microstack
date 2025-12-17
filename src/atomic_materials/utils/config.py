"""Configuration for API keys and settings.

Set your API keys here or via environment variables.
Environment variables take precedence over values set in this file.
"""

import os
from typing import Literal, Optional, Dict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Look for .env in the current directory and parent directories
    load_dotenv(override=False)  # Don't override existing env vars
except ImportError:
    pass  # python-dotenv not installed, will use system env vars only

# =============================================================================
# API Keys
# =============================================================================

# Anthropic API key for Claude (DEFAULT AGENT)
# Get yours at: https://console.anthropic.com/
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# DeepSeek API key for alternative LLM
# Get yours at: https://platform.deepseek.com/
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Materials Project API key
# Get yours at: https://materialsproject.org/api
MATERIALS_PROJECT_API_KEY = os.environ.get("MP_API_KEY", "")

# =============================================================================
# Model Settings
# =============================================================================

# LLM Agent to use: "anthropic" (default) or "deepseek"
# At least one API key must be configured
LLM_AGENT = os.environ.get("LLM_AGENT", "anthropic")

# Claude model to use (when LLM_AGENT=anthropic)
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# DeepSeek model to use (when LLM_AGENT=deepseek)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# MACE model settings
MACE_DEVICE = None  # None = auto-detect (cuda if available, else cpu)
MACE_DTYPE = "float32"

# =============================================================================
# Analysis Settings
# =============================================================================

# Default relaxation steps
DEFAULT_RELAXATION_STEPS = 200

# Surfaces to support
SUPPORTED_METALS = ["Cu", "Pt", "Au", "Ag", "Ni", "Pd", "Al", "Fe", "Ir", "Rh"]
SUPPORTED_2D = ["C", "MoS2", "WS2", "MoSe2", "WSe2"]  # C = graphene
SUPPORTED_FACES = ["100", "111", "110", "graphene", "2d"]

# =============================================================================
# Output Configuration
# =============================================================================

# Base output directory
OUTPUT_DIR = Path("./output")


def init_output_dirs():
    """Create the base output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# def create_task_output_dir(element: str, face: str, task_id: str, status: str) -> Path:
#     """
#     Generates and creates a unique output directory for a given task.
#     The directory structure will be: output/relaxation/{element}_{face}_{status}_{task_id}/

#     Args:
#         element: Chemical symbol (e.g., 'Cu', 'Pt').
#         face: Surface face (e.g., '100', '111').
#         task_id: Unique identifier for the task.
#         status: The status of the structure (e.g., 'unrelaxed', 'relaxed').

#     Returns:
#         The Path object for the created output directory.
#     """
#     # Note: Logic updated to match user request for nested relaxation folder
#     # or specific task isolation if needed.
#     task_specific_dir = OUTPUT_DIR / f"{element}_{face}_{status}_{task_id}/relaxation"
#     task_specific_dir.mkdir(parents=True, exist_ok=True)
#     return task_specific_dir


# =============================================================================
# Logging Configuration
# =============================================================================

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Log file path
LOG_FILE = Path(os.environ.get("LOG_FILE", "./atomic.log"))

# Enable console logging
LOG_TO_CONSOLE = os.environ.get("LOG_TO_CONSOLE", "True").lower() == "true"

# Enable file logging
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "True").lower() == "true"

# Enable debug mode (verbose logging, stack traces)
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"


# =============================================================================
# Validation
# =============================================================================


def validate_config():
    """Check if required API keys are set."""
    warnings = []

    if not ANTHROPIC_API_KEY and LLM_AGENT == "anthropic":
        warnings.append(
            "ANTHROPIC_API_KEY not set but LLM_AGENT is 'anthropic'. Discussion generation disabled."
        )

    if not DEEPSEEK_API_KEY and LLM_AGENT == "deepseek":
        warnings.append(
            "DEEPSEEK_API_KEY not set but LLM_AGENT is 'deepseek'. Discussion generation disabled."
        )

    if not MATERIALS_PROJECT_API_KEY:
        warnings.append("MP_API_KEY not set - Will use cached/literature data only")

    return warnings


def get_anthropic_client():
    """Get Anthropic client if API key is available."""
    if not ANTHROPIC_API_KEY:
        return None

    try:
        import anthropic

        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        print("Warning: anthropic package not installed")
        return None


def get_deepseek_client():
    """
    Get DeepSeek client if API key is available.
    Uses the OpenAI SDK as DeepSeek is API-compatible.
    """
    if not DEEPSEEK_API_KEY:
        return None

    try:
        from openai import OpenAI

        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except ImportError:
        print("Warning: openai package not installed (required for DeepSeek client)")
        return None


def get_mp_client():
    """Get Materials Project client if API key is available."""
    if not MATERIALS_PROJECT_API_KEY:
        return None

    try:
        from mp_api.client import MPRester

        return MPRester(MATERIALS_PROJECT_API_KEY)
    except ImportError:
        print("Warning: mp-api package not installed")
        return None
