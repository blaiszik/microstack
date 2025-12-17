"""Configuration for API keys and settings.

Set your API keys here or via environment variables.
Environment variables take precedence over values set in this file.
"""

import os

# =============================================================================
# API Keys
# =============================================================================

# Anthropic API key for Claude
# Get yours at: https://console.anthropic.com/
ANTHROPIC_API_KEY =  os.environ.get("ANTHROPIC_API_KEY", "")

# Materials Project API key
# Get yours at: https://materialsproject.org/api
MATERIALS_PROJECT_API_KEY = os.environ.get("MP_API_KEY", "")

# =============================================================================
# Model Settings
# =============================================================================

# Claude model to use for report generation
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# MACE model settings
MACE_DEVICE = None  # None = auto-detect (cuda if available, else cpu)
MACE_DTYPE = "float32"

# =============================================================================
# Analysis Settings
# =============================================================================

# Default relaxation steps
DEFAULT_RELAXATION_STEPS = 200

# Surfaces to support in MVP
SUPPORTED_METALS = ["Cu", "Pt", "Au", "Ag", "Ni", "Pd"]
SUPPORTED_2D = ["C", "MoS2"]  # C = graphene
SUPPORTED_FACES = ["100", "111", "110", "graphene", "2d"]

# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Check if required API keys are set."""
    warnings = []

    if not ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY not set - Claude discussion generation will be disabled")

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
