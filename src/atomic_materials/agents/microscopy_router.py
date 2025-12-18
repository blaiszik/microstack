"""Microscopy router agent for LangGraph workflow."""

from atomic_materials.agents.state import WorkflowState
from atomic_materials.utils.logging import get_logger

logger = get_logger("agents.microscopy_router")


def check_microscopy(state: WorkflowState) -> WorkflowState:
    """
    Check if microscopy simulation is requested and handle interactive pause.

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state
    """
    logger.info("Checking if microscopy is requested")

    parsed_params = state.parsed_params

    # Check if microscopy type was in the initial query
    if parsed_params and parsed_params.microscopy_type:
        logger.info(f"Microscopy type requested in query: {parsed_params.microscopy_type}")
        state.microscopy_requested = True
        state.microscopy_type = parsed_params.microscopy_type
        state.interactive_pause = False  # Proceed automatically
    else:
        # If only structure was requested, set interactive pause
        logger.info("Only structure requested, setting interactive pause")
        state.microscopy_requested = False
        state.interactive_pause = True

    return state


def route_microscopy(state: WorkflowState) -> str:
    """
    Route to appropriate microscopy agent based on microscopy type.

    Args:
        state: Workflow state object

    Returns:
        Name of next agent/node to execute
    """
    if not state.microscopy_requested:
        logger.info("No microscopy requested, ending workflow")
        return "end"

    microscopy_type = state.microscopy_type

    logger.info(f"Routing to microscopy agent: {microscopy_type}")

    if microscopy_type == "STM":
        return "stm_agent"
    elif microscopy_type == "AFM":
        return "afm_agent"
    elif microscopy_type == "IETS":
        return "iets_agent"
    else:
        logger.warning(f"Unknown microscopy type: {microscopy_type}")
        return "end"
