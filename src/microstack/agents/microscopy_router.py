"""Microscopy router agent for LangGraph workflow."""

from microstack.agents.state import WorkflowState
from microstack.utils.logging import get_logger

logger = get_logger("agents.microscopy_router")


def check_microscopy(state: WorkflowState) -> WorkflowState:
    """
    Check if microscopy simulation is requested and handle interactive pause.
    Supports single or multiple microscopy simulations in execution order.

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state
    """
    logger.info("Checking if microscopy is requested")

    parsed_params = state.parsed_params

    # Check if this is the initial call (microscopy_queue is empty) or a subsequent call (looping back)
    is_first_call = len(state.microscopy_queue) == 0

    # Check if microscopy type was in the initial query
    if parsed_params and parsed_params.microscopy_type:
        logger.info(
            f"Microscopy type requested in query: {parsed_params.microscopy_type}"
        )
        state.microscopy_requested = True
        state.microscopy_type = parsed_params.microscopy_type

        # Only initialize the queue on the first call
        if is_first_call:
            # Handle both single microscopy type and list of types
            if isinstance(parsed_params.microscopy_type, list):
                # Multiple microscopy types requested - queue them in order
                state.microscopy_queue = list(parsed_params.microscopy_type)
                logger.info(
                    f"Multiple microscopy simulations queued in order: {state.microscopy_queue}"
                )
            else:
                # Single microscopy type requested
                state.microscopy_queue = [parsed_params.microscopy_type]
                logger.info(f"Single microscopy simulation queued: {parsed_params.microscopy_type}")
        else:
            logger.info(
                f"Looping back to check_microscopy after simulation, queue has {len(state.microscopy_queue)} items remaining"
            )

        # Set the current microscopy from the queue (already updated by check_next_microscopy)
        state.current_microscopy = state.microscopy_queue[0] if state.microscopy_queue else None
        state.interactive_pause = False  # Proceed automatically
    else:
        # If only structure was requested, set interactive pause
        logger.info("Only structure requested, setting interactive pause")
        state.microscopy_requested = False
        state.interactive_pause = True

    return state


def route_microscopy(state: WorkflowState) -> str:
    """
    Route to appropriate microscopy agent based on current microscopy type.
    Supports sequential execution of multiple microscopy simulations.

    Args:
        state: Workflow state object

    Returns:
        Name of next agent/node to execute
    """
    if not state.microscopy_requested:
        logger.info("No microscopy requested, ending workflow")
        return "end"

    # Use current_microscopy which is set from the queue
    microscopy_type = state.current_microscopy

    if not microscopy_type:
        logger.warning("No current microscopy type set, ending workflow")
        return "end"

    logger.info(f"Routing to microscopy agent: {microscopy_type}")

    if microscopy_type == "STM":
        return "stm_agent"
    elif microscopy_type == "AFM":
        return "afm_agent"
    elif microscopy_type == "IETS":
        return "iets_agent"
    elif microscopy_type == "TEM":
        return "tem_agent"
    else:
        logger.warning(f"Unknown microscopy type: {microscopy_type}")
        return "end"


def check_next_microscopy(state: WorkflowState) -> WorkflowState:
    """
    Check if there are more microscopy simulations in the queue and prepare for next execution.
    Called after each microscopy simulation completes.

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state with next microscopy type or None if queue is empty
    """
    if not state.microscopy_queue:
        logger.info("No more microscopy simulations in queue")
        state.microscopy_requested = False
        state.current_microscopy = None
        return state

    # Remove the completed microscopy from the queue
    completed = state.microscopy_queue.pop(0)
    logger.info(f"Completed microscopy: {completed}")

    if state.microscopy_queue:
        # Set the next microscopy to execute
        state.current_microscopy = state.microscopy_queue[0]
        logger.info(
            f"Next microscopy in queue: {state.current_microscopy} "
            f"({len(state.microscopy_queue)} remaining)"
        )
    else:
        # No more microscopy simulations
        logger.info("All microscopy simulations completed")
        state.microscopy_requested = False
        state.current_microscopy = None

    return state
