"""Session management for maintaining workflow state across queries."""

from typing import Optional, Dict
from microstack.agents.state import WorkflowState
from microstack.utils.logging import get_logger

logger = get_logger("agents.session_manager")

# Global session cache - stores WorkflowState objects keyed by session_id
_SESSION_CACHE: Dict[str, WorkflowState] = {}


def get_session_state(session_id: str) -> Optional[WorkflowState]:
    """
    Retrieve a stored session state.

    Args:
        session_id: Session identifier

    Returns:
        WorkflowState object or None if session not found
    """
    if session_id in _SESSION_CACHE:
        logger.info(f"Found existing session state: {session_id}")
        return _SESSION_CACHE[session_id]
    logger.info(f"No existing session state found: {session_id}")
    return None


def save_session_state(session_id: str, state: WorkflowState) -> None:
    """
    Save or update a session state.

    Args:
        session_id: Session identifier
        state: WorkflowState object to save
    """
    _SESSION_CACHE[session_id] = state
    logger.info(f"Saved session state: {session_id}")


def clear_session(session_id: str) -> None:
    """
    Clear a session state.

    Args:
        session_id: Session identifier to clear
    """
    if session_id in _SESSION_CACHE:
        del _SESSION_CACHE[session_id]
        logger.info(f"Cleared session state: {session_id}")


def list_sessions() -> list[str]:
    """
    List all active session IDs.

    Returns:
        List of session IDs with stored states
    """
    return list(_SESSION_CACHE.keys())


def get_session_summary(session_id: str) -> Dict[str, str]:
    """
    Get a summary of the session state.

    Args:
        session_id: Session identifier

    Returns:
        Dictionary with session info
    """
    state = get_session_state(session_id)
    if state is None:
        return {}

    return {
        "session_id": session_id,
        "formula": state.structure_info.get("formula", "unknown"),
        "atoms": str(state.structure_info.get("num_atoms", "0")),
        "workflow_stage": state.workflow_stage,
        "relaxed": "Yes" if state.atoms_relaxed else "No",
    }
