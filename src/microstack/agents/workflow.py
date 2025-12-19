"""Main LangGraph workflow for µStack multi-agent system."""

from langgraph.graph import StateGraph, END
from typing import Literal

from microstack.agents.state import WorkflowState
from microstack.llm.client import parse_query
from microstack.agents.structure_generator import (
    generate_structure,
    relax_structure,
)
from microstack.agents.microscopy_router import (
    check_microscopy,
    route_microscopy,
    check_next_microscopy,
)

# Import microscopy agents
try:
    from microstack.agents.microscopy.stm import run_stm_simulation
except ImportError:

    def run_stm_simulation(state):
        logger.warning("GPAW not available for STM simulation")
        state.add_error("GPAW not available for STM simulation")
        return state


from microstack.utils.logging import get_logger

logger = get_logger("agents.workflow")


def parse_query_node(state: WorkflowState) -> WorkflowState:
    """Parse user query using LLM."""
    logger.info("Parsing query with LLM")
    try:
        parsed_params = parse_query(state.query)
        state.parsed_params = parsed_params
        state.workflow_stage = "parsed"
        logger.info(
            f"Query parsed: task_type={parsed_params.task_type}, microscopy={parsed_params.microscopy_type}"
        )
    except Exception as e:
        logger.error(f"Query parsing failed: {e}")
        state.add_error(f"Query parsing failed: {str(e)}")
    return state


def structure_generation_node(state: WorkflowState) -> WorkflowState:
    """Generate atomic structure."""
    return generate_structure(state)


def relaxation_node(state: WorkflowState) -> WorkflowState:
    """Relax the generated structure."""
    return relax_structure(state)


def microscopy_check_node(state: WorkflowState) -> WorkflowState:
    """Check if microscopy is requested."""
    return check_microscopy(state)


def check_next_microscopy_node(state: WorkflowState) -> WorkflowState:
    """Check if there are more microscopy simulations in the queue."""
    return check_next_microscopy(state)


def route_next_microscopy(state: WorkflowState) -> str:
    """
    Route to next microscopy agent or end workflow.
    Called after each microscopy simulation completes.
    """
    if state.microscopy_requested and state.current_microscopy:
        logger.info(
            f"More microscopy simulations queued, routing to next: {state.current_microscopy}"
        )
        return "route_microscopy_next"
    else:
        logger.info("All microscopy simulations completed")
        return "end"


def stm_node(state: WorkflowState) -> WorkflowState:
    """Run STM simulation."""
    return run_stm_simulation(state)


def afm_node(state: WorkflowState) -> WorkflowState:
    """Run AFM simulation."""
    # Import here to avoid circular imports
    from microstack.agents.microscopy.afm import run_afm_simulation

    return run_afm_simulation(state)


def iets_node(state: WorkflowState) -> WorkflowState:
    """Run IETS simulation."""
    # Import here to avoid circular imports
    from microstack.agents.microscopy.iets import run_iets_simulation

    return run_iets_simulation(state)


def tem_node(state: WorkflowState) -> WorkflowState:
    """Run TEM simulation."""
    # Import here to avoid circular imports
    from microstack.agents.microscopy.tem import run_tem_simulation

    return run_tem_simulation(state)


def create_workflow() -> StateGraph:
    """
    Create the main LangGraph workflow.

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating µStack LangGraph workflow")

    # Create StateGraph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("generate_structure", structure_generation_node)
    workflow.add_node("relax_structure", relaxation_node)
    workflow.add_node("check_microscopy", microscopy_check_node)
    workflow.add_node("stm_agent", stm_node)
    workflow.add_node("afm_agent", afm_node)
    workflow.add_node("iets_agent", iets_node)
    workflow.add_node("tem_agent", tem_node)
    workflow.add_node("check_next_microscopy", check_next_microscopy_node)

    # Set entry point
    workflow.set_entry_point("parse_query")

    # Linear edges for parsing -> structure -> relaxation -> check
    workflow.add_edge("parse_query", "generate_structure")
    workflow.add_edge("generate_structure", "relax_structure")
    workflow.add_edge("relax_structure", "check_microscopy")

    # Conditional routing for initial microscopy check
    workflow.add_conditional_edges(
        "check_microscopy",
        route_microscopy,
        {
            "stm_agent": "stm_agent",
            "afm_agent": "afm_agent",
            "iets_agent": "iets_agent",
            "tem_agent": "tem_agent",
            "end": END,
        },
    )

    # All microscopy agents go to check_next_microscopy
    for agent in ["stm_agent", "afm_agent", "iets_agent", "tem_agent"]:
        workflow.add_edge(agent, "check_next_microscopy")

    # Conditional routing after each microscopy simulation
    workflow.add_conditional_edges(
        "check_next_microscopy",
        route_next_microscopy,
        {
            "route_microscopy_next": "check_microscopy",
            "end": END,
        },
    )

    # Compile and return
    compiled_workflow = workflow.compile()
    logger.info("Workflow created successfully")

    return compiled_workflow


def run_workflow(query: str, session_id: str) -> WorkflowState:
    """
    Run the complete workflow.

    If a session with the same ID exists, loads and continues from that state.
    Otherwise, creates a new state.

    Args:
        query: User query string
        session_id: Session identifier

    Returns:
        Final workflow state
    """
    from microstack.agents.session_manager import (
        get_session_state,
        save_session_state,
    )

    logger.info(f"Running workflow for session {session_id}: {query}")

    # Try to load existing session state
    initial_state = get_session_state(session_id)

    if initial_state is None:
        # New session - create fresh state
        logger.info(f"Creating new session: {session_id}")
        initial_state = WorkflowState(
            session_id=session_id,
            query=query,
        )
    else:
        # Existing session - update query but keep structure/relaxation data
        logger.info(f"Continuing session: {session_id}")
        initial_state.query = query
        # Clear errors from previous query but keep structure data
        initial_state.errors = []
        initial_state.warnings = []

    # Create and run workflow
    workflow = create_workflow()
    final_state = workflow.invoke(initial_state, {"recursion_limit": 2500})

    # Ensure we return a WorkflowState object (LangGraph might return dict)
    if isinstance(final_state, dict):
        final_state = WorkflowState(**final_state)

    # Save session state for future queries
    save_session_state(session_id, final_state)

    logger.info(f"Workflow completed for session {session_id}")
    logger.info(f"Final state: {final_state.get_summary()}")

    return final_state
