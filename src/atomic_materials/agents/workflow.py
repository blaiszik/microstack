"""Main LangGraph workflow for ATOMIC multi-agent system."""

from langgraph.graph import StateGraph, END
from typing import Literal

from atomic_materials.agents.state import WorkflowState
from atomic_materials.llm.client import parse_query
from atomic_materials.agents.structure_generator import (
    generate_structure,
    relax_structure,
)
from atomic_materials.agents.microscopy_router import (
    check_microscopy,
    route_microscopy,
)

# Import microscopy agents
try:
    from atomic_materials.agents.microscopy.stm import run_stm_simulation
except ImportError:

    def run_stm_simulation(state):
        logger.warning("GPAW not available for STM simulation")
        state.add_error("GPAW not available for STM simulation")
        return state


from atomic_materials.utils.logging import get_logger

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


def stm_node(state: WorkflowState) -> WorkflowState:
    """Run STM simulation."""
    return run_stm_simulation(state)


def afm_node(state: WorkflowState) -> WorkflowState:
    """Run AFM simulation."""
    # Import here to avoid circular imports
    from atomic_materials.agents.microscopy.afm import run_afm_simulation

    return run_afm_simulation(state)


def iets_node(state: WorkflowState) -> WorkflowState:
    """Run IETS simulation."""
    # Import here to avoid circular imports
    from atomic_materials.agents.microscopy.iets import run_iets_simulation

    return run_iets_simulation(state)


def create_workflow() -> StateGraph:
    """
    Create the main LangGraph workflow.

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating ATOMIC LangGraph workflow")

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

    # Set entry point
    workflow.set_entry_point("parse_query")

    # Linear edges for parsing -> structure -> relaxation -> check
    workflow.add_edge("parse_query", "generate_structure")
    workflow.add_edge("generate_structure", "relax_structure")
    workflow.add_edge("relax_structure", "check_microscopy")

    # Conditional routing for microscopy
    workflow.add_conditional_edges(
        "check_microscopy",
        route_microscopy,
        {
            "stm_agent": "stm_agent",
            "afm_agent": "afm_agent",
            "iets_agent": "iets_agent",
            "end": END,
        },
    )

    # All microscopy agents terminate
    for agent in ["stm_agent", "afm_agent", "iets_agent"]:
        workflow.add_edge(agent, END)

    # Compile and return
    compiled_workflow = workflow.compile()
    logger.info("Workflow created successfully")

    return compiled_workflow


def run_workflow(query: str, session_id: str) -> WorkflowState:
    """
    Run the complete workflow.

    Args:
        query: User query string
        session_id: Session identifier

    Returns:
        Final workflow state
    """
    logger.info(f"Running workflow for session {session_id}: {query}")

    # Create initial state
    initial_state = WorkflowState(
        session_id=session_id,
        query=query,
    )

    # Create and run workflow
    workflow = create_workflow()
    final_state = workflow.invoke(initial_state)

    # Ensure we return a WorkflowState object (LangGraph might return dict)
    if isinstance(final_state, dict):
        final_state = WorkflowState(**final_state)

    logger.info(f"Workflow completed for session {session_id}")
    logger.info(f"Final state: {final_state.get_summary()}")

    return final_state
