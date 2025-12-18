"""LangGraph state definition for µ-Stack workflow."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from ase import Atoms

from microstack.llm.models import ParsedQuery


class WorkflowState(BaseModel):
    """State object for the µ-Stack LangGraph workflow.

    Tracks all information through the workflow lifecycle from query parsing
    to structure generation to microscopy simulations.
    """

    # Session information
    session_id: str = Field(
        description="Unique session identifier (UUID, typically 8-char hex)"
    )
    query: str = Field(description="Original user query")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this workflow started"
    )

    # Parsed query information
    parsed_params: Optional[ParsedQuery] = Field(
        default=None, description="Parsed query parameters from LLM"
    )

    # Structure information
    structure_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about generated structure: element, face, formula, num_atoms, etc.",
    )
    structure_uuid: Optional[str] = Field(
        default=None,
        description="UUID of the generated structure for reference across operations in this session",
    )
    atoms_object: Optional[Atoms] = Field(
        default=None,
        description="ASE Atoms object for the current structure (unrelaxed)",
    )
    atoms_relaxed: Optional[Atoms] = Field(
        default=None, description="ASE Atoms object for the relaxed structure"
    )

    # File paths
    file_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths to generated files: unrelaxed_xyz, relaxed_xyz, visualization, etc.",
    )

    # Workflow progress
    workflow_stage: str = Field(
        default="parsing",
        description="Current stage: parsing, structure_generation, relaxation, microscopy, complete",
    )
    microscopy_requested: bool = Field(
        default=False, description="Whether microscopy simulation was requested"
    )
    microscopy_type: Optional[str] = Field(
        default=None, description="Type of microscopy to run (STM, AFM, IETS)"
    )
    interactive_pause: bool = Field(
        default=False, description="Whether workflow should pause for user interaction"
    )
    user_response: Optional[str] = Field(
        default=None, description="User's response to interactive pause (yes/no)"
    )

    # Microscopy results
    microscopy_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from microscopy simulations: images, data, parameters, etc.",
    )

    # Relaxation results
    relaxation_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from structure relaxation: initial_energy, final_energy, energy_change",
    )

    # Errors and warnings
    errors: List[str] = Field(
        default_factory=list, description="Any errors encountered during workflow"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings encountered (e.g., missing parameters, using defaults)",
    )

    class Config:
        """Allow arbitrary types like Atoms objects."""

        arbitrary_types_allowed = True

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def has_structure(self) -> bool:
        """Check if a structure has been generated in this session."""
        return self.structure_uuid is not None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow state."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.workflow_stage,
            "structure_info": self.structure_info,
            "microscopy_requested": self.microscopy_requested,
            "microscopy_type": self.microscopy_type,
            "has_errors": self.has_errors(),
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
        }
