"""Structure generation agent for µStack workflow."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

from ase.io import write, read
from ase import Atoms

from microstack.agents.state import WorkflowState
from microstack.agents.structure_validator import (
    validate_structure,
    fix_structure_vacuum,
)
from microstack.relaxation.generate_surfaces import create_surface
from microstack.relaxation.surface_relaxation import (
    load_model,
    relax_surfaces,
    plot_surface_relaxation,
)
from microstack.utils import config
from microstack.utils.logging import get_logger
from microstack.llm.models import ParsedQuery

logger = get_logger("agents.structure_generator")


def _format_miller_indices(indices) -> str:
    """
    Convert Miller indices from various formats to simple string.

    Args:
        indices: Can be tuple (1,1,0), list [1,1,0], string "110", or None

    Returns:
        Simple string format like "110"
    """
    if indices is None:
        return "unknown"
    if isinstance(indices, str):
        # Remove any spaces or parentheses
        return (
            indices.replace("(", "").replace(")", "").replace(" ", "").replace(",", "")
        )
    if isinstance(indices, (list, tuple)):
        return "".join(str(i) for i in indices)
    return str(indices)


def generate_structure(state: WorkflowState) -> WorkflowState:
    """
    Generate atomic structure using either simple surfaces or SciLink.

    If a structure already exists in the session and no new structure is requested,
    reuse the existing structure instead of generating a new one.

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state with generated structure
    """
    logger.info(f"Starting structure generation for query: {state.query}")
    state.workflow_stage = "structure_generation"

    try:
        parsed_params = state.parsed_params

        # Check if query parsing succeeded
        if parsed_params is None:
            state.add_error("Query parsing failed, no parameters extracted")
            return state

        # Check if structure already exists in session and no new structure requested
        if state.atoms_object is not None and not parsed_params.material_formula:
            logger.info("Reusing existing structure from current session")
            state.add_warning(
                "Using existing structure from session (no new structure specified in query)"
            )
            return state

        # If no structure specified AND no atoms in state, raise error
        if not parsed_params.material_formula and state.atoms_object is None:
            error_msg = (
                "No structure found in session and no material specified in query. "
                "Please provide a structure specification (e.g., 'Al(111)', '2x2 Cu(100)') "
                "or continue from a previous session."
            )
            logger.error(error_msg)
            state.add_error(error_msg)
            return state

        # Determine if we should use SciLink
        use_scilink = (
            parsed_params.use_scilink
            if hasattr(parsed_params, "use_scilink")
            else False
        )

        atoms = None
        if use_scilink or parsed_params.task_type == "SciLink_Structure_Generation":
            logger.info("Using SciLink for structure generation")
            atoms = _generate_with_scilink(state)
            # If SciLink fails, fall back to Materials Project + ASE
            if atoms is None:
                logger.info("SciLink failed, falling back to Materials Project + ASE")
                atoms = _generate_with_materials_project(state)
            # If Materials Project fails, fall back to simple surface generation
            if atoms is None:
                logger.info(
                    "Materials Project failed, falling back to simple surface generation"
                )
                atoms = _generate_simple_surface(state)
        else:
            logger.info("Using simple surface generation")
            atoms = _generate_simple_surface(state)

        if atoms is None:
            state.add_error("Failed to generate structure")
            return state

        # Validate structure before proceeding to relaxation
        is_valid, validation_msg = validate_structure(atoms)

        # Retry full regeneration if structure is invalid
        max_regenerate_attempts = 3
        attempt = 1
        while not is_valid and attempt < max_regenerate_attempts:
            logger.warning(
                f"Structure validation failed (attempt {attempt}/{max_regenerate_attempts}): {validation_msg}"
            )
            state.add_warning(
                f"Structure validation failed: {validation_msg}. Regenerating structure..."
            )

            # Regenerate the entire structure
            logger.info("Regenerating structure...")
            atoms = None

            # Try generation methods in order
            if use_scilink or parsed_params.task_type == "SciLink_Structure_Generation":
                atoms = _generate_with_scilink(state)
                if atoms is None:
                    atoms = _generate_with_materials_project(state)
                if atoms is None:
                    atoms = _generate_simple_surface(state)
            else:
                atoms = _generate_simple_surface(state)

            if atoms is None:
                logger.warning("Failed to regenerate structure, aborting")
                break

            # Re-validate
            is_valid, validation_msg = validate_structure(atoms)
            attempt += 1

        if not is_valid:
            error_msg = f"Structure validation failed after {max_regenerate_attempts} attempts: {validation_msg}"
            logger.error(error_msg)
            state.add_error(error_msg)
            return state

        logger.info(f"Structure validation passed: {validation_msg}")

        # Store atoms object and info
        state.atoms_object = atoms
        state.structure_uuid = state.session_id  # Use session_id as the structure UUID
        state.structure_info = {
            "element": parsed_params.material_formula or "unknown",
            "face": _format_miller_indices(
                getattr(parsed_params, "surface_miller_indices", None)
            ),
            "formula": atoms.get_chemical_formula(),
            "num_atoms": len(atoms),
        }
        logger.info(f"Structure assigned UUID: {state.structure_uuid}")

        # Save unrelaxed structure
        output_dir = (
            Path(config.OUTPUT_DIR)
            / f"{state.structure_info['element']}_{state.structure_info['face']}_{state.session_id}/relaxation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        unrelaxed_file = output_dir / f"{atoms.get_chemical_formula()}_unrelaxed.xyz"
        write(str(unrelaxed_file), atoms)
        state.file_paths["unrelaxed_xyz"] = str(unrelaxed_file)
        state.file_paths["output_dir"] = str(output_dir)
        state.file_paths["structure_dir"] = str(output_dir.parent)

        logger.info(f"Generated structure with {len(atoms)} atoms")

        return state

    except Exception as e:
        logger.error(f"Structure generation failed: {e}", exc_info=True)
        state.add_error(f"Structure generation failed: {str(e)}")
        return state


def _generate_with_scilink(state: WorkflowState) -> Optional[Atoms]:
    """
    Generate structure using SciLink.

    Args:
        state: Workflow state

    Returns:
        ASE Atoms object or None if generation fails
    """
    try:
        import logging
        from microstack.relaxation.scilink_integration import SciLinkIntegration
        from microstack.utils.exceptions import LLMConnectionError

        # Suppress verbose SciLink logging
        logging.getLogger("scilink").setLevel(logging.WARNING)
        logging.getLogger("root").setLevel(logging.WARNING)
        logging.getLogger("edison_client").setLevel(logging.WARNING)

        parsed_params = state.parsed_params
        logger.info("Generating structure with SciLink...")

        # Initialize SciLink integration
        try:
            scilink_client = SciLinkIntegration()
        except LLMConnectionError as e:
            logger.error(f"SciLink initialization failed: {e}")
            state.add_error(f"SciLink not available: {str(e)}")
            return None

        # Generate structure using SciLink
        result = scilink_client.generate_surface_structure(parsed_params)

        # Check if result is None or doesn't have expected structure
        if result is None or not isinstance(result, dict):
            logger.error("SciLink returned invalid result (None or not a dict)")
            return None

        if result["status"] != "success":
            logger.error(
                f"SciLink generation failed: {result.get('message', 'Unknown error')}"
            )
            return None

        # Load the generated structure file
        structure_file = result.get("file_path")
        if not structure_file or not Path(structure_file).exists():
            logger.error("SciLink did not generate a valid structure file")
            return None

        # Read structure from file
        atoms = read(structure_file)
        formula = atoms.get_chemical_formula()
        logger.info(f"Structure generated: {formula} ({len(atoms)} atoms)")

        # Clean up scilink_structures folder after loading structure
        import shutil

        scilink_output_dir = Path(config.OUTPUT_DIR) / "scilink_structures"
        if scilink_output_dir.exists():
            try:
                shutil.rmtree(scilink_output_dir)
            except Exception as e:
                logger.debug(f"Failed to clean up SciLink output directory: {e}")

        return atoms

    except Exception as e:
        logger.error(f"SciLink generation failed: {e}", exc_info=True)
        state.add_error(f"SciLink generation error: {str(e)}")
        return None


def _generate_with_materials_project(state: WorkflowState) -> Optional[Atoms]:
    """
    Generate structure using Materials Project API and ASE.

    Args:
        state: Workflow state

    Returns:
        ASE Atoms object or None if generation fails
    """
    try:
        from pymatgen.ext.matproj import MPRester
        from pymatgen.io.ase import AseAtomsAdaptor
        from ase.build import add_vacuum

        parsed_params = state.parsed_params
        logger.info("Generating structure with Materials Project + ASE...")

        # Get material specification from parsed params
        material_formula = parsed_params.material_formula
        material_id = parsed_params.material_id
        structure_source = parsed_params.structure_source

        if not (material_formula or material_id):
            logger.error(
                "No material formula or ID provided for Materials Project lookup"
            )
            return None

        # Initialize Materials Project API
        try:
            mpr = MPRester()
        except Exception as e:
            logger.error(f"Failed to initialize Materials Project API: {e}")
            state.add_warning(
                "Materials Project API not available (missing API key or network issue)"
            )
            return None

        # Fetch structure from Materials Project
        try:
            if material_id:
                # Use material ID if provided
                logger.info(f"Fetching structure for material ID: {material_id}")
                structure = mpr.get_structure_by_material_id(material_id)
            else:
                # Search by formula
                logger.info(f"Searching Materials Project for: {material_formula}")
                results = mpr.query(
                    criteria={"pretty_formula": material_formula},
                    properties=["structure", "energy_per_atom"],
                )

                if not results:
                    logger.error(f"No results found for formula: {material_formula}")
                    return None

                # Use the most stable structure (lowest energy per atom)
                structure = results[0]["structure"]
                logger.info(
                    f"Using most stable structure: {results[0]['pretty_formula']}"
                )

            # Convert PyMatGen structure to ASE Atoms
            atoms = AseAtomsAdaptor.get_atoms(structure)
            logger.info(f"Fetched bulk structure with {len(atoms)} atoms")

            # Generate surface from bulk structure
            face = getattr(parsed_params, "surface_miller_indices", None)
            if not face:
                face = (1, 0, 0)

            # Convert face to string format if needed
            if isinstance(face, (list, tuple)):
                face_str = "".join(str(i) for i in face)
            else:
                face_str = str(face)

            # Use ASE's surface builder to create surface
            from ase.build import surface as build_surface

            layers = 4  # Default number of layers
            vacuum = parsed_params.vacuum_thickness or parsed_params.vacuum_size or 15.0

            try:
                # Get lattice constant from bulk structure
                surface_atoms = build_surface(
                    atoms,
                    face,
                    layers,
                    vacuum=vacuum / 10,  # Convert to internal ASE units (0.1 nm)
                )
                logger.info(
                    f"Generated {face_str} surface with {len(surface_atoms)} atoms"
                )

                # Set periodic boundary conditions based on vacuum specification
                has_vacuum = vacuum and vacuum > 0
                if has_vacuum:
                    surface_atoms.set_pbc([True, True, False])
                    logger.info("Set PBC: [True, True, False] (slab with vacuum)")
                else:
                    surface_atoms.set_pbc([True, True, True])
                    logger.info("Set PBC: [True, True, True] (fully periodic)")

                return surface_atoms

            except Exception as e:
                logger.error(f"Failed to build surface from bulk structure: {e}")
                logger.info("Falling back to using bulk structure as-is")
                add_vacuum(atoms, vacuum / 10)
                atoms.set_pbc([True, True, True])
                return atoms

        except Exception as e:
            logger.error(f"Failed to fetch structure from Materials Project: {e}")
            state.add_warning(f"Materials Project fetch failed: {str(e)}")
            return None

    except ImportError:
        logger.warning("PyMatGen or Materials Project tools not available")
        state.add_warning("PyMatGen not available for Materials Project integration")
        return None
    except Exception as e:
        logger.error(f"Materials Project generation failed: {e}", exc_info=True)
        state.add_error(f"Materials Project generation error: {str(e)}")
        return None


def _generate_simple_surface(state: WorkflowState) -> Optional[Atoms]:
    """
    Generate simple FCC surface using existing create_surface function.

    Args:
        state: Workflow state

    Returns:
        ASE Atoms object or None if generation fails
    """
    try:
        parsed_params = state.parsed_params
        element = parsed_params.material_formula
        face = getattr(parsed_params, "surface_miller_indices", None)

        # Convert miller indices to face string if needed
        if isinstance(face, (list, tuple)):
            face = "".join(str(i) for i in face)

        if not element or not face:
            logger.error(f"Missing element or face: {element}, {face}")
            return None

        logger.info(f"Generating simple surface: {element}({face})")

        # Use existing create_surface function
        atoms, _ = create_surface(element, face, state.session_id)

        # Determine PBC settings based on vacuum specification
        # If vacuum is explicitly specified with a distance, use non-periodic in z (slab)
        # Otherwise, use fully periodic (bulk)
        has_vacuum = (
            hasattr(parsed_params, "vacuum_thickness")
            and parsed_params.vacuum_thickness is not None
            and parsed_params.vacuum_thickness > 0
        ) or (
            hasattr(parsed_params, "vacuum_size")
            and parsed_params.vacuum_size is not None
            and parsed_params.vacuum_size > 0
        )

        if has_vacuum:
            # Slab structure with vacuum: periodic in x,y, non-periodic in z
            atoms.set_pbc([True, True, False])
            logger.info("Set PBC: [True, True, False] (slab with vacuum)")
        else:
            # Fully periodic structure (bulk)
            atoms.set_pbc([True, True, True])
            logger.info("Set PBC: [True, True, True] (fully periodic)")

        logger.info(f"Simple surface generated: {atoms.get_chemical_formula()}")
        return atoms

    except Exception as e:
        logger.error(f"Simple surface generation failed: {e}", exc_info=True)
        state.add_error(f"Simple surface generation error: {str(e)}")
        return None


def _build_scilink_prompt(parsed_params: ParsedQuery, original_query: str) -> str:
    """
    Build a prompt for SciLink based on parsed parameters.

    Args:
        parsed_params: Parsed query parameters
        original_query: Original user query

    Returns:
        Prompt string for SciLink
    """
    # If user gave a specific query, use it directly
    if "3x" in original_query or "2x" in original_query or "4x" in original_query:
        # Looks like a structured description, use it
        return f"Using ASE build functions, {original_query}. Store result in 'atoms' variable."

    # Build from parsed parameters
    parts = []

    # Get supercell dimensions
    x = parsed_params.supercell_x or 1
    y = parsed_params.supercell_y or 1
    z = parsed_params.supercell_z or 1
    size = f"{x}x{y}x{z}"

    # Get surface face
    element = parsed_params.material_formula or "Cu"
    face = "111"
    if parsed_params.surface_miller_indices:
        face = "".join(str(i) for i in parsed_params.surface_miller_indices)

    # Get vacuum
    vacuum = parsed_params.vacuum_thickness or parsed_params.vacuum_size or 15.0

    # Build prompt
    prompt = f"Using ASE build functions, create a {size} {element}({face}) surface with {vacuum}A vacuum. Store result in 'atoms' variable."

    return prompt


def relax_structure(state: WorkflowState) -> WorkflowState:
    """
    Relax the generated structure using MACE model.

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state with relaxed structure
    """
    logger.info("Starting structure relaxation")
    state.workflow_stage = "relaxation"

    try:
        atoms = state.atoms_object
        if atoms is None:
            state.add_error("No structure to relax")
            return state

        # Check if relaxed structure already exists in session and no new structure was requested
        if state.atoms_relaxed is not None and not (
            state.parsed_params and state.parsed_params.material_formula
        ):
            logger.info("Reusing relaxed structure from current session")
            state.add_warning(
                "Using relaxed structure from session (no new structure specified)"
            )
            return state

        # Check if relaxation was requested
        should_relax = True
        if state.parsed_params and hasattr(state.parsed_params, "relax"):
            should_relax = state.parsed_params.relax

        if not should_relax:
            logger.info("Relaxation not requested, skipping relaxation step")
            state.workflow_stage = "structure_generation"
            return state

        # Load MACE model
        logger.info("Loading MACE model")
        model = load_model()

        # Relax structure
        logger.info(f"Relaxing structure ({config.DEFAULT_RELAXATION_STEPS} steps)")
        relaxed_surfaces, initial_energies, final_energies = relax_surfaces(
            [atoms], model, steps=config.DEFAULT_RELAXATION_STEPS
        )

        relaxed_atoms = relaxed_surfaces[0]
        init_e = initial_energies[0]
        final_e = final_energies[0]

        state.atoms_relaxed = relaxed_atoms
        state.relaxation_results = {
            "initial_energy": float(init_e),
            "final_energy": float(final_e),
            "energy_change": float(final_e - init_e),
        }

        # Save relaxed structure
        output_dir = Path(state.file_paths.get("output_dir", "."))
        formula = atoms.get_chemical_formula()
        relaxed_file = output_dir / f"{formula}_relaxed.xyz"
        write(str(relaxed_file), relaxed_atoms)
        state.file_paths["relaxed_xyz"] = str(relaxed_file)

        # Generate visualization
        viz_file = output_dir / f"{formula}_relaxation.png"
        plot_surface_relaxation(
            [atoms], [relaxed_atoms], [formula], filename=str(viz_file)
        )
        state.file_paths["visualization"] = str(viz_file)

        logger.info(
            f"Relaxation complete: {init_e:.4f} → {final_e:.4f} eV "
            f"(Δ = {final_e - init_e:.4f} eV)"
        )

        return state

    except Exception as e:
        logger.error(f"Relaxation failed: {e}", exc_info=True)
        state.add_error(f"Relaxation failed: {str(e)}")
        return state
