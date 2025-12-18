import os
import logging
import uuid  # New import
from typing import Dict, Any, Optional

from scilink.agents.sim_agents.structure_agent import StructureGenerator
from scilink.executors import DEFAULT_TIMEOUT
from atomic_materials.utils.settings import settings
from atomic_materials.utils.exceptions import LLMConnectionError
from atomic_materials.llm.models import ParsedQuery

logger = logging.getLogger(__name__)


class SciLinkIntegration:
    """
    Integrates SciLink's structure generation capabilities into the project.
    """

    def __init__(self, output_dir: str = "scilink_output"):
        """
        Initializes the SciLinkIntegration client.

        Args:
            output_dir: Directory to save generated structure files.
        """
        self.output_dir = os.path.join(settings.output_dir, "scilink_structures")
        os.makedirs(self.output_dir, exist_ok=True)

        google_api_key = settings.google_api_key
        if not google_api_key:
            raise LLMConnectionError(
                "Google",
                "API key not found. Please set GOOGLE_API_KEY in your environment for SciLink.",
            )

        try:
            self.structure_generator = StructureGenerator(
                api_key=google_api_key,
                model_name=settings.scilink_generator_model,
                executor_timeout=DEFAULT_TIMEOUT,
                generated_script_dir=self.output_dir,
                mp_api_key=settings.mp_api_key,  # Corrected to use settings.mp_api_key
            )
            logger.info("Initialized SciLink StructureGenerator.")
        except Exception as e:
            raise LLMConnectionError("SciLink StructureGenerator", str(e))

    def generate_surface_structure(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """
        Generates a surface structure using SciLink's StructureGenerator based on a parsed query.

        Args:
            parsed_query: A ParsedQuery object containing structure generation parameters.

        Returns:
            A dictionary containing the status and path to the generated structure file.
        """
        if parsed_query.task_type != "SciLink_Structure_Generation":
            return {
                "status": "error",
                "message": "Invalid task type for SciLink structure generation.",
            }

        material_formula = parsed_query.material_formula
        supercell_x = parsed_query.supercell_x or 1
        supercell_y = parsed_query.supercell_y or 1
        supercell_z = parsed_query.supercell_z or 1
        miller_indices = parsed_query.surface_miller_indices or (1, 0, 0)
        vacuum_thickness = parsed_query.vacuum_thickness or 15.0
        output_format = parsed_query.output_format or "xyz"

        if not all(
            [
                material_formula,
                supercell_x,
                supercell_y,
                supercell_z,
                miller_indices,
                vacuum_thickness,
            ]
        ):
            return {
                "status": "error",
                "message": "Missing required parameters for SciLink surface generation.",
            }

        user_request = f"{supercell_x}x{supercell_y}x{supercell_z} {material_formula}{miller_indices} surface with {vacuum_thickness}A vacuum. Save in {output_format} format."
        logger.info(f"SciLink structure generation request: {user_request}")

        try:
            gen_result = self.structure_generator.generate_script(
                original_user_request=user_request,
                attempt_number_overall=1,
                is_refinement_from_validation=False,
            )

            if gen_result["status"] == "success":
                final_structure_path = gen_result["output_file"]
                logger.info(
                    f"SciLink successfully generated structure: {final_structure_path}"
                )
                return {"status": "success", "file_path": final_structure_path}
            else:
                message = gen_result.get(
                    "message", "Unknown error during SciLink structure generation."
                )
                logger.error(f"SciLink structure generation failed: {message}")
                return {"status": "error", "message": message}
        except Exception as e:
            logger.error(
                f"Error during SciLink structure generation: {e}", exc_info=True
            )
            return {
                "status": "error",
                "message": f"Exception during SciLink generation: {e}",
            }
