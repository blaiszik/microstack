import os
import logging
import uuid  # New import
from typing import Dict, Any, Optional

from scilink.agents.sim_agents.structure_agent import StructureGenerator
from scilink.agents.sim_agents import structure_agent
from scilink.executors import DEFAULT_TIMEOUT
from microstack.utils.settings import settings
from microstack.utils.exceptions import LLMConnectionError
from microstack.llm.models import ParsedQuery

# Use the existing max_retries setting for SciLink code generation
# structure_agent.MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS = 15
MAX_SCILINK_RETRIES = 15

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
                mp_api_key=settings.mp_api_key,
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

        # Retry loop with refinement cycles enabled
        last_error = None
        for attempt in range(1, MAX_SCILINK_RETRIES + 1):
            is_refinement = attempt > 1  # Enable refinement after first attempt
            try:
                logger.info(
                    f"SciLink attempt {attempt}/{MAX_SCILINK_RETRIES} "
                    f"(refinement={is_refinement})"
                )
                gen_result = self.structure_generator.generate_script(
                    original_user_request=user_request,
                    attempt_number_overall=attempt,
                    is_refinement_from_validation=is_refinement,
                )

                if gen_result["status"] == "success":
                    final_structure_path = gen_result["output_file"]
                    logger.info(
                        f"SciLink successfully generated structure on attempt {attempt}: "
                        f"{final_structure_path}"
                    )
                    return {"status": "success", "file_path": final_structure_path}
                else:
                    last_error = gen_result.get(
                        "message", "Unknown error during SciLink structure generation."
                    )
                    logger.warning(
                        f"SciLink attempt {attempt}/{MAX_SCILINK_RETRIES} failed: {last_error}"
                    )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"SciLink attempt {attempt}/{MAX_SCILINK_RETRIES} raised exception: {e}"
                )

        # All attempts failed
        logger.error(
            f"SciLink structure generation failed after {MAX_SCILINK_RETRIES} attempts: {last_error}"
        )
        return {"status": "error", "message": last_error}
