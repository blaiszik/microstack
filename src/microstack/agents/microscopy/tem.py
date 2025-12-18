"""TEM (Transmission Electron Microscopy) agent using abTEM for multislice simulation."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
from ase.io import read

from microstack.agents.state import WorkflowState
from microstack.utils.logging import get_logger
from microstack.utils.settings import settings
from microstack.io import save_tem_to_nsid, SIDPY_AVAILABLE

logger = get_logger("agents.microscopy.tem")


def run_tem_simulation(state: WorkflowState) -> WorkflowState:
    logger.info("Starting TEM simulation with abTEM")
    state.workflow_stage = "microscopy"

    try:
        if not state.has_structure():
            state.add_error("No structure found.")
            return state

        relaxed_xyz = state.file_paths.get("relaxed_xyz")
        from abtem import Potential, PlaneWave

        # 1. Load Atoms and Ensure Cell exists
        atoms = read(relaxed_xyz)
        if np.allclose(atoms.get_volume(), 0):
            atoms.center(vacuum=5.0)  # abTEM requires a bounding box/cell

        formula = atoms.get_chemical_formula()
        tem_dir = (
            Path(state.file_paths.get("structure_dir", ".")) / "microscopy" / "tem"
        )
        tem_dir.mkdir(parents=True, exist_ok=True)

        # 2. Setup Potential
        # Ensure gpts is a tuple to define a 2D grid
        gpts = getattr(settings, "tem_gpts", 256)
        if isinstance(gpts, int):
            gpts = (gpts, gpts)

        potential = Potential(
            atoms=atoms,
            gpts=gpts,
            sampling=getattr(settings, "tem_sampling", 0.1),
            slice_thickness=getattr(settings, "tem_slice_thickness", 1.0),
            device=getattr(settings, "tem_device", "cpu"),
        )

        # 3. Setup PlaneWave (TEM Mode)
        wave = PlaneWave(energy=getattr(settings, "tem_energy", 200) * 1e3)
        exit_wave = wave.multislice(potential)

        # 4. MEASUREMENT FIX: Explicitly request spatial intensity
        # exit_wave.intensity() is a measurement object.
        # We call .compute() to get the actual data array.
        measurement = exit_wave.intensity()
        image_data = measurement.compute()

        # Handle abTEM Measurement objects vs raw numpy arrays
        if hasattr(image_data, "array"):
            image_array = image_data.array
        else:
            image_array = np.array(image_data)

        # 5. SHAPE SAFETY CHECK
        logger.info(f"Calculated image shape: {image_array.shape}")

        # If still scalar, it's a simulation setup error (e.g., zero cell)
        if image_array.ndim < 2:
            logger.warning("Simulation produced scalar/1D data. Expanding to 2D.")
            image_array = np.atleast_2d(image_array)
            if image_array.shape == (1, 1):
                # Create a tiny 2x2 grid so matplotlib doesn't complain
                image_array = np.tile(image_array, (2, 2))

        # 6. VISUALIZATION (Avoiding abtem.show() to prevent IndexError)
        image_file = tem_dir / f"{formula}_tem_image.png"
        npy_file = tem_dir / f"{formula}_tem_image.npy"

        np.save(str(npy_file), image_array)

        # Export to NSID format if enabled
        nsid_file = None
        if getattr(settings, "export_nsid", True) and SIDPY_AVAILABLE:
            try:
                nsid_file = tem_dir / f"{formula}_tem.h5"
                save_tem_to_nsid(
                    filepath=nsid_file,
                    image_data=image_array,
                    sampling=getattr(settings, "tem_sampling", 0.1),
                    energy=getattr(settings, "tem_energy", 200),
                    metadata={"formula": formula},
                )
                logger.info(f"Saved NSID file: {nsid_file}")
            except Exception as e:
                logger.warning(f"NSID export failed: {e}")
        elif getattr(settings, "export_nsid", True) and not SIDPY_AVAILABLE:
            logger.warning("NSID export enabled but sidpy/pyNSID not installed")

        plt.figure(figsize=(8, 8))
        # Squeeze removes any singleton dimensions like (1, 256, 256)
        plt.imshow(image_array.squeeze(), cmap="gray", origin="lower")
        plt.title(f"TEM Simulation: {formula}")
        plt.colorbar(label="Intensity")
        plt.savefig(str(image_file), bbox_inches="tight")
        plt.close()

        # 7. Update State
        state.microscopy_results["tem"] = {
            "image_file": str(image_file),
            "data_file": str(npy_file),
            "image_shape": list(image_array.shape),
            "nsid_file": str(nsid_file) if nsid_file else None,
        }
        state.file_paths["microscopy"] = str(tem_dir)

        return state

    except Exception as e:
        logger.error(f"TEM simulation failed: {e}", exc_info=True)
        state.add_error(f"TEM simulation failed: {str(e)}")
        return state
