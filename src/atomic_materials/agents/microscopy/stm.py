"""STM microscopy agent using ASE/GPAW with full parameter control."""

from pathlib import Path
from typing import Optional
import sys
from io import StringIO
import os

import matplotlib.pyplot as plt

# Try to import GPAW
try:
    from ase.dft.stm import STM
    from gpaw import GPAW, restart

    GPAW_AVAILABLE = True
except ImportError:
    GPAW_AVAILABLE = False
    STM = None
    GPAW = None
    restart = None

from atomic_materials.agents.state import WorkflowState
from atomic_materials.utils.logging import get_logger
from atomic_materials.utils.settings import settings

logger = get_logger("agents.microscopy.stm")


def run_stm_simulation(state: WorkflowState) -> WorkflowState:
    """
    Run STM simulation using ASE/GPAW with full parameter control and GPU acceleration.

    Supports:
    - Constant current and constant height scans
    - Scanning Tunneling Spectroscopy (STS)
    - Full GPAW DFT configuration
    - Surface symmetries for LDOS acceleration
    - GPU acceleration via GPAW CUDA backend

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state with STM results
    """
    logger.info("Starting STM simulation with GPAW")
    state.workflow_stage = "microscopy"

    # Check if GPAW is available
    if not GPAW_AVAILABLE:
        logger.error("GPAW not available for STM simulation")
        state.add_error(
            "GPAW not available for STM simulation. Install gpaw to use STM."
        )
        return state

    try:
        # Check if a structure has been generated in this session
        if not state.has_structure():
            error_msg = "Structure has not been generated yet. Please generate a structure first before running STM simulation."
            logger.error(error_msg)
            state.add_error(error_msg)
            return state

        relaxed_xyz = state.file_paths.get("relaxed_xyz")
        if not relaxed_xyz:
            state.add_error("No relaxed structure file for STM simulation")
            return state

        # Load structure from file
        from ase.io import read

        atoms = read(relaxed_xyz)

        # Get STM parameters from parsed_params or settings
        parsed = state.parsed_params or {}

        # Helper function to get parameter with fallback chain
        def get_param(param_name, default_value=None):
            # Try parsed_params first (user-specified)
            if hasattr(parsed, param_name) and getattr(parsed, param_name) is not None:
                return getattr(parsed, param_name)
            # Then try settings
            if hasattr(settings, param_name):
                return getattr(settings, param_name)
            # Finally use provided default
            return default_value

        # Extract all STM parameters
        bias_voltage = get_param(
            "bias_voltage", get_param("stm_bias_voltage", settings.stm_bias_voltage)
        )
        tip_height = get_param(
            "tip_height", get_param("stm_tip_height", settings.stm_tip_height)
        )
        symmetries = get_param("stm_symmetries", settings.stm_symmetries)
        use_density = get_param("stm_use_density", settings.stm_use_density)
        repeat_x = get_param("stm_repeat_x", settings.stm_repeat_x)
        repeat_y = get_param("stm_repeat_y", settings.stm_repeat_y)
        z0 = get_param("stm_z0", settings.stm_z0)
        sts_bias_start = get_param("stm_sts_bias_start", settings.stm_sts_bias_start)
        sts_bias_end = get_param("stm_sts_bias_end", settings.stm_sts_bias_end)
        sts_bias_step = get_param("stm_sts_bias_step", settings.stm_sts_bias_step)
        sts_x = get_param("stm_sts_x", settings.stm_sts_x)
        sts_y = get_param("stm_sts_y", settings.stm_sts_y)
        gpaw_mode = get_param("stm_gpaw_mode", settings.stm_gpaw_mode)
        gpaw_kpts = get_param("stm_gpaw_kpts", settings.stm_gpaw_kpts)
        gpaw_symmetry = get_param("stm_gpaw_symmetry", settings.stm_gpaw_symmetry)
        gpaw_xc = get_param("stm_gpaw_xc", settings.stm_gpaw_xc)
        gpaw_h = get_param("stm_gpaw_h", settings.stm_gpaw_h)

        logger.info(
            f"STM parameters: bias={bias_voltage}V, height={tip_height}Å, symmetries={symmetries}"
        )
        logger.info(
            f"GPAW config: mode={gpaw_mode}, kpts={gpaw_kpts}, xc={gpaw_xc}, h={gpaw_h}Å"
        )

        structure_dir = Path(state.file_paths.get("structure_dir", "."))
        stm_dir = structure_dir / "microscopy" / "stm"
        stm_dir.mkdir(parents=True, exist_ok=True)

        # Run GPAW calculation with full parameter control
        logger.info("Running GPAW calculation for STM")
        formula = atoms.get_chemical_formula()
        gpw_file = stm_dir / f"calculation.gpw"

        # Check if GPAW file already exists
        if gpw_file.exists():
            logger.info(f"Loading existing GPAW file: {gpw_file}")
            atoms_calc, calc = restart(str(gpw_file), txt=None)
        else:
            logger.info("Creating new GPAW calculation with custom parameters")
            logger.info(
                f"GPAW settings: mode={gpaw_mode}, kpts={gpaw_kpts}, xc={gpaw_xc}, symmetry={gpaw_symmetry}, h={gpaw_h}"
            )

            # Create GPAW calculator with all parameters
            calc = GPAW(
                mode=gpaw_mode,
                kpts=gpaw_kpts,
                symmetry=gpaw_symmetry,
                xc=gpaw_xc,
                h=gpaw_h,
                txt=str(stm_dir / "gpaw.txt"),
            )

            atoms.calc = calc
            energy = atoms.get_potential_energy()
            logger.info(f"GPAW total energy: {energy:.4f} eV")
            calc.write(str(gpw_file), "all")
            atoms_calc = atoms

        # Generate STM images with all parameters
        logger.info(
            f"Generating STM images with symmetries={symmetries}, use_density={use_density}..."
        )
        stm = STM(atoms_calc, symmetries=symmetries, use_density=use_density)

        # Get averaged current
        c = stm.get_averaged_current(bias_voltage, tip_height)
        logger.info(f"Averaged current at z={tip_height}Å: {c:.6f}")

        # Constant current scan
        logger.info(
            f"Running constant current scan with repeat=({repeat_x}, {repeat_y})"
        )
        x, y, h = stm.scan(bias_voltage, c, z0=z0, repeat=(repeat_x, repeat_y))

        # Plot constant current
        plt.figure(figsize=(8, 6))
        plt.gca().axis("equal")
        plt.contourf(x, y, h, 40, cmap="gray")
        plt.colorbar(label="Height (Å)")
        plt.title(f"STM Constant Current (bias={bias_voltage}V)")
        cc_file = stm_dir / f"{formula}_stm_constant_current.png"
        plt.savefig(str(cc_file), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved constant current image: {cc_file}")

        # Constant height scan
        logger.info(
            f"Running constant height scan with repeat=({repeat_x}, {repeat_y})"
        )
        x, y, I = stm.scan2(bias_voltage, tip_height, repeat=(repeat_x, repeat_y))

        # Plot constant height
        plt.figure(figsize=(8, 6))
        plt.gca().axis("equal")
        plt.contourf(x, y, I, 40, cmap="gray")
        plt.colorbar(label="Current (A)")
        plt.title(f"STM Constant Height (bias={bias_voltage}V)")
        ch_file = stm_dir / f"{formula}_stm_constant_height.png"
        plt.savefig(str(ch_file), dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved constant height image: {ch_file}")

        # STS (Scanning Tunneling Spectroscopy)
        logger.info(f"Running STS at position ({sts_x}, {sts_y})")
        logger.info(
            f"STS bias range: {sts_bias_start} to {sts_bias_end} eV, step={sts_bias_step} eV"
        )
        sts_file = stm_dir / f"{formula}_stm_sts.png"

        try:
            # Suppress stdout during STS calculation to avoid printing bias values
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                bias, I, dIdV = stm.sts(
                    sts_x,
                    sts_y,
                    tip_height,
                    sts_bias_start,
                    sts_bias_end,
                    sts_bias_step,
                )
            finally:
                sys.stdout = old_stdout

            plt.figure(figsize=(8, 6))
            plt.plot(bias, I, label="I", linewidth=2)
            plt.plot(bias, dIdV, label="dIdV", linewidth=2)
            plt.xlim(sts_bias_start, sts_bias_end)
            plt.xlabel("Bias Voltage (V)")
            plt.ylabel("Current / Conductance")
            plt.title("Scanning Tunneling Spectroscopy (STS)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(str(sts_file), dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved STS spectrum: {sts_file}")
        except Exception as e:
            logger.warning(f"STS calculation failed: {e}")
            state.add_warning(f"STS failed: {str(e)}")

        # Store comprehensive STM results with all parameters
        state.microscopy_results["stm"] = {
            "bias_voltage": float(bias_voltage),
            "tip_height": float(tip_height),
            "symmetries": symmetries,
            "use_density": use_density,
            "repeat": (repeat_x, repeat_y),
            "z0": z0,
            "constant_current_file": str(cc_file),
            "constant_height_file": str(ch_file),
            "sts_file": str(sts_file) if sts_file.exists() else None,
            "sts_position": (float(sts_x), float(sts_y)),
            "sts_bias_range": (float(sts_bias_start), float(sts_bias_end)),
            "sts_bias_step": float(sts_bias_step),
            "gpaw_config": {
                "mode": gpaw_mode,
                "kpts": gpaw_kpts,
                "xc": gpaw_xc,
                "symmetry": gpaw_symmetry,
                "h": float(gpaw_h),
            },
            "gpaw_file": str(gpw_file),
        }

        state.file_paths["microscopy"] = str(stm_dir)
        logger.info("STM simulation complete")

        return state

    except ImportError as e:
        logger.error(f"GPAW not installed: {e}")
        state.add_error(f"GPAW not available: {str(e)}")
        return state
    except Exception as e:
        logger.error(f"STM simulation failed: {e}", exc_info=True)
        state.add_error(f"STM simulation failed: {str(e)}")
        return state
