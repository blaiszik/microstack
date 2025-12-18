"""IETS microscopy agent using PPSTM (pyPPSTM) with GPAW DFT."""

from pathlib import Path
import os

from atomic_materials.agents.state import WorkflowState
from atomic_materials.utils.logging import get_logger
from atomic_materials.utils.settings import settings

logger = get_logger("agents.microscopy.iets")


def run_iets_simulation(state: WorkflowState) -> WorkflowState:
    """
    Run IETS simulation using pyPPSTM with GPAW on relaxed structure.

    Requires:
    - pyPPSTM package with GPAW support
    - GPAW DFT calculation file

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state with IETS results
    """
    logger.info("Starting IETS simulation")
    state.workflow_stage = "microscopy"

    try:
        # Check if a structure has been generated in this session
        if not state.has_structure():
            error_msg = "Structure has not been generated yet. Please generate a structure first before running IETS simulation."
            logger.error(error_msg)
            state.add_error(error_msg)
            return state

        relaxed_xyz = state.file_paths.get("relaxed_xyz")
        if not relaxed_xyz:
            state.add_error("No relaxed structure file for IETS simulation")
            return state

        # Get IETS parameters from parsed_params or settings
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

        # Extract all IETS parameters
        voltage = get_param("iets_voltage", settings.iets_voltage)
        work_function = get_param("iets_work_function", settings.iets_work_function)
        eta = get_param("iets_eta", settings.iets_eta)
        amplitude = get_param("iets_amplitude", settings.iets_amplitude)
        s_orbital = get_param("iets_s_orbital", settings.iets_s_orbital)
        px_orbital = get_param("iets_px_orbital", settings.iets_px_orbital)
        py_orbital = get_param("iets_py_orbital", settings.iets_py_orbital)
        pz_orbital = get_param("iets_pz_orbital", settings.iets_pz_orbital)
        dxz_orbital = get_param("iets_dxz_orbital", settings.iets_dxz_orbital)
        dyz_orbital = get_param("iets_dyz_orbital", settings.iets_dyz_orbital)
        dz2_orbital = get_param("iets_dz2_orbital", settings.iets_dz2_orbital)
        gpaw_file = get_param("iets_gpaw_file", settings.iets_gpaw_file)
        sample_orbs = get_param("iets_sample_orbs", settings.iets_sample_orbs)
        pbc = get_param("iets_pbc", settings.iets_pbc)
        fermi = get_param("iets_fermi", settings.iets_fermi)
        cut_min = get_param("iets_cut_min", settings.iets_cut_min)
        cut_max = get_param("iets_cut_max", settings.iets_cut_max)
        ncpu = get_param("iets_ncpu", settings.iets_ncpu)
        x_range = get_param("iets_x_range", settings.iets_x_range)
        y_range = get_param("iets_y_range", settings.iets_y_range)
        z_range = get_param("iets_z_range", settings.iets_z_range)
        charge_q = get_param("iets_charge_q", settings.iets_charge_q)
        stiffness_k = get_param("iets_stiffness_k", settings.iets_stiffness_k)
        effective_mass = get_param("iets_effective_mass", settings.iets_effective_mass)
        data_format = get_param("iets_data_format", settings.iets_data_format)
        plot_results = get_param("iets_plot_results", settings.iets_plot_results)

        # Also get basic parameters
        tip_height = get_param("tip_height", 5.0)
        energy_range = get_param("energy_range", (-2.5, 2.5))

        logger.info(
            f"IETS parameters: voltage={voltage}V, eta={eta}eV, sample_orbs={sample_orbs}"
        )
        logger.info(f"Grid: x={x_range}, y={y_range}, z={z_range}")

        structure_dir = Path(state.file_paths.get("structure_dir", "."))
        iets_dir = structure_dir / "microscopy" / "iets"
        iets_dir.mkdir(parents=True, exist_ok=True)

        # Set up OpenMP environment for parallelization
        if ncpu > 1:
            os.environ["OMP_NUM_THREADS"] = str(ncpu)
            logger.info(f"Enabled OpenMP parallelization with {ncpu} cores")

        # Try to import PPSTM for full IETS simulation
        try:
            import sys

            # Add root directory's PPSTM to path for pyPPSTM import
            root_dir = Path(__file__).parent.parent.parent.parent.parent
            ppstm_path = root_dir / "PPSTM"
            if ppstm_path.exists() and str(ppstm_path) not in sys.path:
                sys.path.insert(0, str(ppstm_path))
                logger.info(f"Added PPSTM path to sys.path: {ppstm_path}")

            import pyPPSTM as PS
            import pyPPSTM.ReadSTM as RS
            import ppafm.io as io

            logger.info("pyPPSTM found - running full IETS simulation")

            # Load GPAW DFT data
            if gpaw_file is None:
                gpaw_file = structure_dir / "microscopy/calculation.gpw"
                logger.warning(f"No GPAW file specified, using default: {gpaw_file}")

            if not Path(gpaw_file).exists():
                logger.warning(f"GPAW file not found: {gpaw_file}")
                state.add_warning(f"GPAW calculation file not found: {gpaw_file}")
                # Continue with framework setup even if file missing

            logger.info(f"Loading GPAW data from: {gpaw_file}")

            # Load DFT data (if file exists)
            if Path(gpaw_file).exists():
                eigEn, coefs, Ratin = RS.read_GPAW_all(
                    name=str(gpaw_file),
                    fermi=fermi,
                    orbs=sample_orbs,
                    pbc=pbc,
                    cut_min=cut_min,
                    cut_max=cut_max,
                )
                logger.info("GPAW data loaded successfully")

                # Generate tip grid
                logger.info("Generating tip position grid")
                tip_r = RS.mkSpaceGrid(
                    x_range[0],
                    x_range[1],
                    x_range[2],
                    y_range[0],
                    y_range[1],
                    y_range[2],
                    z_range[0],
                    z_range[1],
                    z_range[2],
                )

                # Run IETS_simple calculation
                logger.info("Running IETS_simple calculation")
                iets_result = PS.IETS_simple(
                    voltage,
                    work_function,
                    eta,
                    eigEn,
                    tip_r,
                    Ratin,
                    coefs,
                    orbs=sample_orbs,
                    s=s_orbital,
                    px=px_orbital,
                    py=py_orbital,
                    pz=pz_orbital,
                    dxz=dxz_orbital,
                    dyz=dyz_orbital,
                    dz2=dz2_orbital,
                    Amp=amplitude,
                )

                logger.info(f"IETS_simple completed! Result shape: {iets_result.shape}")

                # Save results
                iets_file = iets_dir / f"iets_results_{data_format}.{data_format}"
                if data_format == "npy":
                    import numpy as np

                    np.save(str(iets_file), iets_result)
                    logger.info(f"Saved IETS results to {iets_file}")

                state.microscopy_results["iets"] = {
                    "voltage": float(voltage),
                    "eta": float(eta),
                    "sample_orbs": sample_orbs,
                    "grid_x": x_range,
                    "grid_y": y_range,
                    "grid_z": z_range,
                    "orbital_coefficients": {
                        "s": float(s_orbital),
                        "px": float(px_orbital),
                        "py": float(py_orbital),
                        "pz": float(pz_orbital),
                        "dxz": float(dxz_orbital),
                        "dyz": float(dyz_orbital),
                        "dz2": float(dz2_orbital),
                    },
                    "results_file": str(iets_file),
                    "method": "IETS_simple",
                }
            else:
                logger.warning("Skipping IETS calculation (GPAW file not found)")
                state.add_warning(
                    "IETS calculation skipped - GPAW calculation file required"
                )

        except ImportError as e:
            logger.warning(f"pyPPSTM not available: {e}")
            logger.info(
                "Creating framework for IETS parameters without full simulation"
            )

            # Create parameter summary file even without full simulation
            iets_params = {
                "voltage": float(voltage),
                "work_function": float(work_function),
                "eta": float(eta),
                "amplitude": float(amplitude),
                "sample_orbs": sample_orbs,
                "grid": {
                    "x": list(x_range),
                    "y": list(y_range),
                    "z": list(z_range),
                },
                "orbital_coefficients": {
                    "s": float(s_orbital),
                    "px": float(px_orbital),
                    "py": float(py_orbital),
                    "pz": float(pz_orbital),
                    "dxz": float(dxz_orbital),
                    "dyz": float(dyz_orbital),
                    "dz2": float(dz2_orbital),
                },
                "charge_q": float(charge_q),
                "stiffness_k": float(stiffness_k),
                "effective_mass": float(effective_mass),
            }

            # Save parameter configuration
            import json

            params_file = iets_dir / "iets_parameters.json"
            with open(params_file, "w") as f:
                json.dump(iets_params, f, indent=2)
            logger.info(f"Saved IETS parameters to {params_file}")

            state.microscopy_results["iets"] = {
                "status": "IETS framework configured (pyPPSTM not available for full simulation)",
                "parameters_file": str(params_file),
                "note": "Install pyPPSTM to enable full IETS simulation",
            }
            state.add_warning(
                "IETS simulation framework configured but pyPPSTM not available. Install pyPPSTM for full simulation."
            )

        state.file_paths["microscopy"] = str(iets_dir)
        logger.info("IETS simulation complete")

        return state

    except Exception as e:
        logger.error(f"IETS simulation failed: {e}", exc_info=True)
        state.add_error(f"IETS simulation failed: {str(e)}")
        return state
