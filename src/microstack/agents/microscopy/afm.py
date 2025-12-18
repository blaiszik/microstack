"""AFM microscopy agent using ppafm (requires OpenCL)."""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from microstack.agents.state import WorkflowState
from microstack.utils.logging import get_logger
from microstack.utils.settings import settings

logger = get_logger("agents.microscopy.afm")


def run_afm_simulation(state: WorkflowState) -> WorkflowState:
    """
    Run AFM simulation using ppafm on relaxed structure.

    Args:
        state: Workflow state object

    Returns:
        Updated workflow state with AFM results
    """
    logger.info("Starting AFM simulation")
    state.workflow_stage = "microscopy"

    try:
        # Check if a structure has been generated in this session
        if not state.has_structure():
            error_msg = "Structure has not been generated yet. Please generate a structure first before running AFM simulation."
            logger.error(error_msg)
            state.add_error(error_msg)
            return state

        relaxed_xyz = state.file_paths.get("relaxed_xyz")
        if not relaxed_xyz:
            state.add_error("No relaxed structure file for AFM simulation")
            return state

        # Try to import ppafm with OpenCL support
        try:
            from ppafm.io import loadXYZ
            from ppafm.ml.AuxMap import (
                AtomicDisks,
                ESMapConstant,
                HeightMap,
                vdwSpheres,
            )
            from ppafm.ocl.AFMulator import AFMulator

            PPAFM_AVAILABLE = True
        except (ImportError, OSError) as e:
            # ppafm or OpenCL not available
            logger.warning(f"ppafm not available or OpenCL not configured: {e}")
            PPAFM_AVAILABLE = False

        # Get AFM parameters from parsed_params or settings
        parsed = state.parsed_params or {}

        # Helper function to get parameter with fallback chain: parsed_params -> settings -> default
        def get_param(param_name, default_value=None):
            # Try parsed_params first (user-specified)
            if hasattr(parsed, param_name) and getattr(parsed, param_name) is not None:
                return getattr(parsed, param_name)
            # Then try settings
            if hasattr(settings, param_name):
                return getattr(settings, param_name)
            # Finally use provided default
            return default_value

        # Extract parameters
        pix_per_angstrome = get_param(
            "afm_pix_per_angstrome", settings.afm_pix_per_angstrome
        )
        scan_dim = get_param("afm_scan_dim", settings.afm_scan_dim)
        scan_window_min = get_param("afm_scan_window_min", settings.afm_scan_window_min)
        scan_window_max = get_param("afm_scan_window_max", settings.afm_scan_window_max)
        i_zpp = get_param("afm_i_zpp", settings.afm_i_zpp)
        qs = get_param("afm_qs", settings.afm_qs)
        qzs = get_param("afm_qzs", settings.afm_qzs)
        sigma = get_param("afm_sigma", settings.afm_sigma)
        a_pauli = get_param("afm_a_pauli", settings.afm_a_pauli)
        b_pauli = get_param("afm_b_pauli", settings.afm_b_pauli)
        fdbm_vdw_type = get_param("afm_fdbm_vdw_type", settings.afm_fdbm_vdw_type)
        d3_params = get_param("afm_d3_params", settings.afm_d3_params)
        lj_vdw_damp = get_param("afm_lj_vdw_damp", settings.afm_lj_vdw_damp)
        df_steps = get_param("afm_df_steps", settings.afm_df_steps)
        tip_r0 = get_param("afm_tip_r0", settings.afm_tip_r0)
        tip_stiffness = get_param("afm_tip_stiffness", settings.afm_tip_stiffness)
        npbc = get_param("afm_npbc", settings.afm_npbc)
        f0_cantilever = get_param("afm_f0_cantilever", settings.afm_f0_cantilever)
        k_cantilever = get_param("afm_k_cantilever", settings.afm_k_cantilever)
        colorscale = get_param("afm_colorscale", settings.afm_colorscale)
        minimize_memory = get_param("afm_minimize_memory", settings.afm_minimize_memory)

        # Also get basic parameters
        tip_height = get_param("tip_height", 5.0)
        scan_size = get_param("scan_size", (10.0, 10.0))

        logger.info(
            f"AFM parameters: pixPerAngstrome={pix_per_angstrome}, scan_dim={scan_dim}, iZPP={i_zpp}"
        )

        structure_dir = Path(state.file_paths.get("structure_dir", "."))
        afm_dir = structure_dir / "microscopy" / "afm"
        afm_dir.mkdir(parents=True, exist_ok=True)

        # Check if ppafm is available
        if not PPAFM_AVAILABLE:
            logger.warning(
                "ppafm with OpenCL support not available. Saving AFM configuration without running simulation."
            )

            # Save AFM configuration as JSON for later use
            afm_config = {
                "status": "pending",
                "reason": "ppafm or OpenCL not available",
                "parameters": {
                    "pix_per_angstrome": pix_per_angstrome,
                    "scan_dim": scan_dim,
                    "scan_window_min": scan_window_min,
                    "scan_window_max": scan_window_max,
                    "i_zpp": i_zpp,
                    "qs": qs if isinstance(qs, list) else qs.tolist(),
                    "qzs": qzs if isinstance(qzs, list) else qzs.tolist(),
                    "sigma": sigma,
                    "a_pauli": a_pauli,
                    "b_pauli": b_pauli,
                    "fdbm_vdw_type": fdbm_vdw_type,
                    "d3_params": d3_params,
                    "lj_vdw_damp": lj_vdw_damp,
                    "df_steps": df_steps,
                    "tip_r0": (
                        tip_r0 if isinstance(tip_r0, (list, tuple)) else list(tip_r0)
                    ),
                    "tip_stiffness": (
                        tip_stiffness
                        if isinstance(tip_stiffness, (list, tuple))
                        else list(tip_stiffness)
                    ),
                    "npbc": npbc if isinstance(npbc, (list, tuple)) else list(npbc),
                    "f0_cantilever": f0_cantilever,
                    "k_cantilever": k_cantilever,
                    "colorscale": colorscale,
                    "minimize_memory": minimize_memory,
                    "tip_height": tip_height,
                    "scan_size": (
                        scan_size
                        if isinstance(scan_size, (list, tuple))
                        else list(scan_size)
                    ),
                },
            }
            config_file = afm_dir / "afm_config.json"
            with open(config_file, "w") as f:
                json.dump(afm_config, f, indent=2)
            logger.info(f"AFM configuration saved to {config_file}")

            state.file_paths["microscopy"] = str(afm_dir)
            state.add_error(
                "AFM simulation skipped: ppafm/OpenCL not available. Install OpenCL runtime and ppafm[opencl] to enable AFM."
            )
            return state

        # Try to run AFM simulation, with graceful degradation on OpenCL errors
        try:
            # Load XYZ file
            logger.info(f"Loading structure from {relaxed_xyz}")
            xyzs, Zs, qs, _ = loadXYZ(relaxed_xyz)

            # Prepare data for ppafm
            xyzqs = np.concatenate([xyzs, qs[:, None]], axis=1)

            logger.info("Setting up AFM simulation")

            # Create AFMulator with all parameters
            afmulator = AFMulator(
                pixPerAngstrome=pix_per_angstrome,
                scan_dim=scan_dim,
                scan_window=(scan_window_min, scan_window_max),
                iZPP=i_zpp,
                Qs=qs,
                QZs=qzs,
                sigma=sigma,
                A_pauli=a_pauli,
                B_pauli=b_pauli,
                fdbm_vdw_type=fdbm_vdw_type,
                d3_params=d3_params,
                lj_vdw_damp=lj_vdw_damp,
                df_steps=df_steps,
                tipR0=tip_r0,
                tipStiffness=tip_stiffness,
                npbc=npbc,
                f0Cantilever=f0_cantilever,
                kCantilever=k_cantilever,
                colorscale=colorscale,
                minimize_memory=minimize_memory,
            )

            # Define scan region for auxiliary maps (2D projection)
            scan_dim_2d = scan_dim[:2]
            scan_window_2d = (scan_window_min[:2], scan_window_max[:2])

            # Create auxiliary maps
            vdw_spheres = vdwSpheres(
                scan_dim=scan_dim_2d, scan_window=scan_window_2d, zmin=-1.5, Rpp=-0.5
            )
            atomic_disks = AtomicDisks(
                scan_dim=scan_dim_2d, scan_window=scan_window_2d, zmin=-1.2
            )
            height_map = HeightMap(scanner=afmulator.scanner, zmin=-2.0)
            es_map = ESMapConstant(
                scan_dim=scan_dim_2d,
                scan_window=scan_window_2d,
                vdW_cutoff=-2.0,
                Rpp=1.0,
            )

            logger.info("Evaluating auxiliary maps")

            # Evaluate maps
            y_spheres = vdw_spheres(xyzqs, Zs)
            y_disks = atomic_disks(xyzqs, Zs)
            y_es = es_map(xyzqs, Zs)

            # AFM simulation
            logger.info("Running AFM simulation")
            afm = afmulator(xyzs, Zs, qs)
            y_height = height_map(xyzqs, Zs)

            # Plot results
            formula = None
            try:
                from ase.io import read

                atoms = read(relaxed_xyz)
                formula = atoms.get_chemical_formula()
            except:
                formula = "structure"

            logger.info("Plotting AFM results")

            fig, axes = plt.subplots(1, 5, figsize=(16, 3), gridspec_kw={"wspace": 0.1})

            # AFM simulation
            axes[0].imshow(afm[:, :, -1].T, origin="lower", cmap="gray")
            axes[0].set_title("AFM Sim.")

            # vdW spheres
            axes[1].imshow(y_spheres.T, origin="lower", cmap="viridis")
            axes[1].set_title("vdW Spheres")

            # Atomic disks
            axes[2].imshow(y_disks.T, origin="lower", cmap="viridis")
            axes[2].set_title("Atomic Disks")

            # Height map
            axes[3].imshow(y_height.T, origin="lower", cmap="viridis")
            axes[3].set_title("Height Map")

            # ES map (symmetric colormap)
            vmax = max(y_es.max(), -y_es.min())
            vmin = -vmax
            axes[4].imshow(
                y_es.T, origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax
            )
            axes[4].set_title("ES Map")

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            afm_file = afm_dir / f"{formula}_afm_auxmaps.png"
            plt.tight_layout()
            plt.savefig(str(afm_file), dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved AFM auxiliary maps: {afm_file}")

            # Store results
            state.microscopy_results["afm"] = {
                "tip_height": tip_height,
                "scan_size": scan_size,
                "auxmaps_file": str(afm_file),
            }

            state.file_paths["microscopy"] = str(afm_dir)
            logger.info("AFM simulation complete")

            return state

        except RuntimeError as e:
            # Catch OpenCL build errors and other runtime issues
            if "BUILD_PROGRAM_FAILURE" in str(e) or "clBuildProgram" in str(e):
                logger.warning(f"OpenCL build failed: {e}")
                logger.info("Saving AFM configuration for later execution")

                # Save AFM configuration as JSON for later use
                afm_config = {
                    "status": "pending",
                    "reason": "OpenCL compilation failed - incompatible pocl/ppafm version",
                    "error": str(e)[:500],  # Limit error message length
                    "parameters": {
                        "pix_per_angstrome": pix_per_angstrome,
                        "scan_dim": scan_dim,
                        "scan_window_min": scan_window_min,
                        "scan_window_max": scan_window_max,
                        "i_zpp": i_zpp,
                        "qs": qs if isinstance(qs, list) else qs.tolist(),
                        "qzs": qzs if isinstance(qzs, list) else qzs.tolist(),
                        "sigma": sigma,
                        "a_pauli": a_pauli,
                        "b_pauli": b_pauli,
                        "fdbm_vdw_type": fdbm_vdw_type,
                        "d3_params": d3_params,
                        "lj_vdw_damp": lj_vdw_damp,
                        "df_steps": df_steps,
                        "tip_r0": (
                            tip_r0
                            if isinstance(tip_r0, (list, tuple))
                            else list(tip_r0)
                        ),
                        "tip_stiffness": (
                            tip_stiffness
                            if isinstance(tip_stiffness, (list, tuple))
                            else list(tip_stiffness)
                        ),
                        "npbc": npbc if isinstance(npbc, (list, tuple)) else list(npbc),
                        "f0_cantilever": f0_cantilever,
                        "k_cantilever": k_cantilever,
                        "colorscale": colorscale,
                        "minimize_memory": minimize_memory,
                        "tip_height": tip_height,
                        "scan_size": (
                            scan_size
                            if isinstance(scan_size, (list, tuple))
                            else list(scan_size)
                        ),
                    },
                }
                config_file = afm_dir / "afm_config.json"
                with open(config_file, "w") as f:
                    json.dump(afm_config, f, indent=2)
                logger.info(f"AFM configuration saved to {config_file}")

                state.file_paths["microscopy"] = str(afm_dir)
                state.add_warning(
                    "AFM simulation skipped: OpenCL compilation failed. Try updating pocl or ppafm packages."
                )
                return state
            else:
                # Re-raise other runtime errors
                raise

    except ImportError as e:
        logger.error(f"ppafm not installed: {e}")
        state.add_error(f"ppafm not available: {str(e)}")
        return state
    except Exception as e:
        logger.error(f"AFM simulation failed: {e}", exc_info=True)
        state.add_error(f"AFM simulation failed: {str(e)}")
        return state
