"""Comparison engine for surface relaxation analysis.

Extracts relaxation metrics from structures and compares with reference data.
"""

import numpy as np
from ase import Atoms
from typing import Optional

from microstack.relaxation import materials_project


def analyze_relaxation(unrelaxed: Atoms, relaxed: Atoms) -> dict:
    """
    Extract detailed relaxation metrics from structures.

    Args:
        unrelaxed: Original (ideal) surface structure
        relaxed: Structure after ML relaxation

    Returns:
        Dictionary containing:
        - interlayer_spacings_before/after
        - layer_changes (percent)
        - atomic_displacements
        - max_displacement
        - mean_displacement
    """
    pos_before = unrelaxed.get_positions()
    pos_after = relaxed.get_positions()

    # Calculate atomic displacements
    displacements = pos_after - pos_before
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    z_displacements = displacements[:, 2]

    # Identify layers by clustering z-coordinates
    z_before = pos_before[:, 2]
    z_after = pos_after[:, 2]

    # Get unique layer z-positions (rounded to identify layers)
    tolerance = 0.5  # Angstrom
    layer_z_before = np.unique(np.round(z_before, decimals=1))

    # Calculate mean z for each layer after relaxation
    layer_z_after = []
    layer_atom_counts = []

    for z_layer in layer_z_before:
        mask = np.abs(z_before - z_layer) < tolerance
        layer_z_after.append(np.mean(z_after[mask]))
        layer_atom_counts.append(np.sum(mask))

    layer_z_after = np.array(layer_z_after)

    # Calculate interlayer spacings
    spacings_before = np.diff(layer_z_before)
    spacings_after = np.diff(layer_z_after)

    # Calculate percent changes
    layer_changes = {}
    if len(spacings_before) > 0:
        spacing_change_pct = (spacings_after - spacings_before) / spacings_before * 100

        for i, change in enumerate(spacing_change_pct):
            layer_changes[f"d{i+1}{i+2}_change"] = float(change)

    return {
        "n_atoms": len(unrelaxed),
        "n_layers": len(layer_z_before),
        "layer_atom_counts": layer_atom_counts,
        # Interlayer spacings
        "interlayer_spacings_before": spacings_before.tolist(),
        "interlayer_spacings_after": spacings_after.tolist(),
        "layer_changes_percent": layer_changes,
        # Atomic displacements
        "z_displacements": z_displacements.tolist(),
        "displacement_magnitudes": displacement_magnitudes.tolist(),
        "max_displacement": float(np.max(displacement_magnitudes)),
        "mean_displacement": float(np.mean(displacement_magnitudes)),
        "max_z_displacement": float(np.max(np.abs(z_displacements))),
        # Layer positions
        "layer_z_before": layer_z_before.tolist(),
        "layer_z_after": layer_z_after.tolist(),
    }


def compare_with_reference(
    ml_analysis: dict, reference: Optional[dict], element: str, face: str
) -> dict:
    """
    Compare ML prediction with reference data.

    Args:
        ml_analysis: Output from analyze_relaxation()
        reference: Reference data from materials_project.get_surface_reference()
        element: Chemical symbol
        face: Surface face

    Returns:
        Dictionary with comparison metrics and agreement scores
    """
    comparison = {
        "element": element,
        "face": face,
        "has_reference": reference is not None,
        "layer_comparisons": [],
        "overall_agreement": None,
        "reference_source": None,
        "reference_method": None,
    }

    if reference is None:
        comparison["message"] = f"No reference data available for {element}({face})"
        return comparison

    comparison["reference_source"] = reference.get("source", "Unknown")
    comparison["reference_method"] = reference.get("method", "Unknown")

    ml_changes = ml_analysis.get("layer_changes_percent", {})
    deviations = []

    # Compare each layer spacing
    for key in ["d12_change", "d23_change", "d34_change"]:
        ml_key = key.replace("_change", "_change")  # Same format
        ml_value = ml_changes.get(ml_key)
        ref_value = reference.get(key)

        if ml_value is not None and ref_value is not None:
            deviation = abs(ml_value - ref_value)
            deviations.append(deviation)

            comparison["layer_comparisons"].append(
                {
                    "layer": key.replace("_change", ""),
                    "ml_prediction": round(ml_value, 2),
                    "reference": ref_value,
                    "deviation": round(deviation, 2),
                    "agreement": (
                        "good"
                        if deviation < 1.0
                        else ("moderate" if deviation < 2.0 else "poor")
                    ),
                }
            )

    # Calculate overall agreement score
    if deviations:
        mean_deviation = np.mean(deviations)
        comparison["mean_deviation"] = round(mean_deviation, 2)

        if mean_deviation < 1.0:
            comparison["overall_agreement"] = "excellent"
            comparison["agreement_description"] = (
                f"Mean deviation {mean_deviation:.1f}% - within experimental uncertainty"
            )
        elif mean_deviation < 2.0:
            comparison["overall_agreement"] = "good"
            comparison["agreement_description"] = (
                f"Mean deviation {mean_deviation:.1f}% - reasonable agreement"
            )
        elif mean_deviation < 3.0:
            comparison["overall_agreement"] = "moderate"
            comparison["agreement_description"] = (
                f"Mean deviation {mean_deviation:.1f}% - qualitative agreement"
            )
        else:
            comparison["overall_agreement"] = "poor"
            comparison["agreement_description"] = (
                f"Mean deviation {mean_deviation:.1f}% - significant discrepancy"
            )

    return comparison


def full_analysis(
    unrelaxed: Atoms,
    relaxed: Atoms,
    element: str,
    face: str,
    initial_energy: float,
    final_energy: float,
) -> dict:
    """
    Perform complete analysis including relaxation metrics and reference comparison.

    Args:
        unrelaxed: Original surface structure
        relaxed: Relaxed surface structure
        element: Chemical symbol
        face: Surface face
        initial_energy: Energy before relaxation (eV)
        final_energy: Energy after relaxation (eV)

    Returns:
        Complete analysis dictionary
    """
    # Get relaxation analysis
    relaxation = analyze_relaxation(unrelaxed, relaxed)

    # Get reference data
    reference_data = materials_project.get_all_reference_data(element, face)

    # Compare with reference
    comparison = compare_with_reference(
        relaxation, reference_data.get("surface"), element, face
    )

    return {
        "element": element,
        "face": face,
        "surface_label": f"{element}({face})",
        # Energy
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "energy_change_eV": final_energy - initial_energy,
        # Relaxation metrics
        "relaxation": relaxation,
        # Reference data
        "reference": reference_data,
        # Comparison
        "comparison": comparison,
    }


if __name__ == "__main__":
    # Test with dummy data
    print("Comparison module loaded successfully")
    print("\nAvailable functions:")
    print("  - analyze_relaxation(unrelaxed, relaxed)")
    print("  - compare_with_reference(ml_analysis, reference, element, face)")
    print("  - full_analysis(unrelaxed, relaxed, element, face, init_e, final_e)")
