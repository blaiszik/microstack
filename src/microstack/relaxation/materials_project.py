"""Materials Project integration for bulk properties.

Provides bulk properties from MP API for surface analysis.
"""

from typing import Optional

from microstack.utils import config

# =============================================================================
# Materials Project IDs for common elements
# =============================================================================

MP_IDS = {
    "Cu": "mp-30",
    "Pt": "mp-126",
    "Au": "mp-81",
    "Ag": "mp-124",
    "Ni": "mp-23",
    "Pd": "mp-2",
    "Fe": "mp-13953",
    "Al": "mp-134",
    "C": "mp-48",  # Graphite
    "Mo": "mp-129",
    "S": "mp-96",
}

# =============================================================================
# Bulk Properties (fallback if MP API unavailable)
# =============================================================================

BULK_PROPERTIES_CACHE = {
    "Cu": {
        "lattice_constant": 3.615,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "density": 8.96,
    },
    "Pt": {
        "lattice_constant": 3.924,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "density": 21.45,
    },
    "Au": {
        "lattice_constant": 4.078,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "density": 19.32,
    },
    "Ag": {
        "lattice_constant": 4.086,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "density": 10.49,
    },
    "Ni": {
        "lattice_constant": 3.524,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "density": 8.91,
    },
    "Pd": {
        "lattice_constant": 3.891,
        "formation_energy": 0.0,
        "band_gap": 0.0,
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "density": 12.02,
    },
}


# =============================================================================
# API Functions
# =============================================================================


def get_bulk_properties(element: str) -> dict:
    """
    Query Materials Project for bulk properties.
    Falls back to cached data if API unavailable.

    Args:
        element: Chemical symbol (e.g., 'Cu', 'Pt')

    Returns:
        Dictionary with bulk properties
    """
    mpr = config.get_mp_client()

    if mpr is not None:
        try:
            mp_id = MP_IDS.get(element)
            if mp_id:
                # Query MP for the material
                docs = mpr.materials.summary.search(
                    material_ids=[mp_id],
                    fields=[
                        "structure",
                        "formation_energy_per_atom",
                        "band_gap",
                        "density",
                        "symmetry",
                    ],
                )
                if docs:
                    doc = docs[0]
                    structure = doc.structure
                    # Convert enum to string for JSON serialization
                    crystal_system = "cubic"
                    if doc.symmetry and doc.symmetry.crystal_system:
                        crystal_system = (
                            str(doc.symmetry.crystal_system.value)
                            if hasattr(doc.symmetry.crystal_system, "value")
                            else str(doc.symmetry.crystal_system)
                        )
                    return {
                        "lattice_constant": float(structure.lattice.a),
                        "formation_energy": float(doc.formation_energy_per_atom or 0.0),
                        "band_gap": float(doc.band_gap or 0.0),
                        "crystal_system": crystal_system,
                        "space_group": (
                            str(doc.symmetry.symbol) if doc.symmetry else "unknown"
                        ),
                        "density": float(doc.density or 0.0),
                        "source": "Materials Project",
                        "mp_id": mp_id,
                    }
        except Exception as e:
            print(f"Warning: MP API query failed: {e}")

    # Fallback to cached data
    if element in BULK_PROPERTIES_CACHE:
        data = BULK_PROPERTIES_CACHE[element].copy()
        data["source"] = "Cached (Materials Project)"
        data["mp_id"] = MP_IDS.get(element, "unknown")
        return data

    return {
        "lattice_constant": None,
        "formation_energy": None,
        "band_gap": None,
        "source": "No data available",
    }


def get_surface_reference(element: str, face: str) -> Optional[dict]:
    """
    Get reference data for surface relaxation.

    Currently returns None - experimental reference data pending verification.

    Args:
        element: Chemical symbol
        face: Surface face ('100', '111', '110', 'graphene', '2d')

    Returns:
        None (reference data not yet available)
    """
    # TODO: Add verified experimental reference data
    return None


def get_all_reference_data(element: str, face: str) -> dict:
    """
    Get combined bulk and surface reference data.

    Args:
        element: Chemical symbol
        face: Surface face

    Returns:
        Dictionary with all available reference data
    """
    bulk = get_bulk_properties(element)
    surface = get_surface_reference(element, face)

    return {
        "bulk": bulk,
        "surface": surface,
        "element": element,
        "face": face,
        "has_surface_reference": surface is not None,
    }


def list_available_references() -> dict:
    """List all available surface reference data."""
    # Currently no verified surface reference data available
    return {}


if __name__ == "__main__":
    # Test the module
    print("Testing Materials Project integration...\n")

    # Test bulk properties
    print("Bulk properties for Cu:")
    props = get_bulk_properties("Cu")
    for k, v in props.items():
        print(f"  {k}: {v}")

    print("\nSurface reference for Cu(100):")
    ref = get_surface_reference("Cu", "100")
    if ref:
        for k, v in ref.items():
            print(f"  {k}: {v}")
    else:
        print("  No reference data available (pending verification)")

    print("\nAvailable references:")
    refs = list_available_references()
    if refs:
        for elem, faces in refs.items():
            print(f"  {elem}: {faces}")
    else:
        print("  None (experimental data pending verification)")
