"""Materials Project integration and literature reference data.

Provides bulk properties from MP API and curated surface relaxation
data from experimental LEED and DFT studies.
"""

from typing import Optional
import config

# =============================================================================
# Literature Reference Data for Surface Relaxations
# =============================================================================

# Curated from experimental LEED measurements and high-quality DFT calculations
# Format: {element: {face: {property: value}}}
# d12_change, d23_change are in percent (negative = contraction)
# surface_energy in J/m²

SURFACE_REFERENCE_DATA = {
    "Cu": {
        "100": {
            "d12_change": -2.1,  # Lindgren et al., Phys. Rev. B 29, 576 (1984)
            "d23_change": 0.8,
            "d34_change": 0.0,
            "surface_energy": 1.79,
            "source": "Lindgren et al., Phys. Rev. B 29, 576 (1984)",
            "method": "LEED"
        },
        "111": {
            "d12_change": -0.7,  # Davis & Noonan, Surf. Sci. 126, 245 (1983)
            "d23_change": 0.2,
            "d34_change": 0.0,
            "surface_energy": 1.52,
            "source": "Davis & Noonan, Surf. Sci. 126, 245 (1983)",
            "method": "LEED"
        },
        "110": {
            "d12_change": -8.5,  # Strong contraction for open surface
            "d23_change": 2.3,
            "d34_change": -0.5,
            "surface_energy": 1.93,
            "source": "Adams et al., Surf. Sci. 187, 313 (1987)",
            "method": "LEED"
        }
    },
    "Pt": {
        "111": {
            "d12_change": 1.0,  # Slight expansion - anomalous
            "d23_change": 0.5,
            "d34_change": 0.0,
            "surface_energy": 2.30,
            "source": "Materer et al., Surf. Sci. 325, 207 (1995)",
            "method": "LEED"
        },
        "100": {
            "d12_change": -1.1,
            "d23_change": 0.6,
            "d34_change": 0.0,
            "surface_energy": 2.47,
            "source": "Heilmann et al., Surf. Sci. 83, 487 (1979)",
            "method": "LEED"
        }
    },
    "Au": {
        "111": {
            "d12_change": 0.1,  # Nearly bulk-like
            "d23_change": 0.3,
            "d34_change": 0.0,
            "surface_energy": 1.50,
            "source": "Harten et al., Phys. Rev. Lett. 54, 2619 (1985)",
            "method": "LEED"
        },
        "100": {
            "d12_change": -1.2,
            "d23_change": 0.8,
            "d34_change": 0.0,
            "surface_energy": 1.63,
            "source": "Gibbs et al., Phys. Rev. Lett. 67, 3117 (1991)",
            "method": "X-ray"
        }
    },
    "Ag": {
        "111": {
            "d12_change": -0.5,
            "d23_change": 0.2,
            "d34_change": 0.0,
            "surface_energy": 1.25,
            "source": "Soares et al., Phys. Rev. B 60, 10768 (1999)",
            "method": "LEED"
        },
        "100": {
            "d12_change": -1.8,
            "d23_change": 0.9,
            "d34_change": 0.0,
            "surface_energy": 1.35,
            "source": "Quinn et al., J. Phys. C 21, L195 (1988)",
            "method": "LEED"
        }
    },
    "Ni": {
        "100": {
            "d12_change": -3.2,  # Strong contraction
            "d23_change": 1.5,
            "d34_change": -0.3,
            "surface_energy": 2.38,
            "source": "Demuth et al., Phys. Rev. Lett. 34, 1149 (1975)",
            "method": "LEED"
        },
        "111": {
            "d12_change": -1.2,
            "d23_change": 0.5,
            "d34_change": 0.0,
            "surface_energy": 2.01,
            "source": "Narasimhan & Vanderbilt, Phys. Rev. Lett. 69, 1564 (1992)",
            "method": "DFT"
        }
    },
    "Pd": {
        "111": {
            "d12_change": 0.0,  # Nearly bulk-terminated
            "d23_change": 0.2,
            "d34_change": 0.0,
            "surface_energy": 2.00,
            "source": "Ohtani et al., Phys. Rev. B 36, 4460 (1987)",
            "method": "LEED"
        },
        "100": {
            "d12_change": -2.5,
            "d23_change": 1.2,
            "d34_change": 0.0,
            "surface_energy": 2.15,
            "source": "Behm et al., J. Chem. Phys. 78, 7486 (1983)",
            "method": "LEED"
        }
    }
}

# 2D Materials reference data
REFERENCE_2D_DATA = {
    "C": {
        "graphene": {
            "c_c_bond": 1.42,  # Angstrom
            "lattice_constant": 2.46,
            "layer_spacing": 3.35,  # In graphite
            "source": "Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009)",
            "method": "Experiment"
        }
    },
    "MoS2": {
        "2d": {
            "lattice_constant": 3.16,
            "layer_thickness": 3.13,  # S-Mo-S sandwich thickness
            "mo_s_bond": 2.41,
            "source": "Splendiani et al., Nano Lett. 10, 1271 (2010)",
            "method": "Experiment"
        }
    }
}

# Materials Project IDs for common elements
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
                    fields=["structure", "formation_energy_per_atom", "band_gap",
                            "density", "symmetry"]
                )
                if docs:
                    doc = docs[0]
                    structure = doc.structure
                    return {
                        "lattice_constant": structure.lattice.a,
                        "formation_energy": doc.formation_energy_per_atom or 0.0,
                        "band_gap": doc.band_gap or 0.0,
                        "crystal_system": doc.symmetry.crystal_system if doc.symmetry else "cubic",
                        "space_group": doc.symmetry.symbol if doc.symmetry else "unknown",
                        "density": doc.density or 0.0,
                        "source": "Materials Project",
                        "mp_id": mp_id
                    }
        except Exception as e:
            print(f"Warning: MP API query failed: {e}")

    # Fallback to cached data
    if element in BULK_PROPERTIES_CACHE:
        data = BULK_PROPERTIES_CACHE[element].copy()
        data["source"] = "Cached literature values"
        data["mp_id"] = MP_IDS.get(element, "unknown")
        return data

    return {
        "lattice_constant": None,
        "formation_energy": None,
        "band_gap": None,
        "source": "No data available"
    }


def get_surface_reference(element: str, face: str) -> Optional[dict]:
    """
    Get reference data for surface relaxation from literature.

    Args:
        element: Chemical symbol
        face: Surface face ('100', '111', '110', 'graphene', '2d')

    Returns:
        Dictionary with reference relaxation data, or None if not available
    """
    # Check for 2D materials
    if face in ["graphene", "2d"]:
        if element in REFERENCE_2D_DATA:
            if face in REFERENCE_2D_DATA[element]:
                return REFERENCE_2D_DATA[element][face].copy()
            # Try '2d' as fallback
            if "2d" in REFERENCE_2D_DATA[element]:
                return REFERENCE_2D_DATA[element]["2d"].copy()
        return None

    # Check for metal surfaces
    if element in SURFACE_REFERENCE_DATA:
        if face in SURFACE_REFERENCE_DATA[element]:
            return SURFACE_REFERENCE_DATA[element][face].copy()

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
        "has_surface_reference": surface is not None
    }


def list_available_references() -> dict:
    """List all available surface reference data."""
    available = {}

    for element, faces in SURFACE_REFERENCE_DATA.items():
        available[element] = list(faces.keys())

    for element, faces in REFERENCE_2D_DATA.items():
        if element not in available:
            available[element] = []
        available[element].extend(faces.keys())

    return available


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

    print("\nAvailable references:")
    for elem, faces in list_available_references().items():
        print(f"  {elem}: {faces}")
