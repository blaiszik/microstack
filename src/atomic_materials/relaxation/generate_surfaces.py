from ase.build import fcc100, fcc111, fcc110, graphene_nanoribbon, mx2
from ase import Atoms
from ase.io import write
import os
import uuid
from pathlib import Path

from atomic_materials.utils import config


def create_surface(
    element: str,
    face: str,
    task_id: str,
    size: tuple[int, int, int] = (3, 3, 4),
    vacuum: float = 10.0,
) -> tuple[Atoms, Path]:
    """
    Create a surface for a given element and face.

    Args:
        element: Chemical symbol (e.g., 'Cu', 'Pt', 'Au', 'C' for graphene)
        face: Surface face ('100', '111', '110', 'graphene', '2d')
        task_id: Unique identifier for the task.
        size: Tuple of (x, y, z) repetitions. z is number of layers.
        vacuum: Vacuum padding in Angstroms

    Returns:
        A tuple containing the ASE Atoms object and the path to the output directory.
    """
    # Special case for Graphene
    if face.lower() == "graphene" or (element == "C" and face == "2d"):
        atoms = graphene_nanoribbon(
            size[0], size[1], type="zigzag", saturated=False, C_C=1.42, vacuum=vacuum
        )
        atoms.pbc = [True, True, False]
    else:
        # Approximate lattice constants for common metals
        lattice_constants = {
            "Cu": 3.61,
            "Pt": 3.92,
            "Au": 4.08,
            "Ag": 4.09,
            "Al": 4.05,
            "Ni": 3.52,
            "Pd": 3.89,
            "Fe": 2.87,  # bcc usually, but let's keep it simple
            "Ir": 3.84,
            "Rh": 3.80,
        }

        a = lattice_constants.get(element)
        if a is None:
            if element in lattice_constants:
                pass
            else:
                a = 4.0

        if face == "100":
            atoms = fcc100(element, size=size, a=a, vacuum=vacuum)
        elif face == "111":
            atoms = fcc111(element, size=size, a=a, vacuum=vacuum)
        elif face == "110":
            atoms = fcc110(element, size=size, a=a, vacuum=vacuum)
        else:
            raise ValueError(
                f"Unsupported face: {face}. Choose from '100', '111', '110', 'graphene', '2d'."
            )

    # Define the output directory for this specific task
    output_dir = (
        Path(config.OUTPUT_DIR) / f"{element}_{face}_unrelaxed_{task_id}/relaxation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save unrelaxed structure
    unrelaxed_filename = output_dir / f"{element}_{face}_unrelaxed.xyz"
    write(str(unrelaxed_filename), atoms)

    return atoms, output_dir


if __name__ == "__main__":
    # Test
    test_task_id = str(uuid.uuid4())
    s, out_path = create_surface("Cu", "100", task_id=test_task_id)
    print(f"Created Cu(100) with {len(s)} atoms, saved to {out_path}")
