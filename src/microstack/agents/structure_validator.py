"""Structure validation utilities for atomic structures."""

import logging
from pathlib import Path
from typing import Tuple, Optional

from ase import Atoms
from ase.io import read, write

from microstack.utils.logging import get_logger

logger = get_logger("agents.structure_validator")


def validate_structure(
    atoms: Atoms,
    min_vacuum: float = 5.0,
    max_atoms_ratio: float = 0.6,
    edge_margin: float = 1.0,
) -> Tuple[bool, str]:
    """
    Validate a structure for microscopy simulations - checks only periodic conditions.

    Checks:
    - Sufficient vacuum spacing (default 5.0 Angstrom for non-periodic directions)
    - Reasonable atom density (max 60% of box volume)
    - Minimal edge margin for non-periodic directions only (default 1.0 Angstrom)

    Args:
        atoms: ASE Atoms object to validate
        min_vacuum: Minimum vacuum spacing in Angstrom (default 5.0)
        max_atoms_ratio: Maximum ratio of atom volume to box volume (default 0.6)
        edge_margin: Minimum distance from atoms to non-periodic boundaries (default 1.0)

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        import numpy as np
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

        # Convert to pymatgen structure
        try:
            structure = AseAtomsAdaptor.get_structure(atoms)
        except Exception as e:
            logger.warning(f"Could not convert to pymatgen structure: {e}")
            structure = None

        # Check 1: Vacuum spacing (for slab structures)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # If non-periodic in z (slab structure), check vacuum in z-direction
        if not pbc[2]:
            positions = atoms.get_positions()
            z_min = positions[:, 2].min()
            z_max = positions[:, 2].max()
            z_length = cell[2, 2]

            bottom_vacuum = z_min
            top_vacuum = z_length - z_max

            if bottom_vacuum < min_vacuum or top_vacuum < min_vacuum:
                msg = (
                    f"Insufficient vacuum: bottom={bottom_vacuum:.2f}Å, "
                    f"top={top_vacuum:.2f}Å (minimum required: {min_vacuum}Å)"
                )
                logger.warning(msg)
                return False, msg

        # Check 2: Atom proximity to cell boundaries (only for non-periodic directions)
        positions = atoms.get_positions()
        for i, pos in enumerate(positions):
            for j in range(3):
                # Only check edge margins for NON-periodic directions
                # For periodic directions, atoms can be at boundaries
                if not pbc[j]:
                    # Non-periodic direction - check if atoms are too close to edges
                    scaled_pos = np.linalg.solve(cell, pos)[j]

                    # For non-periodic, check distance from 0 and from 1
                    dist_from_bottom = scaled_pos
                    dist_from_top = 1.0 - scaled_pos

                    # Convert to Angstrom
                    dist_bottom_ang = dist_from_bottom * np.linalg.norm(cell[j])
                    dist_top_ang = dist_from_top * np.linalg.norm(cell[j])

                    min_dist = min(dist_bottom_ang, dist_top_ang)

                    if min_dist < edge_margin:
                        msg = (
                            f"Atom {i} too close to boundary in dimension {j}: "
                            f"distance={min_dist:.2f}Å "
                            f"(minimum required: {edge_margin}Å)"
                        )
                        logger.warning(msg)
                        return False, msg

        # Check 3: Atom density
        num_atoms = len(atoms)
        box_volume = abs(np.linalg.det(cell))

        # Estimate atom volume (rough approximation: 10 Å³ per atom)
        estimated_atom_volume = num_atoms * 10.0
        ratio = estimated_atom_volume / box_volume

        if ratio > max_atoms_ratio:
            msg = (
                f"Structure too dense: atom volume ratio={ratio:.2%} "
                f"(maximum allowed: {max_atoms_ratio:.0%})"
            )
            logger.warning(msg)
            return False, msg

        # Check 4: Use pymatgen validators if available
        if structure:
            try:
                from pymatgen.analysis.structure_analyzer import (
                    SpacegroupAnalyzer,
                )

                analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
                primitive = analyzer.get_primitive_standard_structure()

                # Check for too-close atoms
                if primitive.is_valid():
                    # Structure passes pymatgen validation
                    logger.info(
                        f"Structure validation passed: {len(atoms)} atoms, "
                        f"volume={box_volume:.1f}Å³"
                    )
                    return True, "Structure valid"
            except Exception as e:
                logger.debug(f"PyMatGen detailed validation skipped: {e}")

        logger.info(
            f"Structure validation passed: {len(atoms)} atoms, "
            f"volume={box_volume:.1f}Å³"
        )
        return True, "Structure valid"

    except ImportError:
        logger.warning("PyMatGen not available, using basic validation")
        return _basic_validate_structure(atoms, min_vacuum, edge_margin)
    except Exception as e:
        logger.error(f"Structure validation error: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"


def _basic_validate_structure(
    atoms: Atoms,
    min_vacuum: float = 8.0,
    edge_margin: float = 2.0,
) -> Tuple[bool, str]:
    """
    Basic structure validation without pymatgen.

    Args:
        atoms: ASE Atoms object to validate
        min_vacuum: Minimum vacuum spacing in Angstrom
        edge_margin: Minimum distance from atoms to cell edges

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        import numpy as np

        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        positions = atoms.get_positions()

        # Check vacuum in z-direction for slab structures
        if not pbc[2]:
            z_min = positions[:, 2].min()
            z_max = positions[:, 2].max()
            z_length = cell[2, 2]

            bottom_vacuum = z_min
            top_vacuum = z_length - z_max

            if bottom_vacuum < min_vacuum or top_vacuum < min_vacuum:
                msg = (
                    f"Insufficient vacuum: bottom={bottom_vacuum:.2f}Å, "
                    f"top={top_vacuum:.2f}Å (minimum required: {min_vacuum}Å)"
                )
                logger.warning(msg)
                return False, msg

        # Check edge margins (only for non-periodic directions)
        for i, pos in enumerate(positions):
            for j in range(3):
                if not pbc[j]:
                    # Non-periodic direction - check if atoms are too close to edges
                    scaled = np.dot(np.linalg.inv(cell), pos)[j]

                    # For non-periodic, check distance from 0 and from 1
                    dist_from_bottom = scaled
                    dist_from_top = 1.0 - scaled

                    # Convert to Angstrom
                    dist_bottom_ang = dist_from_bottom * np.linalg.norm(cell[j])
                    dist_top_ang = dist_from_top * np.linalg.norm(cell[j])

                    min_dist = min(dist_bottom_ang, dist_top_ang)

                    if min_dist < edge_margin:
                        msg = (
                            f"Atom {i} too close to boundary in dimension {j}: "
                            f"distance={min_dist:.2f}Å "
                            f"(minimum required: {edge_margin}Å)"
                        )
                        logger.warning(msg)
                        return False, msg

        logger.info(
            f"Structure validation passed: {len(atoms)} atoms, "
            f"volume={abs(np.linalg.det(cell)):.1f}Å³"
        )
        return True, "Structure valid"

    except Exception as e:
        logger.error(f"Basic validation failed: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"


def fix_structure_vacuum(atoms: Atoms, target_vacuum: float = 10.0) -> Atoms:
    """
    Fix structure by adding/adjusting vacuum spacing and repositioning atoms.

    Args:
        atoms: ASE Atoms object
        target_vacuum: Target vacuum spacing in Angstrom

    Returns:
        Modified Atoms object with adjusted vacuum
    """
    try:
        import numpy as np
        from ase.build import add_vacuum

        # For slab structures, ensure sufficient vacuum
        atoms_copy = atoms.copy()

        pbc = atoms_copy.get_pbc()

        # Expand cell dramatically to ensure atoms are not at edges
        cell = atoms_copy.get_cell()

        # Scale up the cell significantly to provide buffer
        # For periodic directions, increase by 50%
        expansion_factor = 1.5
        for i in range(3):
            if pbc[i]:
                cell[i] *= expansion_factor

        atoms_copy.set_cell(cell, scale_atoms=True)
        logger.info(f"Expanded cell by {(expansion_factor-1)*100:.0f}% in all periodic directions")

        # Now center atoms within the expanded cell
        positions = atoms_copy.get_positions()
        atom_min = positions.min(axis=0)
        atom_max = positions.max(axis=0)
        atom_range = atom_max - atom_min

        # Calculate center position for atoms (leave margin on each side)
        margin = 3.0  # 3 Angstrom margin from edges
        new_cell = atoms_copy.get_cell()

        # Find where atoms should be centered
        for i in range(3):
            if pbc[i]:
                # Center atoms with margin
                available_space = new_cell[i, i] - 2 * margin
                if available_space > 0:
                    center_pos = margin + available_space / 2
                    # Shift atoms
                    shift = center_pos - atom_min[i] - atom_range[i] / 2
                    positions[:, i] += shift

        atoms_copy.set_positions(positions)
        atoms_copy.wrap()
        logger.info("Repositioned atoms with 3.0Å margin from edges")

        if not pbc[2]:
            # Slab structure - add vacuum in z-direction
            current_vacuum = _get_current_vacuum(atoms_copy)
            if current_vacuum < target_vacuum:
                # Need to add more vacuum
                needed_vacuum = target_vacuum - current_vacuum
                # Add vacuum by extending cell in z
                add_vacuum(atoms_copy, needed_vacuum / 2)
                logger.info(
                    f"Added {needed_vacuum:.2f}Å vacuum, "
                    f"new vacuum: {_get_current_vacuum(atoms_copy):.2f}Å"
                )

        return atoms_copy
    except Exception as e:
        logger.error(f"Failed to fix vacuum: {e}")
        return atoms


def _get_current_vacuum(atoms: Atoms) -> float:
    """Get current vacuum spacing in z-direction."""
    try:
        import numpy as np

        positions = atoms.get_positions()
        cell = atoms.get_cell()

        z_min = positions[:, 2].min()
        z_max = positions[:, 2].max()
        z_length = cell[2, 2]

        bottom_vacuum = z_min
        top_vacuum = z_length - z_max

        return min(bottom_vacuum, top_vacuum)
    except Exception:
        return 0.0
