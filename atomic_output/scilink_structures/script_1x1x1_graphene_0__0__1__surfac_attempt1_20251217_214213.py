
from ase.build import graphene
from ase.io import write
import numpy as np

# 1. Create the graphene structure
# The default graphene() function creates a unit cell.
atoms = graphene()

# 2. Add 10.0A vacuum
# Get the current cell from the Atoms object.
current_cell = atoms.get_cell()

# Set the length of the c-vector (along the z-axis) to 10.0 Angstroms.
# For graphene created by ase.build.graphene(), the c-vector is typically along z
# and defines the vacuum.
current_cell[2, 2] = 10.0

# Apply the modified cell back to the Atoms object.
atoms.set_cell(current_cell, scale_atoms=False) # scale_atoms=False to not change atomic positions relative to cell

# Center the atoms in the new cell along the z-axis.
# This ensures the 10.0A vacuum is distributed evenly above and below the graphene layer.
atoms.center(axis=2)

# 3. Save the structure in XYZ format
filename = "graphene_surface.xyz"
write(filename, atoms)

# 4. Print the required confirmation line
print(f"STRUCTURE_SAVED:{filename}")
