
from ase import Atoms
from ase.build import graphene
from ase.io import write
import numpy as np

# 1. Build a 1x1 graphene unit cell
atoms = graphene()

# 2. Get the existing 2D cell vectors
old_cell = atoms.get_cell()

# 3. Define the vacuum thickness
vacuum_thickness = 10.0

# 4. Create a new 3D cell by adding a c-vector for the vacuum
# The a and b vectors remain from the 2D graphene cell
new_c_vector = np.array([0, 0, vacuum_thickness])
new_cell = np.vstack((old_cell[0], old_cell[1], new_c_vector))

# 5. Set the new cell to the Atoms object
atoms.set_cell(new_cell)

# 6. Set periodic boundary conditions for all directions
atoms.set_pbc([True, True, True])

# 7. Center the atoms in the new cell to ensure vacuum is distributed
atoms.center()

# 8. Save the structure in XYZ format
filename = "graphene_vacuum.xyz"
write(filename, atoms)

# 9. Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")
