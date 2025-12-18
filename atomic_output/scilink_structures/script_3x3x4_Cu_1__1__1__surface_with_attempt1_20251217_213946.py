
from ase.build import fcc111
from ase.io import write

# Build the Cu(111) surface
# 'Cu' for the element
# size=(3,3,4) for 3x3 supercell in the surface plane and 4 layers
# a=3.61 for the lattice constant of Cu (fcc)
slab = fcc111('Cu', size=(3,3,4), a=3.61)

# Add 15.0 Å of vacuum
slab.center(vacuum=15.0/2, axis=2) # vacuum is added on both sides, so divide by 2

# Save the structure in XYZ format
filename = 'Cu_111_3x3x4_15A_vac.xyz'
write(filename, slab)

# Print the confirmation line
print(f'STRUCTURE_SAVED:{filename}')
