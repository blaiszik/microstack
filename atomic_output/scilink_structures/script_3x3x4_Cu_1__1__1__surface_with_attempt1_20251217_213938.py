
from ase.build import surface
from ase.io import write

# Build the Cu(111) surface
# 'fcc' for face-centered cubic structure
# (1,1,1) for the surface orientation
# (3,3,1) for the supercell in the surface plane (3x3) and 1 unit cell in the z-direction
# layers=4 for the slab thickness
slab = surface('Cu', (1,1,1), size=(3,3,4), a=3.61)

# Add 15.0 Å of vacuum
slab.center(vacuum=15.0/2, axis=2) # vacuum is added on both sides, so divide by 2

# Save the structure in XYZ format
filename = 'Cu_111_3x3x4_15A_vac.xyz'
write(filename, slab)

# Print the confirmation line
print(f'STRUCTURE_SAVED:{filename}')
