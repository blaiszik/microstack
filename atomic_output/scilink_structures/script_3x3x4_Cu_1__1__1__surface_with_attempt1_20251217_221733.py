
from ase.build import surface
from ase.io import write

# Build the Cu(111) surface with 4 layers and 15.0 Å vacuum.
# The 'layers' parameter controls the number of atomic layers in the surface.
# The 'size' keyword argument is not accepted by the surface function.
cu_111_surface = surface('Cu', (1, 1, 1), layers=4, vacuum=15.0, a=3.61)

# Create the 3x3 supercell from the generated surface.
# The .repeat() method is used to create a supercell.
# (3, 3, 1) means repeating 3 times in the x-direction, 3 times in the y-direction,
# and 1 time in the z-direction (to keep the 4 layers we already defined).
cu_111_supercell = cu_111_surface.repeat((3, 3, 1))

# Define the filename
filename = "cu_111_surface_3x3x4_15A_vac.xyz"

# Save the structure
write(filename, cu_111_supercell)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")
