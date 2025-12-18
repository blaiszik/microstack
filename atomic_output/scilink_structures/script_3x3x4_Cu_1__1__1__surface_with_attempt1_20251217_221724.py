from ase.build import surface
from ase.io import write

# Build the Cu(111) surface with a 3x3x4 supercell and 15.0 Å vacuum
# Corrected: The 'surface' function expects material and miller_indices as positional arguments,
# not 'name' as a keyword argument.
cu_111_surface = surface('Cu', (1, 1, 1), size=(3, 3, 4), vacuum=15.0, a=3.61)

# Define the filename
filename = "cu_111_surface_3x3x4_15A_vac.xyz"

# Save the structure
write(filename, cu_111_surface)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")