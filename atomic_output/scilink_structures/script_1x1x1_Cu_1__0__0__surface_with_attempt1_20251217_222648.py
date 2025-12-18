from ase.build import surface
from ase.io import write

# Build the Cu(100) surface with 3 layers
# The 'size' parameter for surface is (lateral_x, lateral_y, num_layers)
cu_100_surface = surface(metal='Cu', miller=(1, 0, 0), size=(1, 1, 3), vacuum=15.0)

# Define the output filename
output_filename = "cu_100_surface.xyz"

# Save the structure in XYZ format
write(output_filename, cu_100_surface)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{output_filename}")