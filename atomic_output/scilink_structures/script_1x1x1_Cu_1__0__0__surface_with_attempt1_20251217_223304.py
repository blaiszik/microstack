from ase.build import surface
from ase.io import write

# Create a Cu(100) surface with 3 layers
# The 'size' parameter [1, 1, 3] means 1x1 surface unit cell and 3 layers in the z-direction.
atoms = surface(crystalstructure='fcc',
                latticeconstant=3.61,  # Lattice constant for Cu
                miller=(1, 0, 0),
                layers=3,
                vacuum=0) # Initially set vacuum to 0, we'll adjust it later

# Set the vacuum layer
vacuum_thickness = 15.0
atoms.center(vacuum=vacuum_thickness, axis=2) # Center the slab and add vacuum along the z-axis (axis=2)

# Save the structure in XYZ format
filename = "cu_100_surface.xyz"
write(filename, atoms)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")