from ase.build import surface
from ase.io import write

# Create a Cu(100) surface with 1 layer
# Use 'material' instead of 'crystalstructure' and let ASE infer lattice constant
atoms = surface(material='Cu',
                miller=(1, 0, 0),
                layers=1, # 1x1x1 implies 1 layer
                vacuum=0) # Initially set vacuum to 0

# Set the vacuum layer
vacuum_thickness = 15.0
atoms.center(vacuum=vacuum_thickness, axis=2) # Center the slab and add vacuum along the z-axis (axis=2)

# Save the structure in XYZ format
filename = "cu_100_surface.xyz"
write(filename, atoms)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")