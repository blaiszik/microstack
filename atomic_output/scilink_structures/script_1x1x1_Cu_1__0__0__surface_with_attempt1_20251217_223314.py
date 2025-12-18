from ase.build import surface, bulk
from ase.io import write

# Create bulk copper first
# ASE can infer the lattice constant for common materials like 'Cu' with 'fcc'
bulk_cu = bulk('Cu', 'fcc')

# Create a Cu(100) surface with 1 layer from the bulk structure
atoms = surface(bulk_cu,
                miller=(1, 0, 0),
                layers=1, # 1x1x1 implies 1 layer in terms of surface thickness
                vacuum=0) # Initially set vacuum to 0

# Set the vacuum layer
vacuum_thickness = 15.0
atoms.center(vacuum=vacuum_thickness, axis=2) # Center the slab and add vacuum along the z-axis (axis=2)

# Save the structure in XYZ format
filename = "cu_100_surface.xyz"
write(filename, atoms)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")