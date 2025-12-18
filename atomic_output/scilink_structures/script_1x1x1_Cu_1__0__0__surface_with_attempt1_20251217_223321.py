from ase.build import fcc100
from ase.io import write

# Create a 1x1x1 Cu(100) surface with 15.0A vacuum
# fcc100(symbol, size, layers, vacuum)
# size=(1,1) for a 1x1 surface unit cell
# layers=1 for a single layer slab
atoms = fcc100(symbol='Cu', size=(1, 1), layers=1, vacuum=15.0)

# Save the structure in XYZ format
filename = "cu_100_surface.xyz"
write(filename, atoms)

# Print the confirmation line
print(f"STRUCTURE_SAVED:{filename}")