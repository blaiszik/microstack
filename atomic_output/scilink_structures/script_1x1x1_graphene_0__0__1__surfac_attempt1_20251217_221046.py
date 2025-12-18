
from ase.build import graphene
from ase.io import write

# Create 1x1x1 graphene unit cell with 10.0 Angstrom vacuum
atoms = graphene(size=(1, 1, 1), vacuum=10.0)

# Define filename
filename = 'graphene_surface.xyz'

# Save the structure
write(filename, atoms)

# Print confirmation
print(f'STRUCTURE_SAVED:{filename}')
