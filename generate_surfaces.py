
from ase.build import fcc100, fcc111, fcc110, graphene_nanoribbon, mx2
from ase import Atoms

def create_surface(element: str, face: str, size: tuple[int, int, int] = (3, 3, 4), vacuum: float = 10.0) -> Atoms:
    """
    Create a surface for a given element and face.
    
    Args:
        element: Chemical symbol (e.g., 'Cu', 'Pt', 'Au', 'C' for graphene)
        face: Surface face ('100', '111', '110', 'graphene', '2d')
        size: Tuple of (x, y, z) repetitions. z is number of layers.
        vacuum: Vacuum padding in Angstroms
        
    Returns:
        ASE Atoms object representing the surface
    """
    # Special case for Graphene
    if face.lower() == 'graphene' or (element == 'C' and face == '2d'):
        # Create a graphene sheet (using nanoribbon with pbc to make it infinite sheet)
        # size[0] and size[1] control width and length
        atoms = graphene_nanoribbon(size[0], size[1], type='zigzag', saturated=False, C_C=1.42, vacuum=vacuum)
        # By default nanoribbon is not periodic in all directions, let's make it periodic in x and y for a sheet
        atoms.pbc = [True, True, False]
        return atoms

    # Special case for TMDs (Transition Metal Dichalcogenides) like MoS2
    if face.lower() == '2d' and element in ['MoS2', 'WS2', 'MoSe2', 'WSe2']:
        formula = element
        # mx2 creates a 2D layer
        atoms = mx2(formula=formula, kind='2H', a=3.16, thickness=3.19, vacuum=vacuum)
        atoms = atoms.repeat((size[0], size[1], 1))
        return atoms

    # Approximate lattice constants for common metals
    lattice_constants = {
        'Cu': 3.61,
        'Pt': 3.92,
        'Au': 4.08,
        'Ag': 4.09,
        'Al': 4.05,
        'Ni': 3.52,
        'Pd': 3.89,
        'Fe': 2.87, # bcc usually, but let's keep it simple
        'Ir': 3.84,
        'Rh': 3.80
    }
    
    a = lattice_constants.get(element)
    if a is None:
        if element in lattice_constants:
             pass 
        else:
             # Default to 4.0 if unknown
             a = 4.0
    
    if face == '100':
        return fcc100(element, size=size, a=a, vacuum=vacuum)
    elif face == '111':
        return fcc111(element, size=size, a=a, vacuum=vacuum)
    elif face == '110':
        return fcc110(element, size=size, a=a, vacuum=vacuum)
    else:
        raise ValueError(f"Unsupported face: {face}. Choose from '100', '111', '110', 'graphene', '2d'.")

if __name__ == "__main__":
    # Test
    s = create_surface('Cu', '100')
    print(f"Created Cu(100) with {len(s)} atoms")