import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from loggingFunctions import write_xyz, xyz_to_pdb


def get_e_and_e_grad(coords, atomic_symbols=None):
    """
    Calculates the potential energy and gradient of a set of molecular coordinates using the Universal Force Field (UFF).

    Args:
    coords (ndarray): A 1D array of the atomic coordinates, given as [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn].
    atomic_symbols (list, optional): A list of atomic symbols corresponding to the atoms in coords. Default is None.

    Returns:
    energy (float): The potential energy of the molecular system in units of kcal/mol.
    gradient (ndarray): The gradient of the potential energy with respect to the atomic coordinates. Has the same shape as
                        coords, i.e. a 1D array of the form [dx1, dy1, dz1, dx2, dy2, dz2, ..., dxn, dyn, dzn].
    """

    # Write coordinates to an XYZ file and convert to PDB format
    write_xyz(coords, atomic_symbols)
    xyz_to_pdb()

    # Read in the PDB file as a molecule object and get the UFF force field
    mol = Chem.rdmolfiles.MolFromPDBFile('test.pdb', removeHs=False)
    ff = AllChem.UFFGetMoleculeForceField(mol)

    # Calculate the potential energy and gradient using the UFF force field
    energy = ff.CalcEnergy()
    gradient = ff.CalcGrad()

    return energy, gradient


def get_e(x, atomic_symbols=None, calc_type=0):
    """
    Calculates the energy of a molecular geometry using the UFF force field or a test function.

    Args:
        x (list): Cartesian coordinates of the molecular geometry
        atomic_symbols (list): List of atomic symbols (optional, only used if UFF force field is used)
        calc_type (int): Type of energy calculation to perform (0 for UFF force field, 1 for test function 1, 2 for test function 2)

    Returns:
        float: Energy of the molecular geometry
    """
    if calc_type == 0:
        # Calculate energy using UFF force field
        return get_e_and_e_grad(x, atomic_symbols=atomic_symbols)[0]
    elif calc_type == 1:
        # Calculate energy using Schlegel's function
        return 10 * (x[0] ** 4 + x[1] ** 4 - 2 * x[0] ** 2 - 4 * x[1] ** 2 + x[0] * x[1] + 0.2 * x[0] + 0.1 * x[1])
    elif calc_type == 2:
        # Calculate energy using harmonic approximation
        return x[0] ** 2 + x[1] ** 2


def get_e_grad(x, atomic_symbols=None, calc_type=0):
    """
    Calculates the energy gradient at a given set of atomic coordinates using the UFF force field.

    Args:
    - x (ndarray): 1D numpy array of atomic coordinates.
    - atomic_symbols (list, optional): List of atomic symbols in the same order as the atomic coordinates. Default is None.
    - calc_type (int, optional): The type of energy calculation to perform. Default is 0.

    Returns:
    - ndarray: 1D numpy array of energy gradients at the given set of atomic coordinates.
    """
    if calc_type == 0:
        # If using the UFF force field
        return np.asarray(get_e_and_e_grad(x, atomic_symbols=atomic_symbols)[1]).flatten()
    elif calc_type == 1:
        # If using the Schlegel's function
        return 10 * np.asarray([4 * x[0] ** 3 - 4 * x[0] + x[1] + 0.2, 4 * x[1] ** 3 - 8 * x[1] + x[0] + 0.1])
    elif calc_type == 2:
        # If using the simple quadratic function
        return np.asarray([2 * x[0], 2 * x[1]])


def get_path_e(path_coords, atomic_symbols=None, calc_type=None):
    """
    Compute the energy of a path of geometries.

    Args:
        path_coords (numpy.ndarray): A numpy array of shape (n_geoms, 3 * n_atoms) representing the coordinates
            of the path of geometries. n_geoms is the number of geometries and n_atoms is the number of atoms
            in each geometry.
        atomic_symbols (list): A list of atomic symbols. Default is None.
        calc_type (int): An integer specifying the type of calculation. Default is None.

            - calc_type=0: Calculate the energy using a molecular mechanics force field (UFF).
            - calc_type=1: Calculate the energy using the Rosenbrock function.
            - calc_type=2: Calculate the energy using the harmonic oscillator potential.

    Returns:
        list: A list of energies for each geometry in the path.
    """
    all_e = []
    n_atoms = int(len(path_coords[0])/3)
    n_geoms = len(path_coords)
    path_coords = np.reshape(path_coords, (n_geoms, 3 * n_atoms))
    for i in path_coords:
        geom = np.reshape(i, (n_atoms, 3))
        all_e.append(get_e(geom, atomic_symbols=atomic_symbols, calc_type=calc_type))
    return all_e


def grid_gen(x_min, x_max, y_min, y_max, delta):
    """
    Generate a grid of points with x and y values within given ranges and a given spacing.

    Args:
        x_min (float): The minimum x value.
        x_max (float): The maximum x value.
        y_min (float): The minimum y value.
        y_max (float): The maximum y value.
        delta (float): The spacing between adjacent points in the grid.

    Returns:
        list: A list containing two arrays representing the x and y values of the grid points.

    """
    # Generate arrays of x and y values using the given ranges and spacing
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)

    # Create a meshgrid of x and y values to form a grid of points
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    return [x_grid, y_grid]
