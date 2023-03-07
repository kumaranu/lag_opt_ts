import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from loggingFunctions import write_xyz, xyz_to_pdb


def get_e_and_e_grad(coords, atomic_symbols=None):
    write_xyz(coords, atomic_symbols)
    xyz_to_pdb()
    mol = Chem.rdmolfiles.MolFromPDBFile('test.pdb', removeHs=False)
    ff = AllChem.UFFGetMoleculeForceField(mol)
    energy = ff.CalcEnergy()
    gradient = ff.CalcGrad()
    return energy, gradient


def get_e(x, atomic_symbols=None, calc_type=0):
    if calc_type == 0:
        return get_e_and_e_grad(x, atomic_symbols=atomic_symbols)[0]
    elif calc_type == 1:
        return 10 * (x[0] ** 4 + x[1] ** 4 - 2 * x[0] ** 2 - 4 * x[1] ** 2 + x[0] * x[1] + 0.2 * x[0] + 0.1 * x[1])
    elif calc_type == 2:
        return x[0] ** 2 + x[1] ** 2


def get_e_grad(x, atomic_symbols=None, calc_type=0):
    if calc_type == 0:
        return np.asarray(get_e_and_e_grad(x, atomic_symbols=atomic_symbols)[1]).flatten()
    elif calc_type == 1:
        return 10 * np.asarray([4 * x[0] ** 3 - 4 * x[0] + x[1] + 0.2, 4 * x[1] ** 3 - 8 * x[1] + x[0] + 0.1])
    elif calc_type == 2:
        return np.asarray([2 * x[0], 2 * x[1]])


def get_path_e(path_coords, atomic_symbols=None, calc_type=None):
    all_e = []
    n_atoms = int(len(path_coords[0])/3)
    n_geoms = len(path_coords)
    path_coords = np.reshape(path_coords, (n_geoms, 3 * n_atoms))
    for i in path_coords:
        geom = np.reshape(i, (n_atoms, 3))
        all_e.append(get_e(geom, atomic_symbols=atomic_symbols, calc_type=calc_type))
    return all_e


def grid_gen(x_min, x_max, y_min, y_max, delta):
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    x1, y1 = np.meshgrid(x, y, indexing='ij')
    return [x1, y1]
