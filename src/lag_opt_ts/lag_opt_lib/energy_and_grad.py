from rdkit import Chem
from rdkit.Chem import AllChem
from lag_opt_lib.loggingFunctions import xyz_to_pdb, save_all_geoms
import re
import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, utils, torch_tools
from ase import Atoms


def get_e_and_e_grad(coords, atomic_symbols=None, model_software='mace_model',
                     calc_type=0, output_dir=None, ml_model_path=None):
    """
    Calculates the potential energy and gradient of a set of molecular coordinates using
    the Universal Force Field (UFF).

    Args:
    coords (ndarray): A 1D array of the atomic coordinates, given as [x1, y1, z1, x2, y2, z2,
    ..., xn, yn, zn].
    atomic_symbols (list, optional): A list of atomic symbols corresponding to the atoms in
    coords. Default is None.

    Returns:
    energy (float): The potential energy of the molecular system in units of kcal/mol.
    gradient (ndarray): The gradient of the potential energy with respect to the atomic
    coordinates. Has the same shape as coords, i.e. a 1D array of the form [dx1, dy1, dz1,
    dx2, dy2, dz2, ..., dxn, dyn, dzn].
    """
    # Write coordinates to an XYZ file and convert to PDB format
    energies, gradients = [], []
    if calc_type == 0:
        if model_software == 'rdkit':
            save_all_geoms(atomic_symbols, coords, log_file=output_dir + '/test.xyz')
            xyz_to_pdb()
            # Read in the PDB file as a molecule object and get the UFF force field
            mol = Chem.rdmolfiles.MolFromPDBFile('test.pdb', removeHs=False)
            ff = AllChem.UFFGetMoleculeForceField(mol)
            # Calculate the potential energy and gradient using the UFF force field
            energy = ff.CalcEnergy()
            gradient = ff.CalcGrad()
        elif model_software == 'mace_model':
            model = str(ml_model_path)
            atomic_numbers = [ase.data.atomic_numbers[s] for s in atomic_symbols]
            list_of_ase_configs = []
            if np.prod(np.shape(coords)) > 3 * len(atomic_symbols):
                for i in range(len(coords)):
                    list_of_ase_configs.append(Atoms(numbers=atomic_numbers,
                                                     positions=np.reshape(coords[i], (len(atomic_symbols), 3)),
                                                     ))
                energies, gradients = mace_wrapper(ml_model_path, list_of_ase_configs,
                                                   default_dtype='float64')
                energies = list(energies)
                gradients = [-i.flatten() for i in gradients]
            else:
                list_of_ase_configs.append(Atoms(numbers=atomic_numbers,
                                                 positions=np.reshape(coords, (len(atomic_symbols), 3))))
                energies, gradient = mace_wrapper(model, list_of_ase_configs, default_dtype='float64')
                gradients = -gradient[0].flatten()
    elif calc_type == 1:
        # Calculate energy using Schlegel's function
        for coord in coords:
            energy_coord = 10 * (coord[0] ** 4 + coord[1] ** 4 - 2 * coord[0] ** 2 -
                                 4 * coord[1] ** 2 + coord[0] * coord[1] + 0.2 * coord[0] + 0.1 * coord[1])
            gradient_coord = 10 * np.asarray([4 * coord[0] ** 3 - 4 * coord[0] + coord[1] + 0.2,
                                              4 * coord[1] ** 3 - 8 * coord[1] + coord[0] + 0.1])
            energies.append(energy_coord)
            gradients.append(gradient_coord)
    elif calc_type == 2:
        # Calculate energy using harmonic approximation
        for coord in coords:
            energy_coord = coord[0] ** 2 + coord[1] ** 2
            gradient_coord = np.asarray([2 * coord[0], 2 * coord[1]])
            energies.append(energy_coord)
            gradients.append(gradient_coord)
    return energies, gradients


def extract_e_f(n_atoms):
    with open('mace_output.txt', 'r') as file:
        inf = file.read()
    pattern = r'MACE_energy=(-?\d+\.\d+)'
    matches = re.findall(pattern, inf)
    energies, gradients = [0], [[0]]
    if matches:
        energies = [float(match) for match in matches]
    else:
        print('MACE_energy not found')
    pattern = r'^([A-Za-z]+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+' \
              r'([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)$'
    forces = []
    for line in inf.split('\n'):
        match = re.match(pattern, line)
        if match:
            force_values = tuple(map(float, match.group(5, 6, 7)))
            forces.append(force_values)
    gradients = [forces[i:i + n_atoms] for i in range(0, len(forces), n_atoms)]
    return energies, gradients


def get_path_e(path_coords,
               atomic_symbols=None,
               calc_type=None,
               all_path_e_file=None,
               output_dir=None,
               ml_model_path=None):
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
        all_path_e_file (str): File to save energies of all the paths.
        input_dir (str): A string representing the path to the directory containing input files for the calculation.
            Default is None.
        output_dir (str): A string representing the path to the directory to store output files for the calculation.
            Default is None.
        ml_model_path (str): A string representing the path to the machine learning model to use for the calculation.
            Default is None.

    Returns:
        list: A list of energies for each geometry in the path.
        list: A list of energy gradients for each geometry in the path.
    """
    all_e, all_e_grad = get_e_and_e_grad(path_coords,
                                         atomic_symbols=atomic_symbols,
                                         calc_type=calc_type,
                                         output_dir=output_dir,
                                         ml_model_path=ml_model_path)

    if all_path_e_file is not None:
        with open(all_path_e_file, 'a') as f:
            f.write(' '.join(np.asarray(all_e).astype(str)))
            f.write('\n')

    return all_e, all_e_grad


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


def mace_wrapper(model, list_of_ase_configs, default_dtype='double', device='cpu',
                 batch_size=64, compute_stress=False):
    """
    :param model: type(str), Required argument, path to the model
    :param list_of_ase_configs: type(str), Required argument, path to the xyz configurations
    :param default_dtype: type(str), choices[float32, float64, double (default)]
    :param device: type(str), choices['cpu (default), 'cuda']
    :param batch_size: type(int), batch size, default=64
    :param compute_stress: type(bool), compute stress, default=False
    :return: None
    """

    torch_tools.set_default_dtype(default_dtype)
    device = torch_tools.init_device(device)

    # Load model
    model = torch.load(model, map_location=device)

    configs = [data.config_from_atoms(atoms) for atoms in list_of_ase_configs]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max)) for config in configs],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_stress=compute_stress)
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        forces = np.split(torch_tools.to_numpy(output["forces"]), indices_or_sections=batch.ptr[1:], axis=0)
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [forces for forces_list in forces_collection for forces in forces_list]
    return [energies, forces_list]
