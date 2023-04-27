from lag_opt_lib.energy_and_grad import get_e_and_e_grad
from lag_opt_lib.logger import get_logger
from ase import Atoms
from ase.optimize import BFGS
import numpy as np
from mace.calculators import MACECalculator


def steepest_descent(coords0, step_size, eps, max_iter, calc_type=2, labels=(),
                     model_software='mace_model', output_dir=None,
                     ml_model_path=None):
    """
    Performs the steepest descent optimization on a set of atomic coordinates.

    Args:
    - coords0 (list/numpy array): initial atomic coordinates
    - step_size (float): step size for steepest descent
    - eps (float): energy convergence threshold
    - max_iter (int): maximum number of iterations
    - calc_type (int): energy calculation method
    - labels (list): list of atomic symbols

    Returns:
    - list: optimized atomic coordinates, energy, number of iterations, and optimization path
    """
    coords = np.asarray(coords0)  # convert initial coordinates to numpy array
    opt_path = [coords]  # initialize optimization path with initial coordinates
    path_e = []
    energy_old = 10  # set initial energy to a large value

    logger = get_logger('my_logger', log_file=output_dir + '/lag_opt.log')

    for i in range(max_iter):
        energy, grad = get_e_and_e_grad(coords, atomic_symbols=labels,
                                        model_software=model_software,
                                        calc_type=calc_type,
                                        output_dir=output_dir,
                                        ml_model_path=ml_model_path)  # calculate energy

        if abs(energy[0] - energy_old) < eps:  # check if energy has converged
            logger.debug('Minimum energy geometry:\n' + str(coords))
            logger.debug('Minimum energy value: ' + str(energy_old))
            logger.debug('The number of iterations: ' + str(i+1))
            return [coords, energy, i+1, opt_path, path_e]

        coords = coords - step_size * grad  # update coordinates using steepest descent
        opt_path.append(coords)  # add new coordinates to optimization path
        path_e.append(energy[0])
        energy_old = energy[0]  # update old energy

    logger.debug('Minimum energy geometry:\n' + str(coords))
    logger.debug('Minimum energy value: ' + str(energy_old))
    logger.debug('The number of iterations: ' + str(max_iter))
    return [coords, energy_old, max_iter, opt_path, path_e]


def convert_traj_to_xyz(input_file, output_file):
    from ase.io import read, write
    atoms = read(input_file, index=":", format="traj")
    write(output_file, atoms, format="xyz")


def ase_opt(coords0=None,
            labels=None,
            ml_model_path=None):
    calculator = MACECalculator(model_path=ml_model_path,
                                device='cpu')
    atoms = Atoms(symbols=labels,
                  positions=coords0)
    atoms.set_calculator(calculator)
    dyn = BFGS(atoms)
    dyn.run(fmax=0.05)
    opt_geom = atoms.copy()
    optimized_positions = opt_geom.get_positions()
    return optimized_positions.flatten()
