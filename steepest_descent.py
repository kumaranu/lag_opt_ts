import numpy as np
from energy_and_grad import get_e, get_e_grad


def steepest_descent(coords0, alpha, eps, max_iter, calc_type=2, labels=()):
    """
    Performs the steepest descent optimization on a set of atomic coordinates.

    Args:
    - coords0 (list/numpy array): initial atomic coordinates
    - alpha (float): step size for steepest descent
    - eps (float): energy convergence threshold
    - max_iter (int): maximum number of iterations
    - calc_type (int): energy calculation method
    - labels (list): list of atomic symbols

    Returns:
    - list: optimized atomic coordinates, energy, number of iterations, and optimization path
    """
    coords = np.asarray(coords0)  # convert initial coordinates to numpy array
    opt_path = [coords]  # initialize optimization path with initial coordinates
    energy_old = 10  # set initial energy to a large value
    for i in range(max_iter):
        energy = get_e(coords, atomic_symbols=labels, calc_type=calc_type)  # calculate energy
        grad = get_e_grad(coords, atomic_symbols=labels, calc_type=calc_type)  # calculate gradient
        if abs(energy - energy_old) < eps:  # check if energy has converged
            print('minimum:', coords)  # print optimized coordinates
            print('Minimum energy value:', energy_old)  # print optimized energy
            print('num_iter:', max_iter)  # print number of iterations
            return [coords, energy, i+1, opt_path]  # return optimized coordinates, energy, number of iterations,
            # and optimization path

        coords = coords - alpha * grad  # update coordinates using steepest descent
        opt_path.append(coords)  # add new coordinates to optimization path
        energy_old = energy  # update old energy
    # print('opt_path:', opt_path)
    print('minimum:', coords)  # print optimized coordinates
    print('Minimum energy value:', energy_old)  # print optimized energy
    print('num_iter:', max_iter)  # print number of iterations
    return [coords, energy_old, max_iter, opt_path]  # return optimized coordinates, energy, number of iterations,
    # and optimization path
