import numpy as np
from energy_and_grad import get_e, get_e_grad


def steepest_descent(coords0, alpha, eps, max_iter, calc_type=2, labels=()):
    coords = np.asarray(coords0)
    opt_path = [coords]
    energy_old = 10
    for i in range(max_iter):
        energy = get_e(coords, atomic_symbols=labels, calc_type=calc_type)
        grad = get_e_grad(coords, atomic_symbols=labels, calc_type=calc_type)
        if abs(energy - energy_old) < eps:
            print('minimum:', coords)
            print('Minimum energy value:', energy_old)
            print('num_iter:', max_iter)
            return [coords, energy, i+1, opt_path]
        coords = coords - alpha * grad
        opt_path.append(coords)
        energy_old = energy
    # print('opt_path:', opt_path)
    print('minimum:', coords)
    print('Minimum energy value:', energy_old)
    print('num_iter:', max_iter)
    return [coords, energy_old, max_iter, opt_path]
