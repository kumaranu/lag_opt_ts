import numpy as np
import os
import readIn


def compute_initial_points(start, end, number_of_points):
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]
    points = [start * (1 - t) + end * t for t in ts]
    return np.stack(points)


def get_initial_path(calc_type, minima1, special_point, n_images, minima2,
                     path_to_a_script_to_call_geodesic_code=None, xyz_r=None,
                     xyz_p=None, xyz_path=None, xyz_r_p=None, atomic_symbols=None):
    if calc_type == 1:
        initial_points1 = compute_initial_points(np.asarray(minima1), np.asarray(special_point),
                                                 np.asarray(n_images))
        initial_points2 = compute_initial_points(np.asarray(special_point), np.asarray(minima2),
                                                 np.asarray(n_images))
        initial_points = np.vstack([initial_points1, initial_points2])
        return [minima1, minima2, initial_points, atomic_symbols]
    elif calc_type == 0:
        '''
        This line right below should be converted to an in-memory call instead of system call.
        '''

        reference_dir = os.getcwd()
        os.chdir(path_to_a_script_to_call_geodesic_code)
        os.system('geodesic_interpolate ' + str(xyz_r_p) + ' --output output.xyz --nimages ' + str(n_images))
        os.chdir(reference_dir)

        atomic_symbols_r, coords_r = readIn.read_in_one_xyz_file(xyz_r)
        atomic_symbols_p, coords_p = readIn.read_in_one_xyz_file(xyz_p)
        # atomic_symbols_path, coords_path = readIn.read_in_multiple_xyz_file(xyz_path, n_images=n_images)
        atomic_symbols_path, coords_path = readIn.read_in_multiple_xyz_file(
            path_to_a_script_to_call_geodesic_code + '/output.xyz', n_images=n_images)
        n_atoms = len(coords_r)
        coords_r = coords_r.flatten()
        coords_p = coords_p.flatten()
        coords_path = np.reshape(coords_path, (n_images, 3*n_atoms))
        return [coords_r, coords_p, coords_path, atomic_symbols]


def action1(f, point_l, point_r, n=3):
    dr = point_r - point_l
    df = f(point_r) - f(point_l)
    dist_component = ((dr**n).sum())**(1/n)
    graph_component = df ** 2
    action_var = dist_component + graph_component
    return action_var


def grad_action_l1(f, grad_f, point_l, point_r, n=3):
    dr = (point_r - point_l)
    df = (f(point_r) - f(point_l))
    dist_component_grad = -1 * n * dr**(n-1)
    graph_component_grad = -2 * df * grad_f(point_l)
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def grad_action_r1(f, grad_f, point_l, point_r, n=3):
    dr = (point_r - point_l)
    df = (f(point_r) - f(point_l))
    dist_component_grad = 2 * n * dr**(n-1)
    graph_component_grad = 2 * df * grad_f(point_r)
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def action(f, point_l, point_r, a):
    dr = point_r - point_l
    df = f(point_r) - f(point_l)
    dist_component = np.exp(a * (dr**2).sum()) - 1.0
    graph_component = df ** 2
    action_var = dist_component + graph_component
    return action_var


def grad_action_l(calc_type, atomic_symbols, point_l, point_r, a):
    from energy_and_grad_calls_rdkit import get_e, get_e_grad
    dr = point_r - point_l
    df = get_e(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)-get_e(point_l,
                                                                                  atomic_symbols=atomic_symbols,
                                                                                  calc_type=calc_type)
    dist_component_grad = -1 * np.exp(a * (dr**2).sum()) * a * 2 * dr
    graph_component_grad = -2 * df * get_e_grad(point_l, atomic_symbols=atomic_symbols, calc_type=calc_type)
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def grad_action_r(calc_type, atomic_symbols, point_l, point_r, a):
    dr = (point_r - point_l)
    from energy_and_grad_calls_rdkit import get_e, get_e_grad
    df = (get_e(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)
          - get_e(point_l, atomic_symbols=atomic_symbols, calc_type=calc_type))
    dist_component_grad = np.exp(a * (dr**2).sum()) * a * 2 * dr
    graph_component_grad = 2 * df * get_e_grad(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def lagrangian(f, all_pts, a):
    lag = 0
    for i in range(len(all_pts)-1):
        lag += action(f, all_pts[i], all_pts[i+1], a)
    return lag


def grad_lagrangian(calc_type, atomic_symbols, all_pts, a):
    grad_lag = np.zeros((all_pts.shape[0]-2, all_pts.shape[1]))
    for i in range(len(all_pts)-2):
        grad_lag[i] = np.array(grad_action_r(calc_type, atomic_symbols, all_pts[i], all_pts[i+1], a)) + \
                      np.array(grad_action_l(calc_type, atomic_symbols, all_pts[i+1], all_pts[i+2], a))
        # grad_lag[i] = np.array(grad_action_r1(f, grad_f, all_pts[i], all_pts[i + 1], a)) + \
        #          np.array(grad_action_l1(f, grad_f, all_pts[i + 1], all_pts[i + 2], a))
    return grad_lag


def find_critical_path(calc_type, atomic_symbols=None, initial_points=None, start=None, end=None, num_steps=None,
                       step_factor=None, a=None):
    if calc_type == 0:
        result, points = [], initial_points
        for step in range(num_steps):
            all_pts = np.vstack((start, points, end))
            points = points - step_factor * grad_lagrangian(calc_type, atomic_symbols, all_pts, a)
        np.savetxt('outputPath.txt', points, fmt='%.5f')
    elif calc_type == 1:
        result, points = [], initial_points
        print('points:', points)
        for step in range(num_steps):
            all_pts = np.vstack((start, points, end))
            print('all_pts:', all_pts)
            points = points - step_factor * grad_lagrangian(calc_type, atomic_symbols, all_pts, a)
        np.savetxt('outputPath.txt', points, fmt='%.5f')
    return points
