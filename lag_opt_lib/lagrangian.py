# import copy
import sys

import numpy as np
import os
from lag_opt_lib import readIn
from lag_opt_lib.energy_and_grad import get_path_e
from lag_opt_lib.compare import e_r_diff
from lag_opt_lib.loggingFunctions import save_all_geoms

# from lag_opt_lib.logger import get_logger
# logger = get_logger('my_logger', 'lag_opt.log')


def get_initial_path(calc_type,
                     minima1,
                     special_point,
                     n_images,
                     minima2,
                     geodesic_code=None,
                     xyz_r_p=None,
                     atomic_symbols=None,
                     initial_path_type=1):
    """
    Generates the initial path for the optimization.

    Parameters:
    -----------
    calc_type : int
        The type of calculation to perform. 0 for geodesic path optimization, 1 for linear interpolation.
    minima1 : array_like
        The coordinates of the first minimum on the PES.
    special_point : array_like
        The coordinates of the special point on the PES.
    n_images : int
        The number of images to use for optimization.
    minima2 : array_like
        The coordinates of the second minimum on the PES.
    path_to_a_script_to_call_geodesic_code : str or None, optional
        The path to a script to call the geodesic code for calculation (only used if calc_type is 0).
        Default is None.
    xyz_r : str or None, optional
        The path to the reactant geometry file (only used if calc_type is 0). Default is None.
    xyz_p : str or None, optional
        The path to the product geometry file (only used if calc_type is 0). Default is None.
    xyz_path : str or None, optional
        The path to the geometry path file (only used if calc_type is 0). Default is None.
    xyz_r_p : str or None, optional
        The path to the combined reactant-product geometry file (only used if calc_type is 0). Default is None.
    atomic_symbols : list or None, optional
        The list of atomic symbols in the molecule. Default is None.

    Returns:
    --------
    list
        The list containing the reactant, product, path coordinates and atomic symbols.
    """
    if calc_type == 1:
        initial_points = []
        if initial_path_type == 1:
            initial_points = np.linspace(minima1, minima2, n_images+2)[1:-1]
        elif initial_path_type == 0:
            initial_points =\
                np.vstack((minima1,
                           np.linspace(minima1, special_point, n_images)[1:],
                           np.linspace(special_point, minima2, n_images)[1:]))
        return [minima1, minima2, initial_points, atomic_symbols]
    elif calc_type == 0:
        '''
        This line right below should be converted to an in-memory call instead of system call.
        '''
        # Geodesic path optimization
        # Call the geodesic code
        reference_dir = os.getcwd()
        os.chdir(geodesic_code)
        os.system('geodesic_interpolate ' + str(xyz_r_p) + ' --output output.xyz --nimages ' + str(n_images))
        os.chdir(reference_dir)

        # Read in the reactant, product and path geometries
        atomic_symbols, coords_r_p = \
            readIn.read_in_multiple_xyz_file(xyz_r_p,
                                             n_images=n_images)
        # atomic_symbols, coords_r_p =\
        #    readIn.read_in_multiple_xyz_file(xyz_path,
        #                                     n_images=n_images)
        atomic_symbols_path, coords_path =\
            readIn.read_in_multiple_xyz_file(geodesic_code + '/output.xyz',
                                             n_images=n_images)
        n_atoms = len(coords_r_p[0])
        coords_r = coords_r_p[0].flatten()
        coords_p = coords_r_p[1].flatten()
        coords_path = np.reshape(coords_path,
                                 (n_images, 3*n_atoms))
        return [coords_r, coords_p, coords_path, atomic_symbols]


def action(point_l, point_r, energy_l, energy_r, a, action_type=1):
    """
    Calculate the action variable for a given function f and two points point_l and point_r.

    Args:
    - f: a function that takes a point in n-dimensional space and returns a scalar value
    - point_l: a numpy array representing the coordinates of the left endpoint of the interval
    - point_r: a numpy array representing the coordinates of the right endpoint of the interval
    - a: a scalar value representing the strength of the distance component in the action variable

    Returns:
    - action_var: a scalar value representing the action variable for the given interval
    """
    dr = point_r - point_l
    df = energy_r - energy_l
    dist_component = 0
    if action_type == 0:
        dist_component = np.exp(a * (dr**2).sum()) - 1.0
    elif action_type == 1:
        dist_component = a * (dr**2).sum()
    potential_component = df ** 2
    action_var = dist_component + potential_component
    return action_var


def grad_action_l(point_l, point_r, energy_l, energy_r, grad_l, a, action_type=1):
    """
    Computes the gradient of the action with respect to the left point of the interval.

    Args:
        point_l (ndarray): Left point of the interval, shape (3 * n_atoms,).
        point_r (ndarray): Right point of the interval, shape (3 * n_atoms,).
        energy_l (float): Energy at the left point of the interval.
        energy_r (float): Energy at the right point of the interval.
        grad_l (ndarray): Gradient of the energy at the left point of the interval, shape (3 * n_atoms,).
        a (float): Exponential factor for the distance component of the action.
        action_type (int, optional): Type of action to compute the gradient for (0 for exponential, 1 for linear).
            Default is 1.

    Returns:
        action_grad_var (ndarray): Gradient of the action with respect to the left point, shape (3 * n_atoms,).
    """
    # Compute the difference between the right and left points
    d_r = point_r - point_l

    # Compute the difference in energy between the right and left points
    d_e = energy_r - energy_l
    dist_component_grad = np.zeros(len(d_r))
    if action_type == 0:
        # Compute the gradient of the distance component of the action
        dist_component_grad = -1 * np.exp(a * (d_r**2).sum()) * a * 2 * d_r
    elif action_type == 1:
        dist_component_grad = -1 * a * 2 * d_r

    # Compute the gradient of the graph component of the action
    graph_component_grad = -2 * d_e * grad_l

    # Compute the gradient of the action with respect to the left point
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def grad_action_r(point_l, point_r, energy_l, energy_r, grad_r, a, action_type=1):
    """
    Computes the gradient of the action with respect to the right point of the interval.

    Args:
        point_l (ndarray): Left point of the interval, shape (3 * n_atoms,).
        point_r (ndarray): Right point of the interval, shape (3 * n_atoms,).
        energy_l (float): Energy at the left point of the interval.
        energy_r (float): Energy at the right point of the interval.
        grad_r (ndarray): Gradient of the energy at the right point of the interval, shape (3 * n_atoms,).
        a (float): Exponential factor for the distance component of the action.
        action_type (int, optional): Type of action to compute the gradient for (0 for exponential, 1 for linear).
            Default is 1.

    Returns:
        action_grad_var (ndarray): Gradient of the action with respect to the right point, shape (3 * n_atoms,).
    """
    d_r = (point_r - point_l)
    d_e = energy_r - energy_l
    dist_component_grad = np.zeros(len(d_r))
    if action_type == 0:
        dist_component_grad = np.exp(a * (d_r**2).sum()) * a * 2 * d_r
    elif action_type == 1:
        dist_component_grad = a * 2 * d_r
    graph_component_grad = 2 * d_e * grad_r
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def lagrangian(all_pts, all_e, a):
    """
    Compute the Lagrangian for a given function and set of points.

    Args:
    - f: the function to evaluate.
    - all_pts: a list of points that define the action.
    - a: a constant value.

    Returns:
    - The Lagrangian value computed for the function and set of points.
    """

    # Initialize the Lagrangian value to zero.
    lag = 0

    # Compute the action for each interval defined by the points.
    for i in range(len(all_pts)-1):

        # Add the action for the current interval to the Lagrangian value.
        lag += action(all_pts[i], all_pts[i+1], all_e[i], all_e[i+1], a)

    # Return the final Lagrangian value.
    return lag


def grad_lagrangian(all_pts, a, path_e, path_e_grad, action_type=1):
    """
    Compute the gradient of the Lagrangian for a given set of points.

    Args:
    - calc_type: a string indicating which calculation type to use.
    - atomic_symbols: a list of atomic symbols.
    - all_pts: a numpy array of points that define the action.
    - a: a constant value.

    Returns:
    - A numpy array representing the gradient of the Lagrangian.

    """
    # Create an array to hold the gradient of the Lagrangian.
    grad_lag = np.zeros((all_pts.shape[0]-2, all_pts.shape[1]))
    # print('\nShapes of key variables')
    # print('np.shape(all_pts):', np.shape(all_pts))
    # print('np.shape(path_e):', np.shape(path_e))
    # print('path_e:', path_e)
    # print('np.shape(path_e_grad):', np.shape(path_e_grad))
    # print('np.shape(a):', np.shape(a))

    # Compute the gradient of the action for each interval defined by the points.
    for i in range(len(all_pts)-2):
        # print('i', i)
        # Compute the gradient of the action for the left and right intervals and add them up.
        grad_lag[i] =\
            np.array(grad_action_r(all_pts[i],
                                   all_pts[i+1],
                                   path_e[i],
                                   path_e[i+1],
                                   np.asarray(path_e_grad[i+1]).flatten(),
                                   a[i],
                                   action_type=action_type)) + \
            np.array(grad_action_l(all_pts[i+1],
                                   all_pts[i+2],
                                   path_e[i+1],
                                   path_e[i+2],
                                   np.asarray(path_e_grad[i+1]).flatten(),
                                   a[i+1],
                                   action_type=action_type))

    # Return the computed gradient of the Lagrangian.
    return grad_lag


def parallel_perpendicular(u, v):
    proj_v_u = np.dot(u, v) / np.dot(v, v) * v
    perpendicular_v_u = u - proj_v_u
    return proj_v_u, perpendicular_v_u


def path_grad_components(path, grads, perpendicular_type=0, n=10):
    parallel_grads_list, perpendicular_grads_list = [], []
    if perpendicular_type == 0:
        for i in range(1, len(path)-1):
            v = path[i+1] - path[i-1]
            u = grads[i]
            proj_v_u, perpendicular_v_u = parallel_perpendicular(u, v)
            parallel_grads_list.append(proj_v_u)
            perpendicular_grads_list.append(perpendicular_v_u)
    elif perpendicular_type == 1:
        for i in range(1, len(path)-1):
            if i < n:
                v = (path[int(len(path)/2)] - path[0])
            elif i > (len(path)-1-n):
                v = (path[-1] - path[int(len(path)/2)])
            else:
                v = path[i + n] - path[i - n]
            u = grads[i]
            proj_v_u, perpendicular_v_u = parallel_perpendicular(u, v)
            parallel_grads_list.append(proj_v_u)
            perpendicular_grads_list.append(perpendicular_v_u)

    return parallel_grads_list, perpendicular_grads_list


def re_meshing(path,
               path_e,
               path_e_grad,
               a,
               atomic_symbols,
               calc_type,
               ml_model_path,
               re_meshing_type=1,
               change_factor=0.1):
    """
    Refine the path of geometries by adding a new point at the maximum energy point and
    removing the two points adjacent to it, then insert two new points on either side
    of the maximum energy point.

    Args:
        path (numpy.ndarray): A numpy array of shape (n_geoms, 3 * n_atoms) representing the
            coordinates of the path of geometries.
        path_e (list): A list of energies for each geometry in the path.
        path_e_grad (list): A list of energy gradients for each geometry in the path.
        a (numpy.ndarray): A numpy array of shape (n_geoms, 3 * n_atoms) representing the
            forces acting on each atom in the path of geometries.
        atomic_symbols (list): A list of atomic symbols.
        calc_type (int): An integer specifying the type of calculation.

            - calc_type=0: Calculate the energy using a molecular mechanics force field (UFF).
            - calc_type=1: Calculate the energy using the Rosenbrock function.
            - calc_type=2: Calculate the energy using the harmonic oscillator potential.

        ml_model_path (str): The path to the saved machine learning model used to calculate the
            energy and energy gradient of the geometries.
        re_meshing_type (int): An integer to choose the type of meshing to be
        performed. re_meshing=0 means no re-meshing at all.
        re_meshing=1 means re-meshing without changing the force constants.
        re_meshing=2 means re-meshing with weights changed according to a Gaussian
        function.
        change_factor (float): An float specifying the range of Gaussian function
        that is multiplied with the force constants. The range of the Gaussian
        function is (1-change_factor, 1+change_factor). Default is 0.1.

    Returns:
        tuple: A tuple containing the following:

            - path (numpy.ndarray): A numpy array of shape (n_geoms, 3 * n_atoms) representing the
              coordinates of the refined path of geometries.
            - a (numpy.ndarray): A numpy array of shape (n_geoms, 3 * n_atoms) representing the
              forces acting on each atom in the refined path of geometries.
            - path_e (list): A list of energies for each geometry in the refined path.
            - path_e_grad (list): A list of energy gradients for each geometry in the refined path.
    """
    # print('Inside re-meshing')
    # print('np.shape(path):', np.shape(path))
    # print('1:', path_e)
    index_max = np.argmax(path_e)

    mid_1 = 0.5 * (path[index_max-1] + path[index_max])
    mid_2 = 0.5 * (path[index_max] + path[index_max+1])

    e_m1_m2, e_grad_m1_m2 = get_path_e(np.asarray([mid_1, mid_2]),
                                       atomic_symbols=atomic_symbols,
                                       calc_type=calc_type,
                                       ml_model_path=ml_model_path)
    
    m1_e, m1_e_grad, m2_e, m2_e_grad = \
        e_m1_m2[0], e_grad_m1_m2[0], e_m1_m2[1], e_grad_m1_m2[1]

    # print('path_e A:', *path_e, sep='\n')
    path_e.insert(index_max+1, m2_e)
    path_e.insert(index_max, m1_e)
    path_e.pop(1)
    path_e.pop(-2)
    # print('path_e B:', *path_e, sep='\n')
    # print('2:', path_e)

    # print('path_e_grad A:', *path_e_grad, sep='\n')
    path_e_grad.insert(index_max+1, m2_e_grad)
    path_e_grad.insert(index_max, m1_e_grad)
    path_e_grad.pop(1)
    path_e_grad.pop(-2)
    # print('path_e_grad B:', *path_e_grad, sep='\n')

    # Convert 1D arrays to 2D arrays
    start = path[0][np.newaxis, :]
    end = path[-1][np.newaxis, :]
    # print('index_max:', index_max)
    # Stack all arrays
    path = np.vstack((start,
                      path[2:index_max],
                      mid_1[np.newaxis, :],
                      path[index_max],
                      mid_2[np.newaxis, :],
                      path[(index_max+1):-2],
                      end))
    # print('np.shape(path):', np.shape(path))

    if re_meshing_type == 2:
        a = change_forces(a,
                          index_max,
                          change_factor=change_factor,
                          change_force_type=2)
    return path, path_e, path_e_grad, a


def change_forces(a, max_n_index, change_factor=0.3, change_force_type=1):
    if change_force_type == 0:
        return a
    elif change_force_type == 1:
        diminishing_factor = 1 - change_factor
        increasing_factor = 1 + change_factor
        a[1] = a[1] * diminishing_factor
        a[-2] = a[-2] * diminishing_factor
        a[max_n_index] = a[max_n_index] * increasing_factor
        a[max_n_index + 1] = a[max_n_index + 1] * increasing_factor
        return a
    elif change_force_type == 2:
        gauss = [gaussian(i, max_n_index, 2) + 1 - change_factor
                 for i in np.arange(len(a))]
        print('gaussian_function:', *gauss, sep='\n')
        print('change_factor:', change_factor)
        return np.asarray(a) * np.asarray(gauss)


def gaussian(x, mu, sigma):
    """
    Generate a Gaussian function.

    Args:
        x (float or array): Input values at which to evaluate the function.
        mu (float): Mean value of the function.
        sigma (float): Standard deviation of the function.

    Returns:
        The value(s) of the Gaussian function at the given input value(s).
    """
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def find_critical_path(calc_type,
                       atomic_symbols=None,
                       initial_points=None,
                       start=None,
                       end=None,
                       num_steps=None,
                       step_factor=None,
                       action_type=1,
                       path_grad_type=1,
                       a=None,
                       convergence_type=1,
                       eps=0.001,
                       re_meshing_type=2,
                       change_factor=0.1,
                       all_path_e_file='all_path_e.txt',
                       output_path_file='output_path.txt',
                       input_dir=None,
                       output_dir=None,
                       ml_model_path=None,
                       re_mesh_frequency=5):
    """
    Find the critical path using gradient descent.

    Args:
    - calc_type: a number indicating which calculation type to use.
    - atomic_symbols: a list of atomic symbols.
    - initial_points: a numpy array of points.
    - start: a numpy array representing the start point.
    - end: a numpy array representing the end point.
    - num_steps: the number of steps to take.
    - step_factor: the factor by which to scale the step size.
    - a: a constant value.

    Returns:
    - A numpy array representing the points on the critical path.

    """
    # The following if conditions will go away soon.
    result, points = [], initial_points
    path_e_old_max = np.inf
    points_old = np.zeros(np.shape(points))
    e_r_p, e_grad_r_p = get_path_e(np.asarray([start, end]),
                                   atomic_symbols=atomic_symbols,
                                   calc_type=calc_type,
                                   ml_model_path=ml_model_path)
    start_e, start_e_grad, end_e, end_e_grad =\
        e_r_p[0], e_grad_r_p[0], e_r_p[1], e_grad_r_p[1]
    all_path_e, all_path_coords = [], []

    a_list = [a] * (len(initial_points) + 1)
    path_dists = []
    for step in range(num_steps):
        print('step:', step)
        path_e, path_e_grad = get_path_e(points,
                                         atomic_symbols=atomic_symbols,
                                         calc_type=calc_type,
                                         all_path_e_file=all_path_e_file,
                                         output_dir=output_dir,
                                         ml_model_path=ml_model_path)

        path_e.insert(0, start_e)
        path_e.append(end_e)
        path_e_grad.insert(0, start_e_grad)
        path_e_grad.append(end_e_grad)
        all_path_e.append(path_e)
        all_pts = np.vstack((start, points, end))
        path_dists.append(e_r_diff(all_pts))
        print("path_dists:", path_dists)

        # sys.exit()
        # all_pts_old = copy.deepcopy(all_pts)
        if re_meshing_type:
            # print('re_mesh_frequency:', re_mesh_frequency)
            if (step + 1) % re_mesh_frequency == 0:
                # print('Before re-meshing')
                # print('np.shape(path_e):', np.shape(path_e))

                all_pts, path_e, path_e_grad, a_list = re_meshing(all_pts,
                                                                  path_e,
                                                                  path_e_grad,
                                                                  a_list,
                                                                  atomic_symbols,
                                                                  calc_type,
                                                                  ml_model_path,
                                                                  re_meshing_type=re_meshing_type,
                                                                  change_factor=change_factor)
                # print('After re-meshing')
                # print(np.shape(path_e))
            # print('a_list inside lag_opt loop:', a_list)
        '''
        # convert list of lists to list of tuples
        all_pts_old_tuples = [tuple(x) for x in all_pts_old]
        all_pts_tuples = [tuple(x) for x in all_pts]

        # create sets from the list of tuples
        set_all_pts_old = set(all_pts_old_tuples)
        set_all_pts = set(all_pts_tuples)

        # find the symmetric difference between the two sets
        different_pts = set_all_pts_old.symmetric_difference(set_all_pts)

        print('different_pts:', different_pts)
        '''
        if path_grad_type == 1:
            _, perpendicular_grads = path_grad_components(all_pts,
                                                          path_e_grad)
            perpendicular_grads.insert(0, start_e_grad)
            perpendicular_grads.append(end_e_grad)
            # print('all_pts:', *all_pts, sep='\n')
            # '''
            points = all_pts[1:-1] - step_factor * grad_lagrangian(all_pts,
                                                                   a_list,
                                                                   path_e,
                                                                   perpendicular_grads,
                                                                   action_type=action_type)
            # '''
        elif path_grad_type == 0:
            points = all_pts[1:-1] - step_factor * grad_lagrangian(all_pts,
                                                                   a_list,
                                                                   path_e,
                                                                   path_e_grad,
                                                                   action_type=action_type)
        if convergence_type == 1:
            path_e_max = np.max(path_e)
            if path_e_old_max - path_e_max < eps:
                print('Lagrangian path optimization converged in ', step, 'steps.')
                break
            path_e_old_max = path_e_max
        elif convergence_type == 0:
            if np.linalg.norm(points_old - points) < eps:
                print('Lagrangian path optimization converged in ', step, 'steps.')
                break
            points_old = points

        all_path_coords.append(points)
    all_path_coords = np.reshape(all_path_coords,
                                 (len(all_path_coords),
                                  np.prod(np.shape(all_path_coords)[1:])))
    print('np.shape(all_path_coords):', np.shape(all_path_coords))

    if calc_type == 0:
        '''
        counter = 0
        os.mkdir(str(output_dir) + '/lag_opt_paths')
        for path_coords in all_path_coords:
            print('np.shape(path_coords):', np.shape(path_coords))
            save_all_geoms(atomic_symbols,
                           path_coords,
                           log_file=str(output_dir) + '/lag_opt_paths/lag_opt_path' + str(counter) + '.xyz')
            counter = counter + 1
        '''
    elif calc_type == 1:
        np.savetxt(str(output_dir) + '/lag_opt_path.txt',
                   all_path_coords,
                   fmt='%.5f')

    # Save the final points to a file.
    np.savetxt(output_path_file,
               points,
               fmt='%.5f')
    np.savetxt(all_path_e_file,
               np.atleast_2d(all_path_e),
               fmt='%.8f')
    np.savetxt(str(output_dir) + '/path_dists.txt', path_dists, fmt='%.5f')
    # Return the final points.
    return np.vstack((start, points, end))
