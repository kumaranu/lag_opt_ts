import numpy as np
import os
import readIn

from logger import get_logger
logger = get_logger('my_logger', 'lag_opt.log')


def compute_initial_points(start, end, number_of_points):
    """
    Computes a list of initial points along a straight line connecting two given points.

    Args:
        start (array-like): The starting point of the line.
        end (array-like): The end point of the line.
        number_of_points (int): The number of initial points to compute.

    Returns:
        array-like: A 2D numpy array of shape (number_of_points, len(start)), where each row represents an initial point.

    """
    ts = np.linspace(0.0, 1.0, number_of_points+1)[1:]  # Compute a list of values between
    # 0 and 1, with length number_of_points, excluding the endpoints.
    points = [start * (1 - t) + end * t for t in ts]  # Compute a list of points along the
    # line connecting start and end, using the ts values.
    return np.stack(points)  # Convert the list of points to a 2D numpy array and return it.


def get_initial_path(calc_type, minima1, special_point, n_images, minima2,
                     path_to_a_script_to_call_geodesic_code=None, xyz_r=None,
                     xyz_p=None, xyz_path=None, xyz_r_p=None, atomic_symbols=None):
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
        # Linear interpolation
        initial_points1 = compute_initial_points(np.asarray(minima1), np.asarray(special_point),
                                                 n_images)
        initial_points2 = compute_initial_points(np.asarray(special_point), np.asarray(minima2),
                                                 n_images)
        initial_points = np.vstack([initial_points1, initial_points2])
        return [minima1, minima2, initial_points, atomic_symbols]
    elif calc_type == 0:
        '''
        This line right below should be converted to an in-memory call instead of system call.
        '''
        # Geodesic path optimization
        # Call the geodesic code
        reference_dir = os.getcwd()
        os.chdir(path_to_a_script_to_call_geodesic_code)
        os.system('geodesic_interpolate ' + str(xyz_r_p) + ' --output output.xyz --nimages ' + str(n_images))
        os.chdir(reference_dir)

        # Read in the reactant, product and path geometries
        atomic_symbols_r, coords_r = readIn.read_in_one_xyz_file(xyz_r)
        atomic_symbols_p, coords_p = readIn.read_in_one_xyz_file(xyz_p)
        # atomic_symbols_path, coords_path = readIn.read_in_multiple_xyz_file(xyz_path, n_images=n_images)
        atomic_symbols_path, coords_path = readIn.read_in_multiple_xyz_file(
            path_to_a_script_to_call_geodesic_code + '/output.xyz', n_images=n_images)
        n_atoms = len(coords_r)
        coords_r = coords_r.flatten()
        coords_p = coords_p.flatten()
        coords_path = np.reshape(coords_path, (n_images, 3*n_atoms))
        print('coords_path:', coords_path)
        return [coords_r, coords_p, coords_path, atomic_symbols]


def action1(calc_type, atomic_symbols, point_l, point_r, energy_l, energy_r, a):
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
    from energy_and_grad import get_e
    dr = point_r - point_l
    #df = get_e(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)-get_e(
    #    point_l, atomic_symbols=atomic_symbols, calc_type=calc_type)
    df = energy_r - energy_l
    dist_component = np.exp(a * (dr**2).sum()) - 1.0
    dist_component1 = (dr**2).sum()
    potential_component = df ** 2
    action_var = dist_component + potential_component
    return action_var, dist_component1, potential_component


def grad_action_l1(d_r, d_e, e_grad_l, a):
    """
    Computes the gradient of the action with respect to the left point of the interval.

    Args:
        calc_type (int): Calculation type, determines the form of the potential energy function.
        atomic_symbols (list): List of atomic symbols.
        point_l (ndarray): Left point of the interval, shape (3 * n_atoms,).
        point_r (ndarray): Right point of the interval, shape (3 * n_atoms,).
        a (float): Exponential factor for the distance component of the action.

    Returns:
        action_grad_var (ndarray): Gradient of the action with respect to the left point, shape (3 * n_atoms,).
    """
    dist_component_grad = -1 * a * 2 * d_r
    graph_component_grad = -2 * d_e * e_grad_l

    # Compute the gradient of the action with respect to the left point
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def grad_action_r1(d_r, d_e, e_grad_r, a):
    """
    Calculates the gradient of the action variable with respect to the right point in the interval.
    :param calc_type: An integer representing the type of calculation to be performed.
    :param atomic_symbols: A list of atomic symbols.
    :param point_l: A numpy array representing the left point in the interval.
    :param point_r: A numpy array representing the right point in the interval.
    :param a: A float representing the coefficient for the distance component of the action variable.
    :return: A numpy array representing the gradient of the action variable with respect to the right point.
    """
    dist_component_grad = a * 2 * d_r
    graph_component_grad = 2 * d_e * e_grad_r
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def action(calc_type, atomic_symbols, point_l, point_r, a):
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
    from energy_and_grad import get_e
    dr = point_r - point_l
    df = get_e(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)-get_e(
        point_l, atomic_symbols=atomic_symbols, calc_type=calc_type)
    dist_component = np.exp(a * (dr**2).sum()) - 1.0
    potential_component = df ** 2
    action_var = dist_component + potential_component
    return action_var


def grad_action_l(calc_type, atomic_symbols, point_l, point_r, a):
    """
    Computes the gradient of the action with respect to the left point of the interval.

    Args:
        calc_type (int): Calculation type, determines the form of the potential energy function.
        atomic_symbols (list): List of atomic symbols.
        point_l (ndarray): Left point of the interval, shape (3 * n_atoms,).
        point_r (ndarray): Right point of the interval, shape (3 * n_atoms,).
        a (float): Exponential factor for the distance component of the action.

    Returns:
        action_grad_var (ndarray): Gradient of the action with respect to the left point, shape (3 * n_atoms,).
    """
    from energy_and_grad import get_e, get_e_grad

    # Compute the difference between the right and left points
    dr = point_r - point_l

    # Compute the difference in energy between the right and left points
    df = get_e(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)-get_e(
        point_l, atomic_symbols=atomic_symbols, calc_type=calc_type)

    # Compute the gradient of the distance component of the action
    dist_component_grad = -1 * np.exp(a * (dr**2).sum()) * a * 2 * dr

    # Compute the gradient of the graph component of the action
    graph_component_grad = -2 * df * get_e_grad(point_l, atomic_symbols=atomic_symbols,
                                                calc_type=calc_type)

    # Compute the gradient of the action with respect to the left point
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def grad_action_r(calc_type, atomic_symbols, point_l, point_r, a):
    """
    Calculates the gradient of the action variable with respect to the right point in the interval.
    :param calc_type: An integer representing the type of calculation to be performed.
    :param atomic_symbols: A list of atomic symbols.
    :param point_l: A numpy array representing the left point in the interval.
    :param point_r: A numpy array representing the right point in the interval.
    :param a: A float representing the coefficient for the distance component of the action variable.
    :return: A numpy array representing the gradient of the action variable with respect to the right point.
    """
    dr = (point_r - point_l)
    from energy_and_grad import get_e, get_e_grad
    df = (get_e(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)
          - get_e(point_l, atomic_symbols=atomic_symbols, calc_type=calc_type))
    dist_component_grad = np.exp(a * (dr**2).sum()) * a * 2 * dr
    graph_component_grad = 2 * df * get_e_grad(point_r, atomic_symbols=atomic_symbols, calc_type=calc_type)
    action_grad_var = dist_component_grad + graph_component_grad
    return action_grad_var


def lagrangian(f, all_pts, a):
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
        lag += action(f, all_pts[i], all_pts[i+1], a)

    # Return the final Lagrangian value.
    return lag


def grad_lagrangian(calc_type, atomic_symbols, all_pts, a):
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

    # d_r_normalized, d_r_mu, d_r_sigma = standardization(all_pts)
    all_pts_stdized = standardization(all_pts)

    # print('d_r_normalized, d_r_mu, d_r_sigma:', d_r_normalized, d_r_mu, d_r_sigma)
    print('all_pts_stdized:', all_pts_stdized)
    from energy_and_grad import get_path_e
    path_e, path_e_grad = get_path_e(all_pts, atomic_symbols=atomic_symbols, calc_type=calc_type)

    with open('all_path_e.txt', 'a') as f:
        f.write(' '.join(np.asarray(path_e).astype(str)))
        f.write('\n')
    path_e_stdized = standardization(path_e)
    path_e_grad_stdized = standardization(path_e_grad)
    #d_e_normalized, d_e_mu, d_e_sigma = standardization(path_e)
    #print('d_e_normalized, d_e_mu, d_e_sigma:', d_e_normalized, d_e_mu, d_e_sigma)

    # Create an array to hold the gradient of the Lagrangian.
    grad_lag = np.zeros((all_pts.shape[0]-2, all_pts.shape[1]))

    # Compute the gradient of the action for each interval defined by the points.
    for i in range(len(all_pts)-2):

        # Compute the gradient of the action for the left and right intervals and add them up.
        grad_lag[i] = np.array(grad_action_r1(all_pts_stdized[i+1]-all_pts_stdized[i],
                                              path_e_stdized[i+1]-path_e_stdized[i],
                                              np.asarray(path_e_grad_stdized[i+1]).flatten(), a)) + \
                      np.array(grad_action_l1(all_pts_stdized[i+1]-all_pts_stdized[i],
                                              path_e_stdized[i+1]-path_e_stdized[i],
                                              np.asarray(path_e_grad[i+1]).flatten(), a))

    # Return the computed gradient of the Lagrangian.
    return grad_lag


def standardization(x):
    #neighbouring_dists = [np.sqrt((i ** 2).sum()) for i in np.diff(x, axis=0)]
    #mu, std = np.average(neighbouring_dists), np.std(neighbouring_dists)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    # Standardize the vector
    x_stdized = (x - x_mean) / x_std

    return x_stdized


def find_critical_path(calc_type, atomic_symbols=None, initial_points=None,
                       start=None, end=None, num_steps=None, step_factor=None, a=None):
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
    if calc_type == 0:
        result, points = [], initial_points
        for step in range(num_steps):
            all_pts = np.vstack((start, points, end))
            points = points - step_factor * grad_lagrangian(calc_type, atomic_symbols, all_pts, a)
        np.savetxt('outputPath.txt', points, fmt='%.5f')
    elif calc_type == 1:
        result, points = [], initial_points
        from energy_and_grad import get_path_e
        for step in range(num_steps):
            all_pts = np.vstack((start, points, end))
            points = points - step_factor * grad_lagrangian(calc_type, atomic_symbols, all_pts, a)
        # Save the final points to a file.
        np.savetxt('outputPath.txt', points, fmt='%.5f')
    # Return the final points.
    return points
