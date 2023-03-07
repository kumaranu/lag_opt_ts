# Import necessary libraries
import numpy as np
from steepest_descent import steepest_descent
import readIn
import lagrangian
from plot_functions import plot_contour

# Check if the script is being run as the main program
if __name__ == '__main__':

    # Read input arguments from the command line
    args = readIn.read_in()

    # If the job type is optimization, perform steepest descent algorithm
    if args.job_type == 'opt':

        # Perform steepest descent optimization
        x, fx, num_iter, opt_path = steepest_descent(args.init_geom, args.alpha, args.eps, args.max_iter,
                                                     calc_type=args.calc_type, labels=args.atomic_symbols)

        # Plot the optimization path
        plot_contour(args.x_min, args.x_max, args.y_min, args.y_max, args.delta, opt_path, calc_type=args.calc_type,
                     atomic_symbols=args.atomic_symbols)

    # If the job type is transition state optimization, perform Lagrangian path optimization
    elif args.job_type == 'ts_lag':

        # Obtain the initial path using geodesic code
        coords_r, coords_p, coords_path, atomic_symbols = lagrangian.get_initial_path(
            args.calc_type, args.minima1, args.special_point, args.n_images, args.minima2,
            path_to_a_script_to_call_geodesic_code=args.path_to_a_script_to_call_geodesic_code,
            xyz_r=args.xyz_r, xyz_p=args.xyz_p, xyz_path=args.xyz_path, xyz_r_p=args.xyz_r_p,
            atomic_symbols=args.atomic_symbols)

        # If the calculation type is potential energy surface, use minima as starting and ending points
        if args.calc_type == 0:
            args.minima1, args.minima2 = coords_r, coords_p
        result = lagrangian.find_critical_path(args.calc_type, atomic_symbols=atomic_symbols,
                                               initial_points=coords_path, start=args.minima1, end=args.minima2,
                                               num_steps=args.max_iter, step_factor=0.00001, a=100)

        # If the calculation type is potential energy surface, plot both initial path and final result
        if args.calc_type == 0:
            plot_contour(args.x_min, args.x_max, args.y_min, args.y_max, args.delta, coords_path,
                         calc_type=args.calc_type, atomic_symbols=atomic_symbols)
            plot_contour(args.x_min, args.x_max, args.y_min, args.y_max, args.delta, result, calc_type=args.calc_type,
                         atomic_symbols=atomic_symbols)
