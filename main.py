# Import necessary libraries
import readIn
from steepest_descent import steepest_descent
import lagrangian
from plot_functions import plot_contour, plot_multiple_path_es
from logger import get_logger
import os

logger = get_logger('my_logger', log_file='lag_opt.log', log_mode='w')
logger.info('Inside lag-opt package.')

# Check if the script is being run as the main program
if __name__ == '__main__':

    logger.info('Reading in user arguments.')
    # Read input arguments from the command line
    args = readIn.read_in()

    # If the job type is optimization, perform the steepest descent algorithm
    if args.job_type == 'opt':

        logger.info('This is an optimization job. Calling steepest descent function.')

        # Perform steepest descent optimization
        x, fx, num_iter, opt_path = steepest_descent(args.init_geom, args.step_size, args.eps, args.max_iter,
                                                     calc_type=args.calc_type, labels=args.atomic_symbols)

        logger.info('Done with geometry optimization. Plotting the optimization path.')

        # Plot the optimization path
        plot_contour(args.x_min, args.x_max, args.y_min, args.y_max, args.delta, opt_path,
                     calc_type=args.calc_type, atomic_symbols=args.atomic_symbols)

    # If the job type is transition state optimization, perform Lagrangian path optimization
    elif args.job_type == 'ts_lag':

        logger.info('This is a transition state calculation.')
        logger.info('Generating an initial path for the Lagrangian path optimization.')

        # Obtain the initial path using geodesic code
        coords_r, coords_p, coords_path, atomic_symbols = lagrangian.get_initial_path(
            args.calc_type, args.minima1, args.special_point, args.n_images, args.minima2,
            path_to_a_script_to_call_geodesic_code=args.path_to_a_script_to_call_geodesic_code,
            xyz_path=args.xyz_path, xyz_r_p=args.xyz_r_p, atomic_symbols=args.atomic_symbols)

        logger.info('Done with generating initial path generation. Calling the path optimization.')

        # If the calculation requires molecular systems, use reactant and product geometries as two minima
        if args.calc_type == 0:
            args.minima1, args.minima2 = coords_r, coords_p
        if os.path.exists(args.all_path_e_file):
            os.remove(args.all_path_e_file)
        result = lagrangian.find_critical_path(args.calc_type, atomic_symbols=atomic_symbols,
                                               initial_points=coords_path, start=args.minima1,
                                               end=args.minima2, num_steps=args.max_iter,
                                               step_factor=args.step_size, action_type=args.action_type,
                                               a=args.a, eps=args.eps,
                                               all_path_e_file=args.all_path_e_file)
        if args.calc_type == 0:
            plot_nth = args.nth
            plot_multiple_path_es(plot_nth, file_name=args.all_path_e_file)
        elif args.calc_type == 1:
            plot_contour(args.x_min, args.x_max, args.y_min, args.y_max, args.delta, result,
                         calc_type=args.calc_type, atomic_symbols=atomic_symbols)
