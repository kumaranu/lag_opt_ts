# Import necessary libraries
from ase.io import read
from ase.optimize import BFGS
from ase.optimize.precon import PreconLBFGS
from ase.constraints import UnitCellFilter
from ase.io import write
import numpy as np


def calculate_irc(initial_geometry, step, max_steps, threshold_angle, calculator):
    # Start from the transition state geometry as the initial point
    current_geometry = initial_geometry.copy()

    # Initialize the IRC path
    irc_path = [current_geometry]

    for irc_point in range(max_steps):
        # Determine the pivot point at Step/2 distance in the opposite direction of the gradient
        pivot_point = determine_pivot_point(current_geometry, step, calculator)

        for optimization_step in range(max_optimization_steps):
            # Optimize the geometry at the current IRC point using the ASE calculator
            optimized_geometry = optimize_geometry(current_geometry, calculator)

            # Calculate the angle between (pivot-start) and (pivot-final) vectors
            angle = calculate_angle(pivot_point, current_geometry, optimized_geometry)

            if angle < threshold_angle:
                # Reached the vicinity of an endpoint, switch to energy minimization
                optimized_geometry = minimize_energy(current_geometry, calculator)
                break

            elif angle < 120:
                # Current IRC step is canceled, start a new one with half the initial Step
                step /= 2
                break

            # The optimized geometry becomes the starting point for the next IRC step
            current_geometry = optimized_geometry.copy()

        # Add the optimized geometry to the IRC path
        irc_path.append(current_geometry.copy())

    return irc_path


# Helper functions
def determine_pivot_point(current_geometry, step, calculator):
    # Calculate the gradient for the current geometry using the ASE calculator
    gradient = current_geometry.get_forces()

    # Determine the pivot point at Step/2 distance in the opposite direction of the gradient
    pivot_point = current_geometry.get_positions() - step / 2.0 * gradient
    current_geometry.set_positions(pivot_point)

    # Update the potential energy and forces at the pivot point using the ASE calculator
    current_geometry.calc.calculate(current_geometry)

    return pivot_point


def optimize_geometry(geometry, calculator):
    # Perform geometry optimization using the ASE calculator
    # Here, we use the LBFGS algorithm with pre-conditioning
    optimizer = PreconLBFGS(geometry, logfile=None, use_armijo=False)
    dynamics = optimizer.minimize(fmax=0.1, steps=100)

    return geometry


def minimize_energy(geometry, calculator):
    # Perform energy minimization using the ASE calculator
    # Here, we use the BFGS algorithm
    optimizer = BFGS(geometry, logfile=None)
    dynamics = optimizer.minimize()

    return geometry


if __name__ == '__main__':
    # Example usage
    initial_geometry = read('test0_ts.xyz')  # Provide the initial transition state geometry
    step = 0.1  # Specify the initial Step parameter
    max_steps = 100  # Specify the maximum number of IRC steps
    threshold_angle = 90.0  # Specify the threshold angle for endpoint detection

    # Assume you have an ASE calculator assigned to the variable "calculator"
    irc_path = calculate_irc(initial_geometry, step, max_steps, threshold_angle, calculator)

    # Write the IRC path to a trajectory file (optional)
    write('irc_path.traj', irc_path)

    # Print the optimized geometries at each IRC point
    for idx, geometry in enumerate(irc_path):
        print(f"IRC Point {idx + 1}:")
        print(geometry)
