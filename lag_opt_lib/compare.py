import numpy as np
from geodesic_interpolate.coord_utils import get_bond_list
from geodesic_interpolate.coord_utils import morse_scaler
from geodesic_interpolate.coord_utils import compute_wij


def e_r_diff(geoms_list):
    """
    Computes the distances between consecutive `wij` values for a list of geometries.

    Args:
        geoms_list (list): A list of 3D coordinates of atoms in a molecule. Each element in the list
                           is a 2D numpy array of shape `(N, 3)`, where `N` is the number of atoms
                           in the molecule.

    Returns:
        dists (list): A list of distances between consecutive `wij` values. The length of the list
                      is `len(geoms_list) - 1`.
    """
    # Get bond list and equilibrium bond length for the first geometry in geoms_list
    rijlist, re = get_bond_list(geoms_list[0])

    # Compute scaling factor for the Morse potential
    scaler = morse_scaler(alpha=0.7, re=re)

    # Compute wij values for the first geometry and store in wij_old
    wij_old, _ = compute_wij(geoms_list[0], rijlist, scaler)

    dists = []
    distance = 0.0

    # Iterate through remaining geometries in geoms_list
    for i, geom in enumerate(geoms_list[1:]):

        # Compute wij values for current geometry
        wij, _ = compute_wij(geom, rijlist, scaler)
        print('wij:\n', wij)

        # Add distance between wij and wij_old to distance variable and dists list
        distance += np.linalg.norm(wij - wij_old)
        dists.append(distance)

        # Set wij_old to current wij value for next iteration
        wij_old = wij

    return dists


'''
def e_r_diff1(atoms):
    wij_list = []
    for atom in atoms:
        geom = atom.get_positions()
        rijlist, re = get_bond_list(geom)
        scaler = morse_scaler(alpha=0.7, re=re)
        wij, _ = compute_wij(geom, rijlist, scaler)
        wij_list.append(wij)
    diff_internal_coord = np.linalg.norm(wij_list[0] - wij_list[1])
    return diff_internal_coord


def e_r_diff(geoms_list):
    wij_list = []

    rijlist, re = get_bond_list(geoms_list[0])
    scaler = morse_scaler(alpha=0.7, re=re)
    wij_old, _ = compute_wij(geoms_list[0], rijlist, scaler)
    print('wij_old:\n', wij_old)
    dists = []
    distance = 0.0
    for i, geom in enumerate(geoms_list):
        wij, _ = compute_wij(geom, rijlist, scaler)
        print('wij:\n', wij)
        wij_list.append(wij)
        if i < len(geoms_list):
            distance += np.linalg.norm(wij - wij_old)
            dists.append(distance)
        wij_old = wij
    return dists
'''

'''
def e_r_diff(geoms_list):
    wij_list = []
    diffs_int_coord_list = []
    for i, geom in enumerate(geoms_list):
        rijlist, re = get_bond_list(geom)
        scaler = morse_scaler(alpha=0.7, re=re)
        wij, _ = compute_wij(geom, rijlist, scaler)
        wij_list.append(wij)
        diffs_int_coord_list.append(np.linalg.norm(wij_list[i] - wij_list[i+1]))
    return diffs_int_coord_list
'''
