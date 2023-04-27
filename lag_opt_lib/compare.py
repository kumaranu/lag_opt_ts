import numpy as np
from geodesic_interpolate.coord_utils import get_bond_list
from geodesic_interpolate.coord_utils import morse_scaler
from geodesic_interpolate.coord_utils import compute_wij


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
