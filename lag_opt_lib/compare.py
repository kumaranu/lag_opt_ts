from ase.io import read
import geodesic_interpolate
import numpy as np


def e_r_diff1(atoms):
    wij_list = []
    for atom in atoms:
        geom = atom.get_positions()
        rijlist, re = geodesic_interpolate.coord_utils.get_bond_list(geom)
        scaler = geodesic_interpolate.coord_utils.morse_scaler(alpha=0.7, re=re)
        wij, _ = geodesic_interpolate.coord_utils.compute_wij(geom, rijlist, scaler)
        wij_list.append(wij)
    diff_internal_coord = np.linalg.norm(wij_list[0] - wij_list[1])
    return diff_internal_coord


def e_r_diff(geoms_list):
    wij_list = []
    for geom in geoms_list:
        rijlist, re = geodesic_interpolate.coord_utils.get_bond_list(geom)
        scaler = geodesic_interpolate.coord_utils.morse_scaler(alpha=0.7, re=re)
        wij, _ = geodesic_interpolate.coord_utils.compute_wij(geom, rijlist, scaler)
        wij_list.append(wij)
    diff_internal_coord = np.linalg.norm(wij_list[0] - wij_list[1])
    return diff_internal_coord




