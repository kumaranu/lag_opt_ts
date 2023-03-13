import numpy as np
import os


def print_all_geoms(opt_path):
    """
    Saves all geometries in the optimization path to individual text files.

    Arguments:
    opt_path -- list or array containing the optimization path. Each element in the list/array is itself an array
                containing the Cartesian coordinates of the atoms in the molecule.

    Returns:
    None

    """
    # create a directory to store all the geometry files if it doesn't already exist
    if not os.path.exists('allGeoms'):
        os.mkdir('allGeoms')
    os.chdir('allGeoms')

    # loop through all geometries in the optimization path and save them to individual text files
    for i in range(len(opt_path)):
        # create a directory to store geometry files for this geometry if it doesn't already exist
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        os.chdir(str(i))
        # save the geometry to a text file
        np.savetxt(str(i) + '.txt', opt_path[i], fmt='%.5f')
        os.chdir('../')

    # change back to the original directory
    os.chdir('../')


def save_all_geoms(atomic_symbols, opt_path, log_file='allGeoms.xyz'):
    """
    Save all geometries in a single .xyz file

    Args:
    - atomic_symbols (list): list of atomic symbols in the molecule
    - opt_path (list): list of optimized geometry arrays
    - log_file (str): file name to write the geometries in .xyz format, default is 'allGeoms.xyz'

    Returns:
    - None

    Comments:
    - The function takes a list of atomic symbols and a list of optimized geometry arrays
    - It saves all geometries in a single .xyz file with the given file name
    - The format of the .xyz file is as follows:
        - The first line is the number of atoms in the molecule
        - The second line is blank
        - Each subsequent line contains an atomic symbol followed by the x, y, and z coordinates of the atom
    """
    file1 = open(log_file, 'w')
    n_geoms = len(opt_path)
    n_atoms = len(atomic_symbols)
    # print('n_atoms:', n_atoms)
    # print('n_geoms:', n_geoms)
    opt_path = np.reshape(opt_path, (n_geoms, n_atoms, 3))
    for i in range(n_geoms):
        # print('i in write xyz', i)
        file1.write(str(n_atoms) + '\n\n')
        for j in range(n_atoms):
            file1.write(atomic_symbols[j] + ' ' + str(opt_path[i][j][0]) + ' ' + str(opt_path[i][j][1]) +
                        ' ' + str(opt_path[i][j][2]) + '\n')
    file1.close()


def write_xyz(coords, atomic_symbols=None, path_to_write=os.getcwd(), file_name='test'):
    """
    Write the atomic coordinates in xyz format to a file named 'test.xyz'.

    Args:
    coords (array-like): An array of atomic coordinates with shape (n_atoms, 3).
    symbols (list): A list of atomic symbols corresponding to each atom in `coords`.

    Returns:
    None
    """
    n_atoms = len(atomic_symbols)
    coords = np.reshape(coords, (n_atoms, 3))
    file_str = str(n_atoms) + '\n'
    for i in range(n_atoms):
        file_str += '\n' + atomic_symbols[i] + ' ' + str(coords[i][0]) + ' ' + str(coords[i][1]) + ' ' + str(coords[i][2])
    file_str += '\n'

    file1 = open(path_to_write + '/' + file_name, 'w')
    file1.write(file_str)
    file1.close()


def xyz_to_pdb():
    """
    Convert an XYZ file to a PDB file using Open Babel.
    """
    from openbabel import openbabel as ob

    # Specify the input XYZ file and set the format to "xyz"
    xyz_file = "test.xyz"
    ob_conversion = ob.OBConversion()
    ob_conversion.SetInFormat("xyz")

    # Read the molecule from the XYZ file
    mol = ob.OBMol()
    ob_conversion.ReadFile(mol, xyz_file)

    # Specify the output PDB file and set the format to "pdb"
    pdb_file = "test.pdb"
    ob_conversion.SetOutFormat("pdb")

    # Write the molecule to the PDB file
    ob_conversion.WriteFile(mol, pdb_file)
