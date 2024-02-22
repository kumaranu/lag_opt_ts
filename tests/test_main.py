from compare_geoms.main import compare_one_ref_mol

sample_h5_file = 'output_9953.h5'
sample_ref_molecule = '264_noise00.xyz'


def test_compare_one_ref_mol():
    # Test the compare_one_ref_mol function
    compare_one_ref_mol(sample_h5_file, sample_ref_molecule)
