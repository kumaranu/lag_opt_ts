from compare_geoms.data_loading import load_data_from_h5, load_reference_molecule

sample_h5_file = 'output_9953.h5'
sample_ref_molecule = '264_noise00.xyz'


def test_load_data_from_h5():
    # Test the load_data_from_h5 function
    ref_coords, ref_atomic_nums, ref_molecule = load_reference_molecule(sample_ref_molecule)

    data_list = load_data_from_h5(sample_h5_file, ref_coords, ref_atomic_nums, ref_molecule)


def test_load_reference_molecule():
    # Test the load_reference_molecule function
    ref_coords, ref_atomic_nums, ref_molecule = load_reference_molecule(sample_ref_molecule)
