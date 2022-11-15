

import numpy
import h5py

def get_wofry_data(root, dir_out="./", training_ratio=2/3, verbose=1, gs_or_z=0):
    if gs_or_z == 0:
        filename = "%s%s_targets_gs.txt" % (dir_out, root)
    else:
        filename = "%s%s_targets_z.txt" % (dir_out, root)

    tmp = numpy.loadtxt(filename)

    targets = tmp[:,1:].copy()

    if verbose: print(targets.shape, targets)

    h5_file = "%s%s_block.h5" % (dir_out, root)

    f = h5py.File(h5_file, 'r')

    data = f['allsamples/intensity/stack_data'][:]

    f.close()

    if verbose: print(data.shape, data)

    size_data = data.shape[0]
    size_target = targets.shape[0]

    if size_data != size_target:
        raise Exception("Data and targets must have the same size.")


    istart_training = 0
    iend_training = int(training_ratio * size_data)
    istart_test = iend_training
    iend_test = size_data


    return (data[istart_training:iend_training], targets[istart_training:iend_training]), \
           (data[istart_test:iend_test], targets[istart_test:iend_test])


if __name__ == "__main__":


    dir_out = "/users/srio/Oasys/ML_TRAIN2/"  # where the results are going to be written
    root = "tmp_ml"


    (training_data, training_target), (test_data, test_target) = get_wofry_data(root, dir_out=dir_out)

    print("Training [data/target]: ", training_data.shape, training_target.shape)
    print("Test: [data/target]", test_data.shape, test_target.shape)


