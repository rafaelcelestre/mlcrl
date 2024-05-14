

import numpy
from mlcrl.get_wofry_data import get_wofry_data
from mlcrl.create_1d_zernike_basis import create_1d_zernike_basis

from srxraylib.plot.gol import plot, plot_table

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

def dump_file(x, y, z, filename):
    f = open(filename, "w")
    for i in range(numpy.array(x).size):
        f.write("%g    %g    %g   \n" % (x[i], y[i], z[i]))
    f.close()
    print("File %s written to disk." % filename)

if __name__ == "__main__":

    do_plot = 1

    root = "tmp_ml"

    # dir_out = "/scisoft/users/srio/ML_TRAIN2/1000/"
    # model_root = "training_v12"
    # dir_files = "/users/srio/Oasys/ML_TRAIN"
    # n_files = 1000
    # basis_used = 'gs'
    # nbin = 1
    # pstart = 100


    # dir_out = "/scisoft/users/srio/ML_TRAIN2/"
    # model_root = "training_v13"
    # dir_files = "/users/srio/Oasys/ML_TRAIN5000"
    # n_files = 5000
    # basis_used = 'gs'
    # nbin = 1
    # pstart = 100


    # dir_out = "/scisoft/users/srio/ML_TRAIN2/"
    # model_root = "training_v14"
    # dir_files = "/users/srio/Oasys/ML_TRAIN5000"
    # n_files = 5000
    # basis_used = 'z'
    # nbin = 1
    # pstart = 100


    # dir_out = "/scisoft/users/srio/ML_TRAIN2/"
    # model_root = "training_v15"
    # dir_files = "/users/srio/Oasys/ML_TRAIN5000"
    # n_files = 5000
    # basis_used = 'gs'
    # nbin = 2
    # pstart = 100

    # dir_out = "/scisoft/users/srio/ML_TRAIN2/"
    # model_root = "training_v16"
    # dir_files = "/users/srio/Oasys/ML_TRAIN5000"
    # n_files = 5000
    # basis_used = 'gs'
    # nbin = 4
    # pstart = 100

    # dir_out = "/scisoft/users/srio/ML_TRAIN2/"
    # model_root = "training_v19"
    # dir_files = "/users/srio/Oasys/ML_TRAIN5000"
    # n_files = 5000
    # basis_used = 'gs'
    # nbin = -2
    # pstart = 100

    # dir_out = "/scisoft/users/srio/ML_TRAIN2/MULTIMODE/"
    # model_root = "training_v20"
    # dir_files = "/users/srio/Oasys/ML_TRAIN5000"
    # n_files = 5000
    # basis_used = 'gs'
    # nbin = 1
    # pstart = 100

    dir_out = "/scisoft/users/srio/MLCRL/V10/ML_TRAIN2/MULTIMODE/"
    model_root = "training_v20"
    dir_files = "/scisoft/users/srio/MLCRL/V10/ML_TRAIN5000"
    n_files = 5000
    basis_used = 'gs'
    nbin = 1
    pstart = 100

    if basis_used == 'gs':
        (training_data, training_target), (test_data, test_target) = get_wofry_data(root, dir_out=dir_out, verbose=0, gs_or_z=0, nbin=nbin)
    elif basis_used == 'z':
        (training_data, training_target), (test_data, test_target) = get_wofry_data(root, dir_out=dir_out, verbose=0, gs_or_z=1, nbin=nbin)
    else:
        raise Exception("error...")
    print("Training: ", training_data.shape, training_target.shape)
    print("Test: ", test_data.shape, test_target.shape)

    min_training_data = training_data.min()
    max_training_data = training_data.max()

    print("Min, Max of Training: ", min_training_data, max_training_data)

    # data type: images— 4D tensors of shape (samples, height, width, channels) or (samples, channels, height, width)
    #            could also be Timeseries data or sequence data— 3D tensors of shape (samples, timesteps, features)
    #            right now our data is (samples, features (256), timesteps (65))
    training_data = training_data.reshape((training_data.shape[0], training_data.shape[1], training_data.shape[2], 1))


    training_data = training_data.astype('float32')
    training_data = (training_data - min_training_data) / (max_training_data - min_training_data)

    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
    test_data = test_data.astype('float32')
    test_data = (test_data - min_training_data) / (max_training_data - min_training_data)

    #
    # load model
    #
    if True:
        from keras.models import load_model
        import json

        model = load_model('%s/%s.h5' % (dir_out, model_root))

        # learning with v13 and loading training v23: Does not work
        #model = load_model('%s/%s.h5' % (dir_out, "../training_v13"))


        f = open("%s/%s.json" % (dir_out, model_root), "r")
        f_txt = f.read()
        history_dict = json.loads(f_txt)

        print(history_dict.keys())

        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        if False: plot(epochs, loss_values,
             epochs, val_loss_values,
             legend=['loss','val_loss'], xtitle='Epochs', ytitle='Loss', show=0)

        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        # if do_plot: plot(epochs, acc_values,
        #      epochs, val_acc_values,
        #      legend=['accuracy','val_accuracy'], xtitle='Epochs', ytitle='accuracy')
        if do_plot: plot(epochs[::10], val_acc_values[::10],
             epochs[::10], acc_values[::10],
             legend=['accuracy on validation set', 'accuracy on training set'],
             color=['g', 'b'], xtitle='Epochs', ytitle='accuracy', ylog=0)

        if do_plot: plot(epochs[::10], val_acc_values[::10],
             epochs[::10], acc_values[::10],
             legend=['accuracy on validation set', 'accuracy on training set'],
             color=['g', 'b'], xtitle='Epochs', ytitle='accuracy', xrange=[700,15300], yrange=[0.9,1], ylog=0)
        #
        # test evaluation
        #

        test_loss, test_acc = model.evaluate(test_data, test_target)
        #
        print("test_loss: ", test_loss)
        print("test_acc: ", test_acc)


        #
        # predictions
        #

        predictions = model.predict(test_data)
        print(test_data.shape, predictions.shape)

        # numpy.savetxt("predictions.dat", predictions, delimiter=' ')
        # print("File predictions.dat written to disk.")

    else:
        pass
        # predictions = numpy.loadtxt("predictions.dat")
        # print("test_data, test_target, predictions: ", test_data.shape, test_target.shape, predictions.shape)



    #
    # compute predicted profiles
    #
    size = 512
    basis_x, basis_pre = create_1d_zernike_basis(
        size=size,
        noll=[6,   8,  10,  11,  14,  22, 37],  # removed 12!!!!!!!!!!!!!!!!!!
        filename=None,
        width=1500e-6,
        do_plot=False)

    if False: plot_table(basis_x, basis_pre.T, xtitle="position [um]", ytitle="basis",
               title="non-orthonormal basis",
               legend=numpy.arange(basis_pre.shape[1]))

    # orthonormalize (Gram Schmidt)
    basis, R = numpy.linalg.qr(basis_pre)
    print("basis, basis_x: ", basis.shape, basis_x.shape)

    if False: plot_table(basis_x, basis.T, xtitle="position [um]", ytitle="basis",
               title="Gram-Schmidt orthonormal basis",
               legend=numpy.arange(basis.shape[1]))


    for i in range(pstart,predictions.shape[0]):
        print("\n>>>> testing sample: ", i)
        profile_orig = numpy.zeros(size)
        profile_fit = numpy.zeros(size)
        i_file = int(n_files * 2/3 + i)
        ff = "%s/%s%06d.dat" % (dir_files, root, i_file)
        a = numpy.loadtxt(ff)
        print(">>>>", a.shape)

        if basis_used == 'gs':
            for j in range(7):
                print(j,test_target[i,j], predictions[i,j])
                profile_orig += test_target[i,j] * basis[:,j]
                profile_fit += predictions[i, j] * basis[:, j]
        elif basis_used == 'z':
            for j in range(7):
                print(j,test_target[i,j], predictions[i,j])
                profile_orig += test_target[i,j] * basis_pre[:,j]
                profile_fit += predictions[i, j] * basis_pre[:, j]
        else:
            raise Exception("error...")

        plot(basis_x*1e3, profile_orig,
             basis_x*1e3, profile_fit,
             # a[::10,0], a[::10,1]*1e6,
             legend=["original", "prediction (%s)" % basis_used, ff],
             linestyle=[None,None,''],
             marker=[None,None,'.'],
             title="testing sample %d (sample # %d)" % (i, i_file),
             xtitle="abscissas [mm]", ytitle="Profile height [$\mu$m]", )

        dump_file(basis_x*1e3, profile_orig, profile_fit, "tmp_v20_%i.dat" % i)

        # plot(a[:,0]*1e6, a[:,1]*1e6)


