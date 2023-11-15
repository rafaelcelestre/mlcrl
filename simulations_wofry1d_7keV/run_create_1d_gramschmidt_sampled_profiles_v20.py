import numpy
from srxraylib.plot.gol import plot, plot_table
from mlcrl.create_1d_zernike_sampled_profiles import create_1d_zernike_sampled_profiles
from mlcrl.create_1d_zernike_basis import create_1d_zernike_basis


if __name__ == "__main__":

    # read or calculate zernike basis
    read_from_file = False

    if read_from_file:
        filename = "z1d_basis.dat"
        tmp = numpy.loadtxt(filename)
        basis_x = tmp[:,0].copy()
        basis_pre = tmp[:,1:].copy()
        print(">> READ shapes basis_x, basis: ", basis_x.shape, basis_pre.shape)
    else:
        basis_x, basis_pre = create_1d_zernike_basis(
            size=512,
            noll=[6,   8,  10,  11,  14,  22, 37],  # removed 12!!!!!!!!!!!!!!!!!!
            filename=None,
            width=1500e-6,
            do_plot=False)

    plot_table(basis_x, basis_pre.T, xtitle="position [um]", ytitle="basis",
               title="non-orthonormal basis",
               legend=numpy.arange(basis_pre.shape[1]))

    # orthonormalize (Gram Schmidt)
    basis, R = numpy.linalg.qr(basis_pre)

    plot_table(basis_x, basis.T, xtitle="position [um]", ytitle="basis",
               title="Gram-Schmidt orthonormal basis",
               legend=numpy.arange(basis.shape[1]))

    if False: # check orthogonality
        for i in range(basis.shape[1]):
            for j in range(basis.shape[1]):
                z1 = basis[:,i]
                z2 = basis[:,j]
                tmp = numpy.nansum(z1 * z2)
                if numpy.abs(tmp) < 1e-12:
                    tmp = 0
                print(">>> norm V profile %d  %d: %g" % (i, j, tmp))


    # create sampled profiles
    size = basis_x.size
    # nsamples = 5000
    # seed = 69  # seed for generation of the random Zernike profiles

    nsamples = 25000
    seed = 696969  # seed for generation of the random Zernike profiles

    noll         =               [6,   8,  10,  11,  14,  22, 37]  # removed 12!!!!!!!!!!!!!!!!!!
    # distrubution =             ['n', 'n', 'n', 'u', 'n', 'u', 'u']
    # scale        = numpy.array([0.5, 0.5, 0.5, 2.3, .05, 1.0, 0.5]) * 1e-6
    distrubution =             ['u', 'u', 'u', 'u', 'n', 'u', 'u']
    scale        = numpy.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * 1e-6

    C, Y  = create_1d_zernike_sampled_profiles(
        nsamples,
        size=size,
        noll=noll, distrubution=distrubution, scale=scale, factor=1.0, #factor=5.0,
        seed=seed,
        do_plot=0,
        )

    print(Y.shape,C.shape)
    xx = numpy.linspace(-1500.0e-6/2,1500.0e-6/2,size)
    if False:
        plot_table(basis_x*1e6, Y.T*1e6, xtitle="Lens position [um]", ytitle="Sampled thickness [um]",
                   legend=numpy.arange(nsamples), title="sampled profiles")

    # Gram-Schmidt normalization
    Rinv = numpy.linalg.inv(R)

    # fit sampled profiles
    F = numpy.zeros_like(C)
    for i in range(nsamples):
        F[:,i] = numpy.dot(R,C[:,i])



    if False: # plot
        for i in range(nsamples):
            Corig = C[:,i]
            # Cfit = numpy.dot(R,Corig)
            Cfit = F[:,i]
            vinfl = numpy.dot(Rinv, Cfit)

            print("\n\n>>>>> sample index i: ", i)
            print("   Cfit: ", Cfit)
            print("   Corig: ", C[:,i])
            print("   R^-1 . Cfit (= Corig): ", vinfl)

            Yi = Y[:, i]
            y2 = numpy.zeros_like(basis_x)
            for j in range(basis.shape[1]):
                base = basis[:,j]
                y2 += Cfit[j] * base
            plot(basis_x, Yi,
                 basis_x, y2,
                 legend=["data", "from fit"],
                 title="sample: %d" % i, marker=["+",None])


    if True: # write files
        dir = "/nobackup/gurb1/srio/Oasys/ML_TRAIN_V20_25000/"
        root = "tmp_ml"

        for i in range(nsamples):
            #txt
            filename = "%s%s%06d.txt" % (dir, root, i)
            f = open(filename, 'w')

            f.write("# noll ")
            for j in range(len(noll)):
                f.write("%d  " % noll[j])
            f.write("\n")

            f.write("# Zernike coeff: ")
            for j in range(len(noll)):
                f.write("%g  " % C[j,i])
            f.write("\n")

            f.write("# GramSchmidt coeff: ")
            for j in range(len(noll)):
                f.write("%g  " % F[j,i])
            f.write("\n")

            for j in range(len(noll)):
                f.write("%d  %g  %g \n" % (noll[j], C[j,i], F[j,i]))
            f.write("\n")

            print("File written to disk: %s" % filename)

            # dat
            filename = "%s%s%06d.dat" % (dir, root, i)
            f = open(filename, 'w')

            for j in range(size):
                f.write("%g   %g\n" % (xx[j], Y[j,i]))
            f.close()
            print("File written to disk: %s" % filename)


        # overwrite 000 with no deformation
        # txt - no deformation
        filename = "%s%s%06d.txt" % (dir, root, 0)
        f = open(filename, 'w')

        f.write("# noll ")
        for j in range(len(noll)):
            f.write("%d  " % noll[j])
        f.write("\n")

        f.write("# Zernike coeff: ")
        for j in range(len(noll)):
            f.write("%g  " % 0.0)
        f.write("\n")

        f.write("# GramSchmidt coeff: ")
        for j in range(len(noll)):
            f.write("%g  " % 0.0)
        f.write("\n")

        for j in range(len(noll)):
            f.write("%d  %g  %g \n" % (noll[j], 0.0, 0.0))
        f.write("\n")
        print("File written to disk: %s" % filename)

        # dat - no deformation
        filename = "%s%s%06d.dat" % (dir, root, 0)
        f = open(filename, 'w')

        for j in range(size):
            f.write("%g   %g\n" % (xx[j], 0.0))
        f.close()
        print("File written to disk: %s" % filename)
