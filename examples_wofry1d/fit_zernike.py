
from mlcrl.phasenet.zernike import Zernike
import matplotlib.pyplot as plt
import numpy as np
from srxraylib.plot.gol import plot, plot_table
from create_1d_sampled_profiles import create_1d_sampled_profiles


if __name__ == "__main__":



    #
    # retrieve, plot and save 1d (vertical) Zernike 1D non-orthogonal basis to be used in Wofry1D
    #

    if True:
        size = 128 * 4
        # noll = [6, 8, 10, 11, 12, 14, 22, 37]
        noll = [6, 8, 10, 11, 14, 22, 37]  # removed 12!!!!!!!!!!!!!!!!!!
        nnoll = len(noll)


        basis = np.zeros((size,len(noll)))
        basis_x = np.linspace(-1,1,size)
        for i in range(len(noll)):
            z = Zernike(noll[i], order='noll')
            w = z.polynomial_vertical(size)
            basis[:,i] = w

        plot_table(basis_x, basis.T, xtitle="position [um]", ytitle="basis",
                   title="basis (noll-indexed Vertical cut of Zernike polynomials)", legend=noll)
        print(">> shapes basis_x, basis: ", basis_x.shape, basis.shape)

        if True: # write to file
            filename = "z1d_basis.dat"
            f = open(filename,'w')
            for i in range(size):
                f.write("%g  " % (basis_x[i]))
                for j in range(nnoll):
                    f.write("%12.10f  " % (basis[i,j]))
                f.write("\n")
            f.close()
            print("File %s written to disk." % filename)


    #
    # read and plot non-normalized bases from file
    #
    if False:
        filename = "z1d_basis.dat"
        tmp = np.loadtxt(filename)
        basis_x = tmp[:,0].copy()
        basis = tmp[:,1:].copy()
        print(">> READ shapes basis_x, basis: ", basis_x.shape, basis.shape)
        plot_table(basis_x, basis.T, xtitle="position [um]", ytitle="basis",
                   title="READ basis (noll-indexed Vertical cut of Zernike polynomials)",
                   legend=np.arange(basis.shape[1]))

    # #
    # # orthonormalize basis (Ken's method)
    # #
    # if False:
    #     from orangecontrib.wofry.als.util.axo import orthonormalize_a, linear_2dgsfit1, linear_basis
    #     import numpy
    #
    #     filename = "z1d_basis.dat"
    #
    #     from srxraylib.plot.gol import plot, plot_table
    #
    #     input_array = numpy.loadtxt(filename)
    #     norm1 = ((input_array[:,1])**2).sum()
    #
    #     input_array[:,1] =  input_array[:,1] / np.sqrt(norm1) # / (norm1**2).sum()
    #     print(">>>>>>>>>>>>>>>>>>>> normalize first base ", ((input_array[:,1])**2).sum())
    #
    #     abscissas = input_array[:, 0].copy()
    #
    #
    #     a = []
    #     for i in range(1,input_array.shape[1]):
    #         a.append({'a': input_array[:, i], 'total_squared': 0})
    #
    #     plot_table(abscissas, input_array[:, 1:].T, title="influence functions",
    #                legend=numpy.arange(input_array.shape[1]-1))
    #
    #     # compute the basis
    #     mask = None  # u
    #     b, matrix = orthonormalize_a(a, mask=mask)
    #
    #     print(">>>>> matrix", matrix.shape)
    #     print(">>>>> len b:", len(b))
    #
    #     # plot basis
    #     b_array = numpy.zeros((input_array.shape[0], input_array.shape[1]-1))
    #     print(">>>>> b_array", b_array.shape)
    #     for i in range(b_array.shape[1]):
    #         # print(">>>>> i: ", i)
    #         b_array[:, i] = b[i]["a"]
    #     plot_table(abscissas, b_array.T, title="basis functions",
    #                legend=numpy.arange(input_array.shape[1]-1))
    #
    #
    #     if True: # write to file
    #         filename = "z1d_basis_normalized.dat"
    #         f = open(filename,'w')
    #         for i in range(b_array.shape[0]):
    #             f.write("%g  " % (abscissas[i]))
    #             for j in range(b_array.shape[1]):
    #                 f.write("%12.10f  " % (b_array[i,j]))
    #             f.write("\n")
    #         f.close()
    #         print("File %s written to disk." % filename)
    #
    #
    #     if False: # example fit
    #         # prepare a Gaussian (data to fit)
    #         sigma = (abscissas[-1] - abscissas[0]) / 30
    #         u = 15 * numpy.exp(- abscissas ** 2 / 2 / sigma)
    #
    #
    #         # perform the fit
    #         v = linear_2dgsfit1(u, b, mask=mask)
    #         print("coefficients for orthonormal basis: ", v)
    #
    #         vinfl = numpy.dot(matrix, v)
    #
    #         print(matrix)
    #         print("coefficients for influence functions basis: ", vinfl.shape, vinfl)
    #
    #         # evaluate the fitted data form coefficients and basis
    #         y = linear_basis(v, b)
    #
    #         # evaluate the fitted data form coefficients and basis
    #         yinfl = linear_basis(vinfl, a)
    #
    #         plot(abscissas, u, abscissas, y, legend=["Data", "Fit (orthonormal)"])
    #         # plot(abscissas,u,abscissas,y,abscissas,yinfl,legend=["Data","Fit (orthonormal)","Fit (sorted influence)"])


    #
    # orthonormalize basis using numpy's method
    #
    if True:
        import numpy

        filename = "z1d_basis.dat"

        from srxraylib.plot.gol import plot, plot_table

        input_array = numpy.loadtxt(filename)
        norm1 = ((input_array[:,1])**2).sum()


        abscissas = input_array[:, 0].copy()
        basis = input_array[:, 1:].copy()

        print("abscisas, basis, input: ", abscissas.shape, basis.shape, input_array.shape)


        plot_table(abscissas, input_array[:, 1:].T, title="influence functions",
                   legend=numpy.arange(input_array.shape[1]-1))

        Q, R = np.linalg.qr(basis)

        print(">>> Q.shape, R.shape: ", Q.shape, R.shape)

        plot_table(abscissas, Q.T, title="GS basis functions",
                   legend=numpy.arange(Q.shape[1]))

        basis = Q
        if True: # check orthogonality
            for i in range(basis.shape[1]):
                for j in range(basis.shape[1]):
                    z1 = basis[:,i]
                    z2 = basis[:,j]
                    tmp = np.nansum(z1 * z2)
                    if np.abs(tmp) < 1e-12:
                        tmp = 0
                    print(">>> norm V profile %d  %d: %g" % (i, j, tmp))


        if True: # write to file
            filename = "z1d_basis_normalized.dat"
            f = open(filename,'w')
            for i in range(basis.shape[0]):
                f.write("%g  " % (abscissas[i]))
                for j in range(basis.shape[1]):
                    f.write("%12.10f  " % (basis[i,j]))
                f.write("\n")
            f.close()
            print("File %s written to disk." % filename)



    #
    # sample polynomials
    #
    if True:
        size = 128 * 4
        nsamples = 5
        seed = 69  # seed for generation of the random Zernike profiles
        # noll = [6, 8, 10, 11, 12, 14, 22, 37]
        noll = [6, 8, 10, 11, 14, 22, 37]  # removed 12!!!!!!!!!!!!!!!!!!

        C, Y  = create_1d_sampled_profiles(nsamples, size=size, noll=noll, factor=5.0, do_plot=0, seed=seed)
        C = np.array(C)

        print(Y.shape,C.shape)
        # Y[:,0] = np.linspace(-1,1,size)
        xx = np.linspace(-1500.0/2,1500.0/2,size)
        plot_table(xx, Y.T, xtitle="Lens position [um]", ytitle="Sampled thickness [um]",
                   legend=np.arange(nsamples), title="sampled profiles")

    #
    # 1D fitting with gram-schmidt polynomials
    #
    if True:

        filename = "z1d_basis.dat"
        tmp = np.loadtxt(filename)
        basis_x = tmp[:,0].copy()
        basis_pre = tmp[:,1:].copy()
        print(">> READ shapes basis_x, basis: ", basis_x.shape, basis_pre.shape)

        basis, R = np.linalg.qr(basis_pre)

        plot_table(basis_x, basis.T, xtitle="position [um]", ytitle="basis",
                   title="NORMALIZED basis",
                   legend=np.arange(basis.shape[1]))

        if True: # check orthogonality
            for i in range(basis.shape[1]):
                for j in range(basis.shape[1]):
                    z1 = basis[:,i]
                    z2 = basis[:,j]
                    tmp = np.nansum(z1 * z2)
                    if np.abs(tmp) < 1e-12:
                        tmp = 0
                    print(">>> norm V profile %d  %d: %g" % (i, j, tmp))


        for i in range(Y.shape[1]):
            Corig = C[:,i]
            Cfit = numpy.dot(R,Corig)
            vinfl = numpy.dot(np.linalg.inv(R), Cfit)

            print("\n\n>>>>> i: ", i)
            print("        >>>>> Cfit: ", Cfit)
            print("        >>>>> Corig: ", C[:,i])
            print("        >>>>> R.Corig: ", numpy.dot(R,C[:,i]))
            print("coefficients for influence functions basis inv: ", vinfl.shape, vinfl)

            Yi = Y[:, i]
            y2 = np.zeros_like(basis_x)
            for j in range(basis.shape[1]):
                base = basis[:,j]
                y2 += Cfit[j] * base
            plot(basis_x, Yi,
                 basis_x, y2,
                 legend=["data", "from fit"],
                 title="sample: %d" % i, marker=["+",None])



    # #
    # # 1D fitting trial not working
    # #
    # if False:
    #     for i in range(nsamples):
    #
    #         Cfit = []
    #         Yi = Y[:, i]
    #         y1 = np.zeros_like(basis_x)
    #         y2 = np.zeros_like(basis_x)
    #         for j in range(len(noll)):
    #             base = basis[:,j]
    #             Ci = (Yi * base).sum() / (2 * size)
    #             Cfit.append (Ci)
    #             y1 += C[j,i] * base
    #             y2 += Ci * base
    #         print("\n\n>>>>> i: ", i)
    #         print("        >>>>> Cfit: ", Cfit)
    #         print("        >>>>> Corig: ", C[:,i])
    #         plot(basis_x, Yi,
    #              basis_x, y1,
    #              basis_x, y2,
    #              legend=["data", "from basis", "from fit"],
    #              title="sample: %d" % i)
    #
    # #
    # # 2D fitting trial not working
    # #
    # if False:
    #     for i in range(nsamples):
    #
    #         Cfit = []
    #
    #         Yi = np.outer(Y[:, i], np.ones(size))
    #         y1 = np.zeros_like(basis_x)
    #         y2 = np.zeros_like(basis_x)
    #
    #         for j in range(len(noll)):
    #             w1 = Zernike(noll[j], order='noll').polynomial(size)
    #             v1 = Zernike(noll[j], order='noll').polynomial_vertical(size)
    #
    #             Ci = np.nansum( (Yi * w1) / size**2)
    #             Cfit.append (Ci)
    #             y1 += C[j,i] * w1[:, size//2]
    #             y2 += Ci * w1[:, size//2]
    #             # print(">>>> w1, y1, y2 shapes: ", w1.shape, y1.shape, y2.shape)
    #             # plot(basis_x, w1[size//2,:],
    #             #      basis_x, Yi[size//2,:],
    #             #      title="noll: %d" % noll[j], legend=["base", "profile"])
    #             # plot(basis_x, v1, basis_x, w1[:,size//2], legend=["pver", "p[]"], title="noll %d" % (noll[j]))
    #         print("\n\n>>>>> i: ", i)
    #         print("        >>>>> Cfit: ", Cfit)
    #         print("        >>>>> Corig: ", C[:,i])
    #         plot(basis_x, Yi[:, size//2],
    #              basis_x, y1,
    #              basis_x, y2,
    #              legend=["data", "from basis", "from fit"],
    #              title="sample: %d" % i,
    #              marker=[None,None,None])


