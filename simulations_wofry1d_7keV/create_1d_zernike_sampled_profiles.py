
from mlcrl.phasenet.zernike import Zernike
import numpy
from srxraylib.plot.gol import plot, plot_table
import matplotlib.pyplot as plt

from mlcrl.create_1d_zernike_sampled_profiles import create_1d_zernike_sampled_profiles


if __name__ == "__main__":

    # For visualizing the Zernikes we define the Zernike object and call `polynomial()`.
    # The size of the 2D array is passed as a parameter

    if True:
        fig, ax = plt.subplots(4,10, figsize=(16,10))
        z_old = Zernike(1, order='noll').polynomial(128)
        for i,a in enumerate(ax.ravel()):
            z = Zernike(i+1, order='noll')
            w = z.polynomial(128)
            print(w.shape)
            a.imshow(w)
            if z.name is not None:
                a.set_title(z.name)
                # a.set_title("Noll: %d (%d,%d)" % (i + 1, z.n, z.m))
            else:
                a.set_title("Noll: %d (%d,%d)" % (i+1, z.n, z.m))
            a.axis('off')
            # print(">>>>>> i.(i-1) = ", numpy.nansum((z_old * w)) )

        plt.show()


    #
    # check normalization 2D
    #
    if True:
        noll = [6, 8, 10, 11, 12, 14, 22, 37]
        nnoll = len(noll)
        size = 128 * 4
        for i in range(nnoll):
            for j in range(nnoll):
                z1 = Zernike(noll[i], order='noll')
                z2 = Zernike(noll[j], order='noll')
                tmp = numpy.nansum(z1.polynomial(size=size) * z2.polynomial(size=size))/size**2
                if numpy.abs(tmp) < 1e-12:
                    tmp = 0
                print(">>> norm noll %d  %d: %g" % (noll[i],noll[j], tmp))

    #
    # check NON-normalization 1D
    #
    if True:
        size = 128 * 4
        noll = [6, 8, 10, 11, 12, 14, 22, 37]
        nnoll = len(noll)
        for i in range(nnoll):
            for j in range(nnoll):
                z1 = Zernike(noll[i], order='noll')
                z2 = Zernike(noll[j], order='noll')
                tmp = numpy.nansum(z1.polynomial_vertical(size=size) * z2.polynomial_vertical(size=size))/size
                if numpy.abs(tmp) < 1e-12:
                    tmp = 0
                print(">>> norm V profile noll %d  %d: %g" % (noll[i],noll[j], tmp))


    #
    # retrieve and plot 1d (vertical) Zernike profiles
    #

    if True:
        size = 128
        noll = [6, 8, 10, 11, 12, 14, 22, 37]

        fig, ax = plt.subplots(3,3, figsize=(10,10))
        A = ax.ravel()

        j = -1
        for i in noll:
            z = Zernike(i, order='noll')
            w = z.polynomial_vertical(size)
            print(i,z.name, z.n, z.m)
            if True: # z.m >= 0:
                j += 1
                a = A[j]
                a.plot(numpy.linspace(-1,1,size), w) #
            a.set_title("%s (%d,%d)" % (z.name, z.n, z.m))
            z_old = w
        plt.show()



    #
    # sampled polynomials
    #
    if True:
        size = 128
        nsamples = 20
        seed = 69  # seed for generation of the random Zernike profiles
        noll         =             [6,     8,  10,  11,  12,  14,  22,  37]
        distrubution =             ['n', 'n', 'n', 'u', 'n', 'n', 'u', 'u']
        scale        = numpy.array([0.5, 0.5, 0.5, 2.3, .05, .05, 1.0, 0.5]) * 1e-6
        width = 1500e-6

        C, Y  = create_1d_zernike_sampled_profiles(nsamples,
                                                   noll=noll, distrubution=distrubution, scale=scale, factor=5.0,
                                                   size=size, seed=seed, do_plot=0)


        print(Y.shape,C)
        xx = numpy.linspace(-width/2,width/2,size)
        plot_table(xx*1e6, Y.T*1e6, xtitle="Lens position [um]", ytitle="Sampled thickness [um]", title="sampled profiles",
                   legend=numpy.arange(nsamples))

        #
        # write files
        #
        dir = "./"
        root = "tmp_ml"

        for i in range(nsamples):
            #txt
            filename = "%s%s%04d.txt" % (dir, root, i+1)
            f = open(filename, 'w')

            f.write("# noll ")
            for j in range(len(noll)):
                f.write("%d  " % noll[j])
            f.write("\n")

            f.write("# coeff ")
            for j in range(len(noll)):
                f.write("%g  " % C[j,i])
            f.write("\n")

            print("File written to disk: %s" % filename)

            # dat
            filename = "%s%s%04d.dat" % (dir, root, i + 1)
            f = open(filename, 'w')

            for j in range(size):
                f.write("%g   %g\n" % (xx[j], Y[j,i]))
            f.close()
            print("File written to disk: %s" % filename)


        # overwrite 000 with no deformation
        # txt - no deformation
        filename = "%s%s%04d.txt" % (dir, root, 0)
        f = open(filename, 'w')

        f.write("# noll ")
        for j in range(len(noll)):
            f.write("%d  " % noll[j])
        f.write("\n")

        f.write("# coeff ")
        for j in range(len(noll)):
            f.write("%g  " % 0.0)
        f.write("\n")

        print("File written to disk: %s" % filename)

        # dat - no deformation
        filename = "%s%s%04d.dat" % (dir, root, 0)
        f = open(filename, 'w')

        for j in range(size):
            f.write("%g   %g\n" % (xx[j], 0.0))
        f.close()
        print("File written to disk: %s" % filename)
