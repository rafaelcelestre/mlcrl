
from mlcrl.phasenet.zernike import Zernike
import matplotlib.pyplot as plt
import numpy as np
from srxraylib.plot.gol import plot, plot_table

def create_1d_sampled_profiles(nsamples, size=128, noll=[6, 8, 10, 11, 12, 14, 22, 37], factor=1.0, do_plot=False):

    Y = np.zeros((size, nsamples))
    C = np.zeros((len(noll), nsamples))

    rg = np.random.default_rng(seed)
    x = np.linspace(-1, 1, size)

    for i in range(nsamples):
        y = np.zeros_like(x)

        for ij, j in enumerate(noll):
            z = Zernike(j, order='noll')

            if j <= 10:
                # Z5 (Zcoeffs[4]) to Z10 (Zcoeffs[9]) are more important and have a higher weight
                c = rg.normal(loc=0, scale=0.5 * factor)
            elif j == 11:
                # Spherical aberration 1st order (Z11)
                c = rg.uniform(-2.3 * factor, 2.3 * factor)
            elif j >= 12 and j <= 21:
                # Z12 (Zcoeffs[11]) to Z21 (Zcoeffs[20]) and Z23 (Zcoeffs[22]) to Z36 (Zcoeffs[35]) are very low
                c = rg.normal(loc=0, scale=0.05 * factor)
            elif j == 22:
                # Spherical aberration 2nd order (Z22)
                c = rg.uniform(-1. * factor, 1. * factor)
            elif j == 37:
                # Spherical aberration 3nd order (Z37)
                c = rg.uniform(-0.5 * factor, 0.5 * factor)
            else:
                raise Exception("Why am I here?")

            w = z.polynomial_vertical(size)
            y += c * w
            C[ij, i] = c
        Y[:, i] = y

        if do_plot: plot(x, y, title="sampled # %s" % (i + 1))

    return C, Y

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
            print(">>>>>> i.(i-1) = ", np.nansum((z_old * w)) )

        plt.show()


    #
    # retrieve and plot 1d (vertical) Zernike profiles to be used in Wofry1D
    #

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
            a.plot(np.linspace(-1,1,size), w) #
        a.set_title("%s (%d,%d)" % (z.name, z.n, z.m))
        z_old = w
    plt.show()



    #
    # sampled polynomials
    #

    nsamples = 20
    seed = 69  # seed for generation of the random Zernike profiles
    noll = [6, 8, 10, 11, 12, 14, 22, 37]

    C, Y  = create_1d_sampled_profiles(nsamples, size=size, noll=noll, factor=5.0, do_plot=0)


    print(Y.shape,C)
    # Y[:,0] = np.linspace(-1,1,size)
    xx = np.linspace(-1500.0/2,1500.0/2,size)
    plot_table(xx, Y.T, xtitle="Lens position [um]", ytitle="Sampled thickness [um]")

    #
    # write files
    #
    dir = "./" # "/users/srio/Oasys/"
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
            f.write("%g   %g\n" % (xx[j]*1e-6, Y[j,i]*1e-6))
        f.close()
        print("File written to disk: %s" % filename)


    # txt none
    # txt
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

    # dat None
    filename = "%s%s%04d.dat" % (dir, root, 0)
    f = open(filename, 'w')

    for j in range(size):
        f.write("%g   %g\n" % (xx[j]*1e-6, 0.0))
    f.close()
    print("File written to disk: %s" % filename)
