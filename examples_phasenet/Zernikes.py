
from mlcrl.phasenet.zernike import Zernike
import matplotlib.pyplot as plt
import numpy as np
from srxraylib.plot.gol import plot

def get_polymonials_vertical(size=128,     noll=[5,6,7,8,9,10,11,12,13,14,22,37]):
    polymonials_vertical = np.zeros((len(noll), size))
    j = -1

    for i in noll:
        z = Zernike(i, order='noll')
        w = z.polynomial_vertical(size)
        print(i, z.name, z.n, z.m)
        if True: # z.m >= 0:
            j += 1
            polymonials_vertical[j, :] = w

    return polymonials_vertical, noll


if __name__ == "__main__":


    if False:
        print(">>", Zernike(5, order='noll')) # can be an integer index following noll index
        print(">>", Zernike(3, order='ansi')) # can be an integer index following ansi index
        print(">>", Zernike((2,-2))) # can be a tuple for (n,m) indexing
        print(">>", Zernike('oblique astigmatism')) # can be a string

        print(">>", Zernike((2,-2)) == Zernike(5, order='noll'))


        # All Zernikes defined in any format (ansi for example here) are internally mapped to their respective index in other formats

        for i in range(15):
            print(Zernike(i, order='ansi'))


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


    # fig, ax = plt.subplots(3,5, figsize=(16,10))
    # for i,a in enumerate(ax.ravel()):
    #     z = Zernike(i+1, order='noll')
    #     w = z.polynomial(128)
    #     print(i,z.name,w.shape)
    #     a.imshow(w)
    #     a.set_title(z.name)
    #     a.axis('off')
    # plt.show()




    #
    # retrieve and plot 1d (vertical) profiles to be used in Wofry1D
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

    seed = 69  # seed for generation of the random Zernike profiles
    rg = np.random.default_rng(seed)

    x = np.linspace(-1,1,size)
    noll = [6, 8, 10, 11, 12, 14, 22, 37]
    n = len(noll)
    nsamples = 10


    for i in range(nsamples):
        Y = np.zeros_like(x)
        for j in noll:
            z = Zernike(j, order='noll')

            if j <= 10:
                # Z5 (Zcoeffs[4]) to Z10 (Zcoeffs[9]) are more important and have a higher weight
                c = rg.normal(loc=0, scale=0.5)
            elif j == 11:
                # Spherical aberration 1st order (Z11)
                c = rg.uniform(-2.3, 2.3)
            elif j >= 12 and j <=21:
                # Z12 (Zcoeffs[11]) to Z21 (Zcoeffs[20]) and Z23 (Zcoeffs[22]) to Z36 (Zcoeffs[35]) are very low
                c = rg.normal(loc=0, scale=0.05)
            elif j == 22:
                # Spherical aberration 2nd order (Z22)
                c = rg.uniform(-1., 1.)
            elif j == 37:
                # Spherical aberration 3nd order (Z37)
                c = rg.uniform(-0.5, 0.5)
            else:
                raise Exception("Why I am here?")

            w = z.polynomial_vertical(size)
            Y += c * w


        plot(x, Y, title="sampled # %s" % (i+1))