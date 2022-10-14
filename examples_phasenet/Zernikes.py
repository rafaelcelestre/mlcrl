
from mlcrl.phasenet.zernike import Zernike
import matplotlib.pyplot as plt
import numpy as np
from srxraylib.plot.gol import plot

def get_polymonials_vertical(size=128):
    polymonials_vertical = np.zeros((6, size))
    j = -1
    for i in range(15):
        z = Zernike(i + 1, order='noll')
        if i > 4:
            # w = z.polynomial(128)
            # print(i,z.name, z.n, z.m)
            # if z.m >= 0:
            #     a.plot(np.linspace(-1,1,128), w[:,128//2]) #

            w = z.polynomial_vertical(size)
            print(i, z.name, z.n, z.m)
            if z.m >= 0:
                j += 1
                polymonials_vertical[j, :] = w
    return polymonials_vertical


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


    fig, ax = plt.subplots(3,5, figsize=(16,10))
    for i,a in enumerate(ax.ravel()):
        z = Zernike(i+1, order='noll')
        w = z.polynomial(128)
        print(w.shape)
        a.imshow(w)
        a.set_title(z.name)
        a.axis('off')
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


    # retrieve
    polymonials_vertical = get_polymonials_vertical(size)


    #
    # plot
    #
    fig, ax = plt.subplots(3,5, figsize=(16,10))
    for i,a in enumerate(ax.ravel()):
        z = Zernike(i + 1, order='noll')
        if i > 4:
            w = z.polynomial_vertical(size)
            print(i,z.name, z.n, z.m)
            if z.m >= 0:
                a.plot(np.linspace(-1,1,size), w) #
        a.set_title("%s (%d,%d)" % (z.name, z.n, z.m))
    plt.show()



    #
    # wavefront deformation
    #
    x = np.linspace(-1,1,size)
    weights = [0.0,0.2,0.0,0.5,0.0,0.0]

    for j in range(10):
        c = np.random.rand(6)
        # c = np.array([0,1,0,0,0,0])

        for i in range(6):
            if i == 0:
                Y = weights[i] * c[i] * polymonials_vertical[i,:]
            else:
                Y += weights[i] * c[i] * polymonials_vertical[i, :]


        plot(x, Y, title=str(c)
             )