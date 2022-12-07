
from mlcrl.phasenet.zernike import Zernike
import numpy
from srxraylib.plot.gol import plot, plot_table, plot_image
import matplotlib.pyplot as plt

def create_2d_zernike_sampled_profiles(nsamples,
                    noll=[6, 8, 10, 11, 12, 14, 22, 37],
                    distrubution=['n','n','n','n','n','n','n','n'],
                    scale=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                    size=128,
                    factor=1.0,
                    seed=69,
                    do_plot=False):

    Z = numpy.zeros((size, size, nsamples))
    C = numpy.zeros((len(noll), nsamples))

    rg = numpy.random.default_rng(seed)
    x = numpy.linspace(-1, 1, size)
    y = numpy.linspace(-1, 1, size)

    # noll       = [6,     8,  10,  11,  12,  14,  22, 37]
    # distrubution = ['n', 'n', 'n', 'u', 'n', 'n', 'u', 'u']
    # scale        = [0.5, 0.5, 0.5, 2.3, .05, .05, 1.0, 0.5]

    for i in range(nsamples):
        zz = numpy.zeros((size,size))

        for ij, j in enumerate(noll):
            z = Zernike(j, order='noll')

            if distrubution[ij] == 'n':
                c = rg.normal(loc=0, scale=scale[ij] * factor)
            elif distrubution[ij] == 'u':
                c = rg.uniform(-scale[ij] * factor, scale[ij] * factor)
            else:
                raise Exception("Why am I here?")

#Generator(PCG64) 0 1.785774589511436e-06 7
# Generator(PCG64) 0 8.465043621172215e-07 7
# Generator(PCG64) 0 6.654259141073615e-06 7
# Generator(PCG64) 0 5.3994894946457855e-06 7
# Generator(PCG64) 0 -3.987190882967194e-07 7
# Generator(PCG64) 0 3.166962286232809e-06 7
# Generator(PCG64) 0 2.231738805778543e-07 7
# Generator(PCG64) 1 -1.3682866656138383e-06 7
# Generator(PCG64) 1 -2.0353079624572042e-07 7
# Generator(PCG64) 1 -8.597814005268755e-07 7
# Generator(PCG64) 1 6.585212425935185e-06 7
# Generator(PCG64) 1 -1.0695773383463991e-07 7
# Generator(PCG64) 1 4.621093285144029e-06 7
#             print(rg,i,c, len(noll))
            w = z.polynomial(size, outside=0.0)
            zz += c * w
            C[ij, i] = c
        Z[:, :, i] = zz

        if do_plot: plot_image(zz, x, y, title="sampled # %s" % (i + 1))


    return C, Z

if __name__ == "__main__":



    # For visualizing the Zernikes we define the Zernike object and call `polynomial()`.
    # The size of the 2D array is passed as a parameter

    if False:
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
    if False:
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
    # sampled polynomials
    #
    if True:
        size = 128
        nsamples = 16
        seed = 69  # seed for generation of the random Zernike profiles
        noll         =             [6,     8,  10,  11,  12,  14,  22,  37]
        distrubution =             ['n', 'n', 'n', 'u', 'n', 'n', 'u', 'u']
        scale        = numpy.array([0.5, 0.5, 0.5, 2.3, .05, .05, 1.0, 0.5]) * 1e-6
        width = 1500e-6

        C, Z  = create_2d_zernike_sampled_profiles(nsamples,
                                                   noll=noll, distrubution=distrubution, scale=scale, factor=5.0,
                                                   size=size, seed=seed, do_plot=0)


        print(Z.shape,C)


        x = numpy.linspace(-width/2,width/2,size)
        y = numpy.linspace(-width / 2, width / 2, size)
        for i in range(Z.shape[2]):
            plot_image(Z[:,:,i], x*1e6, y*1e6, title="sampled # %s" % (i + 1), xtitle="X [um]", ytitle="Y [um]")

        #
        # write files
        #
        # dir = "./"
        # root = "tmp_ml"
        #
        # for i in range(nsamples):
        #     #txt
        #     filename = "%s%s%04d.txt" % (dir, root, i+1)
        #     f = open(filename, 'w')
        #
        #     f.write("# noll ")
        #     for j in range(len(noll)):
        #         f.write("%d  " % noll[j])
        #     f.write("\n")
        #
        #     f.write("# coeff ")
        #     for j in range(len(noll)):
        #         f.write("%g  " % C[j,i])
        #     f.write("\n")
        #
        #     print("File written to disk: %s" % filename)
        #
        #     # dat
        #     filename = "%s%s%04d.dat" % (dir, root, i + 1)
        #     f = open(filename, 'w')
        #
        #     for j in range(size):
        #         f.write("%g   %g\n" % (xx[j], Y[j,i]))
        #     f.close()
        #     print("File written to disk: %s" % filename)
        #
        #
        # # overwrite 000 with no deformation
        # # txt - no deformation
        # filename = "%s%s%04d.txt" % (dir, root, 0)
        # f = open(filename, 'w')
        #
        # f.write("# noll ")
        # for j in range(len(noll)):
        #     f.write("%d  " % noll[j])
        # f.write("\n")
        #
        # f.write("# coeff ")
        # for j in range(len(noll)):
        #     f.write("%g  " % 0.0)
        # f.write("\n")
        #
        # print("File written to disk: %s" % filename)
        #
        # # dat - no deformation
        # filename = "%s%s%04d.dat" % (dir, root, 0)
        # f = open(filename, 'w')
        #
        # for j in range(size):
        #     f.write("%g   %g\n" % (xx[j], 0.0))
        # f.close()
        # print("File written to disk: %s" % filename)
