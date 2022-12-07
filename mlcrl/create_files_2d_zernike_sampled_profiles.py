
from mlcrl.phasenet.zernike import Zernike
import numpy
from srxraylib.plot.gol import plot, plot_table, plot_image

from oasys.util.oasys_util import write_surface_file



def write_generic_h5_surface(s, xx, yy, filename='presurface.hdf5',subgroup_name="surface_file"):
    write_surface_file(s.T, xx, yy, filename, overwrite=True)
    print("write_h5_surface: File for OASYS " + filename + " written to disk.")


def create_files_2d_zernike_sampled_profiles(nsamples,
                    noll=[6, 8, 10, 11, 12, 14, 22, 37],
                    distrubution=['n','n','n','n','n','n','n','n'],
                    scale=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                    size=128,
                    factor=1.0,
                    seed=69,
                    width=1000e-6,
                    dir="./",
                    root = "tmp_ml",
                    do_plot=False):

    # Z = numpy.zeros((size, size, nsamples))
    C = numpy.zeros((len(noll), nsamples))

    rg = numpy.random.default_rng(seed)
    x = numpy.linspace(-1, 1, size)
    y = numpy.linspace(-1, 1, size)

    xout = numpy.linspace(-width / 2, width / 2, size)
    yout = numpy.linspace(-width / 2, width / 2, size)

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

            w = z.polynomial(size, outside=0.0)
            zz += c * w
            C[ij,i] = c
        # Z[:, :, i] = zz
        if do_plot: plot_image(zz, xout, yout, title="sampled # %s" % (i + 1))


        #
        # write file
        #

        if i == 0: # first file has no deformation
            zz * 0
            C[:,i] = 0.0
        #txt
        filename = "%s%s%06d.txt" % (dir, root, i)
        f = open(filename, 'w')

        f.write("# noll ")
        for j in range(len(noll)):
            f.write("%d  " % noll[j])
        f.write("\n")

        f.write("# coeff ")
        for j in range(len(noll)):
            f.write("%g  " % C[j,i])
        f.write("\n")

        for j in range(len(noll)):
            f.write("%d  %g  \n" % (noll[j], C[j, i]))
        f.write("\n")

        print("File written to disk: %s" % filename)

        # h5

        filename = "%s%s%06d.h5" % (dir, root, i)
        write_generic_h5_surface(zz, xout, yout, filename=filename)
        print("File written to disk: %s" % filename)

    return C






if __name__ == "__main__":



    #
    # sampled polynomials
    #
    if True:
        size = 128
        nsamples = 3
        seed = 69  # seed for generation of the random Zernike profiles
        noll         =             [6,     8,  10,  11,  12,  14,  22,  37]
        distrubution =             ['n', 'n', 'n', 'u', 'n', 'n', 'u', 'u']
        scale        = numpy.array([0.5, 0.5, 0.5, 2.3, .05, .05, 1.0, 0.5]) * 1e-6
        width = 1500e-6

        create_files_2d_zernike_sampled_profiles(nsamples,
                                                   noll=noll, distrubution=distrubution, scale=scale, factor=5.0,
                                                   size=size, seed=seed, do_plot=1,
                                                width=width)





        # x = numpy.linspace(-width/2,width/2,size)
        # y = numpy.linspace(-width / 2, width / 2, size)
        # for i in range(Z.shape[2]):
        #     plot_image(Z[:,:,i], x*1e6, y*1e6, title="sampled # %s" % (i + 1), xtitle="X [um]", ytitle="Y [um]")

