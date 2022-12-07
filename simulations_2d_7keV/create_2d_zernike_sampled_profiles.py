import numpy
from srxraylib.plot.gol import plot_image
from mlcrl.create_2d_zernike_sampled_profiles import create_2d_zernike_sampled_profiles
from oasys.util.oasys_util import write_surface_file



def write_generic_h5_surface(s, xx, yy, filename='presurface.hdf5',subgroup_name="surface_file"):
    write_surface_file(s.T, xx, yy, filename, overwrite=True)
    print("write_h5_surface: File for OASYS " + filename + " written to disk.")

if __name__ == "__main__":

    #
    # sampled polynomials
    #
    if True:
        size = 512
        nsamples = 4
        seed = 69  # seed for generation of the random Zernike profiles
        # noll         =             [6,     8,  10,  11,  12,  14,  22,  37]
        # distrubution =             ['n', 'n', 'n', 'u', 'n', 'n', 'u', 'u']
        # scale        = numpy.array([0.5, 0.5, 0.5, 2.3, .05, .05, 1.0, 0.5]) * 1e-6

        noll = [6, 8, 10, 11, 14, 22, 37]  # removed 12!!!!!!!!!!!!!!!!!!
        distrubution = ['n', 'n', 'n', 'u', 'n', 'u', 'u']
        scale = numpy.array([0.5, 0.5, 0.5, 2.3, .05, 1.0, 0.5]) * 1e-6
        width = 1500e-6

        C, Z  = create_2d_zernike_sampled_profiles(nsamples,
                                                   noll=noll, distrubution=distrubution, scale=scale, factor=5.0,
                                                   size=size, seed=seed, do_plot=0)


        print(Z.shape,C)




        x = numpy.linspace(-width/2,width/2,size)
        y = numpy.linspace(-width / 2, width / 2, size)

        if True:
            for i in range(Z.shape[2]):
                plot_image(Z[:,:,i], x*1e6, y*1e6, title="sampled # %s" % (i + 1), xtitle="X [um]", ytitle="Y [um]")


        #
        # write files
        #
        # dir = "./"
        # root = "tmp2D_ml"
        dir = "/scisoft/users/srio/ML_TRAIN2/5000_2Dv1/"
        root = "tmp_ml"

        for i in range(nsamples):
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
            write_generic_h5_surface(Z[:,:,i], x, y, filename=filename)
            print("File written to disk: %s" % filename)


        # overwrite 000 with no deformation
        # txt - no deformation
        filename = "%s%s%06d.txt" % (dir, root, 0)
        f = open(filename, 'w')

        f.write("# noll ")
        for j in range(len(noll)):
            f.write("%d  " % noll[j])
        f.write("\n")

        f.write("# coeff ")
        for j in range(len(noll)):
            f.write("%g  " % 0.0)
        f.write("\n")

        for j in range(len(noll)):
            f.write("%d  %g  \n" % (noll[j], 0.0))
        f.write("\n")

        print("File written to disk: %s" % filename)

        # h5 - no deformation
        filename = "%s%s%06d.h4" % (dir, root, 0)
        write_generic_h5_surface(Z[:, :, 0] * 0.0, x, y, filename=filename)
        print("File written to disk: %s" % filename)