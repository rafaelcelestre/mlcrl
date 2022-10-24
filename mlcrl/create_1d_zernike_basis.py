import numpy
from mlcrl.phasenet.zernike import Zernike
from srxraylib.plot.gol import plot_table


def create_1d_zernike_basis(size=512, noll=[6, 8, 10, 11, 12, 14, 22, 37], width=2.0, filename=None, do_plot=False):

    nnoll = len(noll)

    basis = numpy.zeros((size, len(noll)))
    basis_x = numpy.linspace(-0.5*width, 0.5*width, size)
    for i in range(len(noll)):
        z = Zernike(noll[i], order='noll')
        w = z.polynomial_vertical(size)
        basis[:, i] = w

    if do_plot: plot_table(basis_x, basis.T, xtitle="position [um]", ytitle="basis",
               title="non-orthogonal basis (noll-indexed Vertical cut of Zernike polynomials)", legend=noll)

    if filename is not None:  # write to file
        f = open(filename, 'w')
        for i in range(size):
            f.write("%g  " % (basis_x[i]))
            for j in range(nnoll):
                f.write("%12.10f  " % (basis[i, j]))
            f.write("\n")
        f.close()
        print("File %s written to disk." % filename)

    return basis_x, basis