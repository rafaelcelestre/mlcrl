
import numpy
from srxraylib.plot.gol import plot, plot_show

if __name__ == "__main__":

    v13_101 = numpy.loadtxt('tmp_v13_101.dat')
    v13_102 = numpy.loadtxt('tmp_v13_102.dat')
    v13_103 = numpy.loadtxt('tmp_v13_103.dat')
    v13_104 = numpy.loadtxt('tmp_v13_104.dat')
    v13_105 = numpy.loadtxt('tmp_v13_105.dat')

    v14_101 = numpy.loadtxt('tmp_v14_101.dat')
    v14_102 = numpy.loadtxt('tmp_v14_102.dat')
    v14_103 = numpy.loadtxt('tmp_v14_103.dat')
    v14_104 = numpy.loadtxt('tmp_v14_104.dat')
    v14_105 = numpy.loadtxt('tmp_v14_105.dat')

    import matplotlib
    import matplotlib.pylab as plt
    matplotlib.rcParams.update({'font.size': 14})

    plot(v13_101[:, 0], v13_101[:, 1],
         v13_101[:, 0], v13_101[:, 2],
         v14_101[:, 0], v14_101[:, 2],
         [v14_101[0, 0], v14_101[-1, 0]], [0,0],
         v13_102[:, 0], v13_102[:, 1] + 55,
         v13_102[:, 0], v13_102[:, 2] + 55,
         v14_102[:, 0], v14_102[:, 2] + 55,
         [v14_101[0, 0], v14_101[-1, 0]], [55, 55],
         v13_103[:, 0], v13_103[:, 1] + 55 + 30,
         v13_103[:, 0], v13_103[:, 2] + 55 + 30,
         v14_103[:, 0], v14_103[:, 2] + 55 + 30,
         [v14_101[0, 0], v14_101[-1, 0]], [55 + 30, 55 + 30],
         v13_104[:, 0], v13_104[:, 1] + 55 + 30 + 40,
         v13_104[:, 0], v13_104[:, 2] + 55 + 30 + 40,
         v14_104[:, 0], v14_104[:, 2] + 55 + 30 + 40,
         [v14_101[0, 0], v14_101[-1, 0]], [55 + 30 + 40, 55 + 30 + 40],
         v13_105[:, 0], v13_105[:, 1] + 55 + 30 + 40 + 35,
         v13_105[:, 0], v13_105[:, 2] + 55 + 30 + 40 + 35,
         v14_105[:, 0], v14_105[:, 2] + 55 + 30 + 40 + 35,
         [v14_101[0, 0], v14_101[-1, 0]], [55 + 30 + 40 + 35, 55 + 30 + 40 + 35],
         # legend=['original', 'predicted 1500 epochs',  'predicted 24000 epochs'],
         title='', xtitle=r'abscissas [mm]', ytitle=r'profile height [$\mu$m]', xrange=[-0.75,0.75],
         color=['#1f77b4', '#ff7f0e', '#2ca02c','k',
                '#1f77b4', '#ff7f0e', '#2ca02c','k',
                '#1f77b4', '#ff7f0e', '#2ca02c','k',
                '#1f77b4', '#ff7f0e', '#2ca02c','k',
                '#1f77b4', '#ff7f0e', '#2ca02c','k',
                ],
         linestyle=['--', ':', ':', '-.',
                    '--', ':', ':', '-.',
                    '--', ':', ':', '-.',
                    '--', ':', ':', '-.',
                    '--', ':', ':', '-.',
                    ], figsize=(8,10), show=0)


    plt.savefig('figure4.pdf')
    plot_show()

