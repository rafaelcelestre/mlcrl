
import numpy
from srxraylib.plot.gol import plot, plot_show

if __name__ == "__main__":

    v20_101 = numpy.loadtxt('tmp_v20_101.dat')
    v20_102 = numpy.loadtxt('tmp_v20_102.dat')
    v20_103 = numpy.loadtxt('tmp_v20_103.dat')
    v20_104 = numpy.loadtxt('tmp_v20_104.dat')
    v20_105 = numpy.loadtxt('tmp_v20_105.dat')

    v23_101 = numpy.loadtxt('tmp_v23_101.dat')
    v23_102 = numpy.loadtxt('tmp_v23_102.dat')
    v23_103 = numpy.loadtxt('tmp_v23_103.dat')
    v23_104 = numpy.loadtxt('tmp_v23_104.dat')
    v23_105 = numpy.loadtxt('tmp_v23_105.dat')

    import matplotlib
    import matplotlib.pylab as plt
    matplotlib.rcParams.update({'font.size': 14})

    plot(v20_101[:, 0], v20_101[:, 1],
         v20_101[:, 0], v20_101[:, 2],
         v23_101[:, 0], v23_101[:, 2],
         [v23_101[0, 0], v23_101[-1, 0]], [0,0],
         v20_102[:, 0], v20_102[:, 1] + 55,
         v20_102[:, 0], v20_102[:, 2] + 55,
         v23_102[:, 0], v23_102[:, 2] + 55,
         [v23_101[0, 0], v23_101[-1, 0]], [55, 55],
         v20_103[:, 0], v20_103[:, 1] + 55 + 30,
         v20_103[:, 0], v20_103[:, 2] + 55 + 30,
         v23_103[:, 0], v23_103[:, 2] + 55 + 30,
         [v23_101[0, 0], v23_101[-1, 0]], [55 + 30, 55 + 30],
         v20_104[:, 0], v20_104[:, 1] + 55 + 30 + 40,
         v20_104[:, 0], v20_104[:, 2] + 55 + 30 + 40,
         v23_104[:, 0], v23_104[:, 2] + 55 + 30 + 40,
         [v23_101[0, 0], v23_101[-1, 0]], [55 + 30 + 40, 55 + 30 + 40],
         v20_105[:, 0], v20_105[:, 1] + 55 + 30 + 40 + 35,
         v20_105[:, 0], v20_105[:, 2] + 55 + 30 + 40 + 35,
         v23_105[:, 0], v23_105[:, 2] + 55 + 30 + 40 + 35,
         [v23_101[0, 0], v23_101[-1, 0]], [55 + 30 + 40 + 35, 55 + 30 + 40 + 35],
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


    plt.savefig('figure6.pdf')
    plot_show()

