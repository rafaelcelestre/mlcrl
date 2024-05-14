import numpy

# use testing_multimode_v23.py to create figure5.dat

def inset_plot(epochs, val_acc_values, acc_values):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                      mark_inset)


    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(epochs[::10], val_acc_values[::10], label='accuracy on validation set') #  'x', c='b', mew=2, alpha=0.8)
    ax1.plot(epochs[::10], acc_values[::10], label='accuracy on training set') # , c='m', lw=2, alpha=0.5)
    ax1.set_xlabel(r'Epochs')
    ax1.set_ylabel(r'Accuracy')
    ax1.legend(loc=0)



    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1

    # xrange = [700, 15300], yrange = [0.9, 1]
    x1, x2, y1, y2 = 700, 15300, 0.9, 1
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)

    ip = InsetPosition(ax1, [0.4, 0.2, 0.5, 0.5])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax1, ax2, loc1=3, loc2=1, fc="none", ec='0.5')

    # The data
    ax2.plot(epochs[::10], val_acc_values[::10])
    ax2.plot(epochs[::10], acc_values[::10])

    plt.show()

if __name__ == "__main__":

    a = numpy.loadtxt('figure5.dat')
    epochs = a[:,0]
    val_acc_values = a[:,1]
    acc_values = a[:,2]

    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})

    inset_plot(epochs, val_acc_values, acc_values)

