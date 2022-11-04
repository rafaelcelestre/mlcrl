
#
# Import section
#
import numpy

from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates
from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters

from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D

from wofryimpl.propagator.propagators1D.fresnel import Fresnel1D
from wofryimpl.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
from wofryimpl.propagator.propagators1D.fraunhofer import Fraunhofer1D
from wofryimpl.propagator.propagators1D.integral import Integral1D
from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
from wofryimpl.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D

from srxraylib.plot.gol import plot, plot_image, plot_show, plot_table
plot_from_oe = 100 # set to a large number to avoid plots


##########  SOURCE ##########
def run_source():

    #
    # create output_wavefront
    #
    #
    from wofryimpl.propagator.util.undulator_coherent_mode_decomposition_1d import UndulatorCoherentModeDecomposition1D
    coherent_mode_decomposition = UndulatorCoherentModeDecomposition1D(
        electron_energy=6,
        electron_current=0.2,
        undulator_period=0.018,
        undulator_nperiods=138,
        K=1.85108,
        photon_energy=7000,
        abscissas_interval=0.00025,
        number_of_points=1500,
        distance_to_screen=100,
        scan_direction='V',
        sigmaxx=5.2915e-06,
        sigmaxpxp=1.88982e-06,
        useGSMapproximation=False,)
    # make calculation
    coherent_mode_decomposition_results = coherent_mode_decomposition.calculate()

    mode_index = 0
    output_wavefront = coherent_mode_decomposition.get_eigenvector_wavefront(mode_index)


    if plot_from_oe <= 0: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='SOURCE')

    return output_wavefront

def run_beamline(output_wavefront, file_with_thickness_mesh="tmp_ml0000.dat", distance=3.591600):
    ##########  OPTICAL SYSTEM ##########

    ##########  OPTICAL ELEMENT NUMBER 1 ##########

    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D

    optical_element = WOScreen1D()

    # drift_before 36 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,
                                       coordinates=ElementCoordinates(p=36.000000, q=0.000000,
                                                                      angle_radial=numpy.radians(0.000000),
                                                                      angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront, propagation_elements=propagation_elements)
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('magnification_x', 10.0)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                 handler_name='FRESNEL_ZOOM_1D')

    #
    # ---- plots -----
    #
    if plot_from_oe <= 1: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 1')

    ##########  OPTICAL ELEMENT NUMBER 2 ##########

    input_wavefront = output_wavefront.duplicate()
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(-0.5, 0.5, -0.5, 0.5)
    from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D
    optical_element = WOSlit1D(boundary_shape=boundary_shape)

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)

    #
    # ---- plots -----
    #
    if plot_from_oe <= 2: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 2')

    ##########  OPTICAL ELEMENT NUMBER 3 ##########

    input_wavefront = output_wavefront.duplicate()
    from syned.beamline.shape import Rectangle
    boundary_shape = Rectangle(-0.5, 0.5, -0.5, 0.5)
    from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D
    optical_element = WOSlit1D(boundary_shape=boundary_shape)

    # drift_before 29 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,
                                       coordinates=ElementCoordinates(p=29.000000, q=0.000000,
                                                                      angle_radial=numpy.radians(0.000000),
                                                                      angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront, propagation_elements=propagation_elements)
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('magnification_x', 1.5)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                 handler_name='FRESNEL_ZOOM_1D')

    #
    # ---- plots -----
    #
    if plot_from_oe <= 3: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 3')

    ##########  OPTICAL ELEMENT NUMBER 4 ##########

    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.refractors.lens import WOLens1D

    optical_element = WOLens1D.create_from_keywords(
        name='Lens50um',
        shape=1,
        radius=5e-05,
        lens_aperture=0.0015,
        wall_thickness=5e-05,
        material='Be',
        number_of_curved_surfaces=2,
        n_lenses=1,
        error_flag=1,
        error_file=file_with_thickness_mesh,
        error_edge_management=1,
        write_profile_flag=0,
        write_profile='profile1D.dat',
        mis_flag=0,
        xc=0,
        ang_rot=0,
        wt_offset_ffs=0,
        offset_ffs=0,
        tilt_ffs=0,
        wt_offset_bfs=0,
        offset_bfs=0,
        tilt_bfs=0,
        verbose=0)

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)

    #
    # ---- plots -----
    #
    if plot_from_oe <= 4: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 4')

    ##########  OPTICAL ELEMENT NUMBER 5 ##########

    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D

    optical_element = WOScreen1D()

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)

    #
    # ---- plots -----
    #
    if plot_from_oe <= 5: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 5')

    ##########  OPTICAL ELEMENT NUMBER 6 ##########

    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D

    optical_element = WOScreen1D()

    # drift_before 3.5916 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,
                                       coordinates=ElementCoordinates(p=distance, q=0.000000,
                                                                      angle_radial=numpy.radians(0.000000),
                                                                      angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront, propagation_elements=propagation_elements)
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('magnification_x', 0.1)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                 handler_name='FRESNEL_ZOOM_1D')

    #
    # ---- plots -----
    #
    if plot_from_oe <= 6: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 6')

    return output_wavefront

if __name__ == "__main__":
    import matplotlib.pylab as plt
    from srxraylib.util.h5_simple_writer import H5SimpleWriter


    distance0=3.591600
    delta = 1.0
    npoints = 64

    distance = numpy.linspace(distance0-0.5*delta, distance0+0.5*delta, npoints)

    do_intermediate_plot = 0
    do_plot = 0

    nsamples = 1000
    nsamples_plot = 5

    dir = "/users/srio/Oasys/ML_TRAIN/"  # where the profile files sit
    dir_out = "/scisoft/users/srio/ML_TRAIN2/" # where the results are going to be written
    root = "tmp_ml"

    #
    # calculate
    #
    if True:
        src = run_source()

        for nn in range(nsamples):
            if numpy.mod(nn,50) == 0: print("Calculating sample %d of %d..." % (nn+1,nsamples))
            for i in range(npoints):
                file_with_thickness_mesh = "%s%s%06d.dat" % (dir, root, nn)
                output_wavefront = run_beamline(src, file_with_thickness_mesh=file_with_thickness_mesh, distance=distance[i])

                if do_intermediate_plot: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),
                     title='LAST OPTICAL ELEMENT d=%s' % (distance[i]))

                if i == 0:
                    Z = numpy.zeros((npoints, output_wavefront.get_abscissas().size))
                    x = output_wavefront.get_abscissas()
                Z[i, :] = output_wavefront.get_intensity()

            if nn == 0:
                ZZ = numpy.zeros(((nsamples, npoints, output_wavefront.get_abscissas().size)))

            ZZ[nn,:,:] = Z



    #
    # plot
    #
    if do_plot:
        iplot = -1
        fig, ax = plt.subplots(nsamples_plot, 3, figsize=(10, 15))
        A = ax.ravel()

        for nn in range(nsamples_plot):
            file_with_thickness_mesh = "%s%s%06d.dat" % (dir, root, nn)
            tmp = numpy.loadtxt(file_with_thickness_mesh)

            if do_plot:
                iplot += 1
                print(">>>iplot: ", iplot)
                A[iplot].plot(tmp[:,0] * 1e6, tmp[:,1] * 1e6) #
                A[iplot].set_title("sample %d" % nn)

                iplot += 1
                A[iplot].plot(distance, ZZ[nn, :, Z.shape[1]//2]) #
                A[iplot].set_title("sample %d" % nn)

                iplot += 1
                A[iplot].plot(x * 1e6, ZZ[nn, Z.shape[0] // 2, :],) #
                A[iplot].set_title("sample %d" % nn)


        if do_plot: plt.show()

        print(x.shape, ZZ.shape)
        plot_table(x * 1e6, ZZ[:, 0, :], legend=numpy.arange(nsamples), title="closer")
        plot_table(x * 1e6, ZZ[:,ZZ.shape[1]//2,:], legend=numpy.arange(nsamples), title="center")
        plot_table(x * 1e6, ZZ[:, -1, :], legend=numpy.arange(nsamples), title="farer")

    #
    # write h5 (at every sample)
    #
    if True:

        h5_file = "%s%s.h5" % (dir_out, root)
        h5w = H5SimpleWriter.initialize_file(h5_file,creator="h5_basic_writer.py")

        for i in range(nsamples):
            # create the entry for this iteration and set default plot to "Wintensity"
            h5w.create_entry("sample%06d"%i,nx_default="Intensity")


            file_with_info = "%s%s%06d.txt" % (dir, root, i)
            tmp = numpy.loadtxt(file_with_info)
            h5w.add_key("Noll-Zcoeff-GScoeff",
                        tmp,
                        entry_name="sample%06d"%i)


            # add the images at this entry level
            h5w.add_image(ZZ[i,:,:],distance,1e6*x,
                         entry_name="sample%06d"%i,image_name="Intensity",
                         title_x="distance [m]",title_y="x [um]")

            # add deformation profile
            file_with_thickness_mesh = "%s%s%06d.dat" % (dir, root, i)
            tmp = numpy.loadtxt(file_with_thickness_mesh)
            h5w.add_dataset(1e6*tmp[:,0],1e6*tmp[:,1],
                        entry_name="sample%06d"%i,dataset_name="profile",
                        title_x="X [um]",title_y="Y[um]")

        print("File %s written to disk." % h5_file)

    #
    # write h5 (block data) includes interpolation and base removal
    #
    if True:
        abscissa_new = numpy.linspace(-125e-6,125e-6,256)

        ZZblock = numpy.zeros((nsamples, abscissa_new.size, npoints))
        for i in range(nsamples):
            for j in range(npoints):
                y_orig = ZZ[i,j,:]
                y_int = numpy.interp(abscissa_new, x, y_orig)
                ZZblock[i, :, j] = y_int - y_int.min()  # TODO: remove?


        h5_file = "%s%s_block.h5" % (dir_out, root)
        h5w = H5SimpleWriter.initialize_file(h5_file, creator="h5_basic_writer.py")

        h5w.create_entry("allsamples",nx_default="intensity")

        h5w.add_stack(numpy.arange(nsamples), abscissa_new*1e6, distance, ZZblock,
                  stack_name="intensity", entry_name="allsamples",
                  title_0="sample", title_1="abscissa [um]", title_2="distance [m]",
                      )

        print("File %s written to disk." % h5_file)

    #
    # write file with targets
    #
    if True:
        filename = "%s%s_targets_gs.txt" % (dir_out, root)
        f = open(filename,'w')
        for i in range(nsamples):
            file_with_info = "%s%s%06d.txt" % (dir, root, i)
            tmp = numpy.loadtxt(file_with_info, skiprows=3)
            f.write("%6d   " % i)
            for j in range(tmp.shape[0]):
                f.write("%10.8g   " % (tmp[j,2] * 1e6)) #   in microns!!!!
            f.write("\n")
        print("File %s written to disk." % filename)