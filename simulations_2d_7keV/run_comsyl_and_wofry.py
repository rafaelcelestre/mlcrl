
#
# Import section
#
import numpy

from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.element_coordinates import ElementCoordinates
from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters

from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D

from wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D
from wofryimpl.propagator.propagators2D.fresnel import Fresnel2D
from wofryimpl.propagator.propagators2D.fresnel_convolution import FresnelConvolution2D
from wofryimpl.propagator.propagators2D.fraunhofer import Fraunhofer2D
from wofryimpl.propagator.propagators2D.integral import Integral2D
from wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

from srxraylib.plot.gol import plot, plot_image
plot_from_oe = 500 # set to a large number to avoid plots

def run_source(mode_index = 0):
    ##########  SOURCE ##########


    #
    # create output_wavefront
    #
    #
    from comsyl.autocorrelation.CompactAFReader import CompactAFReader
    filename = "//scisoft/users/srio/COMSYL-SLURM/comsyl/comsyl-submit/calculations/id18_ebs_u18_2500mm_s12.0.npz"
    af_oasys = CompactAFReader.initialize_from_file(filename)


    output_wavefront = af_oasys.get_wavefront(mode_index,normalize_with_eigenvalue=1)


    if plot_from_oe <= 0: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='SOURCE')

    return output_wavefront

def run_beamline(output_wavefront, file_with_thickness_mesh="", distance=3.590000):
    ##########  OPTICAL SYSTEM ##########





    ##########  OPTICAL ELEMENT NUMBER 1 ##########



    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen

    optical_element = WOScreen()

    # drift_before 36 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=36.000000,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront,    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 5.0)
    propagation_parameters.set_additional_parameters('magnification_y', 10.0)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_XY_2D')


    #
    #---- plots -----
    #
    if plot_from_oe <= 1: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='OPTICAL ELEMENT NR 1')


    ##########  OPTICAL ELEMENT NUMBER 2 ##########



    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen

    optical_element = WOScreen()

    # drift_before 29 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=29.000000,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront,    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 2.5)
    propagation_parameters.set_additional_parameters('magnification_y', 1.5)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_XY_2D')


    #
    #---- plots -----
    #
    if plot_from_oe <= 2: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='OPTICAL ELEMENT NR 2')


    ##########  OPTICAL ELEMENT NUMBER 3 ##########



    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.refractors.lens import WOLens

    optical_element = WOLens.create_from_keywords(
        name='Real Lens 2D',
        number_of_curved_surfaces=2,
        two_d_lens=0,
        surface_shape=0,
        wall_thickness=5e-05,
        material='Be',
        lens_radius=5e-05,
        n_lenses=1,
        aperture_shape=0,
        aperture_dimension_h=0.0015,
        aperture_dimension_v=0.0015,
        verbose=1)

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)


    #
    #---- plots -----
    #
    if plot_from_oe <= 3: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='OPTICAL ELEMENT NR 3')


    ##########  OPTICAL ELEMENT NUMBER 4 ##########



    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.refractors.thin_object import WOThinObject

    optical_element = WOThinObject(name='ThinObject',file_with_thickness_mesh=file_with_thickness_mesh,material='Be')

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)


    #
    #---- plots -----
    #
    if plot_from_oe <= 4: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='OPTICAL ELEMENT NR 4')


    ##########  OPTICAL ELEMENT NUMBER 5 ##########



    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen

    optical_element = WOScreen()

    # drift_before 3.59 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=distance,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront,    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('shift_half_pixel', 1)
    propagation_parameters.set_additional_parameters('magnification_x', 0.1)
    propagation_parameters.set_additional_parameters('magnification_y', 0.14)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoomXY2D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_XY_2D')


    #
    #---- plots -----
    #
    if plot_from_oe <= 5: plot_image(output_wavefront.get_intensity(),output_wavefront.get_coordinate_x(),output_wavefront.get_coordinate_y(),aspect='auto',title='OPTICAL ELEMENT NR 5')

    return output_wavefront

if __name__ == "__main__":

    if False:
        wf = run_source(mode_index = 0)
        file_with_thickness_mesh = '/scisoft/users/srio/ML_TRAIN2/5000_2Dv1/tmp_ml000002.h5'
        wf = run_beamline(wf, file_with_thickness_mesh=file_with_thickness_mesh, distance=3.0916)
        # wf = run_beamline(wf, file_with_thickness_mesh=file_with_thickness_mesh, distance=3.590000)
        # wf = run_beamline(wf, file_with_thickness_mesh=file_with_thickness_mesh, distance=4.0916)


    import matplotlib.pylab as plt
    from srxraylib.util.h5_simple_writer import H5SimpleWriter


    distance0=3.591600
    delta = 1.0
    npoints = 5 # 64

    distance = numpy.linspace(distance0-0.5*delta, distance0+0.5*delta, npoints)

    do_intermediate_plot = 0
    do_plot = 0

    nsamples = 3 # 5000

    dir = "/scisoft/users/srio/ML_TRAIN2/5000_2Dv1/"  # where the profile files sit
    dir_out = "/scisoft/users/srio/ML_TRAIN2/RESULTS_2D/" # where the results are going to be written
    root = "tmp_ml"


    #
    # calculate
    #
    if True:
        src = run_source()

        for nn in range(nsamples):
            if numpy.mod(nn,50) == 0: print("Calculating sample %d of %d..." % (nn+1,nsamples))
            for i in range(npoints):
                file_with_thickness_mesh = "%s%s%06d.h5" % (dir, root, nn)
                output_wavefront = run_beamline(src, file_with_thickness_mesh=file_with_thickness_mesh, distance=distance[i])

                output_wavefront_intensity = output_wavefront.get_intensity()
                if do_intermediate_plot:
                    # plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),
                    #  title='LAST OPTICAL ELEMENT d=%s' % (distance[i]))
                    plot_image(output_wavefront_intensity, output_wavefront.get_coordinate_x(),
                               output_wavefront.get_coordinate_y(), aspect='auto', title='LAST OPTICAL ELEMENT d=%s' % (distance[i]))

                if i == 0:
                    s = output_wavefront_intensity.shape
                    Z = numpy.zeros((npoints, s[0], s[1]))
                    x = output_wavefront.get_coordinate_x()
                    y = output_wavefront.get_coordinate_y()
                Z[i, :, :] = output_wavefront_intensity

            if nn == 0:
                s = output_wavefront_intensity.shape
                ZZ = numpy.zeros(((nsamples, npoints, s[0], s[1])))

            ZZ[nn,:,:,:] = Z

    print("Done (calculation).")

    #
    # write file with targets
    #

    if True: # Zernike coeffs
        filename = "%s%s_targets_z.txt" % (dir_out, root)
        f = open(filename,'w')
        for i in range(nsamples):
            file_with_info = "%s%s%06d.txt" % (dir, root, i)
            tmp = numpy.loadtxt(file_with_info, skiprows=2)
            f.write("%6d   " % i)
            for j in range(tmp.shape[0]):
                f.write("%10.8g   " % (tmp[j,1] * 1e6)) #   in microns!!!!
            f.write("\n")
        print("File %s written to disk." % filename)

    #
    # write h5 (block data) includes interpolation and base removal
    #
    if True:
        # abscissa_new = numpy.linspace(-125e-6,125e-6,256)
        #
        # ZZblock = numpy.zeros((nsamples, abscissa_new.size, npoints))
        # for i in range(nsamples):
        #     for j in range(npoints):
        #         y_orig = ZZ[i,j,:]
        #         y_int = numpy.interp(abscissa_new, x, y_orig)
        #         ZZblock[i, :, j] = y_int - y_int.min()  # TODO: remove?


        h5_file = "%s%s_block.h5" % (dir_out, root)
        h5w = H5SimpleWriter.initialize_file(h5_file, creator="h5_basic_writer.py")

        h5w.create_entry("allsamples",nx_default="intensity")

        h5w.add_deepstack([numpy.arange(nsamples), distance, x*1e6, y*1e6], ZZ,
                          stack_name="intensity", entry_name="allsamples",
                          # list_of_axes_labels=["sample", "distance", "X", "Y"],
                          list_of_axes_titles=["sample", "distance [m]", "X [um]", "Y [um]"])

        print("File %s written to disk." % h5_file)