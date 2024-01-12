
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

from srxraylib.plot.gol import plot, plot_image
plot_from_oe = 600 # set to a large number to avoid plots


import matplotlib
matplotlib.rcParams.update({'font.size': 14})

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
        magnification_x_forward=100,
        magnification_x_backward=0.01,
        scan_direction='V',
        sigmaxx=5.2915e-06,
        sigmaxpxp=1.88982e-06,
        useGSMapproximation=False,
        e_energy_dispersion_flag=0,
        e_energy_dispersion_sigma_relative=0.001,
        e_energy_dispersion_interval_in_sigma_units=6,
        e_energy_dispersion_points=11)

    # make calculation
    coherent_mode_decomposition_results = coherent_mode_decomposition.calculate()

    mode_index = 0
    output_wavefront = coherent_mode_decomposition.get_eigenvector_wavefront(mode_index)


    if plot_from_oe <= 0: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='SOURCE')
    return output_wavefront

def run_beamline(output_wavefront, error_file_index=0):
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
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=36.000000,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront,    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('magnification_x', 10.0)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_1D')


    #
    #---- plots -----
    #
    if plot_from_oe <= 1: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR 1')


    ##########  OPTICAL ELEMENT NUMBER 2 ##########



    input_wavefront = output_wavefront.duplicate()
    from syned.beamline.shape import Rectangle
    boundary_shape=Rectangle(-0.5, 0.5, -0.5, 0.5)
    from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D
    optical_element = WOSlit1D(boundary_shape=boundary_shape)

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)


    #
    #---- plots -----
    #
    if plot_from_oe <= 2: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR 2')


    ##########  OPTICAL ELEMENT NUMBER 3 ##########



    input_wavefront = output_wavefront.duplicate()
    from syned.beamline.shape import Rectangle
    boundary_shape=Rectangle(-0.5, 0.5, -0.5, 0.5)
    from wofryimpl.beamline.optical_elements.absorbers.slit import WOSlit1D
    optical_element = WOSlit1D(boundary_shape=boundary_shape)

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
    propagation_parameters.set_additional_parameters('magnification_x', 1.5)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_1D')


    #
    #---- plots -----
    #
    if plot_from_oe <= 3: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR 3')


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
        error_file='/scisoft/users/srio/MLCRL/V10/ML_TRAIN/tmp_ml%06d.dat' % error_file_index,
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
        verbose=1)

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)


    #
    #---- plots -----
    #
    if plot_from_oe <= 4: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR 4')


    ##########  OPTICAL ELEMENT NUMBER 5 ##########



    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D

    optical_element = WOScreen1D()

    # no drift in this element
    output_wavefront = optical_element.applyOpticalElement(input_wavefront)


    #
    #---- plots -----
    #
    if plot_from_oe <= 5: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR 5')


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
    beamline_element = BeamlineElement(optical_element=optical_element,    coordinates=ElementCoordinates(p=3.591600,    q=0.000000,    angle_radial=numpy.radians(0.000000),    angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront,    propagation_elements = propagation_elements)
    #self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('magnification_x', 0.1)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,    handler_name='FRESNEL_ZOOM_1D')


    #
    #---- plots -----
    #
    if plot_from_oe <= 6: plot(output_wavefront.get_abscissas(),output_wavefront.get_intensity(),title='OPTICAL ELEMENT NR 6')
    return  output_wavefront

if __name__ == "__main__":

    #
    # profiles
    #
    a05 = numpy.loadtxt('/scisoft/users/srio/MLCRL/V10/ML_TRAIN/tmp_ml%06d.dat' % 5)
    a12 = numpy.loadtxt('/scisoft/users/srio/MLCRL/V10/ML_TRAIN/tmp_ml%06d.dat' % 12)
    a31 = numpy.loadtxt('/scisoft/users/srio/MLCRL/V10/ML_TRAIN/tmp_ml%06d.dat' % 31)

    plot(a05[:, 0] * 1e6, a05[:, 1] * 1e6,
         a12[:, 0] * 1e6, a12[:, 1] * 1e6,
         a31[:, 0] * 1e6, a31[:, 1] * 1e6,
         xtitle=r'x [$\mu$m]', ytitle=r'y [$\mu$m]',
         legend=['profile 5','profile 12', 'profile 31'], title='')


    #
    # image
    #
    ws = run_source()
    w05 = run_beamline(ws, error_file_index=5)
    w12 = run_beamline(ws, error_file_index=12)
    w31 = run_beamline(ws, error_file_index=31)

    plot(w05.get_abscissas() * 1e6, w05.get_intensity(),
         w12.get_abscissas() * 1e6, w12.get_intensity(),
         w31.get_abscissas() * 1e6, w31.get_intensity(),
         xtitle=r'x [$\mu$m]', ytitle='intensity',
         legend=['profile 5', 'profile 12', 'profile 31'], title='', ylog=0, xrange=[-50, 50])

    # plot(w05.get_abscissas() * 1e6, w05.get_intensity(),
    #      w12.get_abscissas() * 1e6, w12.get_intensity(),
    #      w31.get_abscissas() * 1e6, w31.get_intensity(),
    #      xtitle=r'x [$\mu$m]', ytitle='intensity',
    #      legend=['profile 5', 'profile 12', 'profile 31'], title='', ylog=1, xrange=[-150, 150])