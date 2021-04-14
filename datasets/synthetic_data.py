#!/usr/bin/python
# coding: utf-8
###################################################################################
# beam caustics
# Authors/Contributors: Rafael Celestre
# Rafael.Celestre@esrf.eu
# creation: 01/04/2021
# last update: 14/04/2021 (v0.1)
###################################################################################


import sys

import argparse
import datetime
import logging.handlers
import numpy as np
import os
import time

import h5py

try:
    from oasys_srw.srwlib import *
    from oasys_srw.uti_plot import *
    print('using oasys-srw')
except:
    from srwlib import *
    from uti_plot import *
    print('using SRW')

import barc4ro.barc4ro as b4ro
import barc4ro.wavefront_fitting as b4wf

from skimage.restoration import unwrap_phase


def get_radii(_wfr, stvt_x=50, stvt_y=50, silent=True):
    k = 2 * np.pi / srwl_uti_ph_en_conv(_wfr.mesh.eStart, _in_u='eV', _out_u='m')
    arP1 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)
    srwl.CalcIntFromElecField(arP1, _wfr, 0, 4, 3, _wfr.mesh.eStart, 0, 0)

    wp_phase = np.reshape(arP1, (_wfr.mesh.ny, _wfr.mesh.nx))
    wp_phase_x = wp_phase[int(_wfr.mesh.ny / 2), int(_wfr.mesh.nx / 2) - stvt_x:int(_wfr.mesh.nx / 2) + stvt_x]
    wp_phase_y = wp_phase[int(_wfr.mesh.ny / 2) - stvt_y:int(_wfr.mesh.ny / 2) + stvt_y, int(_wfr.mesh.nx / 2)]

    uwp_phase_x = unwrap_phase(wp_phase_x)
    uwp_phase_y = unwrap_phase(wp_phase_y)

    dx = (_wfr.mesh.xFin - _wfr.mesh.xStart) / _wfr.mesh.nx
    dy = (_wfr.mesh.yFin - _wfr.mesh.yStart) / _wfr.mesh.ny

    nx = wp_phase_x.shape[0]
    ny = wp_phase_y.shape[0]

    xStart = - (dx * (nx - 1)) / 2.0
    xFin = xStart + dx * (nx - 1)

    yStart = - (dy * (ny - 1)) / 2.0
    yFin = yStart + dy * (ny - 1)

    x = np.linspace(xStart, xFin, nx)
    y = np.linspace(yStart, yFin, ny)

    px = np.polynomial.polynomial.polyfit(x, uwp_phase_x, 5)
    Rx = k / (2 * px[2])

    py = np.polynomial.polynomial.polyfit(y, uwp_phase_y, 5)
    Ry = k / (2 * py[2])

    if silent is False:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].set_title("wrapped phase")
        im = axs[0, 0].imshow(wp_phase, extent=[_wfr.mesh.xStart * 1e6, _wfr.mesh.xFin * 1e6, _wfr.mesh.yStart * 1e6,
                                                _wfr.mesh.yFin * 1e6], cmap=plt.cm.binary_r)
        plt.colorbar(im, ax=axs[0, 0])

        axs[0, 1].set_title("unwrapped phase")
        im = axs[0, 1].imshow(unwrap_phase(wp_phase), extent=[_wfr.mesh.xStart * 1e6, _wfr.mesh.xFin * 1e6, _wfr.mesh.yStart * 1e6,
                                                 _wfr.mesh.yFin * 1e6], cmap=plt.cm.jet)
        plt.colorbar(im, ax=axs[0, 1])

        axs[1, 0].set_title("wrapped phase - fit")
        im = axs[1, 0].plot(x * 1e6, wp_phase_x, label='h')
        im = axs[1, 0].plot(y * 1e6, wp_phase_y, label='v')
        axs[1, 0].legend(loc=1)

        axs[1, 1].set_title("unwrapped phase")
        im = axs[1, 1].plot(x * 1e6, uwp_phase_x, label='h')
        im = axs[1, 1].plot(y * 1e6, uwp_phase_y, label='v')
        axs[1, 1].legend(loc=1)

        # Reconstructed phase
        ph_x = px[0] + px[1] * x + px[2] * x ** 2
        ph_y = py[0] + py[1] * x + py[2] * x ** 2

        axs[2, 0].set_title("reconstructed phase")
        im = axs[2, 0].plot(x * 1e6, ph_x, label='h')
        im = axs[2, 0].plot(y * 1e6, ph_y, label='v')
        axs[2, 0].legend(loc=1)

        axs[2, 1].set_title("residues")
        im = axs[2, 1].plot(x * 1e6, uwp_phase_x - ph_x, label='h')
        im = axs[2, 1].plot(y * 1e6, uwp_phase_y - ph_y, label='v')
        axs[2, 1].legend(loc=1)

        fig.tight_layout()
        plt.show()

    return Rx, Ry


def get_mtrl_stats(thickness, _mask=None):

    aux_array = deepcopy(thickness)
    if _mask is None:
        pass
    else:
        aux_array[np.logical_not(_mask)] = np.nan

    x2 = np.multiply(aux_array, aux_array)
    SumX2 = np.nansum(x2)
    RMS = np.sqrt((SumX2 / np.count_nonzero(~np.isnan(x2))))
    PV = np.nanmax(aux_array) - np.nanmin(aux_array)

    return RMS, PV


def get_mask(_mesh, _Dx, _Dy, _shape='r', _ap_or_ob='a'):

    X, Y = np.meshgrid(np.linspace(_mesh.xStart, _mesh.xFin, _mesh.nx), np.linspace(_mesh.yStart, _mesh.yFin, _mesh.ny))

    mask = np.zeros((_mesh.ny, _mesh.nx), dtype=bool)

    if _shape == 'r':
        mask[X < -0.5 * _Dx] = True
        mask[X > 0.5 * _Dx] = True
        mask[Y < -0.5 * _Dy] = True
        mask[Y > 0.5 * _Dy] = True

    if _shape == 'c':
        R = (X ** 2 + Y ** 2) ** 0.5
        mask[R > 0.5 * _Dx] = True

    if _ap_or_ob == 'o':
        mask = np.logical_not(mask)

    return mask


def save_wfr_2_hdf5(_wfr,_filename,_subgroupname="wfr",_complex_amplitude=True,_intensity=False,_amplitude=False,_phase=False,_overwrite=True):
    """
    Auxiliary function to write wavefront data into a hdf5 generic file.
    When using the append mode to write h5 files, overwriting does not work and makes the code crash. To avoid this
    issue, try/except is used. If by any chance a file should be overwritten, it is firstly deleted and re-written.
    :param _wfr: input / output resulting Wavefront structure (instance of SRWLWfr);
    :param _filename: path to file for saving the wavefront
    :param _subgroupname: container mechanism by which HDF5 files are organised
    :param _complex_amplitude: complex electric field for sigma and pi polarisation
    :param _intensity: Single-Electron" Intensity - total polarisation (instance of srwl.CalcIntFromElecField)
    :param _amplitude: S0 Stokes parameter. I = P_0 + P_90 = <Ex^2 + Ey^2>.
    :param _phase: "Single-Electron" Radiation Phase - total polarisation (instance of srwl.CalcIntFromElecField)
    :param _overwrite: flag that should always be set to True to avoid infinity loop on the recursive part of the function.
    """

    def _dump_arr_2_hdf5(_arr, _calculation, _filepath, _subgroupname):
        """
        Auxiliary routine to save_radiograph_2_hdf5()
        :param _arr: (usually 2D) array to be saved on the hdf5 file inside the _subgroupname
        :param _calculation
        :param _filepath: path to file for saving the wavefront
        :param _subgroupname: container mechanism by which HDF5 files are organised
        """
        sys.stdout.flush()
        f = h5py.File(_filepath, 'a')
        try:
            f1 = f.create_group(_subgroupname)
        except:
            f1 = f[_subgroupname]
        # f1[_calculation] = _arr
        fdata = f1.create_dataset(_calculation, data=_arr)
        f.close()

    def _SRW_2_Numpy(_srw_array, _nx, _ny, _ne):
        """
        Converts an SRW array to a numpy.array.
        :param _srw_array: SRW array
        :param _nx: numbers of points vs horizontal positions
        :param _ny: numbers of points vs vertical positions
        :param _ne: numbers of points vs photon energy
        :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
        """
        re = np.array(_srw_array[::2], dtype=np.float)
        im = np.array(_srw_array[1::2], dtype=np.float)

        e = re + 1j * im
        e = e.reshape((_ny, _nx, _ne, 1))
        e = e.swapaxes(0, 2)

        return e.copy()

    try:
        if not os.path.isfile(_filename):  # if file doesn't exist, create it.
            sys.stdout.flush()
            f = h5py.File(_filename, 'w')
            # points to the default data to be plotted
            f.attrs['default']          = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name']        = _filename
            f.attrs['file_time']        = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            f.attrs['creator']          = 'save_wfr_2_hdf5'
            f.attrs['code']             = 'SRW'
            f.attrs['HDF5_Version']     = h5py.version.hdf5_version
            f.attrs['h5py_version']     = h5py.version.version
            f.close()

        if _amplitude:
            ar1 = array('f', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(ar1, _wfr, 6, 0, 3, _wfr.mesh.eStart, 0, 0)
            arxx = np.array(ar1)
            arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T
            arxx = np.sqrt(arxx)
            _dump_arr_2_hdf5(arxx,"amplitude/wfr_amplitude", _filename, _subgroupname)

        if _phase:
            ar1 = array('d', [0] * _wfr.mesh.nx * _wfr.mesh.ny)  # "flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(ar1, _wfr, 0, 4, 3, _wfr.mesh.eStart, 0, 0)
            arxx = np.array(ar1)
            arxx = arxx.reshape((_wfr.mesh.ny, _wfr.mesh.nx))#.T

            _dump_arr_2_hdf5(arxx, "phase/wfr_phase", _filename, _subgroupname)

        if (_complex_amplitude) or (_intensity):
            x_polarization = _SRW_2_Numpy(_wfr.arEx, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # sigma
            y_polarization = _SRW_2_Numpy(_wfr.arEy, _wfr.mesh.nx, _wfr.mesh.ny, _wfr.mesh.ne)   # pi

            e_field = np.concatenate((x_polarization, y_polarization), 3)

            if _complex_amplitude:
                _dump_arr_2_hdf5(e_field[0,:,:,0].T, "wfr_complex_amplitude_s", _filename, _subgroupname)
                _dump_arr_2_hdf5(e_field[0,:,:,1].T, "wfr_complex_amplitude_p", _filename, _subgroupname)

            if _intensity:
                intens = np.abs(e_field[0, :, :, 0]) ** 2 + np.abs(e_field[0, :, :, 1]) ** 2
                _dump_arr_2_hdf5(intens.T,"intensity/wfr_intensity",_filename, _subgroupname)


        f = h5py.File(_filename, 'a')
        f1 = f[_subgroupname]

        # points to the default data to be plotted
        f1.attrs['NX_class'] = 'NXentry'
        f1.attrs['default']  = 'intensity'

        #f1["wfr_method"] = "SRW"
        f1["wfr_photon_energy"] = _wfr.mesh.eStart
        f1["wfr_Rx_dRx"] =  np.array([_wfr.Rx,_wfr.dRx])
        f1["wfr_Ry_dRy"] =  np.array([_wfr.Ry,_wfr.dRy])
        f1["wfr_mesh_X"] =  np.array([_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx])
        f1["wfr_mesh_Y"] =  np.array([_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny])

        # Add NX plot attribites for automatic plot with silx view
        myflags = [_intensity,_amplitude,_phase]
        mylabels = ['intensity','amplitude','phase']
        for i,label in enumerate(mylabels):
            if myflags[i]:

                f2 = f1[mylabels[i]]
                f2.attrs['NX_class'] = 'NXdata'
                f2.attrs['signal'] = 'wfr_%s'%(mylabels[i])
                f2.attrs['axes'] = [b'axis_y', b'axis_x']

                # ds = nxdata.create_dataset('image_data', data=data)
                f3 = f2["wfr_%s"%(mylabels[i])]
                f3.attrs['interpretation'] = 'image'

                # X axis data
                ds = f2.create_dataset('axis_y', data=1e6*np.linspace(_wfr.mesh.yStart,_wfr.mesh.yFin,_wfr.mesh.ny))
                # f1['axis1_name'] = np.arange(_wfr.mesh.ny)
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'Y Pixel Size (microns)'    # suggested X axis plot label
                #
                # Y axis data
                ds = f2.create_dataset('axis_x', data=1e6*np.linspace(_wfr.mesh.xStart,_wfr.mesh.xFin,_wfr.mesh.nx))
                ds.attrs['units'] = 'microns'
                ds.attrs['long_name'] = 'X Pixel Size (microns)'    # suggested Y axis plot label
        f.close()

        FileName = _filename.split("/")
        print(">>>> save_wfr_2_hdf5: file witten/updated %s" %FileName[-1])

    except:
        if _overwrite is not True:
            print(">>>> Bad input argument")
            return
        os.remove(_filename)
        FileName = _filename.split("/")
        print(">>>> save_wfr_2_hdf5: file deleted %s"%FileName[-1])
        save_wfr_2_hdf5(_wfr,_filename,_subgroupname,_complex_amplitude,_intensity,_amplitude,_phase,_overwrite = False)


if __name__ == '__main__':
    startTime = time.time()
    p = argparse.ArgumentParser(description='Beam caustics')
    p.add_argument('-s', '--save', dest='save', metavar='BOOL', default=False, help='enables saving .dat file')
    p.add_argument('-p', '--plots', dest='plots', metavar='BOOL', default=False, help='enables graphical display of the result')
    p.add_argument('-c', '--caustics', dest='caustics', metavar='BOOL', default=False, help='calculates the beam caustics')
    p.add_argument('-e', '--beamE', dest='beamE', metavar='NUMBER', default=0, type=float, help='beam energy in keV')
    p.add_argument('-z', '--z', dest='z', metavar='NUMBER', default=0, type=float, help='position [m] of first optical element')
    p.add_argument('-i', '--illum', dest='illum', metavar='NUMBER', default=1, type=int, help='0 for plane phase; 1 for parabolic phase')
    p.add_argument('-nd', '--delta', dest='delta', metavar='NUMBER', default=1e-23, type=float, help='n=1-delta + i beta')
    p.add_argument('-nb', '--beta', dest='beta', metavar='NUMBER', default=1e-23, type=float, help='n=1-delta + i beta')
    p.add_argument('-l', '--lens', dest='lens', metavar='NUMBER', default=0, type=int, help='0 for ideal lens; 1 for aberrated lens with Zernike polynomials; 2 for metrology data')
    p.add_argument('-sd', '--seed', dest='seed', metavar='NUMBER', default=69, type=int, help='random seed for generation of the surface errors')
    p.add_argument('-cr', '--cst_range', dest='cst_range', metavar='NUMBER', default=1, type=float, help='caustic range around zero [m]')
    p.add_argument('-cp', '--cst_points', dest='cst_points', metavar='NUMBER', default=33, type=float, help='number of points for caustics calcuation')
    p.add_argument('-d', '--defocus', dest='defocus', metavar='NUMBER', default=0, type=float, help='defocus in [m], (-) means before focus, (+) means after focus')
    p.add_argument('-prfx', '--prfx', dest='prfx', metavar='STRING',  help='prefix for saving files')
    p.add_argument('-dir', '--dir', dest='dir', metavar='STRING', default='./', help='directory for saving files')
    p.add_argument('-m', '--mtrl', dest='mtrl', metavar='STRING', default='./', help='complete metrology file address')

    args = p.parse_args()

    #############################################################################
    #############################################################################

    save = eval(args.save)
    plots = eval(args.plots)
    caustics = eval(args.caustics)

    beamE = args.beamE
    cst_range = (-args.cst_range/2, args.cst_range/2)
    cst_pts = int(args.cst_points)
    defocus = args.defocus

    strDataFolderName = args.dir
    prfx = args.prfx

    energy = str(beamE)
    energy = energy.replace('.', 'p')

    position = str(defocus * 1e3)
    position = position.replace('.', 'p')

    string_block = 'int_' + energy + 'keV_d' + position + 'mm'
    strIntPropOutFileName = string_block + '_' + prfx

    #############################################################################
    #############################################################################
    # Logging all logging.infos

    # Get time stamp
    start0 = time.time()
    dt = datetime.datetime.fromtimestamp(start0).strftime('%Y-%m-%d_%H:%M:%S')

    # Initializing logging
    log = logging.getLogger('')
    log.setLevel(logging.INFO)
    format = logging.Formatter('%(levelname)s: %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(strDataFolderName + '/' + dt + '_' + string_block + '.log',
                                              maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    log.addHandler(fh)

    #############################################################################
    #############################################################################
    # Photon source

    wfr_resolution = (256, 256)  # nx, ny
    screen_range = (-0.5E-3, 0.5E-3, -0.5E-3, 0.5E-3)  # x_Start, x_Fin, y_Start, y_Fin
    sampling_factor = 0 # sampling factor for adjusting nx, ny (effective if > 0)
    wavelength = srwl_uti_ph_en_conv(beamE, _in_u='keV', _out_u='m')
    k = 2*np.pi/wavelength
    z = args.z

    # ******************************** Undulator parameters (CPMU18)
    numPer = 111			# Number of ID Periods
    undPer = 0.018		# Period Length [m]
    phB = 0	        	# Initial Phase of the Horizontal field component
    sB = 1		        # Symmetry of the Horizontal field component vs Longitudinal position
    xcID = 0 			# Transverse Coordinates of Undulator Center [m]
    ycID = 0
    zcID = 0
    n = 1
    # ******************************** Storage ring parameters
    eBeam = SRWLPartBeam()
    eBeam.Iavg = 0.2             # average Current [A]
    eBeam.partStatMom1.x = 0
    eBeam.partStatMom1.y = 0
    eBeam.partStatMom1.z = -0.5*undPer*(numPer + 4)    # initial Longitudinal Coordinate (set before the ID)
    eBeam.partStatMom1.xp = 0  					       # initial Relative Transverse Velocities
    eBeam.partStatMom1.yp = 0

    # e- beam paramters (RMS) EBS
    sigEperE = 9.3E-4  # relative RMS energy spread
    sigX = 30.3E-06  # horizontal RMS size of e-beam [m]
    sigXp = 4.4E-06  # horizontal RMS angular divergence [rad]
    sigY = 3.6E-06  # vertical RMS size of e-beam [m]
    sigYp = 1.46E-06  # vertical RMS angular divergence [rad]
    eBeam.partStatMom1.gamma = 6.00 / 0.51099890221e-03  # Relative Energy

    n = 1
    if (2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1)) <= 0:
        n=3
        if (2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1)) <= 0:
            n = 5
            if (2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1)) <= 0:
                n = 7

    K = np.sqrt(2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1))
    B = K / (undPer * 93.3728962)  # Peak Horizontal field [T] (undulator)

    # 2nd order stat. moments
    eBeam.arStatMom2[0] = sigX*sigX			 # <(x-<x>)^2>
    eBeam.arStatMom2[1] = 0					 # <(x-<x>)(x'-<x'>)>
    eBeam.arStatMom2[2] = sigXp*sigXp		 # <(x'-<x'>)^2>
    eBeam.arStatMom2[3] = sigY*sigY		     # <(y-<y>)^2>
    eBeam.arStatMom2[4] = 0					 # <(y-<y>)(y'-<y'>)>
    eBeam.arStatMom2[5] = sigYp*sigYp		 # <(y'-<y'>)^2>
    eBeam.arStatMom2[10] = sigEperE*sigEperE # <(E-<E>)^2>/<E>^2

    # Electron trajectory
    eTraj = 0

    # Precision parameters
    arPrecSR = [0]*7
    arPrecSR[0] = 1		# SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    arPrecSR[1] = 0.01	# relative precision
    arPrecSR[2] = 0		# longitudinal position to start integration (effective if < zEndInteg)
    arPrecSR[3] = 0		# longitudinal position to finish integration (effective if > zStartInteg)
    arPrecSR[4] = 20000	# Number of points for trajectory calculation
    arPrecSR[5] = 1		# Use "terminating terms"  or not (1 or 0 respectively)
    arPrecSR[6] = sampling_factor # sampling factor for adjusting nx, ny (effective if > 0)
    sampFactNxNyForProp = arPrecSR[6] # sampling factor for adjusting nx, ny (effective if > 0)

    und = SRWLMagFldU([SRWLMagFldH(n, 'v', B, phB, sB, 1)], undPer, numPer)

    magFldCnt = SRWLMagFldC([und], array('d', [xcID]), array('d', [ycID]), array('d', [zcID]))

    # Monochromatic wavefront
    wfr = SRWLWfr()
    wfr.allocate(1, wfr_resolution[0], wfr_resolution[1])  # Photon Energy, Horizontal and Vertical Positions
    wfr.mesh.zStart = z
    wfr.mesh.eStart = beamE * 1E3
    wfr.mesh.xStart = screen_range[0]
    wfr.mesh.xFin = screen_range[1]
    wfr.mesh.yStart = screen_range[2]
    wfr.mesh.yFin = screen_range[3]
    wfr.partBeam = eBeam
    meshPartCoh = deepcopy(wfr.mesh)

    #############################################################################
    #############################################################################
    # Wavefront generation

    # ******************************** Calculating Initial Wavefront and extracting Intensity:
    logging.info('- Performing Initial Electric Field calculation ... ')
    srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecSR)

    Rx, Ry = get_radii(wfr, stvt_x=50, stvt_y=50, silent=True)

    logging.info('Initial wavefront:')
    logging.info('Nx = %d, Ny = %d' % (wfr.mesh.nx, wfr.mesh.ny))
    logging.info('dx = %.4f um, dy = %.4f um' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E6 / wfr.mesh.nx,
                                                 (wfr.mesh.yFin - wfr.mesh.yStart) * 1E6 / wfr.mesh.ny))
    logging.info('range x = %.4f mm, range y = %.4f mm' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E3,
                                                           (wfr.mesh.yFin - wfr.mesh.yStart) * 1E3))
    logging.info('- Wavefront curvature:')
    logging.info('SRW native calculation: Rx = %.6f, Ry = %.6f' % (wfr.Rx, wfr.Ry))
    logging.info('Phase fit: Rx = %.6f, Ry = %.6f' % (Rx, Ry))
    logging.info('dRx = %.3f %%, dRy = %.3f %%' % ((Rx - wfr.Rx) * 100 / Rx, (Ry - wfr.Ry) * 100 / Ry))

    #############################################################################
    #############################################################################
    # Beamline assembly
    logging.info('Setting up beamline') if (srwl_uti_proc_is_master()) else 0

    if args.illum == 0: # plane wave illumination
        oeThinLens = SRWLOptL(_Fx=Rx, _Fy=Ry)

    # ============= Single lens parameters =================================#
    '''Single lens definition'''

    R = 50 * 1E-6          # CRL radius at the parabola appex - TODO: consider changing it to a parameter in the future
    CRLAph = 444 * 1E-6    # CRL aperture
    CRLApv = 444 * 1E-6
    wt = 20. * 1E-6         # CRL wall thickness [um]
    shp = 1                 # 1- parabolic, 2- circular (spherical)
    foc_plane = 3           # plane of focusing: 1- horizontal, 2- vertical, 3- both

    ContainerThickness = 2e-3
    f_CRL = R/(2*args.delta)
    oeApCRL = SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=CRLAph, _Dy=CRLApv)
    oeCRL = srwl_opt_setup_CRL(foc_plane, args.delta, wavelength/(4*np.pi*args.beta), shp, CRLAph, CRLApv, R,  1, wt,
                               _xc=0, _yc=0, _nx=2001, _ny=2001)

    # ============= Generation of figure errors ============================#

    if args.lens == 1:
        rg = np.random.default_rng(args.seed)

        # Z1 (Zcoeffs[0]) to Z4 (Zcoeffs[3]) are set to zero
        Zcoeffs = np.zeros([37])

        # Z5 (Zcoeffs[4]) to Z10 (Zcoeffs[9]) are more important and have a higher weight
        for i in range(4,10):
            Zcoeffs[i] = rg.normal(loc=0, scale=0.5)
        # Spherical aberration 1st order (Z11)
        Zcoeffs[10] = rg.uniform(-2.3, 2.3)

        # Z12 (Zcoeffs[11]) to Z21 (Zcoeffs[20]) and Z23 (Zcoeffs[22]) to Z36 (Zcoeffs[35]) are very low
        for i in range(11,37):
            Zcoeffs[i] = Zcoeffs[i] = rg.normal(loc=0, scale=0.05)

        # Spherical aberration 2nd order (Z22)
        Zcoeffs[21] = rg.uniform(-1., 1.)

        # Spherical aberration 2nd order (Z22)
        Zcoeffs[36] = rg.uniform(-0.5, 0.5)

        oe_crl_error = b4ro.srwl_opt_setup_CRL_errors(Zcoeffs*1e-6, 'c', args.delta, wavelength/(4*np.pi*args.beta), CRLAph,
                                                      CRLApv, _xc=0, _yc=0, _nx=2001, _ny=2001)

        x, y, thcknss_err = b4ro.polynomial_surface_2D(_z_coeffs=Zcoeffs*1e-6, _pol='c', _apert_h=CRLAph, _apert_v=CRLApv,
                                              _nx=2001, _ny=2001)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        mesh = SRWLRadMesh()
        mesh.eStart = -1
        mesh.eFin = -1
        mesh.ne = 1
        mesh.nx = thcknss_err.shape[1]
        mesh.xStart = - (dx * (mesh.nx - 1)) / 2.0
        mesh.xFin = mesh.xStart + dx * (mesh.nx - 1)
        mesh.ny = thcknss_err.shape[0]
        mesh.yStart = - (dy * (mesh.ny - 1)) / 2.0
        mesh.yFin = mesh.yStart + dy * (mesh.ny - 1)

        mask = get_mask(mesh, _Dx=mesh.xFin-mesh.xStart, _Dy=mesh.yFin-mesh.yStart, _shape='c', _ap_or_ob='a')
        err_rms, err_pv = get_mtrl_stats(thcknss_err*1e6, _mask=np.logical_not(mask))
        logging.info('Fig.err. rms: ' + str(err_rms) + ' [um] and PV:' + str(err_pv) + ' [um]')
        logging.info('Z coefficients (um): \n' + str(Zcoeffs))

    elif args.lens == 2:
        logging.info('>>> metrology file: ' + args.mtrl.split('/')[-1])

        amp_coef = 1
        thcknss_err, mesh = srwl_uti_read_intens_ascii(args.mtrl)
        oe_crl_error = b4ro.srwl_opt_setup_CRL_metrology(thcknss_err, mesh, args.delta, wavelength/(4*np.pi*args.beta),
                                                         _amp_coef=-amp_coef)
        Zcoeffs, fit, residues = b4wf.fit_zernike_circ(np.reshape(thcknss_err, (mesh.ny, mesh.nx)),
                                                       nmodes=37, startmode=1, rec_zern=False)

        mask = get_mask(mesh, _Dx=mesh.xFin-mesh.xStart, _Dy=mesh.yFin-mesh.yStart, _shape='c', _ap_or_ob='a')
        err_rms, err_pv = get_mtrl_stats(np.reshape(thcknss_err, (mesh.ny, mesh.nx))*1e6, _mask=np.logical_not(mask))
        logging.info('Fig.err. rms: ' + str(err_rms) + ' [um] and PV:' + str(err_pv) + ' [um]')
        logging.info('Z coefficients (um): \n' + str(Zcoeffs*1e6))

    if caustics:
        if args.illum == 0:
            Drift = SRWLOptD(f_CRL + cst_range[0] + defocus)
            logging.info('Caustics begin at: %.6f' % (f_CRL + cst_range[0] + defocus))
        else:
            q = 1/(1/f_CRL - 1/z)
            Drift = SRWLOptD(q + cst_range[0] + defocus)
            logging.info('Caustics begin at: %.6f' % (q + cst_range[0] + defocus))

    else:
        if args.illum == 0:
            Drift = SRWLOptD(f_CRL + defocus)
            logging.info('Image at: %.6f' % (f_CRL + cst_range[0] + defocus))
        else:
            q = 1/(1/f_CRL - 1/z)
            Drift = SRWLOptD(q + defocus)
            logging.info('Image at: %.6f' % (q + defocus))

    # ============= Wavefront Propagation Parameters =======================#

    #               [ 0] [1] [2]  [3]  [4]  [5]  [6]  [7]   [8]  [9] [10] [11]
    ppApCRL =       [0,   0,  1,   0,   0,   1,   5,   1,    5,   0,   0,   0]
    ppThinLens =    [0,   0,  1,   0,   0,   1,   1,   1,    1,   0,   0,   0]
    ppCRL =         [0,   0,  1,   1,   0,   1,   1,   1,    1,   0,   0,   0]
    ppDrift =       [0,   0,  1,   1,   0,   1,   1,   1,    1,   0,   0,   0]
    ppDrift_cstc =  [0,   0,  1,   0,   0,   1,   1,   1,    1,   0,   0,   0]

    '''
    [ 3]: Type of Free-Space Propagator:
           0- Standard Fresnel
           1- Fresnel with analytical treatment of the quadratic (leading) phase terms
           2- Similar to 1, yet with different processing near a waist
           3- For propagation from a waist over a ~large distance
           4- For propagation over some distance to a waist
    [ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
    [ 6]: Horizontal Resolution modification factor at Resizing
    [ 7]: Vertical Range modification factor at Resizing
    [ 8]: Vertical Resolution modification factor at Resizing
    '''

    srw_oe_array = []
    srw_pp_array = []

    srw_oe_array.append(oeApCRL)
    srw_pp_array.append(ppApCRL)

    if args.illum == 0:
        srw_oe_array.append(oeThinLens)
        srw_pp_array.append(ppThinLens)

    srw_oe_array.append(oeCRL)
    srw_pp_array.append(ppCRL)

    if args.lens == 1 or args.lens == 2:
        srw_oe_array.append(oe_crl_error)
        srw_pp_array.append(ppCRL)

    srw_oe_array.append(Drift)
    srw_pp_array.append(ppDrift)

    optBL = SRWLOptC(srw_oe_array, srw_pp_array)

    #############################################################################
    #############################################################################
    # Electric field propagation
    logging.info('- Simulating Electric Field Wavefront Propagation ... ')
    srwl.PropagElecField(wfr, optBL)

    Rx, Ry = get_radii(wfr, stvt_x=100, stvt_y=100, silent=True)

    logging.info('Propagated wavefront:')
    logging.info('Nx = %d, Ny = %d' % (wfr.mesh.nx, wfr.mesh.ny))
    logging.info('dx = %.4f um, dy = %.4f um' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E6 / wfr.mesh.nx,
                                                 (wfr.mesh.yFin - wfr.mesh.yStart) * 1E6 / wfr.mesh.ny))
    logging.info('range x = %.4f mm, range y = %.4f mm' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E3,
                                                           (wfr.mesh.yFin - wfr.mesh.yStart) * 1E3))
    logging.info('- Wavefront curvature:')
    logging.info('SRW native calculation: Rx = %.6f, Ry = %.6f' % (wfr.Rx, wfr.Ry))
    logging.info('Phase fit: Rx = %.6f, Ry = %.6f' % (Rx, Ry))
    logging.info('dRx = %.3f %%, dRy = %.3f %%' % ((Rx - wfr.Rx) * 100 / Rx, (Ry - wfr.Ry) * 100 / Ry))

    if save is True or plots is True:
        arI1 = array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
        arP1 = array('d', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" array to take 2D phase data (note it should be 'd')
        srwl.CalcIntFromElecField(arP1, wfr, 0, 4, 3, wfr.mesh.eStart, 0, 0)

    if save is True and caustics is False:
        # srwl_uti_save_intens_ascii(arI1, wfr.mesh, os.path.join(strDataFolderName,  strIntPropOutFileName), 0)
        save_wfr_2_hdf5(wfr, _filename=os.path.join(strDataFolderName,  strIntPropOutFileName),
                        _subgroupname="wfr", _complex_amplitude=False, _intensity=True, _amplitude=False,
                        _phase=False, _overwrite=True)
    logging.info('>> single electron calculations: done')

    if caustics:
        logging.info('- Simulating caustic scan along the optical axis ... ')
        opt_axis = np.linspace(0, args.cst_range, cst_pts)
        cst_drift = SRWLOptD(opt_axis[1]-opt_axis[0])
        k = 0
        for pts in range(cst_pts):
            logging.info('>>>> plane %d out of %d' % (k+1, cst_pts))
            if k == 0:
                pass
            else:
                optBLp = SRWLOptC([cst_drift], [ppDrift])
                srwl.PropagElecField(wfr, optBLp)

            arI1 = array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" 2D array to take intensity data
            srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
            # filename = strDataFolderName + '/' + prfx.replace('XXX', '%.3d')%k
            # srwl_uti_save_intens_ascii(arI1, wfr.mesh, filename, 0)
            # save_wfr_2_hdf5(wfr, _filename=filename, _subgroupname="wfr", _complex_amplitude=False, _intensity=True,
            #                 _amplitude=False, _phase=False, _overwrite=True)

            scan_name = strDataFolderName + '/' + strDataFolderName.split('/')[-1] + '.h5'
            filename = prfx.replace('XXX', '%.3d')%k
            save_wfr_2_hdf5(wfr, _filename=scan_name, _subgroupname=filename, _complex_amplitude=False, _intensity=True,
                            _amplitude=False, _phase=False, _overwrite=True)
            k+=1

    deltaT = time.time() - startTime
    hours, minutes = divmod(deltaT, 3600)
    minutes, seconds = divmod(minutes, 60)
    logging.info(">>>> Elapsed time: " + str(int(hours)) + "h " + str(int(minutes)) + "min " + str(seconds) + "s ") if (srwl_uti_proc_is_master()) else 0

    if plots is True and caustics is False:
        # ********************************Electrical field intensity and phase after propagation
        plotMesh1x = [1E6 * wfr.mesh.xStart, 1E6 * wfr.mesh.xFin, wfr.mesh.nx]
        plotMesh1y = [1E6 * wfr.mesh.yStart, 1E6 * wfr.mesh.yFin, wfr.mesh.ny]
        uti_plot2d(arI1, plotMesh1x, plotMesh1y,['Horizontal Position [um]', 'Vertical Position [um]', 'Intensity After Propagation'])
        uti_plot2d(arP1, plotMesh1x, plotMesh1y,['Horizontal Position [um]', 'Vertical Position [um]', 'Phase After Propagation'])
        uti_plot_show()


