#!/usr/bin/python
# coding: utf-8
###################################################################################
# beam caustics
# Authors/Contributors: Rafael Celestre
# Rafael.Celestre@esrf.eu
# creation: 01/04/2021
# last update: 01/04/2021 (v0.0)
###################################################################################


import sys

import argparse
import datetime
import logging.handlers
import numpy as np
import os
import time

try:
    from oasys_srw.srwlib import *
    from oasys_srw.uti_plot import *
    print('using oasys-srw')
except:
    from srwlib import *
    from uti_plot import *
    print('using SRW')

import barc4ro.barc4ro as b4RO

if __name__ == '__main__':
    startTime = time.time()
    p = argparse.ArgumentParser(description='Beam caustics')
    p.add_argument('-s', '--save', dest='save', metavar='BOOL', default=False, help='enables saving .dat file')
    p.add_argument('-p', '--plots', dest='plots', metavar='BOOL', default=False, help='enables graphical display of the result')
    p.add_argument('-c', '--caustics', dest='caustics', metavar='BOOL', default=False, help='calculates the beam caustics')
    p.add_argument('-e', '--beamE', dest='beamE', metavar='NUMBER', default=0, type=float, help='beam energy in keV')
    p.add_argument('-i', '--illum', dest='illum', metavar='NUMBER', default=0, type=int, help='0 for plane phase; 1 for parabolic phase')
    p.add_argument('-nd', '--delta', dest='delta', metavar='NUMBER', default=1e-23, type=float, help='n=1- delta + i beta')
    p.add_argument('-nb', '--beta', dest='beta', metavar='NUMBER', default=1e-23, type=float, help='n=1- delta + i beta')
    p.add_argument('-cr', '--cst_range', dest='cst_range', metavar='NUMBER', default=1, type=float, help='caustic range around zero [m]')
    p.add_argument('-cp', '--cst_points', dest='cst_points', metavar='NUMBER', default=33, type=float, help='number of points for caustics calcuation')
    p.add_argument('-d', '--defocus', dest='defocus', metavar='NUMBER', default=0, type=float, help='defocus in [m], (-) means before focus, (+) means after focus')
    p.add_argument('-prfx', '--prfx', dest='prfx', metavar='STRING',  help='prefix for saving files')
    p.add_argument('-dir', '--dir', dest='dir', metavar='STRING', default='./', help='directory for saving files')

    args = p.parse_args()


    args = p.parse_args()

    save = eval(args.save)
    plots = eval(args.plots)
    caustics = eval(args.caustics)

    beamE = args.beamE
    cst_range = (-args.cst_range/2, args.cst_range/2)
    cst_pts = int(args.cst_points)     # caustic number of points
    defocus = args.defocus

    #############################################################################
    #############################################################################



    #!/usr/bin/python
    # # coding: utf-8
    # ###################################################################################
    # # beam caustics
    # # Authors/Contributors: Rafael Celestre
    # # Rafael.Celestre@esrf.eu
    # # creation: 08/09/2020
    # # last update: 08/09/2020(v0.0)
    # ###################################################################################
    # import sys
    #
    # sys.path.insert(0, '../../srw_python')
    # # sys.path.insert(0, '../../barc4RefractiveOptics')
    # sys.path.insert(0, '../barc4')
    #
    # import argparse
    # import datetime
    # import logging.handlers
    # import numpy as np
    # import os
    # import time
    #
    # from srwlib import *
    # from uti_plot import *
    # # import barc4RefractiveOptics.barc4RefractiveOptics as b4ro
    # import barc4optics as b4ro
    # from  f_index_of_refraction import index_of_refraction
    #
    # startTime = time.time()
    #
    # if __name__=='__main__':
    #     p = argparse.ArgumentParser(description='Beam caustics')
    #     p.add_argument('-s', '--save', dest='save', metavar='BOOL', default=False, help='enables saving .dat file')
    #     p.add_argument('-p', '--plots', dest='plots', metavar='BOOL', default=False, help='enables graphical display of the result')
    #     p.add_argument('-c', '--caustics', dest='caustics', metavar='BOOL', default=False, help='calculates the beam caustics')
    #     p.add_argument('-e', '--beamE', dest='beamE', metavar='NUMBER', default=0, type=float, help='beam energy in keV')
    #     p.add_argument('-stck', '--stack', dest='stack', metavar='NUMBER', default=1, type=int, help='1 for stack 1 (CDn) 2 for stack_2 (CDo) or 0 for ideal CRL')
    #     p.add_argument('-n', '--n', dest='n', metavar='NUMBER', default=10, type=int, help='number of lenslets')
    #     p.add_argument('-cr', '--cst_range', dest='cst_range', metavar='NUMBER', default=1, type=float, help='caustic range around zero [m]')
    #     p.add_argument('-cp', '--cst_points', dest='cst_points', metavar='NUMBER', default=1001, type=float, help='number of points for caustics calcuation')
    #     p.add_argument('-d', '--defocus', dest='defocus', metavar='NUMBER', default=0, type=float, help='defocus in [m], (-) means before focus, (+) means after focus')
    #     p.add_argument('-prfx', '--prfx', dest='prfx', metavar='STRING',  help='prefix for saving files')
    #     p.add_argument('-dir', '--dir', dest='dir', metavar='STRING', default='./simulations/', help='prefix for saving files')
    #     p.add_argument('-mtr', '--mtr', dest='mtr', metavar='STRING', default='./metrology/', help='prefix for saving files')
    #
    #     args = p.parse_args()
    #
    #     save = eval(args.save)
    #     plots = eval(args.plots)
    #     caustics = eval(args.caustics)
    #
    #     beamE = args.beamE
    #     cst_range = (-args.cst_range/2, args.cst_range/2)
    #     cst_pts = int(args.cst_points)     # caustic number of points
    #     defocus = args.defocus
    #
    #     #############################################################################
    #     #############################################################################
    #
    #     prfx = args.prfx
    #     if args.stack == 1:
    #         prfx += '_CDn_A350um_'
    #     elif args.stack == 2:
    #         prfx += '_CDo_'
    #     elif args.stack == 0:
    #         prfx += '_ideal_'
    #
    #     strDataFolderName = args.dir
    #     metrology = args.mtr
    #
    #     energy = str(beamE)
    #     energy = energy.replace('.', 'p')
    #
    #     position = str(defocus * 1e3)
    #     position = position.replace('.', 'p')
    #
    #     strIntPropOutFileName = prfx + energy + 'keV_d' + position + 'mm_n' + str(args.n) + '_intensity.dat'
    #     strPhPropOutFileName  = prfx + energy + 'keV_d' + position + 'mm_n' + str(args.n) + '_phase.dat'
    #
    #     #############################################################################
    #     #############################################################################
    #     # Logging all logging.infos
    #
    #     # Get time stamp
    #     start0 = time.time()
    #     dt = datetime.datetime.fromtimestamp(start0).strftime('%Y-%m-%d_%H:%M:%S')
    #
    #     # Initializing logging
    #     log = logging.getLogger('')
    #     log.setLevel(logging.INFO)
    #     format = logging.Formatter('%(levelname)s: %(message)s')
    #
    #     ch = logging.StreamHandler(sys.stdout)
    #     ch.setFormatter(format)
    #     log.addHandler(ch)
    #
    #     fh = logging.handlers.RotatingFileHandler(os.path.join(os.getcwd(), strDataFolderName)+'/'+dt+'_'+strIntPropOutFileName.replace('_intensity.dat','.log'), maxBytes=(1048576 * 5), backupCount=7)
    #     fh.setFormatter(format)
    #     log.addHandler(fh)
    #
    #     #############################################################################
    #     #############################################################################
    #     wfr_resolution = (256, 256)  # nx, ny
    #     screen_range = (-0.5E-3, 0.5E-3, -0.5E-3, 0.5E-3)  # x_Start, x_Fin, y_Start, y_Fin
    #     sampling_factor = 0.0 # sampling factor for adjusting nx, ny (effective if > 0)
    #     wavelength = srwl_uti_ph_en_conv(beamE, _in_u='keV', _out_u='m')
    #     z = 60
    #     #############################################################################
    #     #############################################################################
    #     # Photon source
    #
    #     #********************************Undulator parameters (CPMU18)
    #     numPer = 111			# Number of ID Periods
    #     undPer = 0.018		# Period Length [m]
    #     phB = 0	        	# Initial Phase of the Horizontal field component
    #     sB = 1		        # Symmetry of the Horizontal field component vs Longitudinal position
    #     xcID = 0 			# Transverse Coordinates of Undulator Center [m]
    #     ycID = 0
    #     zcID = 0
    #     n = 1
    #     #********************************Storage ring parameters
    #     eBeam = SRWLPartBeam()
    #     eBeam.Iavg = 0.2             # average Current [A]
    #     eBeam.partStatMom1.x = 0
    #     eBeam.partStatMom1.y = 0
    #     eBeam.partStatMom1.z = -0.5*undPer*(numPer + 4)    # initial Longitudinal Coordinate (set before the ID)
    #     eBeam.partStatMom1.xp = 0  					       # initial Relative Transverse Velocities
    #     eBeam.partStatMom1.yp = 0
    #
    #     # e- beam paramters (RMS) EBS
    #     sigEperE = 9.3E-4  # relative RMS energy spread
    #     sigX = 30.3E-06  # horizontal RMS size of e-beam [m]
    #     sigXp = 4.4E-06  # horizontal RMS angular divergence [rad]
    #     sigY = 3.6E-06  # vertical RMS size of e-beam [m]
    #     sigYp = 1.46E-06  # vertical RMS angular divergence [rad]
    #     eBeam.partStatMom1.gamma = 6.00 / 0.51099890221e-03  # Relative Energy
    #
    #     n = 1
    #     if (2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1)) <= 0:
    #         n=3
    #         if (2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1)) <= 0:
    #             n = 5
    #             if (2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1)) <= 0:
    #                 n = 7
    #
    #     K = np.sqrt(2 * (2 * n * wavelength * eBeam.partStatMom1.gamma ** 2 / undPer - 1))
    #     B = K / (undPer * 93.3728962)  # Peak Horizontal field [T] (undulator)
    #
    #     # 2nd order stat. moments
    #     eBeam.arStatMom2[0] = sigX*sigX			 # <(x-<x>)^2>
    #     eBeam.arStatMom2[1] = 0					 # <(x-<x>)(x'-<x'>)>
    #     eBeam.arStatMom2[2] = sigXp*sigXp		 # <(x'-<x'>)^2>
    #     eBeam.arStatMom2[3] = sigY*sigY		     # <(y-<y>)^2>
    #     eBeam.arStatMom2[4] = 0					 # <(y-<y>)(y'-<y'>)>
    #     eBeam.arStatMom2[5] = sigYp*sigYp		 # <(y'-<y'>)^2>
    #     eBeam.arStatMom2[10] = sigEperE*sigEperE # <(E-<E>)^2>/<E>^2
    #
    #     # Electron trajectory
    #     eTraj = 0
    #
    #     # Precision parameters
    #     arPrecSR = [0]*7
    #     arPrecSR[0] = 1		# SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
    #     arPrecSR[1] = 0.01	# relative precision
    #     arPrecSR[2] = 0		# longitudinal position to start integration (effective if < zEndInteg)
    #     arPrecSR[3] = 0		# longitudinal position to finish integration (effective if > zStartInteg)
    #     arPrecSR[4] = 20000	# Number of points for trajectory calculation
    #     arPrecSR[5] = 1		# Use "terminating terms"  or not (1 or 0 respectively)
    #     arPrecSR[6] = sampling_factor # sampling factor for adjusting nx, ny (effective if > 0)
    #     sampFactNxNyForProp = arPrecSR[6] # sampling factor for adjusting nx, ny (effective if > 0)
    #
    #     und = SRWLMagFldU([SRWLMagFldH(n, 'v', B, phB, sB, 1)], undPer, numPer)
    #
    #     magFldCnt = SRWLMagFldC([und], array('d', [xcID]), array('d', [ycID]), array('d', [zcID]))
    #
    #     #********************************Wavefronts
    #
    #     # Monochromatic wavefront
    #     wfr = SRWLWfr()
    #     wfr.allocate(1, wfr_resolution[0], wfr_resolution[1])  # Photon Energy, Horizontal and Vertical Positions
    #     wfr.mesh.zStart = z
    #     wfr.mesh.eStart = beamE * 1E3
    #     wfr.mesh.xStart = screen_range[0]
    #     wfr.mesh.xFin = screen_range[1]
    #     wfr.mesh.yStart = screen_range[2]
    #     wfr.mesh.yFin = screen_range[3]
    #     wfr.partBeam = eBeam
    #     meshPartCoh = deepcopy(wfr.mesh)
    #
    #     #############################################################################
    #     #############################################################################
    #     # Wavefront generation
    #
    #     # if (srwl_uti_proc_is_master()):
    #     # ********************************Calculating Initial Wavefront and extracting Intensity:
    #     logging.info('- Performing Initial Electric Field calculation ... ')
    #     srwl.CalcElecFieldSR(wfr, eTraj, magFldCnt, arPrecSR)
    #     logging.info('Initial wavefront:')
    #     logging.info('Nx = %d, Ny = %d' % (wfr.mesh.nx, wfr.mesh.ny))
    #     logging.info('dx = %.4f um, dy = %.4f um' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E6 / wfr.mesh.nx,
    #                                                  (wfr.mesh.yFin - wfr.mesh.yStart) * 1E6 / wfr.mesh.ny))
    #     logging.info('range x = %.4f mm, range y = %.4f mm' % ((wfr.mesh.xFin - wfr.mesh.xStart) * 1E3,
    #                                                            (wfr.mesh.yFin - wfr.mesh.yStart) * 1E3))
    #     logging.info('Rx = %.6f, Ry = %.6f' % (wfr.Rx, wfr.Ry))
    #
    #     #############################################################################
    #     #############################################################################
    #     # Beamline assembly
    #     logging.info('Setting up beamline') if (srwl_uti_proc_is_master()) else 0
    #
    #     # ============= Ideal lens for beam conditioning =======================#
    #     ''' In order to emulate a plane wave, a ideal thin lens is placed just before the first optical element. Focal distances
    #     are taken from the wavefront radii.'''
    #     oeThinLens = SRWLOptL(_Fx=wfr.Rx, _Fy=wfr.Ry)
    #
    #     # ============= Single lens parameters =================================#
    #     '''Some parameters to define '''
    #     delta, beta, atten_len = index_of_refraction('Be', beamE*1e3) # delta, beta, atten_len
    #     R = 50 * 1E-6          # CRL radius at the parabola appex
    #     nCRL = args.n
    #     CRLAph = 444 * 1E-6    # CRL aperture
    #     CRLApv = 444 * 1E-6
    #     wt = 20. * 1E-6         # CRL wall thickness [um]
    #     shp = 1                 # 1- parabolic, 2- circular (spherical)
    #     foc_plane = 3           # plane of focusing: 1- horizontal, 2- vertical, 3- both
    #     ContainerThickness = 2e-3
    #     drift_lens = SRWLOptD(ContainerThickness)  # container thickness
    #     f_CRL  = R/(2*delta*nCRL) + ((nCRL-1)*ContainerThickness)/6
    #     oeApCRL = SRWLOptA(_shape='c', _ap_or_ob='a', _Dx=350e-6)
    #     oeCRL = srwl_opt_setup_CRL(foc_plane, delta, atten_len, shp, CRLAph, CRLApv, R,  1, wt,_xc=0., _yc=0., _nx=2001, _ny=2001)
    #     DriftCRL = SRWLOptD(ContainerThickness)
    #     # ============= Drift space  ===========================================#
    #     if caustics:
    #         if args.n == 1:
    #             Drift = SRWLOptD(4.6935563955 + cst_range[0])  # container thickness
    #             logging.info('Caustics begin at: %.6f' % (4.6935563955 + cst_range[0]))
    #
    #         elif args.n == 5:
    #             Drift = SRWLOptD(0.9363090268 + cst_range[0])  # container thickness
    #             logging.info('Caustics begin at: %.6f' % (0.9363090268 + cst_range[0]))
    #
    #         elif args.n == 10:
    #             Drift = SRWLOptD(0.4636368601 + cst_range[0])  # container thickness
    #             logging.info('Caustics begin at: %.6f' % (0.4636368601 + cst_range[0]))
    #
    #     else:
    #         if args.n == 1:
    #             Drift = SRWLOptD(4.6935563955 + defocus)  # container thickness
    #             logging.info('Image at: %.6f' % (4.6935563955 + defocus))
    #         elif args.n == 5:
    #             Drift = SRWLOptD(0.9363090268 + defocus)  # container thickness
    #             logging.info('Image at: %.6f' % (0.9363090268 + defocus))
    #         elif args.n == 10:
    #             Drift = SRWLOptD(0.4636368601 + defocus)  # container thickness
    #             logging.info('Image at: %.6f' % (0.4636368601 + defocus))
    #
    #     # ============= Wavefront Propagation Parameters =======================#
    #     #                [ 0] [1] [2]  [3]  [4]  [5]  [6]  [7]   [8]  [9] [10] [11]
    #     ppThinLens		=[ 0,  0, 1.,   0,   0,  1., 10.,  1.,  10.,   0,   0,   0]
    #     ppApCRL    		=[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,   1.,   0,   0,   0]
    #     ppCRL    		=[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,   1.,   0,   0,   0]
    #     ppDrift    		=[ 0,  0, 1.,   1,   0,  1.,  1.,  1.,   1.,   0,   0,   0]
    #     ppDrift_cstc    =[ 0,  0, 1.,   0,   0,  1.,  1.,  1.,   1.,   0,   0,   0]
    #
    #     if args.stack == 0:
    #         if args.n == 1:
    #             ppFinal = [0, 0, 1., 0, 0, 0.4, 2., 0.4, 2., 0, 0, 0]
    #             optBL = SRWLOptC([oeThinLens, oeApCRL, oeCRL,   Drift],
    #                              [ppThinLens, ppApCRL, ppCRL, ppDrift, ppFinal]
    #                              )
    #         elif args.n == 5:
    #             ppFinal = [0, 0, 1., 0, 0, 0.45, 2.5, 0.45, 2.5, 0, 0, 0]
    #             optBL = SRWLOptC([oeThinLens, oeApCRL, oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL,    Drift],
    #                              [ppThinLens, ppApCRL, ppCRL,  ppDrift,
    #                                                    ppCRL,  ppDrift,
    #                                                    ppCRL,  ppDrift,
    #                                                    ppCRL,  ppDrift,
    #                                                    ppCRL,  ppDrift, ppFinal]
    #                              )
    #
    #         elif args.n == 10:
    #             ppFinal = [0, 0, 1., 0, 0, 0.85, 1.17, 0.85, 1.17, 0, 0, 0]
    #             optBL = SRWLOptC([oeThinLens, oeApCRL, oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL, DriftCRL,
    #                                                    oeCRL,    Drift],
    #                              [ppThinLens, ppApCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift,
    #                                                    ppCRL, ppDrift, ppFinal]
    #                              )
    #     else:
    #         if args.stack == 1:
    #             c = 'n'
    #         else:
    #             c = 'o'
    #
    #         # ============= Figure erros  ===========================================#
    #         amp_coef = 1
    #         file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_01.dat'.replace('_x_',c)
    #         heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #         CD01 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #         logging.info('CD01')
    #         if args.n > 1:
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_02.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD02 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD02')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_03.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD03 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD03')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_04.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD04 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD04')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_05.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD05 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD05')
    #         if args.n > 5:
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_06.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD06 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD06')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_07.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD07 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD07')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_08.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD08 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD08')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_09.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD09 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD09')
    #             file_name = 'SRW_M_thk_res_workflow_a_FC_CD_x_10.dat'.replace('_x_',c)
    #             heightProfData, HPDmesh = srwl_uti_read_intens_ascii(os.path.join(os.getcwd(), metrology, file_name))
    #             CD10 = b4ro.srwl_opt_setup_CRL_error(heightProfData, HPDmesh, delta, atten_len, _amp_coef=-amp_coef)
    #             logging.info('CD10')
    #
    #         if args.n == 1:
    #             ppFinal = [0, 0, 1., 0, 0, 0.4, 2., 0.4, 2., 0, 0, 0]
    #
    #             optBL = SRWLOptC([oeThinLens, oeApCRL, oeCRL,  CD01,  Drift],
    #                              [ppThinLens, ppApCRL, ppCRL, ppCRL, ppDrift, ppFinal]
    #                              )
    #         elif args.n == 5:
    #             ppFinal = [0, 0, 1., 0, 0, 0.45, 2.5, 0.45, 2.5, 0, 0, 0]
    #             optBL = SRWLOptC([oeThinLens, oeApCRL, oeCRL,  CD01, DriftCRL,
    #                                                    oeCRL,  CD02, DriftCRL,
    #                                                    oeCRL,  CD03, DriftCRL,
    #                                                    oeCRL,  CD04, DriftCRL,
    #                                                    oeCRL,  CD05,    Drift],
    #                              [ppThinLens, ppApCRL, ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift, ppFinal]
    #                              )
    #
    #         elif args.n == 10:
    #             ppFinal = [0, 0, 1., 0, 0, 0.85, 1.17, 0.85, 1.17, 0, 0, 0]
    #             optBL = SRWLOptC([oeThinLens, oeApCRL, oeCRL,  CD01, DriftCRL,
    #                                                    oeCRL,  CD02, DriftCRL,
    #                                                    oeCRL,  CD03, DriftCRL,
    #                                                    oeCRL,  CD04, DriftCRL,
    #                                                    oeCRL,  CD05, DriftCRL,
    #                                                    oeCRL,  CD06, DriftCRL,
    #                                                    oeCRL,  CD07, DriftCRL,
    #                                                    oeCRL,  CD08, DriftCRL,
    #                                                    oeCRL,  CD09, DriftCRL,
    #                                                    oeCRL,  CD10,    Drift],
    #                              [ppThinLens, ppApCRL, ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift,
    #                                                    ppCRL, ppCRL, ppDrift, ppFinal]
    #                              )
    #
    #     #############################################################################
    #     #############################################################################
    #     # Electric field propagation
    #     logging.info('- Simulating Electric Field Wavefront Propagation ... ')
    #     srwl.PropagElecField(wfr, optBL)
    #
    #     logging.info('Propagated wavefront:')
    #     logging.info('Nx = %d, Ny = %d' % (wfr.mesh.nx, wfr.mesh.ny))
    #     logging.info('dx = %.4f um, dy = %.4f um' % ((wfr.mesh.xFin-wfr.mesh.xStart)*1E6/wfr.mesh.nx,
    #                                                  (wfr.mesh.yFin-wfr.mesh.yStart)*1E6/wfr.mesh.ny))
    #     logging.info('range x = %.4f um, range y = %.4f um' % ((wfr.mesh.xFin-wfr.mesh.xStart)*1E6,
    #                                                            (wfr.mesh.yFin-wfr.mesh.yStart)*1E6))
    #     logging.info('Rx = %.10f, Ry = %.10f' % (wfr.Rx, wfr.Ry))
    #
    #     if save is True or plots is True:
    #         arI1 = array('f', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" 2D array to take intensity data
    #         srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, wfr.mesh.eStart, 0, 0)
    #         arP1 = array('d', [0] * wfr.mesh.nx * wfr.mesh.ny)  # "flat" array to take 2D phase data (note it should be 'd')
    #         srwl.CalcIntFromElecField(arP1, wfr, 0, 4, 3, wfr.mesh.eStart, 0, 0)
    #
    #     if save:
    #         srwl_uti_save_intens_ascii(arI1, wfr.mesh, os.path.join(os.getcwd(), strDataFolderName,  strIntPropOutFileName), 0)
    #         srwl_uti_save_intens_ascii(arP1, wfr.mesh, os.path.join(os.getcwd(), strDataFolderName, strPhPropOutFileName),0)
    #
    #     logging.info('>> single electron calculations: done')
    #
    #     if caustics:
    #         logging.info('- Performing Initial Electric Field calculation ... ')
    #         wftp = deepcopy(wfr)
    #         IntVsZX, IntVsZY, cstcMesh, FWHMxVsZ, FWHMyVsZ = srwl_wfr_prop_drifts(wftp,(cst_range[1] - cst_range[0]) / (cst_pts),cst_pts, ppDrift_cstc, _pol=6, _type=0)
    #
    #         # cstc vs X
    #         mesh_cstc_x = copy(cstcMesh)
    #         mesh_cstc_x.xStart = cst_range[0]
    #         mesh_cstc_x.xFin = cst_range[1]
    #         mesh_cstc_x.nx = cst_pts + 1
    #         mesh_cstc_x.yStart = cstcMesh.xStart
    #         mesh_cstc_x.yFin = cstcMesh.xFin
    #         mesh_cstc_x.ny = cstcMesh.nx
    #
    #         # cstc vs Y
    #         mesh_cstc_y = copy(cstcMesh)
    #         mesh_cstc_y.xStart = cst_range[0]
    #         mesh_cstc_y.xFin = cst_range[1]
    #         mesh_cstc_y.nx = cst_pts + 1
    #         mesh_cstc_y.yStart = cstcMesh.yStart
    #         mesh_cstc_y.yFin = cstcMesh.yFin
    #         mesh_cstc_y.ny = cstcMesh.ny
    #
    #         if save:
    #             srwl_uti_save_intens_ascii(IntVsZX, mesh_cstc_x, os.path.join(os.getcwd(), strDataFolderName,
    #                                                                           strIntPropOutFileName.replace('_intensity',
    #                                                                                                         '_intensity_cstc_X')), 0)
    #
    #             srwl_uti_save_intens_ascii(IntVsZY, mesh_cstc_y, os.path.join(os.getcwd(), strDataFolderName,
    #                                                                           strIntPropOutFileName.replace('_intensity',
    #                                                                                                         '_intensity_cstc_Y')), 0)
    #
    #         logging.info('>> caustic calculations: done')
    #
    #     deltaT = time.time() - startTime
    #     hours, minutes = divmod(deltaT, 3600)
    #     minutes, seconds = divmod(minutes, 60)
    #     logging.info(">>>> Elapsed time: " + str(int(hours)) + "h " + str(int(minutes)) + "min " + str(seconds) + "s ") if (srwl_uti_proc_is_master()) else 0
    #
    #     if plots is True:
    #         # ********************************Electrical field intensity and phase after propagation
    #         plotMesh1x = [1E6 * wfr.mesh.xStart, 1E6 * wfr.mesh.xFin, wfr.mesh.nx]
    #         plotMesh1y = [1E6 * wfr.mesh.yStart, 1E6 * wfr.mesh.yFin, wfr.mesh.ny]
    #         uti_plot2d(arI1, plotMesh1x, plotMesh1y,
    #                    ['Horizontal Position [um]', 'Vertical Position [um]', 'Intensity After Propagation'])
    #         uti_plot2d(arP1, plotMesh1x, plotMesh1y,
    #                    ['Horizontal Position [um]', 'Vertical Position [um]', 'Phase After Propagation'])
    #
    #         if caustics:
    #             if plots:
    #                 plotMesh1x = [1E6 * mesh_cstc_x.xStart, 1E6 * mesh_cstc_x.xFin, mesh_cstc_x.nx]
    #                 plotMesh1y = [1E6 * mesh_cstc_x.yStart, 1E6 * mesh_cstc_x.yFin, mesh_cstc_x.ny]
    #                 uti_plot2d(IntVsZX, plotMesh1x, plotMesh1y,
    #                            ['Horizontal Position [um]', 'Vertical Position [um]', 'Caustic vs X'])
    #
    #                 plotMesh1x = [1E6 * mesh_cstc_y.xStart, 1E6 * mesh_cstc_y.xFin, mesh_cstc_y.nx]
    #                 plotMesh1y = [1E6 * mesh_cstc_y.yStart, 1E6 * mesh_cstc_y.yFin, mesh_cstc_y.ny]
    #                 uti_plot2d(IntVsZY, plotMesh1x, plotMesh1y,
    #                            ['Horizontal Position [um]', 'Vertical Position [um]', 'Caustic vs Y'])
    #         uti_plot_show()
    #
    #
    #