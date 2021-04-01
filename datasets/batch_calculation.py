#!/usr/bin/python
# coding: utf-8

import os



# p.add_argument('-s', '--save', dest='save', metavar='BOOL', default=False, help='enables saving .dat file')
# p.add_argument('-p', '--plots', dest='plots', metavar='BOOL', default=False, help='enables graphical display of the result')
# p.add_argument('-c', '--caustics', dest='caustics', metavar='BOOL', default=False, help='calculates the beam caustics')
# p.add_argument('-e', '--beamE', dest='beamE', metavar='NUMBER', default=0, type=float, help='beam energy in keV')
# p.add_argument('-i', '--illum', dest='illum', metavar='NUMBER', default=0, type=int, help='0 for plane phase; 1 for parabolic phase')
# p.add_argument('-nd', '--delta', dest='delta', metavar='NUMBER', default=1e-23, type=float, help='n=1- delta + i beta')
# p.add_argument('-nb', '--beta', dest='beta', metavar='NUMBER', default=1e-23, type=float, help='n=1- delta + i beta')
# p.add_argument('-cr', '--cst_range', dest='cst_range', metavar='NUMBER', default=1, type=float, help='caustic range around zero [m]')
# p.add_argument('-cp', '--cst_points', dest='cst_points', metavar='NUMBER', default=33, type=float, help='number of points for caustics calcuation')
# p.add_argument('-d', '--defocus', dest='defocus', metavar='NUMBER', default=0, type=float, help='defocus in [m], (-) means before focus, (+) means after focus')
# p.add_argument('-prfx', '--prfx', dest='prfx', metavar='STRING',  help='prefix for saving files')
# p.add_argument('-dir', '--dir', dest='dir', metavar='STRING', default='./', help='directory for saving files')
save = False
plot = True
caustics = False
E = 7
illum = 0
delta = 6.9483563463709E-6
beta = 3.653794312073E-9
cst_rg = 1
cst_pts = 33
d = 0
cmd = 'python caustics.py -s %s -p %s -c %s -e %.4f -i %d -nd %e -nb %e' %(save, plot, caustics, E, illum, delta, beta)

print(cmd)
# cmd = 'python caustics.py -s True -p False -c True -e %f -stck %d -n %d -cr %f -cp %d -prfx "Be" -dir "caustics_focusing"' % (E, stack, n, cst, cst_pts)
# print(cmd)
# os.system(cmd)

