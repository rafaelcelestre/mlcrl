#!/usr/bin/python
# coding: utf-8

import os
import numpy as np
import glob

# Global parameters
save = True       # saves dat file for a particular position (used with caustics = False / psf calculation)
plot = True       # graphical display (used with caustics = False / psf calculation)
caustics = True   # enables the calculation of the 2D cuts along the optical axis
E = 7             # Energy in keV of the simulaitons
z = 75.557433     # position in [m] of the first optical element (X_ray lens) ~20:1 demag

cst_pts = 5      # number of planes
cst_rg = 1        # caustic range in [m]; np.linspace(-cst_rg/2, cst_rg/2, cst_pts)

prfx = 'img_XXX.h5'     # file series name - please, keep the XXX
directory = './example_set_XXX'  # data set number - please, keep the XXX

sets = 2                # number of data sets to be generated using random Zernike polynomials
metrology = True        # it is possible to generate datasets (20) using real metrology data; this adds to "sets"

# parameters we do not need to often change
d = 0             # defocus in [m]; (<0) upstream the image plane; (>0) downstream
illum = 1         # type of illumination: 0 - plane wave; 1 - parabolic wave
delta = 6.9483563463709E-6    # n = 1-delta + i\cdot beata
beta = 3.653794312073E-9

seed = 69         # seed for generation of the random Zernike profiles

k = 0
rg = np.random.default_rng(seed)

for dataset in range(sets):
    sd = rg.integers(1,6969)
    dir = directory.replace('XXX', '%.3d') % k
    cmd = 'mkdir %s' % dir
    print(cmd)
    os.system(cmd)
    if k == 0:    # ideal lens
        lens = 0
        cmd = 'python synthetic_data.py ' \
              '-s %s -p %s -c %s -e %.4f -z %e -i %d -nd %e -nb %e -l %d -sd %d -cr %.6f -cp %d -d %f -prfx %s -dir %s'\
              % (save, plot, caustics, E, z, illum, delta, beta, lens, sd, cst_rg, cst_pts, d, prfx, dir)
    else:
        lens = 1
        cmd = 'python synthetic_data.py ' \
              '-s %s -p %s -c %s -e %.4f -z %e -i %d -nd %e -nb %e -l %d -sd %d -cr %.6f -cp %d -d %f -prfx %s -dir %s'\
              % (save, plot, caustics, E, z, illum, delta, beta, lens, sd, cst_rg, cst_pts, d, prfx, dir)
    print(cmd)
    os.system(cmd)
    k+=1

if metrology:
    lens = 2
    mtrl_files =sorted(glob.glob('./metrology/*.dat'))

    # for dataset in mtrl_files:
    dataset = mtrl_files[0]
    sd = rg.integers(1,6969)
    dir = directory.replace('XXX', '%.3d') % k
    cmd = 'mkdir %s' % dir
    print(cmd)
    os.system(cmd)

    cmd = 'python synthetic_data.py ' \
    '-s %s -p %s -c %s -e %.4f -z %e -i %d -nd %e -nb %e -l %d -sd %d -cr %.6f -cp %d -d %f -prfx %s -dir %s -m %s'\
    % (save, plot, caustics, E, z, illum, delta, beta, lens, sd, cst_rg, cst_pts, d, prfx, dir, dataset)
    print(cmd)
    os.system(cmd)
    k += 1