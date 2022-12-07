Mechanism of a simulation flowchart:

1) create the deformation files and the targets (polynomial coeffs - we use orthonormal polynomials)
   script: run_create_2d_zernike_sampled_profiles.py (first creates the profiles, then write: not very effcient)
   or
   script: run_create_files_2d_zernike_sampled_profiles.py (create and write iteratively: more effcient)

   files: /scisoft/users/srio/ML_TRAIN2/5000_2Dv1/tmp_ml00xxxx.[h5,txt]

2) run comsyl+wofry or srw for the optical system for every deformation file previosly created, and write obtained
   intensities at different (64) image planes
   script: run_.py
   files: /scisoft/users/srio/XXXX   tmp_ml.h5 tmp_ml_targets_gs.txt tmp_ml_targets_z.txt
   OASYS workspace: workspaces/ML_COMSYL_v1.ows

# 3) Define and train CNN
#    script: training_v1.py
#    files: /scisoft/users/srio/ML_TRAIN2/ training_vNN.h5 training_vNN.json
#
#    default: 5000 samples, GS coefficients, 64 planes
#
#    v12 with 1000 samples
#    v13 [default]
#    v14 with Zernike coeffs
#    v15 32 planes
#    v16 16 planes
#    v19 32 planes downstream from focus
#    v19 32 planes downstream from focus
#    v20 multimode standard
#    v23 multimode 25000 epochs
#
# 4) test CNN (loads the CNN from h5 file and runs the test cases)
#    script: testing_v1.py


