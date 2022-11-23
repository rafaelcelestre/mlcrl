Mechanism of a simulation flowchart:

1) create the deformation files and the targets (polynomial coeffs - we use orthonormal polynomials)
   script: create_1d_gramschmidt_sampled_profiles.py
   files: ~/Oasys/ML_TRAIN5000/tmp_ml000011.[dat,txt]

2) run wofry for the optical system for every deformation file previosly created, and write obtained
   intensities at different (64) image planes
   script: run_wofry1d.py
   files: /scisoft/users/srio/ML_TRAIN2/   tmp_ml.h5 tmp_ml_targets_gs.txt tmp_ml_targets_z.txt
   OASYS workspace: ML_Wofry1d_v1.ows

3) Define and train CNN
   script: training_v1.py
   files: /scisoft/users/srio/ML_TRAIN2/ training_vNN.h5 training_vNN.json

   default: 5000 samples, GS coefficients, 64 planes

   v12 with 1000 samples
   v13 [default]
   v14 with Zernike coeffs
   v15 32 planes
   v16 16 planes
   v19 32 planes downstream from focus




