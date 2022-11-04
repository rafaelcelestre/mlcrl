Mechanism of a simulation flowchart:

1) create the deformation files and the targets (polynomial coeffs - we use orthonormal polynomials)
   script: create_1d_gramschmidt_sampled_profiles.py
   files: ~/Oasys/ML_TRAIN/tmp_ml000011.[dat,txt]

2) run wofry for the optical system for every deformation file previosly created, and write obtained
   intensities at different (64) image planes
   script: run_wofry1d.py
   files: /scisoft/users/srio/ML_TRAIN2/*

3) Define and train CNN
   script: training_v1.py
   files: h5 pkl