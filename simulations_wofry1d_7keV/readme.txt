Mechanism of a simulation flowchart:

V10: First version of paper. (single-lens) [this includes multimode that are labeled V2*].

Second version (multi-lens).
V20: using all random over 1500 um
V25: using Rafael recipe over 1500 um
V26: using Rafael recipe recipe over 800 um
V27: using all random over 800 um

1) create the deformation files and the targets (polynomial coeffs - we use orthonormal polynomials)
   # V10
   # script: run_create_1d_gramschmidt_sampled_profiles.py
   # files: ~/Oasys/ML_TRAIN5000/tmp_ml000011.[dat,txt]  -> moved to: /scisoft/users/srio/MLCRL/V10/ML_TRAIN5000
   V 20
   script: run_create_1d_gramschmidt_sampled_profiles_v20.py
   files: ~/Oasys/ML_TRAIN_V20_{5000,25000}/tmp_ml000011.[dat,txt] -> moved to: /scisoft/users/srio/MLCRL/V20/ML_TRAIN25000

2) run wofry for the optical system for every deformation file previosly created, and write obtained
   intensities at different (64) image planes
   # V10
   # script: run_wofry1d.py
   # files: /scisoft/users/srio/MLCRL/V10/ML_TRAIN2/   tmp_ml.h5 tmp_ml_targets_gs.txt tmp_ml_targets_z.txt
   # OASYS workspace: ML_Wofry1d_v1.ows

   V20
   script: run_wofry1d_v20.py
   N=5000:  files:  /scisoft/users/srio/MLCRL/V20/ML_TRAIN2_V20/   tmp_ml.h5 tmp_ml_targets_gs.txt tmp_ml_targets_z.txt
   N=25000: files:  /scisoft/users/srio/MLCRL/V20/ML_TRAIN2_V21/   tmp_ml.h5 tmp_ml_targets_gs.txt tmp_ml_targets_z.txt
   OASYS workspace: ML_Wofry1d_v20.ows

3) Define and train CNN
   # V10
   # script: training_v1.py
   # files: /scisoft/users/srio/MLCRL/V10/ML_TRAIN2/ training_vNN.h5 training_vNN.json
   # v20_1 with 1000 samples
   # v20_2 [default]

   # v14 with Zernike coeffs
   # v15 32 planes
   # v16 16 planes
   # v19 32 planes downstream from focus
   # v19 32 planes downstream from focus
   # v20 multimode standard
   # v23 multimode 25000 epochs
   # default: 5000 samples, GS coefficients, 64 planes

   V20
   script: training_v20.py
   files: /scisoft/users/srio/ML_TRAIN2_V20/1000 training_vNN.h5 training_vNN.json
   files: /scisoft/users/srio/ML_TRAIN2_V20/5000 training_vNN.h5 training_vNN.json


4) test CNN (loads the CNN from h5 file and runs the test cases)
   # V10
   # script: testing_v1.py

   V20
   script: testing_v20.py


