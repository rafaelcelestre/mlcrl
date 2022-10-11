from mlcrl.phasenet.model import PhaseNet, Config, Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

zern = [5]                            # for trial
amp_range = [0.15]*len(zern)
amps = dict(zip(zern, amp_range))

# In[12]:


c = Config(zernike_amplitude_ranges=amps,
           psf_na_detection=1.1,
           psf_units=(0.1,0.086,0.086),
           psf_n=1.33,
           psf_lam_detection=0.515,
           noise_mean=None,
           noise_snr=None,
           noise_sigma=None,
           psf_shape = (32, 64, 64),
           crop_shape = (32//2, 64//2, 64//2),
                    )
print(vars(c))


model = PhaseNet(config=c, name='test', basedir='models')


# ## STEP 3: Train the model

# **Note**: We only set `epochs=20` here for a quick demo. Remove that to train the model properly (will take much longer).

model.train(epochs=2)

print("model.config.zernike_amplitude_ranges", model.config.zernike_amplitude_ranges)
print("model.config.zernike_order", model.config.zernike_order)
print("model.config.zernike_normed", model.config.zernike_normed)
print("model.config.psf_shape", model.config.psf_shape)
print("model.config.psf_units", model.config.psf_units)
print("model.config.psf_na_detection", model.config.psf_na_detection)
print("model.config.psf_lam_detection", model.config.psf_lam_detection)
print("model.config.psf_n", model.config.psf_n)
print("model.config.noise_mean", model.config.noise_mean)
print("model.config.noise_snr", model.config.noise_snr)
print("model.config.noise_sigma", model.config.noise_sigma)
print("model.config.noise_perlin_flag", model.config.noise_perlin_flag)
print("model.config.crop_shape", model.config.crop_shape)
print("model.config.jitter", model.config.jitter)
print("model.config.phantom_params", model.config.phantom_params)
print("model.config.planes", model.config.planes)

data = Data(
    batch_size           = 50,
    amplitude_ranges     = model.config.zernike_amplitude_ranges,
    order                = model.config.zernike_order,
    normed               = model.config.zernike_normed,
    psf_shape            = model.config.psf_shape,
    units                = model.config.psf_units,
    na_detection         = model.config.psf_na_detection,
    lam_detection        = model.config.psf_lam_detection,
    n                    = model.config.psf_n,
    noise_mean           = model.config.noise_mean,
    noise_snr            = model.config.noise_snr,
    noise_sigma          = model.config.noise_sigma,
    noise_perlin_flag    = model.config.noise_perlin_flag,
    crop_shape           = model.config.crop_shape,
    jitter               = model.config.jitter,
    phantom_params       = model.config.phantom_params,
    planes               = model.config.planes,
)

psfs1, amps1 = next(data.generator())
print(psfs1.shape, amps1.shape)


# for i in range(60):
#     print(i)
#     psfs1, amps1 = next(data.generator())
#     # psfs2, amps2 = next(data.generator())
#     print(psfs1.shape, amps1.shape)
#     # print(psfs2.shape, amps2.shape)

