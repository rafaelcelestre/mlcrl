#!/usr/bin/env python
# coding: utf-8

# In[5]:


from mlcrl.phasenet.model import PhaseNet, Config, Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# ## The phasenet model can be trained using the following three steps: 

# ### STEP 1 : Setup the config

# In[6]:


# from IPython.display import Markdown
import re
hlp = '### ' + Config.__doc__
def _replace(s):
    return f"  \n\n`{s.group(1).strip()}`: "
hlp = re.sub(r'\n\s*:param([^:]+):', _replace, hlp)
# display(Markdown(hlp))


# ### Important parameters
# 
# - Please set the Zernike modes for your application (i.e. which modes in what ranges should the neural network be able to predict) by changing the parameter `zernike_amplitude_ranges`. Please modify `zernike_order` if the Zernikes are not provided in Noll index.
# 
# - It is also important to set the PSF parameters (`psf_lam_detection`, `psf_units`, `psf_na_detection`, `psf_n`) to match the physical properties of the microscope being used.
# 
# - It is also advisable to set the noise characteristics (`noise_mean`, `noise_sigma`, `noise_snr`) and phantom parameters (`phantom_params`) to match your image acquisition setup.
# 
# - The training and network parameters (`train_*`, `net_*`) likely don't have to be changed.

# In[11]:


#zern = [5,6,7,8,9,10,11,12,13,14,15] # for final training
zern = [5]                            # for trial
amp_range = [0.15]*len(zern)
amps = dict(zip(zern, amp_range))


# In[12]:


c = Config(zernike_amplitude_ranges=amps, psf_na_detection=1.1, psf_units=(0.1,0.086,0.086), psf_n=1.33,
           psf_lam_detection=0.515, noise_mean=None, noise_snr=None, noise_sigma=None)
vars(c)


# ## STEP 2 : Setup the model

# In[13]:


model = PhaseNet(config=c, name='test', basedir='models')


# ## STEP 3: Train the model

# **Note**: We only set `epochs=20` here for a quick demo. Remove that to train the model properly (will take much longer).

# In[18]:


model.train(epochs=5)



# ### Validation on synthetic data

# In[19]:


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
psfs, amps = next(data.generator())
psfs.shape, amps.shape


# In[20]:


amps_pred = np.array([model.predict(psf) for psf in tqdm(psfs)])


# In[21]:


plt.figure(figsize=(10,8))
ind = np.argsort(amps.ravel())
plt.plot(amps[ind], marker='*', label='gt')
plt.plot(amps_pred[ind], '--', marker='*', label='pred')
# plt.hlines(-0.2, *plt.axis()[:2])
# plt.hlines(+0.2, *plt.axis()[:2])
plt.xlabel('test psf')
plt.ylabel(f'amplitude {tuple(model.config.zernike_amplitude_ranges.keys())[0]}')
plt.legend()
None;
plt.show()


# In[ ]:




