#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mlcrl.phasenet.zernike import ZernikeWavefront, random_zernike_wavefront
import matplotlib.pyplot as plt
import numpy as np


# ## The wavefront can be created by passing a fixed set of amplitudes for different Zernike modes as a list/1D array/dictionary.

# In[2]:


# list of amplitudes starting from piston  #srio: use 4 first polynomials
if False:
    amp = np.random.uniform(-1,1,4)
    f = ZernikeWavefront(amp, order='ansi')
    print(">>> f.zernikes", f.zernikes)
    print(">>> f.amplitudes_noll", f.amplitudes_noll)
    print(">>> f.amplitudes_ansi", f.amplitudes_ansi)

    plt.imshow(f.polynomial(512)); plt.colorbar(); plt.axis('off');
    plt.show()


# In[3]:


# dictionary of amplitudes
if False:
    f = ZernikeWavefront({3:0.1, 5:0.1}, order='ansi')
    print(f.zernikes)

    print(f.amplitudes_noll)
    print(f.amplitudes_ansi)

    plt.imshow(f.polynomial(512)); plt.colorbar(); plt.axis('off');

    plt.show()



# ## A random wavefront can be created by giving an absolute amplitude range as an absolute number or a tuple for different Zernike modes

# In[4]:


if False:
    # random wavefront from a list of absolute amplitudes
    f = random_zernike_wavefront([1,1,1,1], order='ansi')
    print(f.zernikes)

    print(f.amplitudes_noll)
    print(f.amplitudes_ansi)

    plt.imshow(f.polynomial(512)); plt.colorbar(); plt.axis('off');
    plt.show()
#
#
# # In[5]:
#
#
# random wavefront from a list of amplitude ranges given in a tuple
if False:
    f = random_zernike_wavefront([(0,0),(-1,1),(1,2)], order='ansi')
    print(f.zernikes)

    print(f.amplitudes_noll)
    print(f.amplitudes_ansi)

    plt.imshow(f.polynomial(512)); plt.colorbar(); plt.axis('off');
    plt.show()

#
#
# # In[6]:
#
#
# # random wavefront from a dictionary with the range given in a tuple
if True:
    f = random_zernike_wavefront({'defocus':(1,2), (3,-3):5})
    print(f.zernikes)

    print(f.amplitudes_noll)
    print(f.amplitudes_ansi)

    plt.imshow(f.polynomial(512)); plt.colorbar(); plt.axis('off');

    plt.show()
#
