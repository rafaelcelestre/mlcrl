#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from srxraylib.plot.gol import set_qt
set_qt()


from phasenet.zernike import Zernike
import matplotlib.pyplot as plt
import numpy as np



# We can define the Zernike modes by name, noll, ansi or nm nomenclature

# In[2]:


print(Zernike(5, order='noll')) # can be an integer index following noll index
print(Zernike(3, order='ansi')) # can be an integer index following ansi index
print(Zernike((2,-2))) # can be a tuple for (n,m) indexing
print(Zernike('oblique astigmatism')) # can be a string

Zernike((2,-2)) == Zernike(5, order='noll')


# All Zernikes defined in any format (ansi for example here) are internally mapped to their respective index in other formats

# In[3]:


for i in range(15):
    print(Zernike(i, order='ansi'))


# For visualizing the Zernikes we define the Zernike object and call `polynomial()`.  
# The size of the 2D array is passed as a parameter

# In[4]:


fig, ax = plt.subplots(3,5, figsize=(16,10))
for i,a in enumerate(ax.ravel()):
    z = Zernike(i, order='ansi')
    w = z.polynomial(128)
    a.imshow(w)
    a.set_title(z.name)
    a.axis('off')
None;


plt.show()


