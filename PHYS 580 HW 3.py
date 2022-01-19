#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:30:37 2020

@author: ravibrenner
"""

import numpy as np
import matplotlib.pyplot as plt

eps = 10

def ceil(x):
    if x == int:
        return eps*x
    elif x != int :
        return eps*(int(x))

x = np.linspace(0,10,1000) 
f = []
for i in x:
    q = i - ceil(i)
    f.append(q)
f = np.array(f)
v = -1* f**2 * ((np.exp(-f) - 1 - f)**(-1))

plt.plot(abs(f),v, 'r')

plt.xlabel('f arbitrary units')
plt.ylabel('v_drift')
plt.title('epsilon/kT = 10')
plt.figure()
plt.plot(x,f,'r')
plt.xlabel('x')
plt.ylabel('f')
plt.title('epsilon/kT = 10')

#%%
eps = 1
x = np.linspace(0,10,1000) 
f = []
for i in x:
    q = i - ceil(i)
    f.append(q)
f = np.array(f)
v = -1* f**2 * ((np.exp(-f) - 1 - f)**(-1))
plt.figure()
plt.plot(f,v, 'b')
plt.xlabel('f arbitrary units')
plt.ylabel('v_drift')
plt.title('epsilon/kT = 1')
plt.figure()
plt.plot(x,f,'b')
plt.xlabel('x')
plt.ylabel('f')
plt.title('epsilon/kT = 1')