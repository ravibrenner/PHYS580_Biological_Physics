#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:46:44 2020

@author: ravibrenner
"""

import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact

N = 10
x = np.linspace(-N,N,10000)
D = N/2

def mult(N,m):
    return ((fact(N))/(fact(N-m)*fact(m))) /(2**N) 
x_list = np.linspace(0,N,N+1)
x_vals = []
for i in x_list:
    a = mult(N,abs(i))
    x_vals.append(a)

P = (1/(np.sqrt(4*np.pi*D)))*np.exp((-x**2)/(4*D))
#x_array = np.array(x_list)-N/2
x_array = [-10,-8,-6, -4, -2, 0, 2, 4, 6, 8,10]
plt.plot(x,P, 'r')
plt.bar(x_array, x_vals)
plt.xlabel('Steps')
plt.ylabel('Probability')
plt.title('N=10')