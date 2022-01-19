#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:44:56 2020

@author: ravibrenner
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.animation as animation
import math

#%%
'''(1) This is just a basic moran process, displayed on a lattice
Neutral drift, no spatial dependence'''
#Creating numbers to start off with
dim = 10
gens = range(501)
N = dim**2 #this creates a dim x dim lattice
i = 33 #Starting number of A/red/0
a = i
b = (N-i) #starting number of B/blue/1
A = [0]*a
B = [1]*b
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #probability of fixing at a given time

'''function that carries out the moran process with equal probability, and gives
lists of the new i, the probability of A to fix, and returns a list of the new population
'''
def moran(pop):           #runs the moran process, returns a new population
    x = np.random.randint(0,dim)
    y = np.random.randint(0,dim)
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        indiv = np.random.choice(np.ravel(list(pop)))
        if indiv == 0:
            pop[x,y] = 0
        if indiv == 1:
            pop[x,y] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    x_i.append(i/N)
    return pop
#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'], N = 2)

for j in gens:
    im = plt.imshow(moran(pop), cmap, vmin = 0,vmax = 1, animated = True)
    plt.title("Neutral drift w/no spatial dependence. Initial A/Red = %s" %i)
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()





plt.subplot(2,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(2,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(2,2,(3,4))
plt.plot(gens,x_i,'r')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("Neutral Drift With No Spatial Dependence, N = %s, Initial A/Red =%d" %(N,i))

#%%
'''(2) 
Fitness advantage with no spatial dependence'''

dim = 10
gens = range(301)
N = dim**2 #this creates a dim x dim lattice
i = 33 #Starting number of A
a = i
b = (N-i)
A = [0]*a
B = [1]*b
f_a = 1+.02#fitness of each species
f_b = 1
r = (f_a/f_b) 
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #list of fixation probability for A

def x(i):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def moran(pop):           
    x_1 = np.random.randint(0,dim)
    y_1 = np.random.randint(0,dim)
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        a_temp = np.count_nonzero(pop==0)
        b_temp = np.count_nonzero(pop==1)
        p_a = f_a * a_temp / (f_a * a_temp + f_b * b_temp)
        p_b = 1-p_a
        the_list = [0,1]
        indiv = np.random.choice(the_list,p = [p_a,p_b])
        if indiv == 0:
            pop[x_1,y_1] = 0
        if indiv == 1:
            pop[x_1,y_1] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i/N)
    x_i.append(x(i))
    return pop

#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'])

for j in gens:
    im = plt.imshow(moran(pop), cmap = cmap,vmin = 0,vmax = 1, animated=True)
    plt.title("Selective Advantage, Initial A/Red =%s, r = %.2f" %(i,r))
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=500)
plt.show()



plt.subplot(2,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(2,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(2,2,3)
plt.plot(gens, i_new,'r')
plt.xlabel('Number of generations')
plt.ylabel('Fraction that are red')
plt.subplot(2,2,4)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("Selective Advantage, N = %s, Initial A/Red =%d, r = %.2f" %(N,i,r))


#%%
'''(3) This will be the 1D case of neutral drift with spatial dependence on who is chosen for replacement
1D Neutral drift with spatial dependence'''
#Creating numbers to start off with
dim = 10
gens = range(101)
N = dim #this creates a dim lattice
i = 5 #Starting number of A/red/0
a = i
b = (N-i) #starting number of B/blue/1
A = [0]*a
B = [1]*b
pop = A + B
np.random.shuffle(pop)
pop_1 = np.array(pop)
pop = np.array(pop)
i_new = [] #list of the number of A's at a given time
x_i = [] #probability of fixing at a given time
p_i = [] #overall transition probabilities
p_minus = []
p_plus = []

def fix(i,r):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def trans_prob(pop_now,x): #gives normalized probabilities for each individual 
    p_i = []
    p_minus = []
    p_plus = []
    pop_now = list(pop_now)
    if x != 0 and x != (dim-1): #non-end conditions
        for j in list(range(-1,2)): 
            if pop_now[x] == 0:
                if pop_now[x+j] == 0:
                    p_i.append(1/3)
                if pop_now[x+j] == 1:
                    p_plus.append(1/3)
            if pop_now[x] == 1:
                if pop_now[x+j] == 0:
                    p_minus.append(1/3)
                if pop_now[x+j] == 1:
                    p_i.append(1/3)
    if x == 0: #end condition
        if pop_now[x] == 0:
            p_i.append(1/2)
            if pop_now[x+1] == 0:
                p_i.append(1/2)
            if pop_now[x+1] == 1:
                p_plus.append(1/2)
        if pop_now[x] == 1:
            p_i.append(1/2)
            if pop_now[x+1] == 0:
                p_minus.append(1/2)
            if pop_now[x+1] == 1:
                p_i.append(1/2)
    if x == dim - 1: #end condition
        if pop_now[x] == 0:
            p_i.append(1/2)
            if pop_now[x-1] == 0:
                p_i.append(1/2)
            if pop_now[x-1] == 1:
                p_plus.append(1/2)
        if pop_now[x] == 1:
            p_i.append(1/2)
            if pop_now[x-1] == 0:
                p_minus.append(1/2)
            if pop_now[x-1] == 1:
                p_i.append(1/2)
    p_i = sum(p_i)
    p_minus = sum(p_minus)
    p_plus = sum(p_plus)
    return [p_i/N,p_minus/N,p_plus/N]
def transition(pop_now): #gives overall probability of transition at a given time
    p_1 = []
    p_2 = []
    p_3 = []
    for k in list(range(0,dim)):
        p = trans_prob(pop_now,k)
        p_1.append(p[0])
        p_2.append(p[1])
        p_3.append(p[2])
    p_1 = sum(p_1)
    p_2 = sum(p_2)
    p_3 = sum(p_3)
    p_i.append(p_1)
    p_minus.append(p_2)
    p_plus.append(p_3)
    
            
'''function that carries out the moran process with equal probability, and gives
lists of the new i, the transtition probabilities, and returns the new population
'''
def moran(pop):           
    x = np.random.randint(0,dim)
    if x != 0 and x != dim-1:
        x_rand = np.random.choice([-1,0,1])
        x_prime = x + x_rand
    if x == dim-1:
        x_rand = np.random.choice([-1,0])
        x_prime = x + x_rand
    if x == 0:
        x_rand = np.random.choice([0,1])
        x_prime = x+x_rand
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        indiv = pop[x]
        if indiv == 0:
            pop[x_prime] = 0
        if indiv == 1:
            pop[x_prime] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    transition(pop)
    return pop
#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'], N = 2)

for j in gens:
    im = plt.imshow(np.reshape(moran(pop),[1,dim]), cmap, vmin = 0,vmax = 1, animated = True)
    plt.title("1D Spatially Dependent Drift, Initial A/Red =%s" %i)
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()

#These lines clean up any bad values in p_plus and p_minus and r  
p_plus = [x for x in p_plus if x != 'nan']
p_minus = [x for x in p_minus if x != 'nan']
r = np.array(p_plus)/np.array(p_minus)
x_i = fix(i_new,r)
for j in list(range(len(x_i))):
    if math.isnan(x_i[j]) == True:
        x_i[j] = i_new[j]/N
    if math.isnan(x_i[j]) == False:
        x_i[j] = x_i[j]


plt.subplot(3,2,1)
plt.imshow(np.reshape(pop_1,[1,dim]), cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(3,2,2)
plt.imshow(np.reshape(pop,[1,dim]),cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(3,2,3)
plt.plot(gens,np.array(i_new)/N,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Fraction that are A/Red')
plt.subplot(3,2,4)
plt.plot(gens,p_i)
plt.plot(gens,p_minus)
plt.plot(gens,p_plus)
plt.xlabel('Number of Generations')
plt.ylabel('Transition Probabilities')
plt.legend(['$P_{i,i}$','$P_{i,i-1}$','$P_{i,i+1}$'])
plt.subplot(3,2,5)
plt.plot(gens,r,'g')
plt.ylabel('r value')
plt.xlabel('Number of Generations')
plt.subplot(3,2,6)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("1D Spatially Dependent Drift, N = %s, Initial A/Red =%s" %(N,i))

#%%
'''(4) Now we'll do neutral drift, exactly the same as above, just on a square this time
2D neutral drift with spatial dependence'''
#Creating numbers to start off with
dim = 10
gens = range(301)
N = dim**2 #this creates a dim lattice
i = 33 #Starting number of A/red/0
a = i
b = (N-i) #starting number of B/blue/1
A = [0]*a
B = [1]*b
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #probability of fixing at a given time
p_i = []
p_minus = []
p_plus = []

def fix(i,r):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def trans_prob(pop_now,x,y): #gives normalized probabilities for each individual 
    p_i = []
    p_minus = []
    p_plus = []
    if x != 0 and x != (dim-1) and y != 0 and y!=(dim-1): #Center piece condition
        for j in list(range(-1,2)):
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/9)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/9)
    if x == 0 and y == 0: # corner piece conditions
        for j in list(range(0,2)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/4)
    if x == 0 and y ==dim-1: # corner piece conditions
        for j in list(range(0,2)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/4)
    if x == dim-1 and y == 0: # corner piece conditions
        for j in list(range(-1,1)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/4)
    if x == dim-1 and y == dim-1: # corner piece conditions
        for j in list(range(-1,1)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/4)
    if x == 0 and y != 0 and y != dim-1: #these ones are the edge conditions
        for j in list(range(0,2)): 
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/6)
    if x == dim-1 and y != 0 and y != dim-1: #these ones are the edge conditions
        for j in list(range(-1,1)):
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/6)
    if y == 0 and x != 0 and x != dim-1: #these ones are the edge conditions
        for j in list(range(-1,2)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/6)
    if y == dim-1 and x != 0 and x != dim-1: #these ones are the edge conditions
        for j in list(range(-1,2)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(1/6)
    p_i = sum(p_i)
    p_minus = sum(p_minus)
    p_plus = sum(p_plus)
    
    return [p_i/N,p_minus/N,p_plus/N]
def transition(pop_now): #gives overall probability of transition at a given time
    p_1 = []
    p_2 = []
    p_3 = []
    for k in list(range(0,dim)):
        for l in list(range(0,dim)):
            p = trans_prob(pop_now,k,l)
            p_1.append(p[0])
            p_2.append(p[1])
            p_3.append(p[2])
    p_1 = sum(p_1)
    p_2 = sum(p_2)
    p_3 = sum(p_3)
    p_i.append(p_1)
    p_minus.append(p_2)
    p_plus.append(p_3)
    
            
'''function that carries out the moran process with equal probability, and gives
lists of the new i, the transition porbabilities, and returns the new population
'''
def moran(pop):     
    x = np.random.randint(0,dim)
    if x != 0 and x != dim-1:
        x_rand = np.random.choice([-1,0,1])
        x_prime = x + x_rand
    if x == dim-1:
        x_rand = np.random.choice([-1,0])
        x_prime = x + x_rand
    if x == 0:
        x_rand = np.random.choice([0,1])
        x_prime = x+x_rand   
    y = np.random.randint(0,dim)
    if y != 0 and y != dim-1:
        y_rand = np.random.choice([-1,0,1])
        y_prime = y + y_rand
    if y == dim-1:
        y_rand = np.random.choice([-1,0])
        y_prime = y + y_rand
    if y == 0:
        y_rand = np.random.choice([0,1])
        y_prime = y+y_rand
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        indiv = pop[x,y]
        if indiv == 0:
            pop[x_prime,y_prime] = 0
        if indiv == 1:
            pop[x_prime,y_prime] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    transition(pop)
    return pop
#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'], N = 2)

for j in gens:
    im = plt.imshow(moran(pop), cmap, vmin = 0,vmax = 1, animated = True)
    plt.title("2D Spatially Dependent Drift, Initial A/Red =%s" %i)
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()

#replacing bad values with ones that make sense
p_plus = [x for x in p_plus if x != 'nan']
p_minus = [x for x in p_minus if x != 'nan']
r = np.array(p_plus)/np.array(p_minus)
x_i = fix(i_new,r)
for j in list(range(len(x_i))):
    if math.isnan(x_i[j]) == True:
        x_i[j] = i_new[j]/N
    if math.isnan(x_i[j]) == False:
        x_i[j] = x_i[j]

plt.subplot(3,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(3,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(3,2,3)
plt.plot(gens,np.array(i_new)/N,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Fraction that are A/Red')
plt.subplot(3,2,4)
plt.plot(gens,p_i)
plt.plot(gens,p_minus)
plt.plot(gens,p_plus)
plt.xlabel('Number of Generations')
plt.ylabel('Transition Probabilities')
plt.legend(['$P_{i,i}$','$P_{i,i-1}$','$P_{i,i+1}$'])
plt.subplot(3,2,5)
plt.plot(gens,r,'g')
plt.ylabel('r value')
plt.xlabel('Number of Generations')
plt.subplot(3,2,6)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("1D Spatially Dependent Drift, N = %s, Initial A/Red =%s" %(N,i))


#%%
'''(5) 1D fitness advantage with spatial dependence'''
#Creating numbers to start off with
dim = 10
gens = range(101)
N = dim #this creates a dim lattice
i = 5 #Starting number of A/red/0
a = i
b = (N-i) #starting number of B/blue/1
A = [0]*a
B = [1]*b
pop = A + B
np.random.shuffle(pop)
pop_1 = np.array(pop)
pop = np.array(pop)
f_a = 1.02 #fitness of each species
f_b = 1
r_val = f_a/f_b #selective r value, not the same as the overall r value which updates at every time step
radius = 1
i_new = [] #list of the number of A's at a given time
x_i = [] #probability of fixing at a given time
p_i = []
p_minus = []
p_plus = []


def fix(i,r):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def trans_prob(pop_now,x): #gives normalized probabilities for each individual 
    p_i = []
    p_minus = []
    p_plus = []
    pop_now = list(pop_now)
    l_1 = list(range(-radius,radius+1))
    if x != 0 and x != (dim-1): #non-end conditions
        for j in l_1: 
            if pop_now[x] == 0:
                if pop_now[x+j] == 0:
                    p_i.append(chooser(pop)[1][x] * 1/3)
                if pop_now[x+j] == 1:
                    p_plus.append(chooser(pop)[1][x] * 1/3)
            if pop_now[x] == 1:
                if pop_now[x+j] == 0:
                    p_minus.append(chooser(pop)[1][x] * 1/3)
                if pop_now[x+j] == 1:
                    p_i.append(chooser(pop)[1][x] * 1/3)
    if x == 0: #end conditions
        if pop_now[x] == 0:
            p_i.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x+1] == 0:
                p_i.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x+1] == 1:
                p_plus.append(chooser(pop)[1][x] * 1/2)
        if pop_now[x] == 1:
            p_i.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x+1] == 0:
                p_minus.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x+1] == 1:
                p_i.append(chooser(pop)[1][x] * 1/2)
    if x == dim - 1: #end conditions
        if pop_now[x] == 0:
            p_i.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x-1] == 0:
                p_i.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x-1] == 1:
                p_plus.append(chooser(pop)[1][x] * 1/2)
        if pop_now[x] == 1:
            p_i.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x-1] == 0:
                p_minus.append(chooser(pop)[1][x] * 1/2)
            if pop_now[x-1] == 1:
                p_i.append(chooser(pop)[1][x] * 1/2)
    p_i = sum(p_i)
    p_minus = sum(p_minus)
    p_plus = sum(p_plus)
    
    return [p_i,p_minus,p_plus]
def transition(pop_now): #gives overall probability of transition at a given time
    p_1 = []
    p_2 = []
    p_3 = []
    for k in list(range(0,dim)):
        p = trans_prob(pop_now,k)
        p_1.append(p[0])
        p_2.append(p[1])
        p_3.append(p[2])
    p_1 = sum(p_1)
    p_2 = sum(p_2)
    p_3 = sum(p_3)
    p_i.append(p_1)
    p_minus.append(p_2)
    p_plus.append(p_3)
    
def chooser(pop): #this will return the coordinate of either an A or a B according to the probability defined by s and the function f(x)    
    fitness = []
    a_temp = np.count_nonzero(pop==0)
    b_temp = np.count_nonzero(pop==1)
    denom = f_a *a_temp + f_b*b_temp
    for i in list(range(0,dim)):
        if pop[i] == 0:
            fitness.append(f_a)
        if pop[i] == 1:
            fitness.append(f_b)
    fitness = np.array(fitness)/denom
    fudge = 1 - sum(fitness)
    fitness[0] = fitness[0] + fudge
    indiv = np.random.choice(list(range(0,N)),p = fitness)
    x = np.unravel_index(indiv,[dim,1])[0]
    return x,fitness      
'''function that carries out the moran process with equal probability, and gives
lists of the new i, the probability of A to fix, and returns a list of the new population
'''
def moran(pop):           
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        x = chooser(pop)[0]
        if x != 0 and x != dim-1:
            x_rand = np.random.choice([-1,0,1])
            x_prime = x + x_rand
        if x == 0:
            x_rand = np.random.choice([0,1])
            x_prime = x+x_rand
        if x == dim-1:
            x_rand = np.random.choice([-1,0])
            x_prime = x+x_rand
        if pop[x] == 0:
            pop[x_prime] = 0
        if pop[x] == 1:
            pop[x_prime] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    transition(pop)
    return pop
#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'], N = 2)

for j in gens:
    im = plt.imshow(np.reshape(moran(pop),[1,dim]), cmap, vmin = 0,vmax = 1, animated = True)
    plt.title("1D Spatially Dependent Selection, Initial A/Red =%s, $r_{selection}$ =%.2f" %(i,r_val))
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()

p_plus = [x for x in p_plus if x != 'nan']
p_minus = [x for x in p_minus if x != 'nan']
r = np.array(p_plus)/np.array(p_minus)
x_i = fix(i_new,r)
for j in list(range(len(x_i))):
    if math.isnan(x_i[j]) == True:
        x_i[j] = i_new[j]/N
    if math.isnan(x_i[j]) == False:
        x_i[j] = x_i[j]

plt.subplot(3,2,1)
plt.imshow(np.reshape(pop_1,[1,dim]), cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(3,2,2)
plt.imshow(np.reshape(pop,[1,dim]),cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(3,2,(3))
plt.plot(gens,np.array(i_new)/N,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Fraction that are A/red')
plt.subplot(3,2,4)
plt.plot(gens,p_i)
plt.plot(gens,p_minus)
plt.plot(gens,p_plus)
plt.ylabel('Transition Probabilities')
plt.xlabel('Number of Generations')
plt.legend(['$P_{i,i}$','$P_{i,i-1}$','$P_{i,i+1}$'])
plt.subplot(3,2,5)
plt.plot(gens,r,'g')
plt.xlabel('Number of Generations')
plt.ylabel('r value')
plt.subplot(3,2,6)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("1D Spatially Dependent Selection, Initial A/Red =%s, $r_{selection}$ =%.2f" %(i,r_val))

#%%
'''(6) This is to go in section 4.2 of the paper
2D fitness advantage with spatial dependence'''
dim = 8
gens = range(101)
N = dim**2 #this creates a dim x dim lattice
i = 21 #Starting number of A
a = i
b = (N-i)
A = [0]*a
B = [1]*b
f_a = 1 + .02
f_b = 1 # this is the selective advantage of A, done as f_a = 1+s
r_val = f_a/f_b
radius = 1
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #list of fixation probability for A
r_list = []
p_i = []
p_minus = []
p_plus = []

def fix(i,r):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def trans_prob(pop_now,x,y): #gives normalized probabilities for each individual 
    p_i = []
    p_minus = []
    p_plus = []
    l_1 = list(range(-radius,radius+1))
    l_2 = list(range(0,radius+1))
    l_3 = list(range(-radius,1))
    if x != 0 and x != (dim-1) and y != 0 and y!=(dim-1): #Center piece condition
        for j in list(range(-1,2)):
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/9)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] *  1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/9)
    if x == 0 and y == 0: #these ones are the corner piece conditions
        for j in list(range(0,2)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == 0 and y ==dim-1: #these ones are the corner piece conditions
        for j in list(range(0,2)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == dim-1 and y == 0: #these ones are the corner piece conditions
        for j in list(range(-1,1)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == dim-1 and y == dim-1: #these ones are the corner piece conditions
        for j in list(range(-1,1)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == 0 and y != 0 and y != dim-1: #these ones are the edge conditions
        for j in list(range(0,2)): 
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    if x == dim-1 and y != 0 and y != dim-1: #these ones are the edge conditions
        for j in list(range(-1,1)):
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    if y == 0 and x != 0 and x != dim-1: #these ones are the edge conditions
        for j in list(range(-1,2)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    if y == dim-1 and x != 0 and x != dim-1: #these ones are the edge conditions
        for j in list(range(-1,2)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    p_i = sum(p_i)
    p_minus = sum(p_minus)
    p_plus = sum(p_plus)
    
    return [p_i,p_minus,p_plus]
def transition(pop_now): #gives overall probability of transition at a given time
    p_1 = []
    p_2 = []
    p_3 = []
    for k in list(range(0,dim)):
        for l in list(range(0,dim)):
            p = trans_prob(pop_now,k,l)
            p_1.append(p[0])
            p_2.append(p[1])
            p_3.append(p[2])
    p_1 = sum(p_1)
    p_2 = sum(p_2)
    p_3 = sum(p_3)
    p_i.append(p_1)
    p_minus.append(p_2)
    p_plus.append(p_3)

def chooser(pop): #this will return the coordinate of either an A or a B according to the probability defined by s and the function f(x)
    denom = []     
    fitness = []
    a_temp = np.count_nonzero(pop==0)
    b_temp = np.count_nonzero(pop==1)
    for i in list(range(0,dim)):
        for j in list(range(0,dim)):
            if pop[i,j] == 0:
                denom.append(f_a)
                fitness.append(f_a)
            if pop[i,j] == 1:
                denom.append(f_b)
                fitness.append(f_b)
    denom = sum(denom)
    fitness = np.array(fitness)/denom
    fudge = 1 - sum(fitness)
    fitness[0] = fitness[0] + fudge
    indiv = np.random.choice(list(range(0,N)),p = fitness)
    x_ = np.unravel_index(indiv,[dim,dim])[0]
    y_ = np.unravel_index(indiv,[dim,dim])[1]
    fitness = np.reshape(fitness,[dim,dim])
    return x_,y_,fitness

def moran(pop):           
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        x,y = chooser(pop)[0], chooser(pop)[1]
        if x != 0 and x != dim-1:
            x_rand = np.random.choice([-1,0,1])
            x_prime = x + x_rand
        if x == dim-1:
            x_rand = np.random.choice([-1,0])
            x_prime = x + x_rand
        if x == 0:
            x_rand = np.random.choice([0,1])
            x_prime = x+x_rand   
        if y != 0 and y != dim-1:
            y_rand = np.random.choice([-1,0,1])
            y_prime = y + y_rand
        if y == dim-1:
            y_rand = np.random.choice([-1,0])
            y_prime = y + y_rand
        if y == 0:
            y_rand = np.random.choice([0,1])
            y_prime = y+y_rand
        if pop[x,y] == 0: 
            pop[x_prime,y_prime] = 0
        if pop[x,y] == 1:
            pop[x_prime,y_prime] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    transition(pop)
    return pop

#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'])

for j in gens:
    im = plt.imshow(moran(pop), cmap = cmap,vmin = 0,vmax = 1, animated=True)
    plt.title("2D Spatially Dependent Selection, Initial A/Red =%s, $r_{selection}$ =%.2f" %(i,r_val))
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()

p_plus = [x for x in p_plus if x != 'nan']
p_minus = [x for x in p_minus if x != 'nan']
r = np.array(p_plus)/np.array(p_minus)
x_i = fix(np.array(i_new),np.array(r))
for j in list(range(len(x_i))):
    if math.isnan(x_i[j]) == True:
        x_i[j] = i_new[j]/N
    if math.isnan(x_i[j]) == False:
        x_i[j] = x_i[j]

plt.subplot(3,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(3,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(3,2,(3))
plt.plot(gens,np.array(i_new)/N,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Fraction that are A/red')
plt.subplot(3,2,4)
plt.plot(gens,p_i)
plt.plot(gens,p_minus)
plt.plot(gens,p_plus)
plt.xlabel('Number of Generations')
plt.ylabel('Transition Probabilities')
plt.legend(['$P_{i,i}$','$P_{i,i-1}$','$P_{i,i+1}$'])
plt.subplot(3,2,5)
plt.plot(gens,r,'g')
plt.xlabel('Number of Generations')
plt.ylabel('r value')
plt.subplot(3,2,6)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("1D Spatially Dependent Selection, Initial A/Red =%s, $r_{selection}$ =%.2f" %(i,r_val))
#plt.ylim((.9,1))

#%%
'''(7)
2D fitness advantage with spatial dependence on who is chosen to reproduce and who is chosen for replacement'''
dim = 8
gens = range(101)
N = dim**2 #this creates a dim x dim lattice
i = 21 #Starting number of A
a = i
b = (N-i)
A = [0]*a
B = [1]*b
s = 1 # this gives the selective advantage of A, done as f_a = 1+s see function f(x,y) below
radius = 1
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #list of fixation probability for A
r_list = []
p_i = []
p_minus = []
p_plus = []
def f(x,y): #this needs to be a well defined function on the interval [0:dim]. It will get normalized later
    f = 1 + s * (1/((x+1)**2) + 1/((y+1)**2))
    return f

def r(pop):
    f_a = [] #fitness of each species
    f_b = 1
    for i in list(range(0,dim)):
        for j in list(range(0,dim)):
            if pop[i,j] == 0:
                val = f(i,j)
                f_a.append(val)
    f_a = np.mean(f_a)
    r = (f_a/f_b) 
    return(r)

def fix(i,r):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def trans_prob(pop_now,x,y): #gives normalized probabilities for each individual 
    p_i = []
    p_minus = []
    p_plus = []
    l_1 = list(range(-radius,radius+1))
    l_2 = list(range(0,radius+1))
    l_3 = list(range(-radius,1))
    if x != 0 and x != (dim-1) and y != 0 and y!=(dim-1): #Center piece condition
        for j in list(range(-1,2)):
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/9)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] *  1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/9)
    if x == 0 and y == 0: #these ones are the corner piece conditions
        for j in list(range(0,2)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == 0 and y ==dim-1: #these ones are the corner piece conditions
        for j in list(range(0,2)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == dim-1 and y == 0: #these ones are the corner piece conditions
        for j in list(range(-1,1)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == dim-1 and y == dim-1: #these ones are the corner piece conditions
        for j in list(range(-1,1)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/4)
    if x == 0 and y != 0 and y != dim-1: #these ones are the edge conditions
        for j in list(range(0,2)): 
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    if x == dim-1 and y != 0 and y != dim-1: #these ones are the edge conditions
        for j in list(range(-1,1)):
            for k in list(range(-1,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    if y == 0 and x != 0 and x != dim-1: #these ones are the edge conditions
        for j in list(range(-1,2)):
            for k in list(range(0,2)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    if y == dim-1 and x != 0 and x != dim-1: #these ones are the edge conditions
        for j in list(range(-1,2)):
            for k in list(range(-1,1)):
                if pop_now[x,y] == 0:
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
                if pop_now[x,y] == 1:
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/6)
    p_i = sum(p_i)
    p_minus = sum(p_minus)
    p_plus = sum(p_plus)
    
    return [p_i,p_minus,p_plus]
def transition(pop_now): #gives overall probability of transition at a given time
    p_1 = []
    p_2 = []
    p_3 = []
    for k in list(range(0,dim)):
        for l in list(range(0,dim)):
            p = trans_prob(pop_now,k,l)
            p_1.append(p[0])
            p_2.append(p[1])
            p_3.append(p[2])
    p_1 = sum(p_1)
    p_2 = sum(p_2)
    p_3 = sum(p_3)
    p_i.append(p_1)
    p_minus.append(p_2)
    p_plus.append(p_3)

def chooser(pop): #this will return the coordinate of either an A or a B according to the probability defined by s and the function f(x,y)
    denom = []     
    fitness = []
    a_temp = np.count_nonzero(pop==0)
    b_temp = np.count_nonzero(pop==1)
    for i in list(range(0,dim)):
        for j in list(range(0,dim)):
            if pop[i,j] == 0:
                denom.append(f(i,j))
                fitness.append(f(i,j))
            if pop[i,j] == 1:
                denom.append(1)
                fitness.append(1)
    denom = sum(denom)
    fitness = np.array(fitness)/denom
    fudge = 1 - sum(fitness)
    fitness[0] = fitness[0] + fudge
    indiv = np.random.choice(list(range(0,N)),p = fitness)
    x_ = np.unravel_index(indiv,[dim,dim])[0]
    y_ = np.unravel_index(indiv,[dim,dim])[1]
    fitness = np.reshape(fitness,[dim,dim])
    return x_,y_,fitness

def moran(pop):           
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        x,y = chooser(pop)[0], chooser(pop)[1]
        if x != 0 and x != dim-1:
            x_rand = np.random.choice([-1,0,1])
            x_prime = x + x_rand
        if x == dim-1:
            x_rand = np.random.choice([-1,0])
            x_prime = x + x_rand
        if x == 0:
            x_rand = np.random.choice([0,1])
            x_prime = x+x_rand   
        if y != 0 and y != dim-1:
            y_rand = np.random.choice([-1,0,1])
            y_prime = y + y_rand
        if y == dim-1:
            y_rand = np.random.choice([-1,0])
            y_prime = y + y_rand
        if y == 0:
            y_rand = np.random.choice([0,1])
            y_prime = y+y_rand
        if pop[x,y] == 0: 
            pop[x_prime,y_prime] = 0
        if pop[x,y] == 1:
            pop[x_prime,y_prime] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    transition(pop)
    return pop

#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'])

for j in gens:
    im = plt.imshow(moran(pop), cmap = cmap,vmin = 0,vmax = 1, animated=True)
    plt.title("Spatially Dependent Selection 2 ways, Initial A/Red =%s, $r_{selection} = f(r,R)$" %i)
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()

p_plus = [x for x in p_plus if x != 'nan']
p_minus = [x for x in p_minus if x != 'nan']
r = np.array(p_plus)/np.array(p_minus)
x_i = fix(np.array(i_new),np.array(r))
for j in list(range(len(x_i))):
    if math.isnan(x_i[j]) == True:
        x_i[j] = i_new[j]/N
    if math.isnan(x_i[j]) == False:
        x_i[j] = x_i[j]

plt.subplot(3,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(3,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(3,2,(3))
plt.plot(gens,np.array(i_new)/N,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Fraction that are A/red')
plt.subplot(3,2,4)
plt.plot(gens,p_i)
plt.plot(gens,p_minus)
plt.plot(gens,p_plus)
plt.xlabel('Number of Generations')
plt.ylabel('Transition Probabilities')
plt.legend(['$P_{i,i}$','$P_{i,i-1}$','$P_{i,i+1}$'])
plt.subplot(3,2,5)
plt.plot(gens,r,'g')
plt.xlabel('Number of Generations')
plt.ylabel('r value')
plt.subplot(3,2,6)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('$x_i$')
plt.suptitle("Spatially Dependent Selection 2 ways, Initial A/Red =%s, $r_{selection} = f(r,R)$" %i)
#plt.ylim((.9,1))

#%%
'''(8) This is going to be the zombie apocalypse part--Selective advantage for A, increased radius of reproduction for B
Unfortunately, the way I've coded it, it has to be this way
Varying radius of reproduction and seeing how that changes things'''
dim = 6
gens = range(101)
N = dim**2 #this creates a dim x dim lattice
i = 3 #Starting number of A
a = i
b = (N-i)
A = [0]*a
B = [1]*b
f_a = 10
f_b = 1 
radius = 3 #Radius must be <= dim/2 by the way this code is constructed
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #list of fixation probability for A
r_list = []
p_i = []
p_minus = []
p_plus = []


def fix(i,r):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def trans_prob(pop_now,x,y): #gives normalized probabilities for each individual 
    p_i = []
    p_minus = []
    p_plus = []
    l_1 = list(range(-radius,radius+1))
    l_2 = list(range(0,radius+1))
    l_3 = list(range(-radius,1))
    if pop_now[x,y] == 0:
        if x != 0 and x != (dim-1) and y != 0 and y!=(dim-1): #Center piece condition
            for j in list(range(-1,2)):
                for k in list(range(-1,2)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] * 1/9)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/9)
        if x == 0 and y == 0: #these ones are the corner piece conditions
            for j in list(range(0,2)):
                for k in list(range(0,2)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
        if x == 0 and y ==dim-1: #these ones are the corner piece conditions
            for j in list(range(0,2)):
                for k in list(range(-1,1)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
        if x == dim-1 and y == 0: #these ones are the corner piece conditions
            for j in list(range(-1,1)):
                for k in list(range(0,2)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
        if x == dim-1 and y == dim-1: #these ones are the corner piece conditions
            for j in list(range(-1,1)):
                for k in list(range(-1,1)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/4)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/4)
        if x == 0 and y != 0 and y != dim-1: #these ones are the edge conditions
            for j in list(range(0,2)):
                for k in list(range(-1,2)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
        if x == dim-1 and y != 0 and y != dim-1: #these ones are the edge conditions
            for j in list(range(-1,1)):
                for k in list(range(-1,2)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
        if y == 0 and x != 0 and x != dim-1: #these ones are the edge conditions
            for j in list(range(-1,2)):
                for k in list(range(0,2)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
        if y == dim-1 and x != 0 and x != dim-1: #these ones are the edge conditions
            for j in list(range(-1,2)):
                for k in list(range(-1,1)):
                    if pop_now[x+j,y+k] == 0:
                        p_i.append(chooser(pop)[2][x,y] *  1/6)
                    if pop_now[x+j,y+k] == 1:
                        p_plus.append(chooser(pop)[2][x,y] * 1/6)
    if pop_now[x,y] == 1:
        if dim-1-x >= radius and x >= radius and dim-1-y >= radius and y >=radius: #center condition
            for j in l_1:
                for k in l_1:
                    tot = len(l_1) * len(l_1)
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if dim-1-x >= radius and x >= radius and y < radius: #non-center condition
            for j in list(range(-radius,radius+1)):
                for k in list(range(-y,radius+1)):
                    tot = len(list(range(-radius,radius+1))) * len(list(range(-y,radius+1)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if dim-1-x >= radius and x >= radius and dim-1-y < radius: #non-center condition
            for j in list(range(-radius,radius+1)):
                for k in list(range(-radius,dim-y)):
                    tot = len(list(range(-radius,radius+1))) * len(list(range(-radius,dim-y)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if x < radius and dim-1-y >= radius and y >= radius: #non-center condition
            for j in list(range(-x,radius+1)):
                for k in list(range(-radius,radius+1)):
                    tot = len(list(range(-x,radius+1))) * len(list(range(-radius,radius+1)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if x < radius and y < radius: #non-center condition
            for j in list(range(-x,radius+1)):
                for k in list(range(-y,radius+1)):
                    tot = len(list(range(-x,radius+1))) * len(list(range(-y,radius+1)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if x < radius and dim-1-y < radius: #non-center condition
            for j in list(range(-x,radius+1)):
                for k in list(range(-radius,dim-y)):
                    tot = len(list(range(-x,radius+1))) * len(list(range(-radius,dim-y)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)    
        if dim-1-x < radius and dim-1-y >= radius and x >= radius: #non-center condition
            for j in list(range(-radius,dim-x)):
                for k in list(range(-radius,radius+1)):
                    tot = len(list(range(-radius,dim-x))) * len(list(range(-radius,radius+1)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if dim-1-x < radius and y < radius: #non-center condition
            for j in list(range(-radius,dim-x)):
                for k in list(range(-y,radius+1)):
                    tot = len(list(range(-radius,dim-x))) * len(list(range(-y,radius+1)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
        if dim-1-x < radius and dim-1-y < radius: #non-center condition
            for j in list(range(-radius,dim-x)):
                for k in list(range(-radius,dim-y)):
                    tot = len(list(range(-radius,dim-x))) * len(list(range(-radius,dim-y)))
                    if pop_now[x+j,y+k] == 0:
                        p_minus.append(chooser(pop)[2][x,y] * 1/tot)
                    if pop_now[x+j,y+k] == 1:
                        p_i.append(chooser(pop)[2][x,y] * 1/tot)
    p_i = sum(p_i)
    p_minus = sum(p_minus)
    p_plus = sum(p_plus)
    
    return [p_i,p_minus,p_plus]
def transition(pop_now): #gives overall probability of transition at a given time
    p_1 = []
    p_2 = []
    p_3 = []
    for k in list(range(0,dim)):
        for l in list(range(0,dim)):
            p = trans_prob(pop_now,k,l)
            p_1.append(p[0])
            p_2.append(p[1])
            p_3.append(p[2])
    p_1 = sum(p_1)
    p_2 = sum(p_2)
    p_3 = sum(p_3)
    p_i.append(p_1)
    p_minus.append(p_2)
    p_plus.append(p_3)

def chooser(pop): #this will return the coordinate of either an A or a B according to the probability defined by s and the function f(x)
    denom = []     
    fitness = []
    a_temp = np.count_nonzero(pop==0)
    b_temp = np.count_nonzero(pop==1)
    for i in list(range(0,dim)):
        for j in list(range(0,dim)):
            if pop[i,j] == 0:
                denom.append(f_a)
                fitness.append(f_a)
            if pop[i,j] == 1:
                denom.append(f_b)
                fitness.append(f_b)
    denom = sum(denom)
    fitness = np.array(fitness)/denom
    fudge = 1 - sum(fitness)
    fitness[0] = fitness[0] + fudge
    indiv = np.random.choice(list(range(0,N)),p = fitness)
    x_ = np.unravel_index(indiv,[dim,dim])[0]
    y_ = np.unravel_index(indiv,[dim,dim])[1]
    fitness = np.reshape(fitness,[dim,dim])
    return x_,y_,fitness



def moran(pop):           
    if np.count_nonzero(pop ==0) == N:
        pop = pop
    if np.count_nonzero(pop ==1) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N and np.count_nonzero(pop ==1) != N:
        x,y = chooser(pop)[0], chooser(pop)[1]
        if pop[x,y] == 0:
            if x != 0 and x != dim-1:
                x_rand = np.random.choice([-1,0,1])
                x_prime = x + x_rand
            if x == dim-1:
                x_rand = np.random.choice([-1,0])
                x_prime = x + x_rand
            if x == 0:
                x_rand = np.random.choice([0,1])
                x_prime = x+x_rand   
            if y != 0 and y != dim-1:
                y_rand = np.random.choice([-1,0,1])
                y_prime = y + y_rand
            if y == dim-1:
                y_rand = np.random.choice([-1,0])
                y_prime = y + y_rand
            if y == 0:
                y_rand = np.random.choice([0,1])
                y_prime = y+y_rand
            pop[x_prime,y_prime] = 0
        if pop[x,y] == 1:
            if dim-1-x >= radius and x >= radius:
                x_rand = np.random.choice(list(range(-radius,radius+1)))
                x_prime = x + x_rand   
            if x < radius:
                x_rand = np.random.choice(list(range(-x,radius+1)))
                x_prime = x + x_rand
            if dim-1-x < radius:
                x_rand = np.random.choice(list(range(-radius,dim-x)))
                x_prime = x + x_rand
            if dim-1-y >= radius and y >= radius:
                y_rand = np.random.choice(list(range(-radius,radius+1)))
                y_prime = y + y_rand
            if y < radius:
                y_rand = np.random.choice(list(range(-y,radius+1)))
                y_prime = y + y_rand
            if dim-1-y < radius:
                y_rand = np.random.choice(list(range(-radius,dim-y)))
                y_prime = y+y_rand
            pop[x_prime,y_prime] = 1
    i = np.count_nonzero(pop == 0)
    i_new.append(i)
    transition(pop)
    return pop

#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'])

for j in gens:
    im = plt.imshow(moran(pop), cmap = cmap,vmin = 0,vmax = 1, animated=True)
    plt.title("Zombie apocalypse, Initial A/Red =%s, A $r_{selection}$ = %.2f \n B radius = %d" %(i,f_a,radius))
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()

p_plus = [x for x in p_plus if x != 'nan']
p_minus = [x for x in p_minus if x != 'nan']
r = np.array(p_plus)/np.array(p_minus)
x_i = fix(np.array(i_new),np.array(r))
for j in list(range(len(x_i))):
    if math.isnan(x_i[j]) == True:
        x_i[j] = i_new[j]/N
    if math.isnan(x_i[j]) == False:
        x_i[j] = x_i[j]

plt.subplot(3,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(3,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(3,2,(3))
plt.plot(gens,np.array(i_new)/N,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Fraction that are A/red')
plt.subplot(3,2,4)
plt.plot(gens,p_i)
plt.plot(gens,p_minus)
plt.plot(gens,p_plus)
plt.xlabel('Number of Generations')
plt.ylabel('Transition Probabilities')
plt.legend(['$P_{i,i}$','$P_{i,i-1}$','$P_{i,i+1}$'])
plt.subplot(3,2,5)
plt.plot(gens,r,'g')
plt.xlabel('Number of Generations')
plt.ylabel('r value')
plt.subplot(3,2,6)
plt.plot(gens,x_i,'k')
plt.xlabel('Number of Generations')
plt.ylabel('P$x_i$')
plt.suptitle("Zombie apocalypse, Initial A/Red =%s, A $r_{selection}$ = %.2f \n B radius = %d" %(i,f_a,radius))
#plt.ylim((.9,1))




#%%
'''(9) Here we're going to start will a population of all B/blue/1, and there will be a mutation
rate u that will give rise to some A/red/0. Here both will have equal fitness to start off,
so there will be no advantage. In this case, A could fix but B could not (there are no A-->B transitions)
MUTATION'''
dim = 10
gens = range(1001)
N = dim**2 #this creates a dim x dim lattice
i = 0 #Starting number of A
a = i
b = (N-i)
A = [0]*a
B = [1]*b
f_a = 1#fitness of each species
f_b = 1
r = (f_a/f_b) 
u = .01 #mutation rate for A to randomly appear, number between 0 and 1
pop = A + B
np.random.shuffle(pop)
pop_1 = np.reshape(pop,[dim,dim])
pop = np.reshape(pop,[dim,dim])
i_new = [] #list of the number of A's at a given time
x_i = [] #list of fixation probability for A

def x(i):  #this is defining the fixation probability for A at any given time
    return (1-(1/(r**i)))/(1-(1/(r**N)))

def moran(pop):           
    x_1 = np.random.randint(0,dim)
    y_1 = np.random.randint(0,dim)
    x_2 = np.random.randint(0,dim)
    y_2 = np.random.randint(0,dim)
    z = np.random.random()
    if np.count_nonzero(pop == 0) == N:
        pop = pop
    if np.count_nonzero(pop ==0) != N:
        if pop[x_1,y_1] == 1 and z <= u:
            pop[x_2,y_2] = 0
        if pop[x_1,y_1] == 1 and z > u:
            pop[x_2,y_2] = 1
        if pop[x_1,y_1] == 0:
            pop[x_2,y_2] = 0
    i = np.count_nonzero(pop == 0)
    i_new.append(i/N)
    x_i.append(i/N)
    return pop

#Creating an animation of the populations where  0/A = red and 1/B = blue
ims = []
cmap = color.ListedColormap(['r', 'b'])

for j in gens:
    im = plt.imshow(moran(pop), cmap = cmap,vmin = 0,vmax = 1, animated=True)
    ims.append([im])
fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
plt.show()



plt.subplot(2,2,1)
plt.imshow(pop_1, cmap,vmin = 0,vmax = 1)
plt.title('Initial population')
plt.subplot(2,2,2)
plt.imshow(pop,cmap,vmin = 0,vmax = 1)
plt.title('Final Population')
plt.subplot(2,2,(3,4))
plt.plot(gens,x_i,'r')
plt.xlabel('Number of Generations')
plt.ylabel('Probability of fixation')
plt.legend(['Red'])