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
