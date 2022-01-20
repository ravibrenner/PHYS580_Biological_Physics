import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.animation as animation
import math

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
