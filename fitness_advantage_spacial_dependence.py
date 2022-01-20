import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.animation as animation
import math

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
