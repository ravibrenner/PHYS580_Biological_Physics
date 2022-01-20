import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.animation as animation
import math

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
