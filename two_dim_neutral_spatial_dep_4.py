import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.animation as animation
import math

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

