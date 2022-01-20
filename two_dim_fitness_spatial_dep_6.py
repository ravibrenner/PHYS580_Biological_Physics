import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as color
import matplotlib.animation as animation
import math

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
