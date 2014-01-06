#!python
"""
peabox tutorial 
lesson 04 - my first EA homebrew
a) let's make a concoction of simulated annealing and genetic algorithm

test function: CEC-2005 test function 11 --> Weierstrass function in 10 dimensions

what is to be shown here:

 - peabox = quick and easy implementation of new EA idea
 - The code is readable, well at least sort of, meaning you can qualitatively
   grasp in a couple seconds what happens when you look at the core part, the
   generation loop (the codes made with peabox perhaps won't win a speed race,
   but they don't need to fear a readability contest with codes based on most
   other EA libraries out there. Because of the easy readability of the code I
   don't need to write a lot here explaining how exactly I happened to imagine
   bringing SA and GA concepts together, how the concepts interfere in detail.
   Just look at the code below, in particular the generation loop in part 3)

for you to tinker:

 - set the elite size to something above zero
 - turn the exponential decay of mutation step size on or off
 - change the SA/GA ratio

... but don't waste too much time varying parameters here, because in the code
of lesson 4b it will get much easier and the population development plot will
become much more telling.

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
from numpy.random import seed, rand, randn, randint
from numpy import array, arange, asfarray, exp, pi, cos, zeros, where, linspace
import matplotlib as mpl
import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population
from peabox_testfuncs import CEC05_test_function_producer as tfp
from peabox_helpers import parentselect_exp as pse
from peabox_helpers import simple_searchspace

#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

dim=10
f11=tfp(11,dim)   # Weierstrass function in 10 dimensions

searchspace=simple_searchspace(10,-3.,1.)

ps=40    # population size
G=80     # number of generations to go through
parents=Population(Individual,ps,f11,searchspace)
offspring=Population(Individual,ps,f11,searchspace)

#-------------------------------------------------------------------------------
#--- part 2: initialisation ----------------------------------------------------
#-------------------------------------------------------------------------------

seed(1)  # seeding numpy's random number generator so we get reproducibly the same random numbers
parents.new_random_genes()
parents.eval_all()
parents.sort()
parents.update_no()
sa_T=parents[-1].score-parents[0].score # starting temperature
saga_ratio=0.5 # fraction of offspring created by SA and not GA
sa_mstep=0.02 # mutation step size parameter
sa_mprob=0.6 # mutation probability
ga_selp=3. # selection pressure
AE=0.04 # annealing exponent --> exp(-AE) is multiplier for reducing temperature
elite_size=0

g_rec=[]  # for recording x-data for plot, i.e. generation
bs_rec=[]  # for recording y-data for plot, here best score
ws_rec=[]  # for recording y-data for plot, here worst score
T_rec=[]  # for recording y-data for plot, here temperature
ms_rec=[]  # for recording y-data for plot, here mutation step size

# saving initial generation characteristics
g_rec.append(parents.gg)
bs_rec.append(parents[0].score)
ws_rec.append(parents[-1].score)
T_rec.append(sa_T)
ms_rec.append(sa_mstep)


#-------------------------------------------------------------------------------
#--- part 3: the generational loop ---------------------------------------------
#-------------------------------------------------------------------------------

for g in range(G):
    
    # step A: creating and evaluating offspring
    offspring.advance_generation()
    for i,dude in enumerate(offspring):
        oldguy=parents[i]
        if i < elite_size:
            dude.copy_DNA_of(oldguy,copyscore=True)
        elif rand() < saga_ratio:
            dude.copy_DNA_of(oldguy)
            dude.mutate(sa_mprob,sd=sa_mstep)
            dude.evaluate()
            if dude<oldguy:
                pass # accepting improvement
            elif rand() < exp(-(dude.score-oldguy.score)/sa_T):
                pass # accepting improvement
            else:
                dude.copy_DNA_of(oldguy,copyscore=True) # preferring parent DNA
        else:
            pa,pb=pse(ps,ga_selp,size=2)
            dude.CO_from(parents[pa],parents[pb])
            dude.evaluate()
    
    # step B: cooling down temperature and mutation step size parameter
    sa_T*=exp(-AE)
    sa_mstep*=exp(-AE)

    # step C: closing generational cycle --> offspring becomes new parent population
    for pdude,odude in zip(parents,offspring):
        pdude.copy_DNA_of(odude,copyscore=True,copyparents=True)
    parents.advance_generation()
    parents.sort()

    # step D: recording generation characteristics for plot
    g_rec.append(parents.gg)
    bs_rec.append(parents[0].score)
    ws_rec.append(parents[-1].score)
    T_rec.append(sa_T)
    ms_rec.append(sa_mstep)
    
#-------------------------------------------------------------------------------
#--- part 4: plotting ----------------------------------------------------------
#-------------------------------------------------------------------------------

plt.plot(g_rec,bs_rec,'go',label='best')
plt.plot(g_rec,ws_rec,'ro',label='worst')
plt.plot(g_rec,array(T_rec)+90,'b-',lw=2,label='temperature + 90')
plt.axhline(y=90,ls='--',color='grey')
txt='my first EA homebrew:'
txt+=' a concoction of GA with simulated annealing'
txt+='\ntest function: 10D Weierstrass (no. 11 of CEC-2005) with glob. min. = 90.0'
plt.title(txt,fontsize=11)
plt.ylim(85,130)
plt.xlabel('generation')
plt.ylabel('fitness and temperature')
ax1=plt.axes()
ax1t=ax1.twinx()
ax1t.plot(g_rec,ms_rec,'c-',lw=2,label='sa_mstep')
ax1t.set_ylabel('sa_mstep')
handles, labels = ax1.get_legend_handles_labels()
handlest, labelst = ax1t.get_legend_handles_labels()
plt.legend(handles+handlest, labels+labelst, loc='upper right')
plt.show()




