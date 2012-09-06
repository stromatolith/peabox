#!python
"""
peabox tutorial 
lesson 01 - how to use individuals and populations
a) putting together a simple evolution strategy: (4,12)-ES

test function: 8-dimensional parabolic potential

what is to be shown here:
basic operators like Individual.mutate() or Individual.copy_DNA_of()
or Population.sort() make it quick and easy to rapid-prototype evolutionary
algorithm ideas

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population


#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

def parabolic(x):
    return np.sum(x*x)

searchspace=(('p1',-1.,+1.),
             ('p2',-1.,+1.),
             ('p3',-1.,+1.),
             ('p4',-1.,+1.),
             ('p5',-1.,+1.),
             ('p6',-1.,+1.),
             ('p7',-1.,+1.),
             ('p8',-1.,+1.))

mu=4    # size of parent population of (mu,lambda)-ES
la=12   # size of offspring population of (mu,lambda)-ES
dim=len(searchspace)
parents=Population(Individual,mu,parabolic,searchspace)
offspring=Population(Individual,la,parabolic,searchspace)


#-------------------------------------------------------------------------------
#--- part 2: random starting point for search ----------------------------------
#--- and more initialisation ---------------------------------------------------
#-------------------------------------------------------------------------------

npr.seed(1)  # seeding numpy's random number generator so we get reproducibly the same random numbers
startpoint=2*npr.rand(dim)-1     # point in search space where initial population is placed
parents.set_every_DNA(startpoint)
parents.eval_all()
print 'fitness of initial population: '
print [dude.score for dude in parents]
print 'or also via get_scores():'
print parents.get_scores()

mstep=0.002   # mutation step size parameter
print 'initial mstep: ',mstep, 2*'\n'
g_rec=[]  # for recording x-data for plot, i.e. generation
s_rec=[]  # for recording y-data for plot, i.e. score or fitness
ms_rec=[]  # for recording mutation step size history 


#-------------------------------------------------------------------------------
#--- part 3: the loop for (mu,la)-ES -------------------------------------------
#-------------------------------------------------------------------------------

for g in range(50):    # generation loop
    parents.sort()
    successrate=0.
    for dude in offspring:
        parentchoice=npr.randint(mu)
        dude.copy_DNA_of(parents[parentchoice])
        dude.mutate(1,mstep)  # mutate each gene with probability 1 and standard deviation in relation to search space width defined by mstep
        dude.evaluate()
        if dude < parents[parentchoice]:   # comparison operators look at individual's score (the class Individual is just so written)
            successrate+=1
        g_rec.append(g)
        s_rec.append(dude.score)
    ms_rec.append(mstep)
    successrate/=la
    if successrate > 1./5:      # 1/5th success rule is popular because reasonable for evolution strategies
        mstep*=1.2
    else:
        mstep/=1.2
    offspring.sort()
    for i in range(mu):
        parents[i].copy_DNA_of(offspring[i],copyscore=True)
    
print 'fitness of final population: '
print parents.get_scores()
print 'final mstep: ',mstep, 2*'\n'



#-------------------------------------------------------------------------------
#--- part 4: making a plot showing ---------------------------------------------
#--- a) fitness over time and --------------------------------------------------
#--- b) step size over time ----------------------------------------------------
#-------------------------------------------------------------------------------

N=len(s_rec)
g_rec=np.array(g_rec)-0.4+0.8*npr.rand(N)
l1=plt.plot(g_rec,s_rec,'bo',label='fitness')
ax1=plt.axes()
ax2=ax1.twinx()
l2=ax2.plot(ms_rec,'g-',label='mstep')
ax1.set_xlabel('generation')
ax1.set_ylabel('fitness')
ax2.set_ylabel('mutation step size parameter')
plt.legend()
plt.xlim(0,50)
txt='evolution strategies: (4,12)-ES applied to 8D parabolic potential'
txt+='\nstep size grows first and later shrinks when zooming in on optimum'
txt+='\nbut beware: this unimodal test function is way too easy'
plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
plt.show()

