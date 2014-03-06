#!python
"""
peabox tutorial 
lesson 01 - how to use individuals and populations
a) putting together a simple evolution strategy: (4,12)-ES

test function: 8-dimensional parabolic potential

what is to be shown here:
This is more like a demo showing how the code is readable, understandable,
because the functionalities are intuitive. This basic level evolution strategy
could be called the "Hello world!" program of evolutionary computation. So here
is what it looks like with peabox.

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2014
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population


#-------------------------------------------------------------------------------
#--- part 1: setup -------------------------------------------------------------
#-------------------------------------------------------------------------------

def parabolic(x):   # the test function to be minimised
    return np.sum(x*x)

dim=8   # search space dimension
searchspace = [('parameter'+str(i+1),-1.,+1.) for i in range(dim)]
mu=4    # size of parent population of (mu,lambda)-ES
la=12   # size of offspring population of (mu,lambda)-ES
parents=Population(Individual,mu,parabolic,searchspace)
offspring=Population(Individual,la,parabolic,searchspace)
P=1.         # probability that a gene gets modified by the mutation operator
mstep=0.01   # mutation step size parameter
parents.new_random_genes()
parents.sort()

#-------------------------------------------------------------------------------
#--- part 2: the loop for (mu,la)-ES -------------------------------------------
#-------------------------------------------------------------------------------
for g in range(50):    # generation loop
    for dude in offspring:
        parentchoice=npr.randint(mu)  # all among the mu best of the old generation have an equal chance to reproduce
        dude.copy_DNA_of(parents[parentchoice])
        dude.mutate(P,mstep)  # mutate each gene with probability 1 and standard deviation in relation to search space width defined by mstep
        dude.evaluate()
    offspring.sort()
    print 'best current score: {}'.format(offspring[0].score)
    for i in range(mu):
        parents[i].copy_DNA_of(offspring[i],copyscore=True) # select the best mu as parents for the next generation
    

