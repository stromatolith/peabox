#!python
"""
peabox tutorial 
lesson 04 - my first EA homebrew
d) let's put our SAGA homebrew into a class definition

test function: CEC-2005 test function 11 --> Weierstrass function in 10 dimensions

what is to be shown here:

 - the whole algorithm and our favourite plot function are now wrapped up nicely
   in the file SAGA_defs.py
 - now we can really get rolling examining and tuning the algo inside out

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
#from numpy.random import seed, rand, randn, randint
from numpy import zeros, mean, std
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from peabox_individual import Individual
#from peabox_population import Population
from peabox_testfuncs import CEC05_test_function_producer as tfp
from peabox_helpers import simple_searchspace

from peabox_plotting import ancestryplot
from SAGA_defs import SAGA_Population, SAGA, plot_improvement_rates


#-------------------------------------------------------------------------------
#--- part 1: six single test runs ----------------------------------------------
#-------------------------------------------------------------------------------
#"""
# looking at the plots created this way will help to quickly exclude the worst
# parameter settings of our new EA

dim=10
f11=tfp(11,dim)   # Weierstrass function in 10 dimensions

searchspace=simple_searchspace(10,-3.,1.)

ps=40    # population size
G=120     # number of generations to go through

for i,r in enumerate([0.2,0.5,0.8]):
    parents=SAGA_Population(Individual,ps,f11,searchspace); parents.objname='CEC05 f11'
    offspring=SAGA_Population(Individual,ps,f11,searchspace); offspring.objname='CEC05 f11'
    parents.ncase=2+i
    
    ea=SAGA(parents,offspring,userec=True)
    ea.AE=0.04
    ea.saga_ratio=r
    ea.reduce_mstep=False
    ea.sa_mstep=0.05
    ea.rec.set_goalvalue(97)
    ea.simple_run(G)
    txt='best score: {0}'.format(parents[0].score)
    ancestryplot(ea.rec,whiggle=0.6,ylimits=[90,120],addtext=txt)
    plot_improvement_rates(ea.rec)
    
    parents.reset()
    parents.next_subcase()
    ea.rec.clear()
    ea.reduce_mstep=True
    ea.sa_mstep=0.05      # not really necessary as it stayed fix during first run
    ea.simple_run(G)
    txt='best score: {0}'.format(parents[0].score)
    ancestryplot(ea.rec,whiggle=0.6,ylimits=[90,120],addtext=txt)
    plot_improvement_rates(ea.rec)

# ea.saga_ratio=0.5 seems to be the best setting.
# That at this point 100% GA will not work is quite clear, remember, there is
# no mutation operator in the GA offspring production so you would ever just
# recombine the old same snippets of DNA.
# But this leads us to the thought that it might be inefficient to cling to this
# 50%-50% separation between SA and GA with no overlap. Maybe it will be better
# to try some overlap and apply some sort of mutation (not necessarily with same
# intensity) and maybe even the SA energy criterion to the GA offspring. This is
# up to you now: add a "create_offspring_2()" method to the SAGA class or modify
# the old one.

#"""



#-------------------------------------------------------------------------------
#--- part 2: statistics over many runs -----------------------------------------
#-------------------------------------------------------------------------------
"""
# In order to make an objective decision on the performance of an EA, (either for
# comparing it to a different algorithm or for parameter fine tuning,) nothing
# helps, at some point you will have to make statistics over many runs and on
# top applied to different test functions

dim=10
f11=tfp(11,dim)   # Weierstrass function in 10 dimensions

searchspace=simple_searchspace(10,-3.,1.)

ps=40    # population size
G=120     # number of generations to go through
runs=10

for i,AE in enumerate([0.02,0.04,0.06]):
    print 'running now with AE=',AE
    parents=SAGA_Population(Individual,ps,f11,searchspace); parents.objname='CEC05 f11'
    offspring=SAGA_Population(Individual,ps,f11,searchspace); offspring.objname='CEC05 f11'
    parents.ncase=5+i
    
    ea=SAGA(parents,offspring,userec=True)
    ea.AE=AE
    ea.saga_ratio=0.5
    ea.reduce_mstep=True
    ea.rec.set_goalvalue(97)
    
    final_scores=zeros(runs)
    
    for j in range(runs):
        print 'working on run ',j
        ea.userec=False
        if j!=0:
            parents.reset()
            parents.next_subcase()
        if j==runs-1:
            ea.userec=True  # only record the last run
        ea.sa_mstep=0.05      # here it is necessary
        ea.simple_run(G)
        final_scores[j]=parents[0].score
        if j==runs-1:        # plot only last run
            print 'making plots from this run with label ',parents.label
            txt='best score: {0}'.format(parents[0].score)
            ancestryplot(ea.rec,whiggle=0.6,ylimits=[90,120],addtext=txt)
            plot_improvement_rates(ea.rec)
    
    print 80*'_'
    print 'final scores: {0}'.format(np.round(final_scores,2))
    print 'mean: {0:.2f}   standard deviation: {1:.2f}'.format(mean(final_scores),std(final_scores))
    print 80*'_'+2*'\n'
        
"""
