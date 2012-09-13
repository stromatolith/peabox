#!python
"""
peabox tutorial 
lesson 04 - my first EA homebrew
b) making it easier to collect data for plots

test function: CEC-2005 test function 11 --> Weierstrass function in 10 dimensions

what is to be shown here:

 - there is a class "Recorder" to manage these lists collecting data to plot

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

from peabox_recorder import Recorder
from peabox_plotting import ancestryplot

#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

dim=10
f11=tfp(11,dim)   # Weierstrass function in 10 dimensions

searchspace=simple_searchspace(10,-3.,1.)

ps=40    # population size
G=40     # number of generations to go through
parents=Population(Individual,ps,f11,searchspace); parents.objname='CEC05 f11'
offspring=Population(Individual,ps,f11,searchspace); offspring.objname='CEC05 f11'

rec=Recorder(parents) # creating this Recorder instance to collect survey data on parent population

#-------------------------------------------------------------------------------
#--- part 2: initialisation ----------------------------------------------------
#-------------------------------------------------------------------------------

seed(1)  # seeding numpy's random number generator so we get reproducibly the same random numbers
parents.new_random_genes()
parents.eval_all()
parents.sort()
parents.update_no()

rec.save_status()

sa_T=parents[-1].score-parents[0].score # starting temperature
saga_ratio=0.5 # fraction of offspring created by SA and not GA
sa_mstep=0.02 # mutation step size parameter
sa_mprob=0.6 # mutation probability
ga_selp=3. # selection pressure
AE=0.08 # annealing exponent --> exp(-AE) is multiplier for reducing temperature
elite_size=0

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
    rec.save_status()
    
#-------------------------------------------------------------------------------
#--- part 4: plotting ----------------------------------------------------------
#-------------------------------------------------------------------------------

ancestryplot(rec,ylimits=[90,130])
ancestryplot(rec,ylimits=[90,130],whiggle=0.6,suffix='_whiggled')

