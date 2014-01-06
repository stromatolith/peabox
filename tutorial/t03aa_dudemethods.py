#!python
"""
peabox tutorial 
lesson 03 - operators for individuals
b) basic DNA manipulation operators for individuals: old-school CO operators

sample usage of the features presented in lesson 3b

advantages of those features: the code below
 - can be written in a short and readable form
 - still functions after transitioning from minimisation to maximisation

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
import numpy.random as npr
from numpy import amax, linspace
from peabox_individual import Individual
from peabox_population import Population

def parabolic(x):
    return np.sum(x*x)

searchspace=(('p1',-20.,+20.),
             ('p2',-20.,+20.),
             ('p3',-20.,+20.),
             ('p4',-20.,+20.),
             ('p5',-20.,+20.))

N=5
p=Population(Individual,N,parabolic,searchspace)
npr.seed(3)
p.new_random_genes()
p.eval_all()
p.sort()
p.update_no()

w = amax(p[0].widths)           # largest width of the search space
ms=linspace(0.05*w,0.5*w,N)     # a range of mutation step sizes
for dude in p:
    dude.mutate(1,ms[dude.no])  # applying individual mutation step sizes
    

    print "dude {} has score {:.4f} and gets mutated with step size {:.4f}".format(dude.no,dude.score,ms[dude.no])