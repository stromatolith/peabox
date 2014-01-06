#!python
"""
peabox tutorial 
lesson 03 - operators for individuals
b) basic DNA manipulation operators for individuals: old-school CO operators

test function: N-dimensional parabolic potential


what is to be shown here:

 - how to mess with DNA

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
import numpy.random as npr
from numpy import array, arange, asfarray, pi, cos, zeros, ones, where, linspace
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population

def parabolic(x):
    return np.sum(x*x)

#searchspace2=(('p1',-20.,+20.),
#              ('p2',-20.,+20.))
#
#searchspace3=(('p1',-20.,+20.),
#              ('p2',-20.,+20.),
#              ('p3',-20.,+20.))

searchspace5=(('p1',-20.,+20.),
              ('p2',-20.,+20.),
              ('p3',-20.,+20.),
              ('p4',-20.,+20.),
              ('p5',-20.,+20.))

N=6
#p2a=Population(Individual,N,parabolic,searchspace2)
#p3a=Population(Individual,N,parabolic,searchspace3)
p5a=Population(Individual,N,parabolic,searchspace5)
p5b=Population(Individual,N,parabolic,searchspace5)

npr.seed(1)
p5a.marker_genes()
p5b.marker_genes(offset=10)

print 'printing results of get_DNAs()'
print p5a.get_DNAs()
print p5b.get_DNAs()
print '\n'
print 'using p.print_stuff() and p.print_stuff(slim_True)'
p5a.print_stuff(slim=True)
print '\n\nHere comes p5a'
p5a.print_stuff()
print '\nHere comes p5b'
p5b.print_stuff()
print 2*'\n'

print "filling p5a with crossed-over DNAs from first and last of p5b"
parentA=p5b[0]
parentB=p5b[-1]
for dude in p5a:
    dude.CO_from(parentA,parentB)
p5a.print_stuff()

print 'restoring initial setting'
p5a.marker_genes()
p5b.marker_genes(offset=10)
print 2*'\n'

print "having members of p5a cross DNAs with members of p5b"
for adude,bdude in zip(p5a,p5b):
    adude.CO_with(bdude)
p5a.print_stuff()
p5b.print_stuff()
print 'finally restoring initial setting'
p5a.marker_genes()
p5b.marker_genes(offset=10)
print 2*'\n'

print "one more thing about the two uniform CO operators CO_from() and CO_with():"
print "there is a parameter to steer the probability weight between 1st and 2nd parent"
p5c=Population(Individual,2,parabolic,searchspace5)
p5d=Population(Individual,200,parabolic,searchspace5)
parentA,parentB=p5c
parentA.set_DNA(zeros(5))
parentB.set_DNA(ones(5))
for dude in p5d:
    dude.CO_from(parentA,parentB,P1=0.8)  # 80% probability for each gene to come from 1st parent
print 'np.mean(p5d.get_DNAs()) yields ',np.mean(p5d.get_DNAs())
print 2*'\n'

print "another option of generating one offspring from two parents is simple mixing"
parentA.set_DNA(npr.randint(10,size=5))
parentB.set_DNA(npr.randint(10,size=5))
print 'parentA.DNA: ',parentA.DNA
print 'parentB.DNA: ',parentB.DNA
for i,dude in enumerate(p5a):
    dude.become_mixture_of(parentA,parentB,fade=0.2*i)
p5a.print_stuff()


