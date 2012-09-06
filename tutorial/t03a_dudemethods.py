#!python
"""
peabox tutorial 
lesson 03 - operators for individuals
a) basic comparison operators for individuals

test function: 8-dimensional parabolic potential


what is to be shown here:

Once you implement comparison operators like __lt__(self) or __ge__(self) and
tell these operators to go look in dude.score, then suddenly you can do all that shit:

    boolvar = dude1 < dude2
    p=Population(..); p.sort(); p.reverse()

Next, individuals and populations have an additional attribute p.whatisfit
which has the value 'minimize' or 'maximize', this determines the output of the
expressions dude1.isbetter(dude2) and dude1.isworse(dude2).
p.sort() puts 'the best' first, not necessarily the one with lowest score

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
import numpy.random as npr
#from numpy import array, arange, asfarray, pi, cos, zeros, where, linspace
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population

def parabolic(x):
    return np.sum(x*x)

searchspace=(('p1',-1.,+1.),
             ('p2',-1.,+1.),
             ('p3',-1.,+1.))

N=5
p=Population(Individual,N,parabolic,searchspace)

npr.seed(1)
p.new_random_genes()
p.eval_all()

for dude in p:
    dude.score=np.round(dude.score,2)
sc=p.get_scores()
print "the population's scores: ",sc  # should give the list [ 1.22  1.32  0.53  0.17  1.81]
dude0,dude1,dude2,dude3,dude4=p
print 'dude0<dude1 yields ',dude0<dude1,'    and dude0.isbetter(dude1) yields ',dude0.isbetter(dude1)
print 'dude0<dude2 yields ',dude0<dude2,'    and dude0.isbetter(dude2) yields ',dude0.isbetter(dude2)
print 'p.whatisfit and dude0.whatisfit are: ',p.whatisfit,dude0.whatisfit
p.determine_whatisfit('max')
print "now how does i look like after the spell 'p.determine_whatisfit('min')'?"
print 'p.whatisfit and dude0.whatisfit are: ',p.whatisfit,dude0.whatisfit
print 'and the comparisons from above?'
print 'dude0<dude1 yields ',dude0<dude1,'    and dude0.isbetter(dude1) yields ',dude0.isbetter(dude1)
print 'dude0<dude2 yields ',dude0<dude2,'    and dude0.isbetter(dude2) yields ',dude0.isbetter(dude2)
print 2*'\n'

print "the population's scores: ",sc  # should give the list [ 1.22  1.32  0.53  0.17  1.81]
print 'the population itself gets printed out like this:'
print p
print 'now sorting'
p.sort()
print p
print 'now reversing'
p.reverse()
print p
print '\n'
print 'each dude has a number, its attribute "no", and each one has also a stored former number "oldno"'
print 'dude.no: ',[dude.no for dude in p]
print 'dude.oldno: ',[dude.oldno for dude in p]
print "now let's update the numbering so it corresponds to the current sequence"
p.update_no()
print 'dude.no: ',[dude.no for dude in p]
print 'dude.oldno: ',[dude.oldno for dude in p]
print 'and hence the direct printout: ',p,'   which is the output of "Population.__str__()"'
print 2*'\n'

print "now let's sort the population according to the first entry of the DNA vector"
p.sort_for('DNA[0]')  # you can sort_for(somestring) as long as eval('dude.'+somestring) makes some sense
print 'dude.no: ',[dude.no for dude in p]
print 'dude.oldno: ',[dude.oldno for dude in p]
print 'dude.DNA[0]: ',[dude.DNA[0] for dude in p]
print 'the update function for "oldno" gives you the choice of doing it either according to the current ranking'
p.mark_oldno()
print 'after p.mark_oldno()'
print 'dude.oldno: ',[dude.oldno for dude in p]
p.mark_oldno(fromno=True)
print 'or according to the current value of "no"'
print 'after p.mark_oldno(fromno=True)'
print 'dude.oldno: ',[dude.oldno for dude in p]
print '\n'
print 'now if we update "no" again, it all means what it says'
p.update_no()
print 'dude.no: ',[dude.no for dude in p],'   <-- reflects current status'
print 'dude.oldno: ',[dude.oldno for dude in p],'   <-- reflects status before we started messing by sorting for DNA[0]'
print 'dude.DNA[0]: ',[dude.DNA[0] for dude in p],'   <-- reason for current sorting status'
print 2*'\n'

print "Well, that's all a lot of fanciness that might stay unused in many cases,"
print "the only really important thing to remember is just this:"
print "p.sort() takes into account what setting we have in p.whatisfit"
p.sort()
print "after sorting again (remember, we're still shooting for maximisation):"
print 'dude.score: ',[dude.score for dude in p]
p.determine_whatisfit('min')
p.sort()
print "and after a switch to minimisation and renewed sorting:"
print 'dude.score: ',[dude.score for dude in p]






