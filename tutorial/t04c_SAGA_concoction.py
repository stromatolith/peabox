#!python
"""
peabox tutorial 
lesson 04 - my first EA homebrew
c) making it easier to collect data for plots part 2

test function: CEC-2005 test function 11 --> Weierstrass function in 10 dimensions

what is to be shown here:

This is about statistics and visualisation with the goal of finding out what
goes on in the algorithm.

Okay, our algorithm concoction somehow improves the population, but not
spectacularly. In order to find out whether the whole SAGA idea was lame to 
begin with or whether there is some potential to tune better performance into
the algorithm, at some point we will have to do a parameter study.

But before we do that, let's gather some more diagnostic info to visualise.

So we would like to do two more things:
 - colour coding for ancestry: in the population cloud plot we want to see
   SA offspring in a different colour than GA offspring --> maybe one colour
   will always be in bad score regions and we can stop wasting time on producing
   these data points
 - pimp the recorder so it collects the SA acceptance rate and such stuff

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

from os.path import join
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
from peabox_plotting import ancestryplot, ancestcolors, show_these_colormaps

#-------------------------------------------------------------------------------
#--- part 0: class definitions -------------------------------------------------
#-------------------------------------------------------------------------------

# subclassing Population for this purpose:
# giving it more properties for the recorder to call
class SAGA_Population(Population):
    def __init__(self,species,popsize,objfunc,paramspace):
        Population.__init__(self,species,popsize,objfunc,paramspace)
        self.sa_improverate=0.
        self.sa_toleraterate=0.
        self.ga_improverate=0.
        self.sa_bestcontributions=0
        self.ga_bestcontributions=0
        self.sa_beststeps=[]
        self.ga_beststeps=[]


#-------------------------------------------------------------------------------
#--- intermezzo: the colormap used in peabox_plotting.ancestryplot -------------
#-------------------------------------------------------------------------------

show_these_colormaps([ancestcolors],'colormap_used_for_ancestry_coloring.png')

#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

dim=10
f11=tfp(11,dim)   # Weierstrass function in 10 dimensions

searchspace=simple_searchspace(10,-3.,1.)

ps=80    # population size
G=120     # number of generations to go through
parents=SAGA_Population(Individual,ps,f11,searchspace); parents.objname='CEC05 f11'
offspring=SAGA_Population(Individual,ps,f11,searchspace); offspring.objname='CEC05 f11'

parents.ncase=1   # let this be case 1, plots and pickles will contain the number
offspring.ncase=1

rec=Recorder(parents) # creating this Recorder instance to collect survey data on parent population
rec.snames.append('sa_improverate') # appending a scalar property's name to the survey list
rec.snames.append('sa_toleraterate')
rec.snames.append('ga_improverate')
rec.snames.append('sa_bestcontributions')
rec.snames.append('ga_bestcontributions')
rec.reinitialize_data_dictionaries()
rec.set_goalvalue(97.) # note down whether/when this goal was reached

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
sa_mstep=0.05 # mutation step size parameter
sa_mprob=0.6 # mutation probability
ga_selp=3. # selection pressure
AE=0.04 # annealing exponent --> exp(-AE) is multiplier for reducing temperature
elite_size=2

#-------------------------------------------------------------------------------
#--- part 3: the generational loop ---------------------------------------------
#-------------------------------------------------------------------------------

for g in range(G):
    
    # step A: creating and evaluating offspring
    # inside the generation loop this time we give ancestry codes between 0 and
    # 1 to be mapped on the colormap printed out in the intermezzo above
    
    sa_improved=0
    sa_tolerated=0
    sa_events=0
    ga_improved=0
    offspring.advance_generation()
    for i,dude in enumerate(offspring):
        oldguy=parents[i]
        if i < elite_size:
            dude.copy_DNA_of(oldguy,copyscore=True)
            dude.ancestcode=0.18  # blue-purple for conserved dudes
        elif rand() < saga_ratio:
            sa_events+=1
            dude.copy_DNA_of(oldguy)
            dude.mutate(sa_mprob,sd=sa_mstep)
            dude.evaluate()
            if dude<oldguy:
                dude.ancestcode=0.49 # turquoise for improvement through mutation
                sa_improved+=1
            elif rand() < exp(-(dude.score-oldguy.score)/sa_T):
                dude.ancestcode=0.25 # yellow/beige for tolerated dude
                sa_tolerated+=1
            else:
                dude.copy_DNA_of(oldguy,copyscore=True) # preferring parent DNA
                dude.ancestcode=0.18  # blue-purple for conserved dudes
        else:
            pa,pb=pse(ps,ga_selp,size=2)
            dude.CO_from(parents[pa],parents[pb])
            dude.evaluate()
            dude.ancestcode=0.39 # white for CO dude
            if dude.score < min(parents[pa].score,parents[pb].score):
                ga_improved+=1
    
    # step B: cooling down temperature and mutation step size parameter
    sa_T*=exp(-AE)
    sa_mstep*=exp(-AE)

    # an intermediate step: if the best solution was replaced, remember responsibility
    offspring.sort()
    if offspring[0] < parents[0]: # whether a better DNA was found than in last generation
        delta=offspring[0].score-parents[0].score
        if offspring[0].ancestcode==0.39:
            parents.ga_bestcontributions+=1
            parents.ga_beststeps.append(delta)
        else:
            parents.sa_bestcontributions+=1
            parents.sa_beststeps.append(delta)

    # step C: closing generational cycle --> offspring becomes new parent population
    for pdude,odude in zip(parents,offspring):
        pdude.copy_DNA_of(odude,copyscore=True,copyparents=True,copyancestcode=True)
    parents.advance_generation()
    #parents.sort() # not necessary in case offspring has already been sorted

    # step D: recording generation characteristics for plot
    parents.sa_improverate=sa_improved/float(sa_events)
    parents.sa_toleraterate=sa_tolerated/float(sa_events)
    parents.ga_improverate=ga_improved/float(ps-sa_events-elite_size)
    rec.save_status()

#-------------------------------------------------------------------------------
#--- part 4: plotting ----------------------------------------------------------
#-------------------------------------------------------------------------------

# plot A: the usual population score distribution, but this time in colour!!
ancestryplot(rec,ylimits=[90,130],whiggle=0.6)


# plot B: improvement etc rates
gg=rec.gg
saimp=rec.sdat['sa_improverate']
satol=rec.sdat['sa_toleraterate']
gaimp=rec.sdat['ga_improverate']
plt.plot(gg,saimp,'cx',label='SA improve rate')
plt.plot(gg,satol,'yx',label='SA tolerate rate')
plt.plot(gg,gaimp,'g+',label='GA improve rate')

# low-pass filtering:
win=np.hamming(10)
lp_saimp=np.convolve(saimp,win,mode='same')/np.sum(win)
lp_satol=np.convolve(satol,win,mode='same')/np.sum(win)
lp_gaimp=np.convolve(gaimp,win,mode='same')/np.sum(win)
plt.plot(gg,lp_saimp,'c-',lw=2)
plt.plot(gg,lp_satol,'y-',lw=2)
plt.plot(gg,lp_gaimp,'g-',lw=2)

plt.legend()
plt.ylim(0,1)
plt.savefig(join('plots','improrates_'+parents.label+'.png'))
plt.close()


# plot C: contributions for improving best of population
sabcontrib=rec.sdat['sa_bestcontributions']
gabcontrib=rec.sdat['ga_bestcontributions']
plt.plot(gg,sabcontrib,'b-',label='SA')
plt.plot(gg,gabcontrib,'g-',label='GA')
txt="It may seem GA does the local search here, but don't get mistaken: when the CO-operator"
txt+="\ncreated a successful trial, you actually often don't know where the snippets with beneficial"
txt+="\nnew DNA info where created ... but here you do know, by mutation in the SA step."
plt.title(txt,fontsize=9)
plt.xlabel('generation')
plt.ylabel('how often offspring shifted best of population')
plt.legend(loc='upper left')
plt.savefig(join('plots','contributions_shifting_best_'+parents.label+'.png'))
plt.close()


# plot D: how far the best score was moved when improved
sasteps=parents.sa_beststeps
gasteps=parents.ga_beststeps
fig=plt.figure()
ax1=fig.add_subplot('211')
ax2=fig.add_subplot('212')
ax1.plot(arange(len(sasteps)),sasteps,'co',label='SA')
ax2.plot(arange(len(gasteps)),gasteps,'yo',label='GA')
#handles1, labels1 = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
#plt.legend(handles1+handles2, labels1+labels2)
ax2.set_xlabel('improvement count')
ax1.set_ylabel('quantity of score improvement - SA')
ax2.set_ylabel('quantity of score improvement - GA')
plt.savefig(join('plots','quantities_shifting_best_'+parents.label+'.png'))
plt.close()

"""
Okay, now we can build a loop around the whole thing like this:

for i in range(7):
    parents=SAGA_Population(Individual,ps,f11,searchspace); parents.objname='CEC05 f11'
    offspring=SAGA_Population(Individual,ps,f11,searchspace); offspring.objname='CEC05 f11'
    parents.ncase=i   # giving a new case number prevents overwriting the *.png files
    offspring.ncase=i
    
    # and we can systematically vary some parameters like the starting temperature
    sa_T = 6.+2*i
    # or
    saga_ratio=0.1+0.1*i
    # or any other one ...
    # and we can have a look at how this influences stories told by the plots


But meanwhile, the code has grown quite a bit, let us therefore wrap the whole
EA up into a class definition in the next step of the tutorial. The plots can
also become plot functions recieving a recorder instance.
"""
