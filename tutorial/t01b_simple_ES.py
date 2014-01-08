#!python
"""
peabox tutorial 
lesson 01 - how to use individuals and populations
a) putting together a simple evolution strategy: (mu,la)-ES

test function: even particle distribution on ring track

what is to be shown here:
 - another simple test problem without barriers between local minima:
   repelling electrons on a one-dimensional track
 - a simple ES can solve the problem
 - the problem is visualisable, a human can judge a solution plot at a glance
 - BUT: the problem is quite easy, shift the optimal solution by one degree and
   you have a new optimal solution, ergo, the ensemble of global minima forms
   a valley lying diagonally in the search space; secondly, for each marble
   on the necklace there is a force pulling it towards the center position
   between its neighbours, the problem is somehow separable, you can go around
   the ring a few times and optimise one marble at a time
 - thus, the problem is too easy and not serious enough for challenging EAs

by Markus Stokmaier, IKET, KIT, Karlsruhe, January 2014
"""
from os import getcwd
from os.path import join
import numpy as np
from numpy import pi, exp, sin, cos, mod
from numpy import array, zeros, ones, sort, std, roll
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle, Wedge
from peabox_individual import Individual
from peabox_population import Population


#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

class necklace(Individual):
    """
    yet another test problem where the goal is to distribute particles
    evenly along a one-dimensional circular track; to be more precise, the goal
    is to minimise the standard deviation of neighbour distances
    """
    def evaluate(self,usage='no_matter_what',tlock=None):
        if tlock is not None: raise NotImplementedError("Dunno how to deal with that threading and tlock shit!")
        cDNA=array(self.DNA,copy=1); cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        self.score=std(diff)
        return self.score
    def plot_yourself(self,location):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        p=PatchCollection([Wedge((0.,0.), 1.0, phi-3,phi+3, width=0.035) for phi in range(2,360,5)],
                           facecolor='k',edgecolor='k',linewidths=0,zorder=1)
        p2=PatchCollection([Wedge((0.,0.), 1.2, phi-3,phi+3, width=0.035) for phi in [90,270]],
                           facecolor='w',edgecolor='w',linewidths=0,zorder=1)  # only for fixing perspective
        ax.add_collection(p)
        ax.add_collection(p2)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.12) for i,phi in enumerate(self.DNA)],
                          facecolor='c',edgecolor='k',linewidths=2,zorder=3)
        ax.add_collection(c)
        ax.axis('equal')
        ax.axis('off')
        ax.set_ylim([-1.3,1.3])
        plt.title('problem: distribute evenly\ni.e. minimise standard deviation of neighbour distances')
        txt='DNA = {}\nscore = {}'.format(np.round(self.DNA,3),self.score)
        plt.suptitle(txt,x=0.03,y=0.02,ha='left',va='bottom',fontsize=10)
        runlabel='c'+str(self.ncase).zfill(3)+'_g'+str(self.gg).zfill(3)+'_i'+str(self.no).zfill(3)
        plt.savefig(join(location,'necklace_solution_'+runlabel+'.png'))
        plt.close()
    def set_bad_score(self):
        self.score=100000.

def dummyfunc(x):
    return x*x
    
searchspace=(('p1',0.,360.),
             ('p2',0.,360.),
             ('p3',0.,360.),
             ('p4',0.,360.),
             ('p5',0.,360.),
             ('p6',0.,360.),
             ('p7',0.,360.),
             ('p8',0.,360.))

mu=1    # size of parent population of (mu,lambda)-ES
la=5    # size of offspring population of (mu,lambda)-ES
G=160   # number of generations
dim=len(searchspace)
parents=Population(necklace,mu,dummyfunc,searchspace)
offspring=Population(necklace,la,dummyfunc,searchspace)
plotpath=join(getcwd(),'plots')


#-------------------------------------------------------------------------------
#--- part 2: random starting point for search ----------------------------------
#--- and more initialisation ---------------------------------------------------
#-------------------------------------------------------------------------------

npr.seed(11)  # seeding numpy's random number generator so we get reproducibly the same random numbers
startpoint=360*npr.rand(dim)     # point in search space where initial population is placed
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
"""
The big advantage of peabox can be seen in the inner loop below: it is super
easily readable (for dude in offspring: copy .. mutate .. evaluate).
Commenting is not required, because the code is so readable, it looks
like pseudocode. This is the main purpose why I wrote this EA library.
"""
successes=0.2*ones((10,la),dtype=float)
for g in range(G):    # generation loop
    parents.sort()
    successes=roll(successes,-1,axis=0)
    for dude in offspring:
        parentchoice=npr.randint(mu)
        dude.copy_DNA_of(parents[parentchoice])
        dude.mutate(1,mstep)  # mutate each gene with probability 1 and standard deviation in relation to search space width defined by mstep
        dude.evaluate()
        successes[-1,dude.no]= int(dude < parents[parentchoice])   # comparison operators look at individual's score (the class Individual is just so written)
        g_rec.append(g)
        s_rec.append(dude.score)
    ms_rec.append(mstep)
    successrate=np.sum(successes)/float(np.size(successes))
    if successrate > 1./5:      # 1/5th success rule is popular because reasonable for evolution strategies
        mstep*=1.05
    else:
        mstep/=1.05
    offspring.sort()
    for i in range(mu):
        parents[i].copy_DNA_of(offspring[i],copyscore=True)
        
    if mod(g+1,20)==0:
        parents[0].plot_yourself(plotpath)

    parents.advance_generation()    
    
print 'fitness of final population: '
print parents.get_scores()
print 'final mstep: ',mstep, 2*'\n'



#-------------------------------------------------------------------------------
#--- part 4: making a plot showing ---------------------------------------------
#--- a) fitness over time and --------------------------------------------------
#--- b) step size over time ----------------------------------------------------
#-------------------------------------------------------------------------------

N=len(s_rec)
g_rec=np.array(g_rec)-0.4+0.8*npr.rand(N) # a little random horizontal whiggeling for better visibility
l1=plt.plot(g_rec,s_rec,'bo',label='fitness')
ax1=plt.axes()
ax2=ax1.twinx()
l2=ax2.plot(ms_rec,'g-',label='mstep')
ax1.set_xlabel('generation')
ax1.set_ylabel('fitness')
ax2.set_ylabel('mutation step size parameter')
plt.legend()
plt.xlim(0,G)
txt='evolution strategies: ({},{})-ES applied to 8D necklace problem'.format(mu,la)
plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
plt.show()

