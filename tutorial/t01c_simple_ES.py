#!python
"""
peabox tutorial 
lesson 01 - how to use individuals and populations
a) putting together a simple evolution strategy: (mu,la)-ES

test function: 8 electrons on hilly ring track

what is to be shown here:
 - let's modify the necklace problem, instead of minimising the standard
   deviation of neighbour distances let's now talk about energies; let's say
   the particles have repelling 1/r-potentials between neighbouring pairs
 - let's add a second form of potential energy: a hilliness of the track
 - why the problem is still somehow easy: swapping two marbles conserves the
   fitness, there are again many globally optimal solutions; with a little
   luck the marbles fall already into the right valleys by random; if they are
   not completely lopsided initially one has a good chance of finding a very
   good solutions and along the way towards it maybe only one or two marbles
   have to be pushed over a hilltop, and if the valley they have to leave was
   overcrowded then they had been pushed up towards the side anyway
 - that problem is still not serious enough
 - but something can be done: give the marbles individual weights on the hilly
   track, then they become nonidentical particles! (you can do that as a little
   exercise if you want or just fetch the corresponding problem from the
   test functions collection)

by Markus Stokmaier, IKET, KIT, Karlsruhe, January 2014
"""
from os import getcwd
from os.path import join
import numpy as np
from numpy import pi, exp, sin, cos, mod
from numpy import array, arange, zeros, ones, sort, std, roll
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Circle, Wedge
from peabox_individual import Individual
from peabox_population import Population


#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

class hilly(Individual):
    def evaluate(self,tlock=None):
        if tlock is not None: raise NotImplementedError("Dunno how to deal with that threading and tlock shit!")
        # first step: evaluate potential energy of neighbours
        cDNA=array(self.DNA,copy=1); cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        if 0. in diff:
            E_neighbour=2000. # no need to take fancy measures like a smooth and steady
            # projection of [0,inf] onto [0,2000]; if I know that the nondeterministic
            # EA will not be messed up by this fitness function artefact, why bother?
            print 'dude {} in generation {} has 0 in diff'.format(self.no,self.gg)
            print diff
        else:
            E_neighbour=np.sum(1./diff)
        # second step: evaluate potential energy of each marble along the hilly circle track
        E_pot=0.
        for i,marble in enumerate(self.DNA):
            E_pot+=self.trackpotential(marble)
        self.score=E_neighbour+8.*E_pot   #-80.
        return self.score
    def trackpotential(self,phi):
        A=0.6; B=1.0; C=0.8     # coefficients for angular frequencies
        o1=0; o2=40.; o3=10.   # angular offsets
        e_pot=A*(sin(2*pi*(phi+o1)/360.)+1)+B*(sin(4*pi*(phi+o2)/360.)+1)+C*(sin(8*pi*(phi+o3)/360.)+1)
        return e_pot
    def plot_yourself(self,location):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        p=PatchCollection([Wedge((0.,0.), 1.2, phi-3,phi+3, width=0.5) for phi in range(2,360,5)],cmap=plt.cm.bone,edgecolor='white',linewidths=0,zorder=1)
        p.set_array(self.trackpotential(arange(2,360,5)))
        ax.add_collection(p)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.1) for i,phi in enumerate(self.DNA)],
                          facecolor='c',edgecolor='k',linewidths=1.5,zorder=3)
        ax.add_collection(c)
        ax.axis('equal')
        ax.axis('off')
        plt.title('problem: electrons on hilly track\ni.e. minimise two types of potential energies')
        txt='DNA = {}\nscore = {}'.format(np.round(self.DNA,3),self.score)
        plt.suptitle(txt,x=0.03,y=0.02,ha='left',va='bottom',fontsize=10)
        runlabel='c'+str(self.ncase).zfill(3)+'_g'+str(self.gg).zfill(3)+'_i'+str(self.no).zfill(3)
        plt.savefig(join(location,'hilly_solution_'+runlabel+'.png'))
        plt.close()
    def set_bad_score(self):
        self.score=3000.
        
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
parents=Population(hilly,mu,dummyfunc,searchspace)
offspring=Population(hilly,la,dummyfunc,searchspace)
plotpath=join(getcwd(),'plots')


#-------------------------------------------------------------------------------
#--- part 2: random starting point for search ----------------------------------
#--- and more initialisation ---------------------------------------------------
#-------------------------------------------------------------------------------

npr.seed(111)  # seeding numpy's random number generator so we get reproducibly the same random numbers
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
txt='evolution strategies: ({},{})-ES applied to 8D hilly problem'.format(mu,la)
plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
plt.show()

