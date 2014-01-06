#!python
"""
peabox tutorial 
lesson 02 - how to use individuals and populations
a) a simple classic real-coded genetic algorithm (RCGA)

test function: 8-dimensional Rastrigin function


what is to be shown here:

 - use of a recombination operator: uniform crossover
 - see how far you have to pump up population size or the amount of generations
   in order to sometimes find the global minimum (don't forget to stop seeding
   the random number generator with the same number)
 - you may play with mutation type and probability


what might also be taken away (at second glance):

here we also create snapshot sequences of current best solution candidates; I
think looking at such sequences can teach quite a lot ... mainly, it can be the
quickest way to get convinced that it is not useful to waste too much time with
too classic/historic ES and GA variants when ultimately wanting to solve a nasty
enough application problem (luckily we are living in the 21st century and there
more stuff to play with (CMA-ES, SaDE, PSO, Scatter Search, CHC etc) when
confronted with a challenging optimisation problem, ... but ... as one has to
educate oneself towards an informed user somewhere along the way I deem it
important to comparatively tinker around with the algorithms oneself, and this
is what my EA toolbox ultimately is made for --> tinkering and prototyping in
order to learn about the basics)

 - intuitively learn what 'separable problem means' --> by looking at snapshot
   sequences (if you plot each generation and click through the images quickly)
   see how gene values seem to jump across the landscape independently
   until each one finds a good place along its axis --> because they really are
   moving independently as the recombination operator, so heavily used,
   constantly tries to bring the best genes together from different individuals
   (an ES wound make the whole pattern jump at once incrementally instead of
   single genes jumping wiedely)
 - this implies the weakness of classic GAs dealing with nonseparable problems;
   we will see that recording some statistics later; the reason is the low level
   of generating new information in the case of RCGAs with low mutation rates
   (with binary coded DNAs it is different, because two consequtive 1-point-
   crossovers or a 2-point-CO can correspond to a bitflip, at least as long as the
   mutation operator ensures zeros or ones never die out at certain places in
   the gene sequence; so, under these circumstances CO-operators have exploration
   capability, which is however not the case in RCGAs)
 - try an ES on that Rastrigin problem and look at the snapshot sequence:
   rather than each gene value jumping independently you will percieve the whole
   pattern bending incrementally, and you will also see how a simple (M,L)-ES falls
   into each given trap, i.e. pushes each gene into the closest local valley and
   reliably misses the global optimum

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
import numpy.random as npr
from numpy import array, arange, asfarray, pi, cos, zeros, where, linspace
import matplotlib as mpl
import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population

#-------------------------------------------------------------------------------
#--- part 1: Ingredients -------------------------------------------------------
#-------------------------------------------------------------------------------

def rastrigin(x):
    x = asfarray(x)
    n=len(x); A=10.
    return A*n+np.sum(x**2-A*cos(2*pi*x))

class RouletteWheel:
    def __init__(self,dim):
        self.dim=dim
        self.thresh=zeros(self.dim)
    def initialize_thresholds(self,p):
        sc=p.get_scores()
        cs=1/sc
        for i in range(self.dim):
            if p.whatisfit=='minimize':
                self.thresh[i]=np.sum(cs[:i+1])
            if p.whatisfit=='maximize':
                self.thresh[i]=np.sum(sc[:i+1])
        self.thresh/=np.max(self.thresh)
    def choose_parents(self):
        r1,r2=npr.rand(2)
        parenta=np.sum(where(self.thresh<r1,1,0))
        parentb=np.sum(where(self.thresh<r2,1,0))
        return parenta,parentb

searchspace=(('p1',-5.,+5.),
             ('p2',-5.,+5.),
             ('p3',-5.,+5.),
             ('p4',-5.,+5.),
             ('p5',-5.,+5.),
             ('p6',-5.,+5.),
             ('p7',-5.,+5.),
             ('p8',-5.,+5.))

ps=40    # population size
G=41     # number of generations to go through
dim=len(searchspace)
parents=Population(Individual,ps,rastrigin,searchspace)
offspring=Population(Individual,ps,rastrigin,searchspace)
rw=RouletteWheel(dim)


#-------------------------------------------------------------------------------
#--- part 2: random starting point for search ----------------------------------
#--- and more initialisation ---------------------------------------------------
#-------------------------------------------------------------------------------

npr.seed(3) # seed differently or comment out if you want to see an algorithm's average performance over several runs
parents.new_random_genes()
parents.eval_all()
parents.sort()
print 'initial population:\nbest score = ',parents[0].score

# change the plot interval to a low value, e.g. 1, if you want to see
# the different characters of change patterns of the current best solution
#ginter=G/5  # interval to take a snapshot of best solution (for plot variant 2)
ginter=1  # interval to take a snapshot of best solution (for plot variants 1 and 3)
glist=[]; bestDNAs=[]; bestscores=[]


#-------------------------------------------------------------------------------
#--- part 3: the GA generational loop ------------------------------------------
#--- step 1: generate offspring via CO-operator --------------------------------
#--- step 2: mutate offspring with rather small probability --------------------
#-------------------------------------------------------------------------------

for g in range(G):
    rw.initialize_thresholds(parents) # adjusting parent choice probabilities to current fitness distribution
    for dude in offspring:
        pa,pb=rw.choose_parents()
        dude.CO_from(parents[pa],parents[pb])
        dude.insert_random_genes(0.05) # changing each DNA entry with probability P=0.05
    for pdude,odude in zip(parents,offspring):
        pdude.copy_DNA_of(odude)
    parents.eval_all()
    parents.sort()
    if g%ginter==0:
        glist.append(g)
        bestDNAs.append(parents[0].get_copy_of_DNA())
        bestscores.append(parents[0].score)
print 'final population:\nbest score = ',parents[0].score


#-------------------------------------------------------------------------------
#--- part 4: making plots showing development of best solution -----------------
#--- variant 1: gallery to click through ---------------------------------------
#---            use small ginter -----------------------------------------------
#--- variant 2: a couple intermediate steps in one image -----------------------
#---            use larger ginter ----------------------------------------------
#--- variant 3: temporal development of best DNA plotted from left to right ----
#---            use small ginter -----------------------------------------------
#-------------------------------------------------------------------------------

plot_variant=3

rastx=linspace(-5,5,200)
rasty=[rastrigin([x]) for x in rastx]
rasty=array(rasty)
rasty=0.5*dim*rasty/np.max(rasty)

txt2='The unrotated Rastrigin function is a separable problem, that means'
txt2+="\nshifting one gene doesn't worsen (influence at all) the other gene's"
txt2+="\ncontributions. That means we have 8 independent problems, which is a too"
txt2+="\neasy thing. Judging an EA's performance we have to move to nastier stuff."

if plot_variant==1:
    
    for j,g in enumerate(glist):
        fig=plt.figure(figsize=(12,9))
        ax=fig.add_subplot(111)
        for i in range(dim):
            ypos=i+1
            ax.axhspan(ypos-0.2,ypos+0.2,color='b',alpha=0.2)
        ax.plot(rastx,rasty,'k-',alpha=0.4)
        ax.scatter(bestDNAs[j],arange(dim)+1,s=40,c='r')
        ax.set_xlim(-5,5)
        ax.set_ylim(0,dim+0.5)
        ax.set_xlabel('search space coordinate')
        ax.set_ylabel('gene number (i.e. DNA vector index)')
        txt='temporal development of best solution DNA'
        txt+='\nsnapshot of generation '+str(g)
        plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
        plt.suptitle(txt2,x=0.02,y=0.01,ha='left',va='bottom',fontsize=8)
        plt.savefig('RCGA_snapshot_generation_{}.png'.format(str(g).zfill(3)))
        plt.close()

elif plot_variant==2:
    
    fig=plt.figure(figsize=(12,9))
    ax=fig.add_subplot(111)
    for i in range(dim):
        ypos=i+1
        ax.axhspan(ypos-0.2,ypos+0.2,color='b',alpha=0.2)
    ax.plot(rastx,rasty,'k-',alpha=0.4)
    mycolors=[mpl.cm.hot(x) for x in linspace(0,0.75,len(glist))]
    offset=linspace(-0.15,0.15,len(glist))
    for i,g in enumerate(glist):
        ax.scatter(bestDNAs[i],arange(dim)+1+offset[i],s=40,c=mycolors[i])
    ax.set_xlim(-5,5)
    ax.set_ylim(0,dim+0.5)
    ax.set_xlabel('search space coordinate')
    ax.set_ylabel('gene number (i.e. DNA vector index)')
    txt='temporal development of best solution DNA'
    txt+='\nblack is old, yellow is new'
    plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
    plt.suptitle(txt2,x=0.02,y=0.01,ha='left',va='bottom',fontsize=8.5)
    plt.show()  # pump the figure window up to fullscreen to be able to properly read everything

elif plot_variant==3:

    # creating figure
    fig=plt.figure(figsize=(12,9))
    ax=fig.add_subplot(111)
    
    #setting up background
    bgnx,bgny=2,200
    bgx=linspace(-1,glist[-1]*1.+1,bgnx+1)
    bgy=linspace(parents[0].lls[0],parents[0].uls[0],bgny+1)  # equivalent: bgy=linspace(-5,5,101) ... unless searchspace above has been changed
    bgcolumn=[rastrigin([x]) for x in linspace(parents[0].lls[0],parents[0].uls[0],bgny)]
    bgdat=zeros((bgnx,bgny))
    for i in range(bgnx): bgdat[i,:]=bgcolumn
    bgdat/(2*np.max(bgdat))
    ax.pcolor(bgx,bgy,bgdat.T,cmap=plt.cm.bone)  #,alpha=0.6)
    
    bestDNAs=array(bestDNAs)
    mycolors=[mpl.cm.hsv(i/float(dim-1)) for i in range(dim)]
    for i in range(dim):
        ax.plot(glist,bestDNAs[:,i],ls='-',lw=2,color=mycolors[i],alpha=0.6)
        ax.scatter(glist,bestDNAs[:,i],s=50-3*i,facecolor=mycolors[i],edgecolor=mycolors[i])
    ax.set_xlim(-1,glist[-1]+1)
    ax.set_ylim(parents[0].lls[0],parents[0].uls[0])
    ax.set_xlabel('generation')
    ax.set_ylabel('search space coordinate')
    txt='temporal development of best solution DNA (turn to low ginter for this plot)'
    txt+='\neach gene independently is pressured into a low point in a valley'
    txt+='\nGA should let genes jump independently, whereas ES should make pattern move often but incrementally'
    plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
    plt.suptitle(txt2,x=0.02,y=0.01,ha='left',va='bottom',fontsize=8.5)
    plt.show()


#-------------------------------------------------------------------------------
#--- Appendix --------------------------------------- --------------------------
#--- if you want to qualitatively check that the Roulette wheel implementation -
#--- makes sense, then use the part below --------------------------------------
#-------------------------------------------------------------------------------
"""
ps=5    # population size
dim=len(searchspace)
testpop=Population(Individual,ps,rastrigin,searchspace)

# if everybody has the same score, then everyone should be
# chosen as parent with equal probability... let's check that
for dude in testpop:
    dude.score=99.
sc1=testpop.get_scores()
rw=RouletteWheel(dim)
rw.initialize_thresholds(testpop)
histx=arange(ps)
histy1=zeros(ps)
for i in range(1000):
    choices=rw.choose_parents()
    for c in choices:
        if c<0: raise ValueError('parent choice should not be negative')
        if c>=ps: raise ValueError('there are not so many parents')
        histy1[c]+=1

# if everybody has the same score, then everyone should be
# chosen as parent with equal probability... let's check that
for i,dude in enumerate(testpop):
    dude.score=13*(i+1)
sc2=testpop.get_scores()
rw.initialize_thresholds(testpop)
histy2=zeros(ps)
for i in range(1000):
    choices=rw.choose_parents()
    for c in choices:
        if c<0: raise ValueError('parent choice should not be negative')
        if c>=ps: raise ValueError('there are not so many parents')
        histy2[c]+=1

# now let's say higher fitness means higher score
testpop.determine_whatisfit('maximize')
rw.initialize_thresholds(testpop)
histy3=zeros(ps)
for i in range(1000):
    choices=rw.choose_parents()
    for c in choices:
        if c<0: raise ValueError('parent choice should not be negative')
        if c>=ps: raise ValueError('there are not so many parents')
        histy3[c]+=1

width=0.25
plt.bar(histx-1.5*width,histy1,width,color='b')
plt.bar(histx-0.5*width,histy2,width,color='g')
plt.bar(histx+0.5*width,histy3,width,color='r')
txt='blue: scores {0} and goal is minimisation'.format(sc1)
txt+='\ngreen: scores {0} and goal is minimisation'.format(sc2)
txt+='\nred: scores {0} and goal is maximisation'.format(sc2)
plt.suptitle(txt,x=0.5,y=0.98,ha='center',va='top')
plt.show()
"""



