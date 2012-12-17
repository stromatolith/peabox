#!python
"""
peabox tutorial
lesson 05 - FM synthesis (a real-world problem of the CEC-2011 competition)
a) three ways of defining the objective function:
     A: function definition in old-school procedural manner
     B: a simple class definition for the objective function
     C: making more use of object-orientation (next one)

test function: CEC-2011 test function 1 --> Parameter Estimation for Frequency-Modulated (FM) Sound Waves

what is to be shown here:

 - when and how object-orientation makes sense
 - glimpse of advantages of subclassing peabox's individual class

by Markus Stokmaier, IKET, KIT, Karlsruhe, December 2012
"""

from os import getcwd
from os.path import join
import numpy as np
from numpy.random import seed, rand, randn, randint
from numpy import array, arange, zeros, ones
from numpy import mean, std
from numpy import pi, exp, sin, cos
#import matplotlib as mpl
import matplotlib.pyplot as plt
from time import clock
from peabox_individual import Individual
from peabox_population import Population

# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
#--- A: function definition in old-school procedural manner
# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
def FMsynthA(x):
    """ x is a 1D array of length 6 looking like this: [amp1, w1, amp2, w2, amp3, w3]"""
    nt=101
    t=arange(nt,dtype=float)
    theta=2*pi/(nt-1)
    target=1.*sin(5.*t*theta+1.5*sin(4.8*t*theta+2.*sin(4.9*t*theta)))
    trial=x[0]*sin(x[1]*t*theta+x[2]*sin(x[3]*t*theta+x[4]*sin(x[5]*t*theta)))
    return np.sum((trial-target)**2)


# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
#--- B: a simple class definition for the objective function
# --    (a similar definition can also be found in peabox_testfuncs.py)
# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
class FMsynthB(object):
    def __init__(self):
        self.a=[1., 1.5, 2.]
        self.w=[5. ,4.8, 4.9]
        self.nt=101
        self.t=arange(self.nt,dtype=float)
        self.theta=2*pi/(self.nt-1)
        self.trial=zeros(self.nt)
        self.target=zeros(self.nt)
        self.initialize_target()
    def initialize_target(self):
        self.fmwave(self.a,self.w)
        self.target[:]=self.trial[:]
    def fmwave(self,a,w):
        t=self.t; th=self.theta
        self.trial[:]=a[0]*sin(w[0]*t*th+a[1]*sin(w[1]*t*th+a[2]*sin(w[2]*t*th)))
    def call(self,DNA):
        self.fmwave( [DNA[0],DNA[2],DNA[4]] , [DNA[1],DNA[3],DNA[5]] )
        return np.sum((self.trial-self.target)**2)

# search space boundaries:
searchspace=(('amp 1',   -6.4, +6.35),
             ('omega 1', -6.4, +6.35),
             ('amp 2',   -6.4, +6.35),
             ('omega 2', -6.4, +6.35),
             ('amp 3',   -6.4, +6.35),
             ('omega 3', -6.4, +6.35))

ps=4   # population size
dim=len(searchspace)  # search space dimension

# now let's create a population to test version A of the objective function implementation
pA=Population(Individual,ps,FMsynthA,searchspace)
pA.new_random_genes()
pA.eval_all()
print 'DNAs of pA:'
print pA.get_DNAs()
pA.print_stuff(slim=True)

# and here the objective function version B and another population to test it
problem_instance=FMsynthB()
objfuncB=problem_instance.call
pB=Population(Individual,ps,objfuncB,searchspace)
pB.copy_otherpop(pA)
pB.eval_all()
print 'DNAs of pB:'
print pB.get_DNAs()
pB.print_stuff(slim=True)


# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
#--- plot routine: solution candidate visualisation
# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
"""
The really good thing about this FM synthesis test problem is that it is a
difficult six-dimensional test problem, but it is easy to plot a solution
candidate and judge at the first glance whether a solution is good or not. I
think how essential the combination of these two properties is, cannot be
stressed enough.

When inventing or working with evolutionary algorithms, you
definitely need difficult enough test problems. There are many such problems
available, but you always have to make huge statistics in order to get the
performance of an EA on it. But if you have a test problem like this one, just
look at how the plot of the best solution in the population develops over time
while makeing changes to your EA: does the best solution become better? how
quickly does it improve (starting always from random solutions)?

Of course for algorithm fine-tuning you won't be able to avoid huge statistics,
but for first order tinkering, looking at how the best solution develops during
two or three EA runs suffices. Having such a difficult and telling test problem
where you can make a plot and where a human brain can judge on the solution
quality in a couple of miliseconds speeds you up a lot in your trial-and-error
iteration cycle while coding.
"""

def plot_FMsynth_solution(DNA,path):
    nt=101
    t=arange(nt,dtype=float)
    theta=2*pi/(nt-1)
    target=1.*sin(5.*t*theta+1.5*sin(4.8*t*theta+2.*sin(4.9*t*theta)))
    trial=DNA[0]*sin(DNA[1]*t*theta+DNA[2]*sin(DNA[3]*t*theta+DNA[4]*sin(DNA[5]*t*theta)))
    plt.fill_between(t,target,trial,alpha=0.17)
    plt.plot(t,target,'o-',c='k',label='target',markersize=3)
    plt.plot(t,trial,'o-',c='g',label='trial',markersize=3)
    plt.ylim(-7,7)
    plt.title('a solution candidate for the\nCEC-2011 FM synthesis problem')
    txt='DNA = {}\nscore = {}'.format(np.round(DNA,3),FMsynthA(DNA))
    plt.suptitle(txt,x=0.03,y=0.02,ha='left',va='bottom',fontsize=10)
    plt.savefig(path)
    plt.close()

loc=join(getcwd(),'plots')   # make sure there is such a folder to put the pictures
print '\n\nmaking plots ...'
for i,dude in enumerate(pA):
    picname=join(loc,'FMsynth_trial_'+str(i)+'.png')
    plot_FMsynth_solution(dude.DNA,picname)


"""
now for judging the differences between the two problem implementations:
The classical procedural implementation looks of course super-simple, it is
straightforward, it is coded most quickly without much thinking required and
reading and understanding the function's source code goes also quickly and
easily -- so why go through the hassle and seeming complication of object-
orientation, of pumping the simple thing up into several initialisation and
computation subroutines, the class methods?

A first reason is computation efficiency: there is no necessity to recalculate
the array of target values each time. Storing it once and forever when
creating an instance of FMsynthB avoids that. Let's see how big a difference it
makes:
"""
print '\n\ntime comparison procedural vs. object-oriented problem formulation'
m=1000
seed(1)
t0=clock()
for i in range(m):
    pA.new_random_genes()
    pA.eval_all()
t1=clock()
print 'time consumed by ',ps*m,' calls to FMsynthA: ',t1-t0

seed(1)
t0=clock()
for i in range(m):
    pB.new_random_genes()
    pB.eval_all()
t1=clock()
print 'time consumed by ',ps*m,' calls to FMsynthB: ',t1-t0
print 'to understand this work through commented source code\n\n'


"""
The real argument:

Maybe that efficiency stuff is interesting for you, maybe not. For an
engineer's real optimisation problems computation efficiency is often crucial.
But for tinkering around with EA prototypes on simple test problems, it is
rather your own trial and error iteration cycle, your own human thinking
process, which forms the bottleneck, at least from my experience with my brain
this was often so, I can't speak of anybody else.

So here comes the thing:
a) an object keeps variables together in a bunch that logically belong together
b) in an interacting zoo of procedurally programmed functions and subroutines,
after having changed one thing, you often end up having to change a whole lot
of other bits of the source code, and you are likely to miss something some-
times leading to more debugging work
"""

# about a)
# imagine you want to slightly modify the test problem, say w2=4.7 instead of 4.8
# in the case of FMsynthB this is easy:
modified_problem_instance=FMsynthB()
modified_problem_instance.w[1]=4.7
modified_problem_instance.initialize_target()
x=pA[0].get_DNA()
print 'original and modified problem score for DNA ',x
print problem_instance.call(x),modified_problem_instance.call(x),'\n\n'
# imagine you don't remember exactly any more how the two problem instances were created,
# no problem, you can just inquire about the values of the attributes modified_problem_instance.a
# and modified_problem_instance.w, but with FMsynthA there is now way to ask about the internals!

# about b)
# after such a change the old plotting routine of course doesn't fit any more!
# The mess is immediately visible!

"""
So in the next step let's code this up so such problems have no chance of
ever shoing up, i.e. let's produce robust and maintainance-friendly code.
"""
