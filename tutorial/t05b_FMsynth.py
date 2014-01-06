#!python
"""
peabox tutorial
lesson 05 - FM synthesis (a real-world problem of the CEC-2011 competition)
a) three ways of defining the objective function:
     A: function definition in old-school procedural manner (past lesson)
     B: a simple class definition for the objective function (past lesson)
     C: making more use of object-orientation (this one)

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
#--- C: the same by subclassing Individual
# --    (keeps all together that belongs together)
# --    (if you make changes inside the new class, you don't have to worry about any code outside it)
# --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
"""
What we basically wanna do here is creating a more complicated individual,
where the call to Individual.evaluate() does not only point to a call of an
assigned objective function, but where more complex computation and intermediate
data storage is initiated. The drawback is of course that each instance of the
class Individual consumes more memory. But the benefit is that the internals,
like some intermediate computation quantities, can be accessed afterwards
by inspecting the instance's attributes or calling dedicated analysis methods.

In this case here we need a new evaluation function and we will add a plot
utility able to produce snapshots of solution candidates.
"""

def dummyfunc(x):
    return np.sum(x)

class FMsynthC(Individual):
    # we are inheriting everything an individual has or can do and are just adding more methods
    # the only two methods below, where to pay a little bit attention are the constructor and evaluate()
    # these two are not added but overwritten, so one has to be aware of the functionality of the originals
    # in __init__(self) the overwriting problem is solved by calling the original next to executing the replacement
    # in evaluate(self) the problem is solved by creating the useless slot for the second argument
    def __init__(self,dummyfunc,pspace):
        Individual.__init__(self,dummyfunc,pspace)
        self.a=[1., 1.5, 2.]
        self.w=[5. ,4.8, 4.9]
        self.nt=101
        self.t=arange(self.nt,dtype=float)
        self.theta=2*pi/(self.nt-1)
        self.trial=zeros(self.nt)
        self.target=zeros(self.nt)
        self.initialize_target()
        self.plotpath=join(getcwd(),'plots')
    def initialize_target(self):
        self.fmwave(self.a,self.w)
        self.target[:]=self.trial[:]
    def fmwave(self,a,w):
        t=self.t; th=self.theta
        self.trial[:]=a[0]*sin(w[0]*t*th+a[1]*sin(w[1]*t*th+a[2]*sin(w[2]*t*th)))
    def evaluate(self,tlock=None):
        if tlock is not None:  # in case a threading lock is handed over, omg, what the hell is threading??
            raise ValueError('this individual cannot handle threaded processing') # problem solved, no need to know what threading is a this point, but through the error statement we prevent any hidden bug potentially caused by the momentary lack of knowledge - or should we say the momentary lapse of reason
        self.fmwave( [self.DNA[0],self.DNA[2],self.DNA[4]] , [self.DNA[1],self.DNA[3],self.DNA[5]] )
        self.score=np.sum((self.trial-self.target)**2)
        return self.score
    def plot_FMsynth_solution(self):
        plt.fill_between(self.t,self.target,self.trial,alpha=0.17)
        plt.plot(self.t,self.target,'o-',c='k',label='target',markersize=3)
        plt.plot(self.t,self.trial,'o-',c='g',label='trial',markersize=3)
        plt.ylim(-7,7)
        plt.title('a solution candidate for the\nCEC-2011 FM synthesis problem')
        txt='DNA = {}\nscore = {}'.format(np.round(self.DNA,3),self.score)
        plt.suptitle(txt,x=0.03,y=0.02,ha='left',va='bottom',fontsize=10)
        runlabel='c'+str(self.ncase).zfill(3)+'_g'+str(self.gg).zfill(3)+'_i'+str(self.no).zfill(3)
        plt.savefig(join(self.plotpath,'FMsynth_solution_'+runlabel+'.png'))
        plt.close()


"""
Now, with this somewhat more elegant solution above, we avoid most maintainance
problems that could be inferred from the first two attempts. Here we can modify
the values of the objective function on the fly (like demonstrated below with
population D) and the plotting function is robust enough to handle it. The
plotting method above is even robust against more major modifications to how
the wave functions are calculated as it doesn't rely on the formula, but refers
directly to the results stored in self.target and self.trial.
"""

# search space boundaries:
searchspace=(('amp 1',   -6.4, +6.35),
             ('omega 1', -6.4, +6.35),
             ('amp 2',   -6.4, +6.35),
             ('omega 2', -6.4, +6.35),
             ('amp 3',   -6.4, +6.35),
             ('omega 3', -6.4, +6.35))

ps=4   # population size
dim=len(searchspace)  # search space dimension

pC=Population(FMsynthC,ps,dummyfunc,searchspace)
pC.set_ncase(1)
pC.new_random_genes()
pC.eval_all()
print 'DNAs of pC:'
print pC.get_DNAs()
pC.print_stuff(slim=True)
for dude in pC:
    dude.plot_FMsynth_solution()


# and the modified version
pD=Population(FMsynthC,ps,dummyfunc,searchspace)
pD.set_ncase(2)
for dude in pD:
    dude.w[1]=4.7
    dude.initialize_target()
pD.copy_otherpop(pC)
pD.eval_all()
print 'DNAs of pD:'
print pD.get_DNAs()
pD.print_stuff(slim=True)
for dude in pD:
    dude.plot_FMsynth_solution()


"""
An interesting next step  would be inplementing the option of more than 6
problem dimensions ... probably one might also be able to convince oneself that
the problem complexity grows very nonlinearily with problem dimension
"""

