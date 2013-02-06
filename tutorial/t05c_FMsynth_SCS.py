#!python
"""
peabox tutorial
lesson 05 - FM synthesis (a real-world problem of the CEC-2011 competition)
c) applying scatter search to that test function

test function: CEC-2011 test function 1 --> Parameter Estimation for Frequency-Modulated (FM) Sound Waves

what is to be shown here:

 - scatter search is and EA involving different pools containing different amounts of DNAs
   and constantly changing size, but it's okay with this toolbox
   just have a look at the scatter search source code, it is not really commented, but the subroutine
   titles should say everything if you have just been glancing over one of the papers by Glover, Laguna, Marti
 - benefits of having populations behave like lists

what can be learned after having thouroughly read the scatter search papers and after
having had some hands on test function experience with scatter search (e.g. compare it to
the performance of CMA-ES on e.g. the Weierstrass or the eF8F2 test functions of the CEC-2005 suite):
    
 - scatter search stems from the idea of not wasting too much time with random trials
   or more exactly from the wish of avoiding the usage of random where possible
 - difficulty of balancing efforts for global vs. local search

Bibliography:
see scatter search source code

by Markus Stokmaier, IKET, KIT, Karlsruhe, February 2013
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
from peabox_recorder import Recorder
from ScatterSearch import ScatterSearch
from peabox_plotting import ancestryplot

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
    def set_bad_score(self):
        self.score=9999.

def gcb(eaobj):
    sc=eaobj.refset.get_scores()
    bestidx=array(sc).argmin()
    b=eaobj.refset[bestidx]
    oldscore=b.score
    b.evaluate()
    assert b.score==oldscore
    b.plot_FMsynth_solution()
    print 'gcb: score: {}   DNA: {}'.format(b.score,b.DNA)
    print 'gcb: target[:5]: ',b.target[:5]

# search space boundaries:
searchspace=(('amp 1',   -6.4, +6.35),
             ('omega 1', -6.4, +6.35),
             ('amp 2',   -6.4, +6.35),
             ('omega 2', -6.4, +6.35),
             ('amp 3',   -6.4, +6.35),
             ('omega 3', -6.4, +6.35))


dim=len(searchspace)  # search space dimension

ps_ref=20
dset=Population(FMsynthC,40,dummyfunc,searchspace)
rset=Population(FMsynthC,ps_ref,dummyfunc,searchspace)
rec=Recorder(rset)
ea=ScatterSearch(dset,rset,nref=ps_ref,b1=10,recorder=[rec,1])
ea.generation_callback=gcb
ea.gcallback_interval=1

ea.complete_algo(10)

ancestryplot(rec,ylimits=[0,70])


