#!python
"""
a collection of evolutionary algorithms
"""
#from time import time, localtime
from cPickle import Pickler#, Unpickler
#from copy import copy#, deepcopy

import numpy as np
from numpy import exp, mod, hamming, convolve
#from numpy import argmin, argmax, sort, argsort, mod, roll, floor, ceil, clip, shape, size, linspace
#from numpy import sign, sqrt, exp, log, log10, pi, sin, cos, tan, mean, std
#from numpy import arange, array, ones, zeros, diag, eye, ravel, asarray, asfarray
#from numpy import take, where, flipud, fliplr, transpose, float32, append, any, all
#from numpy import add, prod, dot, vdot, inner, outer  # inner=Skalarprodukt, outer=Tensorprodukt
from numpy.random import rand#, randn, randint

from os.path import join
import matplotlib.pyplot as plt

from peabox_population import Population
from peabox_recorder import Recorder
from peabox_helpers import parentselect_exp


class SAGA_Population(Population):
    def __init__(self,species,popsize,objfunc,paramspace):
        Population.__init__(self,species,popsize,objfunc,paramspace)
        self.sa_improved=0
        self.sa_tolerated=0
        self.sa_events=0
        self.sa_improverate=0.
        self.sa_toleraterate=0.
        self.ga_improved=0
        self.ga_events=0
        self.ga_improverate=0.
    def reset_event_counters(self):
        self.sa_improved=0
        self.sa_tolerated=0
        self.sa_events=0
        self.ga_improved=0
        self.ga_events=0
    def calculate_event_rates(self):
        if self.sa_events==0:
            self.sa_improverate=0
        else:
            self.sa_improverate=self.sa_improved/float(self.sa_events)
            self.sa_toleraterate=self.sa_tolerated/float(self.sa_events)
        if self.ga_events==0:
            self.ga_improverate=0
        else:
            self.ga_improverate=self.ga_improved/float(self.ga_events)


class SAGA:
    """
    a concoction of simulated annealing plugged into a GA framework
    """

    def __init__(self,F0pop,F1pop,userec=False):
        self.F0=F0pop  # parent population, nomenclature like in your biology book
        self.F1=F1pop  # offspring population, nomenclature like in your biology book
        self.sa_T=1. # temperature
        self.saga_ratio=0.5 # fraction of offspring created by SA and not GA
        self.sa_mstep=0.05 # mutation step size parameter
        self.sa_mprob=0.6 # mutation probability
        self.ga_selp=3. # selection pressure
        self.AE=0.08 # annealing exponent --> exp(-AE) is multiplier for reducing temperature
        self.elite_size=2
        self.reduce_mstep=False
        if userec:
            self.rec=Recorder(F0pop)
            self.rec.snames.append('sa_improverate') # appending a scalar property's name to the survey list
            self.rec.snames.append('sa_toleraterate')
            self.rec.snames.append('ga_improverate')
            self.rec.reinitialize_data_dictionaries()
            self.userec=True
        else:
            self.rec=None
            self.userec=False
    
    def initialize_temperature(self):
        # assumption: parent population already evaluated and sorted
        self.sa_T=np.fabs(self.F0[-1].score-self.F0[0].score)
    
    def create_offspring(self):
        self.F1.advance_generation()
        self.F0.reset_event_counters()
        for i,dude in enumerate(self.F1):
            oldguy=self.F0[i]
            if i<self.elite_size:
                dude.copy_DNA_of(oldguy,copyscore=True)
                dude.ancestcode=0.18  # blue-purple for conserved dudes
            elif rand() < self.saga_ratio:
                self.F0.sa_events+=1
                dude.copy_DNA_of(oldguy)
                dude.mutate(self.sa_mprob,sd=self.sa_mstep)
                dude.evaluate()
                if dude.isbetter(oldguy):
                    self.F0.sa_improved+=1
                    dude.ancestcode=0.49 # turquoise for improvement through mutation
                elif rand() < exp(-np.fabs(dude.score-oldguy.score)/self.sa_T):  # use of fabs makes it neutral to whatever self.F0.whatisfit
                    self.F0.sa_tolerated+=1
                    dude.ancestcode=0.25 # yellow/beige for tolerated dude
                else:
                    dude.ancestcode=0.18  # blue-purple for conserved dudes
                    dude.copy_DNA_of(oldguy,copyscore=True) # preferring parent DNA
            else:
                self.F0.ga_events+=1
                pa,pb=parentselect_exp(self.F0.psize,self.ga_selp,size=2)
                dude.CO_from(self.F0[pa],self.F0[pb])
                dude.evaluate()
                dude.ancestcode=0.39 # white for CO dude
                # possible changes: accept CO dude only after SA criterion else take better parent (and mutate?)
                if dude.isbetter(self.F0[pa]) and dude.isbetter(self.F0[pb]):
                    self.F0.ga_improved+=1
        self.F0.calculate_event_rates()
        self.F0.neval+=self.F0.psize-self.elite_size  # manually updating counter of calls to the objective function
        # sorry about the line above, I see this is really just a messy workaround
        # there are actually two reasons:
        # a) we evaluated dudes of F1 and we act up as if it happened with F0, but it is even messier
        # b) so far Population.neval is automatically updated only when using a Population method like
        # Population.eval_all() or eval_bunch(), but here we directly had to use the Individual's method evaluate()
            
    def advance_generation(self):
        for pdude,odude in zip(self.F0,self.F1):
            pdude.copy_DNA_of(odude,copyscore=True,copyancestcode=True,copyparents=True)
        self.F0.advance_generation()
    
    def do_step(self):
        self.create_offspring()
        self.advance_generation()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.userec: self.rec.save_status()
            
    def run(self,generations):
        for i in range(generations):
            self.do_step()
            self.sa_T*=exp(-self.AE)
            if self.reduce_mstep:
                self.sa_mstep*=exp(-self.AE)
    
    def simple_run(self,generations,Tstart=None):
        self.F0.new_random_genes()
        self.F0.zeroth_generation()
        if Tstart is not None: self.sa_T=Tstart
        else: self.initialize_temperature()
        if self.userec: self.rec.save_status()
        self.run(generations)



def plot_improvement_rates(rec):
    p=rec.p
    gg=rec.gg
    saimp=rec.sdat['sa_improverate']
    satol=rec.sdat['sa_toleraterate']
    gaimp=rec.sdat['ga_improverate']
    plt.plot(gg,saimp,'cx',label='SA improve rate')
    plt.plot(gg,satol,'yx',label='SA tolerate rate')
    plt.plot(gg,gaimp,'g+',label='GA improve rate')
    
    # low-pass filtering:
    win=hamming(10)
    lp_saimp=convolve(saimp,win,mode='same')/np.sum(win)
    lp_satol=convolve(satol,win,mode='same')/np.sum(win)
    lp_gaimp=convolve(gaimp,win,mode='same')/np.sum(win)
    plt.plot(gg,lp_saimp,'c-',lw=2)
    plt.plot(gg,lp_satol,'y-',lw=2)
    plt.plot(gg,lp_gaimp,'g-',lw=2)
    
    plt.legend()
    plt.ylim(0,1)
    plt.savefig(join(p.plotpath,'improrates_'+p.label+'.png'))
    plt.close()