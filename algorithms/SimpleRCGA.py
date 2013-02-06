#!python
"""
implementation of a simple real-coded genetic algorithm

but please note
a) There is not only the old-scool fitness-value-based selection pressure.
b) By making the F1-population bigger than the F0-population, you can
   create a (mu+lam)-scheme.
"""

import numpy as np
from numpy import zeros, mod, where
from numpy.random import rand, randint
from peabox_helpers import parentselect_exp

class SimpleRCGA:
    def __init__(self,F0pop,F1pop,P_mutation=0.08,CO='uniform',parentselection='roulette',recorder=None):
        self.F0=F0pop
        self.F1=F1pop
        self.m=len(F0pop)     # mu, i.e. size of parent population
        self.l=len(F1pop)     # lambda, i.e. size of F1-generation
        self.Pm=P_mutation    # mutation probability applied to each gene
        self.muttype='randn'  # mutation type: 'randn', 'rand', 'fixstep'
        self.mstep=0.1        # mutation step size in relation to domain width
        self.CO=CO            # crossing-over type: 'uniform', 'BLXa', 'BLXab', 'WHX', or integer means n-point CO
        self.selec=parentselection  # 'roulette' means prop. to fitness, 'expon', 'elite' (then F0pop must be smaller than F1pop)
        self.alpha=0.5        # for BLX operators
        self.beta=0.2         # for BLXab operator
        self.thresh=None      # for fitnessroulette: array containing thresholds to divide the interval [0,1] in segments
        self.generation_callback=None  # any function recieving this EA instance as argument, e.g. plot current best solution
        self.gcallback_interval=10     # execute the generation_callback after every 10th generation
        if recorder is not None:
            self.rec,self.doku=recorder
    def create_offspring(self):
        self.fitnessroulette_ini()
        self.F1.advance_generation()
        for fdude in self.F1:
            if self.selec == 'expon':
                pa=parentselect_exp(self.m,7.)
                pb=parentselect_exp(self.m,7.)
            elif self.selec=='roulette':
                pa,pb=self.fitnessroulette()
                if pb==pa:
                    pb=randint(self.m-1)
                    if pb>=pa: pb+=1
            elif self.selec=='elite':
                pa,pb=randint(self.m,size=2)
                if pb==pa:
                    pb=randint(self.m-1)
                    if pb>=pa: pb+=1
            else:
                raise NotImplementedError("in terms of parent selection the options are 'roulette', 'expon', 'elite'")
            if pa>pb: pa,pb=pb,pa
            if self.CO=='uniform':
                fdude.CO_from(self.F0[pa],self.F0[pb])
                fdude.ancestcode=0.9
            elif self.CO=='BLXa':
                fdude.BLX_alpha(self.F0[pa],self.F0[pb],self.alpha)
                fdude.ancestcode=0.11
            elif self.CO=='BLXab':
                fdude.BLX_alpha_beta(self.F0[pa],self.F0[pb],self.alpha,self.beta)
                fdude.ancestcode=0.19
            elif self.CO=='WHX':
                fdude.WHX(self.F0[pa],self.F0[pb],self.alpha)
                fdude.ancestcode=0.45
            else:
                raise NotImplementedError("in terms of CO the options are 'uniform', 'BLXa', 'BLXab', 'WHX'")
            fdude.mutate(self.Pm,self.mstep)
    def fitnessroulette_ini(self):
        thresh=zeros(self.m)
        sc=self.F0.get_scores()
        cs=1/sc
        for i in range(self.m):
            if self.F0.whatisfit=='minimize':
                thresh[i]=np.sum(cs[:i+1])
            if self.F0.whatisfit=='maximize':
                thresh[i]=np.sum(sc[:i+1])
        thresh/=np.max(thresh)
        self.thresh=thresh
    def fitnessroulette(self):
        r1,r2=rand(2)
        pa=np.sum(where(self.thresh<r1,1,0))
        pb=np.sum(where(self.thresh<r2,1,0))
        return pa,pb
    def advance_generation(self):
        for i,pdude in enumerate(self.F0):
            pdude.copy_DNA_of(self.F1[i],copyscore=True,copyancestcode=True,copyparents=True)
        self.F0.advance_generation()
        self.F0.eval_all()
    def do_step(self):
        self.F0.sort()
        self.create_offspring()
        self.advance_generation()
        self.F0.check_and_note_goal_fulfillment()
        if self.generation_callback is not None and not mod(self.F0.gg,self.gcallback_interval):
            self.generation_callback(self)
        if hasattr(self,'rec'):
            if not mod(self.F0.gg,self.doku): self.rec.save_status()
    def run(self,generations):
        for i in range(generations):
            self.do_step()
            if not mod(i,10): print 'best score: ',self.F0[0].score