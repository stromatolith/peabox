#!python
"""
implementation of a simple real-coded genetic algorithm

but please note
a) There is not only the old-scool fitness-value-based selection pressure.
b) By making the F1-population bigger than the F0-population, you can
   create a (mu+lam)-scheme.
"""

from copy import deepcopy
import numpy as np
from numpy import zeros, mod, where
from numpy.random import rand, randint
from peabox_helpers import parentselect_exp


class SimpleRCGA:

    def __init__(self,F0pop,F1pop,P_mutation=0.08,CO='uniform',parentselection='roulette'):
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
        self.generation_callbacks=[]  # (list of)  any function recieving this EA instance as argument, e.g. plot current best solution
        self.save_best=True
        self.bestdude=None

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

#    def do_step(self):
#        self.F0.sort()
#        self.create_offspring()
#        self.advance_generation()
#        self.F0.check_and_note_goal_fulfillment()
#        if self.save_best:
#            self.update_bestdude()
#        for gc in self.generation_callbacks:
#            gc(self)

    def do_step(self):
        self.create_offspring()
        self.advance_generation()
        #self.mstep_control()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def run(self,generations):
        for i in range(generations):
            self.do_step()
            if not mod(i,10): print 'best score: ',self.F0[0].score

    def zeroth_generation(self,random_ini=True):
        if random_ini:
            self.F0.new_random_genes()
        self.F0.zeroth_generation()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def update_bestdude(self):
        if self.bestdude is None:
            self.bestdude=deepcopy(self.F0[0])
        else:
            if self.F0[0].isbetter(self.bestdude):
                self.bestdude.copy_DNA_of(self.F0[0],copyscore=True,copyparents=True,copyancestcode=True,copymutagenes=True)
                self.bestdude.gg=self.F0[0].gg

    def tell_best_score(self,andDNA=False):
        if (self.bestdude is not None) and (self.save_best is not None) and (self.bestdude.isbetter(self.F0[0])):
            if andDNA:
                return self.bestdude.score,self.bestdude.get_copy_of_DNA()
            else:
                return self.bestdude.score
        else:
            if andDNA:
                return self.F0[0].score,self.F0[0].get_copy_of_DNA()
            else:
                return self.F0[0].score

    def tell_neval(self):
        return self.F0.neval




