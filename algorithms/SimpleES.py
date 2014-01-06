#!python
"""
evolution strategies according to Ingo Rechenberg, TU-Berlin
implementation of (mu,lam)-ES and (mu+lam)-ES
step size control: so far only 1/5th-rule implemented
"""

from copy import deepcopy
import numpy as np
from numpy import mod
from numpy.random import randint



class SimpleES:
    def __init__(self,F0pop,F1pop):
        self.F0=F0pop
        self.F1=F1pop
        self.m=len(F0pop)         # mu, i.e. size of parent population
        self.l=len(F1pop)         # lambda, i.e. size of F1-generation
        self.scheme=','           # should be in [',','Komma','komma','comma'] or ['+','Plus','plus']
        if self.scheme in [',','Komma','komma','comma']:
            assert self.m < self.l    # there must be more offspring than available parent slots, so there is selection pressure
        self.mstep=0.1
        self.adap='1/5th'         # mstep adaptation rule
        self.adapf=1.2            # mstep adaptation factor
        self.generation_callbacks=[]  # any function recieving this EA instance as argument, e.g. plot current best solution
        self.save_best=True
        self.bestdude=None

    def create_offspring(self):
        self.F1.advance_generation()
        for fdude in self.F1:
            fdude.copy_DNA_of(self.F0[randint(self.m)])
            fdude.mutate_fixstep(self.mstep)
        self.F1.eval_all()

    def adapt_stepsize(self):
        if self.adap in ['1/5th','1/5th-rule','1/5-Erfolgsregel','Erfolgsregel']:
            improv=0
            for fdude in self.F1:
                if fdude.isbetter(self.F0[0]): improv+=1
            if improv > self.l/5:  # alternative: if improv > self.l/5+1
                self.mstep*=self.adapf
            else:
                self.mstep/=self.adapf
        elif self.adap in ['const','const.','constant']:
            pass
        else:
            raise NotImplementedError('other stepsize control schemes than the 1/5th success rule are not yet implemented')

    def advance_generation_Plus(self):
        for pdude in self.F0:
            pdude.ancestcode=0.598
        self.F1.reverse()
        for fdude in self.F1:
            if fdude.isbetter(self.F0[-1]):
                self.F0[-1].copy_DNA_of(fdude,copyscore=True)
                self.F0[-1].ancestcode=0.498
                self.F0.sort()
        self.F0.sort()
        self.F0.advance_generation()

    def advance_generation_Komma(self):
        for i,pdude in enumerate(self.F0):
            pdude.copy_DNA_of(self.F1[i],copyscore=True)
            pdude.ancestcode=0.498
        self.F0.sort()  # sort() includes update_scores()   (no sorting needed in principle, F1 should be sorted at this moment)
        self.F0.advance_generation()

    def do_step(self):
        self.F0.sort()
        self.create_offspring()
        self.F1.sort()
        self.adapt_stepsize()
        if self.scheme in [',','Komma','komma','comma']:
            self.advance_generation_Komma()
        elif self.scheme in ['+','Plus','plus']:
            self.advance_generation_Plus()
        else:
            raise NotImplementedError("the ES_scheme constructor argument must be in [',','Komma','komma','comma'] or ['+','Plus','plus']")
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
        return self.F1.neval



