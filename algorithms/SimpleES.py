#!python
"""
evolution strategies according to Ingo Rechenberg, TU-Berlin
implementation of (mu,lam)-ES and (mu+lam)-ES
step size control: so far only 1/5th-rule implemented
"""

import numpy as np
from numpy import mod
from numpy.random import randint



class SimpleES:
    def __init__(self,F0pop,F1pop,mstep,ES_scheme,stepadapt='1/5th',adapfactor=1.2,recorder=None):
        self.F0=F0pop
        self.F1=F1pop
        self.m=len(F0pop)         # mu, i.e. size of parent population
        self.l=len(F1pop)         # lambda, i.e. size of F1-generation
        self.scheme=ES_scheme     # should be in [',','Komma','komma','comma'] or ['+','Plus','plus']
        if self.scheme in [',','Komma','komma','comma']:
            assert self.m < self.l    # there must be more offspring than available parent slots, so there is selection pressure
        self.mstep=mstep
        self.adap=stepadapt
        self.adapf=adapfactor
        self.generation_callback=None  # any function recieving this EA instance as argument, e.g. plot current best solution
        self.gcallback_interval=10     # execute the generation_callback after every 10th generation
        if recorder is not None:
            self.rec,self.doku=recorder
            # the recorder needs to be related to offspring (not parents) if all
            # trials are to be remembered in the case of lambda > mu
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
        if self.generation_callback is not None and not mod(self.F0.gg,self.gcallback_interval):
            self.generation_callback(self)
        if hasattr(self,'rec'):
            if not mod(self.F0.gg,self.doku): self.rec.save_status()
    def run(self,generations):
        for i in range(generations):
            self.do_step()
            if not mod(i,10): print 'best score: ',self.F0[0].score

