#!python
"""
I found out that CMA-ES by Nikolaus Hansen and A. Ostermeier is pretty efficient,
in particular when while averaging over the whole population cloud the search
landscape offers some useful large-scale gradient information, i.e. when a global
potential-bathtub is superpositioned by noise or ripples.

Here I tried to make the algorithm fit for use with my population and individual
classes, not least because now you can compare the ancestry plots with those
produced by other algorithm approaches. But I guess for computational efficiency
this here will be much worse than the original.

Nikolaus Hansen hosts codes for CMA-ES in various languages and very good
presentation slides on his website:
http://www.lri.fr/~hansen/

one of his main papers:
Hansen, N. and A. Ostermeier (2001). Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation, 9(2), pp. 159-195
"""
from __future__ import division

from cPickle import Pickler
from copy import copy, deepcopy

import numpy as np
from numpy import exp, log, sin, cos, tan
from numpy import arange, array, ones, zeros, diag, eye, mod
from numpy import add, prod, dot, vdot, inner, outer  # inner=Skalarprodukt, outer=Tensorprodukt
from numpy.random import rand, randn, randint, multivariate_normal
from numpy.linalg import inv, eig, eigh, qr



class CMAES:

    def __init__(self,F0pop,mu=None,recorder=None):   #,uCS=True):
        self.F0=F0pop
        self.l=len(F0pop)     # lambda, i.e. size of F1-generation
        if mu is not None: self.m=mu
        else: self.m=int(self.l/2)     # mu, i.e. size of parent population
        self.ng=self.F0.ng
        #self.uCS=uCS
        if recorder is not None:
            self.rec=recorder
            self.rec_interval=1
            self.rec.snames.append('mstep')
            self.rec.scmd['mstep']="moreprops['mstep']"
            self.rec.anames.append('mutagenes')
            self.rec.acmd['mutagenes']='[0].mutagenes'
            self.rec.reinitialize_data_dictionaries()
        else:
            self.rec_interval=0
        self.weights = log(self.m+0.5) - log(arange(1,self.m+1)) # recombination weights
        self.weights /= np.sum(self.weights) # normalize recombination weights array
        self.mueff=np.sum(self.weights)**2 / np.sum(self.weights**2) # variance-effectiveness of sum w_i x_i
        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff/self.ng) / (self.ng+4 + 2 * self.mueff/self.ng)  # time constant for cumulation for C
        self.cs = (self.mueff + 2) / (self.ng + self.mueff + 5)  # t-const for cumulation for sigma control
        self.c1 = 2 / ((self.ng + 1.3)**2 + self.mueff)     # learning rate for rank-one update of C
        self.cmu = 2 * (self.mueff - 2 + 1/self.mueff) / ((self.ng + 2)**2 + self.mueff)  # and for rank-mu update
        self.damps = 2 * self.mueff/self.l + 0.3 + self.cs  # damping for sigma, usually close to 1
        # Initialize dynamic (internal) state variables and constants
        self.initialize_internals()
        self.F0.moreprops['mstep']=self.mstep
        self.boundary_treatment='mirror'
        self.save_best=True
        self.bestdude=None
        self.maxsigma=None
        self.c1a_adjustment=True
        self.generation_callback=None  # any function recieving this EA instance as argument, e.g. plot current best solution
        self.gcallback_interval=0     # execute the generation_callback after every 10th generation
        self.more_stop_crits=[]
        
    def initialize_internals(self):
        # Initialize dynamic (internal) state variables and constants
        self.mstep=0.1
        self.xmean=zeros(self.ng)
        self.pc = zeros(self.ng)
        self.ps = zeros(self.ng)  # evolution paths for C and sigma
        self.B = eye(self.ng)   # B defines the coordinate system 
        self.D = ones(self.ng)  # diagonal D defines the scaling
        self.C = eye(self.ng)   # covariance matrix 
        self.invsqrtC = eye(self.ng)  # C^-1/2
        self.F0.reset_all_mutagenes(1.)
        self.goodDNA=zeros((self.m,self.ng))
        self.neval=0
        self.last_cm_update=0

    def sorting_procedure(self):
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
    
    def mvn_mutation(self):
        #  mutates each individual of self.F0 according to the current multivariate normal distribution
        newDNAs=multivariate_normal(self.xmean,self.C,size=self.F0.psize)
        for i,dude in enumerate(self.F0):
            dude.set_uDNA(newDNAs[i,:])
            if self.boundary_treatment=='mirror': dude.mirror_DNA_into_bounds()
            elif self.boundary_treatment=='mirror_and_scatter': dude.mirror_and_scatter_DNA_into_bounds()
            elif self.boundary_treatment=='cycle': dude.cycle_DNA_into_bounds()
            elif self.boundary_treatment=='cycle_and_scatter': dude.cycle_and_scatter_DNA_into_bounds()
            elif self.boundary_treatment=='scatter': dude.randomize_DNA_into_bounds()
            else: pass   #elif self.boundary_treatment=='nobounds': pass

    def do_step(self):
        self.advance_generation()
        self.update_internals()
        if self.generation_callback is not None and self.gcallback_interval and not mod(self.F0.gg,self.gcallback_interval):
            self.generation_callback(self)
        self.record_stuff()
        if self.save_best:
            if self.bestdude is None:
                self.bestdude=deepcopy(self.F0[0])
            else:
                if self.F0[0].isbetter(self.bestdude):
                    self.bestdude.copy_DNA_of(self.F0[0],copyscore=True,copyparents=True,copyancestcode=True)
                    self.bestdude.gg=self.F0[0].gg
        
    def advance_generation(self):
        # Generate and evaluate lam offspring
        self.F0.advance_generation()
        for dude in self.F0:
            dude.set_uDNA(self.xmean + self.mstep*dot(self.B,dude.mutagenes*randn(self.ng)))
            if self.boundary_treatment=='mirror': dude.mirror_DNA_into_bounds()
            elif self.boundary_treatment=='mirror_and_scatter': dude.mirror_and_scatter_DNA_into_bounds()
            elif self.boundary_treatment=='cycle': dude.cycle_DNA_into_bounds()
            elif self.boundary_treatment=='cycle_and_scatter': dude.cycle_and_scatter_DNA_into_bounds()
            elif self.boundary_treatment=='scatter': dude.randomize_DNA_into_bounds()
            else: pass   #elif self.boundary_treatment=='nobounds': pass
        self.F0.eval_all()
        self.neval+=self.F0.psize
        self.sorting_procedure()
        self.F0.check_and_note_goal_fulfillment()
    
    def update_internals(self):
        xold = copy(self.xmean)
        for i,dude in enumerate(self.F0[:self.m]):
            self.goodDNA[i,:]=dude.get_uDNA()
        self.xmean = dot(self.goodDNA.T,self.weights)  # recombination, new mean value
        # Cumulation: Update evolution paths 
        self.ps = (1-self.cs)*self.ps + ((self.cs*(2-self.cs)*self.mueff)**0.5/self.mstep)*dot(self.invsqrtC,(self.xmean-xold))  # see slide 86 in Hansen's PPSN 2006 CMA Tutorial Talk
        hsig = np.sum(self.ps**2)/(1-(1-self.cs)**(2*self.neval/self.l))/self.ng < 2 + 4/(self.ng+1)
        self.pc = (1-self.cc)*self.pc + ((self.cc*(2-self.cc)*self.mueff)**0.5/self.mstep)*hsig*(self.xmean-xold)
        # Adapt covariance matrix C
        Z = (self.goodDNA-xold) / self.mstep
        Z = dot((self.cmu * self.weights) * Z.T, Z)  # learning rate integrated
        if self.c1a_adjustment==True:
            c1a = self.c1 - (1-hsig**2) * self.c1 * self.cc * (2-self.cc)  # minor adjustment for variance loss by hsig
            self.C = (1 - c1a - self.cmu) * self.C + outer(self.c1 * self.pc, self.pc) + Z
        else:
            self.C = (1 - self.c1 - self.cmu) * self.C + outer(self.c1 * self.pc, self.pc) + Z        
        # Adapt step size sigma with factor <= exp(1/2) \approx 1.65
        self.mstep *= exp(np.min((0.5, (self.cs/self.damps) * (np.sum(self.ps**2) / self.ng - 1) / 2),axis=0))  # this is the alternative - see p 18-19 of his tutorial article
        if self.maxsigma is not None: self.mstep=min(self.mstep,self.maxsigma)
        self.F0.moreprops['mstep']=self.mstep        
        # Eigendecomposition: update B, D and invsqrtC from C 
        if self.neval - self.last_cm_update > self.l/(self.c1+self.cmu)/self.ng/10:  # to achieve O(N^2)
            self.D,self.B = eigh(self.C)              # eigen decomposition, B==normalized eigenvectors
            self.D = self.D**0.5   # D contains standard deviations now (being a 1D array)
            Darr = diag(1./self.D)  # is a 2D array
            self.invsqrtC = dot(self.B,dot(Darr,self.B.T))    # proof: print dot(invsqrtC.T,invsqrtC)-inv(C)        
            self.last_cm_update = self.neval
            self.F0.reset_all_mutagenes(self.D)

    def run(self,generations,reset_CMA_state=True,xstart='best'):
        # if xstart can be a DNA vector/list
        # if xstart=='continue': starting point based on weighted DNAs of the best mu individuals
        # if xstart=='best': start with the DNA of self.F0[0]
        # else: start with mean of all current DNAs, i.e. the current center of the cloud
        if reset_CMA_state:
            if self.F0.psize != self.l:
                self.F0.change_size(self.l)    # parent population should only be of size mu, and not as big as lambd; the bigger size here is only for the purpose of scorehistory recording of all trials within the daughter population
            self.F0.reset_all_mutagenes(1.)
            self.initialize_internals()
        for dude in self.F0:
            dude.pa=0; dude.pb=-1; dude.ancestcode=0.87
        if xstart=='best':
            sDNA=self.F0[0].get_uDNA(); self.xmean=sDNA; [dude.set_uDNA(sDNA) for dude in self.F0]
        elif type(xstart) in [np.ndarray,list,tuple]:
            [dude.set_DNA(xstart) for dude in self.F0]; self.xmean=self.F0[0].get_uDNA()
        elif xstart=='continue':
            pass
        else:  # including xstart=='mean'
            self.xmean=self.F0.mean_DNA(uCS=True); [dude.set_uDNA(self.xmean) for dude in self.F0]
        for gg in range(generations):
            self.do_step()
            if len(self.more_stop_crits) != 0:
                morestopvals=[]
                for crit in self.more_stop_crits:
                    morestopvals.append(crit(self))
                if True in morestopvals:
                    print "algorithm's run() terminated because of '+str(morestopvals.index(True))+'th additional stopping criterion"
                    break
        if hasattr(self,'rec'):
            self.rec.save_goalstatus()
        return

    def record_stuff(self):
        #self.pickle_self()
        if hasattr(self,'rec'):
            if self.rec_interval and not mod(self.F0.gg,self.rec_interval):
                #self.F0.pickle_self()
                self.rec.save_status()
    
    def pickle_self(self):
        ofile=open(self.F0.picklepath+'/EA_'+self.F0.label+'.txt', 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()            
         




