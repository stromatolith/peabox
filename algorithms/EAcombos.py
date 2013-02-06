#!python
"""
a collection of EA trials
"""

from cPickle import Pickler
import numpy as np
from numpy import zeros, mod, floor, exp, asfarray, mean
from numpy.random import rand, randn, randint

from peabox_population import population_like
from peabox_helpers import parentselect_exp



class ComboBase:

    def __init__(self,F0pop,F1pop,recorder=None):
        assert len(F0pop)==len(F1pop)
        self.F0=F0pop
        self.F1=F1pop
        self.ps=F0pop.psize
        if recorder is not None:
            self.rec=recorder
        self.rec_interval=1
        self.pickle_interval=10
        self.generation_callback=None  # any function recieving this EA instance as argument, e.g. plot current best solution
        self.gcallback_interval=0     # execute the generation_callback after every 10th generation
        self.more_stop_crits=[]

    def advance_generation(self):
        for i,pdude in enumerate(self.F0):
            pdude.copy_DNA_of(self.F1[i],copyscore=False,copyancestcode=True,copyparents=True)
        self.F0.advance_generation()
        self.F0.eval_all()

    def do_step(self):
        self.create_offspring()
        self.advance_generation()
        self.mstep*=exp(-self.anneal)
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        self.F0.check_and_note_goal_fulfillment()
        if self.generation_callback is not None and self.gcallback_interval and not mod(self.F0.gg,self.gcallback_interval):
            self.generation_callback(self)
        self.record_stuff()

    def run(self,generations):
        for i in range(generations):
            self.do_step()
            if len(self.more_stop_crits) != 0:
                morestopvals=[]
                for crit in self.more_stop_crits:
                    morestopvals.append(crit(self))
                if True in morestopvals:
                    print "algorithm's run() terminated because of '+str(morestopvals.index(True))+'th additional stopping criterion"
                    break

    def simple_run(self,generations,random_starting_DNAs=True):
        if random_starting_DNAs:
            self.F0.new_random_genes()
        self.F0.zeroth_generation()
        if self.generation_callback is not None and self.gcallback_interval and not mod(self.F0.gg,self.gcallback_interval):
            self.generation_callback(self)
        self.record_stuff()
        self.run(generations)

    def simple_run_with_seed_DNAs(self,generations,seedDNAs):
        self.make_bunchlists()
        self.F0.new_random_genes()
        for i,dna in enumerate(seedDNAs):
            self.F0[i].set_DNA(dna)
        self.F0.zeroth_generation()
        if self.generation_callback is not None and self.gcallback_interval and not mod(self.F0.gg,self.gcallback_interval):
            self.generation_callback(self)
        self.record_stuff()
        self.run(generations)

    def record_stuff(self):
        if hasattr(self,'rec') and self.rec_interval and not mod(self.F0.gg,self.rec_interval):
            self.rec.save_status()
            self.rec.save_goalstatus()
        if self.pickle_interval and not mod(self.F0.gg,self.pickle_interval):
            self.F0.pickle_self()
            self.F0.write_popstatus()
            self.pickle_self()

    def pickle_self(self):
        ofile=open(self.F0.picklepath+'/EA_'+self.F0.label+'.txt', 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()




class ComboA(ComboBase):
    """
    ES + GA + DE + exponential decay of mutation step sizes
    """

    def __init__(self,F0pop,F1pop,recorder=None):
        ComboBase.__init__(self,F0pop,F1pop,recorder)
        self.ES_mu=self.ps/3
        self.ES_lam=self.ps
        self.ES_Pm=0.          # mutation probability applied to each gene
        self.GA_Pm=0.          # mutation probability applied to each gene
        self.DE_Pm=0.          # mutation probability applied to each gene
        #self.muttype='randn'   # mutation type: 'randn', 'rand', 'fixstep'
        self.ES_mstep=0.0        # mutation step size in relation to domain width
        self.GA_mstep=0.0        # mutation step size in relation to domain width
        self.DE_mstep=0.0        # mutation step size in relation to domain width
        self.anneal=0.0       # exponential decay of mutation step size from generation to generation
        self.ES_selp=0.         # parent selection pressure for ES
        self.GA_selp=0.         # parent selection pressure for GA
        self.DE_selp=0.         # parent selection pressure for DE
        self.DEsr=[0.2,0.8]     # DE scaling range: difference vectors will be scaled with uniformly sampled random number from within this interval
        self.nES=0
        self.nGA=0
        self.nDE=0
        self.EStier=[]
        self.GAtier=[]
        self.DEtier=[]
        self.initialize_tiers([1.,1.,1.])
    
    def initialize_tiers(self,relsizes):
        relsizes=asfarray(relsizes)
        self.nES=int(np.round(self.ps*relsizes[0]/np.sum(relsizes)))
        self.nGA=int(np.round(self.ps*relsizes[0]/np.sum(relsizes)))
        self.nDE=self.ps-self.nES-self.nGA
        self.EStier=self.F1[:self.nES]
        self.GAtier=self.F1[self.nES:self.nES+self.nGA]
        self.DEtier=self.F1[-self.nDE:]
        for dude in self.F1:
            flag1=int(dude in self.EStier)
            flag2=int(dude in self.GAtier)
            flag3=int(dude in self.DEtier)
            assert flag1+flag2+flag3==1
        assert self.EStier.psize+self.GAtier.psize+self.DEtier.psize==self.ps
        
    def create_offspring(self):
        self.F1.advance_generation()
        self.EStier[0].become_mixture_of_multiple(self.F0[:self.ES_mu])
        for fdude in self.EStier[1:]:   # the f in fdude refers to the latin filia/filius
            fdude.copy_DNA_of(self.F1[0])
            fdude.mutate(self.ES_Pm,self.ES_mstep)
            fdude.ancestcode=0.08
        for fdude in self.GAtier:   # the f in fdude refers to the latin filia/filius
            parenta,parentb=parentselect_exp(self.ps,self.GA_selp,size=2)
            fdude.CO_from(self.F0[parenta],self.F0[parentb])
            fdude.mutate(self.GA_Pm,self.GA_mstep)
            fdude.ancestcode=0.15
        DEsri=self.DEsr[0]; DEsrw=(self.DEsr[1]-self.DEsr[0])
        for fdude in self.DEtier:   # the f in fdude refers to the latin filia/filius
            parenta=parentselect_exp(self.ps,self.GA_selp)
            seq=rand(self.ps).argsort(); seq=list(seq); parenta_index=seq.index(parenta); seq.pop(parenta_index) # exclude already selected parent
            parentb,parentc=seq[:2]  # these two parents chosen without selection pressure; all 3 parents are different
            m=DEsri+DEsrw*rand() # scaling of the difference vector to be added to parenta's DNA
            fdude.set_DNA( self.F0[parenta].DNA  +  m * (self.F0[parentc].DNA-self.F0[parentb].DNA) )
            fdude.mirror_and_scatter_DNA_into_bounds()
            fdude.ancestcode=0.25
            if (self.DE_mstep!=0) and (self.DE_Pm!=0): fdude.mutate(self.DE_Pm,self.DE_mstep)




class ComboB(ComboBase):
    """
    The evolutionary algorithm presented at the GAMM-Tagung 2012 in Darmstadt
    Stokmaier, M. J., Class, A. G., Schulenberg, T. and Lahey, R. T. (2012), Sonofusion: EA optimisation of acoustic resonator. Proc. Appl. Math. Mech., 12: 623-624, doi: 10.1002/pamm.201210300
    http://onlinelibrary.wiley.com/doi/10.1002/pamm.201210300/abstract
    """

    def __init__(self,F0pop,F1pop,recorder=None):
        ComboBase.__init__(self,F0pop,F1pop,recorder)
        self.m=len(F0pop)     # mu, i.e. size of parent population
        self.l=len(F1pop)     # lambda, i.e. size of F1-generation
        self.Pm=0.0           # mutation probability applied to each gene
        #self.muttype='randn'   # mutation type: 'randn', 'rand', 'fixstep'
        self.mstep=0.0        # mutation step size in relation to domain width
        self.anneal=0.0       # exponential decay of mutation step size from generation to generation
        self.eml=0.0          # elite mutation limit, the maximal mutation step size multiplier for elite members
        self.cigar2uniform=0.0# what ratio between the two CO operators
        self.WHX2BLX=0.       # what ratio between the two CO operators
        self.mr=0.0           # mutation strength multiplier for recombinants
        self.selpC=0.         # parent selection pressure for bunch C
        self.selpD=0.         # parent selection pressure for bunch D
        self.selpE=0.         # parent selection pressure for bunch E
        self.selpF=0.         # parent selection pressure for bunch E
        self.cigar_aspect=1.  # aspect ratio for cigar_CO()
        self.bunchsizes=None  # e.g. [4,12,24,30,10]  for the sizes of bunches A-E if the population size is 80
        self.storageA=None    # placeholder for an additional population (needed for fractal_run)
        self.storageB=None    # placeholder for an additional population (needed for fractal_run)

    def set_bunchsizes(self,bs):
        bssum=np.sum(bs)
        if bssum != self.F0.psize:
            raise ValueError('all bunch sizes added together must equal the population size!')
        elif len(bs)!=6:
            raise ValueError('there must be six bunch size parameters!')
        else:
            self.bunchsizes=bs

    def make_bunchlists(self):
        self.F1.update_no()
        cut1=self.bunchsizes[0]; cut2=np.sum(self.bunchsizes[:2]); cut3=np.sum(self.bunchsizes[:3])
        cut4=np.sum(self.bunchsizes[:4]); cut5=np.sum(self.bunchsizes[:5])
        self.bunchA=self.F1[0:cut1]        # those mutate only little, hence cut1 determines size of elite so to speak, though it is no real elite if it mutates as well
        self.bunchB=self.F1[cut1:cut2]     # the part that just mutates
        self.bunchC=self.F1[cut2:cut3]     # one parent chosen with exponential selection pressure, then mutate
        self.bunchD=self.F1[cut3:cut4]     # two parents chosen with exponential selection pressure, then uniform CO  (plus eventual mutation)
        self.bunchE=self.F1[cut4:cut5]     # two parents chosen with exponential selection pressure, then BLX or WHX  (plus eventual mutation)
        self.bunchF=self.F1[cut5:]         # scaled difference vector betw. 2 random individuals added to some dude chosen with exp. sel. pressure  (plus eventual mutation)

    def create_offspring(self):
        # Todo: try: reverse mutation when it worsened the individual
        self.F1.advance_generation()
        N=self.F0.psize
        for i,fdude in enumerate(self.bunchA):
            fdude.copy_DNA_of(self.F0[fdude.no])
            fdude.mutate(0.5*self.Pm,self.eml*self.mstep*float(i)/float(len(self.bunchA)))  # mutate the best few just a little bit and the best one not at all
            fdude.ancestcode=0.09*i/len(self.bunchA)
        for i,fdude in enumerate(self.bunchB):
            fdude.copy_DNA_of(self.F0[fdude.no])
            fdude.mutate(self.Pm,self.mstep)          # just mutation
            fdude.ancestcode=0.103+0.099*i/len(self.bunchB)
        for i,fdude in enumerate(self.bunchC):
            parent=parentselect_exp(N,self.selpC)     # select a parent favoring fitness and mutate
            fdude.copy_DNA_of(self.F0[parent],copyscore=False,copyancestcode=False,copyparents=False)
            fdude.mutate(self.Pm,self.mstep)
            fdude.ancestcode=0.213+0.089*parent/N
        for i,fdude in enumerate(self.bunchD):                    # select two parents favoring fitness, mix by uniform CO, mutate
            parenta=parentselect_exp(N,self.selpD)    # select a parent favoring fitness and mutate
            parentb=parentselect_exp(N,self.selpD)    # select a parent favoring fitness and mutate
            cigar_choice=rand()<self.cigar2uniform
            if cigar_choice:
                fdude.cigar_CO(self.F0[parenta],self.F0[parentb],aspect=self.cigar_aspect,alpha=0.8,beta=0.3,scatter='randn')
                fdude.ancestcode=0.495;
            else:
                fdude.CO_from(self.F0[parenta],self.F0[parentb])
                fdude.ancestcode=0.395;
            if self.mr: fdude.mutate(self.Pm,self.mr*self.mstep)
        for i,fdude in enumerate(self.bunchE):                    # select two parents favoring fitness, mix by WHX or BLX, mutate
            parenta=parentselect_exp(N,self.selpE)    # select a parent favoring fitness and mutate
            parentb=parentselect_exp(N,self.selpE)    # select a parent favoring fitness and mutate
            whx_choice=rand()<self.WHX2BLX
            if whx_choice:
                fdude.WHX(self.F0[parenta],self.F0[parentb]); fdude.ancestcode=0.82
            else:
                fdude.BLX_alpha(self.F0[parenta],self.F0[parentb]); fdude.ancestcode=0.95
            if self.mr: fdude.mutate(self.Pm,self.mr*self.mstep)
        for i,fdude in enumerate(self.bunchF):
            parenta=parentselect_exp(N,self.selpF)
            seq=rand(N).argsort(); seq=list(seq); parenta_index=seq.index(parenta); seq.pop(parenta_index) # exclude already selected parent
            parentb,parentc=seq[:2]  # these two parents chosen without selection pressure; all 3 parents are different
            m=0.2+0.6*rand() # scaling of the difference vector to be added to parenta's DNA
            fdude.set_DNA( self.F0[parenta].DNA  +  m * (self.F0[parentc].DNA-self.F0[parentb].DNA) )
            fdude.mirror_and_scatter_DNA_into_bounds()
            fdude.ancestcode=0.603+0.099*parenta/N
            if self.mr: fdude.mutate(self.Pm,self.mr*self.mstep)

    def simple_run(self,generations,random_starting_DNAs=True):
        self.make_bunchlists()
        ComboBase.simple_run(self,generations,random_starting_DNAs)

    def simple_run_with_seed_DNAs(self,generations,seedDNAs):
        self.make_bunchlists()
        ComboBase.simple_run_with_seed_DNAs(self,generations,seedDNAs)

    def fractal_run(self,g1,g2,fractalfactor=4,msrf=0.3,return_inibest=False):
        mstep_memo=self.mstep; self.mstep*=msrf # msrf stands for mstep reduction factor
        anneal_memo=self.anneal; self.anneal=0.
        self.make_bunchlists()
        self.storageA=population_like(self.F0,size=self.F0.psize)
        self.storageB=population_like(self.F0,size=self.F0.psize)
        if return_inibest==True: inibest,inimean=self.evolve_protopopulations(fractalfactor,g1,elitesize=len(self.bunchA),return_inibest=True)
        else: self.evolve_protopopulations(fractalfactor,g1,elitesize=len(self.bunchA))
        self.mstep=mstep_memo
        self.anneal=anneal_memo
        self.run(g2)
        self.finish()
        if return_inibest: return inibest,inimean
        else: return

    def generate_protopopulation(self,N):
        q=self.F0.psize/N
        for n in range(N):
            if n!=0 or self.F0.gg!=0: self.F0.advance_generation()
            self.F0.new_random_genes()
            self.F0.zeroth_generation()
            self.record_stuff()
            for storedude,f0dude in zip(self.storageA[n*q:(n+1)*q],self.F0[:q]):  # store elite of random population
                storedude.copy_DNA_of(f0dude,copyscore=True,copyancestcode=False,copyparents=True)
                storedude.ancestcode=0.48+n*0.1; storedude.ancestcode-=floor(storedude.ancestcode) # ensures value is in [0,1]
        for f0dude,storedude in zip(self.F0,self.storageA):
            f0dude.copy_DNA_of(storedude,copyscore=True,copyancestcode=True,copyparents=True)
        self.F0.sort(); self.F0.update_no()
        self.F0.advance_generation()
        self.record_stuff()    # so the merged population where no evaluation is necessary shows up in plots



    def evolve_protopopulations(self,N,G,elitesize=4,selp=2.2,return_inibest=False):
        # in series evolve N populations over G generations each time starting with (skimmed) random DNAs
        # return F0 containing a "good" selection of DNAs from the N final states
        # where "good" means (a) containing the best of each and (b) some random choices for diversity
        q=self.F0.psize/N
        if return_inibest:
            inibestscores=zeros(N); inimeanscores=zeros(N)
        for n in range(N):
            self.generate_protopopulation(N)  # merged best parts of N random populations
            if return_inibest:
                sc=self.F0.get_scores()
                inibestscores[n]=sc[0]; inimeanscores[n]=mean(sc)
            self.run(G)
            for i,dude in enumerate(self.storageB[n*q:(n+1)*q]):
                if i < elitesize: dude.copy_DNA_of(self.F0[i],copyscore=True,copyancestcode=False,copyparents=True)
                else: dude.copy_DNA_of(self.F0[parentselect_exp(self.F0.psize,selp)],copyscore=True,copyancestcode=False,copyparents=True)
                dude.ancestcode=0.48+n*0.1; dude.ancestcode-=floor(dude.ancestcode) # ensures value is in [0,1]
        for f0dude,storedude in zip(self.F0,self.storageB):
            f0dude.copy_DNA_of(storedude,copyscore=True,copyancestcode=True,copyparents=True)
        self.F0.sort(); self.F0.update_no()
        self.F0.advance_generation()
        self.record_stuff()    # so the merged population where no evaluation is necessary shows up in plots
        if return_inibest:
            if self.F0.whatisfit=='maximize': inibest=np.max(inibestscores)
            else: inibest=np.min(inibestscores)
            inimean=mean(inimeanscores)
            return inibest,inimean
        else:
            return






