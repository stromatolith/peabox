#!python
"""
peabox - python evolutionary algorithm toolbox
by Markus Stokmaier, IKET, KIT, Karlsruhe, August 2012

In this file:
 - definition of the class Individual and its derivative MOIndividual for MO-optimisation
"""

import numpy as np
from numpy import sqrt, pi, arctan, mean, mod, ceil
from numpy import arange, array, asfarray, ones, zeros, where, argsort
from numpy.random import rand, randn, randint

#--- ToDo:
# BGA mutation operator
# dude.become_DE_child(pa,pb,pc)

#-----------------------------------------------------------------------------------------------------------------------------
#--- The class "Individual" for real-coded single-objective optimisation
#-----------------------------------------------------------------------------------------------------------------------------
class Individual:
    def __init__(self,objfunc,paramspace):
        self.objfunc=objfunc  # the objective function or fitness function
        self.pars=[]; lls=[]; uls=[]
        for elem in paramspace:
            self.pars.append(elem[0]); lls.append(elem[1]); uls.append(elem[2])
        self.lls=array(lls); self.uls=array(uls) # lower and upper limits, i.e. boundaries of the parameter domains
        self.ng=len(paramspace)                  # number of genes (npars stands for number of parameters)
        self.DNA=zeros(self.ng)                  # vector where the value of the genes will be stored in, i.e. the parameter values
        self.widths=self.uls-self.lls            # parameter band widths
        self.mstep=0.1                           # own mutation step size parameter, i.e. relation sd to domain width
        self.mutagenes=ones(self.ng)             # mutation step size multiplier for gene-dependent mutation strengths
        self.mutstepdistrib=None                 # you can later assign a callable function returning distributed mutation steps for use in self.distributed_jump()
        self.score=0                             # goal function to be minimized (or maximized) by evolutionary algorithm
        self.no=0                    # this dude's number within the population
        self.oldno=0                 # old self.no from last generation (helps for instructive coloring of population plots)
        self.ancestcode=0.           # number coding for descent, e.g. mutation step size, crossing-over... (helps for instructive coloring of population plots)
              # values 0...1:  0 to 0.99 -> no to weak mutation,   0.1 to 0.19 -> regular mutation,    0.2 to 0.29 -> mutation with exponential parent choice,   0.35 means recombination,  0.4 means random DNA, rest is still open
        self.pa=-1                   # parent a  (if pa==-1 it means random DNA)
        self.pb=-1                   # parent b  (if pb==-1 it means there was no CO involved in the creation of this individual)
        self.whatisfit='minimize'    # allowed are 'minimize' and 'maximize' telling you what to do with the score function during optimization
        self.gg=0
        self.ncase=0
        self.subcase=0
    def evaluate(self):
        self.score=self.objfunc(array(self.DNA,copy=1))
        return self.score
    def __str__(self):
        return 'dude'+str(self.no)
    def __eq__(self,otherindividual):
        if isinstance(otherindividual,Individual):
            return self.score == otherindividual.score
        else:
            return self.score == otherindividual     # makes comparison operator tolerant for direct use with a scalar number
    def __lt__(self,otherindividual):
        if isinstance(otherindividual,Individual):
            return self.score < otherindividual.score
        else:
            return self.score < otherindividual     # makes comparison operator tolerant for direct use with a scalar number
    def __gt__(self,otherindividual):
        if isinstance(otherindividual,Individual):
            return self.score > otherindividual.score
        else:
            return self.score > otherindividual     # makes comparison operator tolerant for direct use with a scalar number
    def __le__(self,otherindividual):
        if isinstance(otherindividual,Individual):
            return self.score <= otherindividual.score
        else:
            return self.score <= otherindividual     # makes comparison operator tolerant for direct use with a scalar number
    def __ge__(self,otherindividual):
        if isinstance(otherindividual,Individual):
            return self.score >= otherindividual.score
        else:
            return self.score >= otherindividual     # makes comparison operator tolerant for direct use with a scalar number
    def __ne__(self,otherindividual):
        if isinstance(otherindividual,Individual):
            return self.score != otherindividual.score
        else:
            return self.score != otherindividual     # makes comparison operator tolerant for direct use with a scalar number
    def isbetter(self,otherdude):
        if self.whatisfit=='minimize':
            return self < otherdude
        else:
            return self > otherdude
    def isworse(self,otherdude):
        if self.whatisfit=='minimize':
            return self > otherdude
        else:
            return self < otherdude
    def isequallyfit(self,otherdude):
        return self == otherdude
    def get_DNA(self):
        return self.DNA
    def get_copy_of_DNA(self):
        return array(self.DNA,copy=1)        
    def set_DNA(self,params):
        self.DNA[:]=asfarray(params)
    def get_uDNA(self):
        # DNA in a transformed coordinate system corresponding to a unity cube (all parameters range from 0 to 1)
        return (self.DNA-self.lls)/self.widths
    def set_uDNA(self,uparams):
        # set DNA if values are given in terms of an n-dimensional 0-to-1 cube ("u" for unity cube)
        self.DNA[:]=self.widths*uparams+self.lls
    def center_DNA(self):
        # transfer this individual to the center of the search space
        self.set_DNA(0.5*(self.lls+self.uls))
    def print_stuff(self,slim=True):
        if slim:
            print self,';  oldno: ',self.oldno,';  pa: ',self.pa,';  pb: ',self.pb,';  score: ',self.score
        else:
            print self,';  oldno: ',self.oldno,';  pa: ',self.pa,';  pb: ',self.pb,';  score: ',self.score,';  DNA: ',self.DNA
    def set_mutagenes(self,params):
        self.mutagenes[:]=params
    def reset_mutagenes(self,value=1.):
        self.mutagenes[:]=value
    def copy_DNA_of(self,otherindividual,copyscore=False,copyancestcode=False,copyparents=False,copymutagenes=False):
        self.DNA[:]=otherindividual.DNA
        if copymutagenes:
            self.mutagenes[:]=otherindividual.mutagenes
        self.pa=otherindividual.no; self.pb=-1
        if copyscore:
            self.score=otherindividual.score
        if copyancestcode:
            self.ancestcode=otherindividual.ancestcode
        if copyparents:
            self.pa=otherindividual.pa
            self.pb=otherindividual.pb
    def random_DNA(self):
        self.set_DNA(self.lls+rand(self.ng)*self.widths)
        self.pa=-1; self.pb=-1; self.ancestcode=1.

    def set_bad_score(self):
        if self.whatisfit=='minimize':
            self.score=1e32
        else:
            self.score=-1e32

    #----- mutation operators ------------------------------------------------------------------------------------------
        
    def insert_random_genes(self,P):
        newDNA=self.lls+self.widths*rand(self.ng)
        mutflag=where(rand(self.ng)<P,1,0)
        self.set_DNA(where(mutflag==1,newDNA,self.DNA))
    
    def mutate(self,P,sd=None,uCS=True,maxiter=5):
        # each gene mutates with probability P by addition of a normally distributed random number with standard deviation sd
        # the higher maxiter the better fidelity to the distribution shape (noticeable for large sdfrac) but execution time may increase
        # why do I like this mutation operator most? because domain boundaries are taken care of without the normal distribution being affected much
        # (remember: the normal distribution has far-reaching tails,
        # frequently enough creating points several standard deviations away from the center)
        if sd is None: sd=self.mstep
        if uCS:
            DNA=self.get_uDNA(); lls=zeros(self.ng); uls=ones(self.ng); w=ones(self.ng)
        else:
            DNA=self.get_copy_of_DNA(); lls=self.lls; uls=self.uls; w=self.widths
        for i in range(self.ng):
            if rand(1) < P:
                for j in range(maxiter):
                    newval=DNA[i]+randn(1)*sd*w[i]
                    if lls[i]<=newval<=uls[i]: break
                if not lls[i]<=newval<=uls[i]: newval=lls[i]+rand(1)*w[i]
                DNA[i]=newval
        if uCS: self.set_uDNA(DNA)
        else: self.set_DNA(DNA)
    def mutate_quickly(self,P,sd=None,uCS=True,mirrorbds=True):
        # you can use this instead of mutate() when sd is smaller than 0.1 and the population cloud not hovering near a wall
        # this routine has no loops, at least not on the level of python, but at the level of the inner workings of numpy there
        # might be plenty more loop work hidden than what needs to be worked off by mutate(), so one still has to check which one is really quicker
        # speed checking on my fujitsu siemens lifebook E8110: this routine is 1.5 times as fast as mutate() in 3D and 5 times as fast in 20D
        if sd is None: sd=self.mstep
        if uCS: self.DNA+=self.widths*sd*randn(self.ng) * where(rand(self.ng)<P,1,0)
        else: self.DNA+=sd*randn(self.ng) * where(rand(self.ng)<P,1,0)
        if mirrorbds: self.mirror_DNA_into_bounds()
    def mutate_fixstep(self,stepsize=None,uCS=True,mirrorbds=True):
        # mutation as jump into a random direction with fixed step size
        if stepsize is None: stepsize=self.mstep
        step=randn(self.ng); step=stepsize*step/sqrt(np.sum(step*step))
        if uCS:
            DNA=self.get_uDNA(); DNA+=step; self.set_uDNA(DNA)
        else:
            self.DNA+=step
        if mirrorbds: self.mirror_DNA_into_bounds()
    def mutate_with_steprange(self,minstep=0,maxstep=None,uCS=True,mirrorbds=True):
        # mutation as jump into a random direction with uniformly distributed step size between minstep and maxstep
        if maxstep is None: maxstep=self.mstep
        stepsize=minstep+rand()*(maxstep-minstep)
        self.mutate_fixstep(stepsize=stepsize,uCS=uCS,mirrorbds=mirrorbds)
    def varimutate(self,P,mirrorbds=True):
        # mutation step size self.mstep multiplied with gene-dependent multipliers in self.mutagenes
        mutflag=where(rand(self.ng)<P,1,0)
        self.DNA+=self.mutagenes*self.mstep*self.widths*mutflag*randn(self.ng)
        if mirrorbds: self.mirror_DNA_into_bounds()
    def distributed_jump(self,distrib=None,uCS=True,mirrorbds=True):
        if distrib is None: distrib=self.mutstepdistrib
        if uCS:
            DNA=self.get_uDNA(); DNA+=distrib(); self.set_uDNA(DNA)
        else:
            self.DNA+=distrib()
        if mirrorbds: self.mirror_DNA_into_bounds()
    #def BGA_mutation(self):
    #    still has to be implemented
        

    #----- enforcing domain boundaries when mutations kick individuals outside the designated search space ------------------------
    # Something needs to be said about how we should treat DNA vectors that have jumped (e.g. by mutation) outside the search domain.
    # The simple solution often seen in EA literature is to push the point back into the intended domain, i.e. transfer it to the
    # that point on the domain boundaries which is closest to its current illegal position. But if this happens more often, as a consequence
    # the domain boundaries get examined much more thouroughly than the domain volume itself. One can avoid that search density mismatch by
    # letting the domain walls act as mirrors instead. Only in the case of periodic search spaces it would of course make much more sense to
    # let an angle mutate from 352 to 4 degrees instead of mirroring it back to 356 degrees; for this purpose the method cycle_DNA_into_bounds()
    # has been implemented below. For small transgressions of the boundaries mirroring and cycling back should be sufficient. For very far
    # transgressions of the domain boundaries, i.e. points having landed several domain widths far outside, one might argue that (depending on
    # the history that has brought the vector to this point) without loss of information the concerned coordinate can as well be reinitialised
    # by a uniform random distribution spanning the intended domain; the methods mirror_and_scatter_DNA_into_bounds() and
    # cycle_and_scatter_DNA_into_bounds() are designed to serve that purpose by enacting the close/far transgression distinction.
        
    def push_DNA_into_bounds(self):
        self.DNA=where(self.DNA<self.lls,self.lls,self.DNA)
        self.DNA=where(self.DNA>self.uls,self.uls,self.DNA)
    def mirror_DNA_into_bounds(self):
        while np.any(self.DNA-self.lls<0) or np.any(self.DNA-self.uls>0):
            self.DNA+=2*where(self.DNA<self.lls,self.lls-self.DNA,0)
            self.DNA-=2*where(self.DNA>self.uls,self.DNA-self.uls,0)
    def cycle_DNA_into_bounds(self):
        # may be faster than mirror_DNA_into_bounds when mutation to area far outside the domain often occurs
        if np.any(self.DNA-self.lls<0):
            dist=ceil(where(self.DNA-self.lls<0,self.lls-self.DNA,0)/self.widths)
            self.DNA+=dist*self.widths
        if np.any(self.DNA-self.uls>0):
            dist=ceil(where(self.DNA-self.uls>0,self.DNA-self.uls,0)/self.widths)
            self.DNA-=dist*self.widths
    def mirror_and_scatter_DNA_into_bounds(self):
        # case distinction between close transgression (flag closetg) and far transgression (flag fartg)
        tg=where(self.DNA<self.lls,1,0); fartg=where(self.DNA<self.lls-0.5*self.widths,1,0)
        closetg=tg-fartg; self.DNA+=2*where(closetg,self.lls-self.DNA,0)
        tg=where(self.DNA>self.uls,1,0); fartg=where(self.DNA>self.uls+0.5*self.widths,1,0)
        closetg=tg-fartg; self.DNA-=2*where(closetg,self.DNA-self.uls,0)
        # still outside? then this coordinate gets reinitialised by random
        self.randomize_DNA_into_bounds()
    def cycle_and_scatter_DNA_into_bounds(self):
        # case distinction between close transgression (flag closetg) and far transgression (flag fartg)
        fardowntg=where(self.DNA<self.lls-0.5*self.widths,1,0)
        faruptg=where(self.DNA>self.uls+0.5*self.widths,1,0)
        rDNA=self.lls+self.widths*rand(self.ng)
        self.DNA=where(faruptg+fardowntg,rDNA,self.DNA) # reinitialised by random if far upwards or far downwards
        self.cycle_DNA_into_bounds()
    def randomize_DNA_into_bounds(self):
        rDNA=self.lls+self.widths*rand(self.ng)
        self.DNA=where(self.DNA<self.lls,rDNA,self.DNA)
        self.DNA=where(self.DNA>self.uls,rDNA,self.DNA)

    #----- some basic gene mixing mechanisms ------------------------------------------------------------------------------------
            
    def CO_with(self,otherindividual,P=0.5):
        # swap genes with partner with probability P
        DNA1=array(self.DNA[:],copy=1); DNA2=array(otherindividual.DNA[:],copy=1)
        flag=where(rand(self.ng)<P,1,0)
        self.DNA[:]=where(flag,DNA2,DNA1); otherindividual.DNA[:]=where(flag,DNA1,DNA2)
        self.pa=self.no; self.pb=otherindividual.no; self.ancestcode=0.41
    def CO_from(self,dude1,dude2,P1=0.5):
        # compose DNA through uniform crossing-over of two source folks
        # P1: probability that a gene comes from dude1
        dude1flag=where(rand(dude1.ng)<P1,1,0); dude2flag=ones(dude1.ng)-dude1flag
        self.DNA[:]=dude1flag*dude1.DNA+dude2flag*dude2.DNA
        self.pa=dude1.no; self.pb=dude2.no; self.ancestcode=0.45
    def become_mixture_of(self,dude1,dude2,fade=0.5):
        # fade: fading percentage weight between source dudes; fade=0 means 100% dude1, fade=1 means 100% dude2
        self.DNA[:]=(1.-fade)*dude1.DNA+fade*dude2.DNA
        self.pa=dude1.no; self.pb=dude2.no; self.ancestcode=0.49
    def become_mixture_of_multiple(self,dudelist):
        self.DNA[:]=mean(array([dude.DNA for dude in dudelist]),axis=0)
        self.pa=1; self.pb=1; self.ancestcode=0.82

    #----- now follow some crossing-over operators popular in GA tradition, feel free to add more ----------------------------------------------

    def onepoint_CO_with(self,otherdude):
        # random determines whether DNA in front of or behind cutting point will be swapped
        cut=randint(self.ng-1)+1
        storage=self.get_copy_of_DNA()
        if rand()<0.5:
            self.DNA[:cut]=otherdude.DNA[:cut]; otherdude.DNA[:cut]=storage[:cut]
        else:
            self.DNA[cut:]=otherdude.DNA[cut:]; otherdude.DNA[cut:]=storage[cut:]
    def twopoint_CO_with(self,otherdude):
        # random determines whether DNA inside or outside the two points will be swapped
        cut1=randint(self.ng-1)+1; cut2=randint(self.ng-2)+1
        if cut2 >= cut1: cut2+=1
        if cut1>cut2:
            cut1,cut2=cut2,cut1
        storage=self.get_copy_of_DNA()
        if rand()<0.5:
            self.DNA[cut1:cut2]=otherdude.DNA[cut1:cut2]; otherdude.DNA[cut1:cut2]=storage[cut1:cut2]
        else:
            self.DNA[:cut1]=otherdude.DNA[:cut1]; otherdude.DNA[:cut1]=storage[:cut1]
            self.DNA[cut2:]=otherdude.DNA[cut2:]; otherdude.DNA[cut2:]=storage[cut2:]
    def npoint_CO_with(self,otherdude,n):
        if n>=self.ng: raise ValueError('You are trying to cut the DNA into more pieces than there are genes.')
        a=arange(self.ng-1)+1; r=rand(self.ng-1); a=[a[i] for i in argsort(r)]; cuts=[0]+sorted(a[:n])+[self.ng]
        storage=self.get_copy_of_DNA()
        if rand()<0.5:
            self.DNA[:]=otherdude.DNA; otherdude.DNA[:]=storage # swap DNAs
        storage=self.get_copy_of_DNA()
        for i in range(0,n+1,2):
            self.DNA[cuts[i]:cuts[i+1]]=otherdude.DNA[cuts[i]:cuts[i+1]]; otherdude.DNA[cuts[i]:cuts[i+1]]=storage[cuts[i]:cuts[i+1]]
    def onepoint_CO_from(self,dude1,dude2):
        # random determines whether DNA copying begins with dude1's or dude2's DNA
        cut=randint(self.ng-1)+1
        if rand()<0.5:
            self.DNA[:cut]=dude1.DNA[:cut]; self.DNA[cut:]=dude2.DNA[cut:]
        else:
            self.DNA[cut:]=dude1.DNA[cut:]; self.DNA[:cut]=dude2.DNA[:cut]
    def twopoint_CO_from(self,dude1,dude2):
        # random determines whether DNA inside or outside the two points will be swapped
        cut1=randint(self.ng-1)+1; cut2=randint(self.ng-2)+1
        if cut2 >= cut1: cut2+=1
        if cut1>cut2:
            cut1,cut2=cut2,cut1
        if rand()<0.5:
            self.DNA[:cut1]=dude1.DNA[:cut1]; self.DNA[cut1:cut2]=dude2.DNA[cut1:cut2]; self.DNA[cut2:]=dude1.DNA[cut2:]
        else:
            self.DNA[:cut1]=dude2.DNA[:cut1]; self.DNA[cut1:cut2]=dude1.DNA[cut1:cut2]; self.DNA[cut2:]=dude2.DNA[cut2:]
    def npoint_CO_from(self,dude1,dude2,n):
        if n>=self.ng: raise ValueError('You are trying to cut the DNA into more pieces than there are genes.')
        a=arange(self.ng-1)+1; r=rand(self.ng-1); a=[a[i] for i in argsort(r)]; cuts=[0]+sorted(a[:n])+[self.ng]
        if rand()<0.5:
            DNA1=array(dude1.DNA[:],copy=1); DNA2=array(dude2.DNA[:],copy=1)
        else:
            DNA1=array(dude2.DNA[:],copy=1); DNA2=array(dude1.DNA[:],copy=1)
        for i in range(n+1):
            if not mod(i,2):
                self.DNA[cuts[i]:cuts[i+1]]=DNA1[cuts[i]:cuts[i+1]]
            else:
                self.DNA[cuts[i]:cuts[i+1]]=DNA2[cuts[i]:cuts[i+1]]
    def BLX_alpha(self,dude1,dude2,alpha=0.5):
        diff=dude1.DNA-dude2.DNA
        extrapol1=dude1.DNA+alpha*diff; extrapol2=dude2.DNA-alpha*diff
        randvec=rand(self.ng)
        self.DNA[:]=randvec*extrapol1+(1-randvec)*extrapol2
        self.mirror_DNA_into_bounds()
        self.pa=dude1.no; self.pb=dude2.no; self.ancestcode=0.61
    def BLX_alpha_beta(self,dude1,dude2,alpha=0.5,beta=0.2):
        # make sure dude1 is the better one
        diff=dude1.DNA-dude2.DNA
        extrapol1=dude1.DNA+alpha*diff; extrapol2=dude2.DNA-beta*diff
        randvec=rand(self.ng)
        self.DNA[:]=randvec*extrapol1+(1-randvec)*extrapol2
        self.mirror_DNA_into_bounds()
        self.pa=dude1.no; self.pb=dude2.no; self.ancestcode=0.65
    def WHX(self,dude1,dude2,alpha=1):
        # Wright's heuristic CO
        # make sure dude1 is the better one
        diff=dude1.DNA-dude2.DNA
        randvec=rand(self.ng)
        self.DNA[:]=dude1.DNA+randvec*alpha*diff
        self.mirror_DNA_into_bounds()
        self.pa=dude1.no; self.pb=dude2.no; self.ancestcode=0.69

    #----- my CO operator trials ---------------------------------------------------------------------------------------
    # the problem with CO operators like uniform CO or BLX is the coordinate system dependency of the distribution of search points
    # uniform CO --> search points created at the edges of a rectangular structure aligned with the CS
    # BLX and similar --> search points created inside a rectangular structure aligned with the CS
    # popular solutions:
    # e.g. fuzzy-CO (distribution like one electron around two protons) -> you might as well just choose
    #      one of the parents and apply mutation - combination info is lost
    # e.g. pure linear combination of parents, no random like in scatter search -> makes only sense if algorithm is designed for it
    #      (scatter search keeps track and avoids potential repetitions, very low number of parents and recombinations, but safety-net of random is missing)
    # e.g. multivariate normal distributions -> best solution, but: wasting search points around the center-of-mass and thin ends of the eliptical cloud
    # my solution approach: trials lie in a cylindrical volume around the connecting line or deviations of a cylinder
    # cigar shape: very easy to implement
    # devil_stick, i.e. two tall cones pointing at each other or the cigar shape with lower density in the middle -> those two could perhaps make
    # full use of the information lying in the difference vector which is probably the thing making differential evolution so successful, but
    # that's not that easy to implement and maybe I won't have the time

    def cigar_CO(self,dude1,dude2,aspect=6.,alpha=0.6,beta=0.3,scatter='randn',uCS=True,mirrorbds=True):
        # dude1 is assumed to be the better one
        w=1+alpha+beta; x=w*rand()-beta
        self.DNA[:]=x*dude2.DNA+(1-x)*dude1.DNA
        d=dude1.distance_from(dude2,uCS=uCS); d*=w; maxstep=d/aspect
        if scatter=='randn':
            self.mutate(1,maxstep/2.,uCS=uCS)
        elif scatter=='rand':
            self.mutate_with_steprange(maxstep=maxstep,uCS=uCS,mirrorbds=mirrorbds)

    def thinned_cigar_CO(self,dude1,dude2,aspect=6.,alpha=0.6,beta=0.3,scatter='randn',thinning=1.,uCS=True,mirrorbds=True):
        # dude1 is assumed to be the better one
        thisrand=0.5*(1+arctan(thinning*pi*(rand()-0.5))/arctan(thinning*pi*0.5))
        w=1+alpha+beta; x=w*thisrand-beta
        self.DNA[:]=x*dude2.DNA+(1-x)*dude1.DNA
        d=dude1.distance_from(dude2,uCS=uCS); d*=w; maxstep=d/aspect
        if scatter=='randn':
            self.mutate(1,maxstep/2.,uCS=uCS)
        elif scatter=='rand':
            self.mutate_with_steprange(maxstep=maxstep,uCS=uCS,mirrorbds=mirrorbds)
        
    def cyl_CO(self,dude1,dude2,aspect=6.,alpha=0.5,beta=0.2,uCS=True):
        raise NotImplementedError('still to be coded: a CO operator uniformly covering a thin cylindrical volume along the line connecting two individuals')

    def broom_CO(self,dude1,dude2,alpha=0.5,beta=0.2,uCS=True):
        raise NotImplementedError('still to be coded: a CO operator uniformly covering a thin almost cylindrical (except widening ends) volume along the line connecting two individuals')


    #----- the following operators determining inter-idividual distances are essential for Scatter Search but may also be useful in other contexts --------------
        
    def distance_from(self,dude,uCS=False):
        if uCS:
            DNA1=self.get_uDNA(); DNA2=dude.get_uDNA()
            return sqrt(np.sum((DNA1-DNA2)**2))
        else:
            return sqrt(np.sum((self.DNA-dude.DNA)**2))
    def distance_from_point(self,x,uCS=False):
        if uCS:
            DNA1=self.get_uDNA(); DNA2=(x-self.lls)/self.widths
            return sqrt(np.sum((DNA1-DNA2)**2))
        else:
            return sqrt(np.sum((self.DNA-x)**2))
    def avg_distance_to_set(self,someset,uCS=False):
        """average distance between the dude and the members of someset"""
        r=0.
        for dude in someset:
            r+=self.distance_from(dude,uCS=uCS)
        r/=len(someset)
        return r
    def min_distance_to_set(self,someset,uCS=False):
        """distance between the dude and the member of someset which is closest to dude"""
        dmin=self.distance_from(someset[0])
        for dude in someset[1:]:
            d=self.distance_from(dude,uCS=uCS)
            if d<dmin: dmin=d
        dude.dist=dmin
        return dmin




#-----------------------------------------------------------------------------------------------------------------------------
#--- for a first concept of multi-objective optimisation: class "MOIndividual"
#-----------------------------------------------------------------------------------------------------------------------------

class MOIndividual(Individual):
    
    def __init__(self,objectives,paramspace):
        Individual.__init__(self,None,paramspace)
        del self.objfunc
        objfuncs=[]; objnames=[]
        for name,func in objectives:
            objnames.append(name)
            objfuncs.append(func)
        self.objnames=objnames                     # the list of objective function names
        self.objfuncs=objfuncs                     # a list of objective functions or fitness functions
        self.nobj=len(objfuncs)                    # the number of objectives
        self.objvals=zeros(self.nobj)              # this vector contains the separate values returned from the several objective functions
        self.ranks=-ones(self.nobj,dtype=int)      # this list contains the rankings according to the different objectives
        self.overall_rank=-1.                      # weighted combination of this dude's ranks at the different disciplines
        self.rankweights=ones(self.nobj,dtype=float)# the coefficients when summing over the different ranks in order to calculate self.overall_rank
        self.objdirecs=self.nobj * ['min']         # list containing optimisation direction for each objective
        self.whatisfit='minimize'                  # the ordering direction for the overall scalar fitness (if used)
        self.minmaxflip=ones(self.nobj)            # contains a 1 for objectives to be minimised, and a -1 for objectives to be maximised
        self.paretoefficient=False                 # whether this is a member of the Pareto front
        self.paretooptimal=False                   # whether this is a member of the Pareto front and has the best score
        self.paretoking=False                      # whether this dude strictly dominates everybody else
        self.sumcoeffs=ones(self.nobj,dtype=float) # coefficients in case overall score can be calculated as a weighted sum
        self.offset=0.                             # additional constant offset in case score can be calculated as a weighted sum

    def evaluate(self):
        for i,func in enumerate(self.objfuncs):
            self.objvals[i]=func(self.get_copy_of_DNA())
        self.update_score()
        return self.score

    def update_score(self):
        self.score=np.sum(self.sumcoeffs*self.objvals)+self.offset

    def update_overall_rank(self):
        self.overall_rank=np.sum(self.rankweights*self.ranks)

#    def __eq__(self,otherindividual):
#        return self.orderscore == otherindividual.orderscore
#    def __lt__(self,otherindividual):
#        return self.orderscore < otherindividual.orderscore
#    def __gt__(self,otherindividual):
#        return self.orderscore > otherindividual.orderscore
#    def __le__(self,otherindividual):
#        return self.orderscore <= otherindividual.orderscore
#    def __ge__(self,otherindividual):
#        return self.orderscore >= otherindividual.orderscore
#    def __ne__(self,otherindividual):
#        return self.orderscore != otherindividual.orderscore

    def strictly_dominates(self,otherdude):
        bettercrit=0; betterequalcrit=0
        for goal in self.objnames:
            if self.betterobj(otherdude,goal): bettercrit+=1
            if self.beqobj(otherdude,goal): betterequalcrit+=1
        if betterequalcrit==self.nobj and bettercrit >= 1 : result=True
        else: result = False
        return result
        
#    def isbetter(self,otherdude):
#        if self.whatisfit=='minimize':
#            return self < otherdude
#        else:
#            return self > otherdude
#    def isworse(self,otherdude):
#        if self.whatisfit=='minimize':
#            return self > otherdude
#        else:
#            return self < otherdude
#    def isequallyfit(self,otherdude):
#        return self == otherdude
    
    def betterobj(self,otherdude,goal):
        if type(goal)==str:
            goal=self.objnames.index(goal)
        if self.objdirecs[goal]=='min':
            return self.objvals[goal] < otherdude.objvals[goal]
        else:
            return self.objvals[goal] > otherdude.objvals[goal]

    def beqobj(self,otherdude,goal):
        if type(goal)==str:
            goal=self.objnames.index(goal)
        if self.objdirecs[goal]=='min':
            return self.objvals[goal] <= otherdude.objvals[goal]
        else:
            return self.objvals[goal] >= otherdude.objvals[goal]

    def worseobj(self,otherdude,goal):
        if type(goal)==str:
            goal=self.objnames.index(goal)
        if self.objdirecs[goal]=='min':
            return self.objvals[goal] > otherdude.objvals[goal]
        else:
            return self.objvals[goal] < otherdude.objvals[goal]

    def weqobj(self,otherdude,goal):
        if type(goal)==str:
            goal=self.objnames.index(goal)
        if self.objdirecs[goal]=='min':
            return self.objvals[goal] >= otherdude.objvals[goal]
        else:
            return self.objvals[goal] <= otherdude.objvals[goal]

    def equalobj(self,otherdude,goal):
        if type(goal)==str:
            goal=self.objnames.index(goal)
        return self.objvals[goal] == otherdude.objvals[goal]
        
    def calculate_overall_rank(self):
        """calculate the overall ranking of this dude based on a weighted sum over the rankings in the different disciplines"""
        self.overall_rank=np.sum(self.rankweights*self.ranks)/float(self.nobj)
            
    def print_stuff(self,slim=True):
        if slim:
            print self,';  oldno: ',self.oldno,';  pa: ',self.pa,';  pb: ',self.pb,';  score: ',self.score
        else:
            print self,';  oldno: ',self.oldno,';  pa: ',self.pa,';  pb: ',self.pb,';  score: ',self.score,';  DNA: ',self.DNA

    def copy_DNA_of(self,otherindividual,copyscore=False,copyancestcode=False,copyparents=False):
        self.DNA[:]=otherindividual.get_copy_of_DNA()
        self.mutagenes[:]=array(otherindividual.mutagenes,copy=1)
        self.pa=otherindividual.no; self.pb=-1
        if copyscore:
            self.objvals[:]=array(otherindividual.objvals,copy=1)
            self.score=otherindividual.score
        if copyancestcode:
            self.ancestcode=otherindividual.ancestcode
        if copyparents:
            self.pa=otherindividual.pa
            self.pb=otherindividual.pb



#-----------------------------------------------------------------------------------------------------------------------------
#--- to use the CEC-2013 test function suite via ctypes, I implemented the following subclass
#-----------------------------------------------------------------------------------------------------------------------------

class cecIndivid(Individual):
    def __init__(self,ndim):
        self.objfunc=None
        self.pars=['p'+str(i).zfill(2) for i in range(ndim)]
        self.lls=-100*ones(ndim,dtype=float)     # lower limits for DNA vector entries
        self.uls=+100*ones(ndim,dtype=float)     # upper limits for DNA vector entries
        self.ng=ndim                             # number of genes (npars stands for number of parameters)
        self.DNA=zeros(self.ng,dtype=float)      # vector where the value of the genes will be stored in, i.e. the parameter values
        self.widths=self.uls-self.lls            # parameter band widths
        self.mstep=0.1                           # own mutation step size parameter, i.e. relation sd to domain width
        self.mutagenes=ones(self.ng)             # mutation step size multiplier for gene-dependent mutation strengths
        self.mutstepdistrib=None                 # you can later assign a callable function returning distributed mutation steps for use in self.distributed_jump()
        self.score=0                             # goal function to be minimized (or maximized) by evolutionary algorithm
        self.no=0                    # this dude's number within the population
        self.oldno=0                 # old self.no from last generation (helps for instructive coloring of population plots)
        self.ancestcode=0.           # number coding for descent, e.g. mutation step size, crossing-over... (helps for instructive coloring of population plots)
              # values 0...1:  0 to 0.99 -> no to weak mutation,   0.1 to 0.19 -> regular mutation,    0.2 to 0.29 -> mutation with exponential parent choice,   0.35 means recombination,  0.4 means random DNA, rest is still open
        self.pa=-1                   # parent a  (if pa==-1 it means random DNA)
        self.pb=-1                   # parent b  (if pb==-1 it means there was no CO involved in the creation of this individual)
        self.whatisfit='minimize'    # allowed are 'minimize' and 'maximize' telling you what to do with the score function during optimization
        self.gg=0
        self.ncase=0
        self.subcase=0
    def evaluate(self):
        raise TypeError('this is a cecIndivid instance - fitness evaluation must take place on the population level')

    def set_bad_score(self):
        self.score=1e32



