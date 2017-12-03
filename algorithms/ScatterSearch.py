#!python
"""
Here I tried my best at implementing the basics of the Scatter Search Algorithm by Fred Glover, Manuel Laguna, and Rafael Marti
The scheme is based on these two publications:

a) Fred Glover, Manuel Laguna, Rafael Marti: "Scatter Search," Advances in Evolutionary Computing: Theory and Applications,
   A. Ghosh and S.Tsutsui (eds.), Springer-Verlag, New York, 2003, pp. 519-537

b) Fred Glover: "A Template for Scatter Search and Path Relinking", Artificial Evolution, Lecture Notes in Computer Science,
   1363, J.-K. Hao, E. Lutton, E. Ronald , M. Schoenauer and D. Snyers, Eds. Springer, 1998, pp. 13-54.

then there is an additional recombination option based on this publication:

c) F. Herrera, M. Lozano, D. Molina: "Continuous scatter search: An analysis of the integration of some combination methods and
   improvement strategies", European Journal of Operational Research 169 (2) (2006) 450-476

the code below adds the properties self.new, self.dist, self.elite where needed on the fly to instances of the
individual class, sorry for that messiness, I should have created an own Individual subclass.


"""
import numpy as np
from numpy import mod, shape, argmin, argmax
from numpy import arange, zeros, where
from numpy.random import rand
from time import clock

from peabox_population import population_like

class ScatterSearch:

    def __init__(self,divset,refset,nref=20,b1=10):
        self.divset=divset                               # the divset
        self.refset=refset                               # the refset (keep in mind it may have the recorder attached)
        self.rt1=population_like(refset,size=b1)         # the refset's first tier
        self.rt2=population_like(refset,size=nref-b1)    # the refset's second tier
        #self.refset.folks=self.rt1.folks+self.rt2.folks  # the refset (why not self refset=self.rt1+self.rt2? --> because of the recorder eventually attached to the refset)
        #self.refset.psize=len(self.rt1.folks)+len(self.rt2.folks)
        self.refset.folks=self.rt1+self.rt2
        self.refset_update_style='Glover'                # 'Glover' or 'Herrera' , i.e. entry into RefSet via distance criterion active or inactive, respectively
        self.pool=population_like(divset,size=0)             # the pool for newly recombined solution candidates
        self.localpop=population_like(divset,size=0)         # a population for the local search
        self.ndiv=len(divset)
        self.nref=nref
        self.b1=b1                             # b2=nref-b1   Scatter Search terminology: refset consists of two tiers of sizes B1 and B2
        self.recomb='arit'                     # options are 'arit', 'BLXa', 'Glover', and 'Brownlee'   (all recombination methods for subsets of size 2 so far)
        self.ls_depth=100                            # localdepth objective function calls allowed for local search
        self.ls_mstep=0.05                        # mutation step size for local search where needed
        self.ls_adapfactor=1.8
        self.ls_mu=3
        self.ls_lambda=8
        self.localsearch='greedy'              # options (to be implemented) are 'greedy', 'NelderMead', 'SA', 'grad', 'ES', ...
        self.local_ES_initialized=False
        self.local_greedy_initialized=False
        self.local_NM_initialized=False
        self.generation_callbacks=[]  # put functions into this list to be called after each generation with this EA instance as argument, e.g. plot current best solution

    def adjust_segment_probabilities(self,hist):
        N,nd=shape(hist)
        p=zeros((N,nd))
        for i in range(nd):
            if 0 in hist[:,i]:
                p[:,i]=where(hist[:,i]==0,1,0)
            else:
                p[:,i]=1./hist[:,i]
            p[:,i]*=1./np.sum(p[:,i])
        return p

    def random_DNA_in_segments(self,dude,p):
        N,ng=shape(p) # probability matrix for DNA to be invented
        assert ng==dude.ng
        r=rand(2,ng) # two random numbers per gene: 1st for segment choice, 2nd for random number to be projected on segment
        sbounds=zeros((N+1,ng)) # segment boundaries mapped to the interval [0,1]
        skey=zeros(ng,dtype=int)  # segment key, i.e. in which segment should this DNA entry be cre
        for i in range(1,N+1):
            sbounds[i,:]=np.sum(p[:i,:],axis=0)
        for j in range(ng):
            for i in range(N):
                if sbounds[i,j]<r[0,j]<sbounds[i+1,j]:
                    skey[j]=i
        qb=arange(N+1,dtype=float)/float(N)  # quarterboundaries
        for j in range(ng):
            dude.DNA[j]=dude.lls[j] + (qb[skey[j]]+(qb[skey[j]+1]-qb[skey[j]])*r[1,j])*dude.widths[j]

    def renew_divset(self,size=None):
        if size is not None: self.ndiv=size
        if self.divset.psize != self.ndiv: self.divset.change_size(self.ndiv)
        h=zeros((4,self.divset.ng),dtype=int)  # histogram data tracking quarter occurences
        for dude in self.divset:
            p=self.adjust_segment_probabilities(h)  # probabilities for quarter choices
            self.random_DNA_in_segments(dude,p)
            q=4*(dude.DNA-dude.lls)/dude.widths
            for i in range(dude.ng):
                h[int(q[i]),i]+=1
        self.divset.eval_all()
        return

    def prepare_localpop(self):
        if self.localsearch=='greedy':
            if self.localpop.psize != 1:
                self.localpop.change_size(1)
        elif self.localsearch=='ES':
            if self.localpop.psize != self.ls_lambda:
                self.localpop.change_size(self.ls_lambda)
        elif self.localsearch=='NelderMead':
            raise NotImplementedError("The downhill simplex algorithm of Nelder and Mead is not yet implemented")
        else:
            raise NotImplementedError("self.localsearch is incorrect, the options are 'greedy', 'ES', and 'NelderMead' so far")

    def improve(self,somepop):
        self.prepare_localpop()
        if self.localsearch=='greedy':
            for dude in somepop:
                self.local_greedy_search(dude)
        elif self.localsearch=='ES':
            for dude in somepop:
                self.simple_ES(dude)

    def fill_refset_from_divset(self):
        # elite of divset becomes first tier of refset
        for dude in self.rt1:
            dude.copy_DNA_of(self.divset[0],copyscore=True,copyancestcode=True,copyparents=True)
            dude.new=True
            dude.elite=True
            self.divset.pop(0)
        for dude in self.divset:
            dude.dist=dude.min_distance_to_set(self.rt1)
        self.divset.sort_for('dist')
        self.divset.reverse()
        # now second tier of refset gets filled up with far-away dudes for diversity
        for dude in self.rt2:
            dude.copy_DNA_of(self.divset[0],copyscore=True,copyancestcode=True,copyparents=True)
            dude.new=True
            dude.elite=False
            self.divset.pop(0)
        self.refset.update_no()                     # so we only have to care about updating the number labels
        self.refset_ancestcode_update(status='from_divset')
        return
    
    def refset_ancestcode_update(self,status='from_divset'):
        """updating the individual's attribute 'ancestcode' is only important for in what 
        colour they appear in the ancestryplot, this method is not essential to the search algorithm"""
        if status == 'from_divset':
            for i,dude in enumerate(self.rt1):
                dude.ancestcode = float(i)/self.b1 * 0.1
            for i,dude in enumerate(self.rt2):
                dude.ancestcode = float(i)/(self.nref-self.b1) * 0.1 + 0.2
        elif status == 'mainloop':
            if self.refset_update_style == 'Glover':
                for i,dude in enumerate(self.rt1):
                    if dude.new == True:
                        dude.ancestcode = float(i)/self.b1 * 0.1
                    elif dude.ancestcode <= 0.1:  # that means it has been a newcomer in the last iteration
                        dude.ancestcode+=0.1
                for i,dude in enumerate(self.rt2):
                    if dude.new == True:
                        dude.ancestcode = float(i)/(self.nref-self.b1) * 0.1 + 0.2
                    elif 0.2 <= dude.ancestcode <= 0.3:  # that means it has been a newcomer in the last iteration
                        dude.ancestcode+=0.2
            if self.refset_update_style == 'Herrera':
                for i,dude in enumerate(self.refset):
                    if dude.new == True:
                        dude.ancestcode = float(i)/self.b1 * 0.1
                    elif dude.ancestcode <= 0.1:  # that means it has been a newcomer in the last iteration
                        dude.ancestcode+=0.1
                    elif 0.2 <= dude.ancestcode <= 0.3:  # formerly new diverse members from divset (only relevant during first iteration)
                        dude.ancestcode+=0.2

    def local_greedy_search(self,dude):
        """corresponds to (1+1)-ES with component-wise uniform mutation and no step size adaption"""
        evals = 0
        clone=self.localpop[0]; clone.copy_DNA_of(dude)
        while evals <= self.ls_depth:
            clone.mutate(1,self.ls_mstep)
            self.localpop.eval_one_dude(clone.no); evals+=1
            if clone.isbetter(dude):
                dude.copy_DNA_of(clone,copyscore=True,copyancestcode=True,copyparents=True)
            else:
                clone.copy_DNA_of(dude)
        return

    def simple_ES(self, dude):
        """(m/m,l)-ES with step size adaption via 1/5th rule (max_eval will always be rounded up to a multiple of 12)
        m and l stand for mu and lambda, which are the sizes for parent and offspring populations in traditional ES notation
        m is given as argument into this routine, but l comes with the size of this EA's self.localpop"""
        evals=0; thisstep=self.ls_mstep; lambd=self.localpop.psize
        while evals < self.ls_depth-lambd-1:
            # why evals < self.ls_depth-lambd-1 and not evals < self.ls_depth  ?
            # so localdepth is not exceeded after evaluation of next offspring
            # generation and the mixture individual from the best among them
            improv=0
            for ldude in self.localpop:
                ldude.copy_DNA_of(dude)
                ldude.mutate(P=1,sdfrac=thisstep)
            self.localpop.eval_all()
            self.localpop.sort()
            for ldude in self.localpop:
                if ldude.isbetter(dude): improv+=1
            if improv > self.localpop.psize/5+1 : thisstep*=self.ls_adapfactor  # step size adaption pretty sharp because local search supposed to be short
            else:thisstep/=self.ls_adapfactor
            self.localpop[0].become_mixture_of_multiple(self.localpop[:self.ls_mu])
            self.localpop.eval_one_dude(0)
            dude.copy_DNA_of(self.localpop[0],copyscore=True,copyancestcode=True,copyparents=True)
        return

    def select_new_refset_subsets(self):
        b1=self.b1; b2=self.nref-b1; subsets=[]
        for i in range(b1):
            for j in range(b2):
                dude1=self.rt1[i]
                dude2=self.rt2[j]
                if dude1.new or dude2.new:
                    subsets.append([dude1,dude2])    # a new cross-tier subset
        for i in range(b1):
            for j in range(i+1,b1):
                dude1=self.rt1[i]
                dude2=self.rt1[j]
                if dude1.new or dude2.new:
                    subsets.append([dude1,dude2])    # a new elite subset
        return subsets

    def recombine_Glover2003(self,dude1,dude2):
        if dude1.elite and dude2.elite:
            self.pool.change_size(self.pool.psize+4)
            self.pool[-4].become_mixture_of(dude1,dude2,fade=0-0.5*rand())
            self.pool[-3].become_mixture_of(dude1,dude2,fade=0+0.5*rand())
            self.pool[-2].become_mixture_of(dude1,dude2,fade=1-0.5*rand())
            self.pool[-1].become_mixture_of(dude1,dude2,fade=1+0.5*rand())
        elif dude1.elite or dude2.elite:
            self.pool.change_size(self.pool.psize+3)
            self.pool[-3].become_mixture_of(dude1,dude2,fade=0-0.5*rand())
            self.pool[-2].become_mixture_of(dude1,dude2,fade=0+rand())
            self.pool[-1].become_mixture_of(dude1,dude2,fade=1+0.5*rand())
        else:
            self.pool.change_size(self.pool.psize+2)
            self.pool[-2].become_mixture_of(dude1,dude2,fade=0+rand())
            if rand() < 0.5 :
                self.pool[-1].become_mixture_of(dude1,dude2,fade=0-0.5*rand())
            else:
                self.pool[-1].become_mixture_of(dude1,dude2,fade=1+0.5*rand())
        return

    def recombine_BLXa(self,dude1,dude2):
        """in their paper Herrera et al. compared two CO operators: arithmetic mean and BLX-alpha"""
        self.pool.change_size(self.pool.psize+1)
        self.pool[-1].BLX_alpha(dude1,dude2)
        return

    def recombine_arit(self,dude1,dude2):
        """in their paper Herrera et al. compared two CO operators: arithmetic mean and BLX-alpha"""
        self.pool.change_size(self.pool.psize+1)
        self.pool[-1].become_mixture_of(dude1,dude2)
        return

    def recombine_Brownlee(self,dude1,dude2):
        """ Jason Brownlee presents in his book and on his webpage somewhat modified version of scatter seach. Nevertheless, his recombination operator
        represents an intermediate search scrutiny level compared to [Glover2003] on the one hand and [Herrera2006] on the other."""
        self.pool.change_size(self.pool.psize+2)
        self.pool[-2].become_mixture_of(dude1,dude2,fade=0.5+rand())
        self.pool[-1].become_mixture_of(dude1,dude2,fade=0.5-rand())
        return

    def sort_refset_tiers(self):
        """sorts first tier for ascending cost and second tier for ASCENDING distance values"""
        self.rt1.sort()
        for dude in self.rt2:
            dude.dist=dude.min_distance_to_set(self.rt1)
        self.rt2.sort_for('dist')
        self.rt2.reverse()

    def refset_update_from(self,somepop):
        for dude in self.refset:
            dude.new=False
        changed=False
        if self.refset_update_style == 'Glover':
            self.sort_refset_tiers()
            for dude in somepop:
                recently_changed=False
                # following: acces to elite via fitness and access to second tier through far away position
                if dude.isbetter(self.rt1[-1]):
                    self.rt1[-1].copy_DNA_of(dude,copyscore=True,copyancestcode=True,copyparents=True)
                    self.rt1[-1].new=True; changed=True; recently_changed=True
                else:
                    dude.dist=dude.min_distance_to_set(self.rt1)
                    if dude.dist > self.rt2[-1].dist:
                        self.rt2[-1].copy_DNA_of(dude,copyscore=True,copyancestcode=True,copyparents=True)
                        self.rt2[-1].new=True; changed=True; recently_changed=True
                if recently_changed: self.sort_refset_tiers()
        elif self.refset_update_style == 'Herrera':
            self.refset.sort()
            for dude in somepop:
                recently_changed=False
                # following: acces to Refset only via fitness; division into two tiers irrelevant now
                if dude.isbetter(self.refset[-1]):
                    self.refset[-1].copy_DNA_of(dude,copyscore=True,copyancestcode=True,copyparents=True)
                    self.refset[-1].new=True; changed=True; recently_changed=True
                if recently_changed: self.refset.sort()
        else:
            raise ValueError("self.refset_update_style must be either 'Glover' or 'Herrera'")
        return changed

    def do_step(self):
        print 'SCS: entering do_step() in generation ',self.refset.gg
        self.pool.change_size(0)
        subsets=self.select_new_refset_subsets()
        for pair in subsets:
            dude1,dude2=pair
            #print 'SCS: scores of subset: ',dude1.score,dude2.score
            if self.recomb=='arit':
                self.recombine_arit(dude1,dude2)
            elif self.recomb=='BLXa':
                self.recombine_BLXa(dude1,dude2)
            elif self.recomb=='Brownlee':
                self.recombine_Brownlee(dude1,dude2)
            elif self.recomb=='Glover':
                self.recombine_Glover2003(dude1,dude2)
            else:
                raise NotImplementedError("you need to set the attribute self.recomb right, the options are 'arit', 'BLXa', 'Brownlee', and 'Glover'")
        self.pool.eval_all()
        print 'SCS: pool: best score of after evaluation: ',np.min(self.pool.get_scores())
        self.improve(self.pool)
        was_change=self.refset_update_from(self.pool)
        self.refset_ancestcode_update(status='mainloop')
        self.advance_generation()
        if self.refset_update_style == 'Glover':
            self.rt1.mark_oldno(); self.rt2.mark_oldno()
            self.rt1.update_no(); self.rt2.update_no()
        elif self.refset_update_style == 'Herrera':
            self.refset.mark_oldno()
            self.refset.update_no()
        print 'SCS: refset: best score of after completing generation {}: {}'.format(self.refset.gg,np.min(self.refset.get_scores()))
        for gc in self.generation_callbacks:
            gc(self)
        return was_change
            
    def advance_generation(self):
        for p in [self.refset,self.divset,self.localpop]:
            p.advance_generation()
        for p in [self.rt1,self.rt2]:
            p.gg+=1

    def run(self,maxgenerations):
        for i in range(maxgenerations):
            was_change=self.do_step()
            if not mod(i,10): print 'SCS: tier 1: best score: ',self.rt1[0].score
            if not was_change: break

    def complete_algo(self,maxgenerations):
        tstart=clock()
        ini_evals=self.refset.neval+self.pool.neval+self.localpop.neval
        for dude in self.refset+self.pool+self.localpop: dude.set_bad_score()
        self.renew_divset()
        self.divset.sort()
        print 'SCS: raw divset: best score and DNA: ',self.divset[0].score,'   ',self.divset[0].DNA
        self.improve(self.divset)
        self.divset.sort()
        print 'SCS: improved divset: best score and DNA of: ',self.divset[0].score,'   ',self.divset[0].DNA
        #self.refset_update_from(self.divset)
        self.fill_refset_from_divset()
        for gc in self.generation_callbacks:
            gc(self)
        print 'SCS: ---------------------------------------- beginning mainloop ----------------------------------------'
        for i in range(maxgenerations):
            was_change=self.do_step()
            if self.refset_update_style == 'Glover':
                print 'SCS: generation ',self.rt1.gg,', tier 1: best score: ',self.rt1[0].score,"  (update style is 'Glover')"
            elif self.refset_update_style == 'Herrera':
                print 'SCS: generation ',self.refset.gg,', refset: best score: ',self.refset[0].score,"  (update style is 'Herrera')"
            if not was_change:
                print 'SCS: There was no change in the refset over the last iteration. Scatter Search stopped.'
                break
        print 'SCS: ----------------------------------------- ending mainloop ------------------------------------------'
        if self.refset.whatisfit in ['min','minimise','minimize']:
            best_idx=argmin(self.refset.get_scores())
        else:
            best_idx=argmax(self.refset.get_scores())
        print 'SCS: final refset: best score and DNA: ',self.refset[best_idx].score,'   ',self.refset[best_idx].DNA
        tend=clock()
        final_evals=self.refset.neval+self.pool.neval+self.localpop.neval
        print 'SCS summary: final score {} after {} function evaluations in {} seconds'.format(np.min(self.refset.get_scores()),final_evals-ini_evals,tend-tstart)
        return




