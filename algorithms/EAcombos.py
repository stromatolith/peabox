#!python
"""
a collection of EA trials
"""

from os.path import join
from cPickle import Pickler
from copy import copy, deepcopy
import numpy as np
from numpy import array, arange, asfarray, zeros, ones, mod, floor, where
from numpy import exp, log, mean, argsort, eye, fabs, sqrt
from numpy import add, prod, dot, vdot, inner, outer, diag  # inner=Skalarprodukt, outer=Tensorprodukt
from numpy.linalg import inv, eig, eigh, qr
from numpy.random import rand, randn, randint, multivariate_normal

from peabox_population import population_like
#from peabox_cecpop import population_like
from peabox_helpers import parentselect_exp, condense
from peabox_helpers import diverse_selection_from as divsel
from peabox_helpers import less_aggressive_diverse_selection_from as ladivsel
#from peabox_recorder import Recorder
try:
    from local1Dsearch import NMStepper, GoldenStepper
except:
    pass


class ComboBase(object):

    def __init__(self,F0pop,F1pop):
        assert len(F0pop)==len(F1pop)
        self.F0=F0pop; self.F0.ownname='F0pop'
        self.F1=F1pop; self.F1.ownname='F1pop'
        for dude in self.F0: dude.set_bad_score()
        for dude in self.F1: dude.set_bad_score()
        self.ps=F0pop.psize
        self.generation_callbacks=[]  # put functions into this list to be called after each generation with this EA instance as argument, e.g. plot current best solution
        self.more_stop_crits=[stopper]
        self.status='ini'
        self.ownname='EAC'
        self.maxeval=1e4
        self.breaking_distance=100   # if you want to stop so you still have 100 trials left before maxeval is reached, then use the function stop_at_distance(eaobj) which is defined below
        self.save_best=True
        self.bestdude=None
        # for mstep control:
        self.ini_mstep=0.15
        self.mstep=0.15
        self.anneal=0.04       # exponential decay of mutation step size from generation to generation
        self.heatups=0
        self.ini_smallstep=0.
        self.smallstep=0.
        self.heatup_factor=300.
        self.hutshrink=1./3.2  # shrink facto for heatup threshold, i.e. smallstep reduction factor

    def mstep_reset(self):
        self.mstep=self.ini_mstep
        self.heatups=0
        self.smallstep=self.ini_smallstep

    def advance_generation(self):
        for i,pdude in enumerate(self.F0):
            pdude.copy_DNA_of(self.F1[i],copyscore=False,copyancestcode=True,copyparents=True)
        self.F0.advance_generation()
        self.F0.eval_all()

    def do_step(self):
        self.create_offspring()
        self.advance_generation()
        self.mstep_control()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def do_random_step(self,update_mstep=False):
        self.status='random_dudes'
        self.F1.new_random_genes()
        self.advance_generation()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if update_mstep: self.mstep_control()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def do_fake_step(self,status='normal'):
        self.status=status
        self.F0.advance_generation()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def stationary_step(self):
        # mstep (the single algo internal dynamic state variable) stays constant
        self.create_offspring()
        self.advance_generation()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        for gc in self.generation_callbacks:
            gc(self)
            
    def mstep_control(self):
        self.mstep*=exp(-self.anneal)
        if self.mstep<=self.smallstep:
            self.mstep*=self.heatup_factor
            self.smallstep*=self.hutshrink
            self.heatups+=1

    def run(self,generations,status='normal'):
        self.status=status
        for i in range(generations):
            self.do_step()
            if self.other_stopcrit_returned_True():
                break

    def other_stopcrit_returned_True(self):
        if len(self.more_stop_crits) != 0:
            morestopvals=[]
            for crit in self.more_stop_crits:
                morestopvals.append(crit(self))
            if True in morestopvals:
                #print "algorithm's run() terminated because of '+str(morestopvals.index(True))+'th additional stopping criterion"
                return True
            else:
                return False
        else:
            return False

    def zeroth_generation(self,random_ini=True):
        if random_ini:
            self.status='random_dudes'
            self.F0.new_random_genes()
        self.F0.zeroth_generation()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def simple_run(self,generations,random_starting_DNAs=True):
        self.mstep_reset()
        if random_starting_DNAs:
            self.F0.new_random_genes()
        self.F0.zeroth_generation()
        self.status='random_dudes'
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)
        self.run(generations,status='normal')

    def simple_run_with_seed_DNAs(self,generations,seedDNAs):
        self.mstep_reset()
        self.F0.new_random_genes()
        for i,dna in enumerate(seedDNAs):
            self.F0[i].set_DNA(dna)
        self.F0.zeroth_generation()
        self.status='random_dudes'
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)
        self.run(generations,status='normal')
    
    def change_psize(self,newsize):
        self.F0.change_size(newsize)
        self.F1.change_size(newsize)
        self.ps=newsize

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
    
    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        # remove unpicklable ctypes stuff from the dictionary used by the pickler
        del odict['more_stop_crits']
        del odict['generation_callbacks']
        return odict


    def pickle_self(self):
        ofile=open(join(self.F0.picklepath,self.ownname+'_'+self.F0.label+'.txt'), 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()


#--- fractal-structured population merging scheme, generalised, as addition to ComboBase
class ComboBaseFractal(ComboBase):
    
    def __init__(self,F0pop,F1pop):
        ComboBase.__init__(self,F0pop,F1pop)
        self.frc_B=3       # branching factor
        self.frc_L=4       # total number of layers
        self.frc_l=0       # current layer
        self.frc_n=0       # current counter
        self.frc_G1=10     # how many generations per population in all the branches
        self.frc_G2=30     # how many generations should the final population evolve
        self.cpres=2.      # selection pressure applied during condensation
        self.celite=max(2,self.F0.psize/10)   # elite size for condensation
        self.sclist=[]     # list for lists of active populations
        self.completed=[]  # lists number of completed and erased bunches in each layer
        self.ready=[]      # lists number of populations ready to use in each layer
        self.storage=[]    # list holding an amount of B populations per layer
        self.frac_mem=[]   # save some basic data to enable plotting sth like score improvements in the tree structure
        self.score_memory=0.  # for remembering initially best score of a population
        self.mstep_memory=0.  # for remembering initial mstep of a population
        self.condensation_memory=[]  # remembering dude.no, dude.pa and dude.ancestcode occuring after condensation for making a histogram on all selection choices cumulating all condensation steps (purpose: verify selection as intended)
        self.frac_reset()  # remake lists of empty lists, fill storage with populations
        self.big_final=True
    
    def optimal_fractal_dimensions(self,B=None,G=None,endrun_thresh=3,maxeval=None,NM=True):
        if B is not None: self.frc_B=B
        if G is not None: self.frc_G1=G
        if maxeval is not None: self.maxeval=maxeval
        g2_to_g1=[]; l=0; g2=self.maxeval
        nm_evals=self.F0.ng*70*NM  # planning to leave dimension*70 calls to NM
        while g2 > 2*self.frc_G1:
            l+=1
            tree_evals=array([self.frc_B**i for i in range(1,l)]) # how many populations there will be in each layer of the tree
            tree_evals = np.sum(tree_evals)*self.ps*self.frc_G1  # how many evaluations the tree will cost (without the stem)
            tree_evals-=self.frc_B**(l-1)*self.ps # account for the fake generations without evaluation calls during condensation steps
            g2 = (self.maxeval-tree_evals-nm_evals) / (self.ps * max(1, self.big_final*self.frc_B))
            g2_to_g1.append(g2/float(self.frc_G1))
        g2_to_g1=array(g2_to_g1)
        self.frc_L=np.sum(where(g2_to_g1>endrun_thresh,1,0))   # use last number of layers where threshold was met
        tree_evals = np.sum(array([self.frc_B**i for i in range(1,self.frc_L)]))*self.ps*(self.frc_G1-1) + self.frc_B**(self.frc_L-1)*self.ps
        #print 'tree evaluations: ',tree_evals
        self.frc_G2 = (self.maxeval - tree_evals - nm_evals) / (self.ps*max(1, self.big_final*self.frc_B)) + 1
        #print 'G2 = ({} - {} - {}) / {} = {}'.format(self.maxeval,tree_evals,nm_evals,self.ps,self.frc_G2)
        stem_evals=(self.frc_G2-1) * self.ps * max(1, self.big_final*self.frc_B)
        #print 'stem evaluations: ',stem_evals
        nm_evals=self.maxeval-tree_evals-stem_evals
        #print 'remaining for NM: ',nm_evals
        print '\n\n'+80*'-'+'\n'+10*'-'+'   fractal dimensioning report   '+30*'-'
        print 'g2_to_g1: ',g2_to_g1
        print 'where threshold met: ',where(g2_to_g1>endrun_thresh,1,0)
        print 'ps={}, maxeval={}  endrun multiplicator: {}   endrun-ps={}'.format(self.ps,self.maxeval,endrun_thresh, self.ps * max(1, self.big_final*self.frc_B))
        print 'with G1={} and B={}--> optimal:  L={}'.format(self.frc_G1,self.frc_B,self.frc_L)
        msg='--> {} generations remain for final population'.format(self.frc_G2)
        if NM: msg+=' and {} trials for Nelder-Mead'.format(self.maxeval-tree_evals-stem_evals)
        print msg+'\ntotal: {} + {} + {} = {} calls to the objective function'.format(tree_evals,stem_evals,nm_evals, tree_evals+stem_evals+nm_evals)
        print 'L={} would look like this:'.format(self.frc_L+1)
        tree_evals+=self.frc_B**self.frc_L*self.ps*self.frc_G1 - self.frc_B**(self.frc_L-1)*self.ps
        print 'total: {} - {} - {} = {} calls remaining for stem'.format(self.maxeval,tree_evals,nm_evals,self. maxeval-tree_evals-nm_evals)
        print 80*'-'+'\n\n'
        self.frac_reset()
    
    def set_heatup_ratio(self,ratio):
        # populations run for self.frc_G1 generations and that leads to a certain decrease in self.mstep
        # if ratio is 1/3 then the heatup_factor will be set so it rolls back the decrease that has occured during the last third of that run
        self.heatup_factor=exp(self.frc_G1*float(ratio)*self.anneal)
        

    def frac_reset(self):
        self.sclist=[]     # list for lists of active populations
        self.completed=[]  # lists number of completed and erased bunches in each layer
        self.ready=[]      # lists number of populations ready to use in each layer
        self.storage=[]
        self.frac_mem=[]
        self.condensation_memory=[]  # remembering dude.no, dude.pa and dude.ancestcode occuring after condensation for making a histogram on all selection choices cumulating all condensation steps (purpose: verify selection as intended)
        for l in range(self.frc_L):
            self.sclist.append([])
            self.completed.append(0)
            self.ready.append(0)
            tmplist=[]
            for i in range(self.frc_B):
                tmplist.append(population_like(self.F0,size=self.F0.psize))
            self.storage.append(tmplist)
            self.frac_mem.append([])
                    
    def condense(self,pops):
        B=self.frc_B; ps=self.F0.psize;  n1=self.celite
        for j,p in enumerate(pops):
            #print 'population {} is giving DNA'.format(p.ownname)
            for i,dude in enumerate(p[:n1]):
                self.F0[j*n1+i].copy_DNA_of(dude,copyscore=True); self.F0[j*n1+i].ancestcode=np.round(j%B*(1./B),3)
        for i,dude in enumerate(self.F0[B*n1:]):
            num=parentselect_exp(ps,self.cpres)
            dude.copy_DNA_of(pops[i%B][num],copyscore=True,copyparents=False); dude.ancestcode=np.round(i%B*(1./B),3)

    def new_layer_zero_bunch(self):
        for i in range(self.frc_B):
            self.mstep=self.ini_mstep; self.mstep_memory=self.mstep
            self.do_random_step(); self.score_memory=self.F0[0].score
            self.run(self.frc_G1-1)
            self.sclist[self.frc_l].append(self.F0[0].score); self.frc_n+=1
            self.storage[self.frc_l][self.ready[self.frc_l]].copy_otherpop(self.F0,copyscore=True,copyancestcode=True,copyparents=True)
            self.storage[self.frc_l][self.ready[self.frc_l]].ownname='p'+str(self.frc_n).zfill(2)
            self.frac_mem_entry(False)
            self.ready[self.frc_l]+=1

    def summation(self):
        self.mstep=self.ini_mstep*exp(-(self.frc_l+1)*self.frc_G1*self.anneal)
        self.mstep*=self.heatup_factor**(self.frc_l+1)
        #print 'p {} mstep set using l={} (summation)'.format(self.frc_n+1,self.frc_l)
        self.condense(self.storage[self.frc_l])
        self.condensation_memory_entry()
        self.do_fake_step(); self.mstep_memory=self.mstep; self.score_memory=self.F0[0].score
        if self.frc_l+1 == self.frc_L-1:
            #print 'l={}   long evo G2={}'.format(self.frc_l,self.frc_G2)
            self.run(self.frc_G2-1)
        else:
            #print 'l={}   short evo G1={}'.format(self.frc_l,self.frc_G1)
            self.run(self.frc_G1-1)
        self.sclist[self.frc_l+1].append(self.F0[0].score); self.frc_n+=1
        self.storage[self.frc_l+1][self.ready[self.frc_l+1]].copy_otherpop(self.F0,copyscore=True,copyancestcode=True,copyparents=True)
        self.storage[self.frc_l+1][self.ready[self.frc_l+1]].ownname='p'+str(self.frc_n).zfill(2)
        print '{} recieved DNA from {} after {} generations and {} calls'.format('p'+str(self.frc_n).zfill(2), [p.ownname for p in self.storage[self.frc_l]], self.F0.gg, self.tell_neval())
        self.frac_mem_entry(True)
        self.ready[self.frc_l+1]+=1
        self.ready[self.frc_l]=0
        self.completed[self.frc_l]+=1
        self.sclist[self.frc_l]=[]
    
    def combine(self):
        oldps=self.ps; B=self.frc_B
        self.change_psize(self.frc_B*self.ps)
        self.mstep=self.ini_mstep*exp(-(self.frc_l+1)*self.frc_G1*self.anneal)
        self.mstep*=self.heatup_factor**(self.frc_l+1)
        #print 'p {} mstep set using l={} (combine)'.format(self.frc_n+1,self.frc_l)
        for j,p in enumerate(self.storage[self.frc_l]):
            for i,dude in enumerate(p):
                self.F0[j*oldps+i].copy_DNA_of(dude,copyscore=True); self.F0[j*oldps+i].ancestcode=np.round(j%B*(1./B),3)
        self.do_fake_step(); self.mstep_memory=self.mstep; self.score_memory=self.F0[0].score
        self.run(self.frc_G2-1)
        self.sclist[self.frc_l+1].append(self.F0[0].score); self.frc_n+=1
        print '{} recieved DNA from {} after {} generations and {} calls'.format('p'+str(self.frc_n).zfill(2), [p.ownname for p in self.storage[self.frc_l]], self.F0.gg, self.tell_neval())
        self.frac_mem_entry(True)
        self.ready[self.frc_l+1]+=1
        self.ready[self.frc_l]=0
        self.completed[self.frc_l]+=1
        self.sclist[self.frc_l]=[]
        
    def real_fractal_run(self):
        while len(self.sclist[-1])==0:
            if (len(self.sclist[self.frc_l]) >=0) and (len(self.sclist[self.frc_l]) < self.frc_B):
                if self.frc_l==0:
                    self.new_layer_zero_bunch()
                    self.frc_l+=1
                else:
                    self.frc_l=0
            elif len(self.sclist[self.frc_l])==self.frc_B:
                if self.big_final and (self.frc_l+1 == self.frc_L-1):
                    self.combine()
                else:
                    self.summation()
                self.frc_l+=1
            else:
                raise ValueError('sclist[{}] has wrong length {}'.format(self.frc_l,len(self.sclist[self.frc_l])))
    
    def real_fractal_run_only_tree(self):
        while len(self.sclist[-1])==0:
            if (len(self.sclist[self.frc_l]) >=0) and (len(self.sclist[self.frc_l]) < self.frc_B):
                if self.frc_l==0:
                    self.new_layer_zero_bunch()
                    self.frc_l+=1
                else:
                    self.frc_l=0
            elif len(self.sclist[self.frc_l])==self.frc_B:
                if self.frc_l+1 == self.frc_L-1:
                    break
                else:
                    self.summation()
                self.frc_l+=1
            else:
                raise ValueError('sclist[{}] has wrong length {}'.format(self.frc_l,len(self.sclist[self.frc_l])))
        return self.storage[self.frc_l]
    
    def condensation_memory_entry(self):
        d={'n':self.frc_n,
           'ac':self.F0.get_ancestcodes(),
           'pa':[dude.pa for dude in self.F0],
           'pb':[dude.pb for dude in self.F0]}
        self.condensation_memory.append(d)
    
    def frac_mem_entry(self,sumflag):
        if sumflag:
            layer=self.frc_l+1
        else:
            layer=self.frc_l
        d={'n':self.frc_n,
           'l':layer,
           'bunch':self.completed[layer],
           'ready':self.ready[layer],
           'score_ini':self.score_memory,
           'score_fin':self.F0[0].score,
           'mstep_ini':self.mstep_memory,
           'mstep_fin':self.mstep}
        self.frac_mem[layer].append(d)
            
    def pickle_condensation_memory(self):
        ofile=open(join(self.F0.picklepath,'condensation_memory_'+self.F0.label+'.txt'),'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self.condensation_memory)
        ofile.close()
                
    def pickle_fractal_memory(self):
        ofile=open(join(self.F0.picklepath,'fractal_memory_'+self.F0.label+'.txt'),'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self.frc_B)
        einmachglas.dump(self.frc_L)
        einmachglas.dump(self.frc_n)
        einmachglas.dump(self.frc_G1)
        einmachglas.dump(self.frc_G2)
        einmachglas.dump(self.frac_mem)
        einmachglas.dump(self.anneal)
        ofile.close()
        
        
#--- local search along straight lines as addition to ComboBase
class ComboBaseWLS(ComboBase):
    """
    of the F0-population inside here each member is able to independently do a
    1D local search along a straight line cutting through the search space, but
    the thing is all members do each next step at the same time, so you can
    stick to F0pop.eval_all() which allows you tu use this type of local
    search with a cecPop population for access to the CEC-2013 test functions.
    """

    def __init__(self,F0pop,F1pop,lstype='golden'):
        ComboBase.__init__(self,F0pop,F1pop)
        self.searchers=[]
        self.focusbunch=self.ps/3
        if lstype=='golden':
            for dude in self.F0:
                self.searchers.append(GoldenStepper(dude,0.2*200))
        elif lstype=='NM':
            for dude in self.F0:
                self.searchers.append(NMStepper(dude,0.2*200))
        else:
            raise ValueError('invalid argument for local search type in constructor of EA combo class')

    def turn_all_search_lines_towards(self,point,perturbation=0,reset_delta_factor=None):
        for s in self.searchers:
            s.turn_line_towards(point,perturbation,reset_delta_factor)
            
    def search1D(self,G,inistep_factor=0.2,perturb=0.2):
        self.status='1D_search'
        allDNAs=array(self.F0.get_DNAs()); focuspoint=mean(allDNAs[:self.focusbunch,:],axis=0)
        self.turn_all_search_lines_towards(focuspoint,perturbation=perturb,reset_delta_factor=inistep_factor)
        for s in self.searchers:
            s.status='reset'
        for i,dude in enumerate(self.F0):
            dude.no=i; dude.oldno=i; dude.pa=i; dude.pb=-1; dude.ancestcode=i/float(self.ps-1)
        for g in range(G):
            self.do_1D_step()
            if self.other_stopcrit_returned_True(): break
        self.do_last_1D_step()
    
    def do_1D_step(self):
        self.F0.advance_generation()
        for s in self.searchers:
            #print 'now follows 1D search step of dude ',s.dude.no
            s.dude_evaluated=True
            s.step()
        self.F0.eval_all()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)
        
    def do_last_1D_step(self):
        self.status='finalizing_1D_search'
        self.F0.advance_generation()
        for s in self.searchers:
            s.leave()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)



#--- downhill-simplex search by Nelder and Mead as addition to ComboBase
class ComboBaseWNM(ComboBase):

    def __init__(self,F0pop,F1pop,simpop,trialpop):
        ComboBase.__init__(self,F0pop,F1pop)
        self.N=simpop.ng
        if simpop.psize != self.N+1: simpop.change_size(self.N+1,sortbefore=True,updateno=True)
        if trialpop.psize != 1: trialpop.change_size(1,sortbefore=True,updateno=True)
        self.sim=simpop; self.sim.ownname='simplex'
        self.trials=trialpop; self.trials.ownname='trials'
        for dude in self.sim: dude.set_bad_score()
        for dude in self.trials: dude.set_bad_score()
        self.probe=trialpop[0]
        self.rho = 1
        self.chi = 2
        self.psi = 0.5
        self.sigma = 0.5
        self.xbar=zeros(self.N); self.xr=zeros(self.N); self.xe=zeros(self.N); self.xc=zeros(self.N); #self.xcc=zeros(self.N)

    def initialize_simplex(self,delta=0.05):
        for k in range(1,self.N+1):
            self.sim[k].copy_DNA_of(self.sim[0])
            self.sim[k].DNA[k-1]+=delta*self.sim[k].widths[k-1]
        self.sim.eval_all()
        self.sim.mark_oldno(); self.sim.sort(); self.sim.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def NMrun(self, xtol=1e-4, ftol=1e-4, inidelt=0.05, maxiter=None, maxfun=None, reset_internal_state=True,startgg=None):
        self.status='Nelder_Mead'
        if startgg is not None:
            self.sim.set_generation(startgg)
            self.trials.set_generation(startgg)
        inineval=self.tell_neval()
        if reset_internal_state:
            self.initialize_simplex(delta=inidelt)
        iterations=1
        while True:
            #print 'iteration ',iterations
            max_distance_to_best=0.; max_fdiff=0.
            for dude in self.sim[1:]:
                d=dude.distance_from(self.sim[0],uCS=True)
                if d > max_distance_to_best: max_distance_to_best=d
                fd=abs(self.sim[0].score-dude.score)
                if fd > max_fdiff: max_fdiff=fd
            if maxiter is not None and (iterations >= maxiter):
                print 'NelderMead.run() terminated because of maxiter'
                break
            if maxfun is not None and ((self.tell_neval()-inineval) >= maxfun):
                print 'NelderMead.run() terminated because of calls = {} >= maxfun = {}'.format(self.tell_neval()-inineval, maxfun)
                break
            # break if widest simplex span <=xtol and function value differences <= ftol
            if (max_distance_to_best <= xtol)  and (max_fdiff <= ftol):
                print 'NelderMead.run() terminated because of xtol and ftol'
                break
            if self.other_stopcrit_returned_True():
                print 'NelderMead.run() terminated because of one of the additional stopping criterion'
                break
            self.NMstep_serial()
            #print 'step done, best: {} 2nd-worst: {} worst: {} trial {}'.format(self.sim[0].score,self.sim[-2].score,self.sim[-1].score,self.trials[0].score)
            iterations += 1

    def NMstep_serial(self):
        self.sim.advance_generation(); self.trials.advance_generation()
        for i,dude in enumerate(self.sim):
            dude.ancestcode=0.301+(float(i)/float(self.N))*0.098
        best=self.sim[0]; worst=self.sim[-1]; secondworst=self.sim[-2]
        self.xbar*=0.
        for dude in self.sim[:-1]:
            self.xbar+=dude.DNA
        self.xbar/=float(self.N)  # mean of all corners except the worst one
        #print 'xbar: ',self.xbar
        doshrink = 0
        #print 'scores at beginning of step: ',self.sim.get_scores()
        self.probe.set_DNA((1+self.rho)*self.xbar - self.rho*worst.DNA) # xr: mirror worst beyond xbar (rho=1 constantly)   ... is same as sim[-1]+2*(xbar-sim[-1])
        self.trials.eval_all(); self.xr[:]=self.probe.DNA; fxr=self.probe.score
        #print 'xr has score {} with DNA {}'.format(fxr,self.xr)
        if self.probe.isbetter(best): # if mirror image of worst better than best --> go further that way by factor 2 (chi is also constant)
            #print 'entering branch A'
            self.probe.set_DNA((1+self.rho*self.chi)*self.xbar - self.rho*self.chi*worst.DNA) # xe: twice as far from mean as xr in same direction  -->  expeanded mirrored simplex
            self.trials.eval_all(); self.xe[:]=self.probe.DNA; fxe=self.probe.score
            #print 'xe has score {} with DNA {}'.format(fxe,self.xe)
            if self.probe.isbetter(fxr):
                worst.set_DNA(self.xe); worst.score=fxe  # if improved then keep it that way else keep the unstretched mirror image
                worst.ancestcode=0.17
                #print 'kept: xe'
            else:
                worst.set_DNA(self.xr); worst.score=fxr  # if expanded reflection didn't work out stay with the simple reflection
                worst.ancestcode=0.07
                #print 'kept: xr in favour of xe'
        else: # fsim[0] <= fxr
            #print 'entering branch B'
            if secondworst.isworse(fxr):   # i.e. not worst any more
                worst.set_DNA(self.xr); worst.score=fxr  # if not worst any more stay with the simple reflection
                worst.ancestcode=0.05
                #print 'kept: xr because better than second worst'
            else: # fxr >= fsim[-2]  # reflected x is still the worst
                if worst.isworse(fxr):   # reflected x is still the worst but still better than the original worst
                    self.probe.set_DNA((1+self.psi*self.rho)*self.xbar - self.psi*self.rho*worst.DNA) # xc: half as far from mean as xr in same direction  -->  reduced mirrored simplex
                    self.trials.eval_all(); self.xc[:]=self.probe.DNA; fxc=self.probe.score
                    if self.probe.isbetter(fxr):
                        worst.set_DNA(self.xc); worst.score=fxc # if pulling back xr was helpful, then keep it
                        worst.ancestcode=0.27
                        #print 'kept: xc'
                    else:
                        doshrink=1 # best one pulls all other corners closer to himself
                else:  # Perform an inside contraction, because reflection was a bad thing and worsened the worst
                    self.probe.set_DNA((1-self.psi)*self.xbar + self.psi*worst.DNA) # xcc: in the middle in between original worst point and mean  -->  reduced original simplex
                    self.trials.eval_all(); #xcc[:]=self.probe.DNA; #fxcc=self.probe.score
                    if self.probe.isbetter(worst):
                        worst.copy_DNA_of(self.probe,copyscore=True) # that helped, so keep it
                        worst.ancestcode=0.47
                        #print 'kept: xcc'
                    else:
                        doshrink = 1 # best one pulls all other corners closer to himself
                if doshrink: # done when a) reflection was mildly helpful, but pulling it back worsened it again or b) pulling initial worst closer towards the others did not help
                    for i,dude in enumerate(self.sim[1:]): # best one pulls all other corners closer to himself
                        oldDNA=dude.get_copy_of_DNA(); dude.set_DNA(best.DNA + self.sigma*(oldDNA - best.DNA))
                        dude.ancestcode=1.-(float(i)/float(self.N))*0.099
                    self.sim.eval_all()
                    #print 'simplex shrunken'
        #print 'scores at end of step: ',self.sim.get_scores()
        self.sim.mark_oldno(); self.sim.sort(); self.sim.update_no()
        self.F0[0].copy_DNA_of(self.sim[0],copyscore=True)
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)
        return

    def standard_NM_end(self, xtol=1e-16, ftol=1e-16, warnlim=None, pumpup=3, maxiter=None):
        self.sim[0].copy_DNA_of(self.F0[0],copyscore=True)
        remaining_trials=self.maxeval-self.tell_neval()
        print 'remaining trials: {} - {} = {}'.format(self.maxeval,self.tell_neval(),remaining_trials)
        if warnlim is None: warnlim=40*self.F0.ng
        if remaining_trials < warnlim:
            raise ValueError('remaining trials left: '+str(remaining_trials))
        oldscore=self.sim[0].score
        self.NMrun(xtol=xtol, ftol=ftol, inidelt=pumpup*self.mstep, maxiter=maxiter, maxfun=remaining_trials, reset_internal_state=True,startgg=self.F0.gg)
        print 'effect of NM: score {} --> {}'.format(oldscore,self.sim[0].score)

    def tell_neval(self):
        return self.F0.neval+self.sim.neval+self.trials.neval
    
    def tell_best_score(self,andDNA=False):
        if (self.bestdude is not None) and (self.save_best is not None) and (self.bestdude.isbetter(self.F0[0])) and (self.bestdude.isbetter(self.sim[0])):
            if andDNA:
                return self.bestdude.score,self.bestdude.get_copy_of_DNA()
            else:
                return self.bestdude.score
        elif self.status=='Nelder_Mead':
            if andDNA:
                return self.sim[0].score,self.sim[0].get_copy_of_DNA()
            else:
                return self.sim[0].score
        else:
            if andDNA:
                return self.F0[0].score,self.F0[0].get_copy_of_DNA()
            else:
                return self.F0[0].score

    def update_bestdude(self):
        if self.status == 'Nelder_Mead':
            if self.bestdude is None:
                self.bestdude=deepcopy(self.sim[0])
            else:
                if self.sim[0].isbetter(self.bestdude):
                    self.bestdude.copy_DNA_of(self.sim[0],copyscore=True,copyparents=True,copyancestcode=True,copymutagenes=True)
                    self.bestdude.gg=self.sim[0].gg
        else:
            ComboBase.update_bestdude(self)


class ComboA(ComboBase):
    """
    ES + GA + DE + exponential decay of mutation step sizes
    """

    def __init__(self,F0pop,F1pop):
        ComboBase.__init__(self,F0pop,F1pop)
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
        self.ownname='eacA'

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

    def __init__(self,F0pop,F1pop):
        ComboBase.__init__(self,F0pop,F1pop)
        self.m=len(F0pop)     # mu, i.e. size of parent population
        self.l=len(F1pop)     # lambda, i.e. size of F1-generation
        self.Pm=0.6           # mutation probability applied to each gene
        #self.muttype='randn'   # mutation type: 'randn', 'rand', 'fixstep'
        self.mstep=0.15        # mutation step size in relation to domain width
        self.anneal=0.03       # exponential decay of mutation step size from generation to generation
        self.eml=0.2          # elite mutation limit, the maximal mutation step size multiplier for elite members
        self.cigar2uniform=0.2# what ratio between the two CO operators
        self.WHX2BLX=0.2       # what ratio between the two CO operators
        self.mr=0.5           # mutation strength multiplier for recombinants
        self.selpC=1.         # parent selection pressure for bunch C
        self.selpD=2.         # parent selection pressure for bunch D
        self.selpE=2.         # parent selection pressure for bunch E
        self.selpF=3.         # parent selection pressure for bunch E
        self.DEsr=[0.0,0.4]     # DE scaling range: difference vectors will be scaled with uniformly sampled random number from within this interval
        self.cigar_aspect=10.  # aspect ratio for cigar_CO()
        self.bunchsizes=None  # e.g. [4,12,24,30,10]  for the sizes of bunches A-E if the population size is 80
        self.storageA=None    # placeholder for an additional population (needed for fractal_run)
        self.storageB=None    # placeholder for an additional population (needed for fractal_run)
        self.ownname='eacB'

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
            fdude.mutate(1.0,self.eml*self.mstep*float(i)/float(len(self.bunchA)))  # mutate the best few just a little bit and the best one not at all
            fdude.ancestcode=0.09*i/len(self.bunchA)  # black-read
        for i,fdude in enumerate(self.bunchB):
            fdude.copy_DNA_of(self.F0[fdude.no])
            fdude.mutate(self.Pm,self.mstep)          # just mutation
            fdude.ancestcode=0.103+0.099*i/len(self.bunchB)  # blue-violet
        for i,fdude in enumerate(self.bunchC):
            parent=parentselect_exp(N,self.selpC)     # select a parent favoring fitness and mutate
            fdude.copy_DNA_of(self.F0[parent],copyscore=False,copyancestcode=False,copyparents=False)
            fdude.mutate(self.Pm,self.mstep)
            fdude.ancestcode=0.213+0.089*parent/N    # golden-brown
        for i,fdude in enumerate(self.bunchD):                    # select two parents favoring fitness, mix by uniform CO, mutate
            parenta=parentselect_exp(N,self.selpD)    # select a parent favoring fitness and mutate
            parentb=parentselect_exp(N,self.selpD)    # select a parent favoring fitness and mutate
            cigar_choice=rand()<self.cigar2uniform
            if cigar_choice:
                #fdude.cigar_CO(self.F0[parenta],self.F0[parentb],aspect=self.cigar_aspect,alpha=0.8,beta=0.3,scatter='randn')
                fdude.cigar_CO(self.F0[parenta],self.F0[parentb],aspect=self.cigar_aspect,alpha=0.3,beta=0.1,scatter='rand')
                fdude.ancestcode=0.495;              # turquise / cyan
            else:
                fdude.CO_from(self.F0[parenta],self.F0[parentb])
                fdude.ancestcode=0.395;              # almost white
            if self.mr: fdude.mutate(self.Pm,self.mr*self.mstep)
        for i,fdude in enumerate(self.bunchE):                    # select two parents favoring fitness, mix by WHX or BLX, mutate
            parenta=parentselect_exp(N,self.selpE)    # select a parent favoring fitness and mutate
            parentb=parentselect_exp(N,self.selpE)    # select a parent favoring fitness and mutate
            whx_choice=rand()<self.WHX2BLX
            if whx_choice:
                fdude.WHX(self.F0[parenta],self.F0[parentb]); fdude.ancestcode=0.82    # light blue
            else:
                fdude.BLX_alpha(self.F0[parenta],self.F0[parentb]); fdude.ancestcode=0.95  # light green
            if self.mr: fdude.mutate(self.Pm,self.mr*self.mstep)
        DEsri=self.DEsr[0]; DEsrw=(self.DEsr[1]-self.DEsr[0])
        for i,fdude in enumerate(self.bunchF):
            parenta=parentselect_exp(N,self.selpF)
            seq=rand(N).argsort(); seq=list(seq); parenta_index=seq.index(parenta); seq.pop(parenta_index) # exclude already selected parent
            parentb,parentc=seq[:2]  # these two parents chosen without selection pressure; all 3 parents are different
            m=m=DEsri+DEsrw*rand()  #0.2+0.6*rand() # scaling of the difference vector to be added to parenta's DNA
            fdude.set_DNA( self.F0[parenta].DNA  +  m * (self.F0[parentc].DNA-self.F0[parentb].DNA) )
            fdude.mirror_and_scatter_DNA_into_bounds()
            fdude.ancestcode=0.603+0.099*parenta/N             # yellow-green
            if self.mr: fdude.mutate(self.Pm,self.mr*self.mstep)

    def simple_run(self,generations,random_starting_DNAs=True):
        self.make_bunchlists()
        ComboBase.simple_run(self,generations,random_starting_DNAs)

    def simple_run_with_seed_DNAs(self,generations,seedDNAs):
        self.make_bunchlists()
        ComboBase.simple_run_with_seed_DNAs(self,generations,seedDNAs)

    def fractal_run(self,g1,g2,fractalfactor=4,msrf=0.3,return_inibest=False):
        self.mstep_reset()
        mstep_memo=self.mstep; self.mstep*=msrf # msrf stands for mstep reduction factor
        anneal_memo=self.anneal; self.anneal=0.
        self.make_bunchlists()
        self.storageA=population_like(self.F0,size=self.F0.psize)
        self.storageB=population_like(self.F0,size=self.F0.psize)
        if return_inibest==True: inibest,inimean=self.evolve_protopopulations(fractalfactor,g1,elitesize=len(self.bunchA),return_inibest=True)
        else: self.evolve_protopopulations(fractalfactor,g1,elitesize=len(self.bunchA))
        self.mstep=mstep_memo
        self.anneal=anneal_memo
        self.run(g2,status='normal')
        if return_inibest: return inibest,inimean
        else: return
        
#    def real_fractal_run(self,g1,g2=None,fractalfactor=4,layers=3):
#        self.set_bunchsizes(self.bunching(guideline='limit_elite'))
#        self.make_bunchlists()
#        self.mstep_reset()
#        self.storage=[]
#        for l in range(layers):
#            tmplist=[]
#            for frf in range(fractalfactor):
#                tmplist.append(population_like(self.F0,size=self.F0.psize))
#            self.storageA.append(tmplist)
#
#        ff=fractalfactor; L=layers; l=0; count=0; pcount=0
#        ready=zeros(L) # shows for each layer how many evolved populations are ready to use
#        
#        while ready[-1]==0:
#            count+=1; #print 'loop ',count
#        
#            if (ready[l] >=0) and (ready[l] < ff):
#                if l==0:
#                    for i in range(ff):
#                        if self.F0.gg!=0: self.F0.advance_generation()
#                        self.zeroth_generation(random_ini=True)
#                        self.run(g1,status='normal')
#                        self.storage[l][i].copy_otherpop(self.F0)
#                        ready[l]+=1; pcount+=1
#                    print '{} randomly initialised populations in layer {} with best scores {}'.format(ready[l],l,[p[0].score for p in self.storage[l]])
#                    l+=1
#                else:
#                    l=0
#            elif ready[l]==ff:
#                condense(self.storage[l],self.F0,selp=2.)
#                
#                
#                plist[l+1].append(np.sum(plist[l])); pcount+=1
#                print '          adding {} to p{}'.format(plist[l+1][-1],l+1)
#                plist[l]=[]
#                l+=1
#            else:
#                raise ValueError('p{} has wrong length {}'.format(l,ready[l]))

    def cec_run_01(self,g1,ff,maxeval=None):
        self.cigar2uniform=0.4
        self.anneal=0.03
        self.ini_mstep=0.08
        self.ini_smallstep=1.5e-3
        if maxeval is None: self.maxeval=maxeval
        else: self.maxeval=int(1e4)
        self.fractal_run(g1,1000000,fractalfactor=ff)

    def cec_poor_man_run(self):
        self.cigar2uniform=0.5
        self.anneal=0.09
        self.ini_mstep=0.18
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.fractal_run(12,56,fractalfactor=4,msrf=0.5)

    def run_for_3k(self):
        self.cigar2uniform=0.5
        self.anneal=0.06
        self.ini_mstep=0.18
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.fractal_run(17,150,fractalfactor=4,msrf=0.5)

    def cec_rich_man_run_A(self):
        self.cigar2uniform=0.5
        self.anneal=0.06
        self.ini_mstep=0.2
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.fractal_run(24,180,fractalfactor=4,msrf=0.5)

    def cec_rich_man_run_B(self):
        self.cigar2uniform=0.5
        self.anneal=0.07
        self.ini_mstep=0.18
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.fractal_run(16,100,fractalfactor=8,msrf=0.5)

    def cec_rich_man_run_C(self):
        self.cigar2uniform=0.5
        self.anneal=0.08
        self.ini_mstep=0.2
        self.ini_smallstep=0.0
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.fractal_run(16,100,fractalfactor=4,msrf=0.5)

    def generate_protopopulation(self,N):
        q=self.F0.psize/N
        for n in range(N):
            if n!=0 or self.F0.gg!=0: self.F0.advance_generation()
            self.zeroth_generation(random_ini=True)  # self status is set here automatically to 'random_dudes'
            for storedude,f0dude in zip(self.storageA[n*q:(n+1)*q],self.F0[:q]):  # store elite of random population
                storedude.copy_DNA_of(f0dude,copyscore=True,copyancestcode=False,copyparents=True)
                storedude.ancestcode=0.48+n*0.1; storedude.ancestcode-=floor(storedude.ancestcode) # ensures value is in [0,1]
        for f0dude,storedude in zip(self.F0,self.storageA):
            f0dude.copy_DNA_of(storedude,copyscore=True,copyancestcode=True,copyparents=True)
        self.F0.sort(); self.F0.update_no()
        self.F0.advance_generation()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)



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
            self.run(G,status='proto_populations')
            for i,dude in enumerate(self.storageB[n*q:(n+1)*q]):
                if i < elitesize: dude.copy_DNA_of(self.F0[i],copyscore=True,copyancestcode=False,copyparents=True)
                else: dude.copy_DNA_of(self.F0[parentselect_exp(self.F0.psize,selp)],copyscore=True,copyancestcode=False,copyparents=True)
                dude.ancestcode=0.48+n*0.1; dude.ancestcode-=floor(dude.ancestcode) # ensures value is in [0,1]
        for f0dude,storedude in zip(self.F0,self.storageB):
            f0dude.copy_DNA_of(storedude,copyscore=True,copyancestcode=True,copyparents=True)
        self.F0.sort(); self.F0.update_no()
        self.F0.advance_generation()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)    # so the merged population where no evaluation is necessary shows up in plots
        if return_inibest:
            if self.F0.whatisfit=='maximize': inibest=np.max(inibestscores)
            else: inibest=np.min(inibestscores)
            inimean=mean(inimeanscores)
            return inibest,inimean
        else:
            return

    def bunching(self,guideline='limit_elite',weights=None,elim=16):
        ps=self.ps
        if weights is not None:
            nweights=asfarray(weights)/np.sum(weights)
            bunches=np.floor(nweights*ps)
            bunches=[int(b) for b in bunches]
            ford=[3,5,2,1,4]; m=len(ford)  # fillup order
        elif guideline=='standard':
            bunches=6*[0]
            if ps<=100:
                bunches[0]=int(sqrt(floor(ps/4.)))
            else:
                nh=ps/100.
                b1a=int(sqrt(floor(100/4.)))
                b1b=b1a*nh
                bunches[0]=int(round(b1b))
            ford=[3,5,2,1]; m=len(ford)  # fillup order
            weights=[0,1,1,1,0,1]
        elif guideline=='limit_elite':
            bunches=6*[0]
            if ps<=100:
                bunches[0]=int(sqrt(floor(ps/4.)))
            else:
                nh=ps/100.
                b1a=int(sqrt(floor(100/4.)))
                b1b=b1a*nh
                bunches[0]=int(round(b1b))
            if bunches[0]>elim: bunches[0]=elim
            ford=[3,5,2,1]; m=len(ford)  # fillup order
            weights=[0,1,1,1,0,1]
        else:
            raise NotImplementedError('either a list with weights must be supplied or a known guideline')
        rest=ps-np.sum(bunches)
        i=0
        while rest>0:
            #print 'i = ',i
            j=ford[i%m]         #ford[m%i]
            if weights[j]==0:
                i+=1; #print 'not filled: ',j
                continue
            bunches[j]+=1; #print 'I filled up ',j,'   because i was ',i
            rest-=1
            i+=1
        return bunches
    
    def bunch_sorting_checkprint(self,atxt):
        print self.F1.ownname+': checking bunch sorting in generation ',self.F0.gg,'   ',atxt
        txt=''
        for b,blab in zip([self.bunchA,self.bunchB,self.bunchC,self.bunchD,self.bunchD,self.bunchE,self.bunchF],['A','B','C','D','E','F']):
            txt+='bunch{}: {}   '.format(blab,[dude.no for dude in b])
        print txt
        print 'numbers: ',[dude.no for dude in self.F1]


class ComboB_with_NM(ComboB,ComboBaseWNM):
    
    def __init__(self,F0pop,F1pop,simpop,trialpop):
        ComboB.__init__(self,F0pop,F1pop)
        ComboBaseWNM.__init__(self,F0pop,F1pop,simpop,trialpop)
        self.ownname='eacBwNM'

    def cec_poor_man_run_with_NM(self):
        self.maxeval=10000
        self.cec_poor_man_run()
        self.standard_NM_end(warnlim=20,pumpup=3)

    def cec_rich_man_run_with_NM(self,version,pumpup=3):
        self.maxeval=10000*self.F0.ng
        if version=='A':
            self.cec_rich_man_run_A()
        elif version=='B':
            self.cec_rich_man_run_B()
        elif version=='C':
            self.cec_rich_man_run_C()
        elif version=='3k':
            self.maxeval=30000
            self.run_for_3k()
        self.standard_NM_end(pumpup=pumpup)



class ComboB_DeLuxe(ComboB_with_NM, ComboBaseWLS, ComboBaseFractal):

    def __init__(self,F0pop,F1pop,simpop,trialpop,lstype='golden'):
        ComboB_with_NM.__init__(self,F0pop,F1pop,simpop,trialpop)  # I assume the rebinding of self.F0 and so on doesn't change a thing
        ComboBaseWLS.__init__(self,F0pop,F1pop,lstype)
        ComboBaseFractal.__init__(self,F0pop,F1pop)
        self.ownname='eacBdLux'
    
    def cec_fracrun_44A(self):
        # 4 fractal layers and branching factor is also 4
        assert self.F0.psize == 4*self.F0.ng
        self.optimal_fractal_dimensions(B=4,G=26,endrun_thresh=1.7,maxeval=int(1e4*self.F0.ng),NM=True)
        assert self.frc_L == 4
        self.cigar2uniform=0.5
        self.anneal=0.1
        self.ini_mstep=0.17
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.set_bunchsizes(self.bunching()); self.make_bunchlists()
        self.set_heatup_ratio(0.3)   
        self.real_fractal_run()
        self.standard_NM_end()

    def cec_fracrun_44A_part1(self):
        # 4 fractal layers and branching factor is also 4
        assert self.F0.psize == 4*self.F0.ng
        self.optimal_fractal_dimensions(B=4,G=26,endrun_thresh=1.7,maxeval=int(1e4*self.F0.ng),NM=True)
        assert self.frc_L == 4
        self.cigar2uniform=0.5
        self.anneal=0.1
        self.ini_mstep=0.17
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.set_bunchsizes(self.bunching()); self.make_bunchlists()
        self.set_heatup_ratio(0.3)
        storelist=self.real_fractal_run_only_tree()
        return storelist
    
    def combine_from(self,poplist):
        for j,p in enumerate(poplist):
            for i,dude in enumerate(p):
                self.F0[j*p.psize+i].copy_DNA_of(dude,copyscore=True); self.F0[j*p.psize+i].ancestcode=np.round(j%self.frc_B*(1./self.frc_B),3)
        self.do_fake_step(); self.mstep_memory=self.mstep; self.score_memory=self.F0[0].score
        self.run(self.frc_G2-1)
        self.sclist[self.frc_l+1].append(self.F0[0].score); self.frc_n+=1
        #print '{} recieved DNA from {} after {} generations and {} calls'.format('p'+str(self.frc_n).zfill(2), [p.ownname for p in self.storage[self.frc_l]], self.F0.gg, self.tell_neval())
        print '{} recieved DNA from {} after {} generations and {} calls'.format('p'+str(self.frc_n).zfill(2), [p.ownname for p in poplist], self.F0.gg, self.tell_neval())
        self.frac_mem_entry(True)
        self.ready[self.frc_l+1]+=1
        self.ready[self.frc_l]=0
        self.completed[self.frc_l]+=1
        self.sclist[self.frc_l]=[]


    def cec_fracrun_44A_part2(self,oea,storelist):
        """
        What is oea --> that's the other EA of same sort like this one but with different population size
        Why transfer stuff from other EA to this one with different population size instead of using ComboBase.change_psize() ???
        --> Because, if you use the CEC-2013 test functions via ctypes with my crappy code, and if you repeat size changes, then
        all the unused C-arrays will not be tidied up and cram the memory as you continue to reallocate arrays of different sizes.
        """
        self.fetch_some_important_stuff(oea)
        self.mstep*=self.heatup_factor
        self.anneal*=0.6
        self.combine_from(storelist)
        self.standard_NM_end(pumpup=1e3)
    
    def cec_fracrun_34A(self):
        # 4 fractal layers and branching factor is also 4
        assert self.F0.psize == 6*self.F0.ng
        self.optimal_fractal_dimensions(B=3,G=32,endrun_thresh=1.7,maxeval=int(1e4*self.F0.ng),NM=True)
        assert self.frc_L == 4
        self.cigar2uniform=0.5
        self.anneal=0.08
        self.ini_mstep=0.17
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.set_bunchsizes(self.bunching()); self.make_bunchlists()
        self.set_heatup_ratio(0.3)   
        self.real_fractal_run()
        self.standard_NM_end()

    def cec_fracrun_34A_part1(self):
        # 4 fractal layers and branching factor is also 4
        assert self.F0.psize == 6*self.F0.ng
        self.optimal_fractal_dimensions(B=3,G=32,endrun_thresh=1.7,maxeval=int(1e4*self.F0.ng),NM=True)
        assert self.frc_L == 4
        self.cigar2uniform=0.5
        self.anneal=0.08
        self.ini_mstep=0.2
        self.ini_smallstep=0.
        self.mr=0.
        self.DEsr=[0.0, 0.3]
        self.set_bunchsizes(self.bunching()); self.make_bunchlists()
        self.set_heatup_ratio(0.4)
        storelist=self.real_fractal_run_only_tree()
        return storelist
    
    def cec_fracrun_34A_part2(self,oea,storelist):
        """
        What is oea --> that's the other EA of same sort like this one but with different population size
        Why transfer stuff from other EA to this one with different population size instead of using ComboBase.change_psize() ???
        --> Because, if you use the CEC-2013 test functions via ctypes with my crappy code, and if you repeat size changes, then
        all the unused C-arrays will not be tidied up and cram the memory as you continue to reallocate arrays of different sizes.
        """
        self.fetch_some_important_stuff(oea)
        self.mstep*=self.heatup_factor
        self.anneal*=0.6
        self.combine_from(storelist)
        self.standard_NM_end(pumpup=1e3)
    
    def fetch_some_important_stuff(self,oea):
        """
        I made this as a workaround, the selection below is pretty random
        don't use it for another case without reviewing, better write a more thouroughly
        thought through function fetch_important_stuff()
        """
        # ComboBase attributes
        self.generation_callbacks=oea.generation_callbacks
        self.more_stop_crits=oea.more_stop_crits
        self.maxeval=oea.maxeval
        self.breaking_distance=oea.breaking_distance
        self.save_best=oea.save_best
        self.bestdude=deepcopy(oea.bestdude)
        self.F0.neval=oea.F0.neval
        self.F0.gg=oea.F0.gg
        # for mstep control:
        self.ini_mstep=oea.ini_mstep
        self.mstep=oea.mstep
        self.anneal=oea.anneal
        self.heatups=oea.heatups
        self.ini_smallstep=oea.ini_smallstep
        self.smallstep=oea.smallstep
        self.heatup_factor=oea.heatup_factor
        self.hutshrink=oea.hutshrink
        # ComboB attributes
        self.set_bunchsizes(self.bunching()); self.make_bunchlists()
        self.Pm=oea.Pm
        self.cigar2uniform=oea.cigar2uniform
        self.mr=oea.mr
        self.selpC=oea.selpC
        self.selpD=oea.selpD
        self.selpE=oea.selpE
        self.selpF=oea.selpF
        self.DEsr=oea.DEsr
        self.cigar_aspect=oea.cigar_aspect
        # ComboBaseFractal
        self.frc_B=oea.frc_B
        self.frc_L=oea.frc_L
        self.frc_l=oea.frc_l
        self.frc_n=oea.frc_n
        self.frc_G1=oea.frc_G1
        self.frc_G2=oea.frc_G2
        self.cpres=oea.cpres
        #self.celite=max(2,self.F0.psize/10)   # elite size for condensation
        self.sclist=oea.sclist
        self.completed=oea.completed
        self.ready=oea.ready
        self.storage=[]    # list holding an amount of B populations per layer
        self.frac_mem=oea.frac_mem
        self.score_memory=0.  # for remembering initially best score of a population
        self.mstep_memory=0.  # for remembering initial mstep of a population
        self.condensation_memory=[]  # remembering dude.no, dude.pa and dude.ancestcode occuring after condensation for making a histogram on all selection choices cumulating all condensation steps (purpose: verify selection as intended)
        self.big_final=True
        # ComboBaseWNM attribute
        self.sim.neval=oea.sim.neval
        self.trials.neval=oea.trials.neval
        self.sim.gg=oea.sim.gg
        self.trials.gg=oea.trials.gg

    def cec_run_02(self):
        self.set_bunchsizes([4,19,19,19,0,19])
        self.make_bunchlists()
        self.maxeval=4e4
        self.anneal=0.04
        self.ini_mstep=0.15
        self.ini_smallstep=0.002
        self.heatup_factor=40.
        self.mstep_reset()
        self.zeroth_generation(random_ini=True)
        while True:
            self.status='normal'
            for g in range(20):
                self.stationary_step()
                self.mstep*=exp(-self.anneal)
                if self.other_stopcrit_returned_True(): break
            self.search1D(16,inistep_factor=0.25,perturb=0.05)
            self.status='normal'
            while self.mstep>self.smallstep:
                self.stationary_step()
                self.mstep*=exp(-self.anneal)
                if self.other_stopcrit_returned_True(): break
            self.search1D(16,inistep_factor=0.25,perturb=0.05)
            self.status='normal'
            for g in range(20):
                self.stationary_step()
                self.mstep*=exp(-self.anneal)
                if self.other_stopcrit_returned_True(): break
            self.sim[0].copy_DNA_of(self.F0[0],copyscore=True)
            self.NMrun(xtol=1e-8, ftol=1e-8, inidelt=3*self.mstep, maxiter=self.F0.ng+50, reset_internal_state=True,startgg=self.F0.gg)
            self.F0.set_generation(self.sim.gg)
            self.mstep*=self.heatup_factor
            self.smallstep*=self.hutshrink
            self.heatups+=1
            if self.other_stopcrit_returned_True(): break


#------------------------------------------------------------------------------
#--- Combo with CMA-ES mutation distribution
#------------------------------------------------------------------------------

class ComboC_Base(ComboBase):
    """
    in a pythonic expression: from CMA-ES import mutation operator
    """

    def __init__(self,F0pop,F1pop):
        ComboBase.__init__(self,F0pop,F1pop)
        self.mstep=0.15
        self.mu=self.F0.psize/2    # the mu best mutation steps are used for the PCA (it's the mu in (mu,lambda)-ES)
        self.lbd=self.F0.psize     # lbd is the lambda in (mu,lambda)-ES
        self.boundary_treatment='mirror'
        self.weights = log(self.mu+0.5) - log(arange(1,self.mu+1)) # recombination weights
        self.weights /= np.sum(self.weights) # normalize recombination weights array
        self.mueff=np.sum(self.weights)**2 / np.sum(self.weights**2) # variance-effectiveness of sum w_i x_i
        # Initialize dynamic (internal) state variables and constants
        self.initialize_static_internals()
        self.initialize_dynamic_internals()
        self.boundary_treatment='mirror'
        #self.save_best=True
        #self.bestdude=None
        self.maxsigma=None
        self.c1a_adjustment=True
        self.useMVN=False
        self.mstep_rules='hansen'   # other possibilities: 'anneal', 'anneal with heatups', 'hansen with heatups'
        self.ini_neval=0
        self.meanD=1.
        self.ownname='eacCbase'
    
    def tell_cmaes_neval(self):
        return self.tell_neval()-self.ini_neval

    def initialize_static_internals(self,damps=None):
        self.ini_neval=self.tell_neval()
        # Strategy parameter setting: Adaptation
        self.cc = (4. + self.mueff/self.F0.ng) / (self.F0.ng+4. + 2. * self.mueff/self.F0.ng)  # time constant for cumulation for C
        self.cs = (self.mueff + 2) / (self.F0.ng + self.mueff + 5.)  # t-const for cumulation for sigma control
        self.c1 = 2. / ((self.F0.ng + 1.3)**2 + self.mueff)     # learning rate for rank-one update of C
        self.cmu = 2. * (self.mueff - 2. + 1./self.mueff) / ((self.F0.ng + 2.)**2 + self.mueff)  # and for rank-mu update
        if damps is None:
            self.damps = 2. * self.mueff/self.lbd + 0.3 + self.cs  # damping for sigma, usually close to 1
        else:
            self.damps = damps

    def initialize_dynamic_internals(self,reset_mstep=True):
        # Initialize dynamic (internal) state variables and constants
        if reset_mstep: self.mstep_reset()
        self.xmean=zeros(self.F0.ng)
        self.xold=zeros(self.F0.ng)
        self.pc = zeros(self.F0.ng)
        self.psig = zeros(self.F0.ng)  # evolution paths for C and sigma
        self.hsig = 0.0
        self.B = eye(self.F0.ng)   # B defines the coordinate system 
        self.D = ones(self.F0.ng)  # diagonal D defines the scaling
        self.C = eye(self.F0.ng)   # covariance matrix 
        self.invsqrtC = eye(self.F0.ng)  # C^-1/2
        self.F0.reset_all_mutagenes(1.) # same as D
        self.F1.reset_all_mutagenes(1.) # same as D
        self.meanD=1.
        self.goodDNA=zeros((self.mu,self.F0.ng))
        self.last_cm_update=self.tell_neval()

    def create_offspring_cmaes(self):
        if self.useMVN:
            #  mutates each individual of self.F0 according to the current multivariate normal distribution
            newDNAs=multivariate_normal(self.xmean,self.C,size=self.F0.psize)
            for i,dude in enumerate(self.F1):
                dude.set_uDNA(newDNAs[i,:])
                if self.boundary_treatment=='mirror': dude.mirror_DNA_into_bounds()
                elif self.boundary_treatment=='mirror_and_scatter': dude.mirror_and_scatter_DNA_into_bounds()
                elif self.boundary_treatment=='cycle': dude.cycle_DNA_into_bounds()
                elif self.boundary_treatment=='cycle_and_scatter': dude.cycle_and_scatter_DNA_into_bounds()
                elif self.boundary_treatment=='scatter': dude.randomize_DNA_into_bounds()
                elif self.boundary_treatment=='nobounds': pass
                elif self.boundary_treatment=='penalty': pass
                else: raise ValueError('invalid flag for boundary treatment')
                dude.pa=0; dude.pb=-1; dude.ancestcode=0.85
        else:
            for dude in self.F1:
                dude.set_uDNA(self.xmean + self.mstep*dot(self.B,dude.mutagenes*randn(self.F0.ng)))
                if self.boundary_treatment=='mirror': dude.mirror_DNA_into_bounds()
                elif self.boundary_treatment=='mirror_and_scatter': dude.mirror_and_scatter_DNA_into_bounds()
                elif self.boundary_treatment=='cycle': dude.cycle_DNA_into_bounds()
                elif self.boundary_treatment=='cycle_and_scatter': dude.cycle_and_scatter_DNA_into_bounds()
                elif self.boundary_treatment=='scatter': dude.randomize_DNA_into_bounds()
                elif self.boundary_treatment=='nobounds': pass
                elif self.boundary_treatment=='penalty': pass
                else: raise ValueError('invalid flag for boundary treatment')
                dude.pa=0; dude.pb=-1; dude.ancestcode=0.85
    
    def turn_dude_into_cmaes_offspring(self,dude):
        #if dude.no==0: print 'this xmean: ',self.xmean
        dude.set_uDNA(self.xmean + self.mstep*dot(self.B,dude.mutagenes*randn(dude.ng)))
        if self.boundary_treatment=='mirror': dude.mirror_DNA_into_bounds()
        elif self.boundary_treatment=='mirror_and_scatter': dude.mirror_and_scatter_DNA_into_bounds()
        elif self.boundary_treatment=='cycle': dude.cycle_DNA_into_bounds()
        elif self.boundary_treatment=='cycle_and_scatter': dude.cycle_and_scatter_DNA_into_bounds()
        elif self.boundary_treatment=='scatter': dude.randomize_DNA_into_bounds()
        elif self.boundary_treatment=='nobounds': pass
        elif self.boundary_treatment=='penalty': pass
        else: raise ValueError('invalid flag for boundary treatment')
        dude.pa=0; dude.pb=-1; dude.ancestcode=0.85

    def stateupdate(self):
        self.update_evolution_paths()
        self.cma()
        self.mstep_control_cmaes()
            
    def do_cmaes_step(self,stateupdate=True):
        #print 'mstep: ',self.mstep
        for dude in self.F1:
            self.turn_dude_into_cmaes_offspring(dude)
        self.advance_generation()
        if self.boundary_treatment=='penalty':
            for dude in self.F0:
                if dude.violates_boundaries(): dude.set_bad_score()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if stateupdate:
            self.stateupdate()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)
    
    def step_cmaes_do(self):
        self.stateupdate()
        for dude in self.F1:
            self.turn_dude_into_cmaes_offspring(dude)
        self.advance_generation()
        if self.boundary_treatment=='penalty':
            for dude in self.F0:
                if dude.violates_boundaries(): dude.set_bad_score()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)
    
#    def update_bestdude(self):
#        if self.bestdude is None:
#            self.bestdude=deepcopy(self.F0[0])
#        else:
#            if self.F0[0].isbetter(self.bestdude):
#                self.bestdude.copy_DNA_of(self.F0[0],copyscore=True,copyparents=True,copyancestcode=True,copymutagenes=True)
#                self.bestdude.gg=self.F0[0].gg

    def update_evolution_paths(self):
        self.xold[:] = self.xmean  #copy(self.xmean)
        for i,dude in enumerate(self.F0[:self.mu]):
            self.goodDNA[i,:]=dude.get_uDNA()
        self.xmean[:] = dot(self.goodDNA.T,self.weights)  # recombination, new mean value
        #print 'mu: ',self.mu
        #print 'goodDNA: ',self.goodDNA[:4,:4]
        #print 'new xold: ',self.xold
        #print 'new xmean: ',self.xmean
        # Cumulation: Update evolution paths 
        self.psig[:] = (1.-self.cs)*self.psig + ((self.cs*(2.-self.cs)*self.mueff)**0.5/self.mstep)*dot(self.invsqrtC,(self.xmean-self.xold))  # see slide 86 in Hansen's PPSN 2006 CMA Tutorial Talk
        self.hsig = np.sum(self.psig**2)/(1.-(1.-self.cs)**(2.*self.tell_neval()/float(self.lbd)))/self.F0.ng < 2. + 4./(self.F0.ng+1.)
        self.pc[:] = (1.-self.cc)*self.pc + ((self.cc*(2.-self.cc)*self.mueff)**0.5/self.mstep)*self.hsig*(self.xmean-self.xold)
        #print 'generation {}: psig[:3] = {} and pc[:3] = {}'.format(self.F0.gg,self.psig[:3],self.pc[:3])
        

    def cma(self):
        # Adapt covariance matrix C
        Z = (self.goodDNA-self.xold) / self.mstep
        Z = dot((self.cmu * self.weights) * Z.T, Z)  # learning rate integrated
        if self.c1a_adjustment==True:
            c1a = self.c1 - (1.-self.hsig**2) * self.c1 * self.cc * (2.-self.cc)  # minor adjustment for variance loss by hsig
            self.C = (1. - c1a - self.cmu) * self.C + outer(self.c1 * self.pc, self.pc) + Z
        else:
            self.C = (1. - self.c1 - self.cmu) * self.C + outer(self.c1 * self.pc, self.pc) + Z        
        if self.tell_neval() - self.last_cm_update > self.lbd/(self.c1+self.cmu)/float(self.F0.ng)/10.:  # to achieve O(N^2)
            self.Cdecomp()
    
    def Cdecomp(self):
        self.D,self.B = eigh(self.C)              # eigen decomposition, B==normalized eigenvectors
        self.D = self.D**0.5   # D contains standard deviations now (being a 1D array)
        Darr = diag(1./self.D)  # is a 2D array
        self.invsqrtC = dot(self.B,dot(Darr,self.B.T))    # proof: print dot(invsqrtC.T,invsqrtC)-inv(C)        
        self.last_cm_update = self.tell_neval()
        self.F0.reset_all_mutagenes(self.D)
        self.F1.reset_all_mutagenes(self.D)
        self.meanD=mean(self.D)

    def mstep_control_cmaes(self):
        if 'hansen' in self.mstep_rules:
            # Adapt step size sigma with factor <= exp(1/2) \approx 1.65
            self.mstep *= exp(np.min((0.5, (self.cs/self.damps) * (np.sum(self.psig**2) / self.F0.ng - 1.) / 2.),axis=0))  # this is the alternative - see p 18-19 of his tutorial article
            if self.maxsigma is not None: self.mstep=min(self.mstep,self.maxsigma)
        if 'anneal' in self.mstep_rules:
            self.mstep*=exp(-self.anneal)
        if 'heatups' in self.mstep_rules:
            if self.mstep<=self.smallstep:
                self.mstep*=self.heatup_factor
                self.smallstep*=self.hutshrink
                self.heatups+=1
    
    def zeroth_generation(self,random_ini=True):
        self.status='random_dudes'
        self.F0.new_random_genes()
        self.F0.zeroth_generation()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def reset_CMA_state(self,reset_mstep=False,damps=None):
        self.F0.reset_all_mutagenes(1.)
        self.F1.reset_all_mutagenes(1.)
        self.meanD=mean(1)
        self.initialize_static_internals(damps=damps)   # able to account for changed self.lbd
        self.initialize_dynamic_internals(reset_mstep=reset_mstep)

    def run_pure_cmaes(self,generations,reset_CMA_state=True,reset_population=True,
                       xstart='best',reset_mstep=True,reset_bestdude=True,damps=None):
        # if xstart can be a DNA vector/list
        # if xstart=='continue': starting point based on weighted DNAs of the best mu individuals
        # if xstart=='best': start with the DNA of self.F0[0]
        # else: start with mean of all current DNAs, i.e. the current center of the cloud
        if reset_bestdude:
            self.bestdude=None
        if reset_population:
            self.zeroth_generation()
        if reset_CMA_state:
            self.reset_CMA_state(reset_mstep=False,damps=damps)
        #print 'damps after resetting CMA state: ',self.damps
        if damps is not None:
            self.damps=damps  # needed in case CMA state reset has not been called
        if reset_mstep: self.mstep_reset()
        if xstart=='best':
            self.xmean[:]=self.F0[0].get_uDNA()
        elif xstart=='best_mu':
            self.xmean[:]=mean(self.F0.get_uDNAs()[:self.mu,:],axis=0)
        elif type(xstart) in [np.ndarray,list,tuple]:
            [dude.set_DNA(xstart) for dude in self.F0]; self.xmean=self.F0[0].get_uDNA()
        elif xstart=='continue':
            pass
        else:  # including xstart=='mean'
            self.xmean[:]=self.F0.mean_DNA(uCS=True); [dude.set_uDNA(self.xmean) for dude in self.F0]
        #print 'starting with best DNA: ',self.xmean
        #print 'damps=',self.damps
        self.status='pure_cmaes'
        for gg in range(generations):
            self.do_cmaes_step()
            if self.other_stopcrit_returned_True():
                break
        return

#    def tell_best_score(self,andDNA=False):
#        if (self.save_best is not None) and (self.bestdude.isbetter(self.F0[0])):
#            if andDNA:
#                return self.bestdude.score,self.bestdude.get_copy_of_DNA()
#            else:
#                return self.bestdude.score
#        else:
#            if andDNA:
#                return self.F0[0].score,self.F0[0].get_copy_of_DNA()
#            else:
#                return self.F0[0].score


class ComboC(ComboB, ComboC_Base):

    def __init__(self,F0pop,F1pop):
        ComboB.__init__(self,F0pop,F1pop)
        ComboC_Base.__init__(self,F0pop,F1pop)
        self.bunchC_mode='intermediate'
        self.mstep_rules='anneal with heatups'   #'hansen'   # other possibilities: 'anneal', 'anneal with heatups', 'hansen with heatups'
        self.use_meanD=True
        self.ownname='eacC'

    def cmaes_mutation(self,dude,fade=1.):
        dude.set_uDNA(dude.get_uDNA() + fade*self.mstep*dot(self.B,dude.mutagenes*randn(dude.ng)))
        if self.boundary_treatment=='mirror': dude.mirror_DNA_into_bounds()
        elif self.boundary_treatment=='mirror_and_scatter': dude.mirror_and_scatter_DNA_into_bounds()
        elif self.boundary_treatment=='cycle': dude.cycle_DNA_into_bounds()
        elif self.boundary_treatment=='cycle_and_scatter': dude.cycle_and_scatter_DNA_into_bounds()
        elif self.boundary_treatment=='scatter': dude.randomize_DNA_into_bounds()
        elif self.boundary_treatment=='nobounds': pass
        elif self.boundary_treatment=='penalty': pass
        else: raise ValueError('invalid flag for boundary treatment')

    def create_offspring(self):
        # Todo: try: reverse mutation when it worsened the individual
        self.F1.advance_generation()
        N=self.F0.psize
        if self.use_meanD:
            mdcompens=1./self.meanD
        else:
            mdcompens=1.
        for i,fdude in enumerate(self.bunchA):
            fdude.copy_DNA_of(self.F0[fdude.no])
            #fdude.mutate(0.5,self.eml*self.mstep*float(i)/float(len(self.bunchA)))  # mutate the best few just a little bit and the best one not at all
            self.cmaes_mutation(fdude,fade=self.eml*float(i)/float(len(self.bunchA))*mdcompens)
            fdude.ancestcode=0.09*i/len(self.bunchA)  # black-read
        for i,fdude in enumerate(self.bunchB):
            fdude.copy_DNA_of(self.F0[fdude.no])
            #fdude.mutate(self.Pm,self.mstep)          # just mutation
            self.cmaes_mutation(fdude,fade=1.*mdcompens)
            fdude.ancestcode=0.103+0.099*i/len(self.bunchB)  # blue-violet
        for i,fdude in enumerate(self.bunchC):
            if self.bunchC_mode=='oldschool':
                parent=parentselect_exp(N,self.selpC)     # select a parent favoring fitness and mutate
                fdude.copy_DNA_of(self.F0[parent],copyscore=False,copyancestcode=False,copyparents=False)
                fdude.mutate(self.Pm,self.mstep)
                fdude.ancestcode=0.213+0.089*parent/N    # golden-brown
            elif self.bunchC_mode=='intermediate':
                parent=parentselect_exp(N,self.selpC)     # select a parent favoring fitness and mutate
                fdude.copy_DNA_of(self.F0[parent],copyscore=False,copyancestcode=False,copyparents=False)
                self.cmaes_mutation(fdude,fade=1.*mdcompens)
                fdude.ancestcode=0.213+0.089*parent/N    # golden-brown
            elif self.bunchC_mode=='cmaes':
                self.turn_dude_into_cmaes_offspring(fdude)
                fdude.ancestcode=0.205  #+0.089*parent/N    # golden-brown
            else:
                raise ValueError('invalid mode for bunch C: '+self.bunchC_mode)
        for i,fdude in enumerate(self.bunchD):                    # select two parents favoring fitness, mix by uniform CO, mutate
            parenta=parentselect_exp(N,self.selpD)    # select a parent favoring fitness and mutate
            parentb=parentselect_exp(N,self.selpD)    # select a parent favoring fitness and mutate
            cigar_choice=rand()<self.cigar2uniform
            if cigar_choice:
                #fdude.cigar_CO(self.F0[parenta],self.F0[parentb],aspect=self.cigar_aspect,alpha=0.8,beta=0.3,scatter='randn')
                fdude.cigar_CO(self.F0[parenta],self.F0[parentb],aspect=self.cigar_aspect,alpha=0.3,beta=0.1,scatter='rand')
                fdude.ancestcode=0.495;              # turquise / cyan
            else:
                fdude.CO_from(self.F0[parenta],self.F0[parentb])
                fdude.ancestcode=0.395;              # almost white
            if self.mr: self.cmaes_mutation(fdude,fade=self.mr*mdcompens)
        for i,fdude in enumerate(self.bunchE):                    # select two parents favoring fitness, mix by WHX or BLX, mutate
            parenta=parentselect_exp(N,self.selpE)    # select a parent favoring fitness and mutate
            parentb=parentselect_exp(N,self.selpE)    # select a parent favoring fitness and mutate
            whx_choice=rand()<self.WHX2BLX
            if whx_choice:
                fdude.WHX(self.F0[parenta],self.F0[parentb]); fdude.ancestcode=0.82    # light blue
            else:
                fdude.BLX_alpha(self.F0[parenta],self.F0[parentb]); fdude.ancestcode=0.95  # light green
            if self.mr: self.cmaes_mutation(fdude,fade=self.mr*mdcompens)
        DEsri=self.DEsr[0]; DEsrw=(self.DEsr[1]-self.DEsr[0])
        for i,fdude in enumerate(self.bunchF):
            parenta=parentselect_exp(N,self.selpF)
            seq=rand(N).argsort(); seq=list(seq); parenta_index=seq.index(parenta); seq.pop(parenta_index) # exclude already selected parent
            parentb,parentc=seq[:2]  # these two parents chosen without selection pressure; all 3 parents are different
            m=m=DEsri+DEsrw*rand()  #0.2+0.6*rand() # scaling of the difference vector to be added to parenta's DNA
            fdude.set_DNA( self.F0[parenta].DNA  +  m * (self.F0[parentc].DNA-self.F0[parentb].DNA) )
            fdude.mirror_and_scatter_DNA_into_bounds()
            fdude.ancestcode=0.603+0.099*parenta/N             # yellow-green
            if self.mr: self.cmaes_mutation(fdude,fade=self.mr*mdcompens)
    
    def do_step(self,stateupdate=True):
        self.create_offspring()
        self.advance_generation()
        if self.boundary_treatment=='penalty':
            for dude in self.F0:
                if dude.violates_boundaries(): dude.set_bad_score()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if stateupdate:
            self.stateupdate()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def step_do(self):
        self.stateupdate()
        self.create_offspring()
        self.advance_generation()
        if self.boundary_treatment=='penalty':
            for dude in self.F0:
                if dude.violates_boundaries(): dude.set_bad_score()
        self.F0.mark_oldno()
        self.F0.sort()
        self.F0.update_no()
        if self.save_best:
            self.update_bestdude()
        for gc in self.generation_callbacks:
            gc(self)

    def ini_and_run(self,generations,reset_CMA_state=True,xstart='best_mu',reset_mstep=True,reset_bestdude=False):
        self.make_bunchlists()
        if reset_bestdude:
            self.bestdude=None
        if reset_CMA_state:
            self.reset_CMA_state(reset_mstep=False)
        if reset_mstep: self.mstep_reset()
        if xstart=='best':
            self.xmean[:]=self.F0[0].get_uDNA()
        elif xstart=='best_mu':
            self.xmean[:]=mean(self.F0.get_uDNAs()[:,self.mu,:],axis=0)
        elif type(xstart) in [np.ndarray,list,tuple]:
            self.xmean[:]=xstart
        elif xstart=='continue':
            pass
        else:  # including xstart=='mean'
            self.xmean[:]=self.F0.mean_DNA(uCS=True); [dude.set_uDNA(self.xmean) for dude in self.F0]
        self.run(generations,status='normal_cmaes')

    def random_ini_and_run(self,generations,reset_CMA_state=True,xstart='best_mu',reset_mstep=True,reset_bestdude=False):
        if reset_bestdude:
            self.bestdude=None
        self.zeroth_generation(random_ini=True)
        self.ini_and_run(generations, reset_CMA_state=reset_CMA_state, xstart=xstart, reset_mstep=reset_mstep, reset_bestdude=False)


class ComboC_with_NM(ComboB_with_NM, ComboC):

    def __init__(self,F0pop,F1pop,simpop,trialpop):
        ComboB_with_NM.__init__(self,F0pop,F1pop,simpop,trialpop)
        ComboC.__init__(self,F0pop,F1pop)
        self.ownname='eacCwNM'

    def tell_neval(self):
        return self.F0.neval+self.sim.neval+self.trials.neval

    def tell_best_score(self,andDNA=False):
        if self.status=='Nelder_Mead':
            if andDNA:
                return self.sim[0].score,self.sim[0].get_copy_of_DNA()
            else:
                return self.sim[0].score
        else:
            if (self.save_best is not None) and (self.bestdude.isbetter(self.F0[0])):
                if andDNA:
                    return self.bestdude.score,self.bestdude.get_copy_of_DNA()
                else:
                    return self.bestdude.score
            else:
                if andDNA:
                    return self.F0[0].score,self.F0[0].get_copy_of_DNA()
                else:
                    return self.F0[0].score

    def run_for_10k(self):
        self.maxeval=10000
        self.ini_mstep=0.3
        self.mstep_rules='hansen'
        self.run_pure_cmaes(119,xstart='best_mu',damps=None)
        self.sim[0].copy_DNA_of(self.F0[0],copyscore=True)
        remaining_trials=self.maxeval-self.tell_neval()
        print 'remaining trials: {} - {} = {}'.format(self.maxeval,self.tell_neval(),remaining_trials)
        if remaining_trials < 20:
            raise ValueError('remaining trials left: '+str(remaining_trials))
        self.NMrun(xtol=1e-16, ftol=1e-16, inidelt=3*self.mstep*np.max(self.meanD), maxiter=None, maxfun=remaining_trials, reset_internal_state=True,startgg=self.F0.gg)


    def cmaes_run_01(self):
        self.maxeval=10000*self.F0.ng
        self.breaking_distance=100*self.F0.ng
        self.ini_mstep=0.1
        self.mstep_rules='hansen'
        self.bestdude=None
        self.zeroth_generation()
        self.xmean[:]=mean(self.F0.get_uDNAs()[:self.mu,:],axis=0)
        self.status='pure_cmaes'
        self.reset_CMA_state(reset_mstep=True)
        NM_flag=True
        while True:
            self.do_cmaes_step()
            if self.maxeval-self.tell_neval() <= self.breaking_distance:
                break
            if fabs(self.F0[0].score+1000.)<1e-8:
                NM_flag=False
                break
        if NM_flag:
            self.sim[0].copy_DNA_of(self.F0[0],copyscore=True)
            remaining_trials=self.maxeval-self.tell_neval()
            print 'remaining trials: {} - {} = {}'.format(self.maxeval,self.tell_neval(),remaining_trials)
            if remaining_trials < 20:
                raise ValueError('remaining trials left: '+str(remaining_trials))
            self.NMrun(xtol=1e-16, ftol=1e-16, inidelt=3*self.mstep*np.max(self.meanD), maxiter=None, maxfun=remaining_trials, reset_internal_state=True,startgg=self.F0.gg)

    def cmaes_with_periodic_restarts(self,G):
        self.maxeval=10000*self.F0.ng
        self.breaking_distance=100*self.F0.ng
        self.ini_mstep=0.3
        self.mstep_rules='hansen'
        self.bestdude=None
        while self.tell_neval() < self.maxeval-self.breaking_distance:
            if self.F0.gg==0:
                self.zeroth_generation()
            else:
                self.do_random_step()
            self.xmean[:]=mean(self.F0.get_uDNAs()[:self.mu,:],axis=0)
            self.reset_CMA_state(reset_mstep=True)
            self.status='pure_cmaes'
            if self.maxeval-self.tell_neval() <= 3*self.ps*G+self.breaking_distance:
                G*=2
            for gg in range(G):
                self.do_cmaes_step()
                if self.maxeval-self.tell_neval() <= self.breaking_distance:
                    break
        self.sim[0].copy_DNA_of(self.F0[0],copyscore=True)
        remaining_trials=self.maxeval-self.tell_neval()
        print 'remaining trials: {} - {} = {}'.format(self.maxeval,self.tell_neval(),remaining_trials)
        if remaining_trials < 20:
            raise ValueError('remaining trials left: '+str(remaining_trials))
        self.NMrun(xtol=1e-16, ftol=1e-16, inidelt=3*self.mstep*np.max(self.meanD), maxiter=None, maxfun=remaining_trials, reset_internal_state=True,startgg=self.F0.gg)


class ComboD(object):
    
    def __init__(self,populations):
        self.populations=populations
        self.F0b, self.F1b, self.F0c, self.F1c, self.sim, self.trials = populations
        self.EAb=ComboB_with_NM(self.F0b,self.F1b,self.sim,self.trials)
        self.EAc=ComboC_Base(self.F0c,self.F1c)
        self.EAb.more_stop_crits=[]
        self.EAc.more_stop_crits=[]
        self.more_stop_crits=[stopper]
        self.maxeval=0
        self.mstep_dominance='cmaes'
        self.mstep_independence=False
        self.strong_coupling=10  # amount of DNAs to transfer
        self.weak_coupling=5  # amount of DNAs to transfer
        self.selp=3.  # selection pressure used when selecting DNAs for swapping
        self.status='ini'
        self.active_algos=[]
        self.F0b.ownname='population_EAbF0'
        self.F1b.ownname='population_EAbF1'
        self.F0c.ownname='population_EAcF0'
        self.F1c.ownname='population_EAcF1'
        self.sim.ownname='population_sim'
        self.trials.ownname='population_trials'
        self.generation_callbacks=[]
        self.ownname='eacD'

    def tell_neval(self):
        return np.sum([p.neval for p in self.populations])
    
    def zeroth_generation(self,random_ini=True):
        self.active_algos=['EAb','EAc']
        if random_ini:
            self.status='random_dudes'
            self.EAb.status='random_dudes'
            self.EAc.status='random_dudes'
            self.EAb.F0.new_random_genes()
            self.EAc.F0.new_random_genes()
        self.EAb.F0.zeroth_generation()
        self.EAc.F0.zeroth_generation()
        if self.EAc.save_best:
            self.EAc.update_bestdude()
        for gc in self.EAb.generation_callbacks:
            gc(self)
        for gc in self.EAc.generation_callbacks:
            gc(self)
        for gc in self.generation_callbacks:
            gc(self)

    def other_stopcrit_returned_True(self):
        if len(self.more_stop_crits) != 0:
            morestopvals=[]
            for crit in self.more_stop_crits:
                morestopvals.append(crit(self))
            if True in morestopvals:
                #print "algorithm's run() terminated because of '+str(morestopvals.index(True))+'th additional stopping criterion"
                return True
            else:
                return False
        else:
            return False
    
    def update_population_sorting(self):
        self.EAb.F0.sort()
        self.EAc.F0.sort()
        self.EAb.F0.update_no()
        self.EAc.F0.update_no()
    
    def give_cmaes_eff_mstep(self):
        return self.EAc.mstep*self.EAc.meanD
        
    def do_step(self,cma_stateupdate=True):
        if self.weak_coupling: self.bidir_DNA_transfer(self.weak_coupling)
        assure_sortedness(self.EAb.F0)
        assure_sortedness(self.EAc.F0)
        self.EAb.do_step()
        if cma_stateupdate:
            self.EAc.step_cmaes_do()
        else:
            self.EAc.do_cmaes_step(stateupdate=False)
        if not self.mstep_independence:
            if self.mstep_dominance=='cmaes':
                self.EAb.mstep=self.EAc.mstep*self.EAc.meanD   #np.max(self.EAc.F0[0].mutagenes)
            elif self.mstep_dominance=='combob':
                self.reset_cma_state_if_unreasonable()
                self.EAc.mstep=self.EAb.mstep/self.EAc.meanD   #np.max(self.EAc.F0[0].mutagenes)
            else:
                raise ValueError('invalid mstep dominance flag (cmaes or combob): '+self.mstep_dominance)
        #self.EAb.bunch_sorting_checkprint('after ComboD step')
        for gc in self.generation_callbacks:
            gc(self)
    
    def reset_cma_state_if_unreasonable(self):
        if (self.EAc.meanD<1e-32) or (self.EAc.meanD>1e+32):
            print 'in generation {} the CMA state was reset due to meanD = {}'.format(self.F0c.gg,self.EAc.meanD)
            for bdude,cdude in zip(self.F0b,self.F0c):
                cdude.copy_DNA_of(bdude,copyscore=True)
            self.EAc.initialize_dynamic_internals()
            self.EAc.xmean[:]=mean(self.F0c.get_uDNAs(),axis=0)
            self.EAc.mstep=self.EAb.mstep
            #self.EAc.update_evolution_paths()
            #self.EAc.cma()
                
    def run(self,generations,status=['normal','normal','normal_cmaes']):
        self.status=status[0]; self.EAb.status=status[1]; self.EAc.status=status[2]
        self.active_algos=['EAb','EAc']
        for i in range(generations):
            self.do_step()
            #dna1=self.F0b[0].DNA
            #dna2=self.F0c[0].DNA
            #distance=sqrt(np.sum((dna2-dna1)**2))
            #print 'generation {} / {} completed; distance of the best is {}'.format(self.EAb.F0.gg,self.EAc.F0.gg,distance)
            if self.other_stopcrit_returned_True():
                break

    def ini_and_run(self,generations,eac_xstart='best_mu'):
        self.EAb.make_bunchlists(); self.EAb.mstep_reset()
        self.EAc.bestdude=None
        self.EAc.reset_CMA_state(reset_mstep=True)
        if eac_xstart=='best':
            self.EAc.xmean[:]=self.EAc.F0[0].get_uDNA()
        elif eac_xstart=='best_mu':
            self.EAc.xmean[:]=mean(self.EAc.F0.get_uDNAs()[:self.EAc.mu,:],axis=0)
        elif type(eac_xstart) in [np.ndarray,list,tuple]:
            self.EAc.xmean[:]=eac_xstart
        elif eac_xstart=='continue':
            pass
        else:  # including xstart=='mean'
            self.EAc.xmean[:]=self.EAc.F0.mean_DNA(uCS=True); [dude.set_uDNA(self.EAc.xmean) for dude in self.EAc.F0]
        #print 'starting with best DNA: ',self.EAc.xmean
        #print 'mstep: ',self.EAc.mstep
        self.status='normal'; self.EAb.status='normal'; self.EAc.status='normal_cmaes'
        self.active_algos=['EAb','EAc']
        self.do_step(cma_stateupdate=False)
        self.run(generations-1)

    def random_ini_and_run(self,generations,eac_xstart='best_mu'):
        self.EAc.bestdude=None
        self.EAb.mstep_reset(); self.EAc.mstep_reset()
        #self.EAb.bunch_sorting_checkprint('before zeroth generation')
        self.zeroth_generation(random_ini=True)
        #self.EAb.bunch_sorting_checkprint('after zeroth generation')
        self.ini_and_run(generations, eac_xstart=eac_xstart)

    def cec_straight_run(self,generations):
        self.maxeval=1e4*self.F0b.ng
        self.EAb.ini_mstep=0.3; self.EAc.ini_mstep=0.3
        self.mstep_dominance='combob'
        self.EAb.anneal=0.05
        self.strong_coupling=self.F0b.psize/6  # amount of DNAs to transfer
        self.weak_coupling=self.F0b.psize/12  # amount of DNAs to transfer
        self.selp=1.4  # selection pressure used when selecting DNAs for swapping
        self.EAb.set_bunchsizes(self.EAb.bunching(guideline='limit_elite'))
        self.mstep_independence=False
        self.random_ini_and_run(generations=generations)

    def cec_run_01(self):
        self.cec_straight_run(309)
        self.sim[0].copy_DNA_of(self.F0b[0],copyscore=True)
        remaining_trials=self.maxeval-self.tell_neval()
        print 'remaining trials: {} - {} = {}'.format(self.maxeval,self.tell_neval(),remaining_trials)
        if remaining_trials < 500:
            raise ValueError('remaining trials left: '+str(remaining_trials))
        self.EAb.NMrun(xtol=1e-16, ftol=1e-16, inidelt=3*self.EAb.mstep, maxiter=None,
                       maxfun=remaining_trials, reset_internal_state=True,startgg=self.F0b.gg)


    def cec_run_frac_nocom(self,generations):
        # phase 1: independent evolution of the two EAs (proto-populations in EAb, meanwhile normal process in EAc)
        self.EAb.ini_mstep=0.18; self.EAc.ini_mstep=0.18; ff=4; G1=12; G2=10
        self.EAb.set_bunchsizes(self.EAb.bunching(guideline='limit_elite'))
        self.mstep_independence=True
        self.active_algos=['EAb']; self.status='EAb_fractal_run'
        self.EAb.fractal_run(G1,G2,fractalfactor=ff,msrf=0.5) # proto-populations evolve untouched
        G=self.EAb.F0.gg
        self.active_algos=['EAc']; self.status='EAc_random_dudes'
        self.EAc.zeroth_generation(random_ini=True)
        self.EAc.reset_CMA_state(reset_mstep=True)
        self.EAc.xmean[:]=mean(self.EAc.F0.get_uDNAs()[:self.EAc.mu,:],axis=0)
        self.status='EAc_random_dudes'; self.EAc.status='normal_cmaes'
        for g in range(G-1):
            self.EAc.do_cmaes_step(stateupdate=True)  # CMA-ES evolves untouched
        self.EAc.do_cmaes_step(stateupdate=False)
        # phase 2: after proto-populations: one EA needs to catch up so both msteps will have shrunken to a similar magnitude
        if self.give_cmaes_eff_mstep() < 0.9*self.EAb.mstep:
            self.unidir_DNA_transfer('CtoB',self.strong_coupling)
            while self.EAb.mstep > self.give_cmaes_eff_mstep():
                self.EAb.do_step()
        elif self.EAb.mstep < 0.9*self.give_cmaes_eff_mstep():
            self.unidir_DNA_transfer('BtoC',self.strong_coupling)
            self.EAc.stateupdate()
            while self.give_cmaes_eff_mstep() > self.EAb.mstep:
                self.EAc.step_cmaes_do()
        print 'msteps before alignment: ',self.EAb.mstep,self.EAb.mstep*self.EAc.meanD,self.EAc.mstep
        self.EAb.mstep=max(self.EAb.mstep*self.EAc.meanD,self.EAc.mstep)
        self.EAc.mstep=self.EAb.mstep/self.EAc.meanD
        print 'msteps after alignment: ',self.EAb.mstep,self.EAb.mstep*self.EAc.meanD,self.EAc.mstep
        # phase 3: now that mstep are equalized, start co-evolution
        self.mstep_independence=False
        self.mstep_dominance='cmaes'
        while True:
            self.do_step()
            if self.other_stopcrit_returned_True():
                break
            
    def cec_run_frac(self,generations):
        self.EAb.ini_mstep=0.18; self.EAc.ini_mstep=0.18; ff=4; G1=12; G2=10
        # phase 1: proto-populations (DNA flows only from EAb to EAc)
        q=self.ps/ff
        self.zeroth_generation(random_ini=True)
        self.mstep_independence=True
        self.EAb.fractal_run(G1,G2,fractalfactor=ff,msrf=0.5) # proto-populations evolve untouched
        self.EAc.F0.reset_all_mutagenes(1.)
        self.EAc.F1.reset_all_mutagenes(1.)
        self.EAc.meanD=1.
        self.EAc.initialize_static_internals()   # able to account for changed self.lbd
        self.EAc.initialize_dynamic_internals(reset_mstep=True)
        for n in range(ff): # in this loop CMA-ES periodically gets infusions from best of EAb's proto-populations
            nums=ladivsel(self.EAb.storageB[n*q:(n+1)*q],self.strong_coupling,mindist=0.08,uCS=True,selp=self.selp)
            for i,dude in enumerate(self.EAc.F0[-self.strong_coupling:]):
                dude.copy_DNA_of(self.EAb.storageB[nums[i]],copyscore=True,copymutagenes=False)
            if n==0:
                self.EAc.xmean[:]=mean(self.EAc.F0.get_uDNAs()[:self.EAc.mu,:],axis=0)
            else:
                self.EAc.stateupdate()
            for g in range(G1-1):
                self.EAc.do_cmaes_step(stateupdate=True)
            self.EAc.do_cmaes_step(stateupdate=False)
        #self.EAc.stateupdate()
        # phase 2: after proto-populations: one EA needs to catch up so both msteps will have shrunken to a similar magnitude
        if self.EAc.mstep < 0.9*self.EAb.mstep:
            self.unidir_DNA_transfer('CtoB',self.strong_coupling)
            while self.EAb.mstep > self.EAc.mstep:
                self.EAb.do_step()
        elif self.EAb.mstep < 0.9*self.EAc.mstep:
            self.unidir_DNA_transfer('BtoC',self.strong_coupling)
            self.EAc.stateupdate()
            while self.EAc.mstep > self.EAb.mstep:
                self.EAc.step_cmaes_do()
        print 'msteps before alignment: ',self.EAb.mstep,self.EAc.mstep
        self.EAb.mstep=self.EAc.mstep=max(self.EAb.mstep,self.EAc.mstep)
        print 'msteps after alignment: ',self.EAb.mstep,self.EAc.mstep
        # phase 3: now that mstep are equalized, start co-evolution
        self.mstep_independence=False
        self.mstep_dominance='cmaes'
        while True:
            self.do_step()
            if self.other_stopcrit_returned_True():
                break
    
    def unidir_DNA_transfer(self,direc,amount):
        if direc == 'BtoC':
            nums=ladivsel(self.EAb.F0,amount,mindist=2*self.EAb.mstep,uCS=True,selp=self.selp)
            for i,dude in enumerate(self.EAc.F0[-amount:]):
                dude.copy_DNA_of(self.EAb.F0[nums[i]],copyscore=True,copymutagenes=False)
                dude.ancestcode=0.58
            self.EAc.F0.sort()
            self.EAc.F0.update_no()
        elif direc == 'CtoB':
            nums=ladivsel(self.EAc.F0,amount,mindist=2*self.EAc.mstep,uCS=True,selp=self.selp)
            for i,dude in enumerate(self.EAb.F0[-amount:]):
                dude.copy_DNA_of(self.EAc.F0[nums[i]],copyscore=True,copymutagenes=False)
                dude.ancestcode=0.58
            self.EAb.F0.sort()
            self.EAb.F0.update_no()
        else:
            raise ValueError("invalid direction label ('BtoC' or 'CtoB'): "+direc)
        
    def bidir_DNA_transfer(self,amount):
        # in EAc F1 population becomes a copy of the original F0 population
        for fdude,pdude in zip(self.EAc.F1,self.EAc.F0):
            fdude.copy_DNA_of(pdude,copyscore=True,copymutagenes=False)
        assure_sortedness(self.EAb.F0)
        assure_sortedness(self.EAc.F0)
        assure_sortedness(self.EAc.F1)
        # first DNA transfer B--> C
        nums=ladivsel(self.EAb.F0,amount,mindist=2*self.EAb.mstep,uCS=True,selp=self.selp)
        if not (0 in nums): print 'BtoC alarm, 0 is not in nums in generation ',self.EAb.F0.gg
        for i,dude in enumerate(self.EAc.F0[-amount:]):
            dude.copy_DNA_of(self.EAb.F0[nums[i]],copyscore=True,copymutagenes=False)
            dude.ancestcode=0.58
        # next DNA transfer C--> B (from untouched storage)
        nums=ladivsel(self.EAc.F1,amount,mindist=2*self.EAc.mstep,uCS=True,selp=self.selp)
        if not (0 in nums): print 'CtoB alarm, 0 is not in nums in generation ',self.EAb.F0.gg
        for i,dude in enumerate(self.EAb.F0[-amount:]):
            dude.copy_DNA_of(self.EAc.F1[nums[i]],copyscore=True,copymutagenes=False)
            dude.ancestcode=0.58
        self.update_population_sorting()

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        # remove unpicklable ctypes stuff from the dictionary used by the pickler
        del odict['more_stop_crits']
        del odict['generation_callbacks']
        return odict

    def pickle_self(self):
        ofile=open(join(self.EAb.F0.picklepath,self.ownname+'_'+self.EAb.F0.label+'.txt'), 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()

    def tell_best_score(self,andDNA=False):
        dudelist=[self.F0b[0],self.F0c[0],self.sim[0]]
        if self.EAc.bestdude is not None:
            dudelist.append(self.EAc.bestdude)
        idx=0
        for i,dude in enumerate(dudelist):
            if dude.isbetter(dudelist[i]):
                idx=i
        if andDNA:
            return dudelist[idx].score,dudelist[idx].get_copy_of_DNA()
        else:
            return dudelist[idx].score




#------------------------------------------------------------------------------
#--- utilities
#------------------------------------------------------------------------------

def stopper(eaobj):
    if eaobj.tell_neval()>=eaobj.maxeval:
        return True
    else:
        return False

def stop_at_distance(eaobj):
    if eaobj.tell_neval()>=eaobj.maxeval-eaobj.breaking_distance:
        return True
    else:
        return False



def assure_sortedness(p):
    """check whether a population is well sorted"""
    last_no=-1
    sc=p.get_scores()
    errlist=[]
    if p.whatisfit in ['minimize','min']:
        last_s=np.min(sc)-10
        for i,dude in enumerate(p):
            if dude.no <= last_no:
                errlist.append([i,dude.no,dude.score,'wrong numbering'])
            if dude.score < last_s:
                errlist.append([i,dude.no,dude.score,'wrong score sorting'])
    else:
        last_s=np.max(sc)+10
        for i,dude in enumerate(p):
            if dude.no <= last_no:
                errlist.append([i,dude.no,dude.score,'wrong numbering'])
            if dude.score > last_s:
                errlist.append([i,dude.no,dude.score,'wrong score sorting'])
    if len(errlist) != 0:
        print 'population not correctly sorted: {} in generation {}'.format(p.ownname,p.gg)
        print errlist

class ScoreDistribRecorder:
    
    def __init__(self,eac):
        self.eac=eac
        self.p=eac.F0
        self.gg=[]
        self.bestDNA=[]
        self.score000=[]  # best score
        self.score025=[]  # best score of 2nd quarter
        self.score050=[]  # median score
        self.score075=[]  # best score of last quarter
        self.score100=[]  # worst score
        self.scorelists=[self.score000,self.score025,self.score050,self.score075,self.score100]
        self.status=[]    # 1 -> random dudes; 2 -> protopopulations; 3 -> normal; 4,5 -> linesearch; 6 -> downhill-simplex; 7 -> pure CMA-ES
        self.ownname='SDRec_of_'+eac.ownname
        self.F0ps=self.eac.F0.psize
        if hasattr(eac,'sim'):
            self.simps=self.eac.sim.psize
            self.simidx=[0, self.simps/4, self.simps/2, 3*self.simps/4, self.simps-1]
        self.F0idx=[0, self.F0ps/4, self.F0ps/2, 3*self.F0ps/4, self.F0ps-1]
    
    def popsize_change(self):
        self.F0ps=self.eac.F0.psize
        self.F0idx=[0, self.F0ps/4, self.F0ps/2, 3*self.F0ps/4, self.F0ps-1]
    
    def save_status(self,eaobj):
        if self.p.psize!=self.F0ps: self.popsize_change()
        if self.eac.status == 'random_dudes':
            self.gg.append(self.eac.F0.gg)
            self.status.append(1)
            self.F0_snapshot()
        elif self.eac.status == 'proto_populations':
            self.gg.append(self.eac.F0.gg)
            self.status.append(2)
            self.F0_snapshot()
        elif self.eac.status == 'normal':
            self.gg.append(self.eac.F0.gg)
            self.status.append(3)
            self.F0_snapshot()
        elif self.eac.status == '1D_search':
            self.gg.append(self.eac.F0.gg)
            self.status.append(4)
            self.unsorted_F0_snapshot()
        elif self.eac.status == 'finalizing_1D_search':
            self.gg.append(self.eac.F0.gg)
            self.status.append(5)
            self.F0_snapshot()
        elif self.eac.status == 'Nelder_Mead':
            self.gg.append(self.eac.sim.gg)
            self.status.append(6)
            self.simplex_snapshot()
        elif self.eac.status in ['normal_cmaes','pure_cmaes']:
            self.gg.append(self.eac.F0.gg)
            self.status.append(7)
            self.F0_snapshot()
        else:
            raise ValueError("Recorder can't resolve this EA status "+self.eac.status)
    
    def F0_snapshot(self):
        self.bestDNA.append(self.eac.F0[0].get_copy_of_DNA())
        for container,idx in zip(self.scorelists,self.F0idx):
            container.append(self.eac.F0[idx].score)
    
    def unsorted_F0_snapshot(self):
        sc=self.eac.F0.get_scores()
        if self.eac.F0.whatisfit in ['minimize','minimise','min','mn']:
            bidx=sc.argmin(); seq=argsort(sc)
        else:
            bidx=sc.argmax(); seq=argsort(-sc)
        self.bestDNA.append(self.eac.F0[bidx].get_copy_of_DNA())
        for container,idx in zip(self.scorelists,self.F0idx):
            container.append(self.eac.F0[seq[idx]].score)
    
    def simplex_snapshot(self):
        self.bestDNA.append(self.eac.sim[0].get_copy_of_DNA())
        for container,idx in zip(self.scorelists,self.simidx):
            container.append(self.eac.sim[idx].score)
    
    def clear(self):
        self.gg=[]
        self.bestDNA=[]
        self.score000=[]  # best score
        self.score025=[]  # best score of 2nd quarter
        self.score050=[]  # median score
        self.score075=[]  # best score of last quarter
        self.score100=[]  # worst score
        self.status=[]    # 1 -> random dudes; 2 -> protopopulations; 3 -> normal; 4 -> linesearch; 5 -> downhill-simplex
        self.scorelists=[self.score000,self.score025,self.score050,self.score075,self.score100]

    def pickle_self(self):
        ofile=open(join(self.eac.F0.picklepath,self.ownname+'_'+self.eac.F0.label[:-5]+'.txt'), 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()


class ScoreDistribRecorderD:
    
    def __init__(self,eacD):
        self.eac=eacD
        self.sdrb=ScoreDistribRecorder(eacD.EAb)
        self.sdrc=ScoreDistribRecorder(eacD.EAc)
        eacD.EAb.generation_callbacks.append(self.sdrb.save_status)
        eacD.EAc.generation_callbacks.append(self.sdrc.save_status)
        self.status=[]
        self.ownname='SDRec_of_'+eacD.ownname
        
    def clear(self):
        self.sdrb.clear()
        self.sdrc.clear()
        
    def pickle_self(self):
        ofile=open(join(self.eac.EAb.F0.picklepath,self.ownname+'_'+self.eac.EAb.F0.label[:-5]+'.txt'), 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()




