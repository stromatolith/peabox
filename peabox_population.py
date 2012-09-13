#!python
"""
peabox - python evolutionary algorithm toolbox
by Markus Stokmaier, IKET, KIT, Karlsruhe, August 2012

In this file:
 - definition of the class Population and its derivatives for MO-optimisation
 - a copy function for population instances
"""

from os import getcwd
from os.path import join
from cPickle import Pickler
from copy import copy
from threading import Thread, Lock
from time import time, localtime

import numpy as np
import numpy.random as npr

from scipy.optimize import brent
from scipy.stats import pearsonr

#-----------------------------------------------------------------------------------------------------------------------------
from peabox_individual import Individual  #, MOIndividual
#-----------------------------------------------------------------------------------------------------------------------------

def population_like(otherpop,size=0):
    if isinstance(otherpop,wTOPopulation):
        objectives=[[name,func] for name,func in zip(otherpop.objnames,otherpop.objfuncs)]
        p=wTOPopulation(otherpop.species,size,objectives,otherpop.pars)
        p.ncase=otherpop.ncase
        p.gg=otherpop.gg
        p.subcase=otherpop.subcase
        p.nthreads=otherpop.nthreads
        p.determine_whatisfit(otherpop.whatisfit,otherpop.objdirecs)
        p.goal=copy(otherpop.goal)
        p.rankweights=copy(otherpop.rankweights)
        p.sumcoeffs=copy(otherpop.sumcoeffs)
        p.offset=otherpop.offset
        p.path=otherpop.path
        p.picklepath=otherpop.picklepath
        p.plotpath=otherpop.plotpath
        p.moreprops=copy(otherpop.moreprops)
        p.scoreformula=otherpop.scoreformula
        p.sortstrategy=otherpop.sortstrategy
        p.weights_flip=otherpop.weights_flip
        p.weights_rand=otherpop.weights_rand
        p.weights_rand_factor=otherpop.weights_rand_factor
        p.weights_fric=otherpop.weights_fric
        p.optsx=otherpop.optsx
        p.optrx=otherpop.optrx
        p.optw_successrates=copy(otherpop.optw_successrates)
        for key in p.optw_successrates: p.optw_successrates[key]=0
        p.update_label()
        for dude in p:
            dude.sumcoeffs[:]=otherpop.sumcoeffs
            dude.offset=otherpop.offset
            dude.rankweights[:]=otherpop.rankweights
        return p
    elif isinstance(otherpop,MOPopulation):
        objectives=[[name,func] for name,func in zip(otherpop.objnames,otherpop.objfuncs)]
        p=MOPopulation(otherpop.species,size,objectives,otherpop.pars)
        p.ncase=otherpop.ncase
        p.gg=otherpop.gg
        p.subcase=otherpop.subcase
        p.nthreads=otherpop.nthreads
        p.determine_whatisfit(otherpop.whatisfit,otherpop.objdirecs)
        p.goal=copy(otherpop.goal)
        p.rankweights=copy(otherpop.rankweights)
        p.sumcoeffs=copy(otherpop.sumcoeffs)
        p.offset=otherpop.offset
        p.path=otherpop.path
        p.picklepath=otherpop.picklepath
        p.plotpath=otherpop.plotpath
        p.moreprops=copy(otherpop.moreprops)
        p.scoreformula=otherpop.scoreformula
        p.update_label()
        for dude in p:
            dude.sumcoeffs[:]=otherpop.sumcoeffs
            dude.offset=otherpop.offset
            dude.rankweights[:]=otherpop.rankweights
        return p
    else:
        p=Population(otherpop.species,size,otherpop.objfunc,otherpop.pars)
        p.objname=otherpop.objname
        p.ncase=otherpop.ncase
        p.gg=otherpop.gg
        p.subcase=otherpop.subcase
        p.nthreads=otherpop.nthreads
        p.determine_whatisfit(otherpop.whatisfit)
        p.goal=copy(otherpop.goal)
        p.path=otherpop.path
        p.picklepath=otherpop.picklepath
        p.plotpath=otherpop.plotpath
        p.moreprops=copy(otherpop.moreprops)
        p.scoreformula=otherpop.scoreformula
        p.update_label()
        return p
    
    


class Population:
    
    def __init__(self,species,popsize,objfunc,paramspace):
        ngenes=len(paramspace)
        self.pars=tuple(paramspace)
        self.parnames=tuple([el[0] for el in self.pars])
        self.species=species # should be the class Individual or a subclass
        self.objfunc=objfunc
        self.objname=str(self.objfunc).split()[1]
        self.psize=int(popsize)   # population size
        self.ng=ngenes       # number of parameters, i.e. number of genes for each member of population
        self.gg=0            # generation counter
        self.ncase=0         # case or study number, which will used for labelling all output; helps you keep track of what you did
        self.subcase=0       # if you do not want to waste case numbers when looping over many evolution runs for statistics
        self.nthreads=1      # if 0 use self.get_scores() else use self.get_scores_threaded(self.nthreads) ... threaded so far just possible for ansys batch jobs
        self.leastthreads=1  # lowest desired number of threads when using multithreading
        self.folks=[]
        self.whatisfit='minimize'
        for i in range(self.psize):
            newdude=self.species(objfunc,paramspace)
            newdude.no=i; newdude.oldno=i; newdude.ncase=self.ncase
            self.folks.append(newdude)
        self.goal={'goalvalue':0,'fulfilltime':-1, 'fulfillcalls':-1}   # you can define a convergence criterium goalvalue and note down the generation when there was the first Individual getting the corresponding score
        self.neval=0                    # counter for score evaluations
        self.path=getcwd()
        self.datapath=join(self.path,'out')
        self.plotpath=join(self.path,'plots')
        self.picklepath=join(self.path,'pickles')
        self.scoreformula='scoreformula not yet defined'
        self.update_label()
        self.moreprops={}
        return
    
    def __len__(self):
        return self.psize
    
    def __getitem__(self,key):
        return self.folks[key]
    
    def __setitem__(self,key,newDNA):
        self.folks[key].set_DNA(newDNA)
        self.folks[key].score=0.
        self.folks[key].pa=-1
        self.folks[key].pb=-1
        
    def __delitem__(self,key):
        del self.folks[key]
        self.psize-=1
        
    def __contains__(self,somedude):
        for dude in self.folks:
            if somedude is dude:
                return True
        return False
    
    def __add__(self,otherdudes):
        # otherdudes may be population, Individual, or list of Individuals, population.append(stuff) eats it all
        newpop=population_like(self,size=0)
        newpop.append(self)
        newpop.append(otherdudes)
        return newpop
    
    def __getslice__(self,i,j):
        newpop=population_like(self,size=0)
        newpop.append(self.folks[i:j])
        return newpop
    
    def __str__(self):
        s='Population {objective = '+self.whatisfit+' '+self.objname+'} [ '
        for dude in self:
            s+=str(dude)+' '
        s+=']'
        return s

    def pop(self,key=None):
        self.psize-=1
        if key is not None:
            return self.folks.pop(key)
        else:
            return self.folks.pop()
        
    def append(self,newdude_or_dudes):
        if isinstance(newdude_or_dudes,Population):
            n=len(newdude_or_dudes)
            if n==0:
                pass
            elif newdude_or_dudes.objfunc is self.objfunc:
                [self.folks.append(dude) for dude in newdude_or_dudes.folks]
                self.psize+=n
            else:
                raise TypeError('can only concatenate populations of Individuals with same objective function')
        elif isinstance(newdude_or_dudes,Individual):
            if newdude_or_dudes.objfunc is self.objfunc:
                self.folks.append(newdude_or_dudes)
                self.psize+=1
            else:
                raise TypeError('can only concatenate populations of Individuals with same objective function')
        elif isinstance(newdude_or_dudes,list):
            n=len(newdude_or_dudes)
            if n==0:
                pass
            else:
                if isinstance(newdude_or_dudes[0],Individual):
                    if newdude_or_dudes[0].objfunc is self.objfunc:
                        [self.folks.append(dude) for dude in newdude_or_dudes]
                        self.psize+=n
                    else:
                        raise TypeError('can only concatenate populations of Individuals with same objective function')
                else:
                    raise TypeError('only instances of class Individual or subclasses can become member of a Population')
            
    def sort(self):
        self.folks.sort()
        if self.whatisfit=='maximize': self.folks.reverse()

    def sort_for(self,attributename):
        """The argument for this function can be any string that matches a valid Individual attribute of scalar value"""
        values=[eval('dude.'+attributename) for dude in self.folks]
        self.folks=[self.folks[i] for i in np.argsort(values)]

    def sort_for_DNApar(self,parametername):
        """The argument for this function can be any string that matches one of the parameter names or its index in self.parnames"""
        if type(parametername) is str:
            idx=self.parnames.index(parametername)
        elif type(parametername) is int:
            idx=parametername
        else:
            raise TypeError('argument of self.sort_for_DNApar() must be of type str or int')
        values=[eval('dude.DNA['+str(idx)+']') for dude in self.folks]
        self.folks=[self.folks[i] for i in np.argsort(values)]

    def reverse(self):
        self.folks.reverse()
        
    def index(self,somedude):
        for i,dude in enumerate(self.folks):
            if dude is somedude:
                return i
        raise ValueError('that dude is not an inhabitant of the population')
    
    def update_no(self):
        for i,dude in enumerate(self):
            dude.no=i
            
    def mark_oldno(self,fromno=False):
        if fromno:
            for i,dude in enumerate(self):
                dude.oldno=dude.no
        else:
            for i,dude in enumerate(self):
                dude.oldno=i
            
    def new_random_genes(self):      
        for dude in self:
            dude.random_DNA()
            
    def eval_all(self):
        if self.nthreads==1:
            self.eval_all_serial()
        else:
            self.eval_all_threaded(low_n=self.leastthreads, desired_n=self.nthreads)
            
    def eval_all_serial(self):
        for i,dude in enumerate(self):
            dude.evaluate()
        self.neval+=self.psize

    def eval_all_threaded(self,low_n=None,desired_n=None):
        jobcounter=0
        while jobcounter < self.psize:
            n=determine_best_thread_amount(low_n=low_n, desired_n=desired_n)
            threads=[]; thrlock=Lock()
            for nn in range(n):
                threads.append(single_eval_thread(self.folks[jobcounter],thrlock))
                jobcounter+=1
                if jobcounter==self.psize: break
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()
        self.neval+=self.psize
            
    def eval_bunch(self,i_key,f_key):
        if self.nthreads==1:
            self.eval_bunch_serial(i_key,f_key)
        else:
            self.eval_bunch_threaded(i_key,f_key,desired_n=self.nthreads)
        
    def eval_bunch_serial(self,i_key,f_key):
        dudeslist=self[i_key:f_key]
        for dude in dudeslist:
            dude.evaluate()
        self.neval+=len(dudeslist)
        
    def eval_bunch_threaded(self,i_key,f_key,low_n=None,desired_n=None):
        dudeslist=self[i_key:f_key]; nbunch=len(dudeslist)
        jobcounter=0
        while jobcounter < nbunch:
            n=determine_best_thread_amount(low_n=low_n, desired_n=desired_n)
            threads=[]; thrlock=Lock()
            for nn in range(n):
                threads.append(single_eval_thread(dudeslist[jobcounter],thrlock))
                jobcounter+=1
                if jobcounter==nbunch: break
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()
        self.neval+=self.psize
            
    def eval_one_dude(self,key):
        self[key].evaluate()
        self.neval+=1

    def get_DNAs(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.DNA for dude in self[i_key:f_key]],dtype=float,copy=1)

    def get_uDNAs(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.get_uDNA() for dude in self[i_key:f_key]],dtype=float,copy=1)

    def get_scores(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.score for dude in self[i_key:f_key]],dtype=float,copy=1)

    def get_ancestcodes(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.ancestcode for dude in self[i_key:f_key]],dtype=float,copy=1)

    def get_mutagenes(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.mutagenes for dude in self[i_key:f_key]],dtype=float,copy=1)

    def change_size(self,newsize,sortbefore=True,updateno=True):
        # allows you to play with population bottlenecks, or to apply two different EAs to one population in series, i.e. gives a lot of freedom if you want to continue a good population with a modified EA
        if sortbefore: self.sort()
        if updateno: self.update_no()
        if newsize<self.psize:
            for i in range(newsize,self.psize):
                self.pop()
        elif newsize>self.psize:
            for i in range(self.psize,newsize):
                newdude=self.species(self.objfunc,self.pars)
                newdude.no=i; newdude.oldno=i; newdude.random_DNA(); newdude.reset_mutagenes()
                self.append(newdude)
        else:
            return
        return
    
    def set_parameter_limits(self,lolims,hilims):
        lolims=np.asfarray(lolims); hilims=np.asfarray(hilims)
        for dude in self.folks:
            dude.lls[:]=lolims; dude.uls[:]=hilims; dude.widths[:]=hilims-lolims
            
    def set_every_DNA(self,params,uCS=False):
        if uCS:
            for dude in self:
                dude.set_uDNA(params)
        else:
            for dude in self:
                dude.set_DNA(params)
                
    def reset_individual_attributes(self):
        for i,dude in enumerate(self):
            dude.oldno=i; dude.ancestcode=1.; dude.pa=-1; dude.pb=-1; dude.score=0.; dude.gg=self.gg
            
    def reset_all_mutagenes(self,value=1.):
        for dude in self:
            dude.reset_mutagenes(value=value)
            
    def marker_genes(self,factor=0.1,offset=0.):
        # for debugging and testing purposes: if you want the DNAs to consist of numbers you can easily recognize after e.g. crossing-over
        for i,dude in enumerate(self):
            dude.set_DNA(factor*np.arange(self.ng,dtype=float)+i+offset)
            
    def determine_whatisfit(self,direction):
        # if direction = 'minimize' -> goal is to find the minimum of the score function
        # if direction = 'maximize' -> goal is to find the maximum of the score function
        if direction in ['minimise','min','mn','down','downwards']: direction='minimize'
        if direction in ['maximise','max','mx','up','upwards']: direction='maximize'
        assert direction in ['minimize','maximize']  # sorry for the Brits ....
        self.whatisfit=direction
        for dude in self:
            dude.whatisfit=direction

    def update_label(self):
        self.label='c'+str(self.ncase).zfill(3)+'_sc'+str(self.subcase).zfill(3)+'_g'+str(self.gg).zfill(3)
        
    def advance_generation(self):
        self.gg+=1;
        for dude in self:
            dude.gg=self.gg
        self.update_label()
        
    def set_ncase(self,newcase):
        self.ncase=newcase; self.subcase=0; self.update_label()
        for dude in self:
            dude.ncase=newcase
            dude.subcase=0
            
    def set_subcase(self,sc):
        self.subcase=sc; self.update_label()
        for dude in self:
            dude.subcase=sc
        
    def next_ncase(self):
        self.ncase+=1; self.subcase=0; self.update_label()
        for dude in self:
            dude.ncase=self.ncase
            dude.subcase=0
        
    def next_subcase(self):
        self.subcase+=1; self.update_label()
        for dude in self:
            dude.subcase=self.subcase
        
    def mean_DNA(self,i_key=None,f_key=None,uCS=False):
        if uCS:
            muDNA=np.zeros(self.ng)
            for dude in self[i_key:f_key]:
                muDNA+=dude.get_uDNA()
            muDNA/=len(self[i_key:f_key])
            return muDNA
        else:
            mDNA=np.zeros(self.ng)
            for dude in self[i_key:f_key]:
                mDNA+=dude.DNA
            mDNA/=len(self[i_key:f_key])
            return mDNA
        
    def set_goalvalue(self,value,reset=True):
        self.goal['goalvalue']=value
        if reset:
            self.goal['fulfilltime']=-1; self.goal['fulfillcalls']=-1
        
    def check_and_note_goal_fulfillment(self):
        sc=self.get_scores()
        factor=1.
        if self.whatisfit=='maximize': factor=-1.
        if self.goal['fulfilltime']==-1 and np.min(factor*sc) <= factor*self.goal['goalvalue']:
            self.goal['fulfilltime']=self.gg-1
            self.goal['fulfillcalls']=self.neval

    def reset(self):
        self.gg=0
        self.reset_individual_attributes()
        self.update_label()
        self.neval=0
        self.goal['fulfilltime']=-1; self.goal['fulfillcalls']=-1
            
    def zeroth_generation(self):
        self.mark_oldno()
        self.eval_all()
        self.sort()
        self.update_no()
        self.check_and_note_goal_fulfillment()
        
    def print_stuff(self,slim=False):
        print '-----------------------------------------------------------------------------------------------------------'
        print '-------------------------------- *** summary of the population *** ----------------------------------------'
        print 'ncase: '+str(self.ncase)+'     subcase: '+str(self.subcase)+'     generation: '+str(self.gg)
        if slim:
            print '  ranking of dude  ---  his former ranking  ---  parent a  ---  parent b  ---  score '
        else:
            print '  ranking of dude  ---  his former ranking  ---  parent a  ---  parent b  ---  score  ---  DNA '
        for dude in self.folks:
            dude.print_stuff(slim=slim)
        print '-------------------------------- *** end of population summary *** ----------------------------------------'
        print '-----------------------------------------------------------------------------------------------------------'
        
    def pickle_self(self):
        ofile=open(self.picklepath+'/pop_'+self.label+'.txt', 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()
        
    def write_popstatus(self):
        ofile=open(self.datapath+'/evostatus_'+self.label+'.txt', 'w')
        ofile.write('current no. | old no. | parent a | parent b | ancestcode |                DNA:   ')
        for parstr in self.pars:
            ofile.write(parstr[0]+'    ')
        ofile.write('     |  score\r\n')
        for i,dude in enumerate(self):
            ofile.write(str(i)+'    '+str(dude.oldno)+'    '+str(dude.pa)+'    '+str(dude.pb)+'    '+str(dude.ancestcode)+'    ')
            for gene in dude.DNA:
                ofile.write(str(gene)+'    ')
            ofile.write(str(dude.score)+'\r\n')
        ofile.write('\r\n\r\n\r\n')
        ofile.write('psize: '+str(self.psize)+'\r\n')
        ofile.write('goal: '+str(self.goal)+'\r\ngg: '+str(self.gg)+'\r\nwhatisfit: '+str(self.whatisfit)+'\r\n')
        ofile.close()
        return
    
    def recover_popstatus(self,path_fname,howmany=None):
        if howmany is None: howmany=self.psize
        ofile=open(path_fname,'r')
        ofile.readline()
        for i in range(howmany):
            words=ofile.readline().split()
            self[i].no=int(float(words[0]))
            self[i].oldno=int(float(words[1]))
            self[i].pa=int(float(words[2]))
            self[i].pb=int(float(words[3]))
            self[i].ancestcode=int(float(words[4]))
            self[i].set_DNA([float(words[j]) for j in range(5,5+self.ng)])
            self[i].score=float(words[self.ng+5])
        ofile.close()
        self.sort()
        return





#-----------------------------------------------------------------------------------------------------------------------------
#           for a first concept of multi-objective optimisation: class "MOPopulation" for use with "MOIndividual"
#-----------------------------------------------------------------------------------------------------------------------------

class MOPopulation(Population):
    
    def __init__(self,species,popsize,objectives,paramspace):
        ngenes=len(paramspace)
        self.pars=tuple(paramspace)
        self.species=species             # should be the class MOIndividual or a subclass
        objfuncs=[]; objnames=[]
        for name,func in objectives:
            objnames.append(name)
            objfuncs.append(func)
        self.objnames=objnames
        self.objfuncs=objfuncs
        self.nobj=len(objfuncs)                    # the number of objectives
        self.objdirecs=self.nobj * ['min']         # list containing optimisation directions
        self.minmaxflip=np.ones(self.nobj,dtype=int)  # same as objdirecs in the form of an array with elements being 1 or -1 (the latter where 'max')
        self.whatisfit='minimize'                   # the ordering direction for the overall scalar fitness (if used)
        self.psize=int(popsize)   # population size
        self.ng=ngenes       # number of parameters, i.e. number of genes for each member of population
        self.gg=0            # generation counter
        self.ncase=0         # case or study number, which will used for labelling all output; helps you keep track of what you did
        self.subcase=0       # if you do not want to waste case numbers when looping over many evolution runs for statistics
        self.nthreads=1      # if 0 use self.get_scores() else use self.get_scores_threaded(self.nthreads) ... threaded so far just possible for ansys batch jobs
        self.folks=[]
        for i in range(self.psize):
            newdude=self.species(objectives,paramspace)
            newdude.no=i; newdude.oldno=i; newdude.ncase=self.ncase
            self.folks.append(newdude)
        self.paretofront=[]
        self.goal={'goalvalue':0, 'fulfilltime':-1, 'fulfillcalls':-1,
                   'objgoalvalues':np.zeros(self.nobj), 'objfulfilltimes':-np.ones(self.nobj), 'objfulfillcalls':-np.ones(self.nobj)}
        self.rankweights=np.ones(self.nobj,dtype=float)/self.nobj# the coefficients when summing over the different ranks in order to calculate self.overall_rank
        self.sumcoeffs=np.ones(self.nobj,dtype=float)/self.nobj # coefficients in case overall score can be calculated as a weighted sum
        self.offset=0.                             # additional constant offset in case score can be calculated as a weighted sum
        self.neval=0                    # counter for score evaluations
        self.path=getcwd()
        self.plotpath=join(self.path,'plots')
        self.picklepath=join(self.path,'pickles')
        self.scoreformula='scoreformula not yet defined'
        self.update_label()
        self.moreprops={}

    def __setitem__(self,key,newDNA):
        self.folks[key].set_DNA(newDNA)
        self.folks[key].score=0.
        self.folks[key].overall_rank=self.psize
        self.folks[key].objvals[:]=0.
        self.folks[key].paretoefficient=False
        self.folks[key].paretooptimal=False
        self.folks[key].paretoking=False
        self.folks[key].pa=-1
        self.folks[key].pb=-1
        
    def __str__(self):
        s='Population {objectives = '+[self.objdirecs[i]+' '+self.objnames[i]+', ' for i in range(self.nobj)]+'} [ '
        for dude in self:
            s+=str(dude)+' '
        s+=']'
        return s
        
    def append(self,newdude_or_dudes):
        if isinstance(newdude_or_dudes,MOPopulation):
            n=len(newdude_or_dudes)
            if n==0:
                pass
            elif all([ofun1 is ofun2 for ofun1,ofun2 in zip(self.objfuncs,newdude_or_dudes.objfuncs)]):
                [self.folks.append(dude) for dude in newdude_or_dudes.folks]
                self.psize+=n
            else:
                raise TypeError('can only concatenate populations of MOIndividuals with same list of objective functions')
        elif isinstance(newdude_or_dudes,Individual):
            if all([ofun1 is ofun2 for ofun1,ofun2 in zip(self.objfuncs,newdude_or_dudes.objfuncs)]):
                self.folks.append(newdude_or_dudes)
                self.psize+=1
            else:
                raise TypeError('can only concatenate populations of Individuals with same list ofobjective functions')
        elif isinstance(newdude_or_dudes,list):
            n=len(newdude_or_dudes)
            if n==0:
                pass
            else:
                if isinstance(newdude_or_dudes[0],Individual):
                    if all([ofun1 is ofun2 for ofun1,ofun2 in zip(self.objfuncs,newdude_or_dudes[0].objfuncs)]):
                        [self.folks.append(dude) for dude in newdude_or_dudes]
                        self.psize+=n
                    else:
                        raise TypeError('can only concatenate populations of Individuals with same objective function')
                else:
                    raise TypeError('only instances of class Individual or subclasses can become member of a Population')
            
#    def sort(self):
#        self.folks.sort()
#        if self.orderfit=='maximize': self.folks.reverse()

    def sort_for(self,critname):
        """The argument for this function can be any string that matches either one of the objective names or any
        valid MOIndividual attribute of scalar value"""
        if critname in self.objnames:
            oidx=self.objnames.index(critname)
            values=[dude.objvals[oidx] for dude in self.folks]
        else:
            values=[eval('dude.'+critname) for dude in self.folks]
        self.folks=[self.folks[i] for i in np.argsort(values)]

    def get_objvals(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.objvals for dude in self[i_key:f_key]],dtype=float,copy=1)

#    def get_orderscores(self,i_key=None,f_key=None):
#        if i_key is None: i_key=0
#        if f_key is None: f_key=self.psize
#        return array([dude.orderscore for dude in self[i_key:f_key]],dtype=float,copy=1)

    def get_ranks(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.ranks for dude in self[i_key:f_key]],dtype=float,copy=1)

    def get_overall_ranks(self,i_key=None,f_key=None):
        if i_key is None: i_key=0
        if f_key is None: f_key=self.psize
        return np.array([dude.overall_rank for dude in self[i_key:f_key]],dtype=float,copy=1)
    
    def update_scores(self):
        for dude in self: dude.update_score()
    
    def update_overall_ranks(self):
        for dude in self: dude.update_overall_rank()
        
    def update_rankings(self):
        for i,oname in enumerate(self.objnames):
            self.sort_for(oname)
            if self.objdirecs[i]=='max': self.reverse()
            for j,dude in enumerate(self):
                dude.ranks[i]=j
        self.update_overall_ranks()
    
    def update_pareto_front(self,update_rankings=True):
        # this routine should be okay for most real-coded problems where the probability of same DNA vector entries is small
        # if you however should care about > vs >= or < vs <= ... then better go with strictly_update_pareto_front()
        if update_rankings: self.update_rankings()
        ov=self.get_objvals()
        self.paretofront=[]
        for dude in self:
            dude.paretoefficient=False
            betterdudes=np.zeros((self.nobj,self.psize),dtype=int)
            for i in range(self.nobj):
                betterdudes[i,:]=np.where(self.minmaxflip[i]*ov[:,i] < self.minmaxflip[i]*dude.objvals[i],1,0)
            better_everywhere=np.prod(betterdudes,axis=0)
            if np.sum(better_everywhere)==0:
                dude.paretoefficient=True  # no other dude is better at all disciplines at the same time
                self.paretofront.append(dude)
            if np.sum(betterdudes)==0: dude.paretooptimal=True          # at each single discipline there cannot be found a better one
    
    def strictly_update_pareto_front(self):
        # this routine makes sure that only members strictly dominating everybody else enter the pareto front
        # it looks shorter, but I guess it might take longer to execute than update_pareto_front() in case of large populations
        self.paretofront=[]
        for dude in self:
            dude.paretoefficient=True
            for otherdude in self:
                if otherdude.strictly_dominates(dude): dude.paretoefficient=False
            if dude.paretoefficient==True: self.paretofront.append(dude)
                
    def print_MO_stuff(self):
        print 20*'-'+'MO property summary'+50*'-'
        print 'objectives: '+str(self.objnames)+'\n'+80*'-'
        for i,dude in enumerate(self):
            t='{0:2}: dude {1:3}   overall score = {4:10.3f}'.format(i,dude.no,dude.score)
            t+='   overall_rank = {2:5.2f}   pe={3:1}   po={4:1}'.format(dude.ranks[0],dude.ranks[1],dude.overall_rank,dude.paretoefficient,dude.paretooptimal)
            if self.nobj>3: t+='\n'
            t+='objective values: '+str(np.round(dude.objvals,3))
            if self.nobj>3: t+='\n'
            t+='objective-related ranks: '+str(dude.ranks)
            print t
            if self.nobj>3: print 80*'-'
        
    def reset_Individual_attributes(self):
        for i,dude in enumerate(self):
            dude.oldno=i; dude.ancestcode=1.; dude.pa=-1; dude.pb=-1
            dude.score=0; dude.overall_rank=-1.; dude.ranks[:]=-1; dude.objvals[:]=0.
            
    def determine_whatisfit(self,direction,objdirecs):
        if direction in ['minimise','min','mn','down','downwards']: direction='minimize'
        if direction in ['maximise','max','mx','up','upwards']: direction='maximize'
        assert direction in ['minimize','maximize']  # going for AE under the pressure of popularity
        assert len(objdirecs) == len(self.objfuncs)
        for direc in objdirecs: assert direc in ['min','max']
        self.whatisfit=direction
        self.objdirecs=objdirecs
        for i in range(self.nobj):
            if self.objdirecs[i]=='min': self.minmaxflip[i]=1
            else: self.minmaxflip[i]=-1
        for dude in self:
            dude.whatisfit=direction
            dude.objdirecs=copy(objdirecs)
            dude.minmaxflip[:]=self.minmaxflip
        
    def set_objgoalvalues(self,values):
        self.goal['objgoalvalues'][:]=values
        
#    def set_ordergoalvalue(self,value):
#        self.ordergoal['goalvalue']=value
        
    def set_sumcoeffs(self,values):
        self.sumcoeffs[:]=values
        for dude in self: dude.sumcoeffs[:]=values
                
    def set_offset(self,value):
        self.offset=value
        for dude in self: dude.offset=value
                
    def set_rankweights(self,values):
        self.rankweights[:]=values
        for dude in self: dude.rankweights[:]=values
        
    def check_and_note_goal_fulfillment(self):
        ov=self.get_objvals()
        os=self.get_scores()
        for i,factor in enumerate(self.minmaxflip):
            if self.goal['objfulfilltimes'][i]==-1 and np.min(factor*ov[:,i]) <= factor*self.goal['objgoalvalues'][i]:
                self.goal['objfulfilltimes'][i]=self.gg-1
                self.goal['objfulfillcalls'][i]=self.neval
        if self.whatisfit=='minimize': factor=1.
        else: factor=-1.
        if self.goal['fulfilltime']==-1 and np.min(factor*os) <= factor*self.goal['goalvalue']:
            self.goal['fulfilltime']=self.gg-1
            self.goal['fulfillcalls']=self.neval
            
    def reset(self):
        self.gg=0
        self.neval=0
        self.goal['objfulfilltimes'][:]=-1; self.goal['objfulfillcalls'][:]=-1
        self.goal['fulfilltime']=-1; self.goal['fulfillcalls']=-1
            
    def write_popstatus(self):
        ofile=open(self.picklepath+'/evostatus_'+self.label+'.txt', 'w')
        ofile.write('current no. | old no. | parent a | parent b | ancestcode |                DNA:   ')
        for parstr in self.pars:
            ofile.write(parstr[0]+'    ')
        ofile.write('     |           objvals        |  score\r\n')
        for i,dude in self:
            ofile.write(str(i)+'    '+str(dude.oldno)+'    '+str(dude.pa)+'    '+str(dude.pb)+'    '+str(dude.ancestcode)+'    ')
            for gene in dude.DNA:
                ofile.write(str(gene)+'    ')
            for ov in dude.objvals:
                ofile.write(str(ov)+'    ')
            ofile.write(str(dude.score)+'\r\n')
        ofile.write('\r\n\r\n\r\n')
        ofile.write('psize: '+str(self.psize)+'\r\n')
        ofile.write('gg: '+str(self.gg)+'\r\n')
        ofile.write('objdirecs: '+str(self.objdirecs)+'\r\n')
        ofile.write('whatisfit: '+str(self.whatisfit)+'\r\n')
        ofile.write('goal: '+str(self.goal)+'\r\n')
        ofile.close()
        return
    
    def recover_popstatus(self,path_fname,howmany=None):
        if howmany is None: howmany=self.psize
        ofile=open(path_fname,'r')
        ofile.readline()
        for i in range(howmany):
            words=ofile.readline().split()
            self[i].no=int(float(words[0]))
            self[i].oldno=int(float(words[1]))
            self[i].pa=int(float(words[2]))
            self[i].pb=int(float(words[3]))
            self[i].ancestcode=int(float(words[4]))
            self[i].set_DNA([float(words[j]) for j in range(5,5+self.ng)])
            self[i].objvals[:]=[float(words[j]) for j in range(5+self.ng,5+self.ng+self.nobj)]
            self[i].score=float(words[self.ng+self.nobj+5])
        ofile.close()
        self.sort()
        return    




#-----------------------------------------------------------------------------------------------------------------------------
#           for the special case of two objectives: the weighted two-objective population
#           experimenting with ranking for dynamically weighted objective values or weighted rankings
#-----------------------------------------------------------------------------------------------------------------------------

class wTOPopulation(MOPopulation):
    
    def __init__(self,species,popsize,objectives,paramspace):
        MOPopulation.__init__(self,species,popsize,objectives,paramspace)
        if self.nobj != 2: raise ValueError('this population subclass is only intended for 2 objectives')
        self.sortstrategy='swc2'    # which criterion (a) sw->sumweights or (b) rw->rankweights   for each crit 1 or 2
        self.weights_flip=0   # if nonzero, e.g. 10, then flip weights every 10 generations (weights are either sumcoeffs or rankweights)
        self.weights_rand=0   # if 'rnad' or 'randn' then add som random to the weights optimisation result
        self.weights_rand_factor=0.3 # random distribution width multiplicator
        self.weights_fric=0.7 # friction or viscosity for change of weights, i.e. what weight the old setting has when mixing in the new value
        self.optsx=0.5        # currently optimal score weight ratio from the two criteria
        self.optrx=0.5        # currently optimal rank weight ratio from the two criteria
        self.optw_successrates={'swc1_s':0, 'swc1_f':0, # counting successes and failures
                                'swc2_s':0, 'swc2_f':0, # of calls to optimal_fraction()
                                'rwc1_s':0, 'rwc1_f':0, # s for sum weights (sumcoeffs) and r for rankweights
                                'rwc2_s':0, 'rwc2_f':0} # 1 or 2 for the two criteria (triangle-stuff or stuff with Pearson's r)

    def sort(self):
        self.update_rankings()
        self.update_pareto_front()
        if self.sortstrategy.startswith('s'):
            self.optimize_weights()
            self.sort_for('score')
            if self.whatisfit=='maximize': self.folks.reverse()
        elif self.sortstrategy.startswith('r'):
            self.optimize_weights()
            self.sort_for('overall_rank')
        else:
            MOPopulation.sort(self)
    
    def reset(self):
        MOPopulation.reset(self)
        self.optsx=0.5        # currently optimal score weight ratio from the two criteria
        self.optrx=0.5        # currently optimal rank weight ratio from the two criteria
        for key in self.optw_successrates:
            self.optw_successrates[key]=0
        self.set_sumcoeffs([0.5,0.5])

    def optimal_fraction(self,criterion,critkey):
        wkey=critkey[0]; x1=0.; x2=1.; i=0
        if wkey=='s': xmid=self.sumcoeffs[0]
        elif wkey=='r': xmid=self.rankweights[0]
        while i<10:
            # this loop optimises the ratio between the two objectives' weights
            # if the weights are very unequal it makes sure the resolution gets finer
            xtol=1e-2*(x2-x1)
            try:
                x=brent(criterion,brack=(x1,xmid,x2),args=(0,wkey),tol=xtol)  # with xold in the middle of the bracket should be preferred
            except:
                trials=0; maxtrials=5
                while trials<maxtrials:
                    trials+=1
                    lastfailed=False
                    try: x=brent(criterion,brack=(x1,x1+0.05+0.9*npr.rand()*(x2-x1),x2),args=(0,wkey),tol=xtol)
                    except: lastfailed=True
                if trials==maxtrials and lastfailed:
                    self.optw_successrates[critkey+'_f']+=1
                    return -i
            distancetoborder=min(x,1-x)
            if distancetoborder > 8*xtol: break
            x1=max(0.,x-2*xtol); x2=min(1.,x+2*xtol); xmid=0.5*(x1+x2); i+=1
        if i==10:
            self.optw_successrates[critkey+'_f']+=1
            return -i
        else:
            self.optw_successrates[critkey+'_s']+=1
            return x
    
    def optimize_weights(self):
        self.update_rankings(); sxold=self.sumcoeffs[0]; rxold=self.rankweights[0]
        if self.sortstrategy[-1]=='1': criterion=self.ranking_triangles_twoobj
        elif self.sortstrategy[-1]=='2': criterion=self.correlations_criterion
        x=self.optimal_fraction(criterion,self.sortstrategy)
        if 0.<x<1.:
            if self.sortstrategy[0]=='s':
                self.optsx=x
                xnew=(1-self.weights_fric)*x+self.weights_fric*sxold; #print 'x (sumcoeffs) old and new: ',sxold,xnew
                self.set_sumcoeffs([xnew,1-xnew]); self.messup_weights('s')
                self.update_scores(); self.sort_for('score')
                if self.whatisfit=='maximize': self.folks.reverse()
            elif self.sortstrategy[0]=='r':
                self.optrx=x
                xnew=(1-self.weights_fric)*x+self.weights_fric*rxold; #print 'x (rankweights) old and new: ',rxold,xnew
                self.set_rankweights([xnew,1-xnew]); self.messup_weights('r')
                self.update_overall_ranks(); self.sort_for('overall_rank')
            return
        else:
            print 'no optimal weights could be found this time, restoring old (except for messing) setting...'
            if self.sortstrategy[0]=='s':
                self.set_sumcoeffs([sxold,1-sxold]); self.messup_weights('s')
                self.update_scores(); self.sort_for('score')
                if self.whatisfit=='maximize': self.folks.reverse()
            elif self.sortstrategy[0]=='r':
                self.set_rankweights([rxold,1-rxold]); self.messup_weights('r')
                self.update_overall_ranks(); self.sort_for('overall_rank')
            return
    
    def messup_weights(self,wkey):
        # if intended, flip weights every other generation or mess up weights with a bit of random
        # if wkey=='s' then messup sumcoeffs; if wkey=='r' then messup rankweights
        if wkey=='s':
            x=self.sumcoeffs[0]; setfunction=self.set_sumcoeffs
        elif wkey=='r':
            x=self.rankweights[0]; setfunction=self.set_rankweights
        if self.weights_flip and np.mod(self.gg,self.weights_flip)==0 and self.gg!=0:
            setfunction([1-x,x])
        else:
            s=min(x,1-x); s*=self.weights_rand_factor
            if self.weights_rand=='rand':
                x=x-s+2*s*npr.rand()
                setfunction([x,1-x])
            elif self.weights_rand=='randn':
                x+=0.5*s*npr.randn()
                if x<0: x=-x
                elif x>1: x=2-x
                setfunction([x,1-x])

    def triangles__criterion(self,x,*args):
        outflag,wkey=args
        if wkey[0]=='s':
            self.set_sumcoeffs([x,1-x]); self.update_scores(); self.sort_for('score')
            if self.whatisfit=='maximize': self.folks.reverse()
        elif wkey[0]=='r':
            self.set_rankweights([x,1-x]); self.update_overall_ranks(); self.sort_for('overall_rank')
        N=self.psize; mid=N/2.-0.5; firstthird=range(N/3); lastthird=range(N-N/3,N)  #; nft=len(firstthird); nlt=len(lastthird)
        sqdft=np.zeros(2); sqdlt=np.zeros(2)  # mean ranking deviations from mid for first (ft) and last third (lt) for the two objectives
        for i in firstthird:
            sqdft[:]+=(self[i].ranks[:]-mid)**2
        for i in lastthird:
            sqdlt[:]+=(self[i].ranks[:]-mid)**2
        sqdft=np.sqrt(sqdft)  # minimising the two values of this vector means pushing the ranks of the first third to the middle
        sqdlt=np.sqrt(sqdlt)  # maximising these values is intended to mean requiring that last third members are good at one goal and bad at the other one, but the mistake is that more stuff would be necessary to prove anticorrelation
        if outflag: return sqdft,sqdlt
        else: return np.prod(sqdft) / np.prod(sqdlt)  # minimising this means minimising sqdft and maximising sqdlt
    
    def correlations_criterion(self,x,*args):
        outflag,wkey=args
        if wkey[0]=='s':
            self.set_sumcoeffs([x,1-x]); self.update_scores(); self.sort_for('score')
            if self.whatisfit=='maximize': self.folks.reverse()
        elif wkey[0]=='r':
            self.set_rankweights([x,1-x]); self.update_overall_ranks(); self.sort_for('overall_rank')
        no=np.arange(self.psize)
        rks=self.get_ranks()
        r1,pfakt1=pearsonr(no,rks[:,0])
        r2,pfact2=pearsonr(no,rks[:,1])
        if outflag: return r1,r2
        else: return abs(r1-r2)*max(abs(r1),abs(r2))  



class single_eval_thread(Thread):
    def __init__(self,dude,schloss):
        # what is argument "dude"? -> associate one individual with this thread, one individual whose score has to be evalued
        # what is argument "schloss"? -> this refers to a threading.Lock() instance, which all threads have to share in order to prevent them
        #                                from interfering, e.g. when using os.chdir()
        self.dude=dude
        self.tlock=schloss
        Thread.__init__(self)
    def run(self):
        #print 'starting evaluate() on dude ',self.dude.no
        self.dude.evaluate(tlock=self.tlock)


def determine_best_thread_amount(low_n=None, desired_n=None, daytime=[6,21], workdays=[0,5]):
    """
    This is just one possible implementation which had been suitable for my application
    where it was okay to use two licences on weekdays and 8 at nicght and over the weekend.
    For you it might look completely different, it might get so simple as to return just a constant.
    So why not erase the function? Simply because it is used in two places, eval_all_threaded() and
    eval_bunch_threaded(), and one easily forgets to modify the other one too after having changed one of them.
    """
    if low_n is None: low_n=2
    if desired_n is None: desired_n=8
    tm=localtime(time())
    n=low_n
    if (tm[6] in range(6,21)) and (tm[3] in range(0,5)):
        n=low_n
    else:
        n=desired_n
    return n



