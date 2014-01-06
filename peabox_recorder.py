#!python
"""
peabox - python evolutionary algorithm toolbox
by Markus Stokmaier, IKET, KIT, Karlsruhe, August 2012

The recorder classes in this file are designed to keep track of population histories.
"""

from os.path import join
from cPickle import Pickler
from copy import copy

import numpy as np

class Recorder:
    
    def __init__(self,population):
        self.p=population
        self.gg=[]
        self.snames=[]
        self.anames=['DNAs','scores','ancestcodes']
        self.goaldictnames=['goal']
        self.scmd={}
        self.acmd={'DNAs':'get_DNAs()', 'scores':'get_scores()', 'ancestcodes':'get_ancestcodes()'}
        self.reinitialize_data_dictionaries()
        self.ownname='rec'
        self.goal={'goalvalue':0,'fulfilltime':-1, 'fulfillcalls':-1}   # you can define a convergence criterium goalvalue and note down the generation when there was the first Individual getting the corresponding score
        
    def reinitialize_data_dictionaries(self):
        self.sdat={}               # scalar data
        self.adat={}               # array data
        self.gddat={}              # goal dictionaries and other dictionaries for data summarising an evolution run
        for name in self.snames:
            self.sdat[name]=[]
        for name in self.anames:
            self.adat[name]=[]
            
    def save_status(self):
        self.gg.append(self.p.gg)
        for name in self.snames:
            if self.scmd.has_key(name):
                cmd=self.scmd[name]
            else:
                cmd=name
            if name.startswith('['):
                self.sdat[name].append(eval('self.p'+cmd))
            else:
                self.sdat[name].append(eval('self.p.'+cmd))
        for name in self.anames:
            if self.acmd.has_key(name):
                cmd=self.acmd[name]
            else:
                cmd=name
            if cmd.startswith('get'):
                self.adat[name].append(eval('self.p.'+cmd))
            elif cmd.startswith('['):
                self.adat[name].append(eval('self.p'+cmd))
            elif cmd.startswith('dude'):
                data=np.zeros(self.p.psize)
                for i,dude in enumerate(self.p):
                    data[i]=eval(cmd)
                self.adat[name].append(data)
            else:
                self.adat[name].append(eval('array(self.p.'+cmd+',copy=1)'))
        self.check_and_note_goal_fulfillment()

    def set_goalvalue(self,value,reset=True):
        self.goal['goalvalue']=value
        if reset:
            self.goal['fulfilltime']=-1; self.goal['fulfillcalls']=-1

    def check_and_note_goal_fulfillment(self):
        sc=self.p.get_scores()
        factor=1.
        if self.p.whatisfit=='maximize': factor=-1.
        if self.goal['fulfilltime']==-1 and np.min(factor*sc) <= factor*self.goal['goalvalue']:
            self.goal['fulfilltime']=self.p.gg
            self.goal['fulfillcalls']=self.p.neval

    def clear(self):
        self.gg=[]
        self.reinitialize_data_dictionaries()
        self.goal['fulfilltime']=-1
        self.goal['fulfillcalls']=-1

    def pickle_self(self):
        ofile=open(join(self.p.picklepath,self.ownname+'_'+self.p.label+'.txt'), 'w')
        einmachglas=Pickler(ofile)
        einmachglas.dump(self)
        ofile.close()
        
        

class MORecorder(Recorder):
    """for a MOPopulation instance"""
    def __init__(self,population):
        self.p=population
        self.gg=[]
        self.snames=[]
        self.anames=['DNAs','scores','objvals','sumcoeffs','ranks','oaranks','ancestcodes','pe','po','pk']
        self.goaldictnames=['goal']
        self.scmd={}
        self.acmd={'DNAs':'get_DNAs()', 'scores':'get_scores()', 'objvals':'get_objvals()', 'sumcoeffs':'sumcoeffs',
                   'ranks':'get_ranks()', 'rankweights':'rankweights', 'oaranks':'get_overall_ranks()',
                   'ancestcodes':'get_ancestcodes()', 'pe':'dude.paretoefficient', 'po':'dude.paretooptimal', 'pk':'dude.paretoking'}
        self.reinitialize_data_dictionaries()
        self.ownname='MOrec'    
        self.save_goalstatus()



class wTORecorder(MORecorder):
    """for a wTOPopulations instance"""
    def __init__(self,population):
        MORecorder.__init__(self,population)
        self.snames+=['optsx','optrx']
        self.goaldictnames+=['optw_successrates']
        self.acmd['optw_successrates']='optw_successrates'
        self.reinitialize_data_dictionaries()
        self.ownname='wTOrec' 