#!python
"""
data logger class as utility for
benchmarking of myEA on the CEC-2013 test function suite
"""
from os.path import join  #, basename, dirname
from cPickle import Pickler  #, Unpickler
from numpy import ceil  #, array
#from pylab import plt
from EAcombos import ComboBase, ComboC_Base, ComboD
#from nhansens_cmaes import CMAES

class cec_data_logger:
    def __init__(self,eaobj):
        self.ea=eaobj
        if isinstance(eaobj,ComboD):
            self.p=eaobj.F0b
        else:
            self.p=eaobj.F0
        self.thrcount=0
        self.thr=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        if self.p.func_num <= 14:
            self.goalval=-1500+self.p.func_num*100.
        else:
            self.goalval=-1400+self.p.func_num*100.
        self.memory={'g':[], 'DNA':[], 'score':[], 'neval':[]}
    def start(self):
        path=join(self.p.path,'logs')
        self.p.update_label()
        fext='.txt'
        fn=self.ea.ownname+'_cecLog_'+self.p.label[:-5]
        self.outf=open(join(path,fn+fext),'w')
        text='function number {0}, {1} run with population size {2}'.format(self.p.func_num,self.ea.ownname,self.p.psize)
        text+='\nassumed goalvalue: {0} (errorvalue calculated correspondingly)\n\n\n'.format(self.goalval)
        self.outf.write(text)
        fns=fn+'_short'
        self.outfs=open(join(path,fns+fext),'w')
        self.thrcount=0
    def gcb(self,eaobj):
        if self.thrcount>=len(self.thr):
            pass
        elif self.ea.tell_neval()>=self.thr[self.thrcount]*self.ea.maxeval:
            #print 'gcb: ',self.ea.tell_neval(),self.thr[self.thrcount]
            self.logline()
            self.thrcount+=1
            self.add_memory_entry()
    def logline(self):
        errval=self.ea.tell_best_score()-self.goalval
        text='{0}*maxFES, i.e. neval={1}: error={2}\n'.format(self.thr[self.thrcount],self.ea.tell_neval(),errval)
        texts='{0}   {1}   {2}\n'.format(self.thr[self.thrcount],self.ea.tell_neval(),errval)
        self.outf.write(text)
        self.outfs.write(texts)
    def add_memory_entry(self,fakeneval=None):
        score, dna = self.ea.tell_best_score(andDNA=True)
        self.memory['g'].append(self.p.gg)
        self.memory['DNA'].append(dna)
        self.memory['score'].append(score)
        if fakeneval is not None:
            self.memory['neval'].append(fakeneval)
        else:
            self.memory['neval'].append(self.ea.tell_neval())
    def return_errval(self):
        return self.ea.tell_best_score()-self.goalval
    def pickle_the_memory(self):
        outfile=open(join(self.p.datapath,self.ea.ownname+'_cecLog_memory_'+self.p.label[:-5]+'.txt'),'w')
        einmachglas=Pickler(outfile)
        einmachglas.dump(self.memory)
        outfile.close()
    def finish(self):
        self.outf.close()
        self.outfs.close()
        self.pickle_the_memory()
    def reset_memory(self):
        self.memory={'g':[], 'DNA':[], 'score':[], 'neval':[]}
    def early_finish(self):
        errval=self.ea.tell_best_score()-self.goalval
        while self.thrcount < len(self.thr):
            fakeneval=int(ceil(self.thr[self.thrcount]*self.ea.maxeval))
            text='{0}*maxFES, i.e. neval={1}: error={2}\n'.format(self.thr[self.thrcount],fakeneval,errval)
            texts='{0}   {1}   {2}\n'.format(self.thr[self.thrcount],fakeneval,errval)
            self.outf.write(text)
            self.outfs.write(texts)
            self.thrcount+=1
            self.add_memory_entry(fakeneval=fakeneval)
        self.finish()
        

#def plot_cdl_memory(filepath,offset=0.):
#    infile=open(filepath,'r')
#    einmachglas=Unpickler(infile)
#    thedict=einmachglas.load()
#    infile.close()
#    g=thedict['g']
#    dna=thedict['DNA']
#    sc=thedict['score']
#    nev=thedict['neval']
#    plt.plot(nev,array(sc)+offset,'bo')
#    plt.semilogy()
#    plt.title('final score {} after {} evaluations'.format(sc[-1],nev[-1]))
#    plotloc=join(dirname(dirname(filepath)),'plots')
#    nametrunk=basename(filepath).split('.')[0]
#    plt.savefig(join(plotloc,nametrunk+'.png'))
#    plt.close()
