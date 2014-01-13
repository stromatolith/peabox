#!python
"""
run an EA 51 times and use a ceclog instance to record the data for a CEC-2013 competition on real-parameter optimization
Markus Stokmaier, IKET, KIT, Karlsruhe, January 2014
"""
#import sys   # in case you wanna call this from outside, e.g. by a shell script going automatically through all the test functions
from os import getcwd
from os.path import join
from time import clock
from numpy.random import randint, seed
from peabox_population import cecPop
from EAcombos import ComboB_DeLuxe
from ceclog import cec_data_logger

func_num=19           #int(float(sys.argv[1]))   # in case you wanna call this from outside, e.g. by a shell script going automatically through all the test functions
ncase=func_num
runs=range(5)  #range(51)
ps=80
ndim=10

pa0=cecPop(ps,func_num,ndim,evaluatable=True)      # F0 generation for EA (parents)
pa1=cecPop(ps,func_num,ndim,evaluatable=False)     # F1 generation for EA (offspring)
pb0=cecPop(ndim+1,func_num,ndim,evaluatable=True)  # Nelder-Mead: the simplex
pb1=cecPop(1,func_num,ndim,evaluatable=True)       # Nelder-Mead: the trials
plist=[pa0,pa1,pb0,pb1]

for p in plist:
    p.set_ncase(ncase)
    p.set_subcase(0)
ea=ComboB_DeLuxe(pa0,pa1,pb0,pb1)       # instanciate the algorithm from library
cdl=cec_data_logger(ea)
cdl.set_logtarget(pa0)
ea.generation_callbacks.append(cdl.gcb)

def ftol_stopper(eaobj):
    bval=eaobj.tell_best_score()
    if (bval-cdl.goalval <= 1e-9):   # beyond that limit it doesn't count anyway in the CEC-2013 competition
        return True
    else:
        return False

ea.more_stop_crits.append(ftol_stopper)

toteatime=0
loc=getcwd()
runlog=open(join(loc,'logs',ea.ownname+'_runlog_c'+str(ncase)+'.txt'),'w')

saat=randint(1000)
runlog.write('seeding with {}\n'.format(saat))   # remember that so you can prove later that you weren't making things up
seed(saat)
for i in runs:
    t0=clock()
    if i!=runs[0]:
        for p in plist:
            p.reset()
            p.next_subcase()
        ea.bestdude=None
    cdl.start()
    ea.set_bunchsizes(ea.bunching()); ea.make_bunchlists(); #print 'bunch sizes: ',ea.bunchsizes
    ea.cec_poor_man_run_with_NM()
    cdl.early_finish()
    cdl.reset_memory()
    t1=clock()

    eatime=t1-t0
    atxt= 'score {0:.3f} and error {1} after {2} calls and {3:.3f} seconds'.format(ea.tell_best_score(),cdl.return_errval(),ea.tell_neval(),eatime)
    print atxt
    runlog.write(atxt+'\n')
    toteatime+=eatime
    
runlog.write('at the end: {0} runs completed in {1} seconds\n'.format(i, toteatime))
runlog.close()

