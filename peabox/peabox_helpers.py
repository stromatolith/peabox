#!python
"""
peabox - python evolutionary algorithm toolbox
by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

from os import getcwd
from os.path import join
from cPickle import Pickler, Unpickler
#from copy import copy
from time import time, localtime

import numpy as np
from numpy import sqrt, pi, sin, cos, exp, log, ceil, floor, where
from numpy import asarray, array, arange, asfarray, linspace, zeros, dot, prod, roll
from numpy.random import rand, randn, randint


def simple_searchspace(dim,l_bound,u_bound):
    searchspace=[]
    for i in range(dim):
        searchspace.append(eval("tuple(['par"+str(i)+"',"+str(l_bound)+","+str(u_bound)+"])"))
    return tuple(searchspace)

def parentselect_exp(N,selp,maxiter=3,size=1):
    """
    Parent choice function for simulating selection pressure of exponential shape.
    The function returns numbers from zero to N-1 and the probability to get i,
    P(i), decreases exponentially with i. The probability ratio P(N-1)/P(0) does
    not depend on N.
    
    This function returns one number in [0,...,N-1] (or an array of such numbers
    if keyword argument size>1) that help you select from N parents in such a
    way that a selection pressure of exponential shape is applied.
    Argument selp sets the selection pressure:
    0 < selp << 1 --> flat
    1 < selp < 10 --> reasonable values
    selp > 10 --> very steep, almost only 0 will be returned
    for a script to get nice statistics see source code below function definition
    
    about keyword argument maxiter: it is only influential for selp < 5, where
    higher maxiter increases fidelity to exponential distribution because it delays
    resorting to flat randint(N) when repeatedly guessing numbers >= N; I guess in
    most cases the default setting will be okay, because if the pure exponential
    distribution is pretty flat already, its dilution by the randint contribution
    doesn't hurt much anyway.
    """
    if size==1:
        # returns a number in [0,1,...,N-1] suitable for parent selection under exponential selection pressure
        for i in range(maxiter):
            parent=int(floor(-(float(N)/float(selp))*log(rand(1))))    # select a parent number favoring lower numbers by a sort of Boltzmann distribution
            if parent < N: break
        if parent >= N : parent=randint(N)
        return parent
    else:
        # returns an array of numbers in [0,1,...,N-1] suitable for parent selection under exponential selection pressure
        parents=asarray(floor(-(float(N)/float(selp))*log(rand(size))),dtype=int)
        if np.sum(where(parents>=N,1,0))!=0:
            for i in range(maxiter):
                moreparents=asarray(floor(-(float(N)/float(selp))*log(rand(size))),dtype=int)    # select a parents number favoring lower numbers by a sort of Boltzmann distribution
                parents=where(parents>=N,moreparents,parents)
                if np.sum(where(parents>=N,1,0))==0: break
        if np.sum(where(parents>=N,1,0))!=0: parents=where(parents>=N,randint(N,size=size),parents)
        return parents

#--- plot code
"""
# and here a script to show statistics on return values of parentselect_exp()
# in particular it shows that the ratio of probabilities of choosing best and
# worst (i.e. probabilities of return values 0 and N-1) does not depend on the
# population size N

from pylab import *
from peabox_helpers import parentselect_exp

selp=2.
psizes=[4,8,12,20,40]  #,80,160,400]
hists=[]
nhists=[]
binmids=[]
xdats=[]
for ps in psizes:
    N=600*ps
    dat=parentselect_exp(ps,selp,size=N)
    bins=arange(ps+1)-0.5
    hist,edges=np.histogram(dat,bins=bins)
    hists.append(hist)
    nhists.append(asfarray(hist)/hist[0])  #normed
    binmids.append(arange(ps))
    xdats.append(arange(ps,dtype=float)/ps)

nsets=len(psizes)
myc=[cm.jet(float(i)/(nsets-1)) for i in range(nsets)]
for i in range(nsets):
    plt.plot(xdats[i],nhists[i],color=myc[i],lw=2)
plt.xlabel('$i/N$')
plt.ylabel('$P(i) / P(0)$')
plt.title('exponential parent choice distribution parentselect_exp()')
plt.show()

# you should be able to reproduce these statistics:
# probabilities of N/2 and N-1 in comparison to probability of getting 0
# selp    P(N/2)/P(0)  P(N-1)/P(0)
# 0.1       0.98          0.95
# 0.5       0.83          0.7
# 1.0       0.6           0.4
# 2.0       0.4           0.15
# 4.0       0.13          0.03
# 6.0       0.05          0.0
# 10.0      0.0           0.0  (only best 20% have a chance higher than 0.1*P_best)
"""


def diverse_selection_from(population,howmany,mindist=0.1,uCS=True,iniselp=4.):
    """
    This function tries hard to avoid similar DNAs, it will choose low quality
    individuals if this concession is needed to ensure sufficient distance
    between the DNA vectors, and it can be wasteful with resources.
    """
    # first approach: if the first couple members are not similiar, return their numbers
    assert len(population)>howmany
    nums=[0]
    for i in range(1,howmany):
        d=[population[i].distance_from(population[j],uCS=uCS) for j in nums]
        if np.min(d) > mindist:
            nums.append(i)
    if len(nums)==howmany:
        return nums  # first approach was enough
    # second approach: fill up list using parentselect_exp and checking for enough distance everytime
    maxtrials=10*howmany
    for reduc in [1., 0.5, 0.25]: # for lowering the minimal distance requirement
        count=0
        relax_interval=int(ceil(maxtrials/iniselp))
        selp=iniselp  #selection pressure
        while count<=maxtrials:
            count+=1
            num=parentselect_exp(len(population),selp)
            d=[population[num].distance_from(population[j],uCS=uCS) for j in nums]
            if np.min(d) > mindist*reduc:
                nums.append(num)
            if np.mod(count,relax_interval)==0:
                selp-=1. # relax selection pressure (at a rate so it reaches the bottom approximately when maxtrials are used up)
                selp=max(selp,1.) # ensure it doesn't drop below a reasonable limit
            if len(nums)==howmany:
                return nums
    # third approach (there's still no complete list): we don't care no more about diversity
    # we only still care about nonidentity
    count=0
    while count<=maxtrials:
        count+=5
        r=randint(population.psize,size=5)
        for trial in r:
            if trial not in nums:
                nums.append(trial)
            if len(nums)==howmany:
                return nums
    # if at this point still inside the function, should there be a fourth approach?
    # Just to make the function really robust, so it also returns a list when forced to include duplicate numbers?
    # I think this would be no good idea, because that silence may prevent you from discovering a coding error.
    # It would let you sleep well thinking you are working with diverse sets even if it is not the case.
    raise ValueError('Too many trials; this function gives up; it could not ensure a set without duplicates.')


def less_aggressive_diverse_selection_from(population,howmany,mindist=0.1,uCS=True,selp=4.):
    """
    This function initially also cares about DNA diversity and selection pressure, but it
    relaxes quickly on the DNA distance requirements while keeping up selection pressure.
    """
    # first approach: if the first couple members are not similiar, return their numbers
    assert len(population)>howmany
    nums=[0]
    for i in range(1,howmany):
        d=[population[i].distance_from(population[j],uCS=uCS) for j in nums]
        if np.min(d) > mindist:
            nums.append(i)
    if len(nums)==howmany:
        return nums  # first approach was enough
    # second approach: fill up list using parentselect_exp and checking for enough distance everytime
    maxtrials=howmany
    for reduc in [1., 0.5, 0.25, 0.1]: # for lowering the minimal distance requirement
        count=0
        while count<=maxtrials:
            count+=1
            num=parentselect_exp(len(population),selp)
            d=[population[num].distance_from(population[j],uCS=uCS) for j in nums]
            if np.min(d) > mindist*reduc:
                nums.append(num)
            if len(nums)==howmany:
                return nums
    # third approach: fill up ignoring duplicate numbers
    missing=howmany-len(nums)
    addins=parentselect_exp(len(population),selp,size=missing)
    if type(addins) in [int,float]:
        return nums+[addins]
    else:
        return nums+list(addins)

#--- plot code
'''
If you want to get a feeling of how the above selection routines work out,
find below a code producing histograms of:
    a) numbers of the chosen individuals
    b) scores of the chosen individuals
    c) distances between the chosen individual's DNAs



#!python
"""
plot distributions coming out of the routines for diverese parent selection
data to plot:
    a) distribution of scores among selected relative to scores available
    b) distribution of distances among the selected showing how often
       the threshold for minimal distance needed to be breached
"""
from os import getcwd
from os.path import join
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros
from peabox_helpers import parentselect_exp as psel
from peabox_helpers import diverse_selection_from as divsel
from peabox_helpers import less_aggressive_diverse_selection_from as ladivsel
from peabox_population import cecPop
from EAcombos import ComboB

def scale(x,mn,mx):
    return (x-mn)/(mx-mn)

def distances(p,nl):
    dl=[]
    N=len(nl)
    for i in range(N):
        for j in range(i+1,N):
            dl.append(p[nl[i]].distance_from(p[nl[j]],uCS=False))
    return dl

func_num=9
ncase=func_num
ndim=5
G=20


psizes=[20,50,100,100,100,100]
bunches=[4,10,10,25,50,70]
eabunches=[[2,4,4,5,0,5],[4,11,11,12,0,12]]+4*[[4,24,24,24,0,24]]
jobs=len(psizes)
rounds=180
loc=getcwd()

for i in range(jobs):
    ps=psizes[i]
    bs=eabunches[i]
    nsel=bunches[i]
    sel1=[]; sel2=[]; sel3=[]
    d1=[]; d2=[]; d3=[]
    s1=[]; s2=[]; s3=[]
    for j in range(rounds):
        pa0=cecPop(ps,func_num,ndim,evaluatable=True)
        pa1=cecPop(ps,func_num,ndim,evaluatable=False)
        ea=ComboB(pa0, pa1)
        ea.set_bunchsizes(bs)
        ea.make_bunchlists()
        ea.anneal=0.06
        ea.simple_run(G,mstep=0.1)
        ms=ea.mstep
        sc=pa0.get_scores(); minsc=np.min(sc); maxsc=np.max(sc)
        scs=scale(sc,minsc,maxsc)
        
        nums1=psel(ps,4.,size=nsel)
        nums2=divsel(pa0,nsel,mindist=3*ms)
        nums3=ladivsel(pa0,nsel,mindist=3*ms)
        
        sel1+=list(nums1)
        sel2+=list(nums2)
        sel3+=list(nums3)
        
        s1+=[scs[k] for k in nums1]
        s2+=[scs[k] for k in nums2]
        s3+=[scs[k] for k in nums3]
        
        d1+=distances(pa0,nums1)
        d2+=distances(pa0,nums2)
        d3+=distances(pa0,nums3)
    
    dat=array([sel1,sel2,sel3]).T
    clist=['DarkBlue','darkRed','DarkCyan']
    labs=['psel','divsel','ladivsel']
    plt.hist(dat, 20, normed=0, histtype='bar', color=clist, label=labs)
    plt.xlabel('ranking')
    plt.ylabel('how often selected')
    plt.title('showing parent numbers\nwhen selecting {} from {}'.format(nsel,ps))
    plt.legend()
    plt.savefig(join(loc,'plots','nums_ps'+str(ps).zfill(3)+'_nsel'+str(nsel).zfill(2)+'.png'))
    plt.close()

    dat=array([s1,s2,s3]).T
    clist=['DarkBlue','darkRed','DarkCyan']
    labs=['psel','divsel','ladivsel']
    plt.hist(dat, 50, normed=0, histtype='bar', color=clist, label=labs)
    plt.xlabel('score')
    plt.ylabel('how often selected')
    plt.title('showing scores\nwhen selecting {} from {}'.format(nsel,ps))
    plt.legend()
    plt.savefig(join(loc,'plots','scores_ps'+str(ps).zfill(3)+'_nsel'+str(nsel).zfill(2)+'.png'))
    plt.close()

    dat=array([d1,d2,d3]).T
    clist=['DarkBlue','darkRed','DarkCyan']
    labs=['psel','divsel','ladivsel']
    plt.hist(dat, 50, normed=0, histtype='bar', color=clist, label=labs)
    plt.xlabel('distances')
    plt.ylabel('how often occuring')
    plt.title('showing occuring DNA distances\nwhen selecting {} from {}'.format(nsel,ps))
    plt.legend()
    plt.axvline(ms*pa0[0].widths[0])   # mstep
    plt.axvline(3*ms*pa0[0].widths[0]) # minimally required distance
    plt.savefig(join(loc,'plots','distances_ps'+str(ps).zfill(3)+'_nsel'+str(nsel).zfill(2)+'.png'))
    plt.close()
'''

def condense(many_p, one_p, selp=2.2, elitesize=None):
    N=len(many_p); q=one_p.psize/N
    if elitesize is None: elitesize=max(1,many_p[0].psize/20)
    rest=one_p.psize
    for n,p in enumerate(many_p):
        print 'now copying from ',p.ownname
        for i,dude in enumerate(one_p[n*q:(n+1)*q]):
            if i < elitesize:
                dude.copy_DNA_of(p[i],copyscore=True,copyancestcode=False,copyparents=True)
                print '{} dude {} gets new DNA from {} elite dude {} with score {}'.format(one_p.ownname,dude.no,p.ownname,p[i].no,p[i].score)
            else:
                choice=parentselect_exp(p.psize,selp)
                dude.copy_DNA_of(p[choice],copyscore=True,copyancestcode=False,copyparents=True)
                print '{} dude {} gets new DNA from {} random dude {} with score {}'.format(one_p.ownname,dude.no,p.ownname,p[choice].no,p[choice].score)
            dude.ancestcode=0.48+n*0.1; dude.ancestcode-=floor(dude.ancestcode) # ensures value is in [0,1]
            rest-=1
    if rest > 0:
        for i in range(rest):
            pidx=randint(N)
            one_p[-i-1].copy_DNA_of(many_p[pidx][parentselect_exp(many_p[pidx].psize,selp)],copyscore=True,copyancestcode=False,copyparents=True)
        



