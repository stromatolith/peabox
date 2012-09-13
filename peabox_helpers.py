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

