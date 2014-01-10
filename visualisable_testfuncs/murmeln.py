#!python
"""
-------------------------------------------------------------------------------
SPOILER WARNING: If you want to make a similar invention yourself, then close
this file again and don't use it! let yourself inspire by the invention task
description in the readme.md of the visualisable_testfuncs folder instead.
-------------------------------------------------------------------------------

The class "murmeln" is the visualisable test problem I presented at CEC-2013
in this paper:

DOI: 10.1109/CEC.2013.6557837
http://dx.doi.org/10.1109/CEC.2013.6557837
http://ieeexplore.ieee.org/xpl/articleDetails.jsp?tp=&arnumber=6557837&queryText%3Dstokmaier

The problems "necklace" and "hilly" are the simpler problems leading step by
step to the harder one. "twotrack" is a complication suggestion for "murmeln".
More complication suggestions can be found in the paper ... or preferrably
don't yet read the paper and just start inventing yourself after having looked
at and understood only the simple necklace problem - how would you make it more
nasty? Try to come up with better ideas than I did.

Markus Stokmaier, IKET, KIT, Karlsruhe, January 2014
"""
import numpy as np
from numpy import array, arange, zeros, ones, where, sort, roll, mean, std
from numpy import exp, pi, sin, cos
from numpy.random import rand, randn, randint
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge

from peabox_individual import Individual
from peabox_plotting import murmelfarbe





class murmeln(Individual):
    # looking for an easy to calculate optimisation problem having two features:
    #  - local optima
    #  - there must be a way to plot it so one can see at a glance whether it is a good individual or a bad one
    # idea: modify the above problem (even distribution of points on a circle line) so that score, which has to be
    # minimised consists of those two things:
    #  - repulsive potential of neighbours, imagine the particles have the same electric charge
    #  - the circle track has a potential itself, marbles do not want to be on hilltops
    #  - if you really want to have just one ideal solution, give the marbles different weight -> preferring heaviest marble in lowest valley
    def __init__(self,paramspace,ncase):
        Individual.__init__(self,paramspace,ncase)
        self.weights=ones(self.ng)-arange(self.ng,dtype=float)*0.8/float(self.ng-1)
        self.radii=0.2*(0.75*self.weights/pi)**0.333
    def evaluate(self,tlock=None):
        if tlock is not None: raise NotImplementedError('threaded evaluation not implemented')
        # first step: evaluate potential energy of neighbours
        cDNA=array(self.DNA,copy=1)
        if np.any(cDNA<0) or np.any(cDNA>360):
            self.score=3000.
            return self.score
        cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        if 0. in diff:
            E_neighbour=2000.
            print 'dude {} in generation {} has 0 in diff'.format(self.no,self.gg)
            print diff
        else:
            E_neighbour=sum(1./diff)
        # second step: evaluate potential energy of each marble along the hilly circle track
        E_pot=0.
        for i,marble in enumerate(self.DNA):
            E_pot+=self.trackpotential(marble,self.weights[i])
        self.score=E_neighbour+2.5*E_pot   #-80.
        return self.score
    def trackpotential(self,phi,mass):
        A=0.6; B=1.0; C=0.8     # coefficients for angular frequencies
        o1=0; o2=40.; o3=10.   # angular offsets
        e_pot=mass*(A*(sin(2*pi*(phi+o1)/360.)+1)+B*(sin(4*pi*(phi+o2)/360.)+1)+C*(sin(8*pi*(phi+o3)/360.)+1))
        return e_pot
    def rather_good_DNA(self,sigma=5,a1=0):
        arr=arange(self.ng,dtype='float')/float(self.ng)*360+sigma*randn(self.ng)+a1
        arr=where(arr>360,arr-360,arr)
        arr=where(arr<0,arr+360,arr)
        self.set_DNA(arr)
    def draw_geometry(self,a):
        # draw own look into a, which is a matplotlib axes/subplot instance
        p=PatchCollection([Wedge((0.,0.), 1.2, phi-3,phi+3, width=0.5) for phi in range(2,360,5)],cmap=plt.cm.bone,edgecolor='white',linewidths=0,zorder=1)
        p.set_array(self.trackpotential(arange(2,360,5),1))
        a.add_collection(p)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.2*(0.75*self.weights[i]/pi)**0.333) for i,phi in enumerate(self.DNA)],
                          cmap=murmelfarbe,zorder=3)
        c.set_array(arange(self.ng)/(self.ng-1.))
        a.add_collection(c)
        #a.scatter(cos(phi),sin(phi),marker='o',s=100,c=arange(self.ng),cmap=plt.cm.jet,zorder=3)
        #a.axis([-1.2,1.2,-1.2,1.2],'equal'
        a.axis('equal')
        atxt='generation: {0}\nscore: {1:.3f}'.format(self.gg,self.score)
        a.text(0,0,atxt, ha='center',va='center')   #, fontsize=8)
    def set_bad_score(self):
        self.score=3000.

class necklace(Individual):
    def evaluate(self,tlock=None):
        if tlock is not None: raise NotImplementedError('threaded evaluation not implemented')
        cDNA=array(self.DNA,copy=1); cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        self.score=std(diff)
        return self.score
    def draw_geometry(self,a):
        # draw own look into a, which is a matplotlib axes/subplot instance
        p=PatchCollection([Wedge((0.,0.), 1.0, phi-3,phi+3, width=0.035) for phi in range(2,360,5)],
                           facecolor='k',edgecolor='k',linewidths=0,zorder=1)
        a.add_collection(p)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.12) for i,phi in enumerate(self.DNA)],
                          facecolor='c',edgecolor='k',linewidths=2,zorder=3)
        a.add_collection(c)
        a.axis('equal')
        plt.axis('off')
    def set_bad_score(self):
        self.score=100000.

class hilly(Individual):
    def evaluate(self,tlock=None):
        if tlock is not None: raise NotImplementedError('threaded evaluation not implemented')
        # first step: evaluate potential energy of neighbours
        cDNA=array(self.DNA,copy=1); cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        if 0. in diff:
            E_neighbour=2000.
            print 'dude {} in generation {} has 0 in diff'.format(self.no,self.gg)
            print diff
        else:
            E_neighbour=sum(1./diff)
        # second step: evaluate potential energy of each marble along the hilly circle track
        E_pot=0.
        for i,marble in enumerate(self.DNA):
            E_pot+=self.trackpotential(marble)
        self.score=E_neighbour+8.*E_pot   #-80.
        return self.score
    def trackpotential(self,phi):
        A=0.6; B=1.0; C=0.8     # coefficients for angular frequencies
        o1=0; o2=40.; o3=10.   # angular offsets
        e_pot=A*(sin(2*pi*(phi+o1)/360.)+1)+B*(sin(4*pi*(phi+o2)/360.)+1)+C*(sin(8*pi*(phi+o3)/360.)+1)
        return e_pot
    def draw_geometry(self,a):
        # draw own look into a, which is a matplotlib axes/subplot instance
        p=PatchCollection([Wedge((0.,0.), 1.2, phi-3,phi+3, width=0.5) for phi in range(2,360,5)],cmap=plt.cm.bone,edgecolor='white',linewidths=0,zorder=1)
        p.set_array(self.trackpotential(arange(2,360,5)))
        a.add_collection(p)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.1) for i,phi in enumerate(self.DNA)],
                          facecolor='c',edgecolor='k',linewidths=1.5,zorder=3)
        a.add_collection(c)
        a.axis('equal')
        plt.axis('off')
        atxt='generation: {0}\nscore: {1:.3f}'.format(self.gg,self.score)
        a.text(0,0,atxt, ha='center',va='center')   #, fontsize=8)
    def set_bad_score(self):
        self.score=3000.


class twotrack(Individual):
    def __init__(self,paramspace,ncase):
        Individual.__init__(self,paramspace,ncase)
        self.weights=ones(self.ng)-arange(self.ng,dtype=float)*0.8/float(self.ng-1)
        self.radii=0.2*(0.75*self.weights/pi)**0.333
    def evaluate(self,tlock=None):
        if tlock is not None: raise NotImplementedError('threaded evaluation not implemented')
        # first step: evaluate potential energy of neighbours
        cDNA=array(self.DNA,copy=1); cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        if 0. in diff:
            E_neighbour=2000.
            print 'dude {} in generation {} has 0 in diff'.format(self.no,self.gg)
            print diff
        else:
            E_neighbour=sum(1./diff)
        # second step: evaluate potential energy of each marble along the hilly circle track
        E_pot=0.
        for i,marble in enumerate(self.DNA):
            if np.mod(i,2)==0:
                E_pot+=self.innertrack(marble,self.weights[i])
            else:
                E_pot+=self.outertrack(marble,self.weights[i])
        self.score=0.1*E_neighbour+8.*E_pot   #-80.
        return self.score
    def innertrack(self,phi,mass):
        A=0.6; B=1.0; C=0.8     # coefficients for angular frequencies
        o1=0; o2=40.; o3=10.   # angular offsets
        e_pot=mass*(A*(sin(2*pi*(phi+o1)/360.)+1)+B*(sin(4*pi*(phi+o2)/360.)+1)+C*(sin(8*pi*(phi+o3)/360.)+1))
        return e_pot
    def outertrack(self,phi,mass):
        A=1.2; B=2.0
        w=2.8 # frequency omega
        fade=phi/360.
        e_pot=mass*(A*(1-fade)*exp(-fade) + B*fade*sin(w*2*pi*phi/360.))+2.4
        return e_pot
    def rather_good_DNA(self,sigma=5,a1=0):
        arr=arange(self.ng,dtype='float')/float(self.ng)*360+sigma*randn(self.ng)+a1
        arr=where(arr>360,arr-360,arr)
        arr=where(arr<0,arr+360,arr)
        self.set_DNA(arr)
    def draw_geometry(self,a):
        # inner ring
        pin=PatchCollection([Wedge((0.,0.), 1.25, phi-3,phi+3, width=0.3) for phi in range(2,360,5)],cmap=plt.cm.bone,edgecolor='white',linewidths=0,zorder=1)
        pin.set_array(self.innertrack(arange(2,360,5),1))
        a.add_collection(pin)
        # outer ring
        po=PatchCollection([Wedge((0.,0.), 1.6, phi-3,phi+3, width=0.3) for phi in range(2,360,5)],cmap=plt.cm.bone,edgecolor='white',linewidths=0,zorder=1)
        po.set_array(self.outertrack(arange(2,360,5),1))
        a.add_collection(po)
        # marbles
        radii=1.15+0.25*where(np.mod(arange(self.ng),2)==0,0.,1.)
        c=PatchCollection([Circle((radii[i]*cos(2.*pi*phi/360.),radii[i]*sin(2.*pi*phi/360.)),0.13*(0.75*self.weights[i]/pi)**0.333) for i,phi in enumerate(self.DNA)],
                          cmap=murmelfarbe,zorder=3)
        c.set_array(arange(self.ng)/(self.ng-1.))
        a.add_collection(c)
        #a.scatter(cos(phi),sin(phi),marker='o',s=100,c=arange(self.ng),cmap=plt.cm.jet,zorder=3)
        #a.axis([-1.2,1.2,-1.2,1.2],'equal'
        a.axis('equal')
        atxt='generation: {0}\nscore: {1:.3f}'.format(self.gg,self.score)
        a.text(0,0,atxt, ha='center',va='center')   #, fontsize=8)
    def set_bad_score(self):
        self.score=3000.

        
def dummyfunc(x):
    return np.sum(x)