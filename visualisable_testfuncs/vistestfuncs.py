#!python
"""
peabox - python evolutionary algorithm toolbox
by Markus Stokmaier, IKET, KIT, Karlsruhe, June 213

This file hosts some test functions with the property of offering an intuitive
solution visualisation.
"""

from os import getcwd
from os.path import join
from time import time
import numpy as np
import numpy.random as npr
from numpy import sqrt, pi, sin, cos, exp, log, ceil, where, sort, mean, std
from numpy import array, arange, asfarray, linspace, ones, zeros, diag, dot, prod, roll
from numpy import transpose, append
from scipy import r_
import matplotlib.pyplot as plt

import getfem as gf

from peabox_individual import Individual
from peabox_plotting import murmelfarbe, blue_red4

class FMsynth(Individual):
    # we are inheriting everything an individual has or can do and are just adding more methods
    # the only two methods below, where to pay a little bit attention are the constructor and evaluate()
    # these two are not added but overwritten, so one has to be aware of the functionality of the originals
    # in __init__(self) the overwriting problem is solved by calling the original next to executing the replacement
    # in evaluate(self) the problem is solved by creating the useless slot for the second argument
    def __init__(self,dummyfunc,pspace):
        Individual.__init__(self,dummyfunc,pspace)
        self.a=[1., 1.5, 2.]
        self.w=[5. ,4.8, 4.9]
        self.nt=101
        self.t=arange(self.nt,dtype=float)
        self.theta=2*pi/(self.nt-1)
        self.trial=zeros(self.nt)
        self.target=zeros(self.nt)
        self.initialize_target()
        self.plotpath=join(getcwd(),'plots')
        self.kline=None
        self.gline=None
        self.fb=None
    def initialize_target(self):
        self.fmwave(self.a,self.w)
        self.target[:]=self.trial[:]
    def fmwave(self,a,w):
        t=self.t; th=self.theta
        self.trial[:]=a[0]*sin(w[0]*t*th+a[1]*sin(w[1]*t*th+a[2]*sin(w[2]*t*th)))
    def evaluate(self,tlock=None):
        if tlock is not None: # in case a threading lock is handed over, omg, what the hell is threading??
            raise ValueError('this individual cannot handle threaded processing') # problem solved, no need to know what threading is a this point, but through the error statement we prevent any hidden bug potentially caused by the momentary lack of knowledge - or should we say the momentary lapse of reason
        self.fmwave( [self.DNA[0],self.DNA[2],self.DNA[4]] , [self.DNA[1],self.DNA[3],self.DNA[5]] )
        self.score=np.sum((self.trial-self.target)**2)
        return self.score
    def plot_FMsynth_solution_old(self):
        plt.fill_between(self.t,self.target,self.trial,alpha=0.17)
        plt.plot(self.t,self.target,'o-',c='k',label='target',markersize=3)
        plt.plot(self.t,self.trial,'o-',c='g',label='trial',markersize=3)
        plt.ylim(-7,7)
        plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi])
        ax=plt.axes()
        plt.xlim(0,2*pi)
        ax.set_xticklabels([0,r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
        plt.xlabel(r'wave signal timeline $t$')
        plt.ylabel(r'signal $s(t)$ (green: test signal, black: target)')
        plt.title('a solution candidate for the\nCEC-2011 FM synthesis problem')
        txt='DNA = {}\nscore = {}'.format(np.round(self.DNA,3),self.score)
        plt.suptitle(txt,x=0.03,y=0.02,ha='left',va='bottom',fontsize=10)
        runlabel='c'+str(self.ncase).zfill(3)+'_g'+str(self.gg).zfill(3)+'_i'+str(self.no).zfill(3)
        plt.savefig(join(self.plotpath,'FMsynth_solution_'+runlabel+'.png'))
        plt.close()
    def plot_FMsynth_solution(self):
        #plt.figure(dpi=160)
        plt.fill_between(self.t*self.theta,self.target,self.trial,alpha=0.17)
        plt.plot(self.t*self.theta,self.target,'o-',c='k',label='target',markersize=3)
        plt.plot(self.t*self.theta,self.trial,'o-',c='g',label='trial',markersize=3)
        plt.ylim(-4,4)
        plt.xticks([0,0.5*pi,pi,1.5*pi,2*pi])
        ax=plt.axes()
        plt.xlim(0,2*pi)
        ax.set_xticklabels([0,r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
        plt.xlabel(r'wave signal timeline $t$')
        plt.ylabel(r'signal $s(t)$ (green: test signal, black: target)')
        txt1='DNA = {}\nscore = {}'.format(np.round(self.DNA,4),self.score)
        txt2=r'signal $s(t)=A_1 \sin(\omega_1 t + A_2 \sin(\omega_2 t + A_3\sin(\omega_3 t)))$'
        txt2+='\nscore = sum of square distances at points '+r'$t_i=2\pi i/100$  $i \in [0,100]$'
        txt2+='\n'+r'target DNA = [$A_1,\omega_1,A_2,\omega_2,A_3,\omega_3$]'
        txt2+=' = [1.0  5.0  1.5  4.8  2.0  4.9]'
        plt.suptitle(txt1,x=0.5,y=0.12,ha='center',va='bottom',fontsize=12)
        plt.suptitle(txt2,x=0.5,y=0.88,ha='center',va='top',fontsize=12)
        runlabel='c'+str(self.ncase).zfill(3)+'_g'+str(self.gg).zfill(3)+'_i'+str(self.no).zfill(3)
        plt.savefig(join(self.plotpath,'FMsynth_solution_'+runlabel+'.png'),dpi=160)
        plt.close()
    def plot_into_axes(self,ax):
        #self.fb=ax.fill_between(self.t*self.theta,self.target,self.trial,alpha=0.17)
        self.kline,=ax.plot(self.t*self.theta,self.target,'o-',c='k',label='target',markersize=3)
        self.gline,=ax.plot(self.t*self.theta,self.trial,'o-',c='g',label='trial',markersize=3)
        ax.set_xlim([0,2*pi])
        ax.set_ylim([-4,4])
        ax.set_xticks([0,0.5*pi,pi,1.5*pi,2*pi])
        ax.set_xticklabels([0,r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
        ax.set_xlabel(r'wave signal timeline $t$')
        ax.set_ylabel(r'signal $s(t)$ (green: test signal, black: target)')
    def update_plot(self,ax):
        self.kline.set_data([self.t*self.theta,self.target])
        self.gline.set_data([self.t*self.theta,self.trial])
        #self.fb.set_data([self.t*self.theta,self.target,self.trial])
        #plt.draw()
    def set_bad_score(self):
        self.score=9999.


class necklace(Individual):
    def evaluate(self,usage='no_matter_what',tlock=False):
        if tlock: print '!!!*!*!    star_py cannot be evalued in parallel so far; going the old serial way     !*!*!!'
        cDNA=array(self.DNA,copy=1); cDNA=2.*pi*cDNA/360.
        rays=array(sort(cDNA),copy=1); diff=rays-roll(rays,1)
        diff[0]+=2.*pi
        self.score=std(diff)
        return self.score
    def plot_into_axes(self,ax):
        # draw own look into a, which is a matplotlib axes/subplot instance
        p=PatchCollection([Wedge((0.,0.), 1.0, phi-3,phi+3, width=0.035) for phi in range(2,360,5)],
                           facecolor='k',edgecolor='k',linewidths=0,zorder=1)
        ax.add_collection(p)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.12) for i,phi in enumerate(self.DNA)],
                          facecolor='c',edgecolor='k',linewidths=2,zorder=3)
        ax.add_collection(c)
        ax.axis('equal')
        ax.axis('off')
    def update_plot(self,ax):
        ax.cla()
        self.plot_into_axes(ax)
    def set_bad_score(self):
        self.score=100000.

class hilly(Individual):
    def evaluate(self,usage='no_matter_what',tlock=False):
        if tlock: print '!!!*!*!    star_py cannot be evalued in parallel so far; going the old serial way     !*!*!!'
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
    def plot_into_axes(self,a):
        # draw own look into a, which is a matplotlib axes/subplot instance
        p=PatchCollection([Wedge((0.,0.), 1.2, phi-3,phi+3, width=0.5) for phi in range(2,360,5)],cmap=plt.cm.bone,edgecolor='white',linewidths=0,zorder=1)
        p.set_array(self.trackpotential(arange(2,360,5)))
        a.add_collection(p)
        c=PatchCollection([Circle((cos(2.*pi*phi/360.),sin(2.*pi*phi/360.)),0.1) for i,phi in enumerate(self.DNA)],
                          facecolor='c',edgecolor='k',linewidths=1.5,zorder=3)
        a.add_collection(c)
        a.axis('equal')
        a.axis('off')
    def update_plot(self,ax):
        ax.cla()
        self.plot_into_axes(ax)
    def set_bad_score(self):
        self.score=3000.

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
    def evaluate(self,usage='no_matter_what',tlock=False):
        if tlock: print '!!!*!*!    star_py cannot be evalued in parallel so far; going the old serial way     !*!*!!'
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
        arr=arange(self.ng,dtype='float')/float(self.ng)*360+sigma*npr.randn(self.ng)+a1
        arr=where(arr>360,arr-360,arr)
        arr=where(arr<0,arr+360,arr)
        self.set_DNA(arr)
    def plot_into_axes(self,a):
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
        a.axis('off')
    def update_plot(self,ax):
        ax.cla()
        self.plot_into_axes(ax)
    def set_bad_score(self):
        self.score=3000.


class trussobj:
    def __init__(self,point_coords,element_nodes,spring_constant=1.):
        self.pts=array(point_coords,copy=1)
        self.elnodes=array(element_nodes,copy=1)
        self.npts=len(self.pts)
        self.nels=len(self.elnodes)
        self.mesh = gf.Mesh('empty', 2)
        for i in range(self.npts):
            self.mesh.add_point(self.pts[i,:])
        for i in range(self.nels):
            self.mesh.add_convex( gf.GeoTrans('GT_PK(1,1)'), transpose(self.pts[self.elnodes[i,:]]) )
        self.fem = gf.MeshFem(self.mesh);
        self.ptdm=zeros((self.npts,2),dtype=int)    # point-to-dof-map i.e. rhs=self.model.rhs() then rhs[ptdm[p,i]] applies to point p and coordinate axis i
        self.fem.set_qdim(2);
        self.fem.set_classical_fem(1); # classic means Lagrange polynomial
        self.mim = gf.MeshIm(self.mesh, gf.Integ('IM_GAUSS1D(2)'))
        self.model = gf.Model('real') # 2D model variable U is of real value
        self.EA = spring_constant  # stiffness k = E*A
        self.model.add_initialized_data('lambda', self.EA )   # define Lame coefficients lambda and mu
        self.model.add_initialized_data('mu', 0);    # define Lame coefficients lambda and mu
        self.model.add_fem_variable('U', self.fem)
        self.model.add_isotropic_linearized_elasticity_brick(self.mim, 'U', 'lambda', 'mu')
        self.nreg=0   # how many regions are defined on the mesh
        self.fixreg=0  # how many regions have been created for the purpose of fixing nodes
        self.loadreg=0  # how many regions have been created for the purpose of loading nodes
        self.s_pts=zeros((self.npts,2))   # shifted points
        self.lengths=zeros(self.nels)   # element lengths
        self.s_lengths=zeros(self.nels)  # element lengths in deformed geometry
        self.elong=zeros(self.nels)   # element elongation factor
        self.p_elong=zeros(self.nels)   # element percentage of elongation
        self.delta_l=zeros(self.nels)   # element length differences
        self.thescore=0
    def determine_point_to_dof_map(self):
        cf_map=-ones((2,self.npts),dtype=int)
        for i in range(self.npts):
            cf=self.mesh.faces_from_pid(i)    # inquire which faces of which convexes are on that point (for a truss element a face is just a point)
            # cf is now an array of shape(2xN) (where N is the total number of points), then the first row contains the convex ids (for each point
            # the convex with the lowest id number), the second row the corresponding face ids
            cf_map[:,i]=cf[:,0]
        for i in range(self.npts):
            cvdof,dummy=self.fem.basic_dof_from_cvid([cf_map[0,i]])
            self.ptdm[i,0]=cvdof[2*(1-cf_map[1,i])]   # Why 1-cf_map[1,i]? -> because face 1 opposes point 0 and vice versa.
            self.ptdm[i,1]=cvdof[2*(1-cf_map[1,i])+1]   # Why 1-cf_map[1,i]? -> because face 1 opposes point 0 and vice versa.
        return self.ptdm
    def fix_points(self,pl):
        # pl: point list , i.e. list of point id numbers
        # returns: region number and variable name connected to the Dirichlet condition with penalisation applied
        self.nreg+=1; self.fixreg+=1
        n=len(pl); cf_map=-ones((2,n),dtype=int)
        for i in range(n):
            cf=self.mesh.faces_from_pid(pl[i])    # inquire which faces of which convexes are on that point (for a truss element a face is just a point)
            # cf is now an array of shape(2xN) (where N is the total number of points), then the first row contains the convex ids (for each point
            # the convex with the lowest id number), the second row the corresponding face ids
            cf_map[:,i]=cf[:,0]
        self.mesh.set_region(self.nreg,cf_map)
        self.model.add_initialized_data('R'+str(self.fixreg), [0,0]);
        self.model.add_Dirichlet_condition_with_penalization(self.mim, 'U', 1e20, self.nreg, 'R'+str(self.fixreg));
        return self.nreg, 'R'+str(self.fixreg)
    def load_points(self,pl,lvl):
        # pl: point list
        # lvl: load vector list
        # returns: the explicit right hand side added to the system
        n=len(pl); cf_map=-ones((2,n),dtype=int)
        for i in range(n):
            cf=self.mesh.faces_from_pid(pl[i])    # inquire which faces of which convexes are on that point (for a truss element a face is just a point)
            # cf is now an array of shape(2xM) (if there are M convexes sharing that point), then the first row contains the convex ids,
            # the second row the corresponding face ids
            cf_map[:,i]=cf[:,0]#; cf_map[1,i]=cf[1,0]
        nodal_load = zeros(self.fem.nbdof(), dtype=float)
        for i in range(n):
            cvdof,dummy=self.fem.basic_dof_from_cvid([cf_map[0,i]])
            nodal_load[cvdof[2*(1-cf_map[1,i])]:cvdof[2*(1-cf_map[1,i])]+2]=transpose(lvl[i,:])   # Why 1-cf_map[1,i]? -> because face 1 opposes point 0 and vice versa.
        self.model.add_explicit_rhs('U', nodal_load)
        return nodal_load
    def load_points_via_map(self,pl,lvl):
        # pl: point list
        # lvl: load vector list
        # returns: the explicit right hand side added to the system
        n=len(pl)
        nodal_load = zeros(self.fem.nbdof(), dtype=float)
        for i in range(n):
            nodal_load[self.ptdm[pl[i],0]:self.ptdm[pl[i],0]+2]=transpose(lvl[i,:])   # Why 1-cf_map[1,i]? -> because face 1 opposes point 0 and vice versa.
        self.model.add_explicit_rhs('U', nodal_load)
        return nodal_load
    def solve(self):
        t_start=time()
        self.model.solve()
        self.U = self.model.variable('U')
        #print 'U:'
        #print self.U
        t_stop=time()
        return t_stop-t_start
    def postprocess(self):
        for i in range(self.npts):
            self.s_pts[i,:]=self.pts[i,:]+self.U[self.ptdm[i,:]]
        for i in range(self.nels):
            self.lengths[i]=sqrt((self.pts[self.elnodes[i,1],0]-self.pts[self.elnodes[i,0],0])**2+(self.pts[self.elnodes[i,1],1]-self.pts[self.elnodes[i,0],1])**2)
            self.s_lengths[i]=sqrt((self.s_pts[self.elnodes[i,1],0]-self.s_pts[self.elnodes[i,0],0])**2+(self.s_pts[self.elnodes[i,1],1]-self.s_pts[self.elnodes[i,0],1])**2)
            self.elong[i]=self.s_lengths[i]/self.lengths[i]   # elongation factor
            self.delta_l[i]=self.s_lengths[i]-self.lengths[i]
        self.p_elong=100*(array(self.elong,copy=1)-1)   # percentage elongation
        self.thescore = sum( where(self.delta_l>=0, 0.125*self.delta_l*self.lengths, -1*self.delta_l*self.lengths) )   # 8 times lighter weight of ropes by introducing factor 0.125
    def save_deformed_plot(self,location,label):
        #self.postprocess2()   # Why do I have to do this each time? Where does the offset in the second save_deformed_plot call come from if you don't?
        f=plt.figure(); a=f.add_subplot(111)
        a.add_collection(self.s_pointcol)                         # points
        for i in range(self.npts):
            a.text(self.pts[i,0],self.pts[i,1],str(i),color='k',fontsize=20)            # point numbers
        for i in range(self.nels):
            a.text(mean(self.pts[self.elnodes[i,:],0]),mean(self.pts[self.elnodes[i,:],1]),str(i),color='k',fontsize=20)       # plot element numbers
        a.add_collection(self.lincol)
        # loaded geometry:
        # that would be the same as above, just using s_pts instead of pts and with line coloring correspondint to element elongation
        ax2=plt.axes()
        ax2.add_collection(self.s_lincol)
        cb=plt.colorbar(self.s_lincol); cb.set_label('% element elongation')
        #plt.title('score is '+str(self.thescore))
        a.set_title('score is '+str(self.thescore))
        f.savefig(location+'/gf_7el_bridge_'+label+'.png')
        del a; del f
    def plot_into_axes(self,a):
        if not hasattr(self,'nels'):
            self.evaluate()
        # a is a subplot/axes instance; cb is a colorbar or is zero if not needed
        # draw own look into a, which is a matplotlib axes/subplot instance
        lincol=LineCollection([zip(self.pts[self.elnodes[i,:],0],self.pts[self.elnodes[i,:],1]) for i in range(self.nels)], linewidths = 8,linestyles = 'solid', color='grey')
        maxabs_p_elong=max(abs(self.p_elong))
        l_elnodes=r_[array(self.elnodes[:2],copy=1),array(self.elnodes[:],copy=1)]
        l_p_elong=append(array(self.p_elong[:2],copy=1),array(self.p_elong[:],copy=1))
        l_nels=self.nels+2
        l_p_elong[0],l_p_elong[1]=-maxabs_p_elong, maxabs_p_elong
        s_lincol=LineCollection([zip(self.s_pts[l_elnodes[i,:],0],self.s_pts[l_elnodes[i,:],1]) for i in range(l_nels)], linewidths = 4,linestyles = 'solid', cmap=blue_red4)
        s_lincol.set_array(l_p_elong)
        a.scatter(self.pts[:,0],self.pts[:,1],marker='o',s=120,color='grey')
        a.add_collection(lincol, autolim=True)
        a.add_collection(s_lincol, autolim=True)
        a.axis('equal')
        #cb=plt.colorbar(s_lincol,ax=a); cb.set_label('% element elongation')
#        else:
#            a.plot(range(3),ones(3))
    def update_plot(self,ax):
        ax.cla()
        self.plot_into_axes(ax)


##    simple truss structure
##         1-------3                                   constrained: no displacement at all for points 0 and 4
##        / \     / \
##       /   \   /   \           ^ v                   free for evolutionary optimisation: vertical position of points 1, 2, and 3 can vary by +-h
##      /     \ /     \          |
##     0-------2-------4         ----> u
##

class gf_7el_bridge(trussobj,Individual):
    def __init__(self,paramspace,ncase):
        Individual.__init__(self,paramspace,ncase)
        #a = 1.0;  h = a*math.sqrt(3)/2
        #self.pts = array([[0,0],[0.5*a, h],[a, 0.],[1.5*a, h],[2*a, 0.0]], dtype=float32)   # just to draw the initial geometry plot before evolution starts
        #self.elnodes = array([[0,1],[0,2],[1,2],[1,3],[2,3],[2,4],[3,4]])   # just to draw the initial geometry plot before evolution starts
        #self.npts=len(self.pts); self.nels=len(self.elnodes)

    def evaluate(self,usage='no_matter_what',tlock=False):
        if tlock: print '!!!*!*!    threaded score evaluation not verified yet for gf_7el_bridge, but maybe it works     !*!*!!'
        a = 1.0;
        h = a*sqrt(3.)/2
        # point coordinates
        pts = array([[0,0],
                    [0.5*a, h*(1+self.DNA[0])],
                    [a, h*self.DNA[1]],
                    [1.5*a, h*(1+self.DNA[2])],
                    [2*a, 0.0]], dtype=float)
        # array elnodes contains point numbers of points forming the convexes
        # rows: elements, columns: nodes
        elnodes = array([[0,1],
                         [0,2],
                         [1,2],
                         [1,3],
                         [2,3],
                         [2,4],
                         [3,4]])
        fixpoints=array([0,4])
        loadpoints=array([1,2,3])  # which points (by their id) should get loaded
        loadvecs=zeros((3,2))      # to note down x- ynd y-loads for the points in the above list loadpoints
        loadvecs[0,0]=1; loadvecs[0,1]=-1
        loadvecs[1,0]=0; loadvecs[1,1]=0
        loadvecs[2,0]=0; loadvecs[2,1]=0
        trussobj.__init__(self,pts,elnodes,spring_constant=60.)
        self.determine_point_to_dof_map()
        fixregion,fix_inidata=self.fix_points(fixpoints)
        fem_eq_rhs=self.load_points(loadpoints,loadvecs)
        soltime=self.solve()
        self.postprocess()
        #self.postprocess2()
        self.score=self.thescore
        return self.score
    def set_bad_score(self):
        self.score=1e6

##!      j0         j1         j2         j3         j4         j5         j6         j7
##! wall  *---t0-----*---t1----*---t2-----*---t3-----*---t4-----*---t5-----*---t6-----*        -----
##!                 /I         /I         /I         /I         /I         /I         /           I
##!                / I        / I        / I        / I        / I        / I        /            I
##!               /  I       /  I       /  I       /  I       /  I       /  I       /             I
##!              /  t13     /  t14     /  t15     /  t16     /  t17     /  t18     /              I
##!             /    I     /    I     /    I     /    I     /    I     /    I     /             height h
##!           t19    I   t20    I   t21    I   t22    I   t23    I   t24    I   t25               I
##!           /      I   /      I   /      I   /      I   /      I   /      I   /                 I
##!          /       I  /       I  /       I  /       I  /       I  /       I  /                  I
##!         /        I /        I /        I /        I /        I /        I /                   I
##!        /         I/         I/         I/         I/         I/         I/                    I
##! wall  *---t7-----*---t8-----*---t9-----*---t10----*---t11----*---t12----*                   -----
##!      j8         j9         j10        j11        j12        j13        j14
##!
##!                             I--- d ----I
##!                               distance

class gf_kragtraeger(trussobj,Individual):
    def __init__(self,paramspace,ncase):
        Individual.__init__(self,paramspace,ncase)
        #a = 1.0;  h = a*math.sqrt(3)/2
        #self.pts = array([[0,0],[0.5*a, h],[a, 0.],[1.5*a, h],[2*a, 0.0]], dtype=float32)   # just to draw the initial geometry plot before evolution starts
        #self.elnodes = array([[0,1],[0,2],[1,2],[1,3],[2,3],[2,4],[3,4]])   # just to draw the initial geometry plot before evolution starts
        #self.npts=len(self.pts); self.nels=len(self.elnodes)

    def evaluate(self,usage='no_matter_what',tlock=False):
        if tlock: print '!!!*!*!    threaded score evaluation not verified yet for gf_7el_bridge, but maybe it works     !*!*!!'
        a = 0.5;
        h = 1.
        # point coordinates
        pts = array([[0*a,h],
                     [1*a,h],
                     [2*a,h],
                     [3*a,h],
                     [4*a,h],
                     [5*a,h],
                     [6*a,h],
                     [7*a,h],
                     [0*a,0],
                     [1*a,0],
                     [2*a,0],
                     [3*a,0],
                     [4*a,0],
                     [5*a,0],
                     [6*a,0]], dtype=float)
        for i in range(6):
            pts[i+9,0]+=self.DNA[i]*a
            pts[i+9,1]+=self.DNA[i+6]*h
        # array elnodes contains point numbers of points forming the convexes
        # rows: elements, columns: nodes
        elnodes = array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],
                         [8,9],[9,10],[10,11],[11,12],[12,13],[13,14],
                         [1,9],[2,10],[3,11],[4,12],[5,13],[6,14],
                         [1,8],[2,9],[3,10],[4,11],[5,12],[6,13],[7,14]])
        fixpoints=array([0,8])
        loadpoints=array([1,2,3,4,5,6,7])  # which points (by their id) should get loaded
        loadvecs=zeros((7,2))      # to note down x- ynd y-loads for the points in the above list loadpoints
        loadvecs[0,0]=0; loadvecs[0,1]=-1
        loadvecs[1,0]=0; loadvecs[1,1]=-1
        loadvecs[2,0]=0; loadvecs[2,1]=-1
        loadvecs[3,0]=0; loadvecs[3,1]=-1
        loadvecs[4,0]=0; loadvecs[4,1]=-1
        loadvecs[5,0]=0; loadvecs[5,1]=-1
        loadvecs[6,0]=0; loadvecs[6,1]=-1
        trussobj.__init__(self,pts,elnodes,spring_constant=260e4)
        self.determine_point_to_dof_map()
        fixregion,fix_inidata=self.fix_points(fixpoints)
        fem_eq_rhs=self.load_points(loadpoints,loadvecs)
        soltime=self.solve()
        self.postprocess()
        #self.postprocess2()
        self.score=self.thescore
        return self.score
    def set_bad_score(self):
        self.score=1e6


class rastrigin(Individual):
    def __init__(self,dummyfunc,pspace):
        Individual.__init__(self,dummyfunc,pspace)
        self.dots=None
    def evaluate(self):
        cDNA=array(self.DNA,copy=1)
        self.score=self.rastrigin(cDNA)
        return self.score
    def rastrigin(self,x):
        n=len(x); A=10.
        r=A*n+sum(x**2-A*cos(2*pi*x))
        return r
    def rastrigin1D(self,x):
        A=10.
        r=A+x**2-A*cos(2*pi*x)
        return r
    def plot_into_axes(self,a):
        # show yourself on axes/subplot a
        x=self.widths[0]*arange(160,dtype=float)/159.+self.lls[0]; rast1d=self.rastrigin1D(x); rast1d=(self.ng+1)*rast1d/max(rast1d)
        a.plot(x,rast1d,color='grey')
        self.dots,=a.plot(self.DNA,arange(self.ng),'bo')
        a.axis((np.min(self.lls),np.max(self.uls),-0.5,self.ng+0.5))
    def update_plot(self,ax):
        self.dots.set_data([self.DNA,arange(self.ng)])
    def set_bad_score(self):
        self.score=30

class sphere(Individual):
    def __init__(self,dummyfunc,pspace):
        Individual.__init__(self,dummyfunc,pspace)
        self.dots=None
    def evaluate(self):
        self.score=sqrt(np.sum(self.DNA**2))
        return self.score
    def parabel(self,x):
        r=x**2
        return r
    def plot_into_axes(self,a):
        # show yourself on axes/subplot a
        x=self.widths[0]*arange(160,dtype=float)/159.+self.lls[0]; sph1d=self.parabel(x); sph1d=(self.ng+1)*sph1d/max(sph1d)
        a.plot(x,sph1d,color='grey')
        self.dots,=a.plot(self.DNA,arange(self.ng),'bo')
        a.axis((np.min(self.lls),np.max(self.uls),-0.5,self.ng+0.5))
    def update_plot(self,ax):
        self.dots.set_data([self.DNA,arange(self.ng)])
    def set_bad_score(self):
        self.score=1e6
