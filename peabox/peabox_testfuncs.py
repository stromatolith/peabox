#!python
"""
peabox - python evolutionary algorithm toolbox
by Markus Stokmaier, IKET, KIT, Karlsruhe, August 2012

This file hosts some test functions.
"""

import numpy as np
import numpy.random as npr
from numpy import sqrt, pi, sin, cos, exp, log, ceil, where
from numpy import array, arange, asfarray, linspace, zeros, diag, dot, prod, roll
from numpy.linalg import qr
from scipy.io import loadmat


#--- popular old-scool test functions
#--- beware: lack of nastiness

def rosenbrock(x):
    """Rosenbrock function as found in scipy.optimize"""
    x = asfarray(x)
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0,axis=0)    #    (useful domain proposal: [-100,100])

def rastrigin(x):
    x = asfarray(x)
    n=len(x); A=10.
    return A*n+np.sum(x**2-A*cos(2*pi*x))

def griewank(x):
    # common optimization test problem
    # allow parameter range from -1000 to +1000
    x = asfarray(x); ndim=len(x)
    return np.sum(0.00025*x**2)-20*prod(1+cos(x/sqrt(arange(ndim,dtype=float)+1)))


def ackley(x):
    # common optimization test problem
    # allow parameter range -32.768<=x(i)<=32.768, global minimum at x=(0,0,...,0)
    x = asfarray(x); ndim=len(x)
    a=20.; b=0.2; c=2.*pi
    return -a*exp(-b*sqrt(1./ndim*np.sum(x**2)))-exp(1./ndim*np.sum(cos(c*x)))+a+exp(1.)

#--- CEC2005 test functions ----------------------------------------------------

def CEC05_test_function_producer(number,dim):
    """
    return a callable function representing one of the 25 test functions
    presented at the CEC-2005 conference in the desired dimensionality.
    """
    if number in [1,6,7,8,10,11,13]:
        c=eval('CEC05_f'+str(number)+'_container(dim)')
        return c.call
    else:
        msg="sorry, not done yet - but implementing f{} shouldn't be too difficult looking at the other ones".format(number)
        raise NotImplementedError(msg)

class TFContainer:
    def __init__(self,dim):
        self.o=zeros(dim)
        self.M=zeros((dim,dim))
        self.fbias=0.
        self.dim=dim
        self.check_data_availability()
    def call(self,x):
        raise NotImplementedError("you should only use a subclass of TFContainer where a suitable method 'call(self,x)' has been implemented")
    def get_data_1(self,numero,radical,suffix='_func_data'):
        fbd=loadmat('./CEC05_files/fbias_data.mat')  # the f_bias dictionary
        fb=fbd['f_bias'][0,:]                        # the f_bias array
        fbias=fb[numero-1]                             # the desired f_bias for this function
        od=loadmat('./CEC05_files/'+radical+suffix+'.mat')    # the shift data dictionary
        o=od['o'][0,:]                                           # the shift data array
        return fbias,o
    def get_data_2(self,numero,radical,suffix='_func_data'):
        fbd=loadmat('./CEC05_files/fbias_data.mat')  # the f_bias dictionary
        fb=fbd['f_bias'][0,:]                        # the f_bias array
        fbias=fb[numero-1]                             # the desired f_bias for this function
        od=loadmat('./CEC05_files/'+radical+suffix+'.mat')    # the shift data dictionary
        o=od['o'][0,:]                                           # the shift data array
        if self.dim in [2,10,30,50]:
            Md=loadmat('./CEC05_files/'+radical+'_M_D'+str(self.dim)+'.mat')    # the rotation matrix dictionary
            M=Md['M']                                          # the rotation matrix array
        else:
            M=None
        return fbias,o,M
    def check_data_availability(self):
        try:
            loadmat('./CEC05_files/fbias_data.mat')
        except:
            msg="the CEC-2005 shift and rotation data arrays (the matlab matrices *.mat)"
            msg+=" must be made available in a subfolder called 'CEC05_files'; "
            msg+="you can download them from http://www.ntu.edu.sg/home/epnsugan/"
            raise UserWarning(msg)

class CEC05_f1_container(TFContainer):
    # shifted sphere function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o=self.get_data_1(1,'sphere')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            self.o[:]=-100+200*npr.rand(self.dim)
        self.fbias=fbias
    def call(self,x):
        z=asfarray(x)-self.o
        return np.sum(z**2)+self.fbias
    
class CEC05_f6_container(TFContainer):
    # shifted Rosenbrock's function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o=self.get_data_1(6,'rosenbrock')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            self.o[:]=-90+180*npr.rand(self.dim)
        self.fbias=fbias
    def call(self,x):
        z=asfarray(x)-self.o+1
        return np.sum(100.0*(z[:-1]**2.0-z[1:])**2.0 + (z[:-1]-1)**2.0) + self.fbias
    
class CEC05_f7_container(TFContainer):
    # shifted rotated Griewank's function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o,M=self.get_data_2(7,'griewank')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            self.o[:]=-90+180*npr.rand(self.dim)
        if M is None:
            if self.dim in [2,10,30,50]:
                raise StandardError('coding error, program should never end up here')
            else:
                M=random_rot_matrix(self.dim,3)
                M=M*(1.+0.3*npr.randn(self.dim,self.dim))
        self.fbias=fbias
        self.M[:,:]=M
    def call(self,x):
        z=asfarray(x)-self.o
        z=dot(z,self.M)
        f=1.
        for i in range(self.dim):
            f*=cos(z[i]/sqrt(float(i+1)));
        return np.sum(z**2)/4000.-f+1+self.fbias;
    
class CEC05_f8_container(TFContainer):
    # shifted rotated Griewank's function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o,M=self.get_data_2(8,'ackley')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            self.o[:]=-30+60*npr.rand(self.dim)
        self.o[0::2]=-32
        if M is None:
            if self.dim in [2,10,30,50]:
                raise StandardError('coding error, program should never end up here')
            else:
                M=random_rot_matrix(self.dim,100)
        self.fbias=fbias
        self.M[:,:]=M
    def call(self,x):
        z=asfarray(x)-self.o
        z=dot(z,self.M)
        f=np.sum(z**2)
        f=20-20*exp(-0.2*sqrt(f/self.dim))-exp(np.sum(cos(2*pi*z))/self.dim)+exp(1)
        return f+self.fbias
    
class CEC05_f10_container(TFContainer):
    # shifted rotated Rastrigin's function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o,M=self.get_data_2(10,'rastrigin')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            self.o[:]=-5+10*npr.rand(self.dim)
        if M is None:
            if self.dim in [2,10,30,50]:
                raise StandardError('coding error, program should never end up here')
            else:
                M=random_rot_matrix(self.dim,2)
        self.fbias=fbias
        self.M[:,:]=M
    def call(self,x):
        z=asfarray(x)-self.o
        z=dot(z,self.M)
        return np.sum(z**2-10*cos(2*pi*z)+10)+self.fbias
        
class CEC05_f11_container(TFContainer):
    # shifted rotated Weierstrass function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o,M=self.get_data_2(11,'weierstrass',suffix='_data')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            #self.o[:]=-0.5+0.5*npr.rand(self.dim)  # how it is implemented in benchmark_func.m, probably a mistake
            #self.o[:]=-0.5+1.0*npr.rand(self.dim)  # like this it looks more logic
            self.o[:]=-0.4+0.8*npr.rand(self.dim)  # however, this seems to be the range of numbers covered by the array o in weierstrass_data.mat
        if M is None:
            if self.dim in [2,10,30,50]:
                raise StandardError('coding error, program should never end up here')
            else:
                M=random_rot_matrix(self.dim,5)
        self.fbias=fbias
        self.M[:,:]=M
    def w(self,x):
        kmax=20; a=0.5; b=3.
        c1=a**arange(kmax+1)
        c2=2*pi*b**arange(kmax+1)
        if type(x)==float:
            ksum=np.sum(c1*cos(c2*(x+0.5)))
        else:
            ksum=zeros(len(x))
            for i in range(len(x)):
                ksum[i]=np.sum(c1*cos(c2*(x[i]+0.5)))
        return ksum
    def call(self,x):
        z=asfarray(x)-self.o
        z=dot(z,self.M)
        c=self.dim*self.w(0.)
        f=np.sum(self.w(z))
        return f-c+self.fbias


        
def F8F2(xi,xj):
    f2=100.*(xi**2-xj)**2+(xi-1.)**2
    f=f2**2/4000.-cos(f2)+1.;
    return f

class CEC05_f13_container(TFContainer):
    # expanded function F8F2 from Griewank's and Rosenbrock's function
    def __init__(self,dim):
        TFContainer.__init__(self,dim)
        self.settle_data()
    def settle_data(self):
        fbias,o=self.get_data_1(13,'EF8F2')
        if self.dim<=len(o):
            self.o[:]=o[:self.dim]
        else:
            #self.o[:]=-1+1*npr.rand(self.dim)  # how it is implemented in benchmark_func.m, probably a mistake
            self.o[:]=-1+2*npr.rand(self.dim)  # deemed reasonable looking at numbers in EF8F2_func_data.mat
        self.fbias=fbias
    def call(self,x):
        z=asfarray(x)-self.o+1
        f=0.
        for i in range(self.dim-1):
            f+=F8F2(z[i],z[i+1])
        f+=F8F2(z[-1],z[0])
        return f+self.fbias
    

def random_rot_matrix(D,c):
    A=npr.randn(D,D);
    P,R=qr(A);
    A=npr.randn(D,D);
    Q,R=qr(A);
    u=npr.rand(D);
    D=c**((u-np.min(u))/(np.max(u)-np.min(u)));
    D=diag(D);
    M=dot(P,dot(D,Q))
    return M


#--- CEC-2011 test functions (real-world problems) -----------------------------


class CEC10_FMsynth(object):
    def __init__(self):
        self.a=[1., 1.5, 2.]
        self.w=[5. ,4.8, 4.9]
        self.nt=101
        self.t=arange(self.nt,dtype=float)
        self.theta=2*pi/(self.nt-1)
        self.trial=zeros(self.nt)
        self.fmwave(self.a,self.w)
        self.target=array(self.trial,copy=1)
    def fmwave(self,a,w):
        t=self.t; th=self.theta
        self.trial[:]=a[0]*sin(w[0]*t*th+a[1]*sin(w[1]*t*th+a[2]*sin(w[2]*t*th)))
    def call(self,DNA):
        self.fmwave( [DNA[0],DNA[2],DNA[4]] , [DNA[1],DNA[3],DNA[5]] )
        return np.sum((self.trial-self.target)**2)
        
class CEC10_FMsynth_flexible(object):
    def __init__(self,n,amplitudes=[1., 1.5, 2.],frequencies=[5. ,4.8, 4.9],bds=[-6.4,6.35],nt=101):
        #assert n==6   # possible problem dimensions keeping fidelity to CEC-2010 tech report
        assert n<=6  # possible problem dimensions; I introduced a bit more flexibility
        self.a=amplitudes
        self.w=frequencies
        self.nt=nt
        self.t=arange(self.nt,dtype=float)
        self.theta=2*pi/(self.nt-1)
        self.trial=zeros(nt)
        self.fmwave(self.a,self.w)
        self.target=array(self.trial,copy=1)
        goalDNA=[self.a[0],self.w[0],self.a[1],self.w[1],self.a[2],self.w[2]]
        self.fixedDNA=goalDNA[6-n:]
    def fmwave(self,a,w):
        t=self.t; th=self.theta
        self.trial[:]=a[0]*sin(w[0]*t*th+a[1]*sin(w[1]*t*th+a[2]*sin(w[2]*t*th)))
    def call(self,DNA):
        DNA=list(DNA)+self.fixedDNA
        self.fmwave( [DNA[0],DNA[2],DNA[4]] , [DNA[1],DNA[3],DNA[5]] )
        return np.sum((self.trial-self.target)**2)



#--- test functions by Darrell Whitley's genitor group -------------------------

"""
# source code from http://www.cs.colostate.edu/~genitor/functions.html
double f101_eval(const vector<double> &params){
  double x=params[0];
  double y=params[1];
  double sum=0.0;
  int dim = params.size();
  for(int i=0; i<dim-1; i++){
      x=params[i];
      y=params[i+1];
      sum+= (-x*sin(sqrt(fabs(x-(y+47))))-(y+47)*sin(sqrt(fabs(y+47+(x/2)))));
  }
  return sum; 
}
"""
def genitor_f101(x):
    x=asfarray(x)
    s=0.
    for i in range(len(x)-1):
        s+= (-x[i]*sin(sqrt(np.fabs(x[i]-(x[i+1]+47))))-(x[i+1]+47)*sin(sqrt(np.fabs(x[i+1]+47+(x[i]/2)))))
    return s


