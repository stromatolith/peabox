#!python
"""
I want to learn how to complile C-code into a dll and use it from within python
why: to be able to use the CEC-2013 test function suite

once the shared library test_func.so exists, with this script you can use and
plot the test functions
"""
import numpy as np
from numpy import pi, zeros, linspace, flipud
from pylab import plt
from ctypes import cdll
import ctypes as ct

# I found out when looping over k only the first test function called would yield a correct plot
# and this happened even when reloading the whole library each time
# for k in range(1,29):  # so, this didn't work
k=28  # which function to plot

fnames={1:'sphere',
        2:'rot. ellipse',
        3:'rot. bent cigar',
        4:'rot. discus',
        5:'diff. powers',
        6:'rot. Rosenbrock',
        7:'rot. Schaffer F7',
        8:'rot. Ackley',
        9:'rot. Weierstrass',
        10:'rot. Griewank',
        11:'Rastrigin',
        12:'rot. Rastrigin',
        13:'noncont. rot. Rastrigin',
        14:'Schwefel',
        15:'rot. Schwefel',
        16:'rot. Katsuura',
        17:'Lunacek bi-R.',
        18:'rot. Lunacek bi-R.',
        19:'exp. Grie+Ros',
        20:'exp. Schaffer F6',
        21:'comp. function 1',
        22:'comp. function 2',
        23:'comp. function 3',
        24:'comp. function 4',
        25:'comp. function 5',
        26:'comp. function 6',
        27:'comp. function 7',
        28:'comp. function 8',}

tf = cdll.LoadLibrary('./test_func.so')

tf.test_func.argtypes=[ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_int,ct.c_int,ct.c_int]
tf.test_func.restype=None

n=2; m=200; h=180
xlim=[-27.,-17.]; ylim=[7.,17.];
xwidth=xlim[1]-xlim[0]; ywidth=ylim[1]-ylim[0];
dx=xwidth/(m-1.); dy=ywidth/(h-1.);
x=linspace(xlim[0],xlim[1],m+1)
y=linspace(ylim[0],ylim[1],h+1)

npdat=zeros(n*m)
dat = (ct.c_double * len(npdat))()
for i,val in enumerate(npdat):
    dat[i] = val

npf=zeros(m)
f = (ct.c_double * len(npf))()
for i,val in enumerate(npf):
    f[i] = val

rarr=zeros((h,m))

# initially the k-loop was here, and it didn't work for any plot except the first one
# for k in range(1,29):  # so, this didn't work
print 'now function ',k
for i in range(h):
    yc=ylim[1]-i*dy
    for j in range(m):
        xc=xlim[0]+j*dx;
        dat[j*n]=xc;
        dat[j*n+1]=yc;
    #print "first DNA: ",dat[0],dat[1];
    #print "2nd DNA: ",dat[2],dat[3];
    #print "last DNA: ",dat[2*m-2],dat[2*m-1];
    r1=tf.test_func(dat,f,ct.c_int(n),ct.c_int(m),ct.c_int(k))
    rarr[i,:]=[f[j] for j in range(m)]



plt.pcolor(x,y,flipud(rarr),vmin=np.min(rarr),vmax=np.max(rarr))
plt.colorbar()
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('CEC-2013 test function suite:\nno. {0}: {1}'.format(k,fnames[k]))
plt.suptitle('evaluated using ctypes',x=0.02,y=0.02,ha='left',va='bottom',fontsize=9)
#plt.show()
plt.savefig('./pics/test_func_'+str(k).zfill(2)+'_using_ctypes_zoomlevel_2.png')
plt.clf()
plt.close('all')

