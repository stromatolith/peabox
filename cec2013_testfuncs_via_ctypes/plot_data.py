#!python
from pylab import *
from os import getcwd
from os.path import join
loc=getcwd()
dat=loadtxt(join(loc,'output','test_data.txt'))

lim=50
nx,ny=shape(dat)
x=linspace(-lim,lim,nx+1)
y=linspace(-lim,lim,ny+1)
plt.pcolor(y,x,flipud(dat))
plt.colorbar()
plt.show()
#plt.savefig('test_func_09.png')
plt.close()


