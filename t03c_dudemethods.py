#!python
"""
peabox tutorial 
lesson 03 - operators for individuals
b) basic DNA manipulation operators for individuals: mutation operators

test function: N-dimensional parabolic potential


what is to be shown here:

 - how to mess with DNA

by Markus Stokmaier, IKET, KIT, Karlsruhe, September 2012
"""

import numpy as np
import numpy.random as npr
from numpy import array, arange, asfarray, pi, cos, zeros, ones, where, linspace
import matplotlib as mpl
import matplotlib.pyplot as plt
from peabox_individual import Individual
from peabox_population import Population

def parabolic(x):
    return np.sum(x*x)

searchspace2=(('p1',-20.,+20.),
              ('p2',-20.,+20.))

searchspace3=(('p1',-20.,+20.),
              ('p2',-20.,+20.),
              ('p3',-20.,+20.))

searchspace5=(('p1',-20.,+20.),
              ('p2',-20.,+20.),
              ('p3',-20.,+20.),
              ('p4',-20.,+20.),
              ('p5',-20.,+20.))

N=10
p2a=Population(Individual,N,parabolic,searchspace2)
p3a=Population(Individual,N,parabolic,searchspace3)
p5a=Population(Individual,N,parabolic,searchspace5)
P=p2a+p3a+p5a
npr.seed(1)
x0,y0=-12,5
for dude in P:
    dude.DNA[:2]=x0,y0
#p5a.print_stuff()
startDNA2=p2a[0].get_copy_of_DNA()
startDNA3=p3a[0].get_copy_of_DNA()
startDNA5=p5a[0].get_copy_of_DNA()

# now mutate by adding a vector of well-defined length D=4 into a random direction
D=0.3
nn=5000
plotdat2D=zeros((N*nn+2,2))
plotdat3D=zeros((N*nn+2,2))
plotdat5D=zeros((N*nn+2,2))

mirror_or_cycle_into_bounds='mirror'

if mirror_or_cycle_into_bounds=='mirror':
    for i in range(nn):
        for j in range(N):
            
            p2a[j].set_DNA(startDNA2)
            p2a[j].mutate_fixstep(stepsize=D,uCS=True,mirrorbds=True)
            plotdat2D[i*N+j,:]=p2a[j].DNA[:2]
            
            p3a[j].set_DNA(startDNA3)
            p3a[j].mutate_fixstep(stepsize=D,uCS=True,mirrorbds=True)
            plotdat3D[i*N+j,:]=p3a[j].DNA[:2]
            
            p5a[j].set_DNA(startDNA5)
            p5a[j].mutate_fixstep(stepsize=D,uCS=True,mirrorbds=True)
            plotdat5D[i*N+j,:]=p5a[j].DNA[:2]

elif mirror_or_cycle_into_bounds=='cycle':
    for i in range(nn):
        for j in range(N):
            
            p2a[j].set_DNA(startDNA2)
            p2a[j].mutate_fixstep(stepsize=D,uCS=True,mirrorbds=False)
            p2a[j].cycle_DNA_into_bounds()
            plotdat2D[i*N+j,:]=p2a[j].DNA[:2]
            
            p3a[j].set_DNA(startDNA3)
            p3a[j].mutate_fixstep(stepsize=D,uCS=True,mirrorbds=False)
            p3a[j].cycle_DNA_into_bounds()
            plotdat3D[i*N+j,:]=p3a[j].DNA[:2]
            
            p5a[j].set_DNA(startDNA5)
            p5a[j].mutate_fixstep(stepsize=D,uCS=True,mirrorbds=False)
            p5a[j].cycle_DNA_into_bounds()
            plotdat5D[i*N+j,:]=p5a[j].DNA[:2]

for dim,dat in zip([2,3,5],[plotdat2D,plotdat3D,plotdat5D]):
    dat[-2,:]=[-20,-20]
    dat[-1,:]=[+20,+20]
    plt.hexbin(dat[:,0],dat[:,1], cmap=mpl.cm.gist_stern,gridsize=40)
    plt.colorbar()
    #plt.axis('equal')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.xlabel('DNA[0]')
    plt.ylabel('DNA[1]')
    plt.title('DNA vector distribution after mutation with fixed\nstep size in a {}-dimensional space'.format(dim))
    plt.savefig('distrib_after_fixstep_dim{0}{1}.png'.format(dim,mirror_or_cycle_into_bounds))
    plt.close()
    
print "now try for yourself:"
print "other mutation operators implemented in the Individual class"
