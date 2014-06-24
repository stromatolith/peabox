#!python
"""
how does a particle trajectory look like?
"""
from numpy import ones
import matplotlib.pyplot as plt
from peabox_testfuncs import CEC05_test_function_producer as tfp
from peabox_helpers import simple_searchspace

from PSO_defs import PSOSwarm_standard2D as Pswarm, PSO_Individ #as Pind

class Memorizing_PSO_Individ(PSO_Individ):
    def __init__(self,objfunc,paramspace):
        PSO_Individ.__init__(self,objfunc,paramspace)
        self.traj=[]
        self.trajqual=[]
    def memorize(self):
        self.traj.append(self.get_copy_of_DNA())
        self.trajqual.append(self.score)
    def do_step(self):
        PSO_Individ.do_step(self)
        self.memorize()
    def get_trajectory_components(self,dims):
        output=[]
        for d in dims:
            output.append([self.traj[i][d] for i in range(len(self.traj))])
        return output

dim=2
f1=tfp(1,dim)   # Weierstrass function in 2 dimensions
f11=tfp(11,dim)   # Weierstrass function in 2 dimensions
searchspace=simple_searchspace(dim,-3.,1.)
p=Pswarm(Memorizing_PSO_Individ,12,f1,searchspace,[3,4],1)
p.random_ini()
#p.alpha=0.95


if True:
    dude=p[0]
    #dude.random_influence='1D'
    dude.add_attractor(ones(dim),1.)
    dude.speed[:]=1,0
    
    for i in range(20):
        dude.do_step()
    
    x,y=dude.get_trajectory_components([0,1])
    plt.plot(x,y,'bo-')
    plt.plot([1],[1],'rd')
    plt.title('one particle with the attractor indicated by the red diamond')
    plt.show()

if False:
    for i in range(20):
        p.do_step()
        print p[0].score, p[-1].DNA
        
    for dude in p:
        x,y=dude.get_trajectory_components([0,1])
        plt.plot(x,y)
    plt.title('tracks of the whole swarm')
    plt.show()





    