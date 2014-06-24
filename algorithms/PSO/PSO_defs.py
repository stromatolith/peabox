#!python
"""
my trial at implementing the particle swarm optimisation algorithm

What is PSO? Imagine you're sitting in a rolling supermarket trolley with a
bungee rope and an anchor in your hand, that allows you to hook up t a truck to
get pulled along or to a lantern pole. Attached to a lantern pole you will be
on an elliptic orbit around it and due to the friction of the trolley wheels
you will be slowly spiralling ever closer to the pole. The lantern pole is
the attractor of your orbit. In PSO the attractors are points in the search
space marked by others as their best point found so far. Secondly, every agent
can have several attractors. Thirdly, the attractor influence is scaled by
random numbers, the scaling value is renewed in each time step, and it is a
different value in each dimension. But the last and perhaps most important
characteristic of PSO is the choice of attractors, and here the word swarm
comes in. A swarm is a communicating group of agents exhibiting collective
behaviour. One can imagine that two herrings swimming closely together in a
swarm are sharing information about their swimming speed and direction, and
that each herring most of the time follows the external influences and sometimes
gives an own little impulse. The unintuitive thing about PSO is that the
closeness in the information sharing network has nothing to do with the
closeness in the search space. It's like your facebook network that might
consist of friends far away. If you are an agent in a traditional PSO setup
then you have two attractors: one is the best spot in your own past trajectory
and the other one is the best memory from the dudes you usually talk to; and 
the guys you usually talk to are "the local neighbourhood of degree N in the
communication topology" which is an extra thing being created at the beginning
and remaining constant throughout the search, i.e. the guys you usually talk to
remain the same subgroup of the swarm no matter where everybody moves in the
search space.

Now thinking of attractors as in a planetary system. If you have two attractors
then you are effectively orbiting around the center of mass of the two drawing
nice and smooth orbits around it. In PSO this is not so due to the random force
scalings on the one hand and the intentionally coarse time stepping (widely
scanning the search space is what counts, and not the fine resolution of the
trajectory and the realism of some quasi-physical behaviour) on the
other hand.

One more thing. Making a tangential step forward from a position in an orbit
always increases the radius of the orbit. The radius increase is stronger the
coarser the time stepping is. This is the reason for the strong inertia damping
in PSO, which is necessary to prevent the particle speeds from diverging and
the swarm from exploding.

(There are some more explaining notes at the end of the code, justifying some
choices made for the code implementation below.)
"""
from numpy import zeros, ones, array, asfarray, mod, where, argsort, argmin, argmax, fabs
from numpy.random import rand, randint
from peabox_individual import Individual
from peabox_population import Population
#from peabox_helpers import parentselect_exp

def rand_topo_2D(m,n):
    r=rand(m*n)
    s=argsort(r)
    return s.reshape(m,n)



class PSO_Topo(object):

    def __init__(self,nh_degree):
        self.tsize=0 # topology size, i.e. how many members
        self.nhd=nh_degree    # the neighbourhood degree, i.e. only comunicating with first degree neighbours or up to 4th degree or so



class PSO_Topo_standard2D(PSO_Topo):

    def __init__(self,shape,nh_degree):
        PSO_Topo.__init__(self,nh_degree)
        dim1,dim2=shape
        self.tsize=dim1*dim2
        self.dim1=dim1        # how many lines in the topology array
        self.dim2=dim2        # how many columns in the topology array
        self.map=rand_topo_2D(dim1,dim2)  # the map, i.e. the 2D grid containing the individual numbers
        self.imap=zeros((self.tsize,2),dtype=int)  # the inverse map
        self.update_imap()
    
    def is_of_size(self,n):
        return self.dim1*self.dim2 == n
    
    def update_imap(self):
        for i in range(self.dim1):
            for j in range(self.dim2):
                self.imap[self.map[i,j],:]=i,j
    
    def get_indices_of(self,num):
        """wasteful method that should in principle always be obsolete, because the
        information is to be made available by the inverse map"""
        idx=argmin(fabs(self.map-num))
        k=idx/self.dim2; l=idx%self.dim2
        return k,l
    
    def get_neighbourhood(self,n,deg):
        k,l=self.imap[n,:] # the indices of that individual in the grid
        nbs=[]  # the list of neighbours
        for i,j in enumerate(range(k-deg,k+1)):
            jj=j%self.dim1
            if i==0:
                nbs.append(self.map[jj,l])
            else:
                nbs.append(self.map[jj,(l-i)%self.dim2])
                nbs.append(self.map[jj,(l+i)%self.dim2])
        for i,j in enumerate(range(k+deg,k,-1)):
            jj=j%self.dim1
            if i==0:
                nbs.append(self.map[jj,l])
            else:
                nbs.append(self.map[jj,(l-i)%self.dim2])
                nbs.append(self.map[jj,(l+i)%self.dim2])
        return nbs

    def get_neighbourhood_upto(self,n,deg):
        k,l=self.imap[n,:] # the indices of that individual in the grid
        nbs=[]  # the list of neighbours
        for i,j in enumerate(range(k-deg,k+1)):
            jj=j%self.dim1
            if i==0:
                nbs.append(self.map[jj,l])
            else:
                llo=(l-i)%self.dim2; lhi=(l+i)%self.dim2
                for ll in range(llo,lhi+1):
                    nbs.append(self.map[jj,ll])
        for i,j in enumerate(range(k+deg,k,-1)):
            jj=j%self.dim1
            if i==0:
                nbs.append(self.map[jj,l])
            else:
                llo=(l-i)%self.dim2; lhi=(l+i)%self.dim2
                for ll in range(llo,lhi+1):
                    nbs.append(self.map[jj,ll])
        return nbs



class PSO_Individ(Individual):

    def __init__(self,objfunc,paramspace):
        Individual.__init__(self,objfunc,paramspace)
        self.attractors=[]
        self.attweights=[]
        self.speed=zeros(self.ng,dtype=float)
        self.memory={}  # will contain 'bestx', 'bestval', 'nhbestx', and 'nhbestval
        self.nh=[]      # own neighbourhood
        self.egofade=0.5  # if 1 then don't listen to neighbourhood, if 0 then don't listen to yourself
        self.swarm=None # slot for link to own population
        self.random_influence='ND' # '1D' or 'ND'

    def delete_attractors(self):
        self.attractors=[]
        self.attweights=[]

    def add_attractor(self,coords,weight):
        self.attractors.append(asfarray(coords))
        self.attweights.append(weight)
    
    def do_step(self):
        #iniDNA=self.get_copy_of_DNA()
        self.speed*=self.swarm.alpha
        #print 'this guy {}'.format(self.no)
        #print 'DNA ',self.DNA
        #print 'attractors ',self.attractors
        for a,w in zip(self.attractors,self.attweights):
            if self.random_influence=='1D': r=rand()
            elif self.random_influence=='ND': r=rand(self.ng)
            else: raise ValueError("random_influence must be either '1D' or 'ND'")
            self.speed += w*r * (a-self.DNA); #print 'adding {}'.format(w*r * (a-self.DNA))
        self.DNA += self.speed
        #endDNA=self.get_copy_of_DNA()
        #print 'last step: {} and speed {}'.format(endDNA-iniDNA,self.speed)
    
    def update_bestx(self):
        #print 'hello from dude {} beginning update'.format(self.no)
        if self.isbetter(self.memory['bestval']):
            self.memory['bestx']=array(self.DNA,copy=1)
            self.memory['bestval']=self.score
            #print 'dude {}  has updated'.format(self.no)
    
    def update_nh(self):
        self.nh=self.swarm.get_neighbourhood_upto(self.no,self.swarm.nhd)
        
    def update_nh_bestx(self):
        for i,num in enumerate(self.nh):
            if i==0:
                nhbestx=array(self.swarm[num].memory['bestx'],copy=1)
                nhbestval=self.swarm[num].memory['bestval']
            else:
                if self.whatisfit=='minimize':
                    if self.swarm[num].memory['bestval'] < nhbestval:
                        nhbestx[:]=self.swarm[num].memory['bestx']
                        nhbestval=self.swarm[num].memory['bestval']
                elif self.whatisfit=='maximize':
                    if self.swarm[num].memory['bestval'] > nhbestval:
                        nhbestx[:]=self.swarm[num].memory['bestx']
                        nhbestval=self.swarm[num].memory['bestval']
                else:
                    raise ValueError("self.whatisfit should be either 'minimize' or 'maximize', but it is '{}'".format(self.whatisfit))
        self.memory['nhbestx']=nhbestx
        self.memory['nhbestval']=nhbestval
        #print 'dude {} updates nhbestx {} and nhbestval {}'.format(self.no,nhbestx,nhbestval)



class PSOSwarm_standard2D(Population,PSO_Topo_standard2D):

    def __init__(self,species,popsize,objfunc,paramspace,toposhape,nh_degree):
        Population.__init__(self,species,popsize,objfunc,paramspace)
        PSO_Topo_standard2D.__init__(self,toposhape,nh_degree)
        if not self.is_of_size(self.psize):
            raise ValueError("The topology shape does not fit the population size.")
        self.alpha=0.7298
        self.psi=2.9922
        self.attractor_boost=1.0
        for dude in self:
            dude.swarm=self # making individuals conscious of the general population
    
    def initialize_memories(self):
        for dude in self:
            dude.memory['bestx']=array(dude.DNA,copy=1)
            dude.memory['bestval']=dude.score
            dude.memory['nhbestx']=array(dude.DNA,copy=1)
            dude.memory['nhbestval']=dude.score
        for dude in self:
            dude.update_nh()
            dude.update_nh_bestx()

    def random_ini(self):    
        self.new_random_genes()
        self.eval_all()
        self.initialize_memories()
        self.advance_generation()

    def do_step(self):
        for dude in self:
            # empty and refill attractor list, then move
            #print '************* next dude {} ***************'.format(dude.no)
            #print 'difference between DNA and nhbestx: ',dude.memory['nhbestx']-dude.DNA
            #dude.update_bestx()
            #dude.attractors=[dude.memory['bestx']]
            #dude.attweights=[dude.egofade*(self.psi/2.)*self.attractor_boost]
            #dude.update_nh_bestx()
            #dude.attractors.append(dude.memory['nhbestx'])
            #dude.attweights.append((1.-dude.egofade)*(self.psi/2.)*self.attractor_boost)
            #print 'difference between DNA and nhbestx: ',dude.memory['nhbestx']-dude.DNA
            #print '-------------- update finished -------------'
            dude.delete_attractors()
            dude.update_bestx()
            dude.add_attractor(dude.memory['bestx'], dude.egofade*(self.psi/2.)*self.attractor_boost)
            dude.update_nh_bestx()
            dude.add_attractor(dude.memory['nhbestx'], (1.-dude.egofade)*(self.psi/2.)*self.attractor_boost)
            dude.do_step()
        self.eval_all()
        self.advance_generation()

    def advance_generation(self):
        self.mark_oldno()
        self.sort()
        self.update_no()
        for dude in self:
            k,l=self.imap[dude.oldno]
            self.map[k,l]=dude.no
        self.update_imap()
        Population.advance_generation(self)
        



"""
What is particle swarm optimisation PSO?
Imagine you roll through town sitting in a supermarket trolley, and the special
equipment in your hand is one or several bungee ropes with anchors at their ends.
This allows you to hook yourself up to cars and trucks rolling by, or to statues
lanterns, park benches and fountains. If you let go from a truck and hook
yourself up to a lantern, then you will stay in a circular but more often rather
elipptical orbit and depending on the friction of the trolley wheels you will be
spiralling ever closer towards it more and more relaxing the rubber rope. PSO
means there is a swarm of trolley riders and the poles you hook up with are
markers indicating your best found spot so far and somebody else's best spot,
or more precisely, the best spot found within a subgroup of the swarm, those
few ones you communicate with all the time, which are your neighbours in the
communication network, but which may be placed far away in the swarm in the
actual search space.

Before I started to write the code, these were the options I've been thinking of:

a) create a population and a communication topology separately, then hand it over
   to the main algorithm

b) subclassing the Population class and equip it with the communication topology
   and the corresponding functionalities

c) create a separate class for the communication topology and let the particle
   swarm inherit from both population and topology on an equal level, i.e.
   code the particle swarm as a subclass of both

I don't like (a) because you always will have to write algo.topology.update()
and algo.population.do_stuff() and I think this separation into two targets for
your commands is a deviation from the ideal case of letting one single thing,
the swarm, behave its way. If it is good to understand a swarm as a population
of agents with interaction patterns leading to a overall behaviour that can be
described as collective properties of the swarm, then I think it is good to
keep the whole thing together also in your thinking and coding.

Possibility (b) seems like not too much fun for the coding work, because from
the beginning you have to deal with this rather big population class.

I like the idea of coding up and testing the topology class separately, i.e.
taking option (c), that allows testing the thing while it is slim and little.
The only thing to pay attention to will be not to use attribute names which
already exist in the population. If there is only one little topology class one
might as well just add it to the population class while subclassing it immediately,
but it seems there are tons of options for coding up all kinds of 1D, 2D, nD
topologies or networks of whatever shapes like rings of thinly connected cloud
insulas. This means one might end up wanting to write a hierarchy of topology
classes, and during coding there needs to be testing of course. And maybe I
wanna reuse the topology stuff later outside the current context. These were my
arguments for going with option (c).
"""
