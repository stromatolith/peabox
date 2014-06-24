What is PSO?
------------

PSO (particle swarm optimisation) is a nature-inspired global search algorithm, and I would say it is clearly also an evolutionary algorithm. It is based on the idea that agents scan the search space and share information about what they see with other agents, and that this shall lead to a beneficial collective search behaviour of the swarm.

####It is a swarm with individual memories as collective attractors

What is PSO? Imagine you're sitting in a rolling supermarket trolley with a bungee rope and an anchor in your hand, that allows you to hook up to a truck to get pulled along or to a lantern pole. Attached to a lantern pole you will be on an elliptic orbit around it and due to the friction of the trolley wheels you will be slowly spiralling ever closer to the pole. The lantern pole is the attractor of your orbit. In PSO the attractors are points in the search space marked by others as their best point found so far. Secondly, every agent can have several attractors. Thirdly, the attractor influence is scaled by random numbers, the scaling value is renewed in each time step, and it is a different value in each dimension.

####The communication topology of the swarm

But the last and perhaps most important characteristic of PSO is the choice of attractors, and here the word swarm comes in. A swarm is a communicating group of agents exhibiting collective behaviour. One can imagine that two herrings swimming closely together in a swarm are sharing information about their swimming speed and direction, and that each herring most of the time follows the external influences and sometimes gives an own little impulse. The unintuitive thing about PSO is that the closeness in the information sharing network has nothing to do with the closeness in the search space. It's like your facebook network that might consist of friends far away. If you are an agent in a traditional PSO setup then you have two attractors: one is the best spot in your own past trajectory and the other one is the best memory from the dudes you usually talk to; and the guys you usually talk to are "the local neighbourhood of degree N in the communication topology" which is an extra thing being created at the beginning and remaining constant throughout the search, i.e. the guys you usually talk to remain the same subgroup of the swarm no matter where everybody moves in the search space.

####Randomness kicks you out of orbit planes

Now thinking of attractors as in a planetary system. If you were a planet having two attractors (two suns) then you would effectively be orbiting around the center of mass of the two drawing nice and smooth orbits around it. And no matter around what a planet orbits, the orbit stays in one plane. In PSO this is not so due to the random force scalings on the one hand (if the scaling of the force differs in different dimensions, then the orbit plane is not conserved, and two attractors don't have the effect of one single attractor in between them if their scalings have nothing incommon) and the intentionally coarse time stepping (widely scanning the search space is what counts, and not the fine resolution of the trajectory and the realism of some quasi-physical behaviour) on the other hand.

####Damping prevents swarm explosion

One more thing. Making a tangential step forward from a position in an orbit always increases the radius of the orbit. The radius increase is stronger the coarser the time stepping is. This is the reason for the strong inertia damping in PSO, which is necessary to prevent the particle speeds from diverging and the swarm from exploding.

####Why is PSO an EA?
With the argument that all agents live eternally, that there is no selection pressure and no survival of the fittest some have said that PSO sould not be called an EA. But the selection pressure happens on the level of information sharing. Communicating forward only the best memories and not the worse ones is a filter enacting the principle of survival of the fittest on the level of shared memories. And if we see the evolutionary process as the transmission from generation to generation and incremental mutation of meaningful genetic material (whereby the genes are the position vector components) in competition, then there is no problem arising.