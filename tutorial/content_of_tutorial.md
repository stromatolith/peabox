peabox tutorial
===============


### lesson 1: simple evolution strategy (mu,lambda)-ES
- create parent and offspring populations
- create new offspring by choosing a parent and copying its DNA
- apply mutation operator
- evaluate new DNAs
- test function: n-dim. parabolic potential
- plot fitness distribution and mutation step size over time

### lesson 2: simple GA
primary goals:
- test function: unrotated n-dim. Rastrigin function
- create offspring applying CO-operator on two parents
- apply different mutation operator
- plot development of best DNA in current population over time

secondary goals:
- meaning of the term "separable problem"
- why classic RCGAs are okay on separable problems
- meaning of the term "multimodal problem"
- how a simple (mu,la)-ES fails on that

### lesson 3: methods of the Individual class
- lesson 3a: comparing individuals
- lesson 3b: recombining DNAs
- lesson 3c: mutation operators
   * plot DNA vector distributions after mutation
   * compare searchspaces: 2D, 3D, and 5D
   * mutation: (a) fixed length step in random direction, (b) normally distributed numbers
   * treating mutation steps transgressing the search domain boundary




