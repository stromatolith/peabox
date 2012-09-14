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
   * compare searchspace dimensionalities: 2D, 3D, and 5D
   * mutation: (a) fixed length step in random direction
   * mutation: (b) adding normally distributed numbers: ` + scaling_vector * randn(dim)`
   * treating mutation steps transgressing the search domain boundary

### lesson 4: my first EA homebrew: throwing simulated annealing and GA together
Assume we have a new evolutionary algorithm idea: in each generation we generate N offspring
individuals from the parent population of same size. We would like to have X% of the new trials
created following the simulated annealing concept and 100-X% by choosing 2 parents, mixing them
with a recombination operator and perhaps also applying a mutation operator. Along the way we will
somehow learn good settings of X, the annealing temperature cooldown, beneficial mutation
probabilities and distributions, usefulness of cooling down the mutation step size in parallel,
whether there should be an elite transfered untouched into the next generation or not.

Always, the sketched out code of the first day will look very different from what grew out of it
within the following days. Some will start from a procedural skech and keep it proceduarally the
whole time until the EA idea becomes a convenient function call. Others will at some point shift
to object-oriented class definitions, or some will even begin like that. This tutorial lesson
begins procedurally and ends OOP-style. I think the latter style recommends itself by the clearness
of the source code based on the logical splitting of functionalities. Note how there are no comment
lines necessary in the SAGA class definition, because what used to be the comment lines in the
initial procedural loop (chapter headings of steps A, B, C for offspring generation, closing the
generational cycle etc.) have now become the method names of the SAGA class. Among these also note
the method "simple_run": these couple lines easily added without cluttering any source code
clearness offer a convenient kick-starter for one whole algorithm run from random initialisation
to the end. The next logical step would be to add more such starters, each allowing you to explore
one new idea after just 5 more minutes of coding: one with cyclically pulsating and slowly decaying
temperature, maybe introducing a phase shift between temperature and mutation step size pulsation,
one treating a first global and a second local search phase with two different parameter settings,
and so on.

Now, after lesson 4d it should have become clear, that always a bit of work on wrapping up code
for convenient usage calls (either through class or through enveloping function definitions) is
necessary to seriously explore a new EA idea. And I think the peabox library can help go through
the process much more quickly, so you don't throw your EA idea into the trash can too early,
just because you happened to have one poor parameter setting in you initial code sketch but never
got far enough to find out.

- lesson 4a: a first algorithmic sketch
- lesson 4b: plotting population score distribution over time - introducing the recorder
- lesson 4c: more visualisation
- lesson 4d: efficient EA fine tuning - yes, but after decluttering our head and program (via outside definitions / OOP)



