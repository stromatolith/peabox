explanation (for EA newbies)
----------------------------

#### What is an evolutionary algorithm (EA) and when do I need it?

**First in one sentence:** An EA is an optimisation routine that can do parameter tuning based on the principle of the survival of the fittest and with a good portion of mutating randomness, and it allows you to efficiently find near-optimal solutions to all kinds of parameter optimisation problems which are challenging because there are many local optima distributed in irregular patterns in the search space and which otherwise let many deterministic methods of a mathematician's toolbox fail.

**And now in some more sentences:** Imagine you have a parameter tuning problem like e.g. one of these
- What are the ideal truss lengths of the various trusses in my new bridge design?
- What are the ideal sizes of various resistors, condensators, inductors in my circuit?
- What is the ideal set of decision thresholds for my lego mindstorms robot or logistics software or trading bot?
- What is the ideal combination of wingspan, -thickness, -angle, engine size, tank location for my new airplane?
- What are the ideal masses, stiffnesses and geometry parameters that give a vibrating structure the desired properties?
- What are the ideal engine firing times and directions if you want to get to Neptun with a couple of swing-by maneuvers?

Let's assume at least that you can simulate one solution, then take the next parameter combination, simulate or check again and compare and so on. If you have such a parameter tuning problem, then the parameter-tuning task, the question which parameter combination to check next, can be more or less challenging, more or less straightforward to solve. Accordingly, these examples are more or less suited to be tackled with either ab initio thinking and your wits, or by deterministic mathematical methods like Newton's method, the downhill-simplex method or gradient-testing methods, or by completely desparate methods like systematic area-by-area search space probing or even random search ... or with the help of an EA.

#### Sometimes parameter tuning is easy ...
Let's first look at the truss bridge problem where one would rather not consider an EA, because it is easy enough for other methods. Wherein lies the easiness? Imagine the x- and y-positions of the truss joints are the parameters to tune, that list of numbers shall be the DNA to mutate, and you want to find the geometry that allows the most lightweight construction for a static load case. Here it is much more efficient to go through a series of similar incremental modifications than to try out big unconnected mutation jumps into always new directions. In this case each number has one direction where things get better and one where things get worse, and these directions don't change that often when the other parameters change. During optimisation a truss construction will take a more and more regular, natural, smooth look, zigzag lines will turn into arcs, chaos into regular meshes. It's as if each joint is magnetically pulled into the right location. A physicist would say that in the search space we're running down into a potential valley and there seem to be no hilly barriers misleading the run towards the deepest point. A deterministic mathematical method continuing the downhill run can work in such search spaces and is in fact much more efficient due to not waste computer time with mutations into random directions. The airplane design problem is similarly well-behaved, there are many odd setups and with incremental steps of only improvements you can get to the really interesting ones, plus for the engineer some parameters have priority over others so one develops a multi-stage design recipe, a chain of simplified subproblems with own optimisation goals, and that brings the efficiency.

#### ... but at other times not so
But then there are the other parameter tuning problems, ones that are more challenging because there is no obvious decomposition into a recipe, ones where modifying one parameter at a time brings you nowehere, ones where there are many local optima distributed throughout the search space with regions of bad solutions separating them, ones where the curse of dimensionality (you can check three parameters in 10*10*10 trials, but how about ten parameters?) lets systematic search space scanning and random search appear ridiculous. What then?

#### Choosing the right EA
With these problems it is worth checking out two or three different EAs before investing huge amounts of thinking and working in characterising the problem. This means treating the box with the dozen knobs on it to tune as a black box and to leave the tuning work up to the controlled randomness of an evolutionary process. It means giving up trying to thouroughly understand the shape of the search space and hand this information gathering and exploration task over to a search algorithm. It means being inspired by how evolution works in nature where random mutations and selection pressure have composed masses of meaningful genome literature. It means you as the EA applicant restrict your interest about the character of the search space to a few generalities and the indirect knowledge about which EA concept deals better with it.

#### Reading vs. experience
And now comes the thing: trying EAs on a given problem often turns itself into a trial and error quest. Reading wikipedia may help you a bit with a reasonable EA choice and setup, reading scientific literature may help a bit more, but in the end it is some experience and some insight-based intuition that brings you forward. I wrote this library for the purpose of making it easy to gain that experience in python. It shall make it simple to quickly code down an algorithm you got from the literature or to compose from scratch and further develop your own algorithm ideas. It includes visualisable test functions posing more and less challenging optimisation problems for gaining intuition while watching trials and errors. And, as always with python, it shall lead to ejoyably readable code. Therefore it is not
```python
A=population.EAdata.arrays.floatDNA[k:l,:]
lims=population.searchspace.get_limits()
A[:,:]+=myfloatmutation42(A,P,sigma,limits=lims)
```
but rather
```python
bunch=population[k:l]
for dude in bunch:
    dude.mutate()
    #dude.mutate(P,sigma)
```
... and I hope I have managed to abide as much as possible to that philosophy, and that the peabox library has taken a shape that invites your EA tinkering.
