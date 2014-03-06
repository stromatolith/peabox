peabox
======

an evolutionary algorithm toolbox written in python

Use it if you want to tinker around with evolutionary algorithms and you do not yet know wheter it will take you into the direction of GA, ES, PSO, DE, simulated annealing etc or if you decline that decision at all. The library is geared towards being intuitive to use and thus enabling rapid prototyping of evolutionary algorithms.

**newest features:**
- how to call the CEC-2013 test function suite, which is coded in C, from within python via ctypes
- algorithm codes: simple ES, CMA-ES, simple RCGA, scatter search, some EA combination examples


motivation
----------
Learning by doing is the only way to familiarise oneself with evolutionary algorithms. Hence, the interest in rapid algorithm prototyping.
So, wouldn't it be nice to be able to hack away like this?
~~~~~~ python
p1=Population(fitfunc,psize,searchspace)
p2=Population(fitfunc,2*psize,searchspace)

p1.new_random_genes()
p1.eval_all()
print [dude.score for dude in p1]
dude1,dude2=p1[:2]           # now we can address these two guys individually
print dude1<dude2            # wanting to implicitly compare dude1.score with dude2.score
print dude1.isbetter(dude2)  # dudes should know whether goal is to minimise or maximise score
p1.sort()                    # should be based on the isbetter operator

p2[0].copy_DNA_of(p1[0])     # conserve best

for dude in p2[1:]:
    if in_the_mood_for_CO():
        parentA,parentB=randint(psize,size=2)
        dude.CO_from(p1[parentA],p1[parentB])  # uniform crossing-over
    elif in_the_mood_for_mutation():
        dude.copy_DNA_of(p1[randint(psize)])
        dude.mutate(P,standarddeviation)       # adding a vector of normally distributed numbers as one mutation option
    elif in_the_mood_for_averaging():
        parentA,parentB=randint(psize,size=2)
        dude.become_mixture_of(p1[parentA],p1[parentB])   # center of connecting line, i.e. mean DNA vector
    elif in_the_mood_for_DE():
        pA,pB,pC=find_three_DE_parents(p1)
        dude.become_DE_child(pA,pB,pC)         # differential evolution step
    else:
        dude.whatever()   # easily invent a new dude.method() based on basal Individual's methods

p2.eval_all()
p2.sort()
~~~~~
Often it makes sense to work with test problems of which you can easily visualise a solution candidate (e.g. 2D truss bridge or FM-synthesis wave matching), which will allow you to quickly judge online whether the best solution found in a population improves over time or not. In that case I'd like to have the possibility of subclassing the `Individual` and add a specific `plot_yourself()` function.
~~~~~~ python
bestdude=p2[0]
bestdude.plot_yourself(path) 

p2.sort_for('otherprop')  # assuming each Individual has a property dude.otherprop
funnydude=p2[0]
funnydude.plot_yourself(path2)
~~~~~
And some more generally desired behaviour ...
~~~~~~ python
p3=p1+p2                  # we want to be able to merge populations
p3.sort()
p4=p3[:10]                # of course then we also want to be able to take a slice of a population

print p1[0] is p3[0]      # did the best of p1 stay the best in p3?
print p1[0] is p2[0]      # and did it stay the best in p2? no, hold on, p2 is now sorted the other way, but wait...
print p1[0] is bestdude   # ... that's how to ask that question

p3.pickle_self(path)      # being able to pickle and unpickle populations definitely makes sense
~~~~~
We want to be able to pull out important data in the form of lists or arrays:
~~~~~~ python
finalDNAs=p4.get_DNAs()
finalscores=p4.get_scores()
bestDNA=bestdude.get_copy_of_DNA()
bestDNApointer=bestdude.DNA
~~~~~
And after a couple of generations, maybe you want to plot something like this:
~~~~~~ python
p1.plot_score_history_cloud(plotpath)
p1.plot_best_DNA_development(plotpath)
~~~~~



Well, my answer was yes, it definitely would be nice to have such shortcut methods, so I coded up two classes for `Individual` and `Population` and some test function and plotting routines. I really think it made experimenting with evolutionary algorithms much easier and faster for me.

In order to share the library with you on github, these days I am stripping the code of parts too specific to my application, hoping to get rid of everything potentially annoying and conserving the core utilities of general interest.


purpose of this project
-----------------------
class library for rapid prototyping of evolutionary optimisers
 - focus on real parameter optimisation
 - flexibility implementing complex optimisation problems (by subclassing `Individual`)
 - implementing solution candidate plotting is easy (adding plot routine to Individual subclass)
 - learning about EA with the help of test functions with visualisable solution candidates (using Matplotlib)
 - experimenting with useful visualisation of EA run statistics (using Matplotlib)
 
The goal is a rapid iteration cycle (algorithm idea -> code -> testing -> new idea) for the experimenting architect of evolutionary algorithms.


current features
----------------
 - `Individual` class with operators for dealing with own and fellow DNA (i.e. copying, mutating, crossover like uniform, BLX, WHX ...)
 - `Population` class with functionality for sorting, splitting, merging, of populations ensuring freedom for EA invention
 - populations behave like a python list, so you can append and slice them
 - individuals have comparison operators looking at the fitness for asking `dude1<dude2` and `dude1.isbetter(dude2)`
 - hence, a population `p` can easily be sorted:
   * `p.sort()` sorts according to score/fitness of Individuals
   * `p.sort_for('anything')` sorts for that if individuals have a property `dude.anything`
 - mutation and recombination operators are able to respect search domain boundaries
 - some CEC-2005 test functions and some other popular test functions
 - **new:** how to call the CEC-2013 test function suite from within python via ctypes
 - **new:** basic evolution strategies (ES)
 - **new:** a basic real-coded genetic algorithm (RCGA) with several options for selection pressure
 - **new:** scatter search (SCS or SS)
 - **new:** EA combination examples, e.g. ES+GA+DE-combination
 - **new:** own folder with visualisable test functions and motivating thoughts
 - a recorder class for regularly taking notes on population status
 - utilities for plotting population histories based on data from recorder objects
 - a little tutorial

#### tutorial lessons
 - tutorial lesson 1: simple evolution strategy, a (mu,lambda)-ES
 - tutorial lesson 2: simple genetic algorithm (GA) with roulette wheel parent selection
 - tutorial lesson 3a: comparison operators of the Individual class
 - tutorial lesson 3b: old-school crossing-over operators for individual's DNAs
 - tutorial lesson 3c: mutation operators - plot DNA vector density distributions
 - tutorial lesson 4: an EA homebrew - stages of exploring a new EA concept
 - tutorial lesson 5: FM synthesis wave matching - object-oriented implementation of a real-world problem with a candidate solution plotting method


interesting features still missing
----------------------------------
**If you enjoy python coding and EAs belong to your interests, I would be happy about you joining the team!**
 - your EA idea
 - PSO
 - SaDE
 - NSGA-II
 - maybe binary DNAs would be interesting for the sake of testing the CHC-GA
 - BGA mutation operator and other popular GA operators


keywords
--------
evolutionary algorithm, evolutionary computation, evolutionary optimisation, global search, derivative-less optimisation


not featured
------------
 - binary DNA
 - operators (mutation, CO) for integer-coded DNA
 - DNA of symbols
 - DNA of variable length within one population
 - program installation --> just copy the files you need into your work folder (sorry, I didn't take the time yet to figure out how to use distutils or other such stuff)



