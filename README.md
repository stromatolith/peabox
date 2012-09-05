peabox
======

an evolutionary algorithm toolbox written in python

motivation
==========
Learning by doing is the only way to familiarise oneself with evolutionary algorithms. Hence, the interest in rapid algorithm prototyping.
So, wouldn't it be nice to be able to hack away like this?


    p1=Population(fitfunc,psize,searchspace)
    p2=Population(fitfunc,2*psize,searchspace)
    
    p1.new_random_genes()
    p1.eval_all()
    [print dude.score for dude in p1]
    dude1,dude2=p1[:2]           # now we can address these two guys individually
    print dude1<dude2            # wanting to implicitly compare dude1.score with dude2.score
    print dude1.isbetter(dude2)  # dudes should know whether goal is to minimise or maximise score
    p1.sort()                    # should be based on the isbetter operator
    
    p2[0].copy_DNA_of(p1[0])     # conserve best
    
    for dude in p2[1:]:
        if in_the_mood_for_CO():
            parentA,parentB=randint(psize,size=2)
            dude.CO_from(p1[parentA],p1[parentB])
        elif in_the_mood_for_mutation():
            dude.copy_DNA_of(p1[randint(psize)])
            dude.mutate(P,standarddeviation)
        elif in_the_mood_for_averaging():
            parentA,parentB=randint(psize,size=2)
            dude.become_mixture_of(p1[parentA],p1[parentB])
        elif in_the_mood_for_DE():
            pA,pB,pC=find_three_DE_parents(p1)
            dude.become_DE_child(pA,pB,pC)
        else:
            dude.whatever()
    
    p2.eval_all()
    p2.sort()

    # often it makes sense with to work with test problems of which you can easily visualise
    # a solution candidate, e.g. 2D truss bridge or FM-synthesis wave matching
    bestdude=p2[0]
    bestdude.plot_yourself(path) 
    
    p2.sort_for('otherprop')  # assuming each Individual has a property dude.otherprop
    funnydude=p2[0]
    funnydude.plot_yourself(path2)
    
    p3=p1+p2                  # we want to be able to merge populations
    p3.sort()
    p4=p3[:10]                # of course then we also want to be able to take a slice of a population
    
    print p1[0] is p3[0]      # did the best of p1 stay the best in p3?
    print p1[0] is p2[0]      # and did it stay the best in p2? no, hold on, p2 is now sorted the other way, but wait...
    print p1[0] is bestdude   # ... that's how to ask that question
    
    p3.pickle_self(path)      # being able to pickle and unpickle populations definitely makes sense
    
    
    finalDNAs=p4.get_DNAs()
    finalscores=p4.get_scores()

And after a couple of generations, maybe you want to plot something like this:

    p1.plot_score_history_cloud(plotpath)
    p1.plot_best_DNA_development(plotpath)




Well, my answer was yes, it definitely would be nice to have such shortcut methods, so I coded up two classes for `Individual` and `Population` and some test function and plotting routines. I really think it made experimenting with evolutionary algorithms much easier for me.

In order to share the library with you on github, these days I am stripping the code of parts too specific to my application, hoping to get rid of everything potentially annoying and conserving the core utilities of general interest.


current features
----------------
 - Individual class with operators for dealing with own and fellow DNA (i.e. copying, mutating, crossover like uniform, BLX, WHX ...)
 - Population class with functionality for sorting, merging, slicing, pulling array data (and threaded evaluation on a low level)


features to be added soon
-------------------------
 - some CEC-2005 test functions and some other popular test functions
 - popular EAs as class definition: evolution strategy (ES), genetic algorithm (GA), differential evolution (DE), scatter search (SCS or SS)
 - a recorder class for regularly taking notes on population status
 - utilities for plotting population histories based on data from recorder objects



finally, just testing how to get python syntax highlighting in this readme file
-------------------------------------------------------------------------------
~~~~~~ python
from pylab import *
a=randint(100,size=12).reshape(3,4)
b=3.3
c=max(b,1e4)
d=np.max(a,axis=0)
print 'Hallo Welt!'
print 'max is {}'.format(c)
for i in range(3):
    a+=1
~~~~~
