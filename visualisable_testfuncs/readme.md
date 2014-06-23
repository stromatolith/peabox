visualisable test functions to challenge EAs
--------------------------------------------

With a nicely visualisable test function you can watch and judge your evolutionary algorithm doing its work online. But only in case the test function is challenging enough, the whole thing will be useful.

##### spoiler warning:
A solution is presented here to the problem that EA test functions never seem to be both visualisable and challenging at the same time. If you want to try to invent a similar solution without my taken thinking path restricting your creativity, then don't look at the python files in this folder. Only go through this readme and the invention task description at its end and then start inventing and experimenting yourself.

### background thought: intuitive interpretation of optical information
Your eyes got optimised through evolution. Your brain got optimised to quickly interpret what you see through millions of years of trial and error and by your own trials and errors while you were playing out in the dirt. When you climb a tree for escaping a crocodile, you should only use the stable branches and not the moldy ones or those forming a narrow and sharp chamfer with the stem.

This is why you have the ability to tell after only a quick glance which one has the better aerodynamic properties, a Tin Lizzy or a Porsche 911. This is also the reason why in most cases we find rope bridges, truss bridges, and gothic cathedrals where the principle of form follows function holds esthetic. Good engineering is when you follow your intuition as far as it works and start with the slow and expensive computer-supported optimisation as late as possible, save it up for the higher-order details with unintuitive solutions. Your intuition doesn't go that far when you need to determine e.g. the exact position and height of airflow break-away edges on a car.

### the target of EAs: nonintuitive problems
Then there are those optimisation problems where human intuition is not at all applicable, e.g. finding the optimal combination of dozens of control parameters of a chemical plant like a refinery or better yet a chemoreactor with instabilities. Another one? Finding the optimal thrust times and directions for a space vehicle doing swing-by maneuvres in our planetary system.

Here's the thing: numerical optimisers like evolutionary algorithms need to be efficient at solving these nasty and unintuitive problems, and programmers developing EAs are happy about the quickly interpretable problems, where you can tell in seconds (and without large statistics) whether a searcher explores the right directions or not.

**Therefore,** I have gone myself through working with several visualisable test functions and I've invented some own ones. This compilation is what you find in this folder. If you want to go through the invention process yourself, then don't look at the code, only go through the detailed description of the problem and the invention task below.

# detailed description

### visualisable + challenging = good!
Serious comparisons of two EAs needs large statistics over several test problems of different character. But that takes time. While you construct a new EA after having had a genial idea you make a lot of decisions on the structure of the algorithm. You make many of them intuitively and only for some you afford the time of experimental program runs. When you afford the time and effort for experimental runs there is the question of whether the experiment should be quick and dirty or a serious statistics. The more of the serious ones you make in the process, the more broken up is your creative thinking process and the longer the whole thing takes. Every interruption by a long statistics run is a drag on your creativity while inventing the algorithm and the program code. If you have a challenging and visualisable test problem at your hand then there will be more occasions of useful quick-and-dirty experiments.

The usage case: you want to check whether switching a new feature on makes your EA more efficient and during the next hour you want to check out a whole list of such new feature ideas queing up in your brain. For each EA setup you let the algorithm run 3-5 times on the visualisable test problem and you observe how quickly the current best solution plot takes shape and how the optimisation trajectories tend to look like. ...or maybe there is no progress at all in your new setup,, the EA always stalls in a certain way; then you have learnt something and maybe five minutes later you have a new general idea and three algorithm versions to try that out.

### visualisable + not challenging = bad!
Because it's such a hassle to go through serious statistics with serious test one uses shit like the sphere or the Rastrigin function to make sure along the way that the EA still does some optimisation job. There you don't do statistics, you just make sure that the numbers forming the chromosome go to zero quickly. But if you work the whole week with shitty test functions and only on Friday with something serious, what will the outcome be?

### the visualisable vs. challenging conflict
Check this out: http://www.bionik.tu-berlin.de/institut/s2anima.html
Here, Ingo Rechenberg and colleagues of the Institute of Bionics at the TU Berlin put together a few very nice real-parameter optimisation test problems with the property that they represent visualisable real-world problems. You can look at the development of the current best solution and you know immediately whether the optimiser does a good job or a bad one. I like the "Evolution eines Kragtraegers" (http://www.bionik.tu-berlin.de/institut/krag2s.exe) the most, because when watching it several times you can observe the search algorithm falling into one of several local optima each time. And after you have compared the final scores of some of them there is the situation that you are all-knowing while you watch the stupid algorithm, you know always how you would help the poor little optimiser, by pushing some of the truss joints down over the energy barrier onto the lower side of the cantilever. The amazing thing: while the EA is sweating doing its work and while the truss construction seems to whiggle rubber-like in the process, you as lazy observer begin to get a feeling for the energies involved in the elastic whiggeling, if you want, you can begin to imagine a haptic feeling of the energy barrier between the local optima, the barriers the locally searching evolution strategy in the example almost never overcomes. Imagine: in a more than 10-dimensional search space you know in what direction to go next, how far, and what type of barrier you will have to overcome to get there. It's great. Find me another 10D topology with local minima that is so easily graspable by a human brain!

**The problem:** The optimisation problems are **just way too easy!** If they were challenging, they would be such a great help, you could change your EA setup, you could watch three, four runs, and just seconds later you would know whether the code change bumped up the algorithm performance or not. Well, nobody stops you from doing it, the only thing is that knowing whether the code change bumped up the performance or not *on that class of problem* is an almost useless piece of information. The topologies of the corresponding search landscapes are too simple, e.g. for the Kragtraeger: several similarly deep local minima separated by low and smooth barriers and all in one single valley with smoothly and monotonically ascending slopes. The local minima form a somehow spiralling chain in the search space, and they're ordered according to depth along the chain.

### So here's the task:
Find a test problem with the same intuitive graspability ("solution quality judgability") by a human brain which is however seriously challenging for an optimisation algorithm, i.e. less regular distribution of local minima, separated by energy barriers of different shape and height, high enough dimensions so the "curse of dimensionality" is at play, meaning that systematic or random search is hopeless and so on. Find a test problem which is helpful for the creative EA developer in the above sense, one which allows useful quick-and-dirty testing, one where a human is all-knowing, where you are just waiting for the searcher to push the focus into the right direction and where you get almost emotionally involved if it keeps doing the wrong moves. Find a test problem where you can go and look at one optimisation history in detail and judge which mutations were good and which ones bad. The human judging on solution quality comparisons must happen within fractions of a second, it must be easy and quick like comparing two truss bridge solution candidates and not slow like comparing lists of floating point numbers. And the computation time should be short.

