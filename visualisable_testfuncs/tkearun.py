#!python
"""
a little Tkinter app to compare the performance of diverse EAs on diverse
visualisable optimisation problems by watching optimisation histories online
and to convey the message from my conference paper:

DOI: 10.1109/CEC.2013.6557837
http://dx.doi.org/10.1109/CEC.2013.6557837
http://ieeexplore.ieee.org/xpl/articleDetails.jsp?tp=&arnumber=6557837&queryText%3Dstokmaier

Markus Stokmaier, IKET, KIT, Karlsruhe, January 2014
"""

import numpy as np
from peabox_population import Population
from vistestfuncs import FMsynth, necklace, hilly, murmeln, gf_kragtraeger
from vistestfuncs import rastrigin, sphere
from peabox_helpers import simple_searchspace
from peabox_recorder import Recorder
from EAcombos import ComboB
from SimpleRCGA import SimpleRCGA
from SimpleES import SimpleES
from comboCbeta import ComboC_beta
from tkframe import TKEA_win
import matplotlib.pyplot as plt
#plt.ion()

def dummyfunc(x):
    return np.sum(x)
    
ndim=8
ps=80

problem = 'gf_kragtraeger'
EA_type = 'CMAES'

if problem == 'FMsynth':
    space=simple_searchspace(ndim, -6.4, +6.35)
    p0=Population(FMsynth,ps,dummyfunc,space)
    p1=Population(FMsynth,ps,dummyfunc,space)
    rec=Recorder(p0)
    
elif problem == 'necklace':
    space=simple_searchspace(ndim, 0., 360.)
    p0=Population(necklace,ps,dummyfunc,space)
    p1=Population(necklace,ps,dummyfunc,space)
    rec=Recorder(p0)

elif problem == 'hilly':
    space=simple_searchspace(ndim, 0., 360.)
    p0=Population(hilly,ps,dummyfunc,space)
    p1=Population(hilly,ps,dummyfunc,space)
    rec=Recorder(p0)

elif problem == 'murmeln':
    space=simple_searchspace(ndim, 0., 360.)
    p0=Population(murmeln,ps,dummyfunc,space)
    p1=Population(murmeln,ps,dummyfunc,space)
    rec=Recorder(p0)

elif problem == 'gf_kragtraeger':
    parameterspace=(('x_9',-1.,1.),
                    ('x_10',-1.,1.),
                    ('x_11',-1.,1.),
                    ('x_12',-1.,1.),
                    ('x_13',-1.,1.),
                    ('x_14',-1.,1.),
                    ('y_9',0.,2.),
                    ('y_10',0.,2.),
                    ('y_11',0.,2.),
                    ('y_12',0.,2.),
                    ('y_13',0.,2.),
                    ('y_14',0.,2.))
    p0=Population(gf_kragtraeger,ps,dummyfunc,parameterspace)
    p1=Population(gf_kragtraeger,ps,dummyfunc,parameterspace)
    rec=Recorder(p0)

elif problem == 'rastrigin':
    space=simple_searchspace(ndim, -6., 6.)
    p0=Population(rastrigin,ps,dummyfunc,space)
    p1=Population(rastrigin,ps,dummyfunc,space)
    rec=Recorder(p0)
    
elif problem == 'sphere':
    space=simple_searchspace(ndim, -2., 2.)
    p0=Population(sphere,ps,dummyfunc,space)
    p1=Population(sphere,ps,dummyfunc,space)
    rec=Recorder(p0)
    
if EA_type == 'THEA':
    eac=ComboB(p0,p1)
    bs=eac.bunching()
    eac.set_bunchsizes(bs)
    eac.make_bunchlists()
    
    varlist=[]
    varlist.append({'name':'selpC',          'type':float,  'inival':1.0})
    varlist.append({'name':'selpD',          'type':float,  'inival':2.0})
    varlist.append({'name':'selpE',          'type':float,  'inival':2.0})
    varlist.append({'name':'selpF',          'type':float,  'inival':3.0})
    varlist.append({'name':'mstep',          'type':float,  'inival':0.18})
    varlist.append({'name':'Pm',             'type':float,  'inival':0.6})
    varlist.append({'name':'mr',             'type':float,  'inival':0.5})
    varlist.append({'name':'cigar2uniform',  'type':float,  'inival':0.3})
    varlist.append({'name':'cigar_aspect',   'type':float,  'inival':0.3})
    varlist.append({'name':'WHX2BLX',        'type':float,  'inival':0.0})
    varlist.append({'name':'DEsr',           'type':list,  'inival':[0.0,0.4]})
    
    
if EA_type == 'GA1':
    eac=ComboB(p0,p1)
    bs=[0,0,0,ps,0,0]
    eac.set_bunchsizes(bs)
    eac.make_bunchlists()
    
    varlist=[]
    varlist.append({'name':'selpD',          'type':float,  'inival':3.0})
    varlist.append({'name':'mstep',          'type':float,  'inival':0.05})
    varlist.append({'name':'Pm',             'type':float,  'inival':0.05})
    varlist.append({'name':'mr',             'type':float,  'inival':1.0})
    varlist.append({'name':'cigar2uniform',  'type':float,  'inival':0.0})
    varlist.append({'name':'anneal',         'type':float,  'inival':0.0})
    
if EA_type == 'GA2':
    eac=ComboB(p0,p1)
    bs=[0,0,0,ps/2,ps-ps/2,0]
    eac.set_bunchsizes(bs)
    eac.make_bunchlists()
    
    varlist=[]
    varlist.append({'name':'selpD',          'type':float,  'inival':3.0})
    varlist.append({'name':'selpE',          'type':float,  'inival':3.0})
    varlist.append({'name':'mstep',          'type':float,  'inival':0.05})
    varlist.append({'name':'Pm',             'type':float,  'inival':0.05})
    varlist.append({'name':'mr',             'type':float,  'inival':1.0})
    varlist.append({'name':'cigar2uniform',  'type':float,  'inival':0.0})
    varlist.append({'name':'WHX2BLX',        'type':float,  'inival':0.0})
    varlist.append({'name':'anneal',         'type':float,  'inival':0.0})
    
if EA_type == 'GA3':
    eac=SimpleRCGA(p0,p1)
    
    varlist=[]
    varlist.append({'name':'Pm',             'type':float,  'inival':0.05})
    varlist.append({'name':'muttype',        'type':str,    'inival':'randn'})
    varlist.append({'name':'mstep',          'type':float,  'inival':0.05})
    varlist.append({'name':'CO',             'type':str,    'inival':'uniform'})
    varlist.append({'name':'selec',          'type':str,    'inival':'roulette'})
    varlist.append({'name':'alpha',          'type':float,  'inival':0.5})
    varlist.append({'name':'beta',           'type':float,  'inival':0.2})
    
if EA_type == 'ES':
    p0.change_size(4)
    p1.change_size(12)
    eac=SimpleES(p0,p1)
    
    varlist=[]
    varlist.append({'name':'scheme',         'type':str,    'inival':'+'})
    varlist.append({'name':'mstep',          'type':float,  'inival':0.2})
    varlist.append({'name':'adap',           'type':str,    'inival':'1/5th-rule'})
    varlist.append({'name':'adapf',          'type':float,  'inival':1.2})
    
if EA_type == 'CMAES':
    eac=ComboC_beta(p0,p1)
    
    varlist=[]
    varlist.append({'name':'xstart',         'type':str,    'inival':'best_mu'})
    
if problem == 'FMsynth':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_type='semilogy'
    tw.acp_ylim=[1e-1,1e3]

elif problem == 'necklace':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_type='semilogy'
    tw.acp_ylim=[1e-2,1e3]

elif problem == 'hilly':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_ylim=[68,100]

elif problem == 'murmeln':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_ylim=[29,80]

elif problem == 'gf_kragtraeger':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_type='semilogy'
    tw.acp_ylim=[1e-5,1e-3]

elif problem == 'rastrigin':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_type='semilogy'
    tw.acp_ylim=[1e-1,2e2]

elif problem == 'sphere':
    tw=TKEA_win(eac,p0,rec,varlist)
    tw.acp_type='semilogy'
    tw.acp_ylim=[1e-4,1e2]


tw.acp_freq=10
tw.appear()
tw.mainloop()

