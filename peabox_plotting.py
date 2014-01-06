#!python

from os.path import join
from time import time, localtime

#from pylab import *

import numpy as np
from numpy import array, asfarray, zeros, ones, arange, flipud, linspace, prod, where
from numpy import floor, ceil, log10
from numpy.random import rand, randn, randint

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import cm

from matplotlib.colors import LinearSegmentedColormap, rgb2hex
#from matplotlib.collections import LineCollection, CircleCollection, PatchCollection
#from matplotlib.patches import Circle, Wedge, Polygon
#from matplotlib.lines import Line2D
#from matplotlib import font_manager

from peabox_population import MOPopulation, wTOPopulation


#-------------------------------------------------------------------------------
#--- part 1: colormaps ---------------------------------------------------------
#-------------------------------------------------------------------------------

# a color map from blue over almost neutrally yellowish white to red
cdict4 = {'red':  ((0.0, 0.0, 0.0),(0.25,0.0, 0.0),(0.5, 1.0, 1.0),(0.75,1.0, 1.0),(1.0, 0.3, 1.0)),
         'green': ((0.0, 0.0, 0.0),(0.25,0.0, 0.0),(0.5, 1.0, 1.0),(0.75,0.0, 0.0),(1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.0, 0.3),(0.25,1.0, 1.0),(0.5, 0.5, 0.5),(0.75,0.0, 0.0),(1.0, 0.0, 0.0))}
blue_red4 = LinearSegmentedColormap('BlueRed4', cdict4)
cdict5a = {'red':  ((0.0, 0.0, 0.0),(0.25,0.0, 0.0),(0.5, 1.0, 1.0),(0.75,1.0, 1.0),(1.0, 0.3, 1.0)),
          'green': ((0.0, 0.0, 0.2),(0.25,0.4, 0.0),(0.5, 1.0, 1.0),(0.75,0.0, 0.3),(1.0, 0.1, 0.0)),
          'blue':  ((0.0, 0.0, 0.3),(0.25,1.0, 1.0),(0.5, 0.5, 0.5),(0.75,0.0, 0.0),(1.0, 0.0, 0.0))}
blue_red5a = LinearSegmentedColormap('BlueRed5a', cdict5a)
cdict5b = {'red':  ((0.0, 0.0, 0.0),(0.25,0.0, 0.0),(0.5, 1.0, 1.0),(0.75,1.0, 1.0),(1.0, 0.3, 1.0)),
          'green': ((0.0, 0.0, 0.0),(0.25,0.0, 0.4),(0.5, 1.0, 1.0),(0.75,0.3, 0.0),(1.0, 0.0, 0.0)),
          'blue':  ((0.0, 0.0, 0.3),(0.25,1.0, 1.0),(0.5, 0.5, 0.5),(0.75,0.0, 0.0),(1.0, 0.0, 0.0))}
blue_red5b = LinearSegmentedColormap('BlueRed5b', cdict5b)
cdict5c = {'red':  ((0.0, 0.0, 0.0),(0.25,0.0, 0.0),(0.5, 1.0, 1.0),(0.75,1.0, 1.0),(1.0, 0.3, 1.0)),
          'green': ((0.0, 0.0, 0.0),(0.25,0.0, 0.2),(0.5, 1.0, 1.0),(0.75,0.2, 0.0),(1.0, 0.0, 0.0)),
          'blue':  ((0.0, 0.0, 0.3),(0.25,1.0, 1.0),(0.5, 0.5, 0.5),(0.75,0.0, 0.0),(1.0, 0.0, 0.0))}
blue_red5c = LinearSegmentedColormap('BlueRed5c', cdict5c)
cdict6b = {'red':  ((0.0, 0.0, 0.0),(0.125,0.00, 0.00),(0.25,0.0, 0.1),(0.375,0.00, 0.00),(0.5, 1.0, 1.0),(0.625,1.00, 1.00),(0.75,0.8, 1.0),(0.875,0.65, 0.75),(1.0, 0.3, 0.0)),
          'green': ((0.0, 0.0, 0.0),(0.125,0.20, 0.00),(0.25,0.0, 0.0),(0.375,0.75, 0.40),(0.5, 1.0, 1.0),(0.625,0.40, 0.75),(0.75,0.0, 0.0),(0.875,0.00, 0.20),(1.0, 0.0, 0.0)),
          'blue':  ((0.0, 0.0, 0.3),(0.125,0.75, 0.65),(0.25,1.0, 0.8),(0.375,0.75, 0.80),(0.5, 1.0, 1.0),(0.625,0.20, 0.25),(0.75,0.1, 0.0),(0.875,0.00, 0.00),(1.0, 0.0, 0.0))}
blue_red6b = LinearSegmentedColormap('BlueRed6b', cdict6b)
cdict6d = {'red':  ((0.0, 0.0, 0.0),(0.125,0.00, 0.00),(0.25,0.3, 0.0),(0.375,0.00, 0.00),(0.5, 1.0, 1.0),(0.625,1.00, 1.00),(0.75,0.8, 1.0),(0.875,0.35, 0.75),(1.0, 0.2, 0.0)),
          'green': ((0.0, 0.0, 0.0),(0.125,0.30, 0.00),(0.25,0.0, 0.4),(0.375,0.75, 0.40),(0.5, 1.0, 1.0),(0.625,0.40, 0.75),(0.75,0.2, 0.0),(0.875,0.00, 0.30),(1.0, 0.0, 0.0)),
          'blue':  ((0.0, 0.0, 0.2),(0.125,0.75, 0.35),(0.25,1.0, 0.4),(0.375,0.75, 0.80),(0.5, 1.0, 1.0),(0.625,0.20, 0.25),(0.75,0.0, 0.1),(0.875,0.00, 0.00),(1.0, 0.0, 0.0))}
blue_red6d = LinearSegmentedColormap('BlueRed6d', cdict6d)
cdict7 = {'red':  ((0.0,0.3, 0.3),(0.5,0.2, 0.2),(1.0, 0.8, 0.0)),
         'green': ((0.0,0.1, 0.1),(0.5,0.2, 0.2),(1.0, 1.0, 0.0)),
         'blue':  ((0.0,0.0, 0.0),(0.5,1.0, 1.0),(1.0, 0.9, 0.0))}
blue_red7 = LinearSegmentedColormap('BlueRed7', cdict7)

# a colormap consisting of ten little colormaps in series
#                           red-violet    blueish        gold         silver         green        braun        gelbgruen        rose        himmelblau
cdict_ac={'red':  ((0.0,0.0,0.0),(0.1,1.0,0.0),(0.2,0.6,0.3),(0.3,1.0,0.4),(0.4,1.0,0.0),(0.5,0.0,0.0),(0.6,0.4,0.2),(0.7,0.9,0.7),(0.8,0.9,0.4),(0.9,1.0,1.0),(1.0,0.0,0.0)),
         'green': ((0.0,0.0,0.0),(0.1,0.1,0.0),(0.2,0.3,0.1),(0.3,0.9,0.4),(0.4,1.0,0.3),(0.5,1.0,0.0),(0.6,0.2,0.4),(0.7,1.0,0.3),(0.8,0.5,0.4),(0.9,1.0,1.0),(1.0,0.4,0.0)),
         'blue':  ((0.0,0.0,0.0),(0.1,0.3,0.4),(0.2,1.0,0.0),(0.3,0.1,0.4),(0.4,1.0,0.1),(0.5,0.7,0.0),(0.6,0.0,0.0),(0.7,0.1,0.0),(0.8,1.0,1.0),(0.9,1.0,1.0),(1.0,0.0,0.0))}
ancestcolors = LinearSegmentedColormap('ancestcolors', cdict_ac)

# a colormap where 0.5 gives white, and where you can register even the slightest deviations upwards or downwards from 0.5
cdict_sidekick = {'red':  ((0.00, 1.0, 1.0),
                           (0.12, 0.9, 0.9),
                           (0.25, 0.5, 0.5),
                           (0.33, 0.1, 0.1),
                           (0.40, 0.2, 0.2),
                           (0.45, 0.4, 0.4),
                           (0.50, 1.0, 1.0),
                           (0.55, 0.0, 0.0),
                           (0.60, 0.1, 0.1),
                           (0.67, 0.1, 0.1),
                           (0.75, 0.0, 0.0),
                           (0.86, 0.0, 0.0),
                           (1.00, 0.9, 0.9)),
                  'green':((0.00, 1.0, 1.0),
                           (0.12, 0.4, 0.4),
                           (0.25, 0.1, 0.1),
                           (0.33, 0.0, 0.0),
                           (0.40, 0.0, 0.0),
                           (0.45, 0.0, 0.0),
                           (0.50, 1.0, 1.0),
                           (0.55, 0.4, 0.4),
                           (0.60, 0.2, 0.2),
                           (0.67, 0.1, 0.1),
                           (0.75, 0.4, 0.4),
                           (0.86, 0.7, 0.7),
                           (1.00, 1.0, 1.0)),
                  'blue': ((0.00, 0.7, 0.7),
                           (0.12, 0.0, 0.0),
                           (0.25, 0.0, 0.0),
                           (0.33, 0.0, 0.0),
                           (0.40, 0.3, 0.3),
                           (0.45, 0.0, 0.0),
                           (0.50, 1.0, 1.0),
                           (0.55, 0.4, 0.4),
                           (0.60, 0.2, 0.2),
                           (0.67, 0.5, 0.5),
                           (0.75, 0.2, 0.2),
                           (0.86, 0.0, 0.0),
                           (1.00, 0.2, 0.2))}
sidekick = LinearSegmentedColormap('sidekick', cdict_sidekick)


cdict_murmel = {'red':  ((0.0, 0.0, 0.3),(1.0,1.0, 0.0)),
               'green': ((0.0, 0.0, 0.2),(1.0,0.9, 0.0)),
               'blue':  ((0.0, 0.0, 0.0),(1.0,0.2, 1.0))}
murmelfarbe = LinearSegmentedColormap('murmelfarbe', cdict_murmel)

def show_these_colormaps(cmlist,picname):
    nmaps=len(cmlist)
    a = np.linspace(0, 1, 256).reshape(1,-1)
    a = np.vstack((a,a))
    fig = plt.figure(figsize=(5,5))
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.2, right=0.99)
    for i,m in enumerate(cmlist):
        ax = plt.subplot(nmaps, 1, i+1)
        plt.axis("off")
        plt.imshow(a, aspect='auto', cmap=m, origin='lower')
        pos = list(ax.get_position().bounds)
        fig.text(pos[0] - 0.01, pos[1], m.name, fontsize=10, horizontalalignment='right')
    plt.savefig(picname)
    plt.close()



def bluered4hex(x):
    # x must be a float ranging from 0 to 1
    r=blue_red4(x)
    return rgb2hex(r[:-1])
def bluered7hex(x):
    # x must be a float ranging from 0 to 1
    r=blue_red7(x)
    return rgb2hex(r[:-1])


def make_colorrange(vec,cmap):
    c=[]
    for v in vec:
        c.append(rgb2hex(cmap(v)[:-1]))
    return c



#-------------------------------------------------------------------------------
#--- part 2: utilities ---------------------------------------------------------
#-------------------------------------------------------------------------------

def give_datestring():
    # return something like 'November 11th 2010' corresponding to local date
    tm=localtime(time())   # tupel of year, month, day, hour, second, ...
    months=('January','February','March','April','May','June','July','August','September','October','November','December')
    year=tm[0]; month=tm[1]; day=tm[2]; #hour=tm[3]
    if day in (1,21,31):
        daysuffix='st'
    elif day in (2,22):
        daysuffix='nd'
    elif day in (3,23):
        daysuffix='rd'
    else:
        daysuffix='th'
    yearstr=str(year); monthstr=str(months[month-1]); daystr=str(day)+daysuffix
    datestr=monthstr+' '+daystr+' '+yearstr
    return datestr
    



#-------------------------------------------------------------------------------
#--- part 3: add convenient plot routines here ---------------------------------
#-------------------------------------------------------------------------------

def mstepplot(rec,path,title=None,addtext=None,xscale='linear',yscale='linear',ylimits=None,picname=None):
    p=rec.p; gg=rec.gg
    mstep=[]; mutagenes=[]
    for i,g in enumerate(gg):
        mstep.append(rec.sdat['mstep'][i])
        mutagenes.append(rec.adat['mutagenes'][i])
    mutagenes=asfarray(mutagenes); n=len(mutagenes[0,:])
    c=make_colorrange(linspace(0,0.2,n),cm.Dark2)
    ax1=plt.axes(); ax2=ax1.twinx()
    for i in range(n):
        ax2.plot(gg,mutagenes[:,i],c=c[i],alpha=0.6,lw=2)
    ax1.plot(gg,mstep,c='b',lw=2)
    if yscale=='log':
        ax1.semilogy(); ax2.semilogy()
    ax1.set_xlabel('generations'); ax1.set_ylabel('mstep'); ax2.set_ylabel('mutagenes')
    if title is None:
        title='mutation step size control parameters\ncase {0} generation {1}'.format(p.ncase,gg[-1])
    plt.title(title, fontsize=12)
    date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, horizontalalignment='right',verticalalignment='bottom', fontsize=8)
    if addtext is not None:
        plt.suptitle(addtext,x=0.02,y=0.05,horizontalalignment='left',verticalalignment='bottom',fontsize=8)
    if picname is not None:
        plt.savefig(join(path,picname))
    else:
        plt.savefig(join(path,'mstepplot_c'+str(p.ncase)+'_sc'+str(p.subcase).zfill(3)+'_g'+str(p.gg)+'.png'))
    plt.close()

def ancestryplot(reclist,ginter=None,path=None,title=None,addtext=None,textbox=None,yscale='linear',ylimits=None,
                 yoffset=None,whiggle=0,picname=None,ec='same',bg=cm.bone(0.18),suffix=''): # old: ec='k',bg='w'
    """
    instructive plot of how cloud of scores developes over time; color codes for ancestry situation
    argument reclist is expected to be a list of peabox_recorder.Recorder instances (but a Recorder instance not in a list will be handled)
    """
    if type(reclist)!=list: reclist=[reclist]
    #p=rec.p; gg=rec.gg
    x=[]; y=[]; c=[]; allgg=[]
    for rec in reclist:
        allgg+=rec.gg
        for i,g in enumerate(rec.gg):
            for s,ac in zip(rec.adat['scores'][i],rec.adat['ancestcodes'][i]):
                x.append(g)
                y.append(s)
                c.append(ancestcolors(ac))
    if rec.p.whatisfit=='minimize':
        best_score=np.min(array(y))
    else:
        best_score=np.max(array(y))
    x.append(np.min(allgg)-1); y.append(np.mean(reclist[0].adat['scores'][0])); c.append(ancestcolors(0.)) # need to cover whole interval [0,1] ...
    x.append(np.min(allgg)-1); y.append(np.mean(reclist[0].adat['scores'][0])); c.append(ancestcolors(1.)) # ... so colormap works all right
    x=flipud(array(x)); y=flipud(array(y)); c=flipud(array(c)); # why flipud? --> a zorder issue
    if whiggle: x=x+whiggle*rand(len(x))-0.5*whiggle
    if yoffset is not None:
        y+=yoffset
    if ec=='same':
        plt.scatter(x,y,marker='o',c=c,edgecolors=c,cmap=ancestcolors,zorder=True)
    else:
        plt.scatter(x,y,marker='o',c=c,edgecolors=ec,cmap=ancestcolors,zorder=True)
    ax=plt.axes(); ax.set_axis_bgcolor(bg)
    fftimes=array([rec.goal['fulfilltime'] for rec in reclist])
    gvals=array([rec.goal['goalvalue'] for rec in reclist])
    assert np.min(gvals)==np.max(gvals)
    if np.all(fftimes==-1):
        goaltext='goal (score={0}) not met'.format(rec.goal['goalvalue'])
    else:
        goal_reached=where(fftimes>=0,1,0); whichrec=list(goal_reached).index(1); rec=reclist[whichrec]
        goaltext='goal (score={0}) met after {1} generations and {2} calls'.format(rec.goal['goalvalue'],rec.goal['fulfilltime'],rec.goal['fulfillcalls'])
        plt.axvline(x=rec.goal['fulfilltime'],color='b')
    plt.suptitle(goaltext,x=0.022,y=0.015,horizontalalignment='left',verticalalignment='bottom',fontsize=8)
    if yscale=='log':
        plt.semilogy()
    if ylimits is not None:
        plt.ylim(ylimits)
    if ginter==None:
        plt.xlim(np.min(allgg)-1,np.max(allgg)+1)
    else:
        gini,gend=ginter; plt.xlim(gini,gend)
    #plt.colorbar()
    p=reclist[0].p
    if title is None:
        if isinstance(p,wTOPopulation):
            title='case {1} subcase {2} generation {3}\nwTOO with objectives {0}'.format(p.objnames,p.ncase,p.subcase,p.gg)
        if isinstance(p,MOPopulation):
            title='case {1} subcase {2} generation {3}\nMOO with objectives {0}'.format(p.objnames,p.ncase,p.subcase,p.gg)
        else:
            title='case {1} subcase {2} generation {3}\nSOO with objective {0}'.format(p.objname,p.ncase,p.subcase,p.gg)
        title+=r'  $\rightarrow$  final score = {}'.format(best_score)
    plt.title(title, fontsize=10)
    plt.xlabel('generations'); 
    if yoffset is None:
        plt.ylabel('score')
    else:
        plt.ylabel('score with offset '+str(yoffset))
    date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, ha='right',va='bottom', fontsize=8)
    if addtext is not None:
        plt.suptitle(addtext,x=0.02,y=0.04,ha='left',va='bottom',fontsize=6)
    if textbox is not None:
        boxtext,fsize=textbox
        #tbx=plt.suptitle(boxtext,x=0.93,y=0.93,ha='right',va='top',fontsize=fsize)
        tbx=plt.text(0.93,0.93,boxtext,transform=plt.axes().transAxes,ha='right',va='top',fontsize=fsize)
        tbx.set_bbox(dict(facecolor='gray', alpha=0.25))
    if path is None: path=rec.p.plotpath
    if picname is None:
        if ginter is None: picname='ancestryplot_'+reclist[-1].p.label
        else: picname='ancestryplot_c'+str(p.ncase).zfill(3)+'_sc'+str(p.subcase).zfill(3)+'_g'+str(gini)+'to'+str(gend)
    plt.savefig(join(path,picname+'_'+suffix+'.png'))
    plt.close()
    
def paretoplots(rec,path,xcrit=0,ycrit=1,colordata='alloscores',
               title=None,addtext=None,xscale='linear',yscale='linear',xlimits=None,ylimits=None,picname=None):
    # plot population cloud with respect to two of the objective functions
    # and also show Pareto front
    # arguments xcrit and ycrit must be integers or strings referring to objective names that the recorder's population has
    # scattered individuals will be colored differently according to the individual's data you choose
    # for that purpose the argument colordata must be a string matching the name of some scalar data stored
    # in rec for each individual
    if type(xcrit) is str: xcrit=rec.p.objnames.index(xcrit)
    if type(ycrit) is str: ycrit=rec.p.objnames.index(ycrit)
    for i,g, in enumerate(rec.g_all):
        x=[]; y=[]; c=[]; xpe=[]; ype=[]; xpk=[]; ypk=[]
        # xpe and ype are the points forming thePareto front; xpo and ypo form the paretooptimal point if there is one
        for j in range(len(rec.allpe[i])):
            x.append(rec.allobjvals[i][j,xcrit])
            y.append(rec.allobjvals[i][j,ycrit])
            c.append(eval('rec.'+colordata+'['+str(i)+']['+str(j)+']'))
            if rec.allpe[i][j]:
                xpe.append(rec.allobjvals[i][j,xcrit])
                ype.append(rec.allobjvals[i][j,ycrit])
            if rec.allpk[i][j]:
                xpk.append(rec.allobjvals[i][j,xcrit])
                ypk.append(rec.allobjvals[i][j,ycrit])
        x=flipud(array(x)); y=flipud(array(y)); c=flipud(array(c)) # so better individuals are in the foreground
        f=plt.figure(); a=f.add_subplot(111)
        if len(xpk): a.scatter(xpk,ypk,marker='s',s=140,edgecolor='g') # the king strictly dominating anybody else
        a.scatter(xpe,ype,marker='+',s=120,edgecolor='g') # the pareto front
        thedots=a.scatter(x,y,marker='o',c=c,cmap=cm.gist_stern,zorder=True) # all dudes
        if xscale=='log'and yscale=='log': plt.loglog()
        elif xscale=='log': plt.semilogx()
        elif yscale=='log': plt.semilogy()
        if xlimits is not None: plt.xlim(xlimits)
        if ylimits is not None: plt.ylim(ylimits)
        if title is not None: a.set_title(title, fontsize=12)
        a.set_xlabel(rec.p.objnames[xcrit]); a.set_ylabel(rec.p.objnames[ycrit])
        cb=plt.colorbar(thedots); cb.set_label(colordata)
        date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, horizontalalignment='right',verticalalignment='bottom', fontsize=8)
        if addtext is not None:
            plt.suptitle(addtext,x=0.02,y=0.02,horizontalalignment='left',verticalalignment='bottom',fontsize=8)
        if picname is not None:
            plt.savefig(join(path,picname))
        else:
            plt.savefig(join(path,'paretoplot_c'+str(rec.p.ncase)+'_sc'+str(rec.p.subcase).zfill(3)+'_g'+str(g)+'.png'))
        plt.close()

def pparetoplot(popul,path,xcrit=0,ycrit=1,colordata='score',
               title=None,addtext=None,xscale='linear',yscale='linear',xlimits=None,ylimits=None,picname=None):
    # plot population cloud with respect to two of the objective functions
    # and also show Pareto front
    # arguments xcrit and ycrit must be integers or strings referring to objective names that the recorder's population has
    # scattered individuals will be colored differently according to the individual's data you choose
    # for that purpose the argument colordata must be a string matching an attribute of the individual of scalar value
    if type(xcrit) is str: xcrit=popul.objnames.index(xcrit)
    if type(ycrit) is str: ycrit=popul.objnames.index(ycrit)
    x=[]; y=[]; c=[]; xpe=[]; ype=[]; xpk=[]; ypk=[]
    # xpe and ype are the points forming thePareto front; xpo and ypo form the paretooptimal point if there is one
    for dude in popul:
        x.append(dude.objvals[xcrit])
        y.append(dude.objvals[ycrit])
        c.append(eval('dude.'+colordata))
        if dude.paretoefficient:
            xpe.append(dude.objvals[xcrit])
            ype.append(dude.objvals[ycrit])
        if dude.paretoking:
            xpk.append(dude.objvals[xcrit])
            ypk.append(dude.objvals[ycrit])
    x=flipud(array(x)); y=flipud(array(y)); c=flipud(array(c)) # so better individuals are in the foreground
    f=plt.figure(); a=f.add_subplot(111)
    if len(xpk): a.scatter(xpk,ypk,marker='s',s=140,edgecolor='g') # the king strictly dominating anybody else
    thefront=a.scatter(xpe,ype,marker='+',s=120,edgecolor='g')
    thedots=a.scatter(x,y,marker='o',c=c,cmap=cm.gist_stern,zorder=True)
    if xscale=='log'and yscale=='log': plt.loglog()
    elif xscale=='log': plt.semilogx()
    elif yscale=='log': plt.semilogy()
    if title is not None:
        a.set_title(title, fontsize=12)
    a.set_xlabel(popul.objnames[xcrit]); a.set_ylabel(popul.objnames[ycrit])
    cb=plt.colorbar(thedots); cb.set_label(colordata)
    date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, horizontalalignment='right',verticalalignment='bottom', fontsize=8)
    if addtext is not None:
        plt.suptitle(addtext,x=0.02,y=0.02,horizontalalignment='left',verticalalignment='bottom',fontsize=8)
    if picname is not None:
        plt.savefig(join(path,picname))
    else:
        plt.savefig(join(path,'paretoplot_c'+str(popul.ncase)+'_sc'+str(popul.subcase).zfill(3)+'_g'+str(popul.gg)+'.png'))
    plt.close()

def orderplot(popul,path,picname=None,title=None):
    x=[]; y1=[]; y2=[]; y3=[]; c=[]
    for i,dude in enumerate(popul):
        x.append(i)
        y1.append(dude.no)
        y2.append(dude.oldno)
        y3.append(dude.score)
        c.append(dude.score)
    f=plt.figure(figsize=(8,12)); a1=f.add_subplot(211); a2=f.add_subplot(212)
    a1.scatter(x,y2,c=c,cmap=cm.gist_stern)
    a2.scatter(x,y3,c=c,cmap=cm.gist_stern)
    a1.set_xlabel('place in population'); a1.set_ylabel('dude.oldno')
    a2.set_xlabel('place in population'); a2.set_ylabel('dude.score')
    if title is not None: plt.figtext(0.5, 0.98,title,va='top',ha='center', color='black', weight='bold', size='large')
    if picname is not None:
        plt.savefig(join(path,picname))
    else:
        plt.savefig(join(path,'orderplot_c'+str(popul.ncase)+'_sc'+str(popul.subcase).zfill(3)+'_g'+str(popul.gg)+'.png'))
    plt.close()

def MOorderplot(popul,path,picname=None,title=None):
    x=[]; y1=[]; y2=[]; y3=[]; y4=[]; y5=[]; y6=[]; c=[]
    for i,dude in enumerate(popul):
        x.append(i)
        y1.append(dude.no)
        y2.append(dude.oldno)
        y3.append(dude.ranks[0])
        y4.append(dude.ranks[1])
        y5.append(dude.score)
        y6.append(dude.overall_rank)
        c.append(dude.score)
    f=plt.figure(figsize=(8,12)); a1=f.add_subplot(321); a2=f.add_subplot(322); a3=f.add_subplot(323); a4=f.add_subplot(324); a5=f.add_subplot(325); a6=f.add_subplot(326)
    a1.scatter(x,y1,c=c,cmap=cm.gist_stern)
    a2.scatter(x,y2,c=c,cmap=cm.gist_stern)
    a3.scatter(x,y3,c=c,cmap=cm.gist_stern)
    a4.scatter(x,y4,c=c,cmap=cm.gist_stern)
    a5.scatter(x,y5,c=c,cmap=cm.gist_stern)
    a6.scatter(x,y6,c=c,cmap=cm.gist_stern)
    a1.set_xlabel('place in population'); a1.set_ylabel('dude.no')
    a2.set_xlabel('place in population'); a2.set_ylabel('dude.oldno')
    a3.set_xlabel('place in population'); a3.set_ylabel('dude.ranks[0]')
    a4.set_xlabel('place in population'); a4.set_ylabel('dude.ranks[1]')
    a5.set_xlabel('place in population'); a5.set_ylabel('dude.score')
    a6.set_xlabel('place in population'); a6.set_ylabel('dude.overall_rank')
    if title is not None: plt.figtext(0.5, 0.98,title,va='top',ha='center', color='black', weight='bold', size='large')
    if picname is not None:
        plt.savefig(join(path,picname))
    else:
        plt.savefig(join(path,'orderplot_c'+str(popul.ncase)+'_sc'+str(popul.subcase).zfill(3)+'_g'+str(popul.gg)+'.png'))
    plt.close()


"""
still to be done:
def rankingplot(...)
    plt.pcolor(each dude's ranking)
"""



def varying_weight_orderplots(p,wkey,xvals):
    """
    recieves a wTOPopulation (weighted 2 objectives) and makes a plot for each weight setting value given in xvals;
    works with both, weighted ranking or weighted sum of objectives, depending on argument wkey being 'r' or 's'
    """
    # remember old setting
    if wkey=='s': xold=p.sumcoeffs[0]
    elif wkey=='r': xold=p.rankweights[0]
    # make the plots
    for i,x in enumerate(xvals):
        if wkey=='s':
            p.set_sumcoeffs([x,1-x]); p.update_scores(); p.sort()
        if wkey=='r':
            p.set_rankweights([x,1-x]); p.update_overall_ranks(); p.sort_for('overall_rank')
        sqdft,sqdlt=p.ranking_triangles_twoobj(x,1,wkey)
        ttxt='weighting factors: '+str(p.sumcoeffs)+'\n'
        ttxt+='sqdft = '+str(sqdft)+',  sqdlf = '+str(sqdlt)
        ttxt+=',  crit 1 '+str(prod(sqdft)/prod(sqdlt))+'\n'
        r1,r2=p.correlations_criterion(x,1,wkey)
        ttxt+='$r_{P,1}$ = '+str(r1)+',  $r_{P,2}$ = '+str(r2)
        ttxt+=',  $crit(r_{P,1},r_{P,2})$ = '+str(abs(r1-r2)*max(abs(r1),abs(r2)))
        MOorderplot(p,join(p.path,'plots2'),title=ttxt,
                    picname='var_'+wkey+'w_orderplot_nc'+str(p.ncase)+'_g'+str(p.gg).zfill(3)+'_op'+str(i).zfill(2)+'.png')
    # restore old order
    if wkey=='s':
        p.set_sumcoeffs([xold,1-xold]); p.update_scores(); p.sort()
    if wkey=='r':
        p.set_rankweights([xold,1-xold]); p.update_overall_ranks(); p.sort_for('overall_rank')
            


def fmsynthplot(problem,individual,pathname=None,title=None,addtext=None,ylimits=None):
    """plot one solution to the CEC-2011 problem no. 1, the FM-synthesis wave fitting problem"""
    DNA=individual.get_copy_of_DNA(); #a=[DNA[0],DNA[2],DNA[4]]; w=[DNA[1],DNA[3],DNA[5]]
    problem.call(DNA); t=problem.t; tgt=problem.target; wave=problem.trial
    plt.fill_between(t,tgt,wave)
    plt.plot(t,tgt,lw=2,color='r')
    plt.plot(t,wave,lw=2,color='c')
    if ylimits is not None: plt.ylim(ylimits)
    date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, horizontalalignment='right',verticalalignment='bottom', fontsize=8)
    if title is not None:
        plt.title(title)
    if addtext is not None:
        plt.suptitle(addtext,x=0.02,y=0.02,horizontalalignment='left',verticalalignment='bottom',fontsize=8)
    if pathname is not None:
        plt.savefig(pathname)
    else:
        plt.savefig('./plots/fmsynthplot_c'+str(individual.ncase).zfill(3)+'_g'+str(individual.gg).zfill(3)+'_i'+str(individual.no).zfill(2)+'.png')
    plt.close()


def find_edges(sequence,value):
    existence=where(sequence==value,1,0)
    ledges=[]; redges=[]
    if sequence[0]==value:
        #print 'hello'
        ledges.append(0)
        if sequence[1]!=value:
            redges.append(0)
    n=len(sequence)
    for i in range(1,n-2):
        if (existence[i-1]==0) and (existence[i]==1):
            ledges.append(i)
        if (existence[i]==1) and (existence[i+1]==0):
            redges.append(i)
    if sequence[-1]==value:
        redges.append(n-1)
        if sequence[-1]!=value:
            redges.append(n-1)
    #print ledges,redges
    return [[le,re] for le,re in zip(ledges,redges)]

def scoredistribplot(rec,ginter=None,path=None,title=None,addtext=None,textbox=None,yscale='linear',ylimits=None,
                     yoffset=None,whiggle=0,picname=None,suffix=''):
    p=rec.p
    myc=['g',cm.gist_rainbow(0.38),cm.bwr(0.4),cm.Blues(0.5),cm.Blues(0.9),'y',cm.cool(0.6)]
    st=array(rec.status)
    for i in range(7):
        edges=find_edges(st,i+1)
        for le,re in edges:
            plt.axvspan(rec.gg[le]-0.5,rec.gg[re]+0.5,color=myc[i],alpha=0.5)
    if yoffset is None: yoffset=0.
    plt.plot(rec.gg,asfarray(rec.score100)+yoffset,'-',color=cm.afmhot(0.6),lw=2)
    plt.plot(rec.gg,asfarray(rec.score075)+yoffset,'-',color=cm.afmhot(0.4),lw=2)
    plt.plot(rec.gg,asfarray(rec.score050)+yoffset,'-',color=cm.afmhot(0.2),lw=2)
    plt.fill_between(rec.gg,asfarray(rec.score000)+yoffset,asfarray(rec.score025)+yoffset,color='k',alpha='0.7')
    plt.plot(rec.gg,asfarray(rec.score025)+yoffset,'k-',lw=2)
    plt.plot(rec.gg,asfarray(rec.score000)+yoffset,'k-',lw=2)
    if yscale=='log':
        plt.semilogy()
    if ylimits is not None:
        plt.ylim(ylimits)
    else:
        if p.whatisfit=='minimize':
            minyval=np.min(array(rec.score000)+yoffset); maxyval=np.max(array(rec.score100)+yoffset)
        else:
            minyval=np.min(array(rec.score100)+yoffset); maxyval=np.max(array(rec.score000)+yoffset)
        if yscale=='log':
            lopower=floor(log10(minyval)); minyval=10**lopower
            hipower=ceil(log10(maxyval)); maxyval=10**hipower
        plt.ylim(minyval,maxyval)
    if ginter is not None:
        gini,gend=ginter; plt.xlim(gini,gend)
    else:
        plt.xlim(-1,rec.gg[-1]+1)
    if title is None:
        title='case {1} subcase {2} generation {3}\nSOO with objective {0}'.format(p.objname,p.ncase,p.subcase,p.gg)
        title+=r'  $\rightarrow$  final score = {}'.format(p[0].score)
    plt.title(title, fontsize=10)
    plt.xlabel('generations'); 
    if yoffset is None:
        plt.ylabel('score')
    else:
        plt.ylabel('score with offset '+str(yoffset))
    date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, horizontalalignment='right',verticalalignment='bottom', fontsize=8)
    if addtext is not None:
        plt.suptitle(addtext,x=0.02,y=0.04,horizontalalignment='left',verticalalignment='bottom',fontsize=6)
    if textbox is not None:
        boxtext,fsize=textbox
        #tbx=plt.suptitle(boxtext,x=0.93,y=0.93,ha='right',va='top',fontsize=fsize)
        tbx=plt.text(0.6,0.93,boxtext,transform=plt.axes().transAxes,ha='left',va='top',fontsize=fsize)
        tbx.set_bbox(dict(facecolor='gray', alpha=0.25))
    if path is None: path=rec.p.plotpath
    if picname is None:
        if ginter is None: picname='scoredistrib_'+p.label
        else: picname='scoredistrib_c'+str(p.ncase).zfill(3)+'_sc'+str(p.subcase).zfill(3)+'_g'+str(gini)+'to'+str(gend)
    plt.savefig(join(path,picname+'_'+suffix+'.png'))
    plt.close()


# debugging shit to be erased
#def scoredistribplot2(rec,ginter=None,path=None,title=None,addtext=None,textbox=None,yscale='linear',ylimits=None,
#                     yoffset=None,whiggle=0,picname=None,suffix=''):
#    p=rec.p
#    myc=['g',cm.gist_rainbow(0.38),cm.bwr(0.4),cm.Blues(0.5),cm.Blues(0.9),'y',cm.cool(0.6)]
#    st=array(rec.status)
#    for i in range(7):
#        edges=find_edges(st,i+1)
#        for le,re in edges:
#            plt.axvspan(rec.gg[le]-0.5,rec.gg[re]+0.5,color=myc[i],alpha=0.5)
#    if yoffset is None: yoffset=0.
#    #yoffset=0.
#    plt.plot(rec.gg,asfarray(rec.score100)+yoffset,'-',color=cm.afmhot(0.6),lw=2)
#    plt.plot(rec.gg,asfarray(rec.score075)+yoffset,'-',color=cm.afmhot(0.4),lw=2)
#    plt.plot(rec.gg,asfarray(rec.score050)+yoffset,'-',color=cm.afmhot(0.2),lw=2)
#    plt.fill_between(rec.gg,asfarray(rec.score000)+yoffset,asfarray(rec.score025)+yoffset,color='k',alpha='0.7')
#    plt.plot(rec.gg,asfarray(rec.score025)+yoffset,'k-',lw=2)
#    plt.plot(rec.gg,asfarray(rec.score000)+yoffset,'k-',lw=2)
#    if yscale=='log':
#        plt.semilogy()
#    if ylimits is not None:
#        plt.ylim(ylimits)
#    else:
#        if p.whatisfit=='minimize':
#            minyval=np.min(rec.score000); maxyval=np.max(rec.score100)
#        else:
#            minyval=np.min(rec.score100); maxyval=np.max(rec.score000)
#        print 'plot y interval: ',minyval,maxyval
#        plt.ylim(minyval,maxyval)
#    if ginter is not None:
#        gini,gend=ginter; plt.xlim(gini,gend)
#    else:
#        plt.xlim(-1,rec.gg[-1]+1)
#    if title is None:
#        title='case {1} subcase {2} generation {3}\nSOO with objective {0}'.format(p.objname,p.ncase,p.subcase,p.gg)
#        title+=r'  $\rightarrow$  final score = {}'.format(p[0].score)
#    plt.title(title, fontsize=10)
#    plt.xlabel('generations'); 
#    if yoffset is None:
#        plt.ylabel('score')
#    else:
#        plt.ylabel('score with offset '+str(yoffset))
#    date=give_datestring(); plt.suptitle(date,x=0.97,y=0.02, horizontalalignment='right',verticalalignment='bottom', fontsize=8)
#    if addtext is not None:
#        plt.suptitle(addtext,x=0.02,y=0.04,horizontalalignment='left',verticalalignment='bottom',fontsize=6)
#    if textbox is not None:
#        boxtext,fsize=textbox
#        #tbx=plt.suptitle(boxtext,x=0.93,y=0.93,ha='right',va='top',fontsize=fsize)
#        tbx=plt.text(0.6,0.93,boxtext,transform=plt.axes().transAxes,ha='left',va='top',fontsize=fsize)
#        tbx.set_bbox(dict(facecolor='gray', alpha=0.25))
#    if path is None: path=rec.p.plotpath
#    if picname is None:
#        if ginter is None: picname='scoredistrib_'+p.label
#        else: picname='scoredistrib_c'+str(p.ncase).zfill(3)+'_sc'+str(p.subcase).zfill(3)+'_g'+str(gini)+'to'+str(gend)
#    plt.savefig(join(path,picname+'_'+suffix+'.png'))
#    #plt.savefig(join(rec.p.plotpath,'sd_testplot.png'))
#    plt.close()