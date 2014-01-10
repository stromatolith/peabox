#!python
"""
a new trial of coming up with a sort-of GUI ... a window with parameter setting
capability and a start button to kick-off the EA and a canvas for plotting
the current bes solution

(For the lack of time of learning something new, I stuck to the low-level Tkinter.)

Markus Stokmaier, IKET, KIT, Karlsruhe, January 2014
"""
import numpy as np
import numpy.random as npr
from numpy import array, log10, clip, flipud
from Tkinter import Tk, Frame, LabelFrame, Label, Button, Radiobutton, Checkbutton
from Tkinter import Entry, Canvas, Text, StringVar, IntVar, DoubleVar
from Tkinter import LEFT, RIGHT, TOP, BOTTOM, BOTH, DISABLED, NORMAL, RIDGE, END
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from peabox_population import population_like
from peabox_plotting import bluered4hex, ancestcolors

class TKEA_win(object):

    def __init__(self,ea,p,rec,varlist):
        self.vl=varlist               # list of EA attributes which you wanna make accessible via window
        self.ea=ea                    # the EA instance
        self.ea.generation_callbacks.append(self.update_solu_canvas)
        self.ea.generation_callbacks.append(self.update_mstep_bar)
        self.ea.generation_callbacks.append(self.rec_save_status)
        self.ea.generation_callbacks.append(self.acp_update)
        self.acp_type='linear'    # also allowed up to date: 'semilogy'
        self.acp_ylim=False
        self.acp_freq=1
        self.update_freq=1
        self.p=p                      # main population
        self.rec=rec
        tmppop=population_like(p,size=1)
        self.frontman=tmppop[0]       # an extra individual, the one bein plotted
    
    def appear(self):
        self.mwin=Tk()                # root or main window
        self.mwin.title('EA progress visualisation')
        self.setupFig()
        self.setupWindow()

    def setupFig(self):
        self.fig=plt.figure(figsize=(8,7), dpi=80)
        self.sp1=self.fig.add_subplot(211) # additional colorbar might be created under the name self.sp1_cb
        self.sp2=self.fig.add_subplot(212) # additional colorbar might be created under the name self.sp2_cb
        self.sp1t=self.sp1.set_title('ini')
        self.sp2t=self.sp2.set_title('ini')

    def setupWindow(self):
        self.f_inp=Frame(self.mwin); self.f_inp.pack(side='left')
        self.f_plot=Frame(self.mwin); self.f_plot.pack(side='right')
        self.c=FigureCanvasTkAgg(self.fig, master=self.f_plot); self.c.show()
        self.c.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        # setup the input area
        self.f_actn=LabelFrame(self.f_inp,text='action'); self.f_actn.grid(row=0,column=0)
        self.f_gg=Frame(self.f_actn); self.f_gg.pack()
        l_gg=Label(self.f_gg,text='generations'); l_gg.pack()
        self.e_gg = Entry(self.f_gg); self.e_gg.pack(); self.e_gg.insert(0,'40')        # number of generations
        self.b_rini=Button(self.f_actn,text='randini',command=self.randini); self.b_rini.pack()
        self.b_run1=Button(self.f_actn,text='update & run',command=self.run_with_readout); self.b_run1.pack()
        self.b_run2=Button(self.f_actn,text='run (continue)',command=self.run_no_readout); self.b_run2.pack()
        if len(self.vl): self.add_entries(self.vl)
        # draw initial plot
        self.draw_ini_solu()  # sort of setting up instructive initial geometry plot of non-optimised geometry
        self.acp_ini()        
        self.f_tb=Frame(self.f_plot); self.f_tb.pack()
        self.tb=NavigationToolbar2TkAgg(self.c,self.f_tb)
        self.c2=Canvas(self.f_inp,width=80,height=140); self.c2.grid(row=4,column=0)
        self.ini_mstep_bar()
            
    def add_entries(self,vl):
        for el in vl:
            if not hasattr(self.ea,el['name']):
                raise TypeError('you try to set up an entry for a name which is no attribute of the chosen EA')
            fr=Frame(self.f_actn); fr.pack()
            lab=Label(fr,text=el['name']); lab.pack()
            e = Entry(fr); e.pack(); e.insert(0,str(el['inival']))        # number of generations
            el['Entry']=e

    def draw_ini_solu(self):
        self.frontman.plot_into_axes(self.sp1)
        txt='initial DNA'.format(self.frontman.DNA)
        self.sp1t.set_text(txt)

    def draw_solu(self,dude):
        self.frontman.copy_DNA_of(dude,copyscore=True,copyparents=True,copyancestcode=True)
        self.frontman.evaluate()
        self.frontman.update_plot(self.sp1)
        txt='generation {}: score is {:.3f} after {} function calls'.format(self.p.gg,self.ea.bestdude.score,self.ea.tell_neval())
        #self.sp1t.set_text(txt)
        self.sp1.set_title(txt)
        self.c.draw()
    
    def mainloop(self):
        self.mwin.mainloop()
    
    def randini(self):
        self.p.reset()
        self.rec.clear()
        self.ea.bestdude=None
        self.ea.zeroth_generation(random_ini=True)
        self.draw_solu(self.ea.bestdude)
        self.c.draw()
        
    def run_with_readout(self):
        for el in self.vl:
            if el['type'] is float:
                val=float(el['Entry'].get())
                exec('self.ea.'+el['name']+'='+str(val))
            elif el['type'] is int:
                val=int(float(el['Entry'].get()))
                exec('self.ea.'+el['name']+'='+str(val))
            elif el['type'] is str:
                val=el['Entry'].get()
                exec('self.ea.'+el['name']+"='"+val+"'")
            elif el['type'] is list:
                val=el['Entry'].get()
                exec('self.ea.'+el['name']+"="+val)
                print 'string {} and what resulted {}'.format(val,eval('self.ea.'+el['name']))
            else:
                raise NotImplementedError('only float and int parameters cared for at this point')
        self.ea.run(int(float(self.e_gg.get())))
        
    def run_no_readout(self):
        self.ea.run(int(float(self.e_gg.get())))
        
    def update_solu_canvas(self,eaobj):
        if np.mod(self.p.gg,self.update_freq)==0:
            self.draw_solu(self.ea.bestdude)
        
    def ini_mstep_bar(self):
        fg_color=bluered4hex(0.36); #textcolor='white'
        self.c2.create_rectangle(43,0,57,140,fill='white',outline='white')
        mstep_barheight=int(-0.25*log10(self.ea.mstep)*140); mstep_barheight=clip(mstep_barheight,0,140)
        self.mstep_bar=self.c2.create_rectangle(43,mstep_barheight,57,140,fill='green',outline='green')
        for h in [2,35,70,105,140]:
            self.c2.create_line(40,h,60,h,width=2,fill=fg_color)
        for h,poww in zip([6,30,65,100,130],['0','-1','-2','-3','-4']):
            self.c2.create_text(20,h,text='10**'+poww,font=('Courier','6'))
    def update_mstep_bar(self,eaobj):
        mstep_barheight=int(-0.25*log10(self.ea.mstep)*140); mstep_barheight=clip(mstep_barheight,0,140)
        self.c2.coords(self.mstep_bar,43,mstep_barheight,57,140)
        self.mwin.update_idletasks()    

    def rec_save_status(self,eaobj):
        self.rec.save_status()
        
    def acp_ini(self,whiggle=0):
        x=[]; y=[]; farbe=[]
        for i,g in enumerate(self.rec.gg):
            for j in range(self.p.psize):
                x.append(g)
                y.append(self.rec.adat['scores'][i][j])
                farbe.append(self.rec.adat['ancestcodes'][i][j])
        x.append(0); y.append(0); farbe.append(0.)   # for normalisation of color map
        x.append(0); y.append(0); farbe.append(1.)   # for normalisation of color map
        x=flipud(array(x)); y=flipud(array(y)); farbe=flipud(array(farbe))
        if whiggle: x=x+whiggle*npr.rand(len(x))-0.5*whiggle
        self.acdots=self.sp2.scatter(x,y,marker='o',c=farbe,cmap=ancestcolors,zorder=True)
        if self.acp_type=='semilogy':
            self.sp2.semilogy()
        if self.acp_ylim:
            self.sp2.axis((0,self.p.gg,self.acp_ylim[0],self.acp_ylim[1]))
        else:
            self.sp2.axis((0,self.p.gg,0,np.max(y)))
    
    def acp_update(self,eaobj,whiggle=0):
        if np.mod(self.p.gg,self.acp_freq)==0:
            self.sp2.cla()
            self.acp_ini(whiggle=whiggle)
            self.c.draw()
        
        