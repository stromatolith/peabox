#!python
from pylab import plt
import numpy as np
from numpy import zeros, ones, size, shape, array, asfarray, linspace
from numpy import loadtxt, savetxt, median, mean, std, fabs
from os import getcwd, listdir
from os.path import join
from peabox_plotting import give_datestring

def load_cecLog(fpn):
    dat=loadtxt(fpn)
    if (dat[-2,0]==1) and (dat[-1,0]==1):
        return dat[:-1,0],dat[:-1,1],dat[:-1,2]
    else:
        return dat[:,0],dat[:,1],dat[:,2]


def latexnum(n,integ):
    if integ:
        return '${}$'.format(int(n))
    else:
        if fabs(n)<0.1:
            return '${:.3e}$'.format(n)
        elif fabs(n)<1.:
            return '${:.4f}$'.format(n)
        elif fabs(n)<10.:
            return '${:.3f}$'.format(n)
        elif fabs(n)<1000.:
            return '${:.2f}$'.format(n)
        else:
            return '${:.3e}$'.format(n)


loc =getcwd()
#plotloc=join(loc,'summary_plots')
plotloc=join(loc,'compare_summaries')
tabloc=join(loc,'tables')

subcases=range(51)
eat='eacBdLux'
datfolders=['d50a','d50b','d50c','d50d']
dfcontents=[[1,7],[8,14],[15,21],[22,28]]

ftable=zeros((11,51))
all10d=zeros((28,6))
all30d=zeros((28,6))
all50d=zeros((28,6))

for df,dfc in zip(datfolders,dfcontents):
    outloc=join(loc,df,'logs')
    olist=listdir(outloc)
    for nc in range(dfc[0],dfc[1]+1):
        x=[]; y=[]; neval=[]; loaded=[]
        for sc in subcases:
            fn=eat+'_cecLog_c{}_sc{}_short.txt'.format(str(nc).zfill(3),str(sc).zfill(3))
            if fn not in olist:
                print fn,' not there'
                raise ValueError(fn+' not in list of folder items')
            else:
                #print fn,' found'
                xdat,nevdat,ydat=load_cecLog(join(outloc,fn))
                x.append(xdat); y.append(ydat); neval.append(nevdat); loaded.append(sc)
                #print eat, nc, sc,shape(ydat)
                #x.append(array(xdat)); y.append(array(ydat)); neval.append(array(nevdat)); loaded.append(array(sc))

        x=array(x); y=array(y); neval=array(neval)
        
        seq=np.argsort(y[:,10])
        #print seq
        for i,sc in enumerate(seq):
            ftable[:,i]=y[sc,:]
        dim=df[1:-1]; fn='THEA_{}_{}.txt'.format(nc,dim)
        savetxt(join(tabloc,fn),ftable)
        
        fin=ftable[-1,:]
        if dim=='10':
            all10d[nc-1,:]=nc,fin[0],fin[-1],median(fin),mean(fin),std(fin)
        elif dim=='30':
            all30d[nc-1,:]=nc,fin[0],fin[-1],median(fin),mean(fin),std(fin)
        elif dim=='50':
            all50d[nc-1,:]=nc,fin[0],fin[-1],median(fin),mean(fin),std(fin)

#savetxt(join(tabloc,'wholetable_10d.txt'),all10d)
savetxt(join(tabloc,'wholetable_30d.txt'),all30d)

#ofile=open(join(tabloc,'wholetable_10d_b.txt'),'w')
#ofile.write('function   best   worst   median   mean   std\n')
#for line in all10d:
#    txt='{}   {}   {}   {}   {}   {}\n'.format(*list(line))
#    ofile.write(txt)
#ofile.close()
#
#ofile=open(join(tabloc,'wholetable_10d_c.txt'),'w')
#ofile.write('function   best   worst   median   mean   std\n')
#for line in all10d:
#    txt=''
#    for i,number in enumerate(line):
#        if i ==0:
#            txt+=latexnum(number,integ=True)+'  &  '
#        elif i==5:
#            txt+=latexnum(number,integ=False)+'  '
#        else:
#            txt+=latexnum(number,integ=False)+'  &  '
#    ofile.write(txt+r'\\'+'\n')
#ofile.close()

ofile=open(join(tabloc,'wholetable_50d_b.txt'),'w')
ofile.write('function   best   worst   median   mean   std\n')
for line in all50d:
    txt='{}   {}   {}   {}   {}   {}\n'.format(*list(line))
    ofile.write(txt)
ofile.close()

ofile=open(join(tabloc,'wholetable_50d_c.txt'),'w')
ofile.write('function   best   worst   median   mean   std\n')
for line in all50d:
    txt=''
    for i,number in enumerate(line):
        if i ==0:
            txt+=latexnum(number,integ=True)+'  &  '
        elif i==5:
            txt+=latexnum(number,integ=False)+'  '
        else:
            txt+=latexnum(number,integ=False)+'  &  '
    ofile.write(txt+r'\\'+'\n')
ofile.close()



