#!python
from pylab import plt
import numpy as np
from numpy import zeros, ones, size, shape, array, asfarray, linspace, loadtxt, mean, std, isnan
from os import getcwd, listdir
from os.path import join
from peabox_plotting import give_datestring

def load_cecLog(fpn):
    dat=loadtxt(fpn)
    if (dat[-2,0]==1) and (dat[-1,0]==1):
        return dat[:-1,0],dat[:-1,1],dat[:-1,2]
    else:
        return dat[:,0],dat[:,1],dat[:,2]


loc =getcwd()
#plotloc=join(loc,'summary_plots')
plotloc=join(loc,'compare_summaries')

subcases=range(51)
eat='eacBdLux'
datfolders=['d50a','d50b','d50c','d50d']
dfcontents=[[1,7],[8,14],[15,21],[22,28]]
#yldict={}
#yldict={ 1:  [1e-13,1e5],
#         4:  [1e2,1e8],
#         6:  [1e-5,1e4],
#         9:  [1e-1,1e2],
#        13:  [1e0,1e3],
#        21:  [4e1,4e3],
#        25:  [3e1,3e2],
#        27:  [1e2,2e3],
#        28:  [4e1,4e3]}

yldict={ 1:  [1e-13,1e4],
         4:  [1e-3,1e6],
         6:  [1e-5,1e3],
         9:  [1e-1,2e1],
        13:  [1e-3,1e3],
        21:  [4e1,4e3],
        25:  [3e1,3e2],
        27:  [1e2,1e3],
        28:  [4e1,2e3]}

for df,dfc in zip(datfolders,dfcontents):
    outloc=join(loc,df,'logs')
    olist=listdir(outloc)
    for nc in range(dfc[0],dfc[1]+1):
        x=[]; y=[]; neval=[]; loaded=[]
        for sc in subcases:
            fn=eat+'_cecLog_c{}_sc{}_short.txt'.format(str(nc).zfill(3),str(sc).zfill(3))
            if fn not in olist:
                print fn,' not there'
            else:
                print fn,' found'
                xdat,nevdat,ydat=load_cecLog(join(outloc,fn))
                x.append(xdat); y.append(ydat); neval.append(nevdat); loaded.append(sc)
                #print eat, nc, sc,shape(ydat)
                #x.append(array(xdat)); y.append(array(ydat)); neval.append(array(nevdat)); loaded.append(array(sc))

        x=array(x); y=array(y); neval=array(neval)
        ynz,yns=shape(y)
        for i in range(ynz):
            for j in range(yns):
                if isnan(y[i,j]):
                    if j==0:
                        raise ValueError("this shouldn't happen at the beginning")
                    else:
                        y[i,j]=y[i,j-1]
        
        if np.min(y)==0.:
            mn=1e-10
        else:
            mn='{:e}'.format(np.min(y))
            mn=1*10**int(mn[-3:])

        mx='{:e}'.format(np.max(y))
        mx=10*10**int(mx[-3:])

        ttxt='case {} and test function {}: benchmarking {}'.format(nc,nc,eat)
        ttxt+='\nbest error: mean {} and std {} after {} evaluations'.format(mean(y[:,-1]),std(y[:,-1]),mean(neval[:,-1]))
        date=give_datestring()

        plt.figure()
        for i,xval in enumerate(x[0]):
            if i==0:
                rxmin, rxmax = xval-0.004, xval+0.004
            else:
                rxmin, rxmax = xval-0.03, xval+0.03
            rymin, rymax = np.min(y[:,i]), np.max(y[:,i])
            rect = plt.Rectangle((rxmin, rymin), rxmax-rxmin, rymax-rymin, facecolor='grey',alpha=0.4)
            plt.gca().add_patch(rect)
        for i,sc in enumerate(loaded):
            plt.plot(x[i],y[i])
        if nc != 8: plt.semilogy()
        #if nc in yldict: plt.ylim(yldict[nc])
        plt.xlabel('FES / maxFES')
        plt.ylabel(r'error = $f_i(x)-f_i(x^*)$')
        plt.suptitle(ttxt,x=0.5,y=0.98, ha='center',va='top', fontsize=10)
        plt.suptitle(date,x=0.97,y=0.02, ha='right',va='bottom', fontsize=8)
        plt.savefig(join(plotloc,'allruns_c'+str(nc).zfill(3)+'_'+eat+'_'+df[:-1]+'.png'))
        plt.close()

#        for i,xval in enumerate(x[0]):
#            if i==0:
#                rxmin, rxmax = xval-0.004, xval+0.004
#            else:
#                rxmin, rxmax = xval-0.03, xval+0.03
#            rymin, rymax = np.min(y[:,i]), np.max(y[:,i])
#            rect = plt.Rectangle((rxmin, rymin), rxmax-rxmin, rymax-rymin, facecolor='grey',alpha=0.4)
#            plt.gca().add_patch(rect)
#        plt.plot(x[0,:],np.min(y,axis=0),c='c')
#        plt.plot(x[0,:],np.max(y,axis=0),c='c')
#        if nc != 8: plt.semilogy()
#        plt.xlabel('FES / maxFES')
#        plt.ylabel(r'error = $f_i(x)-f_i(x^*)$')
#        plt.suptitle(ttxt,x=0.5,y=0.98, ha='center',va='top', fontsize=10)
#        plt.suptitle(date,x=0.97,y=0.02, ha='right',va='bottom', fontsize=8)
#        plt.savefig(join(loc,'summary_plots',eat+'_allruns_c'+str(nc).zfill(3)+'b_'+eat+suffix+'b.png'))
#        plt.close()


