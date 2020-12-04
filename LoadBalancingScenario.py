
# coding: utf-8

# In[1024]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc
from numpy import linalg as LA
import csv
import sqlite3
from sqlite3 import Error
from pandas import DataFrame
from statistics import mean 
from scipy.stats import sem
from scipy.stats import t
import re
from matplotlib.pyplot import text
import statistics
import math
from pathlib import Path


# In[1025]:


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


# In[1026]:


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return h


# In[1027]:


def mapNodeId(df):
    nodeId=list()
    nodeId=[None]*len(df)
    for i in range(0,len(df)):
        s=df['nodeId'][i]
        digit=re.findall("\d+", s)
        nodeId[i]=int(digit[0])
    return nodeId

def getData(df): 
    df['numId']=mapNodeId(df)
    dfsorted=df.sort_values(["numId"])
    return(dfsorted)


# In[1028]:


def barPlot(data,ylabel,xlabels,path,save=False):
    #fig = plt.figure(figsize=(8,5))
    fig, ax = plt.subplots(figsize=(7,6))
    rects2=ax.bar(range(len(data)), data, width =0.3, color = ('lightblue','darkorange','lightgreen'),
      edgecolor = 'gray', linewidth = 1)
    plt.xticks(range(len(data)), xlabels )
    plt.ylabel(ylabel)
    autolabel(rects2,ax,"right")
    if(save==True):
        plt.savefig(path, format='pdf')
    else:
        plt.show()


# In[1029]:


def autolabel(rects,ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = round(rect.get_height(),2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*1, 1),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')
        


# In[1030]:


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None


# In[1031]:


def getDecisionCV(db_file):
    conn=create_connection(db_file)
    df_mean=pd.read_sql_query("SELECT avg(statMean) mean FROM statistic s where s.statName='decision:stats'",conn)
    df_stdDev=pd.read_sql_query("SELECT avg(statStddev) stdDev FROM statistic s where s.statName='decision:stats'",conn)
    return df_stdDev["stdDev"].squeeze(), df_mean["mean"].squeeze()


# In[1032]:


def selectDecisionFromVec(conn):
    df=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time,vd.value decision FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='decision:vector' and v.moduleName like '%.car[4].%'",conn)
    return df


# In[1033]:


#### Delay
def selectDelayFromVec(conn):
    delay0=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='delay0:vector' and v.moduleName like '%.car[0].%'",conn)
    delay1=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='delay1:vector' and v.moduleName like '%.car[0].%'",conn)
    return delay0,delay1

def selectMeanDelay(conn):
    df=pd.read_sql_query("SELECT avg(v.scalarValue)*1000 delay FROM scalar v where v.scalarName='delay:mean' and  v.scalarValue NOT NULL  and v.moduleName like '%.applLayer[0]'",conn)
    return df

def filterDelayData(path):
    conex=create_connection(path)
    meanDelay=selectMeanDelay(conex)
    return meanDelay.squeeze()


# In[1034]:


#### Throughput
def selectAVBFromVec(conn):
    tr0=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value/1000000  val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='tr0:vector' and v.moduleName like '%.car[0].%'",conn)
    tr1=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value/1000000  val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='tr1:vector' and v.moduleName like '%.car[0].%'",conn)
    return tr0,tr1

def selectMeanThroughput(conn):
    df=pd.read_sql_query("SELECT avg(v.scalarValue)/1000000 th FROM scalar v where v.scalarName='throughput:mean' and  v.scalarValue NOT NULL  and v.moduleName like '%.applLayer[0]'",conn)
    return df

def filterThData(path):
    conex=create_connection(path)
    meanTh=selectMeanThroughput(conex)
    return meanTh.squeeze()


# In[1035]:


def selectMeanCBR(conn):
    cbr0=pd.read_sql_query("SELECT avg(v.scalarValue)*100 cbr FROM scalar v where v.scalarName='cbr0:mean'",conn)
    cbr1=pd.read_sql_query("SELECT avg(v.scalarValue)*100 cbr FROM scalar v where v.scalarName='cbr1:mean'",conn)
    return cbr0,cbr1

def filterCBRData(path):
    conex=create_connection(path)
    meanCBR0,meanCBR1=selectMeanCBR(conex)
    return meanCBR0.squeeze(),meanCBR1.squeeze()


# In[1036]:


def plot2BarPlots(x,x1,ylab,xlab,title,xtickLabel,path,lab1,lab2):
    ind = np.arange(len(x))  # the x locations for the groups    
    
    width = 0.25 # the width of the bars
    fig, ax = plt.subplots(figsize=(11,10))
    rects1 = ax.bar(ind - width/2, x, width,
                label=lab1,alpha=0.7, align='center',color='lightblue',edgecolor='k')
    rects2 = ax.bar(ind + width/2, x1, width,
                label=lab2,alpha=0.7, align='center',color='orange',edgecolor='k')
    plt.xlabel(xlab) 
    ax.set_ylabel(ylab)  
    ax.set_title(title) 
    ax.set_xticks(ind)
    ax.set_xticklabels(xtickLabel) 
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.09),
          fancybox=True, shadow=True, ncol=2)
    matplotlib.rcParams.update({'font.size':12})
    autolabel(rects1,ax, "left")
    autolabel(rects2,ax,"right")
    fig.tight_layout()   
    plt.savefig(path, format='pdf')
    plt.show()
    plt.close(fig)


# In[1037]:


commonPath=str(Path.home())+'/myGitDepot'+'/HeteroSIM/HeteroSIM/simulations/examples/loadBalancing/'


# In[1038]:


def countDecisionChanges(path):
    conex=create_connection(path)
    decision=selectDecisionFromVec(conex)
    return max(decision['decision'].diff().ne(0).cumsum())


# In[1039]:


def treatData(path):
    conex=create_connection(path)
    ## Cbr
    cbr0,cbr1=selectMeanCBR(conex)
    ## Delay
    delay=selectMeanDelay(conex)
    ## th
    th= selectMeanThroughput(conex)    
    return (cbr0.squeeze(),cbr1.squeeze(),delay.squeeze(),th.squeeze())


# ### DATA treatment

# In[1040]:


def Extractrepeat(commonPath,fileName,repeat,start,end,step):
    n=len(list(range(start,end,step)))
    k=0
    
    cbr0_r=[None]*repeat
    cbr1_r=[None]*repeat
    delay_r=[None]*repeat
    th_r=[None]*repeat
    
    cbr0_m=[None]*n
    cbr1_m=[None]*n
    delay_m=[None]*n
    th_m=[None]*n
    
     
    cbr0_err=[None]*n
    cbr1_err=[None]*n
    delay_err=[None]*n
    th_err=[None]*n
    
    
    for i in range(start,end,step):
        for j in range(0,repeat):
            path=commonPath+fileName+str(i)+'-'+str(j)+'.sca'
            cbr0_r[j],cbr1_r[j],delay_r[j],th_r[j]=treatData(path)  
            #print(delay_r[j])
        ### cbr 0
        cbr0_m[k]=mean(cbr0_r)
        cbr0_err[k]=statistics.stdev(cbr0_r)
        ### cbr 1
        cbr1_m[k]=mean(cbr1_r)
        cbr1_err[k]=statistics.stdev(cbr1_r)
        ### delay
        delay_m[k]=mean(delay_r)
        delay_err[k]=statistics.stdev(delay_r)
        ### throughput
        th_m[k]=mean(th_r)
        th_err[k]=statistics.stdev(th_r)
        
        k=k+1
    return (cbr0_m,cbr0_err,cbr1_m,cbr1_err,delay_m,delay_err,th_m,th_err)


# In[1041]:


def setPlotSettings(ylab,path):
    plt.xlabel("Offered load rate (%)")
    plt.ylabel(ylab)
    plt.xticks(np.arange(0, 101, 10))
    plt.legend(loc="best")
    matplotlib.rcParams.update({'font.size':12})
    plt.savefig(path)


# In[1042]:


repeat=100
start=10
end=101
step=10
offeredLoad=list(range(start,end,step))
### Random selection
cbr0R,cbr0R_err,cbr1R,cbr1R_err,delayR,delayR_err,thR,thR_err=Extractrepeat(commonPath,'random-',repeat,start,end,step)
### DCARAT
cbr0T,cbr0T_err,cbr1T,cbr1T_err,delayT,delayT_err,thT,thT_err=Extractrepeat(commonPath,'loadBalancing-',repeat,start,end,step)


# In[1043]:


### Random selection
plt.errorbar(offeredLoad,cbr0R,yerr=cbr0R_err,label="Wlan0-random", color="royalblue",capsize = 5)
plt.errorbar(offeredLoad,cbr1R,yerr=cbr1R_err,label="Wlan1-random", linestyle="--", color="royalblue",capsize = 5)

### DCARAT
plt.errorbar(offeredLoad,cbr0T,yerr=cbr0T_err,label="Wlan0-DCARAT", color="darkorange",capsize = 5)
plt.errorbar(offeredLoad,cbr1T,yerr=cbr1T_err,label="Wlan1-DCARAT",linestyle=":", color="darkorange",capsize = 5)

setPlotSettings("Average CBR (%)", "cbr.pdf")


# ### Throughput  Vs load balancing

# In[1044]:


plt.errorbar(offeredLoad,thR,yerr=thR_err,label="Random selection",marker=".", color="royalblue",capsize = 5)
plt.errorbar(offeredLoad,thT,yerr=thT_err,label="DCARAT",marker="x",  color="darkorange",capsize = 5)
setPlotSettings("Average throughput (Mbps)", "th.pdf")


# ### Delay  Vs load balancing

# In[1045]:


plt.errorbar(offeredLoad,delayR,yerr=delayR_err,label="Random selection",marker=".",  color="royalblue",capsize = 5)
plt.errorbar(offeredLoad,delayT,yerr=delayT_err,label="DCARAT",marker="x",  color="darkorange",capsize = 5)
setPlotSettings("Average End-to-end delay (ms)", "delay.pdf")

