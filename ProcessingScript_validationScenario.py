
# coding: utf-8

# In[1600]:


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


# In[1601]:


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


# In[1602]:


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return h


# In[1603]:


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
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


# In[1604]:


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None


# In[1605]:


def selectMeanThroughput(conn,ModuleName):
    df=pd.read_sql_query("SELECT avg(v.scalarValue) th FROM scalar v where v.scalarName='throughput:mean' and  v.scalarValue NOT NULL  and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    return df

def getMeanThroughput(path,ModuleName):
    conex=create_connection(path)
    throughput=selectMeanThroughput(conex,ModuleName)
    return throughput


# In[1606]:


def selectThroughputFromVec(conn,ModuleName):
    df=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value th FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='throughput:vector' and v.moduleName like ('%' || ? || '%') where time > 10.0",conn,params=(ModuleName,))
    return df

def selectCBRFromVec(conn,ModuleName):
    cbr0=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='cbr0:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    cbr1=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='cbr1:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    return cbr0,cbr1


def selectAvailableBandwidthFromVec(conn,ModuleName):
    tr0=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value/1000000 val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='tr0:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    tr1=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value/1000000 val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='tr1:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    return tr0,tr1


def selectDelayNonDecider(conn,ModuleName):
    delay=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='delay0:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    return delay

def selectDelayFromVec(conn,ModuleName):
    delay0=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value*1000 val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='delay0:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    delay1=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, vd.value*1000 val FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='delay1:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    return delay0,delay1


# In[1607]:


def selectCountDecisionFromVec(conn,ModuleName):
    df1=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, count(vd.value) decision FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='decision:vector' and v.moduleName like ('%' || ? || '%') where vd.value=1",conn,params=(ModuleName,))
    df0=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time, count(vd.value) decision FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='decision:vector' and v.moduleName like ('%' || ? || '%') where vd.value=0",conn,params=(ModuleName,))
    ratio1=df1["decision"]/(df1["decision"]+df0["decision"])
    ratio0=df0["decision"]/(df1["decision"]+df0["decision"])
    return ratio0,ratio1


# In[1608]:


def selectDecisionFromVec(conn,ModuleName):
    df=pd.read_sql_query("SELECT v.moduleName nodeId,vd.simTimeRaw*0.000000000001 time,vd.value decision FROM vector v INNER JOIN vectordata vd ON vd.vectorId=v.vectorId and v.vectorName='decision:vector' and v.moduleName like ('%' || ? || '%')",conn,params=(ModuleName,))
    return df


# In[1609]:


def getDelayOfNonDecider(path, ModuleName):
    conex=create_connection(path)
    delay=selectDelayFromVec(conex,ModuleName)
    return delay


# In[1610]:


def getDelay(path, ModuleName):
    conex=create_connection(path)
    delay0,delay1=selectDelayFromVec(conex,ModuleName)
    return delay0,delay1

def getCBR(path, ModuleName):
    conex=create_connection(path)
    cbr0,cbr1=selectCBRFromVec(conex,ModuleName)
    return cbr0,cbr1

def getAvailableBandwidth(path, ModuleName):
    conex=create_connection(path)
    tr0,tr1=selectAvailableBandwidthFromVec(conex,ModuleName)
    return tr0,tr1


# In[1611]:


def getDecisionFromVec(path, ModuleName):
    conex=create_connection(path)
    decision=selectDecisionFromVec(conex,ModuleName)
    return decision

def getCountDecision(path, ModuleName):
    conex=create_connection(path)
    ratio0,ratio1=selectCountDecisionFromVec(conex,ModuleName)
    return ratio0,ratio1


# In[1612]:


def setPlotSettings(ax,xlabel,ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.3),fancybox=True, shadow=True, ncol=3)
    plt.show


# ###  Example: 


resPath=str(Path.home())+'yourPathTo_Result Database' 
configName='deciderTestThroughput'
pathToResult=resPath+configName+'-0.vec'
#delay0,delay1=getDelay(pathToResult,".car[20].collectStatistics")
bw0,bw1=getAvailableBandwidth(pathToResult,".car[20].collectStatistics")
decision=getDecisionFromVec(pathToResult, ".car[20].decisionMaker")

# In[1614]:


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,8))
### Available bandwidth
ax1.plot(bw0['time'],bw0['val'],label="Interface 0")
ax1.plot(bw1['time'],bw1['val'],label="Interface 1")
setPlotSettings(ax1,"Simulation time (s)","Available bandwidth (Mbps)")
### Delay

#ax1.plot(delay0['time'],delay0['val'],label="Interface 0")
#ax1.plot(delay1['time'],delay1['val'],label="Interface 1")
#setPlotSettings(ax1,"simulation time (s)","Acess delay (ms)")

### Decision
ax2.plot(decision['time'],decision['decision'],label="Decision with TOPSIS",color='g')
setPlotSettings(ax2,"Simulation time (s) ","Decision")
plt.subplots_adjust(hspace=0.6)



