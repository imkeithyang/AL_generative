#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
#import matplotlib.pyplot as plt

import pandas as pd
import math
import pickle


# In[2]:

'''
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
'''


# In[3]:


import os


# In[4]:


import importlib
import detect_bursts
importlib.reload(detect_bursts)


# In[5]:


#import xlrd


# In[6]:


def convertIntoFloatList(str):
    #preprocess data extracted
    #remove the "[" and "]" characters
    str = str[1:-1]
    if str == '':
        return None
    return np.array(str.split(','),dtype = float)

def readData(frame):
    #convert dataframe rows into an entire list
    tp = []
    for ind in frame.index:
        inc = convertIntoFloatList(frame[ind])
        if inc is not None:
            tp.append(inc)
    return tp


# In[7]:

'''
dir = os.getcwd()
#load data from csv file
loadPath = os.path.join(dir,f'ALdata/070921_pre_stim_cleaned.csv')

df = pd.read_csv(loadPath, header = 0)
    
#take samples with stimuli P9_TenThous
Df = df.loc[df['stimuli'] == "P9_TenThous"]

neuronCols = list(Df.columns[3:])
'''


# In[8]:


#load xlsx data, extract column with largest pre_stim
def argMaxPreStim(mothNum):

    dir = os.getcwd()
    #load data from csv file
    loadPath =  os.path.join(dir,f'ALdata/timestamps_{mothNum}.csv')

    df = pd.read_csv(loadPath,header=0)

    argMaxColName = df.columns[np.argmax(df.iloc[0][2:]) + 2]
    #print(f"##########current stimuli referenced: {argMaxColName}##########")
    return argMaxColName


# In[9]:


#load csv data, extract timestamps
def loadData(mothName):

    dir = os.getcwd()
    #load data from csv file
    loadPath =  os.path.join(dir,f'ALdata/{mothName}_pre_stim_cleaned.csv')

    df = pd.read_csv(loadPath, header = 0)
    
    #take samples with stimuli giving largest pre_stim value in first row
    Df = df.loc[df['stimuli'] == argMaxPreStim(mothName)]

    tempDf = []

    neuronCols = list(Df.columns[3:])
    for neuron in neuronCols:
        curArr = readData(Df[neuron])
        tempDf += curArr

    return tempDf


# In[10]:

'''
a = loadData('070906')[0]
lInput = [0] + a
burstIndicator,finalNumBursts,finalBurstRanges,finalBurstSurprises,totalTime = detect_bursts.detectBursts(lInput,0,math.inf,2)
print(finalBurstRanges)
print(finalBurstSurprises)
print(burstIndicator)
'''

# Derive Nine Parameters

# In[10]:


#within-burst number of spikes
def withinBurstNumSpikes(burstIndicator):
    num = len(np.where(burstIndicator == 1)[0])
    percentage = num/burstIndicator.shape[0]
    return num,percentage


# In[11]:


#burst duration and inter-burst interval
def durations(timestamps,finalBurstRanges):
    burst = timestamps[finalBurstRanges[0][1]] - timestamps[finalBurstRanges[0][0]]
    #here we assume interBurst doesn't include the start-first-burst interval or the last-burst-end interval
    interBurstSt = finalBurstRanges[0][1]
    interBurst = 0

    maxSpikingFreq = (finalBurstRanges[0][1] - finalBurstRanges[0][0] + 1)/burst

    #tuple representing current burst
    for tup in finalBurstRanges[1:]:
        burstInc = timestamps[tup[1]] - timestamps[tup[0]]
        burst += burstInc
        interBurst += timestamps[tup[0]] - timestamps[interBurstSt]
        interBurstSt = tup[1]

        #count spikes within current burst
        curSpikingFreq = (tup[1]-tup[0]+1)/burstInc
        if maxSpikingFreq < curSpikingFreq:
            maxSpikingFreq = curSpikingFreq

    return burst,interBurst,maxSpikingFreq

#within-burst spiking frequency
def meanSpikingFreq(num,duration):
    return num/duration


# In[12]:


#surprise values
def surpriseEval(finalBurstSurprises):
    meanSurprise = np.mean(finalBurstSurprises)
    maxSurprise = np.max(finalBurstSurprises)
    return meanSurprise,maxSurprise


# In[13]:


#mean burst frequency
def meanBurstFreq(finalNumBursts,totalTime):
    meanburstFreq = finalNumBursts/totalTime
    return meanburstFreq


# In[15]:


#render 9 parameters for each sample (1 trial of 1 neuron)
def renderParams(timestamps,finalBurstRanges,finalBurstSurprises,burstIndicator,finalNumBursts,totalTime):
    withinBurstSpikeNum,withinBurstSpikePercentage = withinBurstNumSpikes(burstIndicator)
    duration,interBurst,maxSpikingFreq = durations(timestamps,finalBurstRanges)
    meanSpikingFreq = withinBurstSpikeNum/duration
    meanSurprise,maxSurprise = surpriseEval(finalBurstSurprises)
    meanburstFreq = meanBurstFreq(finalNumBursts,totalTime)
    return [duration,meanSpikingFreq,maxSpikingFreq,withinBurstSpikeNum,            interBurst,withinBurstSpikePercentage,meanburstFreq,meanSurprise,maxSurprise]


# In[16]:


#start collecting data for logistic regression
def collectModelData(mothNames):
    #for each moth, for each trial, for each neuron, render 9 parameters
    totalDf = []
    for mothName in mothNames:
        mothDf = loadData(mothName)
        #print(f"########current number of rows: {len(mothDf)}########")
        totalDf += mothDf
    
    #print("############all data loaded############")
    #print row number of totalDf
    #print(f"############totalDf row number{len(totalDf)}############")

    return totalDf

def formulateDataset(totalDf):
    sampleDataset = []
    for dfRow in totalDf:
        lInput = [0] + dfRow
        burstIndicator,finalNumBursts,finalBurstRanges,finalBurstSurprises,totalTime = detect_bursts.detectBursts(lInput,0,math.inf,2)
        sampleDataset.append(renderParams(lInput,finalBurstRanges,finalBurstSurprises,burstIndicator,finalNumBursts,totalTime))
    return sampleDataset


# In[17]:


mothNames = ['070906', '070913', '070921', '070922', '070924_1', '070924_2', '071002']
totalDf = collectModelData(mothNames)


# In[18]:


#formulate dataset
sampleDataset = formulateDataset(totalDf)


# In[ ]:


#save sampleDataset
with open('sampleDataset.pkl', 'wb') as f:
    pickle.dump(sampleDataset, f)


'''
# In[ ]:


#setup unsupervised classification training
#shuffle sample dataset
sampleDataset = shuffle(sampleDataset,random_state=42)

#normalization
scaler = preprocessing.StandardScaler().fit(sampleDataset)
sampleDataset = scaler.transform(sampleDataset)

def KMeansClassification(sampleData,NClusters,NInit = 10):
    #KMeans classification
    kmeans = KMeans(n_clusters=NClusters,n_init = NInit,random_state=42)
    kmeans.fit(sampleData)
    return kmeans


# In[ ]:


#KMeans classification
kmeans = KMeansClassification(sampleDataset,2)

#export trained model for later prediction if necessary

'''