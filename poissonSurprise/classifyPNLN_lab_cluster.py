

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import pickle
import importlib
import detect_bursts
importlib.reload(detect_bursts)
import copy
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


# Loading ALdata


def convertIntoFloatList(str):
    #preprocess data extracted
    #remove the "[" and "]" characters
    str = str[1:-1]
    if str == '':
        return None
    return np.array(str.split(','),dtype = float)

def readData(frame,neuron,mothName):
    #convert dataframe rows into an entire list
    tp = []
    tpI = []
    for ind in frame.index:

        inc = convertIntoFloatList(frame[neuron][ind])
        #get stimuli 
        stimuli = frame['stimuli'][ind]
        if inc is not None:
            #print(inc)
            #print(stimuli,neuron,mothName)
            tp.append(inc)
            tpI.append(np.array([stimuli,neuron,mothName]))
    return tp,tpI


#load csv data, extract timestamps
def loadData(mothNum):

    dir = os.getcwd()
    #load data from csv file
    #remember to replace poissonSurprise with data when running on server
    loadPath =  os.path.join(dir,f'poissonSurprise/ALdata/{mothNum}_pre_stim_cleaned.csv')

    df = pd.read_csv(loadPath, header = 0)
    
    Df = df

    tempDf = []

    tempDfI = []

    neuronCols = list(Df.columns[3:])
    for neuron in neuronCols:
        curArr,curArrI = readData(Df,neuron,mothNum)
        tempDf += curArr
        tempDfI += curArrI

    #add an entire column of mothNums at the end of tempDf
    #tempDf = np.array(tempDf)
    #mothNums = np.full((tempDf.shape[0],1),mothNum)
    #tempDf = np.concatenate((tempDf,mothNums),axis = 1)

    return tempDf,tempDfI


# Derive Nine Parameters

#within-burst number of spikes
def withinBurstNumSpikes(burstIndicator):
    num = len(np.where(burstIndicator == 1)[0])
    percentage = num/burstIndicator.shape[0]
    return num,percentage

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

#surprise values
def surpriseEval(finalBurstSurprises):
    meanSurprise = np.mean(finalBurstSurprises)
    maxSurprise = np.max(finalBurstSurprises)
    return meanSurprise,maxSurprise

#mean burst frequency
def meanBurstFreq(finalNumBursts,totalTime):
    meanburstFreq = finalNumBursts/totalTime
    return meanburstFreq

#render 9 parameters for each sample (1 trial of 1 neuron)
def renderParams(timestamps,finalBurstRanges,finalBurstSurprises,burstIndicator,finalNumBursts,totalTime):
    withinBurstSpikeNum,withinBurstSpikePercentage = withinBurstNumSpikes(burstIndicator)
    duration,interBurst,maxSpikingFreq = durations(timestamps,finalBurstRanges)
    meanSpikingFreq = withinBurstSpikeNum/duration
    meanSurprise,maxSurprise = surpriseEval(finalBurstSurprises)
    meanburstFreq = meanBurstFreq(finalNumBursts,totalTime)
    return [duration,meanSpikingFreq,maxSpikingFreq,withinBurstSpikeNum,            interBurst,withinBurstSpikePercentage,meanburstFreq,meanSurprise,maxSurprise]

#start collecting data for logistic regression
def collectModelData(mothNames):
    #for each moth, for each trial, for each neuron, render 9 parameters
    totalDf = []
    totalDfI = []

    for mothName in mothNames:
        mothDf,mothDfI = loadData(mothName)
        print(f"########current number of rows: {len(mothDf)}########")
        totalDf += mothDf
        totalDfI += mothDfI
    
    print("############all data loaded############")
    #print row number of totalDf
    print(f"############totalDf row number{len(totalDf)}############")

    return totalDf,totalDfI

def formulateDataset(totalDf,totalDfI):
    sampleDataset = []
    sampleDatasetI = [] 
    data_no_burst = []
    data_no_burst_I = []
    for index,dfRow in enumerate(totalDf):
        lInput = [0] + dfRow
        burstIndicator,finalNumBursts,finalBurstRanges,finalBurstSurprises,totalTime \
            = detect_bursts.detectBursts(lInput,0,math.inf,2)
        if finalBurstRanges != []:
            sampleDataset.append\
                (renderParams(lInput,finalBurstRanges,\
                              finalBurstSurprises,burstIndicator,\
                                finalNumBursts,totalTime))
            sampleDatasetI.append(totalDfI[index])
        else:
            #store data and I data into a list
            #try and get max and mean poisson for such data
            data_no_burst.append(dfRow)
            data_no_burst_I.append(totalDfI[index])

    return sampleDataset,sampleDatasetI,data_no_burst,data_no_burst_I



if __name__ == "__main__":

    mothNames = ['070906', '070913', '070921', '070922', '070924_1', '070924_2', '071002']
    preDF,preDFI = collectModelData(mothNames)
    

    #formulate dataset
    sampleDataset,sampleDatasetI,data_no_burst,data_no_burst_I = formulateDataset(preDF,preDFI)

    #save sampleDataset
    #replace debug with refined when running on server
    with open('./sampleDataset_refined_4.pkl', 'wb') as f:
        pickle.dump((sampleDataset,sampleDatasetI,data_no_burst,data_no_burst_I), f)
    