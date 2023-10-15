import numpy as np
import math
import matplotlib.pyplot as plt





'''
#matlab engine
import matlab.engine
eng = matlab.engine.start_matlab()
'''





from mpmath import nsum, exp, inf,fac, nstr
import math

import decimal

import pandas as pd
import math
import pickle

import os

import importlib




#Poisson Surprise Function
def poissonSurprise(numISI,meanFiringRate,interval):
    P = exp(-meanFiringRate*interval)*nsum(lambda i:(meanFiringRate*interval)**i/fac(i),[numISI,inf])
    #mitigate math domain error as we currently haven't found why it occasionally gives zero
    try:
        S = -float(decimal.Decimal(nstr(P)).log10())
    except:
        print(f"Poisson Surprise Error, current P value is: {P}")
        S = -math.inf
    return S





def potentialBursts(timeStamps,maxNumBurstSpikes,numISI):
    #numSpikes considered as an np multidimensional array
    numSpikes = timeStamps.shape[0] - 1
    #return last row value of timeStamps
    totalTime = timeStamps[numSpikes]

    #interval values by discrete difference function
    intervals = np.diff(timeStamps,n=1,axis = 0)
    #mean firing rate
    meanFreq = numSpikes/totalTime
    #mean interspike intervals
    meanISI = 1/meanFreq

    burstRanges,burstSurprises = [],[]
    #Find potential bursts:
    # 1. Check for two consecutive ISI < 0.5*mean_ISI
    # 2. Include consecutive spikes until ISI > mean_ISI
    # 3. Compute surprise for each new inclusion
    # 4. Retain the spike train that has the maximum surprise

    i = 0
    while i < numSpikes - 1:
        burstEndIndex = i
        if intervals[i] < 0.5*meanISI and intervals[i+1] < 0.5*meanISI:
            if i == 0:
                burstStartIndex = 0
            else:
                burstStartIndex = i - 1

            maxSurprise = 0
            
            period = intervals[i] + intervals[i+1]
            number = numISI
            surprise = poissonSurprise(number,meanFreq,period)
            
            if surprise >= maxSurprise:
                burstEndIndex = i + 1
                maxSurprise = surprise
            
            j = i + 2

            while j <= numSpikes-1 and intervals[j] <= meanISI and number <= maxNumBurstSpikes:
                period += intervals[j]
                number += 1
                surprise = poissonSurprise(number,meanFreq,period)

                if surprise >= maxSurprise:
                    burstEndIndex = j
                    maxSurprise = surprise

                j += 1
            
            burstRanges.append([burstStartIndex,burstEndIndex])
            burstSurprises.append(maxSurprise)
        
        i = burstEndIndex + 1
    
    return burstRanges,burstSurprises,intervals,totalTime,meanFreq,numSpikes

                





#Maximize surprise within the detected bursts by cropping spikes at the beginning
def cropBursts(burstRanges,burstSurprises,intervals,meanFreq):
    cropBurstRanges,cropBurstSurprises = [],[]
    numBursts = len(burstRanges)
    for i in range(numBursts):
        surprise = burstSurprises[i]
        maxSurprise = surprise
        cropStartIndex = burstRanges[i][0]

        startIndex = burstRanges[i][0] + 1
        endIndex = burstRanges[i][1]

        while startIndex < endIndex:
            period = sum(intervals[startIndex+1:endIndex+1])
            number = endIndex - startIndex
            surprise = poissonSurprise(number,meanFreq,period)

            if surprise >= maxSurprise:
                cropStartIndex = startIndex
                maxSurprise = surprise

            startIndex += 1
        
        cropBurstRanges.append([cropStartIndex,endIndex])
        cropBurstSurprises.append(maxSurprise)
    
    cropNumBursts = len(cropBurstRanges)

    return cropBurstRanges,cropBurstSurprises,cropNumBursts





#Retain bursts with at least 3 spikes in them.
def finalBursts(cropBurstRanges,cropBurstSurprises,cropNumBursts,minSurprise,numSpikes):   
    #create vector of zeros with same row number as timeStamps (neglecting first row as it's always zero as reference)
    burstIndicator = np.zeros(numSpikes)
    finalBurstRanges,finalBurstSurprises = [],[]
    for i in range(cropNumBursts):
        numBurstSpikes = cropBurstRanges[i][1] - cropBurstRanges[i][0] + 1
        surprise = cropBurstSurprises[i]
        if numBurstSpikes >= 3 and surprise >= minSurprise:
            finalBurstRanges.append(cropBurstRanges[i])
            finalBurstSurprises.append(cropBurstSurprises[i])

            burstIndicator[cropBurstRanges[i][0]:cropBurstRanges[i][1]+1] = 1
    
    finalNumBursts = len(finalBurstRanges)

    return burstIndicator,finalBurstRanges,finalBurstSurprises,finalNumBursts





def detectBursts(timeStamps,minSurprise,maxNumBurstSpikes,numISI):
    burstRanges,burstSurprises,intervals,totalTime,meanFreq,numSpikes = potentialBursts(timeStamps,maxNumBurstSpikes,numISI)

    cropBurstRanges,cropBurstSurprises,cropNumBursts = cropBursts(burstRanges,burstSurprises,intervals,meanFreq)
    
    burstIndicator,finalBurstRanges,finalBurstSurprises,finalNumBursts =          finalBursts(cropBurstRanges,cropBurstSurprises,cropNumBursts,minSurprise,numSpikes)
    
    return burstIndicator,finalNumBursts,finalBurstRanges,finalBurstSurprises,totalTime




