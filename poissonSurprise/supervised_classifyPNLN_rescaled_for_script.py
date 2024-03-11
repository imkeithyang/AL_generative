import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math
import importlib
import detect_bursts
importlib.reload(detect_bursts)
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import optuna


# Load Unlabeled Data

def convertIntoFloatList(str):
    #preprocess data extracted
    #remove the "[" and "]" characters
    str = str[1:-1]
    if str == '':
        return None
    return np.array(str.split(','),dtype = float)

def readData(frame,mothName):
    #convert dataframe rows into an entire list
    tp = []
    nameArr = []
    for ind in frame.index:
        inc = convertIntoFloatList(frame[ind])
        if inc is not None:
            tp.append(inc)
            nameArr.append(mothName)
    return tp,nameArr

#load xlsx data, extract column with largest pre_stim
def argMaxPreStim(mothNum):

    #load data from csv file
    loadPath =  f'ALdata/timestamps_{mothNum}.csv'

    df = pd.read_csv(loadPath,header=0)

    argMaxColName = df.columns[np.argmax(df.iloc[0][2:]) + 2]
    print(f"##########current stimuli referenced: {argMaxColName}##########")
    return argMaxColName

#load xlsx data, extract column with smallest pre_stim
def argMinPreStim(mothNum):

    #load data from csv file
    loadPath =  f'ALdata/timestamps_{mothNum}.csv'

    df = pd.read_csv(loadPath,header=0)

    argMinColName = df.columns[np.argmin(df.iloc[0][2:]) + 2]
    print(f"##########current stimuli referenced: {argMinColName}##########")
    return argMinColName

#load csv data, extract timestamps
def loadData(mothNum):
    #load data from csv file
    loadPath =  f'ALdata/{mothNum}_spontaneous.csv'

    df = pd.read_csv(loadPath, header = 0)

    tempDf = []
    tempName = []
    tempNeuron = []
    tempStimuli = []

    neuronCols = list(df.columns[3:])
    stimuli = df['stimuli'][0]
    
    for neuron in neuronCols:
        curArr,nameArr = readData(df[neuron],mothNum)
        tempDf += curArr
        tempName += nameArr
        tempNeuron += [neuron] * len(curArr)
        tempStimuli += [stimuli] * len(curArr)

    return tempDf,tempName,tempNeuron,tempStimuli


#start collecting data for logistic regression
def collectModelData(mothNames):
    #for each moth, for each trial, for each neuron, render 9 parameters
    totalDf = []
    totalName = []
    totalNeuron = []
    totalStimuli = []
    for mothName in mothNames:
        mothDf,mothNameArr,mothNeuronArr,mothStimuliArr = loadData(mothName)
        print(f"########current number of rows: {len(mothDf)}########")
        totalDf += mothDf
        totalName += mothNameArr
        totalNeuron += mothNeuronArr
        totalStimuli += mothStimuliArr
    
    print("############all data loaded############")
    #print row number of totalDf
    print(f"############totalDf row number{len(totalDf)}############")

    return totalDf,totalName,totalNeuron,totalStimuli

#GIT data for unsupervised learning
mothNames = ['070906', '070913', '070921', '070922', '070924_1', '070924_2', '071002']
totalDf,totalName,totalNeuron,totalStimuli = collectModelData(mothNames)



# Derive Nine Parameters
#from timestamp data count average spike frequency (spike count/duration)
def countSpikingFreq(timestamps):
    sum_duration = 0
    spike_count = 0
    for timestamp in timestamps:
        sum_duration += timestamp[-1] - timestamp[0]
        spike_count += len(timestamp)
    return spike_count/sum_duration

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

    # burst /= 1000
    # interBurst /= 1000
    # maxSpikingFreq *= 1000
    return burst,interBurst,maxSpikingFreq

#within-burst spiking frequency
def meanSpikingFreq(num,duration):
    return num/duration
    # return num/duration * 1000

#surprise values
def surpriseEval(finalBurstSurprises):
    meanSurprise = np.mean(finalBurstSurprises)
    maxSurprise = np.max(finalBurstSurprises)
    return meanSurprise,maxSurprise

#mean burst frequency
def meanBurstFreq(finalNumBursts,totalTime):
    meanburstFreq = finalNumBursts/totalTime
    # meanburstFreq = meanburstFreq* 1000
    return meanburstFreq

#render 9 parameters for each sample (1 trial of 1 neuron)
def renderParams(timestamps,finalBurstRanges,finalBurstSurprises,burstIndicator,finalNumBursts,totalTime):
    withinBurstSpikeNum,withinBurstSpikePercentage = withinBurstNumSpikes(burstIndicator)
    duration,interBurst,maxSpikingFreq = durations(timestamps,finalBurstRanges)
    meanSpikingFreq = withinBurstSpikeNum/duration
    meanSurprise,maxSurprise = surpriseEval(finalBurstSurprises)
    meanburstFreq = meanBurstFreq(finalNumBursts,totalTime)
    return [duration,meanSpikingFreq,maxSpikingFreq,withinBurstSpikeNum,\
            interBurst,withinBurstSpikePercentage,meanburstFreq,meanSurprise,maxSurprise]


def formulateDataset(totalDf,totalName = None,totalLabel = None,p = 0.5):
    sampleDataset = []
    nameRes = []
    labelRes = []

    no_burst_sampleDataset = []
    no_burst_nameRes = []
    no_burst_labelRes = []

    if totalName is None:
        totalName = ['']*(totalDf.shape[0])

    if totalLabel is None:
        totalLabel = [0]*(totalDf.shape[0])
        
    for index,dfRow in enumerate(totalDf):
        lInput = [0] + dfRow
        burstIndicator,finalNumBursts,finalBurstRanges,\
            finalBurstSurprises,totalTime = detect_bursts.detectBursts(lInput,0,math.inf,2,p)
        
        if finalBurstRanges != []:
            sampleDataset.append(renderParams(lInput,finalBurstRanges,\
                                              finalBurstSurprises,burstIndicator,finalNumBursts,totalTime))
            nameRes.append(totalName[index])
            labelRes.append(totalLabel[index])

        else:
            #no burst in current list of timestamps
            no_burst_sampleDataset.append(dfRow)
            no_burst_nameRes.append(totalName[index])
            no_burst_labelRes.append(totalLabel[index])

    return sampleDataset,nameRes,labelRes,no_burst_sampleDataset,no_burst_nameRes,no_burst_labelRes



def formulateDataset_unlabeled(totalDf,totalName = None,totalNeuron = None,totalStimuli = None,p = 0.5):
    sampleDataset = []
    nameRes = []
    neuronRes = []
    stimuliRes = []

    no_burst_sampleDataset = []
    no_burst_nameRes = []
    no_burst_neuronRes = []
    no_burst_stimuliRes = []


    if totalName is None:
        totalName = ['']*(len(totalDf))

    
    if totalNeuron is None:
        totalNeuron = ['']*(len(totalDf))
    
    if totalStimuli is None:
        totalStimuli = ['']*(len(totalDf))
        
    for index,dfRow in enumerate(totalDf):
        lInput = [0] + dfRow
        burstIndicator,finalNumBursts,finalBurstRanges,\
            finalBurstSurprises,totalTime = detect_bursts.detectBursts(lInput,0,math.inf,2,p)
        
        if finalBurstRanges != []:
            sampleDataset.append(renderParams(lInput,finalBurstRanges,\
                                              finalBurstSurprises,burstIndicator,finalNumBursts,totalTime))
            nameRes.append(totalName[index])
            neuronRes.append(totalNeuron[index])
            stimuliRes.append(totalStimuli[index])

        else:
            #no burst in current list of timestamps
            no_burst_sampleDataset.append(dfRow)
            no_burst_nameRes.append(totalName[index])
            no_burst_neuronRes.append(totalNeuron[index])
            no_burst_stimuliRes.append(totalStimuli[index])

    return sampleDataset,nameRes,neuronRes,stimuliRes,no_burst_sampleDataset,\
        no_burst_nameRes,no_burst_neuronRes,no_burst_stimuliRes




#unsupervised -- save data into csv file
def saveData(Dataset,nameRes,neuronRes,stimuliRes):
    df = pd.DataFrame(Dataset,columns=['burst duration','within-burst spiking freq',\
                                       'within-burst max spiking freq','within-burst number of spikes',\
                                        'inter-burst interval','percentage of burst spikes','burst frequency',\
                                            'mean surprise','max surprise'])
    
    #also include subject and stimuli name
    df['Subject'] = nameRes
    df['Neuron'] = neuronRes
    df['Stimuli'] = stimuliRes
    
    # df.to_csv(savePath, index=False)
    return df


#supervised -- save data into csv file
def supervised_saveData(dataset,nameRes,labelRes,save_path = "labeled_data/nine_burst_parameters.csv"):
    df = pd.DataFrame(dataset,columns=                      ['burst duration','within-burst spiking freq',\
                                                             'within-burst max spiking freq','within-burst number of spikes',\
                                                                'inter-burst interval','percentage of burst spikes','burst frequency',                        'mean surprise','max surprise'])
    
    #also include subject and stimuli name
    df['Subject'] = nameRes
    df['label'] = labelRes
    
    # #save parameters into csv file
    # df.to_csv(save_path, index=False)
    return df




def histoResults(DF,lstColumnNames,plotFunc,plotParams,saveFileName = 'compare_histograms.pdf',\
                 barplot = False,figsize = (100,50),main_title = 'Comparison of histograms of 9 parameters',alpha = 0.3):
    #3x3 subplots
    fig,ax = plt.subplots(3,3,figsize=(15,15))
    # fig = plt.figure(figsize=figsize)
    #sns plot with hue of each of the nine parameters on a row for datapoints in each cluster
    num = len(lstColumnNames)
    for j in range(num):
        #use "y = " for barplot, "x =  " otherwise
        #figure into the jth subplot
        # fig.add_subplot(num,1,j+1)
        
        if barplot:
            #call barplot function, add a subplot to the 3x3 grid
            plotFunc(data=DF,y=lstColumnNames[j],ax=ax[j//3,j%3],**plotParams)

            
            for bar in ax[j//3,j%3].containers[0]:
                bar.set_alpha(alpha)

        else:
            #call plot function, add a subplot to the 3x3 grid
            plotFunc(data=DF,x=lstColumnNames[j],ax=ax[j//3,j%3],**plotParams)


        
            
    #main title
    fig.suptitle(main_title)

    #save pdf
    fig.savefig(saveFileName)
    plt.show()
    return


ts = pd.read_pickle("LNandPN.pkl")


# Choose appropriate p values and parameters for each of the three datasets

def get_label_from_true_json(data,name_base,neuron_base,true_json_data):
    true_label = []

    for i,label in enumerate(data):
        true_label.append(true_json_data[name_base[i]][neuron_base[i]]['true'])


    return true_label

true_json_data = json.load(open('unlabeled_pred.json'))



def lr_objective(trial):
    try:
        #lr parameters
        solver = trial.suggest_categorical('solver',['newton-cg','lbfgs','liblinear','sag','saga'])
        C = trial.suggest_loguniform('C',1e-5,1e2)
        tol = trial.suggest_loguniform('tol',1e-5,1e-1)

        #scaler
        scaler_label = trial.suggest_categorical('scaler',['standard','minmax'])
        if scaler_label == 'standard':
            scaler = preprocessing.StandardScaler()
        else:
            scaler = preprocessing.MinMaxScaler()

        #choice of 9 columns
        prune = trial.suggest_categorical('prune',['yes','no'])
        if prune == 'yes':
            pruned_nine_cols = ['within-burst max spiking freq','within-burst number of spikes',\
                                'percentage of burst spikes','burst frequency','mean surprise','max surprise','label']
        else:
            pruned_nine_cols = ['within-burst number of spikes','burst frequency','mean surprise','max surprise','label']

        #p parameter
        p_labeled = trial.suggest_float('p_labeled',0.1,0.9)
        p_unlabeled = trial.suggest_float('p_unlabeled',0.1,0.9)

        #exploratory data analysis
        print(f"#######Start Burst Detection, trial: {trial.number}#######")
        #ts
        ts_sampleDataset,ts_nameRes,ts_labelRes,ts_no_burst_sampleDataset,ts_no_burst_nameRes,\
            ts_no_burst_labelRes = formulateDataset(ts['timestamps'],ts['mothname'],ts['label'],p = p_labeled)
        ts_df = supervised_saveData(ts_sampleDataset,ts_nameRes,ts_labelRes)
        ts_df_pruned = ts_df[pruned_nine_cols]

        #unlabeled
        unlabeled_sampleDataset,unlabeled_nameRes,unlabeled_neuronRes,unlabeled_stimuliRes,\
            unlabeled_no_burst_sampleDataset,unlabeled_no_burst_nameRes,unlabeled_no_burst_neuronRes,\
                unlabeled_no_burst_stimuliRes = formulateDataset_unlabeled(totalDf,totalName,totalNeuron,totalStimuli,p = p_unlabeled)
        unlabeled_df = saveData(unlabeled_sampleDataset,unlabeled_nameRes,unlabeled_neuronRes,unlabeled_stimuliRes)
        #no 'label' column here, so we add :-1
        unlabeled_df_pruned = unlabeled_df[pruned_nine_cols[:-1]]

        #take every fold as a test set and the rest as training set, setup scaler w.r.t. training set
        #preprocessing with scaler on ts data 
        ts_df_pruned_processed = scaler.fit_transform(ts_df_pruned.iloc[:,:-1])

        print(f"#######Burst Detection Finished, trial: {trial.number}#######")

        print(f"#######Start Logistic Regression, trial: {trial.number}#######")


        lr = LogisticRegression(solver = solver,C = C,tol = tol)

        #fit the model
        #display number of columns of ts_df_pruned_processed
        # print(f"#######Number of columns of ts_df_pruned_processed: {ts_df_pruned_processed.shape[1]}#######")
        lr.fit(ts_df_pruned_processed,ts_df_pruned['label'])

        #preprocessing with scaler on unlabeled data
        unlabeled_df_pruned_processed = scaler.fit_transform(unlabeled_df_pruned)
        #display number of columns of unlabeled_df_pruned_processed
        # print(f"#######Number of columns of unlabeled_df_pruned_processed: {unlabeled_df_pruned_processed.shape[1]}#######")
        print(f"#######Logistic Regression Finished, trial: {trial.number}#######")


        #predict on test set
        print(f"#######Start Prediction on Unlabeled Test Set and Compute Accuracy, trial: {trial.number}#######")
        pred_label = lr.predict(unlabeled_df_pruned_processed)


        true_labels = get_label_from_true_json(pred_label,unlabeled_nameRes,unlabeled_neuronRes,true_json_data)
        true_labels_no_burst = get_label_from_true_json(unlabeled_no_burst_nameRes,\
                                                        unlabeled_no_burst_nameRes,unlabeled_no_burst_neuronRes,true_json_data)
        true_labels_extended = true_labels + true_labels_no_burst
        #replace 0 in pred_label as 'LN' and 1 as 'PN'
        pred_label = list(map(lambda x: 'LN' if x == 0 else 'PN',pred_label))
        #append LNs after predicted labels
        unlabeled_pred_extended = list(pred_label + ['LN']*len(unlabeled_no_burst_nameRes))

        #compute accuracy
        pred_accu = accuracy_score(true_labels_extended,unlabeled_pred_extended)
        print(f"#######Output Accuracy: {pred_accu}, trial: {trial.number}#######")
        return pred_accu

    except:
        return 0



study = optuna.create_study(direction='maximize')
study.optimize(lr_objective,n_trials=300)

print(study.best_params,study.best_value)

