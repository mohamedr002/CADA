# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:18:00 2019

@author: MOHAMEDR002
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
#=========================setting up for data processing===================
scaler = preprocessing.MinMaxScaler()
#======================end of setting up===================================

#=================================START OF DATA PROCESSING ============================================

def process_data(data_dir, data_identifier,window_size):
    RUL_01 = np.loadtxt(data_dir + 'RUL_' + data_identifier +'.txt')
    # read training data - It is the aircraft engine run-to-failure data.
    train_01_raw = pd.read_csv(data_dir + '/train_'+data_identifier+'.txt', sep=" ", header=None)
    train_01_raw.drop(train_01_raw.columns[[26, 27]], axis=1, inplace=True)
    train_01_raw.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_01_raw = train_01_raw.sort_values(['id','cycle'])

    # read test data - It is the aircraft engine operating data without failure events recorded.
    test_01_raw = pd.read_csv(data_dir + '/test_'+data_identifier+'.txt', sep=" ", header=None)
    test_01_raw.drop(test_01_raw.columns[[26, 27]], axis=1, inplace=True)
    test_01_raw.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                         's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                         's15', 's16', 's17', 's18', 's19', 's20', 's21']
    test_01_raw = test_01_raw.sort_values(['id','cycle'])
    
    if data_identifier == 'FD002' or data_identifier == 'FD004':
        max_RUL = 130
        print("--multi operating conditions--")

        #standard the setting1 values 
        train_01_raw.loc[train_01_raw['setting1'].between(0.00000e+00, 3.00000e-03), 'setting1'] = 0.0
        train_01_raw.loc[train_01_raw['setting1'].between(9.99800e+00, 1.00080e+01), 'setting1'] = 10.0
        train_01_raw.loc[train_01_raw['setting1'].between(1.99980e+01, 2.00080e+01), 'setting1'] = 20.0
        train_01_raw.loc[train_01_raw['setting1'].between(2.49980e+01, 2.50080e+01), 'setting1'] = 25.0
        train_01_raw.loc[train_01_raw['setting1'].between(3.49980e+01, 3.50080e+01), 'setting1'] = 35.0
        train_01_raw.loc[train_01_raw['setting1'].between(4.19980e+01, 4.20080e+01), 'setting1'] = 42.0

        test_01_raw.loc[test_01_raw['setting1'].between(0.00000e+00, 3.00000e-03), 'setting1'] = 0.0
        test_01_raw.loc[test_01_raw['setting1'].between(9.99800e+00, 1.00080e+01), 'setting1'] = 10.0
        test_01_raw.loc[test_01_raw['setting1'].between(1.99980e+01, 2.00080e+01), 'setting1'] = 20.0
        test_01_raw.loc[test_01_raw['setting1'].between(2.49980e+01, 2.50080e+01), 'setting1'] = 25.0
        test_01_raw.loc[test_01_raw['setting1'].between(3.49980e+01, 3.50080e+01), 'setting1'] = 35.0
        test_01_raw.loc[test_01_raw['setting1'].between(4.19980e+01, 4.20080e+01), 'setting1'] = 42.0

        ######
        # Normalising sensor and settings data
        ######

        # skip the first 2 columns, id and cycle
        train_sensor = train_01_raw.iloc[:, 2:]
        test_sensor = test_01_raw.iloc[:, 2:]

        # Obtain the 21 column names from 's1' to 's21'
        Train_Norm = pd.DataFrame(columns = train_sensor.columns[3:])
        Test_Norm = pd.DataFrame(columns = test_sensor.columns[3:])


        grouped_train = train_sensor.groupby('setting1')
        grouped_test = test_sensor.groupby('setting1')

        for train_idx, train in grouped_train:
            scaled_train = scaler.fit_transform(train.iloc[:, 3:])
            scaled_train_combine = pd.DataFrame(
                    data=scaled_train,
                    index=  train.index,  
                  columns=train_sensor.columns[3:]) 
            Train_Norm = pd.concat([Train_Norm, scaled_train_combine])

            for test_idx, test in grouped_test:
                if train_idx == test_idx:
                    scaled_test = scaler.transform(test.iloc[:, 3:])
                    scaled_test_combine = pd.DataFrame(
                            data=scaled_test,    
                            index=  test.index,  
                          columns=test_sensor.columns[3:]) 
                    Test_Norm = pd.concat([Test_Norm, scaled_test_combine])
        Train_Norm = Train_Norm.sort_index()
        Test_Norm = Test_Norm.sort_index()
        train_01_raw.iloc[:, 2:5] = scaler.fit_transform(train_01_raw.iloc[:, 2:5])
        test_01_raw.iloc[:, 2:5] = scaler.transform(test_01_raw.iloc[:, 2:5])

        Train_Settings = pd.DataFrame(
                            data=train_01_raw.iloc[:, :5],    
                            index=  train_01_raw.index,  
                          columns=train_01_raw.columns[:5])
        Test_Settings = pd.DataFrame(
                            data=test_01_raw.iloc[:, :5],    
                            index=  test_01_raw.index,  
                          columns=test_01_raw.columns[:5])


        #adding the column of 'time'  
        train_01_nor = pd.concat([Train_Settings, Train_Norm], axis = 1)
        test_01_nor = pd.concat([Test_Settings, Test_Norm], axis = 1)
        train_01_nor = train_01_nor.values
        test_01_nor = test_01_nor.values
        train_01_nor = np.delete(train_01_nor, [5,6,9,10,11,12,14,16,17,20,22,23], axis=1) # sensor 1 for index 5   2,3,4,
        test_01_nor = np.delete(test_01_nor, [5,6,9,10,11,12,14,16,17,20,22,23], axis=1)

    else:
        print("--single operating conditions--")
        max_RUL = 130 #125.0
        with np.nditer(train_01_raw['setting1'], op_flags=['readwrite']) as it:
            for x in it:
                x[...] = 0.0
        
        #skip the first 2 columns, id and cycle
        train_01_raw.iloc[:, 2:] = scaler.fit_transform(train_01_raw.iloc[:, 2:])
        test_01_raw.iloc[:, 2:] = scaler.transform(test_01_raw.iloc[:, 2:])
        train_01_nor = train_01_raw
        test_01_nor = test_01_raw
        train_01_nor = train_01_nor.values
        test_01_nor = test_01_nor.values
        train_01_nor = np.delete(train_01_nor, [5,9,10,14,20,22,23], axis=1) # sensor 1 for index 5 2,3,4
        test_01_nor = np.delete(test_01_nor, [5,9,10,14,20,22,23], axis=1) 
   
 # sliding window and split to 5 subsets  
    winSize = window_size;trainX = [];trainY = []     
    testX = [];testY = [];testLen = []
    num_engines=int(np.max(train_01_nor[:,0]))
    fold_length=int(num_engines/5)
    # Fold_1--->  [----/-]
    #training data 
    data_subsets={'data':[],'labels':[]}
    num_subsets=5
    start=1;end=fold_length
    for k in range(num_subsets):
        trainX = [];trainY = []
        for i in range(start,end+1):
            #the data of the the i_th engine
            ind =np.where(train_01_nor[:,0]==i)
            #the id of the ith engine 
            ind = ind[0]
            data_temp = train_01_nor[ind,:]

            for j in range(len(data_temp)-winSize+1):
                trainX.append(data_temp[j:j+winSize,1:].tolist())
                train_RUL = len(data_temp)-winSize-j
                if train_RUL > max_RUL:
                    train_RUL = max_RUL
                trainY.append(train_RUL)
        trainX = np.array(trainX)
        trainY = np.array(trainY)/max_RUL
        trainX = trainX[:, :, 4:]
    
        data_subsets['data'].append(trainX)
        data_subsets['labels'].append(trainY)
        start=end+1
        end=end+fold_length
    # Construct the five folds of training data 
    num_folds=5
    folds={'train':[],
           'validation':[]}
    for n in range(num_folds):
        temp_data,temp_labels = data_subsets['data'][:], data_subsets['labels'][:]
        #getting validation data and labels together
        val_data= temp_data.pop(n),temp_labels.pop(n)
        #getting trainign data
        train_data= np.vstack(temp_data),np.concatenate(temp_labels)
        folds['train'].append(train_data)
        folds['validation'].append(val_data)

    
# sliding window for test data to segment it. 
    for i in range(1,int(np.max(test_01_nor[:,0]))+1):     
        ind =np.where(test_01_nor[:,0]==i)
        ind = ind[0]
        testLen.append(len(ind))
        data_temp = test_01_nor[ind,:]
        
        if len(data_temp)<winSize:
            data_temp_a = []
            for myi in range(data_temp.shape[1]):
                x1 = np.linspace(0, winSize-1, len(data_temp) )
                x_new = np.linspace(0, winSize-1, winSize)
                tck = interpolate.splrep(x1, data_temp[:,myi])
                a = interpolate.splev(x_new, tck)
                data_temp_a.append(a.tolist())
            data_temp_a = np.array(data_temp_a)
            data_temp = data_temp_a.T
            data_temp = data_temp[:,1:]
        else:
            data_temp = data_temp[-winSize:,1:]
            # print('np.shape(data_temp)', np.shape(data_temp))
            # print('data_temp[:,1]', data_temp[:,0])
            # break 
        data_temp = np.reshape(data_temp,(1,data_temp.shape[0],data_temp.shape[1]))
        if i == 1:
            testX = data_temp
        else:
            testX = np.concatenate((testX,data_temp),axis = 0)
        if RUL_01[i-1] > max_RUL:
            testY.append(max_RUL)
        else:
            testY.append(RUL_01[i-1])
            
    #remove the working condition parameters
    testX = testX[:, :, 4:]  
    # trainAux =  trainX[:, :, 0:4]
    # testAux = testX[:, :, 0:4]
    
    return folds, testX, testY#, trainAux, testAux
#=================================END OF DATA PROCESSING ==============================================
