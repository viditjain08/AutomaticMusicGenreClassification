#!/usr/bi, but the fact of the nude therapy was deemed a distinguishing characteristic in itselfn/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:53:53 2018

@author: user
"""
# Importing all the libraries needed
import numpy as np
import pandas as pd
import os
from python_speech_features import mfcc, logfbank, ssc
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import pickle
    

def extract_features():
    
    # Some variable and path initializations
    df = pd.DataFrame(columns=['mfcc_feat','fbank_feat','ssc'],index=range(0,1000))
    y_df = pd.DataFrame(columns=['classification'],index=range(0,1000))
    
    # Getting all the genres
    l = []
    for dirpaths,dirnames,filenames in os.walk(os.getcwd()):
        l.append(dirnames)
    
    song_no = 0
    # Extracting Features of the songs
    for i in l[0]:
        #for genre in genres1
        for x in range(100):
            
            # for song in batch of songs
            
            # Extracting the features
            (rate,sig) = wav.read(i+ "/"+ i + ".000" + "%02d" %x + '.wav')
            mfcc_feat = mfcc(sig,rate,nfft=551)
            fbank_feat = logfbank(sig,rate,nfft=551)
            sig_temp = np.reshape(sig, (sig.shape[0],1))
            ssc_var = ssc(sig_temp,rate,nfft=551)
            
            # Adding features to the pandas dataframe -- all 2985 frames
            df.iloc[song_no][0] = mfcc_feat[0:2985,:]
            df.iloc[song_no][1] = fbank_feat[0:2985,:]
            df.iloc[song_no][2] = ssc_var[0:2985,:]
            y_df.iloc[song_no][0] = i
            
            # Incrementing the song number\n",
            song_no+=1
            
    return df,y_df
      
def shuffling(df,y_df):  
    # Joining the features and classification and Shuffling the songs
    z = df.join(y_df) 
    z_shuffled = shuffle(z)

    return z_shuffled

def dump_shuffled(z_shuffled):
    path = os.getcwd()
    with open( os.path.join(path,'shuffled_features.pickle'),'wb' ) as f:
        pickle.dump(z_shuffled,f)
        
def load_shuffled():
    path = os.getcwd()
    with open( os.path.join(path,'shuffled_features.pickle'),'rb' ) as f:
        z_shuffled = pickle.load(f)
        
    return z_shuffled
    
def expanding_data(z_shuffled):
    # Creating a new pandas dataframe where all features are concatenated  to form ( no_of_songs x (no_of_frames x no_of_features) )  
    # Now each song will be a 2985 rows that have 65 columns in the pandas dataframe
    shuffled_df = pd.DataFrame()
    shuffled_y_df = pd.DataFrame()
    count=0
    for song in range(1000):
        temp = np.concatenate((z_shuffled.iloc[song,0],z_shuffled.iloc[song,1],z_shuffled.iloc[song,2]), axis=1)
        shuffled_df = shuffled_df.append(pd.DataFrame(temp))
        shuffled_y_df = shuffled_y_df.append(pd.DataFrame([z_shuffled.iloc[song,3]]*2985))
        count += 1 
        
    return shuffled_df,shuffled_y_df


def train_SVM(shuffled_df,shuffled_y_df): 
    # Dictionary of classifying values
    d = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
    
    # Storing the SVM Classifiers across the Time Series
    SVM_list = []
    
    predict_matrix = np.zeros( (800,10) )
    predict_test_matrix = np.zeros((200,10))
    frame_number=-1
    x = shuffled_df
    y = shuffled_y_df
    
    while frame_number < 2984:    
        frame_number+=1
        scaler = StandardScaler()
        frame = np.zeros( (1000,65) ) # One song has 65 features for one frame, 1000 songs
        frame_y = np.zeros( (1000,1) ) # Predicting for every song's frame
        
        for song in range(1000):
            frame[song,:] = x.iloc[song*2985 + frame_number,:] # getting the songs
            frame_y[song,0] = d[y.iloc[song*2985 + frame_number,0]] # getting the ground truth
        
        frame_new = scaler.fit_transform(frame) #normalization
    
        # Initializing an SVM (frame wise), fitting and storing it in a list
        clf = SVC(C=100,gamma=0.005)
        clf = clf.fit(frame_new[:800,:], frame_y[:800,:].reshape(800,)) # Fitting for 800 songs
        SVM_list.append(clf)
        
        predicted_train = clf.predict(frame_new[:800,:])
        predicted_test = clf.predict(frame_new[800:,:]) # Prediction for the 200 songs. dimention (200,1)
        
        for i in range(800):
            predict_matrix[i,int(predicted_train[i])] += 1 # updating the prediction matrix for 800 songs
        for j in range(200):
            predict_test_matrix[j,int(predicted_test[j])] +=1 # updating the test matrix for 200 songs
            
    return SVM_list,predict_matrix,predict_test_matrix

def dump_SVM(SVM_list):
    path = os.getcwd()
    with open( os.path.join(path,'SVM_list.pickle'),'wb' ) as f:
        pickle.dump(SVM_list,f)
        
def load_SVM():
    path = os.getcwd()
    with open( os.path.join(path,'SVM_list.pickle'),'rb' ) as f:
        SVM_list = pickle.load(f)
        
    return SVM_list

def split_train_test(shuffled_df,shuffled_y_df):
    # Splitting into train and test 
    data_x_train = shuffled_df.iloc[0:800*2985,:].values
    data_x_test = shuffled_df.iloc[800*2985:,:].values
    data_y_train = shuffled_y_df.iloc[0:800*2985,:].values
    data_y_test = shuffled_y_df.iloc[800*2985:,:].values
    
    return data_x_train,data_x_test,data_y_train,data_y_test

# Random Forest on the Prediction Matrix
def RandomForest(predict_matrix,data_y_train,dictionary):
    
    short_class = np.zeros( shape = (800,1) )
    for k in range(0,800):
        short_class[k,:] = dictionary[ (data_y_train[k*2985,:])[0] ]
    
    # Initializing and training the random forest classifier
    clf = RandomForestClassifier(n_estimators=80)
    clf.fit(predict_matrix, short_class.reshape(800,))
    print(metrics.accuracy_score(clf.predict(predict_matrix),short_class.reshape(800,)))
    
    return clf

# Predicting Accuracy on the Test Set
def CheckAccuracy(predict_test_matrix,clf,data_y_test,dictionary):
    
    short_test = np.zeros( shape = (200,1) )
    for k in range(0,200):
        short_test[k,:] = dictionary[ (data_y_test[(k)*2985,:])[0] ]
        
    # Testing results
    result = metrics.accuracy_score(clf.predict(predict_test_matrix),short_test.reshape(200,))
    print(result)
    

def model(songindex):
    
    # Reverse Dictionary
    drev = { 0:'blues', 1:'classical', 2:'country', 3:'disco', 4:'hiphop', 5:'jazz',6:'metal',7:'pop',8:'reggae',9:'rock'}
    d = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
    
    # Extract features,Shuffle,Dump,Load
    df,y_df = extract_features()
    z_shuffled = shuffling(df,y_df)
    dump_shuffled(z_shuffled)
    #z_shuffled = load_shuffled()
    
    #Expand, train SVM, create prediction  matrix
    shuffled_df,shuffled_y_df = expanding_data(z_shuffled)
    SVM_list,predict_matrix,predict_test_matrix = train_SVM(shuffled_df,shuffled_y_df)
    dump_SVM(SVM_list)
    #SVM_list = load_SVM()
    
    # Splitting  into train and test
    data_x_train,data_x_test,data_y_train,data_y_test = split_train_test(shuffled_df,shuffled_y_df)
    
    # Classifier
    clf = RandomForest(predict_matrix,data_y_train,d)
    CheckAccuracy(predict_test_matrix,clf,data_y_test,d)
    
    print( "Predicted: " + drev[int(clf.predict(predict_test_matrix[songindex,:].reshape(1,-1)))])    
    print( "Actual: " + data_y_test[2985*songindex,0] )


if __name__ == '__main__':
    model(3)








    
    
    
    
    