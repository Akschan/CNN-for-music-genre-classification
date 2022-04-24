################ Creating model fucntions ####################

import json
import os
import librosa
import math
import numpy as np
PATH = "data.json" # path to data
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from keras.regularizers import l2
import matplotlib.pyplot as plt

# load json func
def load_data(path):
    with open(path,"r") as fp:
        data = json.load(fp)
        
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    
    return inputs, targets
  
  #prepare the train validation and test sets + fix input shape it must be (130,13,1)
  def Make_datasets(test_size, validation_size):
    x, y = load_data(PATH)
    
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = test_size)
    xtrain, xvalidation, ytrain, yvalidation = train_test_split(xtrain, ytrain, test_size = validation_size)
    
    xtrain = xtrain[...,np.newaxis]
    xvalidation = xvalidation[...,np.newaxis]
    xtest = xtest[...,np.newaxis]
    
    input_shape = (xtrain.shape[1],xtrain.shape[2],xtrain.shape[3])
    
    return xtrain, xvalidation, xtest, ytrain, yvalidation, ytest, input_shape
  
  
  """ implementing the model with 3 convolutional layers followed by a 1 dense layer and another dense layer 
      with softmax activation so we can take the result with the biggest probalitiy """
  
  def Model(shape):
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(32, (3,3), activation = "relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), input_shape = shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32, (3,3), activation = "relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), input_shape = shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv2D(32, (2,2), activation = "relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), input_shape = shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model
  
  
  # function to see the accuracy and the error of the model
  def plot_history(history):
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"],label='train accuracy')
    axs[0].plot(history.history["val_accuracy"],label='test accuracy')
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    axs[1].plot(history.history["loss"],label='train error')
    axs[1].plot(history.history["val_loss"],label='test error')
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    plt.savefig('CNN.png')
    plt.show()
    
    
    ################ Prediction using model fucntions ####################
    
    def Audio_to_predict(audio_path):
        file_path=r"C:\Users\Ju_Eun\Testing_Model\CNN\rock1.wav"
        SAMPLE_RATE = 22050
        hop_length = 512
        num_segments = 10
        n_fft = 2048
        n_mfcc = 13
        data_test = {"mfcc":[]}
        SAMPLES_IN_TRACK = SAMPLE_RATE * 30
        num_sample_per_seg = int(SAMPLES_IN_TRACK / num_segments)
        expected_num_mfcc_vectors_per_seg = math.ceil(num_sample_per_seg / hop_length)
        signal,sr = librosa.load(file_path,sr=SAMPLE_RATE,offset=80.0, duration=30.0)
        
        for s in range(num_segments):
            start_sample = num_sample_per_seg * s
            finish_sample = start_sample + num_sample_per_seg


            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                        sr=sr,
                                        n_fft=n_fft,
                                        n_mfcc=n_mfcc,
                                        hop_length=hop_length)
            mfcc = mfcc.T
            if len(mfcc) == expected_num_mfcc_vectors_per_seg:
                data_test["mfcc"].append(mfcc.tolist())
                arr = np.asarray(data_test['mfcc'])
        return arr
    
    def predict(model,x):
        x = x[...,np.newaxis]
        prediction = model.predict(x)
        p_index = np.argmax(prediction, axis=1)
        genre = np.array(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"])
        print("Predicted genre is {}".format(*genre[p_index]))
