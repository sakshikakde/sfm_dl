## import basic packages
import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

## import sklearn ML helper functions
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

## import tf and keras functions
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if (len(tf.config.experimental.list_physical_devices('GPU')) > 0) :
       print("################################################ RUNNING IN GPU ")
else:
       print("################################################ NO GPU FOUND" )

## import self written functions
from misc.utils import *
from Network.VGG16 import *
from misc.DataLoader import DataBatches, load_all, preprocess

CheckPointPath= "../Checkpoints/"
LogsPath =  '../Logs/'
NumEpochs = 20
BasePath = '../Data/FishDataset/Fish_Dataset/Fish_Dataset/'
ModelPath = CheckPointPath+'supervisedModel_old.h5'
SavePath = '../Logs/Results/'
training = False
    
## if the folders dont exist, create them.
foldercheck(SavePath)
foldercheck(CheckPointPath)
foldercheck(LogsPath)

#############################################################################################
####################################### Data Pipeline #######################################
#############################################################################################
folders = dict()
i = 0
for f in os.listdir(BasePath):
    if f[-2:]!='.m' and f[-4:]!='.txt':
        folders[i] = f        
        i+=1
        
X=[]
Y=[]
count=0
for i in folders.keys():
    filepath = BasePath + folders[i] + '/' + folders[i] +'/*png'
    im_paths = sorted(glob.glob(filepath))    
    labels = [i]*len(im_paths)
    # print(len(labels), len(im_paths))
    Y.append(labels)
    X.append(im_paths)
    
X,Y = np.array(X).ravel(), np.array(Y).ravel()

X_tmp, X_test_path, Y_tmp, Y_test = train_test_split(X,Y,test_size=0.2,random_state=44)
X_train_path, X_val_path, Y_train, Y_val = train_test_split(X_tmp,Y_tmp,test_size=0.1,random_state=44)


Y_train, Y_test, Y_val = to_categorical(Y_train),  to_categorical(Y_test), to_categorical(Y_val)
print(Y_train.shape, Y_test.shape, Y_val.shape)

if Y_train.shape[1] != Y_test.shape[1] != Y_val.shape:
    print('failed categorical conversion')

if training:    
    batch_size = 32
    num_classes = Y_train.shape[1]
    number_of_training_samples = X_train_path.shape[0]
    iterations = number_of_training_samples//batch_size

    trainLoader = DataBatches(X_train_path, Y_train, batch_size, shuffle=True)
    X_val = load_all(X_val_path)
    val_data = (X_val, Y_val)
    input_shape = X_val[0].shape
    
    #############################################################################################
    ####################################### Model Trainer #######################################
    #############################################################################################

    print(" Supervised  Model Trainer using Keras...")

    model = VGG_16(num_classes, input_shape, weights_path=None)
    adam = optimizers.Adam(lr=0.0001)
    print("################################################ Model and loss defined")

    model.compile(loss= 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print("################################################ Compiling Model, optimizer and loss functions")
    print("Printing model summary ..... \n")
    print(model.summary())

    print("################################################ Define paths of files .... ")
    
    ckptPath = CheckPointPath+ "weights-{epoch:02d}-{loss:.2f}.ckpt"
    checkpoint = ModelCheckpoint(ckptPath, monitor='loss', save_weights_only = True, verbose=1,  save_best_only = True, mode = min)

    print("################################################ Define training parameters .... ")
    num_iterations_per_epoch = int(number_of_training_samples / batch_size)
    print("Number of training samples: ", number_of_training_samples)

    X,y = trainLoader[1]
    print("Batch Shape,:  ", X.shape,y.shape )

    print('Begin Training .....')

    history_callback = model.fit_generator(generator = trainLoader, validation_data = val_data, verbose = 1, 
                                           steps_per_epoch = num_iterations_per_epoch,  epochs = NumEpochs, callbacks=[checkpoint])

    #############################################################################################
    #######################################  Logs ###############################################
    #############################################################################################
    
    loss_history = history_callback.history["loss"]
    np.savetxt(LogsPath+ "loss_history.txt", np.array(loss_history), delimiter=",")

    acc_history = history_callback.history["acc"]
    np.savetxt(LogsPath+ "acc_history.txt", np.array(acc_history), delimiter=",")

    val_loss_history = history_callback.history["val_loss"]
    np.savetxt(LogsPath+ "val_loss_history.txt", np.array(val_loss_history), delimiter=",")

    val_acc_history = history_callback.history["val_acc"]
    np.savetxt(LogsPath+ "val_acc_history.txt", np.array(val_acc_history), delimiter=",")
    print("################################################ Done Training, Saving final model")

    model.save(ModelPath)
    print("################################################ Model Saved")

else:
    def Predict(model, X_test_path, Y_test):
        predictions, true = [], []
        for path, y_test in tqdm(zip(X_test_path, Y_test)):
            ## load and preprocess
            x_test = cv2.imread(path)
            x_test = preprocess(x_test).reshape(1,128,128,3)
            ## perform prediction
            y_pred = model.predict(x_test)
            ## store predictions and ground truths in order
            predictions.append(np.argmax(y_pred))
            true.append(np.argmax(y_test))
        return np.array(predictions), np.array(true)
    print('Loading Model.......')
    model = load_model(ModelPath)
    
    print('Perform Predictions......... ')
    predictions, true = Predict(model,X_test_path, Y_test)
    print("Accuracy :", accuracy_score(np.array(predictions), np.array(true)))
    print("Classification Report :", classification_report(np.array(predictions), np.array(true)))
    
###################################################################################################################    
###################################################################################################################
###################################################################################################################