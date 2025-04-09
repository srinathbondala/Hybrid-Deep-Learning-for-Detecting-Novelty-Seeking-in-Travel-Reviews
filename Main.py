from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from string import punctuation
import numpy as np
import pandas as pd
import pickle
#importing NLP packages for text review processing
from nltk.corpus import stopwords

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import os

from sentence_transformers import SentenceTransformer #loading bert sentence model

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from keras.layers import Bidirectional, GRU, LSTM, Conv1D, MaxPooling1D, RepeatVector#loading GRU, bidriectional, lstm and CNN


from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Hybrid Deep Learning for Detecting Novelty Seeking in Travel Reviews: Integrating BERT, CNN, and BiGRU for Improved Accuracy") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca,bert,index
global accuracy, precision, recall, fscore, values,cnn_model,extension_model,predict,values
precision = []
recall = []
fscore = []
accuracy = []

bert = SentenceTransformer('nli-distilroberta-base-v2')
print("Bert model initialized")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset, labels, values
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    labels, count = np.unique(dataset['label'], return_counts = True)
    labels = ['Novelty Seeking', 'Non-Novelty Seeking']
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.show()

def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, pca, scaler
    text.delete('1.0', END)
    if os.path.exists("model/bert.npy"):
        X = np.load("model/bert.npy")
        Y = np.load("model/label.npy")
    else:
        textdata = []
        labels = []
        for i in range(len(dataset)):#loop all reviews from dataset
            msg = dataset.get_value(i, 'content')#read review content
            label = dataset.get_value(i, 'label')#read label
            msg = msg.strip().lower()      #convert text to lower case  
            msg = cleanPost(msg)#clean the review message
            textdata.append(msg)#add message to textdata array    
            labels.append(label)#adding label to array
        embeddings = bert.encode(textdata, convert_to_tensor=True)#convert all text data into BERT vector
        X = embeddings.numpy()#convert bert vector into numpy for training
        np.save("model/bert", X)#save bert data and labels to model folder
        Y = np.asarray(labels)
        np.save("model/label", Y)
    text.insert(END,"Bert Converted Embedding vector from dataset reviews"+"\n")
    text.insert(END,X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 32, 24))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    print()
    text.insert(END,"\n\nDataset train & test split as 80% dataset for training and 20% for testing"+"\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0])+"\n")
    print()

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    print()
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def trainLSTM():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore,predict,index,values
    text.delete('1.0', END)
    
    lstm = Sequential() #defining deep learning sequential object
    #adding GRU layer with 32 filters to filter given input X train data to select relevant features
    lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    #adding dropout layer to remove irrelevant features
    lstm.add(Dropout(0.3))
    #adding another layer
    lstm.add(LSTM(32))
    lstm.add(Dropout(0.3))
    #defining output layer for prediction
    lstm.add(Dense(y_train.shape[1], activation='softmax'))
    #compile GRU model
    lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm.fit(X_train, y_train, batch_size = 16, epochs = 35, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close() 
    else:
        lstm = load_model("model/lstm_weights.hdf5")
    predict = lstm.predict(X_test)#perfrom prediction on test data
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    #calculate accuracy and other metrics
    calculateMetrics("BERT-LSTM Model", predict, target)

def trainBILSTM():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore,predict,values
    text.delete('1.0', END)
    
    gru_bilstm = Sequential() #defining deep learning sequential object
    #adding GRU layer with 32 filters to filter given input X train data to select relevant features
    gru_bilstm.add(Bidirectional(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    gru_bilstm.add(Dropout(0.3))
    #adding another layer
    gru_bilstm.add(Bidirectional(GRU(32)))#adding bidirectional-GRU layer and peform training on X_train Bert data
    gru_bilstm.add(Dropout(0.3))
    #defining output layer for prediction
    gru_bilstm.add(Dense(y_train.shape[1], activation='softmax'))
    #compile GRU model
    gru_bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #start training model on train data and perform validation on test data
    if os.path.exists("model/bigru_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/bigru_weights.hdf5', verbose = 1, save_best_only = True)
        hist = gru_bilstm.fit(X_train, y_train, batch_size = 16, epochs = 35, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/bigru_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close() 
    else:
        gru_bilstm = load_model("model/bigru_weights.hdf5")
    predict = gru_bilstm.predict(X_test)#perfrom prediction on test data
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    #calculate accuracy and other metrics
    calculateMetrics("BERT-Bi-GRU Model", predict, target)

def runCNN():
    global X_train, y_train, X_test, y_test,extension_model
    global accuracy, precision, recall, fscore,predict,index,values
    text.delete('1.0', END)
    
    extension_model = Sequential()
    #defining CNN layer
    extension_model.add(Conv1D(filters=32, kernel_size = 15, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
    extension_model.add(Conv1D(filters=16, kernel_size = 12, activation = 'relu'))
    #adding maxpool layer
    extension_model.add(MaxPooling1D(pool_size = 2))
    extension_model.add(Dropout(0.3))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(2))
    #adding bidirectional + GRU to CNN layer
    extension_model.add(Bidirectional(GRU(24, activation = 'relu')))
    extension_model.add(Dropout(0.3))
    #defining output layer
    extension_model.add(Dense(units = 33, activation = 'softmax'))
    extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, batch_size = 16, epochs = 35, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model = load_model("model/extension_weights.hdf5")
    #perform prediction on test data using bidirectional LSTM on test data   
    predict = extension_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    target = np.argmax(y_test, axis=1)
    #calculate accuracy and other metrics
    calculateMetrics("Propose BERT-CNN-Bi-GRU Model", predict, target)

def graph():
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    df = pd.DataFrame([['BERT-LSTM','Precision',precision[0]],['BERT-LSTM','Recall',recall[0]],['BERT-LSTM','F1 Score',fscore[0]],['BERT-LSTM','Accuracy',accuracy[0]],
                       ['BERT-Bi-GRU','Precision',precision[1]],['BERT-Bi-GRU','Recall',recall[1]],['BERT-Bi-GRU','F1 Score',fscore[1]],['BERT-Bi-GRU','Accuracy',accuracy[1]],
                       ['Propose BERT-CNN-Bi-GRU','Precision',precision[2]],['Propose BERT-CNN-Bi-GRU','Recall',recall[2]],['Propose BERT-CNN-Bi-GRU','F1 Score',fscore[2]],['Propose BERT-CNN-Bi-GRU','Accuracy',accuracy[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()
    
    
def predict():
    global X_train, y_train, X_test, y_test,predict,extension_model
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    testData = pd.read_csv("Dataset/testData.csv")#reading test data
    for i in range(len(testData)):
        msg = dataset.get_value(i, 'content')#read test review as message
        msgs = msg.strip().lower()        #convert to lower case
        msgs = cleanText(msgs)#clean messages
        data = []
        data.append(msgs)#add message to array
        embeddings = bert.encode(data, convert_to_tensor=True)#convert message review to bert vector
        X = embeddings.numpy()#convert vector to numpy
        X = np.reshape(X, (X.shape[0], 32, 24))#reshape vector
        predict = extension_model.predict(X)#using extension model predict weather test message is Novelty or not
        predict = np.argmax(predict)
        text.insert(END,"Text Review : "+msg+" ===> Predicted As "+labels[predict]+"\n")




font = ('times', 16, 'bold')
title = Label(main, text='Hybrid Deep Learning for Detecting Novelty Seeking in Travel Reviews: Integrating BERT, CNN, and BiGRU for Improved Accuracy')
title.config(bg='gray24', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Attack Database", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=220,y=100)
processButton.config(font=font1)

svmButton = Button(main, text="Run BERT-LSTM", command=trainLSTM)
svmButton.place(x=450,y=100)
svmButton.config(font=font1)

knnButton = Button(main, text="Run BERT-Bi-GRU", command=trainBILSTM)
knnButton.place(x=600,y=100)
knnButton.config(font=font1)

dtButton = Button(main, text="Run Propose BERT-CNN-Bi-GRU", command=runCNN)
dtButton.place(x=770,y=100)
dtButton.config(font=font1)

xgButton = Button(main, text="Comparision Graph", command=graph)
xgButton.place(x=10,y=150)
xgButton.config(font=font1)

dnnButton = Button(main, text="Predict", command=predict)
dnnButton.place(x=200,y=150)
dnnButton.config(font=font1)
main.config(bg='gold')
main.mainloop()


