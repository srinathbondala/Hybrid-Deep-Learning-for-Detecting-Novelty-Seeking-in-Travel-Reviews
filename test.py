from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
from nltk.stem import PorterStemmer
from keras.utils.np_utils import to_categorical
import os
from sklearn.metrics import accuracy_score
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint 
from keras.layers import Bidirectional, GRU, LSTM, Conv1D, MaxPooling1D, RepeatVector
from sentence_transformers import SentenceTransformer #loading bert sentence model
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
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
'''
dataset = pd.read_csv("Dataset/processed.csv")
for i in range(len(dataset)):
    msg = dataset.get_value(i, 'content')
    label = dataset.get_value(i, 'label')
    msg = msg.strip().lower()        
    msg = cleanPost(msg)
    textdata.append(msg)    
    labels.append(label)
    print(label)

print(textdata)
bert = SentenceTransformer('nli-distilroberta-base-v2')
embeddings = bert.encode(textdata, convert_to_tensor=True)
X = embeddings.numpy()
np.save("model/bert", X)
Y = np.asarray(labels)
print(X)
print(Y)
np.save("model/label", Y)
'''

X = np.load("model/bert.npy")
Y = np.load("model/label.npy")
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
print(X)
print(Y)
print(X.shape)
print(Y.shape)

X = np.reshape(X, (X.shape[0], 32, 24))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

gru_bilstm = Sequential() #defining deep learning sequential object
#adding GRU layer with 32 filters to filter given input X train data to select relevant features
gru_bilstm.add(Bidirectional(GRU(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
#adding dropout layer to remove irrelevant features
gru_bilstm.add(Dropout(0.3))
#adding another layer
gru_bilstm.add(Bidirectional(GRU(32)))
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
predict = gru_bilstm.predict(X_test)
predict = np.argmax(predict, axis=1)
target = np.argmax(y_test, axis=1)
acc = accuracy_score(target, predict)
print(acc)


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
predict = lstm.predict(X_test)
predict = np.argmax(predict, axis=1)
target = np.argmax(y_test, axis=1)
acc = accuracy_score(target, predict)
print(acc)

#now define extension model by combining CNN + LSTM + Bidirectional LSTM as this bi-lstm will optimized features from 
#both forward and backward direction so it will have more optimzied features and accuracy will be better
extension_model = Sequential()
#defining CNN layer
extension_model.add(Conv1D(filters=32, kernel_size = 15, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
extension_model.add(Conv1D(filters=16, kernel_size = 12, activation = 'relu'))
#adding maxpool layer
extension_model.add(MaxPooling1D(pool_size = 2))
extension_model.add(Dropout(0.3))
extension_model.add(Flatten())
extension_model.add(RepeatVector(2))
#adding bidirectional + LSTM to CNN layer
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
acc = accuracy_score(target, predict)
print(acc)
