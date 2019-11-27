import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, BatchNormalization, MaxPooling2D, LSTM, GRU,TimeDistributed
from tensorflow.keras import layers
import time
import itertools
import os
#%%
# open dataset
Folder = 'DATASET_FINAL'
path = os.getcwd()+'\\'+Folder
file = 'crusher_.npz'
dataset = np.load(path+'\\'+file)

x_train = np.reshape(np.asarray(dataset['x_train']),[-1,np.asarray(dataset['x_train']).shape[1],np.asarray(dataset['x_train']).shape[2],1])
y_train = np.asarray(dataset['y_train'])

x_test = np.reshape(np.asarray(dataset['x_test']),[-1,np.asarray(dataset['x_test']).shape[1],np.asarray(dataset['x_test']).shape[2],1])
y_test = np.asarray(dataset['y_test'])
#%%
class_weight = {0: 1.,
                1: 300.}

#%% LSTM
model = Sequential()
model.add(Conv2D(filters = 256, kernel_size = (7, 2), activation = 'tanh', input_shape = (x_train.shape[1],x_train.shape[2],1), padding = 'valid'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 256, kernel_size = (7, 1), activation = 'tanh', padding = 'same'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Flatten()))
model.add(GRU(units = 256, return_sequences=True))
model.add(GRU(units = 256, return_sequences=True))
model.add(Flatten())
model.add(Dense(units = 512, activation = 'tanh'))
model.add(Dropout(0.4))
model.add(Dense(units = 128, activation = 'tanh'))

model.add(Dense(units = 1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_history = model.fit(x_train,y_train, batch_size=1024, epochs=50, validation_split = 0.1,  class_weight=class_weight)




#%% CNN
#model = Sequential()
#model.add(Conv2D(filters = 64, kernel_size = (7, 2), activation = 'tanh', input_shape = (x_train.shape[1],x_train.shape[2],1), padding = 'valid'))
#model.add(Conv2D(filters = 32, kernel_size = (3, 1), activation = 'tanh', padding = 'same'))
#model.add(Conv2D(filters = 16, kernel_size = (3, 1), activation = 'tanh', padding = 'valid'))
#
#model.add(Flatten())
#model.add(Dense(units = 128, activation = 'tanh'))
#model.add(Dropout(0.2))
#model.add(Dense(units = 128, activation = 'tanh'))
#model.add(Dropout(0.2))
#
#model.add(Dense(units = 1, activation = 'sigmoid'))
#model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
#model_history = model.fit(x_train,y_train, batch_size=512, epochs=100, validation_split = 0.1,  class_weight=class_weight)

#%%
test_loss,test_accuracy = model.evaluate(x_test, y_test, batch_size=512)
y_pred = model.predict(x_test)
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#%%
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('CNN model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
