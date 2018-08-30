import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('dataset/pima-indians-diabetes.csv')
X = dataset.iloc[:,:8].values
print(X)
Y = dataset.iloc[:,8].values
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(output_dim = 16, init = 'uniform', activation = 'sigmoid', input_dim = 8))
model.add(Dense(output_dim = 8, init = 'uniform', activation = 'sigmoid'))
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print((cm[0][0]+cm[1][1])/cm.sum())

ann_viz(model, view=True, filename="network.gv", title="DiaNN)