'''
#Train a simple MLP on the breast cancer data set.
Apply WisdomNet net
0% error after 6 epochs with learning rate of 0.5.
'''
import numpy as np 
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import Model

data = pd.read_csv("bank_note_data.csv")

X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Scaling the feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Convert the class label vectors to binary class matrix
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

#Create the base MLP model
model = Sequential()
model.add(Dense(4,activation = 'relu', input_dim=4)) # input layer
model.add(Dense(4, activation = 'relu'))   # hidden layer
model.add(Dense(2, activation = 'sigmoid')) # output layer

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=100, epochs=28)

#Evaluate the base model
score = model.evaluate(X_train, y_train, verbose=0)
print('NLP Net Train accuracy:', score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print('MLP Net Test accuracy:', score[1])

#Code for the WisdomNet Net start from here

#Predict the training data by using the base network
y_train_pred = model.predict(X_train)

#Get vector of predicted label form predicted classes matrix
y_train_pred = np.array([np.argmax(y) for y in y_train_pred]) 

# Detect the missclassified training data
count = 0
y_train_u = []
y_train_act = []
misclassified_item_indices = []
for i in range(len(y_train_pred)):
  if y_train_pred[i] != np.argmax(y_train[i]):
    count = count + 1
    y_train_act.append(np.argmax(y_train[i]))
    y_train_u.append(y_train_pred[i])
    misclassified_item_indices.append(i)

#The new set of all misclassified training items by the base network
y_train_u = np.array(y_train_u)
X_train_u = np.array([X_train[i] for i in misclassified_item_indices])

#--Check point of the data set if needed 
#print(count)
#print('u_acture',y_train_act)
#print('u_predicted',y_train_u)
#print(len(y_train_u))
#--

#Asign new label 2 to all undecided training data. The old class label is 0 and 1
for i in range(len(y_train_u)):
  y_train_u[i] = 2

y_train_u = keras.utils.to_categorical(y_train_u, 3)

#Obtain the parameters of the base network output layer
last_layer_old_weights = model.layers.pop().get_weights()

#Create the new weight matrix of the WisdomNet Network output layer
new_layer_weights_1 = last_layer_old_weights[0]
new_layer_weights_0 = np.zeros(shape=[4,1])

W = np.hstack((new_layer_weights_1,new_layer_weights_0))

#Create the new biases of the the WisdomNet Network output layer
new_layer_bais_1 = last_layer_old_weights[1]
new_layer_bais_0 = np.zeros(1)

b = np.hstack((new_layer_bais_1, new_layer_bais_0))

#Checking
model.summary() # Summary of the base model structure

#Pop the last layer out of the base network
model.pop()

#Create new prediction layer
new_layer = Dense(3, activation = 'sigmoid', weights = [W,b])(model.layers[1].output)

#Create the WisdomNet Net by replacing the output layer of the base network
#With the new prediction layer
model2 = Model(inputs = model.input, outputs = new_layer)

#Checking
model2.summary() # Summary of the WisdomNet Net model structure

model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=5),
              metrics=['accuracy'])

#Train the WisdomNet Net on the undecided set of data 
model2.fit(X_train_u, y_train_u,
          batch_size=count,
          epochs=4,
          verbose=1)

#Evaluate the WisdomNet model on the Testing data set 
y_test_pred = model2.predict(X_test)

y_test_pred = np.array([np.argmax(y) for y in y_test_pred]) 

test_c = 0 # Correct count
test_m = 0 # Incorrect count
test_u = 0 # Undecided count
y_test_u = []
misclassified_item_indices = []
for i in range(len(y_test_pred)):
  if y_test_pred[i] == 2:
    test_u = test_u + 1
  elif y_test_pred[i] == np.argmax(y_test[i]):
    test_c = test_c + 1
  elif y_test_pred[i] != np.argmax(y_test[i]):
    test_m = test_m + 1
print('correct', test_c)
print('missed', test_m)
print('undecided', test_u)


print('missed rate', test_m/len(y_test))
print('undecided rate', test_u/len(y_test))
print('Number of testing sample ',len(y_test))
