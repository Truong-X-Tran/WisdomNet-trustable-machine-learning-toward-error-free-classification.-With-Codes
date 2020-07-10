'''Train Purify Net on MNIST with a CNN base network
The base CNN network mode is from keras example code.
For example of CNN net, see https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model

batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('MNIST X_train shape:', X_train.shape)
print('MNIST y_train shape:', y_train.shape)

#Create the training set for 2 and 7 digits
#Get indices of digits 2 and 7
digit_indices = [i for i in range(len(y_train)) if y_train[i] == 2 or y_train[i] == 7]

#Vector of 2 and 7 digits
y_train = np.array([y_train[i] for i in digit_indices])

#Label the digit 2 as 0 and the digit 7 as 1
for i in range(len(y_train)):
  if y_train[i] == 2:
    y_train[i] = 0
  elif y_train[i] == 7:
      y_train[i] =1

#Training input for 2 and 7 digits
X_train = np.array([X_train[i] for i in digit_indices])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

print('MNIST (0-1) X_train shape:', X_train.shape)
print('MNIST (0-1) y_train shape:', y_train.shape)

#Create the Testing Set for 2 and 7 digits
digit_indices = [i for i in range(len(y_test)) if y_test[i] == 2 or y_test[i] == 7]
print(len(digit_indices))

y_test = np.array([y_test[i] for i in digit_indices])

#Label the digit 2 as 0 and the digit 7 as 1
for i in range(len(y_test)):
  if y_test[i] == 2:
    y_test[i] = 0
  elif y_test[i] == 7:
      y_test[i] =1

X_test = np.array([X_test[i] for i in digit_indices])

y_test = keras.utils.to_categorical(y_test, num_classes)

print('MNIST (0-1) X_test shape:',X_test.shape)
print('MNIST (0-1) y_test shape:',y_test.shape)

#Create the base CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) # Input layer
model.add(Conv2D(64, (3, 3), activation='relu')) #CNN layer 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # Hidden layer
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) #Output layer

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Train the base CNN network
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0)


#Evaluate the base CNN model
score = model.evaluate(X_train, y_train, verbose=0)
print('CNN Trainning accuracy:', score[1])
score = model.evaluate(X_test, y_test, verbose=0)
print('CNN Testing accuracy:', score[1])

#Code for the Purify Net start from here

#Predict the training data by using the base network
y_train_pred = model.predict(X_train)

#Get vector of predicted label form predicted classes matrix
y_train_pred = np.array([np.argmax(y) for y in y_train_pred]) 

#Detect the missclassified training data
count = 0
y_train_u = []
digit_indices = []
for i in range(len(y_train_pred)):
  if y_train_pred[i] != np.argmax(y_train[i]):
    count = count + 1
    y_train_u.append(y_train_pred[i])
    digit_indices.append(i)
#The new set of all misclassified training items by the base network
y_train_u = np.array(y_train_u)
X_train_u = np.array([X_train[i] for i in digit_indices])

#Asign new label 2 to all undecided training data. The old class label is 0 and 1
for i in range(len(y_train_u)):
  y_train_u[i] = 2

y_train_u = keras.utils.to_categorical(y_train_u, 3)

#Obtain the parameters of the base network output layer
last_layer_old_weights = model.layers.pop().get_weights()

#Create the new weights matrix and biases of the Purify Network output layer

new_neuron_weights = np.zeros(shape=[128,1])
new_neuron_bais = np.zeros(1)

W = np.hstack((last_layer_old_weights[0],new_neuron_weights))
b = np.hstack((last_layer_old_weights[1], new_neuron_bais))

#Checking
model.summary() # Summary of the base model structure

#Pop the last layer out of the base network
model.pop()

#Create new prediction layer 
#and link it to the last hidden layer of the base model
new_layer = Dense(3, activation='softmax', weights = [W,b])(model.layers[6].output)

#Create the Purify Net by replacing the output layer of the base network
#With the new prediction layer

model2 = Model(input = model.input, outputs = new_layer)

#Checking
model2.summary() # Summary of the Purify Net model structure

model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.8),
              metrics=['accuracy'])

#Train the Purify Net on the undecided set of data 
model2.fit(X_train_u, y_train_u,
          batch_size=batch_size,
          epochs=2,
          verbose=1)

#Evaluate the Purify model on the Testing data set 
y_test_pred = model2.predict(X_test)

y_test_pred = np.array([np.argmax(y) for y in y_test_pred]) 

test_c = 0 # Correct count
test_m = 0 # Incorrect count
test_u = 0 # Undecided count
y_test_u = []
digit_indices = []
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
