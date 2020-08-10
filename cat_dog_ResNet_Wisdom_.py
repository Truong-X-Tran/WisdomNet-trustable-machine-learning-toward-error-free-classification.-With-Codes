#You may refer to the sample keras code to classify cat and dog
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#----------------------------
#This program is to build a WisdomNet on top of ResNet50 network
#----------------------------
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model

from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(1)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'input/train'
validation_data_dir = 'input/validation_400_next'
nb_train_samples = 4000
nb_validation_samples = 400
epochs = 16
batch_size = 8


def save_bottlebeck_features():
    #datagen = ImageDataGenerator(rescale=1. / 255)
    datagen = ImageDataGenerator(rescale=1., featurewise_center=True) #(rescale=1./255)
    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

    # build the ResNet50 network
    model = applications.ResNet50(include_top=False, pooling = 'avg', weights='imagenet')
    #model = applications.(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    #model.summary()
    plot_model(model, show_shapes = True,to_file='Model_ResNet_notop.png')


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    print(train_data.shape)
    dummy = train_data.shape[1:]
    print(dummy)
    train_labels = keras.utils.to_categorical(train_labels, 2)
    validation_labels = keras.utils.to_categorical(validation_labels, 2) 


    print(len(validation_data))

    model = Sequential()

    #model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu', input_shape=train_data.shape[1:]))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

    scoreT = model.evaluate(train_data,train_labels,verbose=0)
    print('Train Test loss:', scoreT[0])
    print('Train Test accuracy:', scoreT[1])

    scoreV = model.evaluate(validation_data,validation_labels,verbose=0)
    print('Val Test loss:', scoreV[0])
    print('Val Test accuracy:', scoreV[1])
    #plot_model(model, show_shapes = True, to_file='Model_dogcat1.png')


    # Add code for N2 net
    y_train_pred = model.predict(train_data)
    print(len(y_train_pred))
    #print(y_train_pred)
    #print(train_labels)
    #print(validation_labels)

    y_train_pred = np.array([np.argmax(y) for y in y_train_pred]) 
    #print(y_train_pred)
    count = 0
    y_train_u = []
    cat_dog_indices = []
    for i in range(len(y_train_pred)):
      if y_train_pred[i] != np.argmax(train_labels[i]):
        count = count + 1
        y_train_u.append(y_train_pred[i])
        cat_dog_indices.append(i)
    y_train_u = np.array(y_train_u)

    print(count)
    #print(y_train_u)
    print(len(y_train_u))
    #print(cat_dog_indices)

    x_train_u = np.array([train_data[i] for i in cat_dog_indices])
    print(train_data.shape)
    print(x_train_u.shape)

    for i in range(len(y_train_u)):
      y_train_u[i] = 2
    #print(y_train_u)

    y_train_u = keras.utils.to_categorical(y_train_u, 3)

  #  print(y_train_u)

    #model.summary()

    last_layer_old_weights = model.layers.pop().get_weights()
 #   print(last_layer_old_weights[1])
    new_neuron_weights = np.zeros(shape=[128,1])
#    print(new_neuron_weights)

    W = np.hstack((last_layer_old_weights[0],new_neuron_weights))

    new_neuron_bais = np.zeros(1)

    b = np.hstack((last_layer_old_weights[1], new_neuron_bais))

#    print(W)
#   print(b)

    model.pop()
    #model.summary()
    new_layer = Dense(3, activation='sigmoid', weights = [W,b])(model.layers[3].output)
    # model.add(new_layer)
    model2 = Model(input = model.input, outputs = new_layer)
    #model2.summary()

    model2.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=1.0),
                  metrics=['accuracy'])

    model2.fit(x_train_u, y_train_u,
              batch_size=count,
              epochs=16,
              verbose=1)
    plot_model(model2, show_shapes = True,to_file='Model_doccat2.png')
    y_train_pred2 = model2.predict(x_train_u)

    #print(y_train_pred2)

    y_test_pred = model2.predict(validation_data)

    y_test_pred = np.array([np.argmax(y) for y in y_test_pred]) 



    test_c = 0
    test_m = 0
    test_u = 0
    u_cat_dog_indices = []
    missed_cat_dog_indices = []
    for i in range(len(y_test_pred)):
      if y_test_pred[i] == 2:
        test_u = test_u + 1
        u_cat_dog_indices.append(i)
      elif y_test_pred[i] == np.argmax(validation_labels[i]):
        test_c = test_c + 1
      elif y_test_pred[i] != np.argmax(validation_labels[i]):
        test_m = test_m + 1
        missed_cat_dog_indices.append(i)
    print('correct',test_c)
    print('missed',test_m)
    print('undecided',test_u)
    print('total',len(validation_labels))
    print('missed index', missed_cat_dog_indices)
    print('undecided index', u_cat_dog_indices)





save_bottlebeck_features()
train_top_model()