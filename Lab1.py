#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# In[3]:


random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[108]:


#ALGORITHM = "guesser"
ALGORITHM = "tf_net"
#ALGORITHM = "tf_conv"


# In[109]:


DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"


# In[110]:


if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
#     pass                                 # TODO: Add this case.
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
#     pass                                 # TODO: Add this case.
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
#     pass                                 # TODO: Add this case.


#=========================<Classifier Functions>================================


# In[111]:


def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6):
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    inShape=(IS,)
    model.add(keras.layers.Dense(512, input_shape= inShape, activation=tf.nn.relu))
    model.add(keras.layers.Dense(NUM_CLASSES,activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss=lossType)
    model.fit(x,y,epochs=eps)
    #pass        #TODO: Implement a standard ANN here.
    return model


def buildTFConvNet(x, y, eps = 1, dropout = True, dropRate = 0.2, isDatagen=False):
    # To get best accuracy for this model keep datagen as False
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32,  padding="same", kernel_size = (3,3), activation="relu", input_shape=(IH,IW,IZ)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.1));
    model.add(keras.layers.Conv2D(filters=64, padding="same", kernel_size = (3,3), activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(dropRate));
    
    model.add(keras.layers.Conv2D(filters=128, padding="same", kernel_size = (3,3), activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.3));
    
    model.add(keras.layers.Conv2D(filters=256, padding="same", kernel_size = (3,3), activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.4));
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation="relu"))
    model.add(keras.layers.Dense(NUM_CLASSES,activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    #pass        #TODO: Implement a CNN here. dropout option is required.
    if (isDatagen == False):
        model.fit(x,y,epochs=10);
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range=90, width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
        datagen.fit(x)
        model.fit_generator(datagen.flow(x,y, batch_size=64), epochs=15, steps_per_epoch=x.shape[0] // 64)
    return model
    
    
#     model = keras.Sequential()
#     model.add(keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(IH,IW,IZ)))
#     model.add(keras.layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
#     model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
#     model.add(keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
#     model.add(keras.layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
#     model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#     model.add(keras.layers.Flatten())
    
#     model.add(keras.layers.Dense(128,activation="relu"))
#     model.add(keras.layers.Dense(NUM_CLASSES,activation="softmax"))

#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
#     #pass        #TODO: Implement a CNN here. dropout option is required.
#     model.fit(x,y,epochs=eps,batch_size=16)
#     return model

#=========================<Pipeline Functions>==================================


# In[ ]:


def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
#         pass      # TODO: Add this case.
    elif DATASET == "cifar_100_f":
        cifar100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode="fine")
#         pass      # TODO: Add this case.
    elif DATASET == "cifar_100_c":
        cifar100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode="coarse")
        pass      # TODO: Add this case.
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain/ 255.0
    xTest = xTest / 255.0
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data,save=False,saved=False):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        if (saved):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = keras.models.model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            model = loaded_model
        else:
            print("Building and training TF_NN.")
            model = buildTFNeuralNet(xTrain, yTrain)
        if (save):
            # Saving the model to disk
            model_json = model.to_json()
            with open("model.json","w") as json_file:
                json_file.write(model_json)
            model.save_weights("model.h5")
        return model
    elif ALGORITHM == "tf_conv":
        if (saved):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = keras.models.model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            model = loaded_model
        else:
            print("Building and training TF_CNN.")
            model = buildTFConvNet(xTrain, yTrain)
        if (save):
            # Saving the model to disk
            model_json = model.to_json()
            with open("model.json","w") as json_file:
                json_file.write(model_json)
            model.save_weights("model.h5")
        return model 
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    #put save=True to train and save a model
    #put saved=True to run an already trained and saved model make sure to keep save=False when keeping saved=True
    model = trainModel(data[0],saved=False,save=False)
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_f', 'cifar_100_c']
tnn_acc=[97.77,87.4,46.64,14.52,28.85]
# These accuracies were achieved with epochs: 5, 10, 10, 35, 10 respectively
cnn_acc=[99.44,92.06,77.64,50.59,58.11]
# ax.bar(langs,tnn_acc)
ax.bar(langs,cnn_acc)
plt.show()


# In[ ]:




