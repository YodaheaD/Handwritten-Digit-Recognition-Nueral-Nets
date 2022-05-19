import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten 
import matplotlib.pyplot as plt

# Creating our Traing and Testing Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 

# We want our Training and Testing values between 0 to 1 so we scale accoringly
x_train = x_train.astype("float32") / 255 
x_test = x_test.astype("float32") / 255 


# Checking Shapes for all variables, printing the shapes if needed
x_train = np.expand_dims(x_train, -1) 
x_test = np.expand_dims(x_test, -1) 
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
#print('Shape of x_train is {}'.format(x_train.shape))
#print('Shape of x_test is {}'.format(x_test.shape))
#print('Shape of y_train is {}'.format(y_train.shape))
#print('Shape of y_test is {}'.format(y_test.shape))
y_train = keras.utils.to_categorical(y_train, 10) 
y_test = keras.utils.to_categorical(y_test, 10) 
#print('Shape of Y_train is {}'.format(y_train.shape))
#print('Shape of Y_test is {}'.format(y_test.shape))



# The Construction of our model with 3 Convolutional layers, 2 Pooling layers,
# a flatten layer,and two dense layers, the last being our output layer
# The last layer has 10 units because we want 10 different classes  through 9

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

# Gathering Model metrics
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 
print(model.summary())
history = model.fit(x_train,y_train,epochs=5,batch_size=200,validation_split=.2,verbose=1)
score= model.evaluate(x_test, y_test, verbose=0)

#Print the results of our model testing and saving the model for use
print("Test loss:", score[0])
print("Test accuracy:", score[1]) 
model.save('/home/pi/Proj/keras_convnet_adam') 