#This program task is to call upon the Camera module and take a picture of the Handwritten digit
#The image is saved into a file that is then converted into readable data for the Convolutional Nueral Network model
#The model is previoulsy trained and saved in another file and will be loaded into this program at the start
#Once the model is done, It will produce an array of values that are probabilites of what the model thinks the digit is.
#Therefore, we read the highest proability and as that is essentially the Model's "Guess"

#Importing Libraries needed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sense_hat import SenseHat
import time
import picamera
import cv2
sense=SenseHat()

#Parameters for resizing image
row=28
colomn=28

#define read_image to read file
def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (row, colomn),interpolation=cv2.INTER_CUBIC)
#return cv2.resize(img, (ROWS, COLOMNS),interpolation=cv2.INTER_CUBIC)

#define invert to invert image color
def invert(file_path):
    return cv2.bitwise_not(file_path)



#Import the previously saved Convulutional Nueral Network model named "keras_convnet_adam"
model = keras.models.load_model("/home/pi/Proj/keras_convnet_adam")



#Capture an image with the Pi camera and name the file "imageTest1". 
#You can find the Image in the Proj directory to see that number is fully inframe

while(1): 
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.capture('imageTest1.jpg')
    file = "imageTest1.jpg"
    test_image = read_image(file)
    
    #Our model is trained with number that use a Black background and white numbers
    #Therefore we must minipulate the image taken to match the format
    #First Convert image to greyscale then invert it, use .cvtColor and apply it to the image
    #Then use the previously define function invert to invert the greyscale image
    
    gray= cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    invert_img= invert(gray)
    
    #Print Image
    #cv2.imshow('invert', invert_img)
    #cv2.imshow('invert',test_image)
    #cv2.waitKey()
    
    #Reshaping is important part of setting up Nueral Networks correctly. 
    #Here we are reshaping the image then feeding it into the model
    
    X_img = invert_img.reshape(1,28,28,1)/255
    Yhat = model.predict(X_img)
    print('Yhat:',Yhat)
    
    #Now we need to find the highest accuracy in the Yhat array 
    #The digit with the highest accuracy is therefore the model's prediction
    max = 0
    for i in range (10):
        if(Yhat[0,i]> max):
            max = Yhat[0,i]
            maxPos= i
    #Display the digit to the Sensehat    
    digit= str(maxPos)
    sense.show_letter(digit)
    
    #Print the Yhat array aswell at the Digit prediction in the terminal
    print('Accuracy:{}'.format(Yhat[0,maxPos]*100))
    print('Digit:{}'.format(digit))
    
    #Wait for 5 milliseconds
    cv2.waitKey(5)
    
    