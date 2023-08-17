import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout,Flatten
import pickle
import math
#from tensorflow.keras.losses import categorical_crossentropy

#from tensorflow import keras
#from tensorflow.keras import layers

#from tensorflow.python.keras import optimizers as opt





#################
path= 'Character_sets'
testRatio=0.2
valRatio=0.2
imageDimensions=(32,32,3)

batchsizeVal=50
epochsVal=10

#stepsPerEpochVal=2000
###################
images= [] ##creating a list for images
classNo=[] ##creating a list for storing each classID
myList =os.listdir(path)
print("Total number of classes detected:",len(myList))
noOfClasses = len(myList)
print("Importing classes.......")
####import all the imgs and put them in a list####

for x in range(0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        currImg= cv2.imread(path+"/"+str(x)+"/"+y)
        currImg= cv2.resize(currImg,(imageDimensions[0],imageDimensions[1])) ##images are 32*32
        images.append(currImg)
        classNo.append(x)
    print(x,end=" ")
print(" ")

images= np.array(images)
classNo = np.array(classNo)
print("Total images in images list=",len(images))
print("Total IDs in classNo list=",len(classNo))

print(images.shape) #we have 32*32 sized images and three channels i.e a coloured image adn of this size we have 19848 images
#print(classNo.shape) #this matrix has all the class numbers which are only values classNo size==image size

####Splitting the data into train test and validation
X_train,X_test,y_train,y_test=train_test_split(images,classNo,test_size= testRatio) #training will be 0.8% and test will be 0.2 i.e 80%training and 20%testing
X_train,X_Validation,y_train,y_validation= train_test_split(X_train,y_train,test_size=valRatio)
print(X_train.shape) #training set
print(X_test.shape) #testing set
print(X_Validation.shape) #validation set

# for checking how many images we have in each class X_train contains images and y_train has IDs of each images
#np.where(y_train==0) #gives the index of where 0 is present
numOfSamples=[]
for x in range(0,noOfClasses):
    #print(len(np.where(y_train==x)[0]))
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)

#####bar  graph to see the distribution of images
plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()
#####################################ends here

print(X_train[50].shape)
##################preprocess the images################
def preProcessing(img):
    img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ##grayscaling the image
    img= cv2.equalizeHist(img) #equalise the img i.e make the lighting of the image distribute evenly
    img= img/255 #normalise and restrict the value from 0 to 1
    return img

#img=preProcessing(X_train[50])
#img = cv2.resize(img,(300,300)) #resizing the image as 32*32 is way too small to see
#cv2.imshow("Preprocessed",img)
#cv2.waitKey(0)

##to preprocess all the images we have
X_train= np.array(list(map(preProcessing,X_train))) #takes each element from training set and send it through preprocessing func and converts it into a numpy array and stores it back to X_train

#print(X_train[50].shape) #has only one channel after preprocessing
X_test= np.array(list(map(preProcessing,X_test)))
X_Validation= np.array(list(map(preProcessing,X_Validation)))

#add depth of 1 to the images which is required for CNN to run properly
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_Validation = X_Validation.reshape(X_Validation.shape[0],X_Validation.shape[1],X_Validation.shape[2],1)

#augmentation of images i.e add zoom, rotations which will make dataset more generic
#keras is used

dataGen = ImageDataGenerator(width_shift_range= 0.1,
                             height_shift_range=0.1,
                             zoom_range= 0.2,
                             shear_range= 0.1,
                             rotation_range= 10) #in degrees
dataGen.fit(X_train) #

#for the network
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

#for linet model
def myModel():
    noOfFilters= 60
    sizeOfFilter1= (5,5)
    sizeOfFilter2= (3,3)
    sizeOfPool= (2,2)
    noOfNode= 500
#addding convolutional layers
    model= Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                                                             imageDimensions[1],
                                                             1),activation='relu'
                                                                )))

    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool)) #adding a pooling layer
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5)) #dropout layer

    model.add(Flatten()) #flattening layer
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model= myModel()
print(model.summary())
##trying to solve error
stepsPerEpochVal = math.floor(len(X_train)/batchsizeVal)

####run the training

history= model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batchsizeVal),
                                    steps_per_epoch=stepsPerEpochVal,
                                    epochs=epochsVal,
                                    validation_data=(X_Validation,y_validation),
                                    shuffle=1)

########to plot how the training was done (variation in the accuracy)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score=model.evaluate(X_test,y_test,verbose=0)
print('Test score=',score[0])
print('Test Accuracy=',score[1])

#saving the file as pickle object
#pickle_out= open("model_trained_10.p","wb")
#pickle.dump(model,pickle_out)
#pickle_out.close()

#as json file
model_json = model.to_json()
with open("model_lenet_10_nd.json","w") as json_file:
    json_file.write(model_json)
#to save
model.save_weights("model_lenet_10_nd.h5")
print("Model saved")