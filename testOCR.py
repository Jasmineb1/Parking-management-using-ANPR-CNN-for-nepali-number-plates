import numpy
import numpy as np
import cv2
import pickle
from keras.models import model_from_json #needed to import model from json
import os

#--------Parameters----------------#
width= 640
height= 480
threshold=0.6
#----------assigning the class names---------#
class_names = {
    0: '0',
    1: '१',
    2: '२',
    3: '३',
    4: '४',
    5: '५',
    6: '६',
    7: '७',
    8: '८',
    9: '९',
    10: 'बा',
    11: 'च',
    12: 'प'

}
cnn_class_names_en = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'ba',
    12: 'pa',
    11: 'cha'
}
#################################################################################

###########################if webcam is used#####################################
#cap=cv2.VideoCapture(1)
#cap.set(3,width)
#cap.set(44,height)

#-----------------------unpickle the obj-------------------
##pickle_in=open("model_trained.p","rb")
##model= pickle.load(pickle_in)


#-------------------------- load json and create model------------------#
json_file = open('model_lenet_10_nd.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load weights into this model
model.load_weights("model_lenet_10_nd.h5")
print("Model has been loaded from disk")

#-------------------for preprocessing---------------------------#
def preProcessing(img):
    img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) ##grayscaling the image
    img= cv2.equalizeHist(img) #equalise the img i.e make the lighting of the image distribute evenly
    img= img/255 #normalise and restrict the value from 0 to 1
    return img

#------------using single image-------------------------------#
# For photo it will be read as numpy array
img_path = 'segtest/3.jpg'
img = cv2.imread(img_path,1)
cv2.imshow("Original Image",img)
# resize the image captured
img = cv2.resize(img,(32,32))
# preprocessing image
img = preProcessing(img)
cv2.imshow("Processed Image",img)
# reshape image before sending it to predictor
img = img.reshape(1,32,32,1)
#--------------predict from model------------------#
#classIndex = int(model.predict_classes(img))#yo garda error aaucha support gardaina
predict_x=model.predict(img)
classes_x=np.argmax(predict_x,axis=1)#error aayera yo gareko esle ni class index nai dincha
predictions = predict_x[0]
max_value = max(predictions)
print(predictions)
print(max_value)
#exit()
if max_value <= 0.05:
     class_id = 999
else:
     class_of_x=np.argmax(predict_x,axis=1)
     class_id = class_of_x[0]
print('Class Id: ',class_id)
print('Class Name: ', class_names[class_id])
print('Class name en:',cnn_class_names_en[class_id])
cv2.waitKey(0)
cv2.destroyAllWindows()