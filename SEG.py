# Import the necessary packages
import argparse
import cv2
import numpy as np
from keras.models import model_from_json #to import model from json
from keras.models import load_model
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
import functools
import matplotlib.pyplot as plt
from YOLO_LICENSEPLATE_DETECTION import plate_detection

def segmentation():
    # Read the image and convert to grayscale
    image= plate_detection()
    #image = cv2.imread(img\8.jpg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()
    # Apply Gaussian blurring and thresholding
    # to reveal the characters on the license plate

    # below 2 lines from testfromvid.py
    # gray= cv2.equalizeHist(gray) #equalise the img i.e make the lighting of the image distribute evenly
    # gray= gray/255 #normalise and restrict the value from 0 to 1
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    plt.imshow(thresh, cmap='gray')

    # Perform connected components analysis on the thresholded image and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 60  # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 15  # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)
        #yo maile comment gareko plt.imshow(mask, cmap='gray')
        #yo maile comment gareko plt.show()

        # Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        # Sort the bounding boxes from left to right, top to bottom
        # sort by Y first, and then sort by X if Ys are similar
        def compare(rect1, rect2):
            if abs(rect1[1] - rect2[1]) > 10:
                return rect1[1] - rect2[1]
            else:
                return rect1[0] - rect2[0]

        boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
        # Define constants
        # TARGET_WIDTH = 128
        # TARGET_HEIGHT = 128

        TARGET_WIDTH = 32
        TARGET_HEIGHT = 32

        # draw contours
        result = mask.copy()
        for rect in boundingBoxes:
            # Get the coordinates from the bounding box
            x, y, w, h = rect
            # Crop the character from the mask
            # and apply bitwise_not because in our training data for pre-trained model
            # the characters are black on a white background
            crop = mask[y:y + h, x:x + w]
            crop = cv2.bitwise_not(crop)
            # Get the number of rows and columns for each cropped image
            # and calculate the padding to match the image input of pre-trained model
            rows = crop.shape[0]
            columns = crop.shape[1]
            paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
            paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

            # Apply padding to make the image fit for neural network model
            crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)
            # Convert and resize image
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

            # Prepare data for prediction
            crop = crop.astype("float") / 255.0
            crop = img_to_array(crop)
            crop = np.expand_dims(crop, axis=0)

            # Show bounding box and prediction on image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.putText(image, chars[idx], (x,y+15), 0, 0.8, (0, 0, 255), 2)

        #maile gareko comment plt.imshow(image)
        #maile gareko comment plt.show()

        # --------Parameters----------------#
        # width= 640
        # height= 480
        # threshold=0.6
        # ----------assigning the class names---------#
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
        # Load the pre-trained CNN model
        model_json = open('model_lenet_10_nd.json', 'r')
        loaded_model_json = model_json.read()
        model_json.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('model_lenet_10_nd.h5')

        # model.save('model_lenet_10_nd.hdf5')
        # model = load_model('model_lenet_10_nd.hdf5')
        vehicle_plate = ""
        # Loop over the bounding boxes
        for rect in boundingBoxes:
            # Get the coordinates from the bounding box
            x, y, w, h = rect
            # Crop the character from the mask
            # and apply bitwise_not because in our training data for pre-trained model
            # the characters are black on a white background
            crop = mask[y:y + h, x:x + w]
            crop = cv2.bitwise_not(
                crop)  # inverts color... but if not done then chahine gari color format nahuna sakcha resize garesi
            # Get the number of rows and columns for each cropped image
            # and calculate the padding to match the image input of pre-trained model
            rows = crop.shape[0]
            columns = crop.shape[1]
            paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
            paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

            # Apply padding to make the image fit for neural network model
            crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)
            #     plt.imshow(crop)
            #     plt.show()
            #     crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

            crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
            crop = cv2.bitwise_not(crop)  # resize garesi color inversion to get the required format

            #yo maile comment gareko plt.imshow(crop, cmap='gray')
            #yo maile comment gareko plt.show()

            # Prepare data for prediction
            crop = crop.astype("float") / 255.0
            crop = img_to_array(crop)
            crop = np.expand_dims(crop, axis=0)
            crop = crop.reshape(1, 32, 32, 1)  # kina error aayeko check garnu parcha chalena bhane
            # yesma reshape bhitra ko sabai lai multiply garda total 3072 hunu parne cha as 1*32*32*3

            #     predict_x = model.predict(crop)
            #     classes_x=np.argmax(predict_x,axis=1)
            #     predictions = predict_x[0]
            #     max_value = max(predictions)

            #     print(predictions)
            #     print(max_value)

            #     if max_value <= 0.05:
            #         class_id = 999
            #     else:
            #         class_of_x=np.argmax(predict_x,axis=1)
            #         class_id = class_of_x[0]
            # #     print('Class Id: ',class_id)
            #     print('Class Name: ', class_names[class_id])
            #     print('Class name en:',cnn_class_names_en[class_id])

            #      predict_x=model.predict(crop)

            #     # Convert and resize image
            #     crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            #     crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
            #     # Prepare data for prediction
            #     crop = crop.astype("float") / 255.0
            #     crop = img_to_array(crop)
            #     crop = np.expand_dims(crop, axis=0)
            #     # Make prediction
            prob = model.predict(crop)[0]
            idx = np.argsort(prob)[-1]
            vehicle_plate += class_names[idx]
        #     # Show bounding box and prediction on image
        #     cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 2)
        #     cv2.putText(image, class_names[idx], (x,y+15), 0, 0.8, (0, 0, 255), 2)
        # # Show final image
        # cv2.imshow('Final', image)
        print("Vehicle plate: " + vehicle_plate)
        # cv2.waitKey(0)
    print("final vehicle plate number:" +vehicle_plate)
    return(vehicle_plate)


#running
segmentation()
