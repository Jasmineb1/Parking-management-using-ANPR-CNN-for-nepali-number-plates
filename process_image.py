import cv2
import numpy as np
import os
import cv2
import matplotlib as plt

def processImage(img):
    kernel = np.ones((3,3),np.uint8)
    # grayscale conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)  # imgf contains Binary image
    #plt.imshow(img)
    #denoising
    #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

    #eroison
    # img = cv2.erode(img,kernel,iterations=1)
    #license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

    #_, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
    # applying Otsu thresholding
    # as an extra flag in binary
    # thresholding
    # params: source, threshold value, max value, thresholding technique
    ret, thresh1 = cv2.threshold(img, 150, 400, cv2.THRESH_BINARY_INV)
    # median blur
    #img = cv2.medianBlur(thresh1, 3)
    #plt.imshow(img)
    # resizing
    # for lenet (32,32)
    # for alexnet (128,128)
    img = cv2.resize(thresh1, (32,32))
    #plt.imshow(img)
    return img


if __name__ == '__main__':
    for class_val in range(12):
        ip_dir_path = 'E:/OCRtraining/trialimages/'
        op_dir_path = 'E:/OCRtraining/trialpreprocess/'
        # create directories in path if not exists
        if not os.path.isdir(op_dir_path):
            os.mkdir(op_dir_path)
        images = os.listdir(ip_dir_path)
        for img_path in images:
            img = cv2.imread(ip_dir_path+img_path)
            img = processImage(img)
            cv2.imwrite(op_dir_path+img_path, img)

    # ip_dir_path = 'test3/0/'
    # op_dir_path = 'test4/0_2/'
    # images = os.listdir(ip_dir_path)
    # for img_path in images:
    #     img = cv2.imread(ip_dir_path+img_path)
    #     img = processImage(img)
    #     cv2.imwrite(op_dir_path+img_path, img)
    print('Preprocessing done')
