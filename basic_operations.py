import cv2 as cv
import sys
import logging
import os
import numpy as np

log = logging.getLogger("basic_operations_Log")
log.setLevel(logging.INFO)

ch = logging.FileHandler(filename="basic_operations.log")
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch.setFormatter(format)
log.addHandler(ch)

class NoImageError(ValueError):
    pass
class UnableToOpenImageError(ValueError):
    pass
class WrongFilePath(ValueError):
    pass

def show_image(img_src_path):
    '''
    test description

    '''
    img = load_image(img_src_path)

    if img is None:
        log.info("img is empty")
        raise NoImageError("Could not read the image.")
    try:
        cv.imshow("Display window", img)
        k = cv.waitKey(0)

    except cv.Error:
        raise UnableToOpenImageError(cv.Error)

def show_matrix(img):

    cv.imshow("Display window", img)
    k = cv.waitKey(0)


def load_image(img_src_path):

    try:
        img = cv.imread(img_src_path)
    except cv.Error:
        raise UnableToOpenImageError(cv.Error)

    if img is None:
        #sys.exit("Could not read the image.")
        raise NoImageError("Could not read the image.")
    return img

def copy_image(img_src_path, img_dest_path):
    src = os.path.isfile(img_src_path)

    if src == False:
        log.info("Wrong file path " + img_src_path)
        raise WrongFilePath('Wrong source file path')

    if(src==True):
        img = cv.imread(img_src_path)
        if img is None:
            log.info("img is empty")
            raise NoImageError("Could not read the image.")
        res = cv.imwrite(img_dest_path, img)
        return res

def save_img_to_file(img, img_dest_path):
    if img is None:
        log.info("file NOT saved error " + str(NameError))
        return 1
    else:

        try:
            cv.imwrite(img_dest_path, img)
        except NameError:
            log.info("file NOT saved error " + str(NameError))

            return 1

        return 0

def create_empty_image_mono(height,width):
    print(np.zeros((height,width,1), np.uint8))
    return np.zeros((height,width,1), np.uint8)

def create_random_image_mono(height,width):
    print(np.random.randint(0,255,(height,width,1), np.uint8))
    return np.random.randint(0,255,(height,width,1), np.uint8)

def create_empty_image_BGR(height,width):
    print(np.zeros((height,width,3), np.uint8))
    return np.zeros((height,width,3), np.uint8)

def create_random_image_BGR(height,width):
    print(np.random.randint(0,255,(height,width,3), np.uint8))
    return np.random.randint(0,255,(height,width,3), np.uint8)

def convert_bgr_to_gray(img_src_path):
    bgr = load_image(img_src_path)
    gray_img = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)

    return gray_img

def circles_detect(src,img):

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 40,
                              param1=10, param2=13, minRadius=5, maxRadius=100)
    circles = np.uint16(np.around(circles))
    circles_counter=0
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(src, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(src, (i[0], i[1]), 2, (0, 0, 255), 3)
        circles_counter+=1
    print('circles= '+ str(circles_counter))
    cv.imshow('detected circles', src)
    cv.waitKey(0)



def s_channel():
    pass

def sobel_edge(img):
    kernel1 = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], np.float32)  # kernel should be floating point type

    kernel2 = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], np.float32)  # kernel should be floating point type
    dst1 = cv.filter2D(img, -1, kernel1)
    dst2 = cv.filter2D(dst1, -1, kernel2)


    return  dst2

def otsu(img):
    img = np.array(img)
    otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    print(otsu)

    return otsu[1]


def otsu_tresholding(img):

    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    list = np.array(hist)
    hist_max = max(list)
    print(hist_max)
    index = np.where(list == hist_max)[0][0]
    print(index)
    print(img[0][0])



    shape = np.shape(img)
    print(shape)


    for i in range(shape[0]):
        for ii in range(shape[1]):

            if(img[i][ii]<210):
                img[i][ii] = 0
            elif (img[i][ii]>210):
                img[i][ii] = 255
            else:
                img[i][ii] = 210

    print(img)
    #img1 = img.astype("uint8")

    return img







