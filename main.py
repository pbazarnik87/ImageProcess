import logging
import cv2 as cv
import sys
import basic_operations
import os
import numpy as np
from matplotlib import pyplot as plt


def main():

    print("test")

    #basic_operations.show_image('C:\\Users\\bazar666\Pictures\\image001.png')
    #basic_operations.copy_image('C:\\Users\\bazar666\Pictures\\image001.png','C:\\Users\\bazar666\Pictures\\copy_image001.png')

  #  img1=basic_operations.create_empty_image_BGR(100,100)
  #  basic_operations.save_img_to_file(img1,'blank_image001.png')
  #  img2=basic_operations.create_random_image_BGR(100,100)
  #  basic_operations.save_img_to_file(img2,'random_image001.png')
#
  #  img3=basic_operations.create_empty_image_mono(100,100)
  #  basic_operations.save_img_to_file(img3,'blank_mono_image001.png')
  #  img4=basic_operations.create_random_image_mono(100,100)
  #  basic_operations.save_img_to_file(img4,'random_mono_image001.png')
#
  #  gray = basic_operations.convert_bgr_to_gray('foto.png')

    #basic_operations.circles_detect()

#---------------------------------------------------------------------------------------------------
    #show original
    original = basic_operations.load_image('cells.png')
    cv.imshow("original", original)
    #cv.waitKey(0)

    #color2gray
    gray =  basic_operations.convert_bgr_to_gray('cells.png')
    cv.imshow("gray",gray)
    #cv.waitKey(0)

    #blur
    blur = cv.GaussianBlur(gray,(5,5),0)
    cv.imshow("blur", blur)
    #cv.waitKey(0)
    print(blur)
    print('huj')

    otsu = basic_operations.otsu_tresholding(blur)
    cv.imshow("otsu", otsu)
    #cv.waitKey(0)

    #sobel_edges
    sobel = basic_operations.sobel_edge(otsu)
    cv.imshow("sobel", sobel)
    # edges
    edges = cv.Canny(otsu,180,200)
    cv.imshow("canny", edges)

    cir = basic_operations.circles_detect(original,otsu)
    #cv.imshow("cir", cir)
    #cv.waitKey(0)
    #


if __name__ == '__main__':
    main()