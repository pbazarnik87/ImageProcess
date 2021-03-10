import cv2 as cv
import basic_operations


def main():
    """
    main
    """
    image = basic_operations.ImageBox('file_path',r'pic\face.png')


    #show original
 #   #original = basic_operations.img(r'pic\face.png')
    image.show_image()
#
 #   # color2gray
  #  image.convert_bgr_to_gray()
    image.show_image()
#
 #   # blur
 #   blur = cv.GaussianBlur(image.img, (5, 5), 0)
 #   image.img = blur
 #   image.show_image()
#
 #   '#otsu'
 #   otsu = image.otsu()
 #   otsu_img = otsu[1]
 #   cv.imshow("otsu", otsu_img)
#
 #   '#edges Canny'
 #   edges = cv.Canny(otsu_img, 0, 200)
 #   cv.imshow("canny", edges)

    '#count circles'
    cir = image.circles_detect()
    cv.imshow('detected circles', cir)
    cv.waitKey(0)

    cam_process = basic_operations.CamBox()

    cam_process.show_image()
    #cap = cv.VideoCapture(0)
    #if not cap.isOpened():
    #    print("Cannot open camera")
    #    exit()
    #while True:
    #    # Capture frame-by-frame
    #    ret, frame = cap.read()
    #    # if frame is read correctly ret is True
    #    if not ret:
    #        print("Can't receive frame (stream end?). Exiting ...")
    #        break
    #    # Our operations on the frame come here
#
    #    cir = frame.circles_detect()
##
    #    cv.imshow('frame', cir)
    #    if cv.waitKey(1) == ord('q'):
    #        break
    ## When everything done, release the capture
    #cap.release()
    #cv.destroyAllWindows()


if __name__ == '__main__':
    main()
