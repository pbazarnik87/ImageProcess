import cv2 as cv
import ImgErrors
import numpy as np
import logging
import os

log = logging.getLogger("basic_operations_Log")
log.setLevel(logging.INFO)

ch = logging.FileHandler(filename="basic_operations.log")
format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch.setFormatter(format)
log.addHandler(ch)


class RandomImg(object):
    """
    class for creating basic random images
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def create_empty_image_mono(self):
        """
        :return:mono image array filled with zeros (black)
        """
        print(np.zeros((self.height, self.width, 1), np.uint8))
        return np.zeros((self.height, self.width, 1), np.uint8)

    def create_random_image_mono(self):
        """
        :return: mono image array filled random values (0-255)
        """
        print(np.random.randint(0, 255, (self.height, self.width, 1), np.uint8))
        return np.random.randint(0, 255, (self.height, self.width, 1), np.uint8)

    def create_empty_image_BGR(self):
        """
        :return: BGR black image array
        """
        print(np.zeros((self.height, self.width, 3), np.uint8))
        return np.zeros((self.height, self.width, 3), np.uint8)

    def create_random_image_BGR(self):
        """
        :return: BGR random values image array
        """
        print(np.random.randint(0, 255, (self.height, self.width, 3), np.uint8))
        return np.random.randint(0, 255, (self.height, self.width, 3), np.uint8)


def copy_image(img_src_path, img_dest_path):
    """
    copy image img matrix
    :param img_src_path:
    :param img_dest_path:
    :return: cv.imwrite()
    """
    src = os.path.isfile(img_src_path)

    if src is False:
        log.info("Wrong file path " + img_src_path)
        raise ImgErrors.WrongFilePath('Wrong source file path')

    if src is True:
        img = cv.imread(img_src_path)
        if img is None:
            log.info("img is empty")
            raise ImgErrors.NoImageError("Could not read the image.")
        res = cv.imwrite(img_dest_path, img)
        return res


def save_img_to_file(img, img_dest_path):
    """
    saves img matrix as file
    :param img_dest_path: destination path
    :param img: image matrix
    :return:
    """
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


class ImageBox(object):
    """
    class ImageBox contains original image array and modified array
    methods for detecting
    """

    def __init__(self, option, img):
        """
        :param img: image data
        :param option: 'cam'|'file_path'
        """
        self.option = option
        self.img = img
        self._img_original = self.img
        super().__init__()

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img):

        __option_define = {'cam', 'file_path'}
        if self.option == 'cam':

            self.__img = img

        elif self.option == 'file_path':
            try:
                img_data = cv.imread(img)
            except cv.Error:
                raise ImgErrors.UnableToOpenImageError(cv.Error)

            if img_data is None:
                # sys.exit("Could not read the image.")
                raise ImgErrors.NoImageError("Could not read the image.")
            self.__img = img_data
        elif self.option not in __option_define:
            raise ImgErrors.WrongOption(r"Wrong option!. Available options are: 'cam'|'file_path'")

    def show_image(self):
        """
        show the image in popup window(openCV)
        :return:
        """

        if self.img is None:
            log.info("img is empty")
            raise ImgErrors.NoImageError("Could not read the image.")
        try:
            cv.imshow("Display window", self.img)
            k = cv.waitKey(0)

        except cv.Error:
            raise ImgErrors.UnableToOpenImageError(cv.Error)

    def convert_bgr_to_gray(self):
        """
        convert image from BRG to GRAY
        """
        self.__img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def circles_detect(self):
        """
        HOUGH circle detection
        :return: original image array with drawn green circles and red center point
        """
        # color2gray
        self.convert_bgr_to_gray()

        # blur
        self.blur()

        # otsu
        self.otsu()

        # edges Canny
        self.edges_canny()

        circles = cv.HoughCircles(self.img, cv.HOUGH_GRADIENT, 1, 15,
                                  param1=50, param2=10, minRadius=12, maxRadius=18)
        print(circles)
        circles_counter = 0
        tmp = self._img_original
        if np.all(circles) is not None:
            circles = np.uint16(np.around(circles))


            for i in circles[0, :]:
                # draw the outer circle
                cv.circle(tmp, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv.circle(tmp, (i[0], i[1]), 2, (0, 0, 255), 3)
                circles_counter += 1
        print('circles= ' + str(circles_counter))
        return tmp

    def sobel_edge(self):
        kernel1 = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], np.float32)  # kernel should be floating point type

        kernel2 = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], np.float32)  # kernel should be floating point type
        dst1 = cv.filter2D(self.img, -1, kernel1)
        dst2 = cv.filter2D(self.img, -1, kernel2)
        dst3 = np.add(dst1, dst2)

        return dst3

    def otsu(self):
        """
        otsu threshold
        """
        otsu = cv.threshold(np.array(self.img), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        self.__img = otsu[1]

    def blur(self):
        """
        blurring the image with Gaussian blur mask
        :return: blurred original image array
        """
        self.__img = cv.GaussianBlur(self.img, (5, 5), 0)

    def edges_canny(self):
        """
        Canny detecting edges
        :return: black image array with white edge lines
        """
        self.__img = cv.Canny(self.img, 0, 200)


class CamBox:

    def show_image(self):
        """
        display popup with cam
        """
        if not self.cam.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            #    Capture frame-by-frame
            ret, frame = self.cam.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            frame_process = ImageBox('cam', frame)
            frame_process.circles_detect()

            cv.imshow('frame', frame_process._img_original)
            if cv.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        self.cam.release()
        cv.destroyAllWindows()

    def __init__(self):
        self.cam = cv.VideoCapture(0)
