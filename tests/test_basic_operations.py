import unittest
import numpy as np
import cv2 as cv
import os
import basic_operations

class BasicOperationsTestCase(unittest.TestCase):


#load_image
    def test_check_if_load_image_is_loaded(self):
        res = basic_operations.load_image(r'tests\foto.png')
        img = cv.imread(r'tests\foto.png')
        self.assertIsNone(np.testing.assert_array_equal(res,img))

    def test_check_if_load_image_is_empty(self):
        self.assertRaises(basic_operations.NoImageError, basic_operations.load_image,'noImage')

    def test_image_file_extension_is_wrong_load_image(self):
        self.assertRaises(basic_operations.NoImageError,basic_operations.load_image,'image.noimage')


#show_image
    def test_check_if_opened_show_image(self):
        res = basic_operations.show_image(r'tests\foto.png')

        self.assertIsNone(res)

    def test_fail_to_open_empty_im_show_image(self):

        self.assertRaises(basic_operations.NoImageError,basic_operations.show_image,'noImage')
        cv.destroyAllWindows()

    def test_image_file_extension_is_wrong_show_image(self):

        self.assertRaises(basic_operations.NoImageError,basic_operations.show_image,'image.noimage')
        cv.destroyAllWindows()


#copy_image
    def test_copy_image_wrong_src_path(self):
        src = 'wrongfilepath'
        dest = 'foto.png'
        self.assertRaises(basic_operations.WrongFilePath,basic_operations.copy_image,src,dest)


    def test_copy_image_wrong_dest_path(self):
        src = r'tests\foto.png'
        dest = 'wrongfilepath'
        self.assertRaises(cv.error, basic_operations.copy_image, src, dest)


    def test_copy_image_correct_deest_src_path(self):
        src = r'tests\foto.png'
        dest = r'D:\PycharmProjects\ImageProcess\tests\foto_copy.png'

        self.assertEquals(basic_operations.copy_image(src, dest),True)