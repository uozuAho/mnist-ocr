import cv2
import subprocess
import os

from utils import classifier as cs


class TesseractNoTrainingSlow(cs.GenericClassifier):
    """ This is WAY too slow to be useful - approx 10 digits/sec.
        Just here as an example.
    """
    def train(self, images, labels):
        pass

    def classify(self, image):
        return classify_digit(image)


def classify_digit(img):
    """ Classify the digit in an image containing a single digit.
        Image requirements (I think):
            - binary, white on black
            - upright, no skew etc
        Returns:
            1-9 (int), or None
    """
    img_path = 'tesseract_digit_img.tmp.jpg'
    txt_path = 'tesseract_classification.tmp.txt'
    cv2.imwrite(img_path, img)
    # tesseract options:
    #   -psm 10:    image contains a single character
    #   digits1-9:  'whitelist' config file that I created under
    #               /usr/share/tesseract-ocr/tessdata/configs
    #               Contents of file:
    #               tessedit_char_whitelist 123456789
    subprocess.check_output(
        "tesseract "+img_path+" "+txt_path[:-4]+" -psm 10 digits1-9", shell=True)
    digit_str = open(txt_path, "r").read()
    digit_int = None
    try:
        digit_int = int(digit_str)
    except:
        pass
    os.remove(img_path)
    os.remove(txt_path)
    return digit_int
