""" kNN digit classifier, preprocessing images before training &
    classification. Preprocessing:
    - de-slant image based on image moments
    - normalise digit size by finding each digit's bounding box,
      cropping image to just the digit, then resizing to a pre-
      determined size.
"""

from __future__ import print_function

import cv2
import numpy as np

from utils import classifier as cs
from utils import knn
from utils import mnist

# Size that digit images are normalised to
NORMALISED_SIZE = (20, 20)


class KnnNormSizeDeslant(knn.KnnDigitClassifier):

    def train(self, images, labels):
        super(KnnNormSizeDeslant, self).train(self.preprocess_all(images), labels)

    def classify(self, image):
        return super(KnnNormSizeDeslant, self).classify(self.preprocess(image))

    def preprocess(self, image):
        desl = deslant(image)
        return normalise_size(desl)

    def preprocess_all(self, images):
        for image in images:
            yield self.preprocess(image)


def normalise_size(img):
    rects = [cv2.boundingRect(c) for c in find_contours(img)]
    # sort by area decreasing
    rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = rects[0]
    cropped_img = img[y:y+h, x:x+w]
    return cv2.resize(cropped_img, NORMALISED_SIZE, interpolation=cv2.INTER_AREA)


def find_contours(img):
    """ Return contours found by cv2.findContours, without affecting the
        input image.
    """
    img_copy, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
    return contours


def deslant(img):
    """ Vertically straighten an image based on image moments.

        Taken from
        https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html#svm-opencv
    """
    h, w = img.shape[:2]
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*h*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(h, h),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img


if __name__ == '__main__':
    NUM_TRAINS = 100
    NUM_TESTS = 100
    runner = cs.ClassifierRunner(KnnNormSizeDeslant())
    runner.train(mnist.training_images(NUM_TRAINS), mnist.training_labels(NUM_TRAINS))
    runner.run(mnist.test_images(NUM_TESTS), mnist.test_labels(NUM_TESTS))
    print(runner.get_report_str())
