""" kNN digit classifier, with no pre processing.

    # Requires
        - python (python2 tested)
        - opencv (3.0.0 tested)
        - numpy

    A disadvantage of this method is that the entire training data
    set is required to be in memory for classification.

    Best accuracy: 96.88% (3.12% error rate), using all training
    and test data. 10s to train, 30ms / digit classification rate.

    # Some results

    Training images: 5000,  test images: 10000
    Accuracy: 95.95%
    Training time: 1.038233s
    Total classification time: 15.916733s (0.001592s / digit)

    Training images: 10000,  test images: 10000
    Accuracy: 96.74%
    Training time: 1.965730s
    Total classification time: 29.044025s (0.002904s / digit)

    Training images: 20000,  test images: 10000
    Accuracy: 97.41%
    Training time: 3.825669s
    Total classification time: 54.745351s (0.005475s / digit)

    Training images: 40000,  test images: 10000
    Accuracy: 97.95%
    Training time: 7.645315s
    Total classification time: 106.770302s (0.010677s / digit)
"""

from __future__ import print_function

import itertools
import time

import cv2
import numpy as np

from utils import knn
from utils import mnist_reader as mnist


# Number of digits to train with (max 60000)
TRAINING_SIZE = 60000

# Number of digits from the test set to classify (max 10000)
TEST_SIZE = 10000

# Size that digit images are normalised to
NORMALISED_SIZE = (20, 20)


def main():
    print("training knn classifier...")
    start = time.clock()
    classifier = knn.KnnDigitClassifier(preprocess, TRAINING_SIZE)
    end = time.clock()
    training_time = end - start
    print("...done""")

    print("classifying test set...")
    num_predictions = 0
    num_correct = 0
    start = time.clock()
    # TODO: I think you can do predictall() or similar, may be faster
    for digit_img, label in itertools.izip(
            mnist.test_images(TEST_SIZE),
            mnist.test_labels(TEST_SIZE)):
        predicted_digit = classifier.predict(digit_img)
        num_predictions += 1
        if predicted_digit == label:
            num_correct += 1
    end = time.clock()
    classification_time = end - start
    print("...done")
    print("Training images: %d,  test images: %d" % (TRAINING_SIZE, TEST_SIZE))
    print("Accuracy: %.2f%%" % (100.0 * num_correct / num_predictions))
    print("Training time: %fs" % training_time)
    print("Total classification time: %fs (%fs / digit)" % (classification_time,
            classification_time / TEST_SIZE))


def preprocess(img):
    desl = deslant(img)
    return normalise_size(desl)


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
    main()
