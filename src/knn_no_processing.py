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

    Training images: 5000,  test images: 1000
    Accuracy: 91.00%
    Training time: 0.887249s
    Total classification time: 2.746573s (0.002747s / digit)

    Training images: 10000,  test images: 1000
    Accuracy: 91.60%
    Training time: 1.651124s
    Total classification time: 5.274939s (0.005275s / digit)

    Training images: 20000,  test images: 1000
    Accuracy: 93.70%
    Training time: 3.244392s
    Total classification time: 10.453117s (0.010453s / digit)

    Training images: 40000,  test images: 1000
    Accuracy: 96.00%
    Training time: 6.432339s
    Total classification time: 20.490807s (0.020491s / digit)

    Training images: 60000,  test images: 10000
    Accuracy: 96.88%
    Training time: 9.480559s
    Total classification time: 303.666520s (0.030367s / digit)
"""

from __future__ import print_function

import itertools
import time

import cv2
import numpy as np

from utils import knn
from utils import mnist_reader as mnist


# Number of digits to train with (max 60000)
TRAINING_SIZE = 1000

# Number of digits from the test set to classify (max 10000)
TEST_SIZE = 1000


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
    return img


if __name__ == '__main__':
    main()
