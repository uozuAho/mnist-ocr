""" Read and display a few training and test images.
    Requires opencv & numpy.
    Tested with python2, opencv3.0.0
"""


from __future__ import print_function

import cv2
import numpy as np
import itertools


# constants

DATA_DIR = '../data'
TRAINING_IMAGES_PATH = DATA_DIR + '/train-images-idx3-ubyte'
TRAINING_LABELS_PATH = DATA_DIR + '/train-labels-idx1-ubyte'
TEST_IMAGES_PATH = DATA_DIR + '/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = DATA_DIR + '/t10k-labels-idx1-ubyte'

DIGIT_HEIGHT = 28
DIGIT_WIDTH = 28
DIGIT_BYTES = DIGIT_HEIGHT * DIGIT_WIDTH

NUM_DIGITS_TRAIN = 60000
NUM_DIGITS_TEST = 10000


def main():
    print("some training set examples")
    for digit_img, label in itertools.izip(
            read_image_set(TRAINING_IMAGES_PATH, 5),
            read_label_set(TRAINING_LABELS_PATH, 5)):
        print(label)
        cv2.imshow('digit', digit_img)
        cv2.waitKey(0)

    print("some test set examples")
    for digit_img, label in itertools.izip(
            read_image_set(TEST_IMAGES_PATH, 5),
            read_label_set(TEST_LABELS_PATH, 5)):
        print(label)
        cv2.imshow('digit', digit_img)
        cv2.waitKey(0)


def training_images(num_images=NUM_DIGITS_TRAIN):
    for img in read_image_set(TRAINING_IMAGES_PATH, num_images):
        yield img


def training_labels(num_labels=NUM_DIGITS_TRAIN):
    for label in read_label_set(TRAINING_LABELS_PATH, num_labels):
        yield label


def test_images(num_images=NUM_DIGITS_TEST):
    for img in read_image_set(TEST_IMAGES_PATH, num_images):
        yield img


def test_labels(num_labels=NUM_DIGITS_TEST):
    for label in read_label_set(TEST_LABELS_PATH, num_labels):
        yield label


def read_image_set(path, num_digits):
    """ Yield the training image set of 60000 digits, 28x28 pixels,
        8 bits per pixel.

        Yields:
            digit image: np.array((28, 28), dtype=np.uint8)
    """
    with open(path, 'rb') as infile:
        magic_num = infile.read(4)
        assert(magic_num == '\x00\x00\x08\x03')
        # read past unused data
        infile.read(12)
        for i in range(num_digits):
            img = np.array(
                [ord(x) for x in infile.read(DIGIT_BYTES)], dtype=np.uint8)
            yield img.reshape((DIGIT_HEIGHT, DIGIT_WIDTH))


def read_label_set(path, num_labels):
    """ Yield the training set labels.

        Yields:
            number in [0, 9]
    """
    with open(path, 'rb') as infile:
        magic_num = infile.read(4)
        assert(magic_num == '\x00\x00\x08\x01')
        # read past unused data
        infile.read(4)
        for i in range(num_labels):
            yield ord(infile.read(1))


if __name__ == '__main__':
    main()
