from __future__ import print_function

import cv2
import numpy as np
import itertools


# training set
DIGIT_HEIGHT = 28
DIGIT_WIDTH = 28
DIGIT_BYTES = DIGIT_HEIGHT * DIGIT_WIDTH

NUM_DIGITS = 60000


def main():
    print("some training set examples")
    i = 0
    for digit_img, label in itertools.izip(
            read_training_image_set('train-images-idx3-ubyte'),
            read_training_label_set('train-labels-idx1-ubyte')):
        print(label)
        cv2.imshow('digit', digit_img)
        cv2.waitKey(0)
        i += 1
        if i > 5:
            break

    print("some test set examples")
    i = 0
    for digit_img, label in itertools.izip(
            read_training_image_set('t10k-images-idx3-ubyte'),
            read_training_label_set('t10k-labels-idx1-ubyte')):
        print(label)
        cv2.imshow('digit', digit_img)
        cv2.waitKey(0)
        i += 1
        if i > 5:
            break


def read_image_set(path, num_digits):
    """ Yield the training image set of 60000 digits, 28x28 pixels,
        8 bits per pixel.

        Yields:
            digit image: np.array((28, 28), dtype=np.uint8)
    """
    with open(path, 'rb') as infile:
        magic_num = infile.read(4)
        assert(magic_num == '\x00\x00\x08\x03')
        # num_images = infile.read(4)
        rows = infile.read(4)
        cols = infile.read(4)
        for i in range(num_digits):
            img = np.array([ord(x) for x in infile.read(DIGIT_BYTES)], dtype=np.uint8)
            yield img.reshape((DIGIT_HEIGHT, DIGIT_WIDTH))


def read_label_set(path, num_labels):
    """ Yield the training set labels.

        Yields:
            number in [0, 9]
    """
    with open(path, 'rb') as infile:
        magic_num = infile.read(4)
        assert(magic_num == '\x00\x00\x08\x01')
        # num_labels = infile.read(4)
        for i in range(num_labels):
            yield ord(infile.read(1))


if __name__ == '__main__':
    main()
