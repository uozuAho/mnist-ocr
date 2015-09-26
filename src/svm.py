from __future__ import print_function

import cv2
import numpy as np

from utils import classifier as cs


DIGIT_SIZE = 28
NUM_CELLS = 4
NUM_BINS = 16
KERNEL_TYPE = cv2.ml.SVM_LINEAR
SVM_TYPE = cv2.ml.SVM_C_SVC
SVM_C = 2.67
SVM_GAMMA = 5.383


class SvmDeslantHog(cs.GenericClassifier):

    def __init__(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setGamma(SVM_GAMMA)
        self.svm.setKernel(KERNEL_TYPE)
        self.svm.setC(SVM_C)
        self.svm.setType(SVM_TYPE)

    def train(self, images, labels):
        """ Train on the given training set """
        # prepare the training data
        training_set = [self.preprocess(img) for img in images]
        training_labels = list(labels)
        # convert the training set into vectors (instead of m*n images)
        training_set = np.array(training_set).reshape(-1, NUM_CELLS * NUM_BINS).astype(np.float32)
        training_labels = np.array(training_labels)
        # train the svm
        self.svm.train(samples=training_set, layout=cv2.ml.ROW_SAMPLE, responses=training_labels)

    def classify(self, image):
        proc = self.preprocess(image)
        arr = np.array(proc).reshape(-1, NUM_CELLS * NUM_BINS).astype(np.float32)
        _, result =  self.svm.predict(arr)
        classification = int(result.ravel()[0])
        return classification

    def preprocess(self, image):
        desl = deslant(image)
        return hog(desl)


def deslant(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5 * DIGIT_SIZE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (DIGIT_SIZE, DIGIT_SIZE),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(NUM_BINS * ang / (2 * np.pi))
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), NUM_BINS) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist
