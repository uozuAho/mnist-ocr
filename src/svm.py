import cv2
import numpy as np

from utils import classifier as cs
from utils import imgproc


DIGIT_SIZE = 28
NUM_CELLS = 4
NUM_BINS = 16


class SvmDeslantHog(cs.GenericClassifier):

    def __init__(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setGamma(5.383)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setC(2.67)
        self.svm.setType(cv2.ml.SVM_C_SVC)

    def train(self, images, labels):
        """ Train on the given training set """
        # prepare the training data
        training_set = [self.preprocess(img) for img in images]
        training_labels = list(labels)
        # convert the training set into vectors (instead of m*n images)
        training_set = np.array(
            training_set).reshape(-1, NUM_CELLS * NUM_BINS).astype(np.float32)
        training_labels = np.array(training_labels)
        # train the svm
        self.svm.train(samples=training_set, layout=cv2.ml.ROW_SAMPLE,
                       responses=training_labels)

    def classify(self, image):
        proc = self.preprocess(image)
        arr = np.array(proc).reshape(-1, NUM_CELLS * NUM_BINS).astype(np.float32)
        _, result = self.svm.predict(arr)
        classification = int(result.ravel()[0])
        return classification

    def preprocess(self, image):
        desl = imgproc.deslant(image)
        return imgproc.hog(desl, NUM_BINS, NUM_CELLS / 2)


class SvmPca(cs.GenericClassifier):

    def __init__(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setGamma(2)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        # self.svm.setDegree(4)
        self.svm.setC(2.67)
        self.svm.setType(cv2.ml.SVM_C_SVC)

    def train(self, images, labels):
        # prepare the training data
        training_set = np.array([self.preprocess(img) for img in images])
        training_labels = list(labels)
        # convert the training set into vectors
        h, w = training_set[0].shape
        training_set = training_set.reshape(-1, h * w).astype(np.float32)
        training_labels = np.array(training_labels)
        # train the svm
        self.svm.train(samples=training_set, layout=cv2.ml.ROW_SAMPLE,
                       responses=training_labels)

    def classify(self, image):
        proc = self.preprocess(image)
        arr = np.array(proc).reshape(1, -1).astype(np.float32)
        _, result = self.svm.predict(arr)
        classification = int(result.ravel()[0])
        return classification

    def preprocess(self, image):
        desl = imgproc.deslant(image)
        w, u, vt = cv2.SVDecomp(desl.astype(np.float32))
        sigma = np.diag(w.ravel())
        n = 26
        return u[:n, :n].dot(sigma[:n, :n]).dot(vt[:n, :n])
