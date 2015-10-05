import cv2
import numpy as np

from utils import classifier as cs
from utils import imgproc


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
