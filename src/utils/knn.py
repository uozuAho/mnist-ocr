import cv2
import numpy as np

import classifier


class KnnDigitClassifier(classifier.GenericClassifier):

    def __init__(self):
        # OpenCV kNN classifier
        self.knn = cv2.ml.KNearest_create()

    def train(self, images, labels):
        """ Train on the given training set """
        # prepare the training data
        training_set = list(images)
        training_labels = list(labels)
        # get preprocessed image shape
        h, w = training_set[0].shape[:2]
        # convert the training set into vectors (instead of m*n images)
        training_set = np.array(training_set).reshape(-1, h*w).astype(np.float32)
        training_labels = np.array(training_labels)
        # create & train the knn object
        self.knn.train(training_set, cv2.ml.ROW_SAMPLE, training_labels)

    def classify(self, image):
        """ Classify the digit in the given image.

            Parameters:
                img (np.array):
                    Image to predict. Expected to be in the same
                    format as images used for training.
        """
        ret, result, neighbours, dist = self.predict_detailed(image)
        return int(ret)

    def predict_detailed(self, image, k=5):
        """ Directly returns results of knn.findNearest().

            Parameters:
                img (np.array):
                    Image to predict. Expected to be a 28x28 digit image
                    from the MNIST dataset.

            Returns:
                (ret, result, neighbours, distance)

                ret -        nearest neighbour: float
                result -     array containing ret??
                neighbours - array of k nearest neighbours
                distance -   array of distances of neighbours the given image
        """
        vec = image.reshape(1, -1).astype(np.float32)
        return self.knn.findNearest(vec, k=k)
