import cv2
import numpy as np

import mnist_reader as mnist


class KnnDigitClassifier:
    def __init__(self, preprocess_func, training_set_size=mnist.NUM_DIGITS_TRAIN):
        """ Init classifier.

            Parameters:
                preprocess_func:
                    Function that takes an image and outputs an image.
                    Used before training and classification.
                    This must output images of consistent size and type.
                training_set_size:
                    Number of images from the MNIST training set to
                    train on.
        """
        # KNN predictor object
        self.knn = None
        # Image preprocessor
        self.preprocess = preprocess_func
        self.__train(training_set_size)

    def predict(self, img, k=5):
        """ Predict the digit in the given image.

            Parameters:
                img (np.array):
                    Image to predict. Expected to be a 28x28 digit image
                    from the MNIST dataset.

            Returns:
                int: predicted digit
        """
        ret, result, neighbours, dist = self.predict_detailed(img, k)
        return int(ret)

    def predict_detailed(self, img, k=5):
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
        proc = self.preprocess(img)
        vec = proc.reshape(1, -1).astype(np.float32)
        return self.knn.findNearest(vec, k=k)

    def __train(self, num_trains=mnist.NUM_DIGITS_TRAIN):
        """ Train on the MNIST training set """
        # prepare the training data
        training_set = [self.preprocess(x) for x in mnist.training_images(num_trains)]
        training_labels = list(mnist.training_labels(num_trains))
        # get preprocessed image shape
        h, w = training_set[0].shape[:2]
        # convert the training set into vectors (instead of m*n images)
        training_set = np.array(training_set).reshape(-1, h*w).astype(np.float32)
        training_labels = np.array(training_labels)
        # create & train the knn object
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(training_set, cv2.ml.ROW_SAMPLE, training_labels)
