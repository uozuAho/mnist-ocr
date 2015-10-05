""" kNN digit classifier, converting images to binary before
    training and classification. Should (or should allow for)
    reduction in kNN object size.
"""

import cv2

from utils import classifier as cs
from utils import knn
from utils import mnist


class KnnBinary(knn.KnnDigitClassifier):

    def train(self, images, labels):
        super(KnnBinary, self).train(
            self.preprocess_all(images), labels)

    def classify(self, image):
        return super(KnnBinary, self).classify(self.preprocess(image))

    def preprocess(self, image):
        # [1]: threshold returns tuple (x, x, img), where x is
        # something I cbf figuring out
        return cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)[1]

    def preprocess_all(self, images):
        for image in images:
            yield self.preprocess(image)


if __name__ == '__main__':
    NUM_TRAINS = 100
    NUM_TESTS = 100
    runner = cs.ClassifierRunner(KnnBinary())
    runner.train(mnist.training_images(NUM_TRAINS), mnist.training_labels(NUM_TRAINS))
    runner.run(mnist.test_images(NUM_TESTS), mnist.test_labels(NUM_TESTS))
    print(runner.get_report_str())
