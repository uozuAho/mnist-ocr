""" kNN digit classifier, with no pre processing.
"""

from utils import mnist
from utils import knn
from utils import classifier

NUM_TRAINS = 1000
NUM_TESTS = 1000


class KnnNoProcessing(knn.KnnDigitClassifier):
    pass


if __name__ == '__main__':
    runner = classifier.ClassifierRunner(KnnNoProcessing())
    runner.train(mnist.training_images(NUM_TRAINS), mnist.training_labels(NUM_TRAINS))
    runner.run(mnist.test_images(NUM_TESTS), mnist.test_labels(NUM_TESTS))
    print(runner.get_report_str())
