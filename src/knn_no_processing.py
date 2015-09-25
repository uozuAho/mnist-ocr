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

from utils import mnist_reader as mnist
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
