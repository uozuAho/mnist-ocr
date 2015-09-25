"""
Run all the classifiers for testing / benchmarking
"""

from __future__ import print_function

from utils import classifier as cf
from utils import mnist_reader as mnist

import knn_no_processing
import knn_normsize_deslant


TEST_TRAIN_SIZE = 100
TEST_TEST_SIZE = 100

BENCHMARK_TRAIN_SIZES = [
    10000,
    # 20000,
    # 40000,
    # 60000
]
BENCHMARK_TEST_SIZE = 10000

CLASSIFIERS = [
    knn_no_processing.KnnNoProcessing(),
    knn_normsize_deslant.KnnNormSizeDeslant()
]


def main():
    # test_all_classifiers()
    benchmark_all_classifiers()


def test_all_classifiers():
    for classifier in CLASSIFIERS:
        run(classifier, TEST_TRAIN_SIZE, TEST_TEST_SIZE, short_report=True)


def benchmark_all_classifiers():
    for classifier in CLASSIFIERS:
        for train_size in BENCHMARK_TRAIN_SIZES:
            run(classifier, train_size, BENCHMARK_TEST_SIZE)


def run(classifier, train_size, test_size, short_report=False):
    print(classifier)
    print("training set size: ", train_size)
    print("test set size:     ", test_size)
    runner = cf.ClassifierRunner(classifier)
    runner.train(mnist.training_images(train_size), mnist.training_labels(train_size))
    runner.run(mnist.test_images(test_size), mnist.test_labels(test_size))
    print(runner.get_report_str(short=short_report))


if __name__ == "__main__":
    main()
