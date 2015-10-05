from utils import classifier as cf
from utils import mnist

import classifiers


TEST_TRAIN_SIZE = 100
TEST_TEST_SIZE = 100


def main():
    for classifier in classifiers.CLASSIFIERS:
        run(classifier, TEST_TRAIN_SIZE, TEST_TEST_SIZE)
    print("OK")


def run(classifier, train_size, test_size):
    print(classifier)
    runner = cf.ClassifierRunner(classifier)
    runner.train(mnist.training_images(train_size), mnist.training_labels(train_size))
    runner.run(mnist.test_images(test_size), mnist.test_labels(test_size))


if __name__ == "__main__":
    main()
