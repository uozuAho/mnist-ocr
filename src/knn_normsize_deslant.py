""" kNN digit classifier, preprocessing images before training &
    classification. Preprocessing:
    - de-slant image based on image moments
    - normalise digit size by finding each digit's bounding box,
      cropping image to just the digit, then resizing to a pre-
      determined size.
"""

from utils import classifier as cs
from utils import imgproc
from utils import knn
from utils import mnist

# Size that digit images are normalised to
NORMALISED_SIZE = (20, 20)


class KnnNormSizeDeslant(knn.KnnDigitClassifier):

    def train(self, images, labels):
        super(KnnNormSizeDeslant, self).train(
            self.preprocess_all(images), labels)

    def classify(self, image):
        return super(KnnNormSizeDeslant, self).classify(self.preprocess(image))

    def preprocess(self, image):
        desl = imgproc.deslant(image)
        return imgproc.crop_to_outer_contour(desl, NORMALISED_SIZE)

    def preprocess_all(self, images):
        for image in images:
            yield self.preprocess(image)


if __name__ == '__main__':
    NUM_TRAINS = 100
    NUM_TESTS = 100
    runner = cs.ClassifierRunner(KnnNormSizeDeslant())
    runner.train(mnist.training_images(NUM_TRAINS), mnist.training_labels(NUM_TRAINS))
    runner.run(mnist.test_images(NUM_TESTS), mnist.test_labels(NUM_TESTS))
    print(runner.get_report_str())
