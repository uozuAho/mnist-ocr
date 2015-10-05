import time


class GenericClassifier(object):
    """
    Interface for all classifiers to implement
    """

    def train(self, images, labels):
        """ Train on the given set of images and labels.
        :param images: list/generator of digit images
        :param labels: list/generator of labels of images
        :return: None
        """
        pass

    def classify(self, image):
        """ Classify a single digit image
        :param image: image of single digit to classify
        :return: int: digit classification
        """
        return 0


class ClassifierRunner(object):
    """
    Class for running classifiers and gathering performance information
    """

    def __init__(self, classifier):
        self.train_time = 0
        self.classify_time = 0
        self.num_classifies = 0
        self.classifier = classifier
        self.stats = ClassifierStats()

    def train(self, images, labels):
        start = time.clock()
        self.classifier.train(images, labels)
        self.train_time = time.clock() - start

    def run(self, images, labels):
        start = time.clock()
        for image, label in zip(images, labels):
            cls = self.classifier.classify(image)
            self.stats.add(label, cls)
            self.num_classifies += 1
        self.classify_time = time.clock() - start

    def get_report_str(self, short=False):
        s =  "train time:    {0:.2f}s\n".format(self.train_time)
        s += "classify time: {0:.2f}s ".format(self.classify_time)
        s += "({0}s / image)\n".format(self.classify_time / self.num_classifies)
        if not short:
            s += str(self.stats)
        return s


class ClassifierStats(object):
    """
    Class for collecting performance stats of a classifier
    """

    def __init__(self):
        self.num_classifications = 0
        self.num_correct = 0
        self.label_stats = {}

    def add(self, label, classification):
        self.num_classifications += 1
        if label == classification:
            self.num_correct += 1
        if label not in self.label_stats:
            self.label_stats[label] = LabelStats(label)
        self.label_stats[label].add(classification)

    def __str__(self):
        s = "Total accuracy: {0:.2f}\n".format(
            float(self.num_correct) / self.num_classifications
        )
        sorted_keys = list(self.label_stats.keys())
        sorted_keys.sort()
        for label in sorted_keys:
            stats = self.label_stats[label]
            s += "{0} (count {1}): Accuracy: {2:.2f}".format(
                label, stats.num_classifications(), stats.get_accuracy()
            )
            incorrect_label, incorrect_fraction = stats.most_incorrects()
            if incorrect_label is not None:
                s += ", most incorrectly classified as {0} ({1:.2f})\n".format(
                    incorrect_label, incorrect_fraction
                )
            else:
                s += "\n"
        return s


class LabelStats(object):
    """ Statistics for a single label of the test set.
    """

    def __init__(self, label):
        self.label = label
        # dict of label : count
        # what labels this label has been classified as
        self.classifications = {}

    def add(self, classification):
        if classification in self.classifications:
            self.classifications[classification] += 1
        else:
            self.classifications[classification] = 1

    def num_classifications(self):
        """
        :return: Total number of classifications made against this label
        """
        return sum(self.classifications.values())

    def get_accuracy(self):
        """ Get how often this label was correctly classified,
            as a fraction of all classifications made.
        """
        if self.label in self.classifications:
            return float(self.classifications[self.label]) / self.num_classifications()
        return 0

    def get_classifications(self):
        """
        :return: a list of (label, count) classifications made against this label,
                 sorted by count (descending).
        """
        classifications = [(l, self.classifications[l]) for l in self.classifications.keys()]
        classifications = sorted(classifications, key=lambda x: x[1], reverse=True)
        return classifications

    def most_incorrects(self):
        label_out = None
        fraction = None
        for label, count in self.get_classifications():
            if label != self.label:
                label_out = label
                fraction = float(count) / self.num_classifications()
                break
        return label_out, fraction


if __name__ == "__main__":
    ls = LabelStats('a')
    ls.add('a')
    ls.add('a')
    ls.add('b')
    ls.add('b')
    ls.add('b')
    assert(ls.num_classifications() == 5)
    assert(ls.get_accuracy() == 2.0 / 5)
    assert(ls.get_classifications() == [('b', 3), ('a', 2)])
    assert(ls.most_incorrects() == ('b', 0.6))

    cs = ClassifierStats()
    cs.add('a', 'a')
    cs.add('a', 'a')
    cs.add('a', 'b')
    cs.add('b', 'b')
    cs.add('b', 'a')
    print(cs)

    images = ['a', 'b', 'c']
    labels = ['a', 'b', 'c']
    cr = ClassifierRunner(GenericClassifier())
    cr.train(images, labels)
    cr.run(images, labels)
    print(cr.get_report_str())
