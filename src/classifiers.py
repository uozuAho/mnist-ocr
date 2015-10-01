import knn_no_processing
import knn_normsize_deslant
import svm


CLASSIFIERS = [
    knn_no_processing.KnnNoProcessing(),
    knn_normsize_deslant.KnnNormSizeDeslant(),
    svm.SvmDeslantHog()
]
