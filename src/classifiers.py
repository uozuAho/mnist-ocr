import knn_binary
import knn_no_processing
import knn_normsize_deslant
import svm


CLASSIFIERS = [
    knn_binary.KnnBinary(),
    knn_no_processing.KnnNoProcessing(),
    knn_normsize_deslant.KnnNormSizeDeslant(),
    svm.SvmDeslantHog()
]
