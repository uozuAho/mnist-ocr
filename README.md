# mnist-ocr
Some attempts at OCR-ing the MNIST datasets.

The [MNIST database site](http://yann.lecun.com/exdb/mnist/) includes training &
test data, plus a review of many techniques and their performance. LeCun 98 (pdf
in papers/) also provides a broad overview of techniques.

To get the MNIST data files, run `get-mnist-data.sh`, or download the files
from the above site and use gunzip to decompress them.

# todo
- get memory usage info of classifiers
- parallelize knn classifiers
- try knn on binary images
- renew benchmark figures of all classifiers
- add SVM example
- test preprocessing stages separately, see what gets best results