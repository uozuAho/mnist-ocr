import cv2
import numpy as np


def deslant(img):
    """ Vertically straighten an image based on image moments.

        Taken from
        https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html#svm-opencv
    """
    h, w = img.shape[:2]
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * h * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (h, h),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def crop_to_outer_contour(img, size=None):
    """ Crop (and resize) the image to the largest outer contour
    :param img:  Image to resize
    :param size: [optional] Resize cropped image to this size (width, height)
    :return:     Cropped (and resized) image
    """
    rects = [cv2.boundingRect(c) for c in find_outer_contours(img)]

    def rect_area(rect):
        return rect[2] * rect[3]

    rects = sorted(rects, key=rect_area, reverse=True)
    x, y, w, h = rects[0]
    cropped_img = img[y:y + h, x:x + w]
    if size is None:
        return cropped_img
    else:
        return cv2.resize(cropped_img, size, interpolation=cv2.INTER_AREA)


def find_outer_contours(img):
    """ Return outer contour(s) of the given image
    :return: list of contours
    """
    # copy image, as findContours modifies the input image
    img_copy, contours, hierarchy = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def hog(img, num_bins, edge_num_cells=2):
    """ Histogram of oriented gradients
    :param img:             image to process
    :param edge_num_cells:  cut img into cells: 2 = 2x2, 3 = 3x3 etc.
    :return:
    """
    if edge_num_cells != 2:
        raise NotImplementedError
    w, h = img.shape[:2]
    cut_x = w / 2
    cut_y = h / 2
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    ang = np.int32(num_bins * ang / (2 * np.pi))
    bin_cells = (ang[:cut_x, :cut_y], ang[cut_x:, :cut_y],
                 ang[:cut_x, cut_y:], ang[cut_x:, cut_y:])
    mag_cells = (mag[:cut_x, :cut_y], mag[cut_x:, :cut_y],
                 mag[:cut_x, cut_y:], mag[cut_x:, cut_y:])
    hists = [np.bincount(
        b.ravel(), m.ravel(), num_bins) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist
