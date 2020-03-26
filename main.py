import random

import cv2
import numpy as np
from scipy import signal


def main():
    image = cv2.imread('binary.png')
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1", binary)
    noise = add_noise(binary, 0.1)
    # er_3 = erosion(noise, shape='cross', size=3)
    # print(noise_percentage(binary, noise))
    # open = opening(noise, shape='square', size=3)
    # close = closing(noise, shape='square', size=3)
    # table = filter_with_table(noise)
    # inner_contour = contour(binary, 'square', 3)
    # outter_contour = contour(binary, 'square', 3, inner=False)
    hor_contour = oriented_contour(binary)
    # ver_contour = oriented_contour(binary)
    # print(np.linalg.norm(inner_contour - inner_contour))
    cv2.imshow("1", noise)
    # cv2.imshow("2", outter_contour)
    cv2.imshow("3", hor_contour)
    # cv2.imshow("4", ver_contour)
    cv2.waitKey(0)


def reverse_decision(probability):
    return random.random() < probability


def add_noise(image, prob):
    normalized_image = image / 255
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i][j] = (normalized_image[i][j] + 1) % 2 if reverse_decision(prob) else normalized_image[i][j]
    return new_image * 255


def dilatation(image, shape='square', size=3):
    normalized_image = image / 255
    result = __dilatation(normalized_image, size, shape)
    return result * 255


def erosion(image, shape='square', size=3):
    normalized_image = image / 255
    result = __erosion(normalized_image, size, shape)
    return result * 255


def __dilatation(image, size=3, shape='square', mask=None):
    mask = (get_square_mask(size) if shape == 'square' else get_cross_mask(size)) if mask is None else mask
    conv_result = signal.convolve2d(image, mask, mode='same', fillvalue=0)
    for i in range(conv_result.shape[0]):
        for j in range(conv_result.shape[1]):
            conv_result[i][j] = 1 if conv_result[i][j] > 0 else 0
    return conv_result


def __erosion(image, size=3, shape='square', mask=None):
    mask = (get_square_mask(size) if shape == 'square' else get_cross_mask(size)) if mask is None else mask
    conv_result = signal.convolve2d(image, mask, mode='same', fillvalue=1)
    elements_in_mask = size ** 2 if shape == 'square' else 2 * size - 1
    for i in range(conv_result.shape[0]):
        for j in range(conv_result.shape[1]):
            conv_result[i][j] = 0 if conv_result[i][j] < elements_in_mask else 1
    return conv_result


def filter_with_table(image):
    normalized_image = image / 255
    mask = get_cross_mask(3)
    conv_result = signal.convolve2d(normalized_image, mask, mode='same', fillvalue=1)
    for i in range(conv_result.shape[0]):
        for j in range(conv_result.shape[1]):
            if normalized_image[i][j] == 0:
                if conv_result[i][j] == 4.:
                    conv_result[i][j] = 1
                else:
                    conv_result[i][j] = 0

            if normalized_image[i][j] == 1:
                if conv_result[i][j] == 0.:
                    conv_result[i][j] = 0
                else:
                    conv_result[i][j] = 1
    return conv_result * 255


def opening(image, size, shape):
    return dilatation(erosion(image, shape, size), shape, size)


def closing(image, size, shape):
    return erosion(dilatation(image, shape, size), shape, size)


def noise_percentage(image, image_with_noise):
    noise_pixels = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] != image_with_noise[i][j]:
                noise_pixels += 1
    return noise_pixels / image_with_noise.size


def get_cross_mask(size):
    mask = np.zeros((size, size))
    mask = mask.astype('int32')
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i == size // 2) or (j == size // 2):
                mask[i][j] = 1
    return mask


def get_square_mask(size):
    mask = np.ones((size, size))
    return mask


def contour(image, shape, size, inner=True):
    operation_result = erosion(image, shape, size) if inner else dilatation(image, shape, size)
    contour_result = (image / 255 + operation_result / 255) % 2 * 255
    return contour_result


def oriented_contour(image, orientation='vertical'):
    image = image / 255
    mask = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) if orientation == 'vertical' else \
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    result = __dilatation(image, mask=mask)
    contour_result = (image + result) % 2 * 255
    return contour_result


main()
