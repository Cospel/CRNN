import numpy as np
import tensorflow as tf
import imutils
import cv2
import random

from scipy.misc import imread, imresize, imsave


def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def read_image(image):
    """ 
        Read Image with scipy im read method to grayscale.
    """
    im_arr = imread(image, mode='L')
    r, c = np.shape(im_arr)
    return im_arr, c

def resize_image(im_arr, input_width, height):
    """
        Resize an image to the "good" input size
    """
    im_arr = imutils.resize(im_arr, height=height)
    r, c = im_arr.shape

    if c > input_width or ((c - 30) > input_width and random.random() < 0.5):
        c = input_width
        ratio = float(input_width) / c
        final_arr = cv2.resize(im_arr, (input_width, height))
    else:
        final_arr = np.zeros((height, input_width))
        im_arr_resized = im_arr
        final_arr[:, 0 : np.shape(im_arr_resized)[1]] = im_arr_resized[:, 0 : np.shape(im_arr_resized)[1]]

    return final_arr, c


def label_to_array(label, char_vector):
    try:
        return [char_vector.index(x) for x in label if x in char_vector]
    except Exception as ex:
        print("Error generating label to array")
        raise ex

def ground_truth_to_word(ground_truth, char_vector):
    """
        Return the word string based on the input ground_truth
    """

    try:
        return ''.join([char_vector[i] for i in ground_truth if i != -1])
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
