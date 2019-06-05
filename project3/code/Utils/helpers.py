import cv2
import numpy as np
import sys
import os
import random
import math
import pickle

def read_image(img_name, img_dir, resize = True):
    ''' Reads image from the specified directory '''
    img_path = os.path.join(os.path.abspath(img_dir), img_name)
    img = cv2.imread(img_path, 0)
    return img if not resize else cv2.resize(img, (24, 24))

def calculate_integral_image(image):    
    ''' Calculates integral image from the image '''
    i = 0
    h, w = image.shape
    integral_img = np.zeros((h, w))

    while i < len(integral_img):
        j = 0
        while j < len(integral_img[0]):
            integral_img[i][j] = image[i][j] + (integral_img[i - 1][j] if i - 1 >= 0 else 0) + (integral_img[i][j - 1] if j - 1 >= 0 else 0) - (integral_img[i - 1][j - 1] if j - 1 >= 0 and i - 1 >= 0 else 0)
            j += 1
        i += 1

    return integral_img

def create_features(height, width):
    ''' Created all 4 types of Haar features '''
    features = []

    if height <= 1 or width <= 1:
        return features

    w = 1
    while w <= width:
        h = 1
        while h <= height:
            j = 0
            while j + w < width:
                i = 0
                while i + h < height:
                    # Type 1 feature
                    if j + 2 * w < width:
                        features.append(([(j+w, i, w, h)], [(j, i, w, h)]))

                    # Type 2 feature
                    if i + 2 * h < height:
                        features.append(([(j, i, w, h)], [(j, i+h, w, h)]))
                    
                    # Type 3 feature
                    if j + 3 * w < width:
                        features.append(([(j+w, i, w, h)], [(j+2*w, i, w, h), (j, i, w, h)]))

                    # Type 4 feature
                    if i + 3 * h < height:
                        features.append(([(j, i+h, w, h)], [(j, i+2*h, w, h), (j, i, w, h)]))

                    i += 1
                j += 1
            h += 1
        w += 1

    return np.array(features)


def compute_feature_value(feature_tuple, integral_image):
    ''' Computes a feature value on an integral image '''
    x, y, w, h = feature_tuple

    if len(integral_image) <= y + h or len(integral_image[0]) <= x + w:
        return 0

    return integral_image[y + h][x + w] + integral_image[y][x] - integral_image[y + h][x] - integral_image[y][x + w]


def apply_features(features, training_data):
    ''' Apply all features to the training dataset '''
    feature_image_matrix = np.zeros((len(features), len(training_data)))

    i = 0
    for feature in features:
        row = []
        for training_obj in training_data:
            positive_val = negative_val = 0
            for pos_feature in feature[0]:
                positive_val += compute_feature_value(pos_feature, training_obj.integral_image)

            for neg_feature in feature[1]:
                negative_val += compute_feature_value(neg_feature, training_obj.integral_image)

            feature_val = positive_val - negative_val
            row.append(feature_val)

        feature_image_matrix[i] = row
        i += 1

    return feature_image_matrix