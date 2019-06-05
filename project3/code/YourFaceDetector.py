import cv2
import numpy as np
import sys
import os
import random
import math
import pickle
import json
from Utils.helpers import *

class TrainingData:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.integral_image = calculate_integral_image(image)

class WeakClassifier:
    def __init__(self, feature, threshold, polarity):
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, integral_image, factor):
        positive_val = negative_val = 0
        for pos_feature in self.feature[0]:
            positive_val += compute_feature_value(pos_feature, integral_image)

        for neg_feature in self.feature[1]:
            negative_val += compute_feature_value(neg_feature, integral_image)

        feature_val = positive_val - negative_val

        return 1 if self.polarity * feature_val < self.polarity * self.threshold else 0

def normalize_vector(vector):
    if len(vector) <= 0:
        return vector

    den = sum([x ** 2 for x in vector])

    i = 0
    while i < len(vector):
        vector[i] = 1.0 * vector[i] / math.sqrt(den)
        i += 1

    return vector
        

def train_layer(weights, features, feature_image_matrix, flags, training_data, layer):
    best_weak_classifiers = []
    alphas = []
    for _ in range(layer):
        weights = normalize_vector(weights)
        best_weak_classifier, error, accuracy = train_classifier(feature_image_matrix, flags, features, weights, training_data)
        beta = error / (1.0 - error)
        for i in range(len(accuracy)):
            weights[i] *= (beta ** (1 - accuracy[i]))
        alphas.append(math.log(1.0/beta))
        best_weak_classifiers.append(best_weak_classifier)

    return best_weak_classifiers, alphas

def train_classifier(feature_image_matrix, flags, features, weights, training_data):
    P = sum([weight for weight, flag in zip(weights, flags) if flag == 1])
    N = sum([weight for weight, flag in zip(weights, flags) if flag == 0])

    classifiers = []
    total_features = len(feature_image_matrix)
    for index, feature_row in enumerate(feature_image_matrix):
        positive_seen_so_far = negative_seen_so_far = positive_weight = negative_weight = 0
        min_error = sys.maxsize
        best_feature = None
        best_threshold = None
        best_polarity = None
        for weight, feature, flag in sorted(zip(weights, feature_row, flags), key=lambda x: x[1]):
            error = min(positive_weight + N - negative_weight, negative_weight + P - positive_weight)
            if error < min_error:
                min_error = error
                best_feature = features[index]
                best_threshold = feature

                if positive_seen_so_far > negative_seen_so_far:
                    best_polarity = 1
                else:
                    best_polarity = -1

            if flag == 1:
                positive_seen_so_far += 1
                positive_weight += weight
            else:
                negative_seen_so_far += 1
                negative_weight += weight
        
        classifiers.append(WeakClassifier(best_feature, best_threshold, best_polarity))

    best_classifier = None
    best_error = sys.maxsize
    best_accuracy = None
    for classifier in classifiers:
        error, accuracy = 0, list()
        for data, weight in zip(training_data, weights):
            correctness = abs(classifier.classify(data.integral_image, 1) - data.type)
            accuracy.append(correctness)
            error += weight * correctness
        error = error / len(training_data)
        if error < best_error:
            best_classifier = classifier
            best_error = error
            best_accuracy = accuracy
    return best_classifier, best_error, best_accuracy

def save_trained_model(classifiers, alphas):
    print("Saving Model")
    with open(os.path.abspath("./Trained/Classifers.pkl"), 'wb') as classifiers_file:
            pickle.dump(classifiers, classifiers_file)

    with open(os.path.abspath("./Trained/Alphas.pkl"), 'wb') as alphas_file:
            pickle.dump(alphas, alphas_file)

    print("Model Saved!")

def train(initial_weights, features, feature_image_matrix, flags, training_data, layers = [5, 10, 25, 50]):
    print("Starting to train")

    all_classifiers = []
    all_alphas = []
    for layer in layers:
        print("In layer {}".format(layer))
        best_weak_classifiers, alphas = train_layer(np.copy(initial_weights), features, feature_image_matrix, flags, training_data, layer)
        all_classifiers.append(best_weak_classifiers)
        all_alphas.append(alphas)

    save_trained_model(all_classifiers, all_alphas)

def load_trained_model():
    with open(os.path.abspath("./Trained/Classifers.pkl"), 'rb') as classifiers_file:
        classifiers = pickle.load(classifiers_file)

    with open(os.path.abspath("./Trained/Alphas.pkl"), 'rb') as alphas_file:
        alphas = pickle.load(alphas_file)

    return classifiers, alphas

def is_overlap(box_tuple, x, y):
    if y > box_tuple[1] and y < box_tuple[1] + box_tuple[3] and x > box_tuple[0] and x < box_tuple[0] + box_tuple[2]:
        return True

    return False


def detect(image, classifiers, alphas):
    max_height, max_width = image.shape

    h = w = MIN_WIDTH = MIN_HEIGHT = 100
    factor = 1

    result = dict()

    while h < image.shape[0] and w < image.shape[1]:
        for i in range(0, image.shape[0] - h, 10):
            for j in range(0, image.shape[1] - w, 10):
                box = cv2.resize(image[i: i + h, j: j + w], (19, 19))
                ii = calculate_integral_image(box)
                is_face = True
                index = 0
                net_total = 0
                while index < len(classifiers):
                    total = 0
                    for classifier, alpha in zip(classifiers[index], alphas[index]):
                        total += alpha * classifier.classify(ii, factor)

                    if total < 0.65 * sum(alphas[index]):
                        is_face = False
                        break

                    net_total += total
                    index += 1

                if is_face:
                    x_center = i + h // 2
                    y_center = j + w // 2
                    has_overlap = False

                    delete_list = []
                    should_include = True
                    for result_box, threshold in result.items():
                        if is_overlap(result_box, x_center, y_center) or is_overlap((i, j, h, w), result_box[0] + result_box[2] // 2, result_box[1] + result_box[3] // 2):
                            has_overlap = True

                            if net_total >= threshold:
                                delete_list.append(result_box)
                            else:
                                should_include = False

                    if not has_overlap or should_include:
                        result[(i, j, h, w)] = net_total

                    for x in delete_list:
                        del result[x]

        h = int(h * 1.5)
        w = int(w * 1.5)
        
    return result


def test_main():
    img_dir = sys.argv[1]
    classifiers, alphas = load_trained_model()

    final_result = list()

    total_files = len(os.listdir(os.path.abspath(img_dir)))
    print("Processing total: {} images".format(total_files))
    count = 0
    for image in os.listdir(os.path.abspath(img_dir)):
        if (image.lower().endswith('jpg') or image.lower().endswith('png') or image.lower().endswith('jpeg')):
            count += 1
            print("Now processing {} of {}".format(count, total_files))
            img = read_image(image, img_dir, resize=False)
            image_result = detect(img, classifiers, alphas)

            for res_tup, _ in image_result.items():
                final_result.append({
                    "iname": image,
                    "bbox": [res_tup[1], res_tup[0], res_tup[3], res_tup[2]]
                })

    with open(os.path.abspath("./results.json"), "w") as out:
        json.dump(final_result, out)

def train_main():
    pos_img_dir = sys.argv[1]
    neg_img_dir = sys.argv[2]

    training_data = []
    pos_count = neg_count = 0
    flags = []

    for image in os.listdir(os.path.abspath(pos_img_dir)):
        if (image.lower().endswith('jpg') or image.lower().endswith('png') or image.lower().endswith('jpeg')):
            img = read_image(image, pos_img_dir)
            training_data.append(TrainingData(img, 1))
            flags.append(1)
            pos_count += 1

    for image in os.listdir(os.path.abspath(neg_img_dir)):
        if (image.lower().endswith('jpg') or image.lower().endswith('png') or image.lower().endswith('jpeg')):
            img = read_image(image, neg_img_dir)
            training_data.append(TrainingData(img, 0))
            flags.append(0)
            neg_count += 1

    initial_weights = np.zeros(len(training_data))
    for i in range(len(training_data)):
        count = pos_count if training_data[i].type == 1 else neg_count
        initial_weights[i] = 1.0 / (2 * count)

    features = create_features(19, 19)

    feature_image_matrix = apply_features(features, training_data)

    train(initial_weights, features, feature_image_matrix, flags, training_data)
    
def main():
    if len(sys.argv) < 2:
        raise ValueError("Insufficient number of arguments")

    if len(sys.argv) < 4:
        test_main()
        return

    if sys.argv[3] == "train":
        train_main()
        return

if __name__ == '__main__':
    main()