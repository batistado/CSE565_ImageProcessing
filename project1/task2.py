"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import utils
from task1 import *   # you could modify this line

CHARACTER = 'a'

threshold_mapping = {
    'a': [0.938, 0, 0],
    'b': [0.97, 0.965, 0.973],
    'c': [0.96, 0, 0]
}

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/proj1-task2.jpeg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="./data/c.jpg",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def extract_template(img, template, threshold, coordinates):
    i = 0
    while i + len(template) < len(img):
        j = 0
        while j + len(template[0]) < len(img[0]):
            num = den_img = den_temp = 0
            for x in range(len(template)):
                for y in range(len(template[0])):
                    num += int(img[i + x][j + y]) * int(template[x][y])
                    den_img += int(img[i + x][j + y]) ** 2
                    den_temp += int(template[x][y]) ** 2
            
            res = num / np.sqrt(den_img * den_temp)

            if res > threshold:
                coordinates.append((i, j))

            j += 1
        
        i += 1

    return coordinates

def sample_small_template(template):
    ''' Sampling small (1/2 the size) using bilinear interpolation '''
    small_template = [[0 for _ in range(len(template[0]) // 2)] for _ in range(len(template) // 2)]

    for i in range(len(small_template)):
        for j in range(len(small_template[0])):
            sum = int(template[2*i + 0][2*j + 0])
            sum += int(template[2*i + 0][2*j + 1])
            sum += int(template[2*i + 1][2*j + 0])
            sum += int(template[2*i + 1][2*j + 1])
            
            small_template[i][j] = sum / 4

    return small_template

def sample_medium_template(template):
    ''' Sampling medium (2/3rd the size) using bilinear interpolation '''
    medium = []
    i = 0
    flag = True
    while i + 2 < len(template):
        row = []
        j = 0
        while j + 2 < len(template[0]):
            if j + 3 <= len(template[0]):
                sum = int(template[i][j])
                sum += int(template[i + 1][j])
                sum += int(template[i][j + 1])
                sum += int(template[i + 1][j + 1])
                row.append(sum / 4)

                sum = int(template[i][j + 1])
                sum += int(template[i + 1][j + 1])
                sum += int(template[i][j + 1 + 1])
                sum += int(template[i + 1][j + 1 + 1])
                row.append(sum / 4)

            elif j + 2 <= len(template[0]):
                sum = int(template[i][j])
                sum += int(template[i + 1][j])
                sum += int(template[i][j + 1])
                sum += int(template[i + 1][j + 1])
                row.append(sum / 4)
            j += 3
        
        if flag:
            i += 1
            flag = False
        else:
            i += 2
            flag = True

        medium.append(row)

    return medium


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    # raise NotImplementedError
    coordinates = []

    if (threshold_mapping[CHARACTER][0] != 0):
        extract_template(img, template, threshold_mapping[CHARACTER][0], coordinates)
    
    if (threshold_mapping[CHARACTER][1] != 0):
        extract_template(img, sample_small_template(template), threshold_mapping[CHARACTER][1], coordinates)

    if (threshold_mapping[CHARACTER][2] != 0):
        extract_template(img, sample_medium_template(template), threshold_mapping[CHARACTER][2], coordinates)

    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)

def main():
    global CHARACTER

    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)

    CHARACTER = os.path.splitext(os.path.split(args.template_path)[1])[0]

    coordinates = detect(img, template)

    template_name = "{}.json".format(CHARACTER)
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
