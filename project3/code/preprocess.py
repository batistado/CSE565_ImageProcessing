import os
import cv2
import string
import random


FOLDS_DIR = './Folds/'
PICS_DIR = './Pics/'
OUT_DIR = './ProcessedData/'

class Point:
    def __init__(self, x_coordinate, y_coordinate):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate

class FaceData:
    def __init__(self, major_radius, minor_radius, center_x, center_y):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.center_x = center_x
        self.center_y = center_y

class FoldData:
    def __init__(self, image_path):
        self.image_path = image_path
        self.faces = []

    def set_count(self, count):
        self.count = count

    def add_face_data(self, major_radius, minor_radius, center_x, center_y):
        self.faces.append(FaceData(major_radius, minor_radius, center_x, center_y))



def main():
    fold_datas = []

    count = 0
    for fold in os.listdir(os.path.abspath(FOLDS_DIR)):
        if fold.lower().endswith('txt'):
            with open(os.path.join(FOLDS_DIR, fold)) as fold_file:
                image_data = []
                isNewImage = True
                face_count = -1
                fold_data = None
                line = fold_file.readline()

                # First line read
                if line:
                    line = line.strip()
                    count += 1
                    fold_data = FoldData(line)
                
                line = fold_file.readline()

                while line:
                    line = line.strip()
                    if isNewImage:
                        fold_data.set_count(int(line))
                        face_count = int(line)
                        isNewImage = False
                    else:
                        if face_count == 0:
                            fold_datas.append(fold_data)

                            if len(line) < 2:
                                break

                            count += 1
                            fold_data = FoldData(line)
                            isNewImage = True
                        else:
                            face_count -= 1
                            line_split = line.split(" ")
                            fold_data.add_face_data(float(line_split[0]), float(line_split[1]), float(line_split[3]), float(line_split[4]))
                    line = fold_file.readline()

                fold_datas.append(fold_data)

        process_fold_data(fold_datas)
    print(count)

def randomString(stringLength=10):
    """ Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def is_overlapping(first_rectangle, second_rectangle):
    l1 = first_rectangle[0]
    r1 = first_rectangle[1]

    l2 = second_rectangle[0]
    r2 = second_rectangle[1]

    if (l1.x_coordinate > r2.x_coordinate or l2.x_coordinate > r1.x_coordinate):
        return False

    if (l1.y_coordinate > r2.y_coordinate or l2.y_coordinate > r1.y_coordinate):
        return False
  
    return True;

def process_non_faces(img, points):
    index = 0
    extension = '.jpg'

    while index < len(points):
        top_left = points[index][0]
        bottom_right = points[index][1]

        x_length = bottom_right.x_coordinate - top_left.x_coordinate
        y_length = bottom_right.y_coordinate - top_left.y_coordinate

        overlapping = True

        count = 0
        while overlapping and count < 10:
            random_x = random.randint(0, img.shape[1] - x_length)
            random_y = random.randint(0, img.shape[0] - y_length)

            random_rectangle = (Point(random_x, random_y), Point(random_x + x_length, random_y + y_length))

            for pt in points:
                overlapping = is_overlapping(random_rectangle, pt)

                if overlapping:
                    break

            if not overlapping:
                crop_img = img[random_y: random_y + int(y_length), random_x: random_x + int(x_length)]

                cv2.imwrite(OUT_DIR + 'NonFaces/' + randomString() + extension, crop_img)

            count += 1

        index += 1

            




def process_fold_data(fold_objs):
    overall_max_x = overall_max_y = -1
    for fold_obj in fold_objs:
        index = 1
        extension = '.jpg'
        img = cv2.imread(PICS_DIR + fold_obj.image_path + extension)

        points = []
        for face in fold_obj.faces:
            x = int(face.center_x - face.minor_radius)
            y = int(face.center_y - face.major_radius)
            crop_img = img[y + 50: y + int(2 * face.major_radius) + 1, x:x + int(2 * face.minor_radius) + 1]

            points.append((Point(x, y), Point(x + crop_img.shape[1], y + crop_img.shape[0])))

            overall_max_x = max(overall_max_x, crop_img.shape[1])
            overall_max_y = max(overall_max_y, crop_img.shape[0])

            name = '_'.join(fold_obj.image_path.split("/"))
            start_index = fold_obj.image_path.rfind('/') + 1
            end_index = fold_obj.image_path.rfind('.')

            cv2.imwrite(OUT_DIR + 'Faces/' + name + '_' + str(index) + extension, crop_img)
            index += 1

        print(PICS_DIR + fold_obj.image_path + extension)
        process_non_faces(img, points)

        
    print(overall_max_x, overall_max_y)

if __name__ == '__main__':
    main()