import cv2
import numpy as np
import sys
import os
import random
import string
from scipy.spatial import distance

orb = cv2.ORB_create(nfeatures = 1000)

def create_P_matrix(source_points, destination_points):
    ''' Helper matrix for creating homography matrix '''
    sub_matrices = []
    for i in range(len(source_points)):
        sub_matrix = np.zeros((2,9))
        sub_matrix[0][0] = -1 * source_points[i][0]
        sub_matrix[0][1] = -1 * source_points[i][1]
        sub_matrix[0][2] = -1
        sub_matrix[0][6] = source_points[i][0] * destination_points[i][0]
        sub_matrix[0][7] = source_points[i][1] * destination_points[i][0]
        sub_matrix[0][8] = destination_points[i][0]

        sub_matrix[1][3] = -1 * source_points[i][0]
        sub_matrix[1][4] = -1 * source_points[i][1]
        sub_matrix[1][5] = -1
        sub_matrix[1][6] = source_points[i][0] * destination_points[i][1]
        sub_matrix[1][7] = source_points[i][1] * destination_points[i][1]
        sub_matrix[1][8] = destination_points[i][1]
        sub_matrices.append(sub_matrix)

    last_row = np.zeros((1, 9))
    last_row[0][8] = 1
    sub_matrices.append(last_row)
    result = np.concatenate(sub_matrices, axis=0)
    return result

def find_homograph(source_points, destination_points):
    ''' Creates homography matrix from 4 source and destination points '''
    a = create_P_matrix(source_points, destination_points)
    b = np.zeros((9, 1))
    b[8][0] = 1
    u, s, v = np.linalg.svd(a)
    c = np.dot(u.T, b)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(v.T, w)
    x.resize((3, 3))
    return x

def ransac(source_points, destination_points, inlier_threshold):
    ''' Used to calculate the best homography matrix based on 4 randomnly chosen points and counting inliers '''
    max_inliers = - sys.maxsize - 1
    best_h = None
    
    for _ in range(500):
        random_pts = np.random.choice(range(0, len(source_points) - 1), 4, replace=False)

        src = []
        dst = []
        for i in random_pts:
            src.append(source_points[i])
            dst.append(destination_points[i])

        h = find_homograph(src, dst)
        
        count = 0
        for index in range(len(source_points)):
            src_pt = np.append(source_points[index], 1)

            dest_pt = np.dot(h, src_pt.T)

            dest_pt = np.true_divide(dest_pt, dest_pt[2])[0: 2]

            if distance.euclidean(destination_points[index], dest_pt) <= inlier_threshold:
                count += 1
                
        if count > max_inliers:
            max_inliers = count
            best_h = h

    return best_h, max_inliers

def read_image(img_name, img_dir):
    ''' Reads image from the specified directory '''
    img_path = os.path.join(os.path.abspath(img_dir), img_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def warp_images(img1, img2, H):
    ''' Warps image 2 and stitches it to image 1 '''
    h1, w1 = img1.shape[ :2]
    h2, w2 = img2.shape[ :2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]],[0, 1, t[1]],[0, 0, 1]])
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

def get_keypoint_descriptors(img):
    ''' Uses ORB to extract keypoints and descriptors from an image '''
    return orb.detectAndCompute(img, None)
    
def find_matches(keypoints1, descriptors1, keypoints2, descriptors2):
    ''' Used to find matches between the descriptors from image1 to image 2 using hamming distance on bits '''
    results = list()
    descriptor_size = len(descriptors1[0])
    overall_min = sys.maxsize
    
    for i in range(len(descriptors1)):
        min_diff = sys.maxsize
        min_right_index = -1
        for j in range(len(descriptors2)):
            total_diff_bits = 0
            for k in range(descriptor_size):
                x = int('{0:08b}'.format(descriptors1[i][k]))
                y = int('{0:08b}'.format(descriptors2[j][k]))
                total_diff_bits += np.count_nonzero(x != y)
            
            if total_diff_bits < min_diff:
                min_diff = total_diff_bits
                min_right_index = j

                if min_diff < overall_min:
                    overall_min = min_diff
        results.append([list(keypoints1[i].pt), list(keypoints2[min_right_index].pt)])

    if overall_min > 10:
        return None

    return results 

def stitcher(img1, img2, gray1, gray2):
    ''' Used to stitch two images if they are stitchable '''
    keypoints1, descriptors1 = get_keypoint_descriptors(gray1)
    keypoints2, descriptors2 = get_keypoint_descriptors(gray2)
    matches = find_matches(keypoints1, descriptors1, keypoints2, descriptors2)

    if matches is None:
        return None, None

    src = np.float32([match[0] for match in matches]).reshape(-1,2)
    dst = np.float32([match[1] for match in matches]).reshape(-1,2)

    homograph = None
    i = 0
    threshold = min(len(descriptors1), len(descriptors2))

    while True:
        homograph, max_inliers = ransac(src, dst, i)

        if 0.1 * threshold <= max_inliers < 0.3 * threshold:
            break
        elif max_inliers < 0.1 * threshold:
            i += 1
        else: i -= 1

    return warp_images(img2, img1, homograph), warp_images(gray2, gray1, homograph)

def write_output_image(img, output_dir, file_name):
    ''' Writes the final image to disk '''
    file_name += '.jpg'
    cv2.imwrite(os.path.join(os.path.abspath(output_dir), file_name), img)

def main():
    # Check if source directory is specified while running the script
    if len(sys.argv) < 2:
        raise ValueError("Insufficient number of arguments")

    img_dir = sys.argv[1]
    image_queue = list()

    # Only add images from source directory to the queue
    for image in os.listdir(os.path.abspath(img_dir)):
        if (image.lower().endswith('jpg') or image.lower().endswith('png') or image.lower().endswith('jpeg')) and image.lower() != 'panorama.jpg':
            image_queue.append(os.path.join(os.path.abspath(img_dir), image))

    # Check if there are atleast two images to stitch
    if len(image_queue) < 2:
        raise ValueError("Need atleast two images to stitch!")

    result_image = None
    result_gray = None

    # Select two images that can be stitched
    for i in range(2):
        image1 = image_queue[i]
        image2 = image_queue[i+1]
        img1, gray1 = read_image(image1, img_dir)
        img2, gray2 = read_image(image2, img_dir)

        result_image, result_gray = stitcher(img1, img2, gray1, gray2)

        if result_image is not None:
            image_queue.remove(image1)
            image_queue.remove(image2)
            break

    # If there are more than two images then stitch this third image to the panorama obtained earlier
    if len(image_queue) > 0:
        image = image_queue[-1]
        img, gray = read_image(image, img_dir)

        result_image, result_gray = stitcher(result_image, img, result_gray, gray)

    # Finally write the panorama to disk
    write_output_image(result_image, img_dir, 'panorama')

if __name__ == "__main__":
    main()