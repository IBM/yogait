import os
import numpy as np
import cv2
import requests
from sklearn.preprocessing import normalize

URL = 'http://localhost:5000/model/predict'

COCO_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
               [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
               [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
               [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
               [255, 0, 170], [255, 0, 85]]


def cart2pol(x, y):
    """
    Convert a Cartesian [x,y] numpy array to a polar [rho, phi] representation.
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def convert(coordinates):
    """
    convert from Cartesian coordinate system to polar so that
    relative position on screen doesn't matter
    input format:  [length] [x, y]
    return format: [length] [rho, phi]
    """
    center = np.mean(coordinates, axis=0, dtype=np.float32)
    x = np.subtract(np.array(coordinates, dtype=np.float32), center)
    rho, phi = cart2pol(x[:, 0], x[:, 1])
    result = np.swapaxes(np.array([rho, phi], dtype=np.float32), 0, 1)

    # normalize rho values to range[0-1]
    result[:, 0] = normalize(result[:, 0].reshape(1, -1), norm='max')
    return result


def get_pose():
    """
    Submit the pose detection request by calling the rest API.
    """
    files = {'file': ('image.jpg', open(
        'assets/image.jpg', 'rb'), 'images/jpeg')}
    result = requests.post(URL, files=files).json()
    img = cv2.imread('assets/image.jpg')[:, :, ::-1]
    return result, img


def get_pose_from_file(f, d):
    """
    Submit the pose detection request by calling the rest API with a file stored in assets/images.
    """
    filepath = 'assets/images/' + str(d) + '/' + str(f)

    with open(filepath, 'rb') as name:
        _, ext = str(f).split('.')
        frame = cv2.imread(filepath)[:, :, ::-1]
        files = {'file': (filepath, name, 'images/' + ext)}
        result = requests.post(URL, files=files).json()

    return result, frame


def draw_pose(preds, img):
    """
    Visualize the detected human poses on the image. The returned JSON result
    contains the pose lines for each person in the input image.
    """
    humans = preds['predictions']
    for human in humans:
        pose_lines = human['pose_lines']
        for i, _ in enumerate(pose_lines):
            line = pose_lines[i]['line']
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), COCO_COLORS[i], 3)


def convert_pose(coordinates, cartesian=True):
    """
    Convert poses then return the normalized coordinates
    """
    if not cartesian:
        coordinates = convert(coordinates)
    else:
        coordinates[:, 0] = normalize(
            coordinates[:, 0].reshape(1, -1), norm='max')
        coordinates[:, 1] = normalize(
            coordinates[:, 1].reshape(1, -1), norm='max')

    return coordinates


def get_labels():
    """
    Returns a complete list of labels as per the folders in assets/images.
    """
    return [name for name in os.listdir(os.getcwd() + '/assets/images') if name != 'test']
