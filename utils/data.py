import os
import traceback
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.helpers import convert_pose, get_pose_from_file


def process_images(cartesian=False):
    """
    Extracts features and labels from assets/images directory
    """
    features = []
    labels = []
    lines = []

    for x in os.walk(str(os.getcwd())+'/assets/images'):
        for d in x[1]:
            if d != 'test':
                print('processing images in {}'.format(d))
                for f in os.listdir(str(os.getcwd())+'/assets/images/'+d):
                    try:
                        preds, _ = get_pose_from_file(f, d)
                        if preds['predictions']:
                            for i in range(len(preds['predictions'])):
                                pose_lines = preds['predictions'][i]['pose_lines']
                                body_parts = preds['predictions'][i]['body_parts']
                                coordinates = np.array([[d['x'], d['y']] for d in body_parts], dtype=np.float32)
                                p = convert_pose(coordinates, cartesian=cartesian)

                                if p != []:
                                    p = fill_empty_vector(p)
                                    features.append(p)
                                    labels.append(d)
                                    lines.append(pose_lines)

                    except Exception:
                        print("Something went wrong")
                        traceback.print_exc()
                        break

    return np.array(features), labels, np.array(lines)


def fill_empty_vector(feature):
    """
    Fill empty values of a vector with zeros.
    """
    return np.concatenate([feature, np.zeros([19-feature.shape[0], 2])])


def augment_data(features, labels, data_size=15, noise_amount=0.03):
    """
    Augment dataset to have an equal number of features per class with specified noise applied to phi and rho values.
    """
    orig_labels_count = len(labels)
    label_frequency = np.unique(labels, return_counts=True)
    while sum(label_frequency[1]) < data_size * len(label_frequency[0]):
        for i in range(orig_labels_count):
            if label_frequency[1][np.where(label_frequency[0] == labels[i])] < data_size:
                feature = features[i]
                new_data = np.zeros((feature.shape), dtype=np.float32)

                # add noise to each body part, making sure to add the same noise to corresponding joints
                for k, part in enumerate(feature):
                    noise = np.random.normal(0, noise_amount, 2)
                    new_0 = part[0] + noise[0]
                    new_1 = part[1] + noise[1]
                    new_data[k] = [new_0, new_1]

                features = np.concatenate(
                    (features, np.expand_dims(new_data, axis=0)))
                labels.append(labels[i])

            label_frequency = np.unique(labels, return_counts=True)

    return features, labels


def get_ideal_pose(features, labels, name_map):
    """
    Gets an ideal pose for each class.
    """
    ideal_poses = {}
    for lab, name in name_map.items():
        feature_list = np.array(
            [feats for i, feats in enumerate(features) if labels[i] == lab])
        ideal_poses[name] = np.average(feature_list, axis=0)

    return ideal_poses


def flatten_dataset(features):
    """
    Flattens a multidimensional dataset down to 1 dimension to feed into SVM.
    """
    nsamples, nx, ny = features.shape
    reshaped_data = features.reshape((nsamples, nx*ny))
    return reshaped_data


def encode_labels(labels):
    """
    Integer-encode labels.
    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    name_map = {key: val for key, val in zip(integer_encoded, labels)}
    return np.array(integer_encoded), name_map
