# source: https://code.oursky.com/tensorflow-svm-image-classifications-engine/
import pickle

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from utils.data import process_images, augment_data, flatten_dataset, encode_labels, get_ideal_pose


def train_svm_classifier(features, labels, model_output_path):
    """
    Training SVM on the processed/augmented/flattened data.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2)
    print('x_train shape: {}\ty_train shape: {}'.format(
        x_train.shape, y_train.shape))
    param = [
        {
            "kernel": ["poly"],
            "degree":[4, 5, 6, 7, 8]
        }
    ]

    # request probability estimation
    svm_model = SVC(max_iter=10000, gamma='scale', probability=True)
    clf = GridSearchCV(svm_model, param, cv=5, n_jobs=2, verbose=3)
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)

    labels = sorted(list(set(labels)))

    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print('\nClassification report:')
    print(classification_report(y_test, y_predict))

    # save the model to disk
    with open(model_output_path, 'wb') as fid:
        pickle.dump(clf, fid)

    return clf, svm_model


if __name__ == '__main__':

    data_size, noise = 100, 0.03

    features, labels, _ = process_images()
    features_aug, labels_aug = augment_data(features, labels, data_size=data_size, noise_amount=noise)
    print(features_aug.shape)
    flat_features = flatten_dataset(features_aug)
    labels_encoded, name_map = encode_labels(labels_aug)
    ideal_poses = get_ideal_pose(features_aug, labels_encoded, name_map)

    # save ideal poses for comparison during inference
    with open('assets/ideal_poses.pkl', 'wb') as fil:
        pickle.dump(ideal_poses, fil, pickle.HIGHEST_PROTOCOL)

    # save class name mappings for usage during inference
    with open('assets/classes.pkl', 'wb') as fil:
        pickle.dump(name_map, fil, pickle.HIGHEST_PROTOCOL)

    classifier, _ = train_svm_classifier(flat_features, labels_encoded, 'assets/classifier.pkl')
