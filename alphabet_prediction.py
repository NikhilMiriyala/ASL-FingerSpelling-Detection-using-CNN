import glob
import sys
import time
from pandas import DataFrame
from sklearn.metrics import classification_report
from skimage.feature import hog
import cv2
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
from tensorflow import keras
load_model = keras.models.load_model


def get_vector_data(files_list):
    model = HandShapeFeatureExtractor.get_instance()
    vectors = []
    video_names = []
    step = int(len(files_list) / 100)
    if step == 0:
        step = 1

    count = 0
    for video_frame in files_list:

        img = cv2.imread(video_frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = model.extract_feature(img)
        results = np.squeeze(results)
        predicted = np.where(results == max(results))[0][0]

        vectors.append(predicted)
        video_names.append(os.path.basename(video_frame))

        count += 1
        if count % step == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    return vectors


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    i = 0

    for id in id_to_labels:
        labels_to_id[id] = i
        i += 1

    return id_to_labels, labels_to_id


def train_and_predict():
    print("Training with the below frames extracted:\n")
    files = []
    train_frames = os.path.join('alphabet_hand_frames')

    path = os.path.join(train_frames, "*.png")
    frames = glob.glob(path)

    frames.sort()
    files = frames

    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)

    X_train = []
    Y_train = []
    for fileName in files:
        img = cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))
        # Hog Feature extraction
        _, img_arr = hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=True, multichannel=False)
        img_arr = img_arr.reshape(200, 200, 1)

        X_train.append(img_arr)
        print(fileName)
        label = fileName.split('\\')[-1].split('.')[0].split("_")[0]
        label_id = labels_to_id[label]
        Y_train.append(label_id)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    model = load_model('cnn_model.h5')
    model.fit(X_train, Y_train, epochs=5)

    files = []
    test_frames = os.path.join('test')

    path = os.path.join(test_frames, "*.png")
    frames = glob.glob(path)

    frames.sort()
    files = frames

    prediction_vector = get_vector_data(files)

    time.sleep(2)
    final_predictions = []

    for i in range(len(prediction_vector)):
        for ins in labels_to_id:
            if prediction_vector[i] == labels_to_id[ins]:
                final_predictions.append(ins)

    print(len(final_predictions))
    print("\n")
    print(final_predictions)
    print("\n")

    pred = final_predictions

    pred_array = []
    for i in range(26):
        pred_array.append([pred[i], id_to_labels[i % 26]])

    df = DataFrame(pred_array, columns=['pred', 'true'])
    print(classification_report(df.pred, df.true))
    df.to_csv(os.path.join(test_frames, 'results.csv'))


if __name__ == "__main__":
    train_and_predict()