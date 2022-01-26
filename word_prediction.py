import glob
import sys
import cv2
import numpy as np
import os
import time
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
from tensorflow import keras
load_model = keras.models.load_model


def get_inference_vector_one_frame_alphabet(files_list):
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

def predict():
    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)

    total_word_count = 0
    predicted_word_count = 0
    total_alphabet_count = 0
    predicted_alphabet_count = 0

    word_frames_path = "posenet/word_hand_frames"
    word_folders = os.scandir(word_frames_path)

    for each_folder in word_folders:

        if each_folder.name == '.DS_Store':
            continue

        word = each_folder.name

        files = []
        test_frames = os.path.join(f'posenet/word_hand_frames/{each_folder.name}')

        path = os.path.join(test_frames, "*.jpg")
        frames = glob.glob(path)
        # sort image frames
        frames.sort()
        files = frames

        prediction_vector = get_inference_vector_one_frame_alphabet(files)

        time.sleep(2)
        final_predictions = []

        for i in range(len(prediction_vector)):
            for ins in labels_to_id:
                if prediction_vector[i] == labels_to_id[ins]:
                    final_predictions.append(ins)

        print()

        final_word = final_predictions[0] + final_predictions[1] + final_predictions[2]

        if final_word == word:
            predicted_word_count += 1

        if final_word[0] == word[0]:
            predicted_alphabet_count += 1

        if final_word[1] == word[1]:
            predicted_alphabet_count += 1

        if final_word[2] == word[2]:
            predicted_alphabet_count += 1

        print("Prediction Obtained for " + word + " = " + final_word)

        total_word_count += 1
        total_alphabet_count += 3

    word_accuracy = (predicted_word_count * 100) / total_word_count
    alphabet_accuracy = (predicted_alphabet_count * 100) / total_alphabet_count

    print()
    print("Total Word Count = " + str(total_word_count))
    print("Predicted Word Count = " + str(predicted_word_count))
    print("Accuracy in terms of Words = " + str(word_accuracy) + " %")
    print("Total Alphabet Count = " + str(total_alphabet_count))
    print("Predicted Alphabet Count = " + str(predicted_alphabet_count))
    print(f"Accuracy in terms of Alphabet = {alphabet_accuracy:.2f}" + " %")


if __name__ == "__main__":
    predict()
