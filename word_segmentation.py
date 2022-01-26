import os
import glob
import cv2
import word_wrist_extraction

def extract_cropped_frames():
    output_input = "temp_input"
    output_output = "temp_output"

    files = os.scandir("word_frames")
    for each in files:
        if not os.path.exists(output_input):
            os.mkdir(output_input)
        # print(each)
        if each.name == '.DS_Store':
            continue
        img_files = glob.glob(os.path.join(each, "*.jpg"))
        # print(img_files)
        for each_img in img_files:
            input_img = cv2.imread(each_img)
            cv2.imwrite(output_input + "\\" + each_img.split("\\")[-1], input_img)
        file_names = os.scandir(output_input)

        print("Extracting cropped images for: " + each.name)

        word_wrist_extraction.main()

        img_output_files = glob.glob(os.path.join(output_output, "*.jpg"))
        final_path = os.path.join("word_hand_frames", each.name)
        if not os.path.exists(final_path):
            os.mkdir(final_path)
        for each_output_img in img_output_files:
            out_put_temp = cv2.imread(each_output_img)
            cv2.imwrite(final_path + "/" + each_output_img.split("\\")[-1], out_put_temp)

        if os.path.exists(output_input):
            files_names_to_remove = glob.glob(os.path.join(output_input, "*.jpg"))
            for each___ in files_names_to_remove:
                os.remove(each___)

        if os.path.exists(output_output):
            files_names_to_remove = glob.glob(os.path.join(output_output, "*.jpg"))
            for each___ in files_names_to_remove:
                os.remove(each___)


if __name__ == "__main__":
    extract_cropped_frames()