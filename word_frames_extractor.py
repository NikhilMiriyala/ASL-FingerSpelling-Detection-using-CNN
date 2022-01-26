import cv2
import os

def word_frame_extractor():
    print("Running Words Frame Extractor")
    word_files_path = "word_videos"
    file_name_list = []
    new_file_names = []
    for entry in os.scandir(word_files_path):
        if entry.path.endswith('.mp4') and entry.is_file():
            filename = entry.name
            file_name_list.append(filename)

    for name in sorted(file_name_list):
        videoname = name
        print("Extracting frames for: " + videoname)
        cap = cv2.VideoCapture(f"./word_videos/{videoname}")

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total number of frames for " + videoname + " = " + str(totalFrames))
        pass
        size = totalFrames / 3
        frames = []
        frames.append(int(0.5 * size))
        frames.append(int(1.5 * size))
        frames.append(int(2.5 * size))

        os.mkdir(f"./word_frames/{videoname.split('.')[0]}")

        index = 1

        for i in frames:
            cap.set(1, i)
            ret, frame = cap.read()  # Read the frame
            name = videoname.split('.')[0]
            path = f"./word_frames/{name}/" + str(index) + '.jpg'
            cv2.imwrite(path, frame)

            # Resizing the image to the required resolution for efficient posenet cropping feature
            src = frame

            width = 1280
            height = 720

            # dsize
            dsize = (width, height)

            # resize image
            output = cv2.resize(src, dsize)

            cv2.imwrite(path, output)

            index += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    word_frame_extractor()