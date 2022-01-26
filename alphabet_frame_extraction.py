import cv2
import os

def frameExtractor(videopath, frames_path):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2)
    cap.set(1,frame_no)
    ret,frame=cap.read()
    name = videopath.split('/')[-1].split('.')[0]
    path = f"{frames_path}/{name}.png"
    cv2.imwrite(path, frame)

    # Resizing the frame for posenet efficient extraction
    src = frame

    width = 1280
    height = 720

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    cv2.imwrite(path, output)


def run_alphabet_frame_extractor():
    videopath = "alphabet_videos"
    frames_path = "alphabet_video_frames"

    file_name_list = []
    new_file_names = []
    for entry in os.scandir(videopath):
        if entry.path.endswith('.mp4') and entry.is_file():
            filename = entry.name
            file_name_list.append(filename)

    for name in sorted(file_name_list):
        print("Extracting frames for: " + name)
        frameExtractor(f'{videopath}/{name}', frames_path)

if __name__ == "__main__":
    run_alphabet_frame_extractor()