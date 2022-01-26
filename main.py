import alphabet_frame_extraction as afe
import alphabet_wrist_extraction as awe
import alphabet_prediction as ap
import word_frames_extractor as wfe
import word_segmentation as ws
import word_prediction as wp


if __name__=='__main__':
    while True:
        print("Available operation to perform:")
        print("1: Predict ASL Alphabets")
        print("2: Predict ASL Words")
        print("3: Quit")
        print("\n")

        choice = input("Choose an option: ")

        if choice == '1':
            try:
                afe.run_alphabet_frame_extractor()
                awe.extract_cropped_frames()
                ap.train_and_predict()
            except:
                print("Exception occured while predicting ASL alphabets")

        elif choice == '2':
            try:
                wfe.word_frame_extractor()
                ws.extract_cropped_frames()
                wp.predict()
            except:
                print("Exception occured while predicting ASL words")

        elif choice == '3':
            print("Bye !")
            break

        else:
            print("Choose a valid option")