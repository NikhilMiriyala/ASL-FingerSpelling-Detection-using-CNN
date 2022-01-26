import tensorflow.compat.v1 as tf
import cv2
import argparse
import os

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./alphabet_video_frames')
parser.add_argument('--output_dir', type=str, default='./alphabet_hand_frames')
args = parser.parse_args()


def extract_cropped_frames():
    wristCoords = []
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if not args.notxt:
                print()
                print("Extracting Cropped image for : %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break

                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        if posenet.PART_NAMES[ki] == 'rightWrist':
                            wristCoords.append(c.tolist())
                            img = cv2.imread(f, 1)

                            startY = int(c.tolist()[1] - 325)
                            endY = int(c.tolist()[1])
                            startX = int(c.tolist()[0] - 100)
                            endX = int(c.tolist()[0] + 200)

                            if (startY < 0):
                                startY = 0
                            if (endY < 0):
                                endY = 0
                            if (startX < 0):
                                startX = 0
                            if (endX < 0):
                                endX = 0

                            crop_img = img[startY:endY, startX:endX]
                            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), crop_img)


if __name__ == "__main__":
    extract_cropped_frames()