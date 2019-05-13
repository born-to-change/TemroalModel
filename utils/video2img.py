import mmcv
import os
import sys
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing

def is_video(x):
    return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')

def video2imgage(input_path, output_path, fps=None, isCrop=False):

    if not os.path.isdir(input_path):
        if is_video(input_path):
            video = mmcv.VideoReader(input_path)
            video.cvt2frames(output_path)
        else:
            sys.stderr.write("Input directory '%s' does not exist!\n" % input_path)
            sys.exit(1)

    vis_existing = [x.split('.')[0] for x in os.listdir(input_path)]
    video_filenames = [x for x in sorted(os.listdir(input_path))
                       if is_video(x) and os.path.splitext(x)[0] in vis_existing]
    if fps is not None:
        for video_file in tqdm(video_filenames):
            #  Tqdm a fast Python progress bar tqdm(iterator)
            output_file_path = output_path + '/' + video_file.split('.')[0]
            if os.path.exists(output_file_path):
                pass
            else:
                os.makedirs(output_file_path)
            try:
                clip = VideoFileClip(os.path.join(input_path, video_file))
            except Exception as e:
                sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_file)
                sys.stderr.write("Exception: {}\n".format(e))
                continue

            video_fps = int(np.round(clip.fps))
            if video_fps < fps:
                sys.stderr.write("Unable transfor video to img when your set fps is larger than video ")
            else:
                for idx, x in enumerate(clip.iter_frames()):
                    if (idx % video_fps) % (video_fps // fps) == 0:
                        cv2.imwrite(output_file_path + '/' + str(idx + 1) + '.jpg', x)

    else:
        pool = multiprocessing.Pool(processes=4)
        for video_file in video_filenames:
            output_file_path = output_path + '/' + video_file.split('.')[0]
            if os.path.exists(output_file_path):
                pass
            else:
                os.makedirs(output_file_path)

            pool.apply_async(apply_cv, (input_path, video_file, output_file_path,))
        pool.close()
        pool.join()


def apply_cv(input_path, video_file, output_file_path):
    video = mmcv.VideoReader(os.path.join(input_path, video_file))
    video.cvt2frames(output_file_path)


    # Go through each video and extract features

