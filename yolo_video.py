import os
import sys
import re
import pathlib
import argparse
import numpy as np
import cv2
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo, images_dire):
    images_dirpath = os.path.abspath(images_dire)
    #while True:
    filepath_list = [
        os.path.join(os.path.abspath(_dirpath), _filename)
        for _dirpath, _dirnames, _filenames in os.walk(images_dirpath)
        for _filename in _filenames
        if re.search(pattern=r'^(?!.*^\.).*\.jpg',
                     string=_filename,
                     flags=re.IGNORECASE) is not None
    ]  # .(ピリオド)で始まるファイルを無視
    for filepath in filepath_list:
        #img = input('Input image filename:')
        try:
            image = Image.open(filepath)
        except:
            print('Open Error! Try again!')
            continue
        else:
            image = np.array(image)
            assert len(image.shape) == 3 or len(image.shape) == 2, "Incorrect Image Format : {}".format(image.shape)
            if len(image.shape) == 3:
                pass
            elif len(image.shape) == 2:
                # グレイスケールはRGB全部同じ値にすれば良いのでBGRでもRGBでも同じ。
                image = cv2.cvtColor(src=image, code=cv2.COLOR_GRAY2BGR)
                image = Image.fromarray(np.uint8(image))

            r_image = yolo.detect_image(image)
            #r_image.show()
            save_file_Path = pathlib.Path("./out") / filepath[len(images_dirpath)+1:]
            save_file_Path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            r_image.save(str(save_file_Path), "JPEG")
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    parser.add_argument(
        '--image_dir', type=str,
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)), images_dire=FLAGS.image_dir)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
