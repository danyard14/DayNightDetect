import numpy as np
import torch
import cv2
from pathlib import Path


def read_movie(source_path: str, destination):
    cap = cv2.VideoCapture(source_path)

    print(f'splitting {source_path}')
    # Read until video is completed
    frame_num = 0

    while True:
        # Capture frame-by-frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret is False:
            cap.release()
            break

        movie_name = str(Path(source_path).with_suffix('').name)
        write_path = f'{destination}/{movie_name}_frame_%04d.jpg' % frame_num
        cv2.imwrite(write_path, frame)
        print(f'wrote frame {frame_num} to path {write_path}')
        frame_num += 60


if __name__ == '__main__':
    read_movie(r'C:\Users\danya\Desktop\san_fran.mp4', r'C:\Users\danya\Desktop\day_night_detection')
