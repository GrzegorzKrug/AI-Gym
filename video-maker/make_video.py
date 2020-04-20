import cv2
import os
import re

VID_SIZE = (1600, 900)


def make_video(dir_path, file_list, video_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{dir_path}.mp4', fourcc, 60.0, video_size)

    for file in file_list:
        img_path = os.path.join(dir_path, file)
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()


found = os.listdir()
for _found in found:
    if not os.path.isdir(_found):
        continue

    name = _found
    files = os.listdir(_found)
    good_files = [file for file in files if file.endswith('jpg') or file.endswith('png')]
    numerated_files = []
    for file in good_files:
        num = int(re.match(r"\d+", file)[0])
        numerated_files.append([num, file])

    numerated_files.sort(key=lambda x: x[0])  # Sort integers
    numerated_files = [file for number, file in numerated_files]

    make_video(name, numerated_files, VID_SIZE)





