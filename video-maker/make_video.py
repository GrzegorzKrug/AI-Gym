import cv2
import os
import re


def make_video(dir_path, file_list):
    first_pic_path = os.path.join(dir_path, file_list[0])
    frame = cv2.imread(first_pic_path)

    video_size = (frame.shape[1], frame.shape[0])
    movie_path = f'{dir_path}.mp4'
    if os.path.isfile(movie_path):
        print(f"Movie {dir_path} already exists!!")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(movie_path, fourcc, 60.0, video_size)
        for file in file_list:
            img_path = os.path.join(dir_path, file)
            # print(img_path)
            frame = cv2.imread(img_path)
            out.write(frame)
        out.release()
        print(f"Saved movie: {movie_path}")


found = os.listdir()
for _found in found:
    if not os.path.isdir(_found):
        continue

    name = _found
    files = os.listdir(_found)
    good_files = [file for file in files if file.endswith(
        'jpg') or file.endswith('png')]
    numerated_files = []
    for file in good_files:
        num = int(re.match(r"\d+", file)[0])
        numerated_files.append([num, file])

    numerated_files.sort(key=lambda x: x[0])  # Sort integers
    numerated_files = [file for number, file in numerated_files]

    make_video(name, numerated_files)
