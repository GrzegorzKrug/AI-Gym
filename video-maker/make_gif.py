from PIL import Image

import os
import re


def make_gif(name, file_list):
    framerate = 30

    duration = 1 / framerate

    frames = []
    for file_path in file_list:
        picture = Image.open(os.path.join(name, file_path))

        pil_im = picture.copy()
        picture.close()
        frames.append(pil_im)

    if len(frames) > 2:
        frames[0].save(f"{name}.gif", format="GIF", append_images=frames[1:],
                       save_all=True, optimize=False, duration=duration, loop=0)
        print(f"Saved gif {name}")
    else:
        print(f"Not saved gif {name}")


found = os.listdir()
for _found in found:
    if not os.path.isdir(_found):
        continue

    if os.path.isfile(f"{_found}.gif"):
        print(f"This gif exists already '{name}.gif'!")
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

    make_gif(name, numerated_files)

print("Finished...")
