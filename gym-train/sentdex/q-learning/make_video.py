import cv2

RUN_NUM = 20


def make_video(num):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'Cart-{num}.mp4', fourcc, 60.0, (1600, 900))

    for i in range(0, 50000, 10):
        img_path = f"qtables_{num}_charts/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video(RUN_NUM)



