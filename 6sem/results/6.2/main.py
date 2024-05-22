from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt

def get_profiles(img):
    return {
        "x": {
            "x": np.sum(img, axis=0),
            "x_range": np.arange(1, img.shape[1] + 1).astype(int)
        },
        "y": {
            "y": np.sum(img, axis=1),
            "y_range": np.arange(1, img.shape[0] + 1).astype(int)
        }
    }

def add_profile(img, type):
    profiles = get_profiles(img)
    plt.figure(figsize=(12,2))

    if type == "x" :
        plt.bar(x=profiles["x"]["x_range"], height=profiles["x"]["x"], width=0.85)
        plt.ylim(0, max(profiles["x"]["x"]))
        plt.xlim(0, max(profiles["x"]["x_range"]))
    else :
        plt.barh(y=profiles["y"]["y_range"], width=profiles["y"]["y"], height=0.85)
        plt.ylim(max(profiles["y"]["y_range"]), 0 )
        plt.xlim(0, max(profiles["y"]["y"]))

    plt.savefig(f"6sem/results/6.2/output/profile/unicode/{type}/profile_{type}.png")
    plt.clf()

def create_profiles(img):
    img_arr = np.array(img)
    img_arr[img_arr <= 75] = 1
    img_arr[img_arr > 75] = 0
    add_profile(img_arr, "x")
    add_profile(img_arr, "y")

def get_segments(img):
    profile = np.sum(img == 0, axis=0)

    in_letter = False
    letter_segment = []

    for i in range(len(profile)):
        if profile[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_segment.append((start - 1, end))

    if in_letter:
        letter_segment.append((start, len(profile)))

    return letter_segment

def crop_segments(img, segments):
    image = Image.fromarray(img)
    i = 0
    for start, end in segments:
        left, right = start, end
        top, bottom = 0, img.shape[0]
        box = (left + 1, top, right, bottom)
        res = image.crop(box)
        res.save(f"6sem/results/6.2/output/letters/unicode/{i + 1}.png")
        i+=1

def main():
    test_img = np.array(Image.open("6sem/results/6.1/output/text_unicode.bmp"))
    segments = get_segments(test_img)
    crop_segments(test_img, segments)
    create_profiles(test_img)

if __name__ == '__main__':
    main()