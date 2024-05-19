from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
import math

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

    plt.savefig(f"6sem/results/6.2/output/profile/{type}/profile_{type}.png")
    plt.clf()

def create_profiles(img):
    img_arr = np.array(img)
    img_arr[img_arr <= 75] = 1
    img_arr[img_arr > 75] = 0
    add_profile(img_arr, "x")
    add_profile(img_arr, "y")

def fix_color(img):  
    return np.asarray(np.asarray(img) < 1, dtype = np.int0)

def get_segments(img):
    img_arr_for_calculations = fix_color(img)
    x_profiles = np.sum(img_arr_for_calculations, axis=0) 
    lst = [] 
    new_lst = []  
    for i in range(len(x_profiles)):   
        if x_profiles[i] == 0:
            lst.append(i)
    lst.append(img.width)  

    for i in range(len(lst)-1):
        if lst[i] + 1 != lst[i+1]:
            new_lst.append(lst[i])
            new_lst.append(lst[i+1])
    new_lst.append(img.width-1)
    new_lst = sorted(list(set(new_lst))) 

    segments = []
    for i in range(0, len(new_lst)-1, 2):
        segments.append((new_lst[i], new_lst[i+1]))
    return segments

def crop_segments():
    img = Image.open("6sem/results/6.1/output/sentence.png").convert('L')
    segments = get_segments(img)
    i = 0
    for segment in segments:
        box = (segment[0] + 1, 0, segment[1] - 1, img.height)
        res = img.crop(box)
        res.save(f"6sem/results/6.2/output/letters/{i + 1}.png")
        i+=1

def main():
    #сегменты вырезаются, не обводятся, нужно добавить удаление белых пиксилей
    test_img = Image.open("6sem/results/6.1/output/sentence.png")
    crop_segments()
    create_profiles(test_img)

if __name__ == '__main__':
    main()