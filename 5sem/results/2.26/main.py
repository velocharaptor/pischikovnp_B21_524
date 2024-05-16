from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

def create_features(image_array):
    img_px = np.zeros(shape=image_array.shape)
    img_px[image_array != 255] = 1

    width, height = image_array.shape[:2]
    size = width * height

    weight, rel_weight,  = 0, 0
    x_avg, y_avg, rel_x_avg, rel_y_avg = 0, 0, 0, 0
    inertia_x, rel_inertia_x, inertia_y, rel_inertia_y = 0, 0, 0, 0

    for i in range(width):   
        for j in range(height):
            if img_px[i, j] == 0: 
                weight += 1
                x_avg += i   
                y_avg += j

    rel_weight = weight / size   

    x_avg /= weight
    y_avg /= weight
    rel_x_avg = (x_avg - 1) / (width - 1)  
    rel_y_avg = (y_avg - 1) / (height - 1)  

    for i in range(width): 
        for j in range(height):
            if img_px[i, j] == 0: 
                inertia_x = (j - x_avg) ** 2
                inertia_y = (i - y_avg) ** 2

    rel_inertia_x = inertia_x / (width ** 2 * height ** 2)  
    rel_inertia_y = inertia_y / (width ** 2 * height ** 2)

    return {
            "Weight" : weight,
            "Normalized Weight" : rel_weight,
            "Mass Center" : (x_avg, y_avg),
            "Normalized Mass Center" : (rel_x_avg, rel_y_avg),
            "Inertia Moments" : (inertia_x, inertia_y),
            "Normalized Inertia Moments" : (rel_inertia_x, rel_inertia_y),
        }


def create_report(letters):
    with open("5sem/results/2.26/output/data.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = 
        ["Letter", "Weight", 
         "Normalized Weight", "Mass Center",
         "Normalized Mass Center", "Inertia Moments", 
         "Normalized Inertia Moments"]
         )
        writer.writeheader()

        for letter in letters:
            img = Image.open(f"5sem/results/1.26/font/{letter}.png").convert('L')
            img_arr = np.array(img, dtype=np.uint8)

            features = create_features(img_arr)
            features['Letter'] = letter

            writer.writerow(features)

def get_profiles(img_arr):
    return {
        "x": {
            "profiles": np.sum(img_arr, axis=0),
            "range": np.arange(1, img_arr.shape[1] + 1)
        },
        "y": {
            "profiles": np.sum(img_arr, axis=1),
            "range": np.arange(1, img_arr.shape[0] + 1)
        }
    }

def add_profile(img, letter, type="x"):
    profiles = get_profiles(img)
    
    if type == 'x':
        plt.bar(x=profiles["x"]["range"], height=profiles["x"]["profiles"], width=0.85)
        plt.ylim(0, max(profiles["x"]["profiles"]))
        plt.xlim(0, max(profiles["x"]["range"]))
    else:
        plt.barh(y=profiles["y"]["range"], width=profiles["y"]["profiles"], height=0.85)
        plt.ylim(max(profiles["y"]["range"]), 0 )
        plt.xlim(0, max(profiles["y"]["profiles"]))

    plt.savefig(f"5sem/results/2.26/output/profiles/{type}/{letter}.png")
    plt.clf()

def create_profiles(letters):
    for letter in letters:
        img = Image.open(f"5sem/results/1.26/font/{letter}.png").convert('L')
        img_arr = np.array(img)

        img_arr[img_arr == 0] = 1
        img_arr[img_arr == 255] = 0

        add_profile(img_arr, letter, "y")
        add_profile(img_arr, letter, "x")

def main():
    letters = "abcdefghijklmnopqrstuvwxyz"
    create_report(letters)
    create_profiles(letters)

if __name__ == "__main__":
    main()