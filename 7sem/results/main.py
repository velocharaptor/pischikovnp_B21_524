import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
import csv
from math import sqrt

def binarization(image, threshold):
    old_image = np.array(image)
    new_image = np.zeros(shape=old_image.shape)
    new_image[old_image > threshold] = 255
    return Image.fromarray(new_image.astype(np.uint8), 'L')

def get_weight(img_px, width, height):
    size = width * height
    weight = 0
    for i in range(width):   
        for j in range(height):
            if img_px[i, j] == 0: 
                weight += 1
    rel_weight = weight / size 

    return weight, rel_weight

def get_avg(img_px, weight, width, height):
    x_avg, y_avg = 0, 0
    for i in range(width):   
        for j in range(height):
            if img_px[i, j] == 0: 
                x_avg += i   
                y_avg += j
    x_avg /= weight
    y_avg /= weight
    rel_x_avg = (x_avg - 1) / (width - 1)  
    rel_y_avg = (y_avg - 1) / (height - 1) 

    return (x_avg, y_avg), (rel_x_avg, rel_y_avg)

def get_inertia(img_px, x_avg, y_avg, width, height):
    inertia_x, inertia_y = 0, 0
    for i in range(width): 
        for j in range(height):
            if img_px[i, j] == 0: 
                inertia_x = (j - x_avg) ** 2
                inertia_y = (i - y_avg) ** 2
    rel_inertia_x = inertia_x / (width ** 2 * height ** 2)  
    rel_inertia_y = inertia_y / (width ** 2 * height ** 2)
    
    return (inertia_x, inertia_y), (rel_inertia_x, rel_inertia_y)

def create_features(image_array):
    img_px = np.zeros(shape=image_array.shape)
    img_px[image_array != 255] = 1

    width, height = image_array.shape[:2]
    size = width * height

    weight, rel_weight = get_weight(img_px, width, height)
    xy_avg, rel_xy_avg = get_avg(img_px, weight, width, height)
    inertia, rel_inertia = get_inertia(img_px, xy_avg[0], xy_avg[1], width, height)
    return {
            "Normalized Weight" : rel_weight,
            "Normalized Mass Center" : rel_xy_avg,
            "Normalized Inertia Moments" : rel_inertia,
        }

def load_features(path):
    with open(path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        result = {}
        for row in reader:
            result[row['Letter']] = {
                'Normalized Weight': float(row['Normalized Weight']),
                'Normalized Mass Center': tuple(map(float, row['Normalized Mass Center'][1:len(row['Normalized Mass Center'])-1].split(', '))),
                'Normalized Inertia Moments': tuple(map(float, row['Normalized Inertia Moments'][1:len(row['Normalized Inertia Moments'])-1].split(', ')))
            }

        return result

def feature_distance(features_1, features_2):
    return sqrt(
        (features_1['Normalized Weight'] - features_2['Normalized Weight'])**2 +
        (features_1['Normalized Mass Center'][0] - features_2['Normalized Mass Center'][0])**2 +
        (features_1['Normalized Mass Center'][1] - features_2['Normalized Mass Center'][1])**2 +
        (features_1['Normalized Inertia Moments'][0] - features_2['Normalized Inertia Moments'][0])**2 +
        (features_1['Normalized Inertia Moments'][1] - features_2['Normalized Inertia Moments'][1])**2
    )

def calculate_distance(features_global, features_local):
    result = {}
    for letter, features in features_global.items():
        result[letter] = feature_distance(features_local, features)

    _max = max(result.values())

    new_result = {}
    for letter, distance in result.items():
        new_result[letter] = (_max - distance) / _max 

    return new_result

def fix_color(img):  
    return np.asarray(np.asarray(img) < 1, dtype = np.int8)

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

def crop_segments(img):
    letters_list = []
    segments = get_segments(img)
    i = 0
    for segment in segments:
        box = (segment[0] + 1, 0, segment[1] - 1, img.height)
        res = img.crop(box)
        #res.save(f"7sem/results/output/letters/{i + 1}.png")
        # i+=1
        letters_list.append(res)
    return letters_list

def create_report(letters):
    with open("7sem/results/output/data.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = 
        ["Letter", "Weight", 
         "Normalized Weight", "Mass Center",
         "Normalized Mass Center", "Inertia Moments", 
         "Normalized Inertia Moments"]
         )
        writer.writeheader()
        symbol = ["s","n","o","w","c","a","t","a","n","d","b","e","a","r"]
        for i, letter in enumerate(letters):
            letter_arr = np.array(letter, dtype=np.uint8)

            features = create_features(letter_arr)
            features['Letter'] = symbol[i]

            writer.writerow(features)

def create_regocnition(path):
    img = Image.open("7sem/results/input/sentence_100.png").convert('L')
    letters_list = crop_segments(img)
    save_features = load_features(path)
    #create_report(letters_list)

    with open("7sem/results/output/output.txt", "a", encoding="utf-8") as file:
        for i, letter in enumerate(letters_list):
            letter_arr = np.array(letter, dtype=np.uint8)
            current_features = create_features(letter_arr)
            grades = calculate_distance(save_features, current_features)
            file.write(f"{i + 1}: {dict(sorted(grades.items(), key=lambda item: item[1], reverse=True))}\n")
            letter = max(grades, key=grades.get)
            print(letter, end=' ')
        file.write(f"\n")

def generate_text(text, size):
    font = ImageFont.truetype("6sem/results/6.1/input/Arial-Italic.ttf", size)
    image = Image.new("L", (1200, size), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text=text, font=font, color="black")
    binarization(image, 100).save(f"7sem/results/input/sentence_{size}.png")

def main():
    generate_text("snow cat and bear", 100)
    create_regocnition("5sem/results/2.26/output/data.csv")
        
if __name__ == '__main__':
    main()