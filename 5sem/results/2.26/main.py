import csv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

start_unicode = ord('a')
end_unicode = ord('z')

ENG_LETTER = [chr(code_point) for code_point in range(start_unicode, end_unicode + 1)]

# def get_weight(img_px, width, height):
#     size = width * height
#     weight = 0
#     for i in range(width):   
#         for j in range(height):
#             if img_px[i, j] == 0: 
#                 weight += 1
#     rel_weight = weight / size 

#     return weight, rel_weight

# def get_avg(img_px, weight, width, height):
#     x_avg, y_avg = 0, 0
#     for i in range(width):   
#         for j in range(height):
#             if img_px[i, j] == 0: 
#                 x_avg += i   
#                 y_avg += j
#     x_avg /= weight
#     y_avg /= weight
#     rel_x_avg = (x_avg - 1) / (width - 1)  
#     rel_y_avg = (y_avg - 1) / (height - 1) 

#     return (x_avg, y_avg), (rel_x_avg, rel_y_avg)

# def get_inertia(img_px, x_avg, y_avg, width, height):
#     inertia_x, inertia_y = 0, 0
#     for i in range(width): 
#         for j in range(height):
#             if img_px[i, j] == 0: 
#                 inertia_x = (j - x_avg) ** 2
#                 inertia_y = (i - y_avg) ** 2
#     rel_inertia_x = inertia_x / (width ** 2 * height ** 2)  
#     rel_inertia_y = inertia_y / (width ** 2 * height ** 2)
    
#     return (inertia_x, inertia_y), (rel_inertia_x, rel_inertia_y)

# def create_features(image_array):
#     img_px = np.zeros(shape=image_array.shape)
#     img_px[image_array != 255] = 1

#     width, height = image_array.shape[:2]
#     size = width * height

#     weight, rel_weight = get_weight(img_px, width, height)
#     xy_avg, rel_xy_avg = get_avg(img_px, weight, width, height)
#     inertia, rel_inertia = get_inertia(img_px, xy_avg[0], xy_avg[1], width, height)
#     return {
#             "Weight" : weight,
#             "Normalized Weight" : rel_weight,
#             "Mass Center" : xy_avg,
#             "Normalized Mass Center" : rel_xy_avg,
#             "Inertia Moments" : inertia,
#             "Normalized Inertia Moments" : rel_inertia,
#         }

def create_features(img):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != 255] = 1

    (h, w) = img_b.shape
    h_half, w_half = h // 2, w // 2
    quadrants = {
        'top_left': img_b[:h_half, :w_half],
        'top_right': img_b[:h_half, w_half:],
        'bottom_left': img_b[h_half:, :w_half],
        'bottom_right': img_b[h_half:, w_half:]
    }
    weights = {k: np.sum(v) for k, v in quadrants.items()}
    rel_weights = {k: v / (h_half * w_half) for k, v in weights.items()}

    total_pixels = np.sum(img_b) 
    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / total_pixels
    x_center_of_mass = np.sum(x_indices * img_b) / total_pixels
    center_of_mass = (x_center_of_mass, y_center_of_mass)

    normalized_center_of_mass = (x_center_of_mass / (w - 1), y_center_of_mass / (h - 1))

    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / total_pixels
    normalized_inertia_x = inertia_x / h ** 2
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / total_pixels
    normalized_inertia_y = inertia_y / w ** 2

    return {
        'weight': total_pixels,
        'weights': weights,
        'rel_weights': rel_weights,
        'center_of_mass': center_of_mass,
        'normalized_center_of_mass': normalized_center_of_mass,
        'inertia': (inertia_x, inertia_y),
        'normalized_inertia': (normalized_inertia_x, normalized_inertia_y)
    }

def create_report(letters):
    with open("5sem/results/2.26/output/csv/data.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = 
        ["letter", "weight", "weights" ,
         "rel_weights", "center_of_mass",
         "normalized_center_of_mass", "inertia", 
         "normalized_inertia"])
        writer.writeheader()

        for i in range(len(letters)):
            img = Image.open(f'5sem/results/1.26/font_Italic/{letters[i]}.png').convert('L')
            img_arr = np.array(img, dtype=np.uint8)
            features = create_features(img_arr)
            features['letter'] = letters[i]
            writer.writerow(features)

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

def add_profile(img, letter, type="x"):
    profiles = get_profiles(img)

    if type == "x" :
        plt.bar(x=profiles["x"]["x_range"], height=profiles["x"]["x"], width=0.85)
        plt.ylim(0, max(profiles["x"]["x"]))
        plt.xlim(0, max(profiles["x"]["x_range"]))
    else :
        plt.barh(y=profiles["y"]["y_range"], width=profiles["y"]["y"], height=0.85)
        plt.ylim(max(profiles["y"]["y_range"]), 0 )
        plt.xlim(0, max(profiles["y"]["y"]))

    plt.savefig(f"5sem/results/2.26/output/profiles_Unicode/{type}/{letter}.png")
    plt.clf()

def create_profiles(letters):
    for letter in letters:
        img = Image.open(f"5sem/results/1.26/font_UNICODE/{letter}.png").convert('L')
        img_arr = np.array(img)

        img_arr[img_arr == 0] = 1
        img_arr[img_arr == 255] = 0

        add_profile(img_arr, letter, "y")
        add_profile(img_arr, letter, "x")

def main():
    create_report(ENG_LETTER)
    #create_profiles(ENG_LETTER)

if __name__ == "__main__":
    main()