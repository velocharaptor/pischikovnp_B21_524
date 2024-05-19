from PIL import Image
import numpy as np
import random
from glob import glob
import os

def binarization(old_image, threshold):
    new_image = np.zeros(shape=old_image.shape)
    new_image[old_image > threshold] = 255
    return new_image.astype(np.uint8)

def Prewitt_opertor(image) :
    height, width = image.shape[:2]
  
    G_x = np.array([[-1, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]])
    G_y = np.array([[-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1],
                    [-1, 0, 0, 0, 1]])
  
    G_x_res = np.zeros_like(image, dtype=np.float32)
    G_y_res = np.zeros_like(image, dtype=np.float32)

    for y in range(2, height - 2):
        for x in range(2, width - 2):
            window = image[y - 2:y + 3, x - 2:x + 3]
            G_x_res[y, x] = np.sum(G_x * window)
            G_y_res[y, x] = np.sum(G_y * window)

    G_res = np.sqrt(np.square(G_x_res) + np.square(G_y_res))#возведение в квадрат поэлементное, а не матричное
  
    G_res = ((G_res - np.min(G_res)) / (np.max(G_res) - np.min(G_res))) * 255

    return (G_x_res.astype(np.uint8),
            G_y_res.astype(np.uint8),
            G_res.astype(np.uint8))


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    names = ['4sem/results/4.2/output/semitone_eye.png']
    for input_path in names:
        input_img = Image.open(input_path)
        image = np.array(input_img)

        #Приютт оператор
        pruitt_x_image, pruitt_y_image, pruitt_image = Prewitt_opertor(image)
        binarized_100_image = binarization(pruitt_image, 100)

        new_img = Image.fromarray(pruitt_image)
        curr_opath = os.path.join(output_path, os.path.splitext("pruit_" + os.path.basename(input_path))[0])
        new_img.save(curr_opath + ".png")

        new_img = Image.fromarray(pruitt_x_image)
        curr_opath = os.path.join(output_path, os.path.splitext("pruit_x_" + os.path.basename(input_path))[0])
        new_img.save(curr_opath + ".png")

        new_img = Image.fromarray(pruitt_y_image)
        curr_opath = os.path.join(output_path, os.path.splitext("pruit_y_" + os.path.basename(input_path))[0])
        new_img.save(curr_opath + ".png")

        new_img = Image.fromarray(binarized_100_image)
        curr_opath = os.path.join(output_path, os.path.splitext("binarized_100_" + os.path.basename(input_path))[0])
        new_img.save(curr_opath + ".png")

if __name__ == "__main__":
    main()