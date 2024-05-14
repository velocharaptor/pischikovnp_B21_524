from PIL import Image
import numpy as np
import random
from glob import glob
import os

def semitone(input_path, output_path):
    input_img = Image.open(input_path)
    input_array = np.array(input_img)
    height, width = input_array.shape[:2]
    output_array = np.zeros((height, width, input_array.shape[2]), dtype=input_array.dtype)
    output_array = (0.3 * input_array[:, :, 0] + 0.59 * input_array[:, :, 1] + 0.11 * input_array[:, :, 2]).astype(np.uint8)
    new_img = Image.fromarray(output_array)
    new_img.save(output_path + ".png")
    print(f"Wrote semitone in {output_path}.png")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    names = ['4sem/result/4.2/input/eye.png']
    for input_path in names:
        curr_opath = os.path.join(output_path, os.path.splitext("semitone_" + os.path.basename(input_path))[0])
        semitone(input_path, curr_opath)

if __name__ == "__main__":
    main()

