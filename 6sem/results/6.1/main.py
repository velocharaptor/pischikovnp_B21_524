from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

def binarization(image, threshold):
    old_image = np.array(image)
    new_image = np.zeros(shape=old_image.shape)
    new_image[old_image > threshold] = 255
    return Image.fromarray(new_image.astype(np.uint8), 'L')

def main():
    font = ImageFont.truetype("6sem/results/6.1/input/Arial-Italic.ttf", 69)
    image = Image.new("L", (615, 80), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text="snow cat and bear", font=font, color="black")
    binarization(image, 50).save(f"6sem/results/6.1/output/sentence.png")

if __name__ == '__main__':
    main()