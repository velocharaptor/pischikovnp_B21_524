from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np

def main():
    letters = "абвгдеёжзийклсмнопрстуфхцчшщъыьэюя"
    font = ImageFont.truetype("5sem/results/1.26/font/Arial-Italic.ttf", 69)

    for letter in letters:
        _, _, width, height = font.getbbox(letter)

        image = Image.new("L", (width, height), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), letter, font=font, color="black")
        image.save(f"5sem/results/1.26/font/{letter}.png")

if __name__ == '__main__':
    main()