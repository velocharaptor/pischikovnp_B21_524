from PIL import Image, ImageDraw, ImageFont
import numpy as np

def binarization(img, threshold=75):
    old_image = np.array(img)
    new_image = np.zeros(shape=old_image.shape)
    new_image[old_image > threshold] = 255
    return new_image.astype(np.uint8)

def main():
    phrase = "show me your power"
    space_len = 5
    phrase_width = sum(ImageFont.truetype("5sem/results/1.26/font_Italic/Arial-Italic.ttf", 52).getsize(char)[0] for char in phrase) + space_len * (len(phrase) - 1)

    height = max(ImageFont.truetype("5sem/results/1.26/font_Italic/Arial-Italic.ttf", 52).getsize(char)[1] for char in phrase)

    img = Image.new("L", (phrase_width, height), color="white")
    draw = ImageDraw.Draw(img)

    current_x = 0
    for letter in phrase:
        width, letter_height = ImageFont.truetype("5sem/results/1.26/font_Italic/Arial-Italic.ttf", 52).getsize(letter)
        draw.text((current_x, height - letter_height), letter, "black", font=ImageFont.truetype("5sem/results/1.26/font_Italic/Arial-Italic.ttf", 52))
        current_x += width + space_len

    img = Image.fromarray(binarization(img))
    img.save("6sem/results/6.1/output/text_power.bmp")

if __name__ == '__main__':
    main()