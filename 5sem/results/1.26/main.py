from PIL import Image, ImageDraw, ImageFont
import numpy as np

start_unicode = ord('a')
end_unicode = ord('z')

ENG_LETTER = [chr(code_point) for code_point in range(start_unicode, end_unicode + 1)]

def binarization(image, threshold):
    old_image = np.array(image)
    new_image = np.zeros(shape=old_image.shape)
    new_image[old_image > threshold] = 255
    return Image.fromarray(new_image.astype(np.uint8), 'L')

def main(letters):
    font = ImageFont.truetype("5sem/results/1.26/font_Italic/Arial-Italic.ttf", 52)

    for i in range(len(letters)):
        letter = letters[i]
        _, _, width, height = font.getbbox(letter)

        image = Image.new("L", (width, height), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), letter, font=font, color="black")

        binarization(image, 50).save(f"5sem/results/1.26/font_Italic/{letter}.png")

if __name__ == '__main__':
    main(ENG_LETTER)