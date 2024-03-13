import numpy as np
from PIL import Image as pim
from glob import glob
import os

def create_integral_image(image):
    height = image.shape[0]
    width = image.shape[1]
    integral_image = np.zeros((height+1, width+1), dtype=np.uint32)

    for y in range(1, height):
        for x in range(1,width):
            top_sum = integral_image[y-1, x] if y > 0 else 0
            left_sum = integral_image[y, x-1] if x > 0 else 0
            top_left_sum = integral_image[y-1, x-1] if y > 0 and x > 0 else 0

            integral_image[y, x] = image[y, x] + top_sum + left_sum - top_left_sum

    return integral_image

def bradley_thresholding(image, window_size, threshold=0.15):
    height = image.shape[0]
    width = image.shape[1]
    integral_image = create_integral_image(image)
    #integral_image = np.zeros((height+1, width+1), dtype=np.int32)
    #integral_image[1:,1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    threshold_image = np.zeros_like(image)

    half_window = window_size // 2
    for y in range(height):
        for x in range(width):
            top = max(0, y - half_window)
            bottom = min(height, y + half_window +1)
            left = max(0, x - half_window)
            right = min(width, x + half_window +1)
            #рассчет среднего значения пикселя с помощью интг окна по апертуре
            window_sum = integral_image[bottom, right] - integral_image[bottom, left] - \
                         integral_image[top, right] + integral_image[top, left]

            pixel_value = image[y, x]
            threshold_value = pixel_value * window_size * window_size #count
            #средняя яркость в окне * константу
            if (window_sum * (1 - threshold)> threshold_value):
              threshold_image[y, x] = 0
            else: threshold_image[y, x] = 255

    return threshold_image

def main():
    names = ['../1/output/semitone_84_3.bmp', '../1/output/semitone_im1.bmp', '../1/output/semitone_198_115.bmp']

    for name in names:
        img_src = pim.open(name)
        image = np.array(img_src)
        output_semitone = pim.fromarray(bradley_thresholding(image, window_size=int(image.shape[0]/15), threshold=0.19))
        s = os.path.splitext(os.path.basename(name))[0]
        output_semitone.save(f"output/res_{s}.bmp", bitmap_format="bmp")

if __name__ == "__main__":
    main()