from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import math

def fetch(segment, img, i):
    box1 = (segment[0] + 1, 0, segment[0]+17, img.height)
    box2 = (segment[0] +18, 0, segment[1]-1, img.height)
    res1 = img.crop(box1)
    res2 = img.crop(box2)
    res1.save(f"6sem/results/6.2/output/letters/{i + 1}.png")
    res2.save(f"6sem/results/6.2/output/letters/{i + 2}.png")

def get_profiles(img):
    return {
        "x": {
            "profiles": np.sum(img, axis=0),
            "range": np.arange(1, img.shape[1] + 1).astype(int)
        },
        "y": {
            "profiles": np.sum(img, axis=1),
            "range": np.arange(1, img.shape[0] + 1).astype(int)
        }
    }

def add_profile(img, type):
    profiles = get_profiles(img)
    plt.figure(figsize=(12,2))

    if type == "x" :
        plt.bar(x=profiles["x"]["range"], height=profiles["x"]["profiles"], width=0.85)
        plt.ylim(0, max(profiles["x"]["profiles"]))
        plt.xlim(0, max(profiles["x"]["range"]))
    else :
        plt.barh(y=profiles["y"]["range"], width=profiles["y"]["profiles"], height=0.85)
        plt.ylim(max(profiles["y"]["range"]), 0 )
        plt.xlim(0, max(profiles["y"]["profiles"]))

    plt.savefig(f"6sem/results/6.2/output/profile/{type}/profile_{type}.png")
    plt.clf()

def create_profiles(img):
    img_arr = np.array(img)
    img_arr[img_arr <= 75] = 1
    img_arr[img_arr > 75] = 0
    add_profile(img_arr, "x")
    add_profile(img_arr, "y")

def fix_color(img):  #преобразование в двумерный массив, где 1 - черный символ а 0 белый фон
    return np.asarray(np.asarray(img) < 1, dtype = np.int0)

def get_segments(img):
    img_arr_for_calculations = fix_color(img)
    x_profiles = np.sum(img_arr_for_calculations, axis=0) 
    lst = [] #индексы где нет черных пикселей
    new_lst = []  #начало и конец сегментов
    for i in range(len(x_profiles)):  #заполняем где нет черных 
        if x_profiles[i] == 0:
            lst.append(i)
    lst.append(img.width)  #добавляем ширину изображения в конец списка для обозначения конца последнего сегмента

    #пары индексов, обозначающих начало и конец
    for i in range(len(lst)-1):
        if lst[i] + 1 != lst[i+1]:
            new_lst.append(lst[i])
            new_lst.append(lst[i+1])
    new_lst.append(img.width-1)
    new_lst = sorted(list(set(new_lst))) #убирем дубликаты и отсортируем список
    
    
    segments = []
    for i in range(0, len(new_lst)-1, 2):
        segments.append((new_lst[i], new_lst[i+1]))
    return segments

# def draw(image, segments):
#     left_color = (0, 0, 255)  #зеленый для левой
#     right_color = (255, 0, 0) #фиолетовый цвет для правой границы
#     result = image.copy().convert('RGB')
#     result_draw = ImageDraw.Draw(im=result)
#     for segment in segments:
#         # Создаем многоугольник из точек, образующих прямоугольник
#         polygon1 = [(segment[0], 0), (segment[0], result.height)]
#         polygon2 = [(segment[1], result.height), (segment[1], 0)]
#         # Поворачиваем многоугольник на 2 градуса вправо
#         rotated_polygon1 = []
#         rotated_polygon2 = []
#         for point in polygon1:
#             rotated_polygon1.append(rotate_point(point, 2))
#         for point in polygon2:
#             rotated_polygon2.append(rotate_point(point, 2))
#         # Рисуем повернутый прямоугольник
#         result_draw.polygon(xy=rotated_polygon1, fill=left_color)
#         result_draw.polygon(xy=rotated_polygon2, fill=right_color)
#     return result

# # Функция для поворота точки вокруг начала координат
# def rotate_point(point, angle):
#     x, y = point
#     angle = math.radians(angle)
#     return (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))

# def create_segments(img):
#     segments = get_segments(img)
#     result = draw(img, segments)
#     result.save("6sem/results/6.2/output/result_sentence.png")

def crop_segments():
    img = Image.open("6sem/results/6.1/output/sentence.png").convert('L')
    segments = get_segments(img)
    i = 0
    for segment in segments:
        if(segment[0] == 169): 
            fetch(segment, img, i)
            i+=2
        else: 
            box = (segment[0] + 1, 0, segment[1] - 1, img.height)
            res = img.crop(box)
            res.save(f"6sem/results/6.2/output/letters/{i + 1}.png")
            i+=1

def main():
    test_img = Image.open("6sem/results/6.1/output/sentence.png")
    
    crop_segments()
    create_profiles(test_img)

if __name__ == '__main__':
    main()