from PIL import Image
import csv
import math
import numpy as np

start_unicode = ord('a')
end_unicode = ord('z')

ENG_LETTER = [chr(code_point) for code_point in range(start_unicode, end_unicode + 1)]

# def calc_black_weight(bd_image):
#     return np.sum(bd_image)

# def calc_rel_black_weight(bd_image):
#     return calc_black_weight(bd_image) / bd_image.size

# def calc_center_of_gravity(bd_image):
#     height, width = bd_image.shape

#     black_weight = calc_black_weight(bd_image)

#     center_x = (np.sum(bd_image, axis=1) @ np.array(range(height))) / black_weight
#     center_y = (np.sum(bd_image, axis=0) @ np.array(range(width))) / black_weight

#     return center_x, center_y

# def calc_rel_center_of_gravity(bd_image):
#     height, width = bd_image.shape

#     center_x, center_y = calc_center_of_gravity(bd_image)

#     return (center_x - 1) / (height - 1), (center_y - 1) / (width - 1)

# def calc_horizontal_inertia_moment(bd_image):
#     _, width = bd_image.shape
#     _, y_center = calc_center_of_gravity(bd_image)

#     return np.sum((np.array(range(width)) - y_center)**2 @ np.transpose(bd_image))

# def calc_vertical_inertia_moment(bd_image):
#     height, _ = bd_image.shape
#     x_center, _ = calc_center_of_gravity(bd_image)
#     return np.sum((np.array(range(height)) - x_center)**2 @ bd_image)

# def calc_rel_horizontal_inertia_moment(bd_image):
#     height, width = bd_image.shape

#     return calc_horizontal_inertia_moment(bd_image) / (height**2 * width**2)

# def calc_rel_vertical_inertia_moment(bd_image):
#     height, width = bd_image.shape

#     return calc_vertical_inertia_moment(bd_image) / (height**2 * width**2)


# def create_features(img: np.array):
#     x, y = calc_rel_center_of_gravity(img)
#     return calc_black_weight(img), x, y, calc_rel_horizontal_inertia_moment(img), calc_rel_vertical_inertia_moment(img)

def create_features(img: np.array):
    img_b = np.zeros(img.shape, dtype=int)
    img_b[img != 255] = 1  

    weight = np.sum(img_b)

    y_indices, x_indices = np.indices(img_b.shape)
    y_center_of_mass = np.sum(y_indices * img_b) / weight
    x_center_of_mass = np.sum(x_indices * img_b) / weight

    inertia_x = np.sum((y_indices - y_center_of_mass) ** 2 * img_b) / weight
    inertia_y = np.sum((x_indices - x_center_of_mass) ** 2 * img_b) / weight

    return weight, x_center_of_mass, y_center_of_mass, inertia_x, inertia_y

def get_segments(img):
    profile = np.sum(img == 0, axis=0)

    in_letter = False
    letter_segment = []

    for i in range(len(profile)):
        if profile[i] > 0:
            if not in_letter:
                in_letter = True
                start = i
        else:
            if in_letter:
                in_letter = False
                end = i
                letter_segment.append((start - 1, end))

    if in_letter:
        letter_segment.append((start, len(profile)))

    return letter_segment

def load_features_1():
    with open('5sem/results/2.26/output/csv/data_unicode.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        result = dict()

        for i, row in enumerate(reader):
            weight = int(row['weight'])
            center_of_mass = tuple(map(float, row['center_of_mass'].strip('()').split(',')))
            inertia = tuple(map(float, row['inertia'].strip('()').split(',')))
            result[ENG_LETTER[i]] = weight, *center_of_mass, *inertia

        return result

def create_regocnition(load_features: dict[chr, tuple], target_features):
    def feature_distance(feature1, feature2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(feature1, feature2)))

    distances = dict()
    for letter, features in load_features.items():
        distance = feature_distance(target_features, features)
        distances[letter] = distance

    max_distance = max(distances.values())

    similarities = [(letter, round(1 - distance / max_distance, 2)) for letter, distance in distances.items()]

    return sorted(similarities, key=lambda x: x[1])


def get_regocnition(text, img: np.array, segments):
    load_features = load_features_1()
    res = []
    for start, end in segments:
        letter_features = create_features(img[:, start: end])
        hypothesis = create_regocnition(load_features, letter_features)
        best_hypotheses = hypothesis[-1][0]
        res.append(best_hypotheses)

    res = "".join(res)

    max_len = max(len(text), len(res))
    orig = text.ljust(max_len)
    detected = res.ljust(max_len)
    with open("7sem/results/output/result_power.txt", 'w') as f:
        correct_letters = 0
        by_letter = ["has | got | correct"]
        for i in range(max_len):
            is_correct = orig[i] == detected[i]
            by_letter.append(f"{orig[i]}\t{detected[i]}\t{is_correct}")
            correct_letters += int(is_correct)
        f.write("\n".join([
            f"phrase:      {orig}",
            f"detected:    {detected}",
            f"correct:     {math.ceil(correct_letters / max_len * 100)}%\n\n"
        ]))
        f.write("\n".join(by_letter))

if __name__ == "__main__":
    text = "show me your power".replace(" ", "")
    img = np.array(Image.open(f'6sem/results/6.1/output/text_power.bmp').convert('L'))
    segments = get_segments(img)
    recognized_text = get_regocnition(text, img, segments)
