from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def semitone(image):
    width = image.size[0]
    height = image.size[1]
    new_image = Image.new('L', (image.width, image.height))
    for x in range(width):
        for y in range(height):
            pix = image.getpixel((x, y))
            sum_ = 0.3 * pix[0] + 0.59 * pix[1] + 0.11 * pix[2]
            new_image.putpixel((x, y), int(sum_))
    return new_image

def get_haralic(img):
    size = 256 

    img_arr = np.asarray(img).transpose()
    matrix = np.zeros((size, size))
    width = img.size[0]
    height = img.size[1]

    for x in range(1, width - 1): 
        for y in range(1, height - 1):
            pixel = img_arr[x, y]

            up_left_pixel = img_arr[x-1, y-1]
            down_left_pixel = img_arr[x-1, y+1]
            up_right_pixel = img_arr[x+1, y-1]
            down_right_pixel = img_arr[x+1, y+1]

            matrix[pixel, up_left_pixel] += 1  
            matrix[pixel, down_left_pixel] += 1
            matrix[pixel, up_right_pixel] += 1
            matrix[pixel, down_right_pixel] += 1

    return Image.fromarray(matrix).convert('L'), matrix

def vector_Pj(i, matrix):  
    Pj = 0
    for j in range(matrix.shape[1]):
        Pj += matrix[i, j]
    return Pj

def vector_Pi(j, matrix):
    Pi = 0
    for i in range(matrix.shape[0]):
        Pi += matrix[i, j]
    return Pi

def get_corr(haralic_matrix): 
    size = haralic_matrix.shape[0]
    u_i, u_j = 0, 0
    sigma_i, sigma_j = 0, 0
    p_i_j = 0  

    for i in range(size):
        u_i += (i+1) * vector_Pj(i, haralic_matrix)
        u_j += (i+1) * vector_Pi(i, haralic_matrix)

    for i in range(size):
        sigma_j += (i+1-u_j)**2 * vector_Pj(i, haralic_matrix)
        sigma_i += (i+1-u_i)**2 * vector_Pi(i, haralic_matrix)

    sigma_I = np.sqrt(sigma_i) 
    sigma_J = np.sqrt(sigma_j)

    for i in range(size):
        for j in range(size):
            p_i_j += (i+1) * (j+1) * haralic_matrix[i, j]

    return (p_i_j - u_i * u_j) / (sigma_I * sigma_J)

#выравнивание гистограммы
def equalize_histogram(image):
    histogram = np.zeros(256)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1

    histogram = histogram / (image.shape[0] * image.shape[1])

    cumulative_histogram = np.zeros(256)
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = cumulative_histogram[image[i, j]] * 255

    return image

def get_hist(matrix):
    shape = np.reshape(matrix, (1, -1))
    plt.figure()
    plt.hist(shape[0], bins=256)
    return plt

def save_imgs(img_path):
    img = Image.open(img_path)
    semitone_img = semitone(img)
    contrast_img = Image.fromarray(equalize_histogram(np.array(semitone_img)))
    semitone_img.save(f"8sem/results/output/semitone/semitone_img.png")
    contrast_img.save(f"8sem/results/output/contrast/contrast_img.png")

    return semitone_img, contrast_img

def save_hist(semitone_img, contrast_img):
    get_hist(np.asarray(semitone_img)).savefig(f"8sem/results/output/semitone/semitone_hist_img.png")
    get_hist(np.asarray(contrast_img)).savefig(f"8sem/results/output/contrast/contrast_hist_img.png")

def save_haralics(semitone_img, contrast_img):
    haralic_img, haralic_matrix = get_haralic(semitone_img)
    contrast_haralic_img, contrast_haralic_matrix = get_haralic(contrast_img)
    haralic_img.save(f"8sem/results/output/semitone/semitone_haralic_img.png")
    contrast_haralic_img.save(f"8sem/results/output/contrast/contrast_haralic_img.png")

    return haralic_matrix, contrast_haralic_matrix

def save_diff_features(haralic_matrix, contrast_haralic_matrix):
    corr = get_corr(haralic_matrix)
    contrast_corr = get_corr(contrast_haralic_matrix)

    with open("8sem/results/output/result.txt", "w") as file:
        file.write(f"Corr: "+str(corr)+"\n")
        file.write(f"Contrast_corr: "+str(contrast_corr)+"\n")
        file.write(f"Diff_Corr: "+str(np.abs(corr - contrast_corr))+"\n")
    file.close()

    return np.abs(corr - contrast_corr)

def get_state(img_path):
    semitone_img, contrast_img = save_imgs(img_path)
    save_hist(semitone_img, contrast_img)
    haralic_matrix, contrast_haralic_matrix = save_haralics(semitone_img, contrast_img)
    save_diff_features(haralic_matrix, contrast_haralic_matrix)

def main():
    get_state("8sem/results/input/page.png")
    
if __name__ == '__main__':
    main()

