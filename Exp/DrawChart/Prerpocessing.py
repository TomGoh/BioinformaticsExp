import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image

img_folder_path = 'C:\\Users\\Tom-G\\OneDrive\\3DOWN\\Bioinformatics\\Exp\\UNet\\ISBI_Dataset'


def load_dataset(img_folder_path):
    """
    Get all file names in the given dataset folder
    :param img_folder_path: dataset folder path
    :return: train and label image name tuple
    """
    os.chdir(img_folder_path)
    X_ids = next(os.walk('train'))[2]
    Y_ids = next(os.walk('label'))[2]
    print(len(X_ids), len(Y_ids))
    X_ids.sort()
    Y_ids.sort()
    return X_ids, Y_ids


def vertical_symmetry(img_input):
    return np.array(img_input)[::-1, :, :]


def horizontal_symmetry(img_input):
    return np.array(img_input)[:, ::-1, :]


def vertical_symmetry_2D(img_input):
    return np.array(img_input)[::-1, :]


def horizontal_symmetry_2D(img_input):
    return np.array(img_input)[:, ::-1]


def cropping(img_input):
    return img_input.crop((0, 0, img_input.width / 2, img_input.height / 2)), img_input.crop(
        (0, img_input.height / 2 - 1, img_input.width / 2, img_input.height - 1)), img_input.crop(
        (img_input.width / 2 - 1, 0, img_input.width - 1, img_input.height / 2 - 1)), img_input.crop(
        (img_input.width / 2 - 1, img_input.height / 2 - 1, img_input.width - 1, img_input.height - 1))


IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT = 3, 512, 512


def generate_train_image(img_folder_path, file_size):
    X_ids, Y_ids = load_dataset(img_folder_path)
    X_train = np.zeros((len(X_ids) * 7, file_size, file_size, 3), dtype=np.float32)
    Y_train = np.zeros((len(Y_ids) * 7, file_size, file_size, 1), dtype=np.bool)
    X_original = np.zeros((len(X_ids), 2 * file_size, 2 * file_size, 3), dtype=np.float32)
    Y_original = np.zeros((len(Y_ids), 2 * file_size, 2 * file_size, 1), dtype=np.float32)
    n = 0
    m = 0
    for id_ in X_ids:
        image = tf.keras.preprocessing.image.load_img(f'{img_folder_path}/train/{id_}',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH))
        # print(n,id_)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)[90:450, 150:406]
        image = tf.keras.preprocessing.image.array_to_img(input_arr, ).resize((file_size, file_size))
        original_image = tf.keras.preprocessing.image.array_to_img(input_arr, ).resize((2 * file_size, 2 * file_size))
        X_train[n] = np.array(image)
        X_original[m] = np.array(original_image)
        img1, img2, img3, img4 = cropping(tf.keras.preprocessing.image.array_to_img(X_original[m]))
        n += 1
        X_train[n] = np.array(img1.resize((file_size, file_size)))
        n += 1
        X_train[n] = np.array(img2.resize((file_size, file_size)))
        n += 1
        X_train[n] = np.array(img3.resize((file_size, file_size)))
        n += 1
        X_train[n] = np.array(img4.resize((file_size, file_size)))
        n += 1
        X_train[n] = np.array(tf.keras.preprocessing.image.array_to_img(
            vertical_symmetry(tf.keras.preprocessing.image.array_to_img(X_train[m])), ).resize(
            (file_size, file_size)))
        n += 1
        X_train[n] = np.array(tf.keras.preprocessing.image.array_to_img(
            horizontal_symmetry(tf.keras.preprocessing.image.array_to_img(X_train[m])), ).resize(
            (file_size, file_size)))
        n += 1
        m += 1

    n = 0
    m = 0
    for id_ in Y_ids:
        image = tf.keras.preprocessing.image.load_img(f'{img_folder_path}/label/{id_}',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")

        # print(n, id_)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)[90:450, 150:406]
        image = tf.keras.preprocessing.image.array_to_img(input_arr, ).resize((file_size, file_size))
        original_image = tf.keras.preprocessing.image.array_to_img(input_arr, ).resize((2 * file_size, 2 * file_size))
        Y_train[n] = np.array(image)[:, :, np.newaxis]
        Y_original[m] = np.array(original_image)[:, :, np.newaxis]

        img1, img2, img3, img4 = cropping(tf.keras.preprocessing.image.array_to_img(Y_original[m]))
        n += 1
        Y_train[n] = np.array(img1.resize((file_size, file_size)))[:, :, np.newaxis]
        n += 1
        Y_train[n] = np.array(img2.resize((file_size, file_size)))[:, :, np.newaxis]
        n += 1
        Y_train[n] = np.array(img3.resize((file_size, file_size)))[:, :, np.newaxis]
        n += 1
        Y_train[n] = np.array(img4.resize((file_size, file_size)))[:, :, np.newaxis]
        n += 1
        Y_train[n] = np.array(vertical_symmetry_2D(tf.keras.preprocessing.image.array_to_img(Y_train[m])), )[:, :,
                     np.newaxis]
        n += 1
        Y_train[n] = np.array(horizontal_symmetry_2D(tf.keras.preprocessing.image.array_to_img(Y_train[m])), )[:, :,
                     np.newaxis]
        n += 1
        m += 1
    # pic = Y_train[0][:, :, 0]
    # plt.imshow(pic)
    return X_train, Y_train


# def draw_img(no, X_train, Y_train):
#     ax1 = plt.subplot(121)
#     ax1.imshow(tf.keras.preprocessing.image.array_to_img(X_train[no]))
#     ax2 = plt.subplot(122)
#     ax2.imshow(tf.keras.preprocessing.image.array_to_img(Y_train[no]))
#     plt.show()

if __name__ == '__main__':
    X_train, Y_train = generate_train_image(img_folder_path, 256)
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # ax1.imshow(tf.keras.preprocessing.image.array_to_img(X_train[10]))
    # ax1.set_title('Original Image')
    # ax2.imshow(vertical_symmetry(tf.keras.preprocessing.image.array_to_img(X_train[10])))
    # ax2.set_title('Vertical Symmetry')
    # ax3.imshow(horizontal_symmetry(tf.keras.preprocessing.image.array_to_img(X_train[10])))
    # ax3.set_title('Horizontal Symmetry')
    # plt.savefig('Rotation.jpg')
    # plt.show()
    img1, img2, img3, img4 = cropping(tf.keras.preprocessing.image.array_to_img(X_train[10]))

    ax5 = plt.subplot(221)
    ax5.imshow(img1)
    ax6 = plt.subplot(222)
    ax6.imshow(img2)
    ax7 = plt.subplot(223)
    ax7.imshow(img3)
    ax8 = plt.subplot(224)
    ax8.imshow(img4)

    plt.suptitle('Cropping')
    plt.show()
