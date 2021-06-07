from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import Preprocessing
from PIL import Image


def draw_data(results):
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def test_img(img_folder_path, model):
    import random
    X_ids, Y_ids = Preprocessing.load_dataset(img_folder_path)
    test_id = random.choice(X_ids)
    print(test_id)
    img = tf.keras.preprocessing.image.load_img(f"{img_folder_path}/train/{test_id}", target_size=(224, 224))
    input_array = tf.keras.preprocessing.image.img_to_array(img)
    input_array_model = np.array([input_array])
    predictions = model.predict(input_array_model)
    ax10 = plt.subplot(121)
    ax10.imshow(np.asarray(Image.open(f"{img_folder_path}/train/{test_id}")))
    ax11 = plt.subplot(122)
    ax11.imshow(tf.keras.preprocessing.image.array_to_img(np.squeeze(predictions)[:, :, np.newaxis]))
    plt.show()

    ax12 = plt.subplot(121)
    ax12.imshow(np.asarray(tf.keras.preprocessing.image.array_to_img(np.squeeze(predictions)[:, :, np.newaxis])))
    ax13 = plt.subplot(122)
    train_id = test_id.replace('volume', 'labels')
    ax13.imshow(np.asarray(Image.open(f"{img_folder_path}/label/{train_id}")))
    plt.show()
