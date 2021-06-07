import tensorflow as tf
from FCN8s import FCN8s
import Preprocessing, PostTrain

model = FCN8s(n_class=2)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

img_folder_path = "C:\\Users\\Tom-G\\OneDrive\\3DOWN\\Bioinformatics\\Exp\FCN\\ISBI_Dataset"
X_train, Y_train = Preprocessing.generate_train_image(img_folder_path, file_size=224)

callback = tf.keras.callbacks.ModelCheckpoint("FCN8s.h5", verbose=1, save_weights_only=True)
results = model.fit(X_train, Y_train, epochs=30, callbacks=[callback], batch_size=2)

PostTrain.draw_data(results)

PostTrain.test_img(img_folder_path, model)


