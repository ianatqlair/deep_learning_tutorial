import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Activation, Flatten, MaxPooling2D
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard

dense_layers = [1, 2]
layer_sizes = [32, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            x = pickle.load(open("x" + str(0) + ".obj", "rb"))
            y = pickle.load(open("y" + str(0) + ".obj", "rb"))

            x = x / 255.0

            NAME = "Dense:" + str(dense_layer) + "-" + "Layer_size:" + str(layer_size) + "-" + "Convs: " \
                   + str(conv_layer) + "-" + "Start_time:" + str(time.time())
            tensorboard = TensorBoard(log_dir=('logs/' + NAME))
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer - 1):
                model.add(tf.keras.layers.Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for d in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=['accuracy'])
            for num in range(6):
                x = pickle.load(open("x" + str(num) + ".obj", "rb"))
                y = pickle.load(open("y" + str(num) + ".obj", "rb"))

                x = x / 255.0

                model.fit(x, y, batch_size=32, validation_split=0.2, epochs=5)
            print()
            print(NAME)
            model.fit(x, y, batch_size=32, validation_split=0.2, epochs=5, callbacks=[tensorboard])
            print()
