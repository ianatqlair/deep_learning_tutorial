import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Activation, Flatten, MaxPooling2D
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard

NAME = "Cats_vs_dogs_cnn_64x2_" + str(int(int(time.time())))
tensorboard = TensorBoard(log_dir=('logs/' + NAME))


x = pickle.load(open("x.obj", "rb"))
y = pickle.load(open("y.obj", "rb"))

x = x/255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x, y, batch_size=32, validation_split=0.2, epochs=3, callbacks=[tensorboard])

