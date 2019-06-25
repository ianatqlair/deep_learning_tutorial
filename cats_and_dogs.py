import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm
import random

CURRENTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = CURRENTDIR + "/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100


def create_training_data():
    training_data = []
    x = []
    y = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception:
                pass
    random.shuffle(training_data)
    for features, label in training_data:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    for num in range(10):
        length = x.shape
        upper_limit = int(((num + 1) * .1) * length[1])
        lower_limit = int((num * .1) * length[1])
        temp_x = x[0][lower_limit:upper_limit]
        temp_y = y[lower_limit:upper_limit]

        pickle_out = open("x" + str(num) + ".obj", "wb")
        pickle.dump(temp_x, pickle_out)
        pickle_out.close()

        pickle_out = open("y" + str(num) + ".obj", "wb")
        pickle.dump(temp_y, pickle_out)
        pickle_out.close()


create_training_data()

