# import tensorflow as tf
# import os
# from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt
# batch_size = 6
# IMG_SIZE = 160
# IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')
# base_model.trainable = False
# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(1)
# model = tf.keras.Sequential([
#   base_model,
#   global_average_layer,
#   prediction_layer
# ])
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     'maskpics',
#     labels='inferred',
#     label_mode='binary',
#     color_mode='rgb',
#     batch_size=batch_size,
#     image_size=(160, 160),
#     shuffle=True,
# )
# ds_test = tf.keras.preprocessing.image_dataset_from_directory(
#     'test',
#     labels='inferred',
#     label_mode='binary',
#     color_mode='rgb',
#     batch_size=batch_size,
#     image_size=(160, 160),
#     shuffle=True,
# )
# base_learning_rate = 0.001
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(ds_train, epochs=8, verbose=2)
# test_loss, test_acc = model.evaluate(ds_test, verbose=2)
# print(test_acc)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = "maskpics"
CATEGORIES = ["bruh", "proper"]

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path): 
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
        break 
    break
IMG_SIZE = 100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training_data = []
def create_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
      except Exception as e:
        pass
create_training_data()
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = np.array(X/255.0)
y = np.array(y)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, batch_size=2, epochs=4, validation_split=0.1)

model.save("mask_CNN.model")