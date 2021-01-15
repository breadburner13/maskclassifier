import cv2 
import tensorflow as tf 

CATEGORIES = ["bruh", "proper"]


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("mask_CNN.model")

prediction = model.predict([prepare('test/proper/comeonbro.jpg')])

prediction2 = model.predict([prepare('test/bruh/business-person-807754.jpg')])

print(prediction)
print(CATEGORIES[int(round(prediction[0][0]))])
print(prediction2)
print(CATEGORIES[int(round(prediction2[0][0]))])