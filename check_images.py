import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import sys

def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

if len(sys.argv) == 2:
    try:
        if sys.argv[1].lower() == 'normal':
            appendDirectory = 'TEMPnormalUNIQUE'
        elif sys.argv[1].lower() == 'upsidedown':
            appendDirectory = 'TEMPupsidedownUNIQUE'
        elif sys.argv[1].lower() == 'missing':
            appendDirectory = 'TEMPmissingUNIQUE'
        else:
            raise ValueError("unknown class")
    except ValueError:
        print("Invalid input. Please provide - 'normal', 'upsidedown', or 'missing'")
        sys.exit(1)
elif len(sys.argv) == 1:
    print("Input missing. Please provide - 'normal', 'upsidedown', or 'missing'")
    sys.exit(1)
else:
    print("Too many arguments inserted. Please provide exactly one argument - 'normal', 'upsidedown', or 'missing'")
    sys.exit(1)


model = tf.keras.models.load_model('D:\Coding\mmargonemAuctionToolBot\captcha_CNN_py\models\research\antiMargoCaptcha.keras')
targetSize = (62, 83)


targetDirectory = 'D:\Coding\margonemAntiCaptcha\captcha_TOOL_py/' + appendDirectory
checkUnqiues = [f for f in os.listdir(targetDirectory) if os.path.isfile(os.path.join(targetDirectory, f))]

for filename in checkUnqiues:
    input_image = preprocess_image(targetDirectory + '\\'+filename, targetSize)
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])  # Get the index of the class with the highest probability
    class_labels = ['missing', 'normal', 'upsideDown']
    predicted_label = class_labels[predicted_class]
    if (predicted_label.lower() != sys.argv[1].lower()):
        print(f'{filename} - Predicted class: {predicted_label}')