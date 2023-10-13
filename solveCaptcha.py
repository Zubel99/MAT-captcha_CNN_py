import tensorflow as tf
from PIL import Image
import cv2
import os
import numpy as np
import json
import sys
import captchaSplitNoSave
import base64
import io
import random
import string

def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

if __name__ == "__main__":
    #output_path = '..\\..\\..\\captcha_TOOL_py\\captchaBUFFER\\'
    output_path = 'D:\\Coding\\margonemAuctionToolBot\\captcha_TOOL_py\\captchaBUFFER\\'
    full_output_path = os.path.join(output_path, generate_random_string(10))
    try:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_filename = 'antiMargoCaptcha.keras'
        model_path = os.path.join(script_directory, model_filename)
        model = tf.keras.models.load_model(model_path)

        imageBlob = sys.argv[1]
        imgdata = base64.b64decode(str(imageBlob))
        img = Image.open(io.BytesIO(imgdata))
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        #cv2.imshow(opencv_img)
        cv2.imwrite(full_output_path + '.jpg', opencv_img)

        imageArray = captchaSplitNoSave.captchaSplitNoSave(opencv_img)
        #print('AFTER SPLITTING')

        #print(imageArray)
        resultArray = []
        for image in imageArray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = cv2.resize(image, (83, 62))  # Resize
            image = image / 255.0  # Normalize pixel values

            # Predict the class probabilities
            predictions = model.predict(np.expand_dims(image, axis=0))

            # Get the predicted class index (the class with the highest probability)
            predicted_class_index = np.argmax(predictions)

            # Map the class index to the actual class label
            class_labels = ['Missing', 'Normal', 'Upside Down']
            predicted_class = class_labels[predicted_class_index]

            print(f"Predicted class: {predicted_class}")
            resultArray.append(predicted_class)
        print('SPLIT-FETCH')
        print(json.dumps(resultArray))

    except Exception as e:
        print(json.dumps({'error': str(e)}))
