import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

class Colors:
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 3 classes: Upside down, Normal, Missing
    ])
    return model

if len(sys.argv) == 2:
    try:
        num_epochs = int(sys.argv[1])
        if num_epochs < 1 or num_epochs > 100:
            raise ValueError("Number of epochs must be between 1 and 100")
    except ValueError:
        print("Invalid input for number of epochs. Please provide an integer between 1 and 100.")
        sys.exit(1)
else:
    num_epochs = 10

# Set your image dimensions and channels
image_height, image_width = 62, 83
num_channels = 3  # Assuming RGB images
input_shape = (image_height, image_width, num_channels)
num_classes = 3  # Upside down, Normal, Missing

# Create the model
modelName = 'antiMargoCaptcha'
modelDirPath = 'D:/Coding/margonemAuctionToolBot/captcha_CNN_py/models/research/' + modelName + '.keras'
if os.path.exists(modelDirPath):
    print(f"{Colors.OKGREEN}File exists. Loading '{modelName}' model.{Colors.ENDC}")
    model = load_model(modelDirPath)
else:
    print(f"{Colors.WARNING}File doesnt exist. Creating new model.{Colors.ENDC}")
    model = create_model(input_shape, num_classes)


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation and preprocessing
batch_size = 32

# Set paths to your data folders
train_data_dir = 'D:\Coding\margonemAuctionToolBot\captcha_TOOL_py\TrainData'
val_data_dir = 'D:\Coding\margonemAuctionToolBot\captcha_TOOL_py\ValidateData'
test_data_dir = 'D:\Coding\margonemAuctionToolBot\captcha_TOOL_py\TestData'

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generators for train, validation, and test sets
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Train the model
# num_epochs = 10
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator
)

# Evaluate the model
# Print training and validation accuracy for each epoch

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} Train acc: {Colors.OKCYAN}{history.history['accuracy'][epoch]:.5f}{Colors.ENDC} "
          f"- Valid acc: {Colors.OKCYAN}{history.history['val_accuracy'][epoch]:.5f}{Colors.ENDC} ||"
          f" Train loss: {Colors.OKCYAN}{history.history['loss'][epoch]:.5f}{Colors.ENDC} "
          f"- Valid loss: {Colors.OKCYAN}{history.history['val_loss'][epoch]:.5f}{Colors.ENDC}")


# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

decisionMade = False
while decisionMade == False:
    userModelSaveDecision = input('Save this state? [Y] - Yes | [N] - No\n')
    if userModelSaveDecision == 'Y':
        model.save(modelName + '.keras')
        print(f"{Colors.OKGREEN}Saved {modelName} to {modelDirPath}{Colors.ENDC}")
        decisionMade = True
    elif userModelSaveDecision == 'N':
        print(f"{Colors.OKGREEN}Discarded changes{Colors.ENDC}")
        decisionMade = True

