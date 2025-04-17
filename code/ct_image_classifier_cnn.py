# Brain CT Image Classification using CNN
# Description: In this project, we build and train a Convolutional Neural Network (CNN) to classify brain CT scans
#             as either "normal" or showing "hemorrhage".

import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np

# ------------------------------------------------------------------------------
# STEP 1: Data Preview
# ------------------------------------------------------------------------------
# We're starting by loading and visualizing a few CT scan images to get a sense
# of what the data looks like. These are grayscale images from the "normal" class.

path = 'data/head_ct_slices/train/normal/'
image_files = os.listdir(path)

plt.figure(figsize=(5,5))
for i in range(9):
    img = plt.imread(os.path.join(path, image_files[random.randrange(0, len(image_files))]))
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()

# ------------------------------------------------------------------------------
# STEP 2: Set Up Image Data Generators
# ------------------------------------------------------------------------------
# We use TensorFlow's ImageDataGenerator to preprocess our data.
# Here, we're normalizing pixel values to the range [0,1] which helps the model
# learn faster and more effectively.

train_data_generator = ImageDataGenerator(rescale=1/255)
validate_data_generator = ImageDataGenerator(rescale=1./255)

# Define paths to training and validation directories
train_folder = "data/head_ct_slices/train/"
validation_folder = "data/head_ct_slices/validate/"

# Load training images in batches and label them automatically based on folder names
train_generator = train_data_generator.flow_from_directory(
    train_folder,
    target_size=(150, 150),  # We resize all images to a uniform shape
    batch_size=10,
    class_mode="binary"      # Binary classification: normal vs hemorrhage
)

# Do the same for validation images
validate_generator = validate_data_generator.flow_from_directory(
    validation_folder,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

# ------------------------------------------------------------------------------
# STEP 3: Build the CNN Model
# ------------------------------------------------------------------------------
# Now we build our CNN model layer by layer. 
# We're using increasing numbers of filters in successive convolutional layers to learn hierarchical features.
# Each convolutional layer is followed by a max-pooling layer to reduce spatial dimensions and computation.

model = models.Sequential([
    # First Convolution + Pooling layer
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(),

    # Second Convolution + Pooling layer
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Third Convolution + Pooling layer
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Fourth Convolution + Pooling layer
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Fifth Convolution + Pooling layer
    # We're adding a deeper layer here to help the model capture more abstract patterns
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    # Flatten the feature maps and connect to fully-connected layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # This layer learns to interpret the high-level features
    layers.Dense(1, activation='sigmoid')  # Sigmoid outputs a probability for binary classification
])

# ------------------------------------------------------------------------------
# STEP 4: Compile the Model
# ------------------------------------------------------------------------------
# We specify loss function, optimizer, and evaluation metric.
# - Binary crossentropy is used for binary classification tasks.
# - Adam is a popular optimizer that adapts the learning rate automatically.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# ------------------------------------------------------------------------------
# STEP 5: Train the Model
# ------------------------------------------------------------------------------
# This is the training loop. We let the model train for 10 epochs.
# Each epoch means the model will go through all 140 training images once.
# The model also evaluates its performance on the 60 validation images after each epoch.

history = model.fit(
    train_generator,
    validation_data = validate_generator,
    epochs=10
)

# ------------------------------------------------------------------------------
# STEP 6: Test the Model on Unseen Data
# ------------------------------------------------------------------------------
# Now, we evaluate the model's predictions on completely new images it hasn't seen before.
# These are located in the "test" folder. For each image, we predict whether it's normal or hemorrhage.

test_images = os.listdir("data/head_ct_slices/test")

for file_name in test_images:
    path = "data/head_ct_slices/test/" + file_name
    test_img = image.load_img(path, target_size = (150,150))
    
    # Convert the image to an array and scale it
    img = image.img_to_array(test_img)
    img /= 255.0
    img = np.expand_dims(img, axis = 0)

    # Make the prediction
    prediction = model.predict(img)
    print(f"Prediction: {prediction[0]}")
    
    # Interpret and print the result
    if prediction[0] < 0.5:
        print(file_name + " is hemorrhage")
    else:
        print(file_name + " is normal")
    print('\\n')

