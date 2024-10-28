import tensorflow as tf
from tensorflow.keras import layers, Model

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import os

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from utils.preprocessing_lesion_images import (
    load_dicom_image,
    dicom_to_gray_scale,
    hair_remove,
    apply_ahe,
    apply_filter_dicom,
)

from models.CustomEfficientNetB7 import Customefficientnetb7

from utils.scheduler import create_early_stopping

from utils.preprocessing_training_tabula import MelanomaDataset


directory = "/kaggle/input/siim-isic-melanoma-classification/"
dataset = MelanomaDataset(directory)
dataset.process_and_save()


# Path to the directory where your DICOM images are stored
image_dir = "/kaggle/input/siim-isic-melanoma-classification/train"

# Load CSV data
train_df = pd.read_csv("new_train.csv")
print(train_df.head())  # Check the first few rows of the DataFrame
print(train_df.shape)  # Check the total number of entries in the DataFrame


# Function to preprocess and resize images
def load_and_preprocess_image(file_path, target_size=[200, 200]):
    original_image = load_dicom_image(file_path)
    gray_sclaed = dicom_to_gray_scale(original_image)
    hair_removed = hair_remove(image=original_image, grayScale=gray_sclaed)
    enhanced_image = apply_ahe(hair_removed)
    denoised = apply_filter_dicom(enhanced_image, filter_type="nlm")
    # Reshape the array to add a channel dimension
    image_array = denoised.reshape(200, 200, 1)
    return image_array


# Load and preprocess images
images = []
targets = []

for idx, row in train_df.iterrows():
    file_path = os.path.join(
        image_dir, row["image_name"] + ".dcm"
    )  # Adjust extension if necessary
    if os.path.exists(file_path):
        image = load_and_preprocess_image(file_path)
        print(image.shape)
        images.append(image)
        targets.append(row["target"])

# Convert lists to numpy arrays
images = np.array(images)
targets = np.array(targets)


# Split data into train and validation sets with stratification
x_train, x_val, y_train, y_val = train_test_split(
    images, targets, test_size=0.2, random_state=42, stratify=targets
)

# Data augmentation generator
data_aug = ImageDataGenerator(
    horizontal_flip=0.5,
    vertical_flip=0.5,
    brightness_range=[0.7, 1.3],
    rescale=1.0 / 255,
    fill_mode="nearest",
)
# Create an instance of the custom model
model = Customefficientnetb7(num_classes=2)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# EarlyStopping callback
early_stopping = create_early_stopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)


# Train model
train_generator = data_aug.flow(x_train, y_train, batch_size=1)
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)


model_save_path = "saved_models/my_efficientnetb7_model"

# Save the entire model to a file
model.save(model_save_path)

print(f"Model saved successfully at {model_save_path}")
