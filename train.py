import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from utils.preprocessing_lesion_images import (
    load_dicom_image,
    dicom_to_gray_scale,
    hair_remove,
    apply_ahe,
    apply_filter_dicom,
)
from models.custom_efficientb7 import Customefficientnetb7
from models.custom_vggnet import CustomVGGNet
from models.custom_model import CustomModel

# Load and preprocess dataset
image_dir = "/kaggle/input/siim-isic-melanoma-classification/train"
train_df = pd.read_csv("new_train.csv")

def load_and_preprocess_image(file_path, target_size=[200, 200]):
    original_image = load_dicom_image(file_path)
    gray_scaled = dicom_to_gray_scale(original_image)
    hair_removed = hair_remove(image=original_image, grayScale=gray_scaled)
    enhanced_image = apply_ahe(hair_removed)
    denoised = apply_filter_dicom(enhanced_image, filter_type="nlm")
    image_array = denoised.reshape(200, 200, 1)
    return image_array

images, targets = [], []

for idx, row in train_df.iterrows():
    file_path = os.path.join(image_dir, row["image_name"] + ".dcm")
    if os.path.exists(file_path):
        image = load_and_preprocess_image(file_path)
        images.append(image)
        targets.append(row["target"])

images = np.array(images)
targets = np.array(targets)

# Split data
x_train, x_val, y_train, y_val = train_test_split(
    images, targets, test_size=0.2, stratify=targets
)

data_aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    rescale=1.0 / 255,
)

# Initialize and compile the custom model
custom_model = CustomModel()
custom_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Train the model
train_generator = data_aug.flow(x_train, y_train, batch_size=16)
history = custom_model.fit(
    train_generator,
    validation_data=(x_val, y_val),
    epochs=15,
)

model_save_path = "saved_models/custom_melanoma_model"
custom_model.save(model_save_path)
print(f"Model saved at {model_save_path}")
