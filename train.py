import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.preprocessing_lesion_images import (
    load_dicom_image,
    dicom_to_gray_scale,
    hair_remove,
    apply_ahe,
    apply_filter_dicom
)
from models.custom_model import CustomModel
from utils.scheduler import create_early_stopping
from utils.preprocessing_training_tabula import MelanomaDataset
import os

from utils.segmentation import apply_segmentation


# Load and preprocess dataset
directory = "/kaggle/input/siim-isic-melanoma-classification/"
dataset = MelanomaDataset(directory)
dataset.process_and_save()

# Load CSV data
train_df = pd.read_csv("new_train.csv")

# Load and preprocess images
image_dir = "/kaggle/input/siim-isic-melanoma-classification/train"
images = []
targets = []

def load_and_preprocess_image(file_path, target_size=(200, 200)):
    original_image = load_dicom_image(file_path)
    print("origginal_image done")

    gray_scaled = dicom_to_gray_scale(original_image)
    print("gray_scaled done")

    hair_removed = hair_remove(original_image, gray_scaled)
    print("hair_removed done")

    enhanced_image = apply_ahe(hair_removed)
    print("enhanced_image done")

    denoised = apply_filter_dicom(enhanced_image, filter_type="nlm")
    print("denoised done")
    
    # Apply segmentation
    segmented_image = apply_segmentation(denoised, method="threshold", threshold=0.5)
    print("segmented_image done")
    
    return segmented_image.reshape(*target_size, 1)


for idx, row in train_df.iterrows():
    file_path = os.path.join(image_dir, row["image_name"] + ".dcm")
    if os.path.exists(file_path):
        image = load_and_preprocess_image(file_path)
        images.append(image)
        targets.append(row["target"])

print("Images loaded and preprocessed")

images = np.array(images)
targets = np.array(targets)

# Split data
x_train, x_val, y_train, y_val = train_test_split(
    images, targets, test_size=0.2, random_state=42, stratify=targets
)

print("Data split done")

# Data augmentation
data_aug = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True, brightness_range=[0.7, 1.3], rescale=1.0/255
)

print("Data augmentation done")

# Initialize and train the model
ensemble_model = CustomModel(num_classes=2)
print("Model initialized")

ensemble_model.build_model(input_shape=(200, 200, 1))
print("Model built")

ensemble_model.compile_and_train(x_train, y_train, x_val, y_val)
# Save the model
ensemble_model.save_model("saved_models/ensemble_model")
