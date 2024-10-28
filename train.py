import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils.preprocessing_lesion_images import (
    load_dicom_image, dicom_to_gray_scale, hair_remove, apply_ahe, apply_filter_dicom
)
from utils.scheduler import create_early_stopping
from models.custom_efficientb7 import Customefficientnetb7
from models.custom_vggnet import CustomVGGNet
from segmentation import watershed_segmentation

directory = "/kaggle/input/siim-isic-melanoma-classification/"
dataset = MelanomaDataset(directory)
dataset.process_and_save()

# Load and preprocess images
def load_and_preprocess_image(file_path, target_size=(200, 200)):
    original_image = load_dicom_image(file_path)
    gray_scaled = dicom_to_gray_scale(original_image)
    hair_removed = hair_remove(image=original_image, grayScale=gray_scaled)
    enhanced_image = apply_ahe(hair_removed)
    denoised = apply_filter_dicom(enhanced_image, filter_type="nlm")
    segmentation_map = watershed_segmentation(denoised)
    image_array = segmentation_map.reshape(200, 200, 1)
    return image_array

# Load the data
image_dir = "/kaggle/input/siim-isic-melanoma-classification/train"
train_df = pd.read_csv("new_train.csv")
images, targets = [], []

for idx, row in train_df.iterrows():
    file_path = os.path.join(image_dir, row["image_name"] + ".dcm")
    if os.path.exists(file_path):
        image = load_and_preprocess_image(file_path)
        images.append(image)
        targets.append(row["target"])

# Convert lists to numpy arrays
images = np.array(images)
targets = np.array(targets)

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(
    images, targets, test_size=0.2, random_state=42, stratify=targets
)

# Data augmentation
data_aug = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    rescale=1.0 / 255,
)

# Model creation
class CustomModel:
    def __init__(self, num_classes=2):
        self.vgg = CustomVGGNet()
        self.efficientnet = Customefficientnetb7(num_classes)
        self.model = None

    def build_model(self, input_shape=(200, 200, 1)):
        inputs = tf.keras.Input(shape=input_shape)
        vgg_features = self.vgg(inputs)
        efficient_output = self.efficientnet(vgg_features)
        self.model = tf.keras.Model(inputs=inputs, outputs=efficient_output)

    def compile_and_train(self, x_train, y_train, x_val, y_val):
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        early_stopping = create_early_stopping(monitor="val_loss", patience=3, restore_best_weights=True)
        train_generator = data_aug.flow(x_train, y_train, batch_size=1)
        self.model.fit(train_generator, epochs=15, validation_data=(x_val, y_val), callbacks=[early_stopping])

    def save_model(self, path="saved_models/ensemble_model"):
        self.model.save(path)
        print(f"Model saved successfully at {path}")

# Instantiate and train the model
ensemble_model = CustomModel(num_classes=2)
ensemble_model.build_model()
ensemble_model.compile_and_train(x_train, y_train, x_val, y_val)
ensemble_model.save_model()
