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


class Customefficientnetb7(Model):
    def __init__(self, num_classes=2):
        super(Customefficientnetb7, self).__init__(name="efficientnetb7")
        # Load the base EfficientNetB7 model with adjusted input shape
        self.base_model = tf.keras.applications.EfficientNetB7(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=(200, 200, 1),
            pooling=None,
        )
        # Add custom top layers
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_pool(x)
        return self.classifier(x)
