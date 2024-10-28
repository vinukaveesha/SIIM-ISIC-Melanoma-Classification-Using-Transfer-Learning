import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

class Customefficientnetb7(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(Customefficientnetb7, self).__init__(name="efficientnetb7")
        self.base_model = tf.keras.applications.EfficientNetB7(
            include_top=False,
            weights="imagenet",  # Use pretrained weights
            input_shape=(200, 200, 3)  # 3 channels
        )
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_pool(x)
        return self.classifier(x)
