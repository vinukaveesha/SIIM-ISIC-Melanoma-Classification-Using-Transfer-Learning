import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D

class CustomVGGNet(tf.keras.Model):
    def __init__(self):
        super(CustomVGGNet, self).__init__()
        # Load VGG19 as a feature extractor
        self.vgg_base = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_shape=(200, 200, 3)  # Use 3 channels for VGG19 input
        )
        self.global_pool = GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.vgg_base(inputs)
        x = self.global_pool(x)
        return x
