import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

class CustomVGGNet(Model):
    def __init__(self):
        super(CustomVGGNet, self).__init__(name="vgg19")
        self.base_model = tf.keras.applications.VGG19(
            include_top=False, weights="imagenet", input_shape=(200, 200, 1)
        )
        self.global_pool = GlobalAveragePooling2D()

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.global_pool(x)
