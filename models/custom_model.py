import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.vgg = CustomVGGNet()
        self.efficientnet = Customefficientnetb7()

    def compile(self, optimizer, loss_fn):
        super(CustomModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        images, targets = data
        with tf.GradientTape() as tape:
            # Pass through VGG
            vgg_features = self.vgg(images)
            # Pass through EfficientNet
            efficientnet_output = self.efficientnet(images)

            # Calculate loss for both models
            vgg_loss = self.loss_fn(targets, vgg_features)
            efficientnet_loss = self.loss_fn(targets, efficientnet_output)
            total_loss = vgg_loss + efficientnet_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": total_loss}
