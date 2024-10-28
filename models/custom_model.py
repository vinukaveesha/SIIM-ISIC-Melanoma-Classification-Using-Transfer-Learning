import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from models.custom_efficientb7 import Customefficientnetb7
from models.custom_vggnet import CustomVGGNet

class CustomModel:
    def __init__(self, num_classes=2):
        self.vgg = CustomVGGNet()
        self.efficientnet = Customefficientnetb7(num_classes)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

    def build_model(self, input_shape=(200, 200, 1)):
        inputs = tf.keras.Input(shape=input_shape)
        vgg_features = self.vgg(inputs)
        efficient_output = self.efficientnet(vgg_features)
        self.model = tf.keras.Model(inputs=inputs, outputs=efficient_output)

    def compute_loss(self, y_true, vgg_pred, efficientnet_pred):
        vgg_loss = self.loss_fn(y_true, vgg_pred)
        efficientnet_loss = self.loss_fn(y_true, efficientnet_pred)
        return vgg_loss + efficientnet_loss, vgg_loss, efficientnet_loss

    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            vgg_pred = self.vgg(x_batch)
            efficientnet_pred = self.efficientnet(vgg_pred)
            total_loss, vgg_loss, efficientnet_loss = self.compute_loss(y_batch, vgg_pred, efficientnet_pred)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"total_loss": total_loss, "vgg_loss": vgg_loss, "efficientnet_loss": efficientnet_loss}

    def compile_and_train(self, x_train, y_train, x_val, y_val, epochs=15, batch_size=1):
        print("Compiling model...")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                losses = self.train_step(x_batch, y_batch)
                print(f"Step {i//batch_size + 1} - Total Loss: {losses['total_loss']:.4f}, VGG Loss: {losses['vgg_loss']:.4f}, EfficientNet Loss: {losses['efficientnet_loss']:.4f}")

    def save_model(self, path="saved_models/ensemble_model"):
        self.model.save(path)
        print(f"Model saved successfully at {path}")
