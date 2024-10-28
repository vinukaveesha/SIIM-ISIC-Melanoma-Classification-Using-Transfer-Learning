import matplotlib.pyplot as plt


def plot_accuracy(history):
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
