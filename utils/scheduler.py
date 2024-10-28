from tensorflow.keras.callbacks import EarlyStopping


def create_early_stopping(monitor="val_loss", patience=5, restore_best_weights=True):
    return EarlyStopping(
        monitor=monitor, patience=patience, restore_best_weights=restore_best_weights
    )


# early_stopping = create_early_stopping(monitor='val_loss', patience=3, restore_best_weights=True)
