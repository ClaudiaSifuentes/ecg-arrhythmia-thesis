from keras.callbacks import EarlyStopping, ModelCheckpoint

class CustomCallbacks:
    def __init__(self, patience=10, model_save_path='model.h5'):
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True)

    def get_callbacks(self):
        return [self.early_stopping, self.model_checkpoint]