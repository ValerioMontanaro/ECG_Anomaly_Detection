from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

class LSTMModel:
    def __init__(self, timesteps, features):
        self.timesteps = timesteps
        self.features = features
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            LSTM(32, activation="tanh", return_sequences=True, input_shape=(self.timesteps, self.features)),
            Dense(16, activation="tanh"),
            TimeDistributed(Dense(self.features))  # output = prossimo battito per ogni timestep
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=64):
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        return history

    def predict(self, X):
        return self.model.predict(X)
