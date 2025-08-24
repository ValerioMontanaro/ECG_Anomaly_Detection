import numpy as np
from sklearn.preprocessing import StandardScaler

class ECGPreprocessor:
    """
    Preprocessing per dati ECG sequence-to-sequence:
    - scala X e y usando StandardScaler
    - mantiene forma (samples, timesteps, 1)
    """

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit_transform(self, X_train, y_train):
        """
        Scala X_train e y_train.
        """
        # Flatten per scaler
        X_flat = X_train.reshape(-1, X_train.shape[-1])
        y_flat = y_train.reshape(-1, y_train.shape[-1])

        X_scaled = self.scaler_X.fit_transform(X_flat)
        y_scaled = self.scaler_y.fit_transform(y_flat)

        # Reshape indietro
        X_scaled = X_scaled.reshape(X_train.shape)
        y_scaled = y_scaled.reshape(y_train.shape)
        return X_scaled, y_scaled

    def transform(self, X, y=None):
        """
        Scala X e opzionalmente y usando scaler già fit.
        """
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler_X.transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)

        if y is not None:
            y_flat = y.reshape(-1, y.shape[-1])
            y_scaled = self.scaler_y.transform(y_flat)
            y_scaled = y_scaled.reshape(y.shape)
            return X_scaled, y_scaled

        return X_scaled

    def inverse_transform_y(self, y_scaled):
        """
        Riporta y alla scala originale.
        """
        y_flat = y_scaled.reshape(-1, y_scaled.shape[-1])
        y_inv = self.scaler_y.inverse_transform(y_flat)
        return y_inv.reshape(y_scaled.shape)
