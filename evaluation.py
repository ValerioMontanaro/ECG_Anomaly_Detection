import numpy as np
import matplotlib.pyplot as plt

def compute_sequence_errors(X, y, model):
    """
    Calcola MSE per sequenza (senza normalizzazione)
    """
    preds = model.predict(X)
    mse_per_seq = np.mean((preds - y) ** 2, axis=(1,2))
    return mse_per_seq

def compute_threshold_mean_std(error_norm, k=3.0):
    """
    Soglia robusta: media + k*std sugli errori normali.
    """
    mu = float(np.mean(error_norm))
    sd = float(np.std(error_norm))
    return mu + k * sd

def apply_window_filter(preds, window=3):
    """
    Anomalia solo se ci sono almeno 'window' positivi consecutivi.
    """
    preds = preds.astype(int)
    out = np.zeros_like(preds)
    if window <= 1:
        return preds
    run = 0
    for i, v in enumerate(preds):
        run = run + 1 if v == 1 else 0
        if run >= window:
            out[i-window+1:i+1] = 1
    return out

def evaluate_anomaly_detection(error_norm, threshold=None, k=3.0, window=3, plot=True):
    """
    Valuta anomaly detection usando solo errori normali per calcolare soglia.
    """
    if threshold is None:
        threshold = compute_threshold_mean_std(error_norm, k=k)

    # Predizioni (1=anomalia, 0=normale)
    y_pred_raw = (error_norm > threshold).astype(int)
    y_pred = apply_window_filter(y_pred_raw, window=window)

    if plot:
        plt.figure(figsize=(12,5))
        plt.hist(error_norm, bins=100, alpha=0.5, label='Normale')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Soglia={threshold:.6g}')
        plt.xlabel("Errore MSE per sequenza")
        plt.ylabel("Frequenza")
        plt.title("Distribuzione errori di predizione (baseline)")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f">>> Threshold scelto: {threshold:.6g}  (window={window})")
    return threshold, y_pred
