import zipfile
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from data_loader import ECGDataLoader
from preprocessing import ECGPreprocessor
from model import LSTMModel
from evaluation import compute_sequence_errors, evaluate_anomaly_detection

# --- Estrarre zip se necessario ---
zip_path = os.path.join("content", "Tracciati_ECG.zip")
extract_path = os.path.join("content", "Tracciati_ECG")

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Cerca cartella con CSV
data_folder = None
for root, dirs, files in os.walk(extract_path):
    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        data_folder = root
        print(f"Cartella CSV trovata: {data_folder}")
        break

if data_folder is None:
    raise RuntimeError("Cartella con CSV non trovata")

# --- Caricamento dati ---
loader = ECGDataLoader(data_folder)
X_train, y_train, X_test_norm, y_test_norm, X_test_anom, y_test_anom = loader.build_datasets()

print("Dati caricati:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test_norm:", X_test_norm.shape)
print("y_test_norm:", y_test_norm.shape)
print("X_test_anom:", X_test_anom.shape)
print("y_test_anom:", y_test_anom.shape)

# --- Preprocessing ---
prep = ECGPreprocessor()
X_train, y_train = prep.fit_transform(X_train, y_train)
X_test_norm, y_test_norm = prep.transform(X_test_norm, y_test_norm)
X_test_anom, y_test_anom = prep.transform(X_test_anom, y_test_anom)

# --- Modello ---
timesteps = X_train.shape[1]
features = X_train.shape[2]

if os.path.exists("ecg_model.h5"):
    print("✅ Carico modello già addestrato...")
    model = load_model("ecg_model.h5", compile=False)
else:
    print("🚀 Alleno nuovo modello...")
    model = LSTMModel(timesteps=timesteps, features=features)
    model.summary()

    history = model.train(
        X_train, y_train,
        X_val=X_test_norm, y_val=y_test_norm,
        epochs=3,
        batch_size=256
    )

    model.model.save("ecg_model.h5")
    print("💾 Modello salvato in ecg_model.h5")

# --- Valutazione ---
print("\n📊 Calcolo errori su test normali e anomali...")
error_norm = compute_sequence_errors(X_test_norm, y_test_norm, model)
error_anom = compute_sequence_errors(X_test_anom, y_test_anom, model)

# usa solo errori normali per scegliere soglia robusta
threshold, y_pred_norm = evaluate_anomaly_detection(
    error_norm,
    threshold=None,
    k=3.0,
    window=3,
    plot=True
)

# Applichiamo la soglia ai dati anomali
y_pred_anom = (error_anom > threshold).astype(int)

print(f"Threshold scelto automaticamente: {threshold:.6f}")

# --- Visualizzazione errori ---
plt.figure(figsize=(15,5))
plt.hist(error_norm, bins=100, alpha=0.5, label='Normale')
plt.hist(error_anom, bins=100, alpha=0.5, label='Anomalo')
plt.axvline(threshold, color='red', linestyle='--', label='Soglia')
plt.legend()
plt.title("Distribuzione errori normali vs anomali")
plt.show()

# --- Analisi qualitativa ---
print("\n🔍 Analisi battiti classificati come anomali...")
abnormal_indices = np.where(y_pred_anom == 1)[0]
print(f"Numero battiti anomali rilevati: {len(abnormal_indices)}")


# --- Ipotesi setup pratico ---
print("\n💡 Setup pratico proposto: consideriamo ANOMALIA solo se ≥ 3 battiti consecutivi superano la soglia.")


# --- Filtraggio anomalie consecutive ---
def find_consecutive_anomalies(y_pred, min_consecutive=3):
    """
    Restituisce gli intervalli di battiti anomali consecutivi.
    
    Args:
        y_pred: array binario (1=anomalia, 0=normale)
        min_consecutive: numero minimo di battiti consecutivi per considerare un'anomalia

    Returns:
        List of tuples (start_idx, end_idx) per ogni episodio anomalo
    """
    anomalies = []
    count = 0
    start_idx = None

    for i, val in enumerate(y_pred):
        if val == 1:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_consecutive:
                anomalies.append((start_idx, i-1))
            count = 0
            start_idx = None

    # Controlla se l'ultima sequenza termina alla fine
    if count >= min_consecutive:
        anomalies.append((start_idx, len(y_pred)-1))

    return anomalies

# Trova episodi anomali reali
consecutive_anomalies = find_consecutive_anomalies(y_pred_anom, min_consecutive=3)
print(f"\n💡 Numero episodi anomali consecutivi rilevati: {len(consecutive_anomalies)}")
print("Primi 5 episodi:", consecutive_anomalies[:5])

# --- Visualizzazione compatta di episodi anomali ---

n_plot = min(5, len(consecutive_anomalies))
if n_plot > 0:
    fig, axes = plt.subplots(n_plot, 1, figsize=(15, 3*n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]  # garantisce che sia iterabile

    for i in range(n_plot):
        start_idx, end_idx = consecutive_anomalies[i]
        axes[i].plot(y_test_anom[start_idx:end_idx+1].reshape(-1), label="Target (anomalo)")
        axes[i].plot(model.predict(X_test_anom[start_idx:end_idx+1]).reshape(-1), label="Predizione")
        axes[i].set_title(f"Episodio anomalo {i+1} (battiti {start_idx} → {end_idx})")
        axes[i].legend()

    plt.tight_layout()
    plt.show()
else:
    print("Nessun episodio anomalo consecutivo trovato da visualizzare.")
