import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

class ECGDataLoader:
    def __init__(self, data_folder, sequence_length=128, sampling_rate=400):
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate

        # Durate fasi baseline (in secondi)
        self.phase_durations = {
            'normal_1': 60,
            'apnea': 30,
            'normal_2': 30,
            'hypoventilation': 30,
            'hyperventilation': 30
        }
        self.phase_limits = self.get_phase_indices()

    def get_phase_indices(self):
        phase_limits = {}
        start = 0
        for phase, duration in self.phase_durations.items():
            end = start + duration * self.sampling_rate
            phase_limits[phase] = (start, end)
            start = end
        return phase_limits

    def load_signals(self):
        """Carica tutti i segnali CSV dalla cartella"""
        all_data = []
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.data_folder, filename))
                if 'Tracciato' not in df.columns:
                    continue
                signal = df['Tracciato'].values
                # Label
                if 'baseline' in filename.lower():
                    label = 'baseline'
                elif 'mental' in filename.lower():
                    label = 'mental_stress'
                elif 'physical' in filename.lower():
                    label = 'physical_stress'
                else:
                    label = 'unknown'
                # Persona
                match = re.search(r'Pers\d+', filename)
                person_id = match.group(0) if match else 'unknown'
                all_data.append({'person_id': person_id, 'label': label, 'signal': signal})
        return all_data

    def organize_by_person(self, all_data):
        """Raggruppa i segnali per persona"""
        people_data = defaultdict(lambda: {'baseline': None, 'mental_stress': None, 'physical_stress': None})
        for entry in all_data:
            people_data[entry['person_id']][entry['label']] = entry['signal']
        return people_data

    def generate_sequences(self, signal):
        """Crea sequenze input-target (shift di 1 campione)"""
        X, y = [], []
        seq_len = self.sequence_length
        for i in range(0, len(signal) - seq_len):
            X.append(signal[i:i+seq_len])
            y.append(signal[i+1:i+seq_len+1])
        return np.array(X), np.array(y)

    def build_datasets(self):
        """Costruisce dataset baseline (train e test) e stress (anomali)"""
        all_data = self.load_signals()
        people_data = self.organize_by_person(all_data)

        X_train, y_train = [], []
        X_test_norm, y_test_norm = [], []
        X_test_anom, y_test_anom = [], []

        for person_id, signals in people_data.items():
            baseline = signals['baseline']
            mental = signals['mental_stress']
            physical = signals['physical_stress']

            # --- BASELINE ---
            if baseline is not None and len(baseline) >= self.sequence_length:
                # TRAIN su normal_1, apnea, normal_2
                for phase in ['normal_1', 'apnea', 'normal_2']:
                    start, end = self.phase_limits[phase]
                    end = min(end, len(baseline))  # sicurezza
                    segment = baseline[start:end]
                    X_tr, y_tr = self.generate_sequences(segment)
                    if len(X_tr) > 0:
                        X_train.append(X_tr)
                        y_train.append(y_tr)

                # TEST normali su hypoventilation e hyperventilation
                for phase in ['hypoventilation', 'hyperventilation']:
                    start, end = self.phase_limits[phase]
                    end = min(end, len(baseline))
                    segment = baseline[start:end]
                    X_te, y_te = self.generate_sequences(segment)
                    if len(X_te) > 0:
                        X_test_norm.append(X_te)
                        y_test_norm.append(y_te)

            # --- STRESS (anomali) ---
            for stress_signal in [mental, physical]:
                if stress_signal is not None and len(stress_signal) >= self.sequence_length:
                    X_an, y_an = self.generate_sequences(stress_signal)
                    if len(X_an) > 0:
                        X_test_anom.append(X_an)
                        y_test_anom.append(y_an)

        # Concatenazione finale
        X_train = np.concatenate(X_train) if X_train else np.empty((0, self.sequence_length))
        y_train = np.concatenate(y_train) if y_train else np.empty((0, self.sequence_length))
        X_test_norm = np.concatenate(X_test_norm) if X_test_norm else np.empty((0, self.sequence_length))
        y_test_norm = np.concatenate(y_test_norm) if y_test_norm else np.empty((0, self.sequence_length))
        X_test_anom = np.concatenate(X_test_anom) if X_test_anom else np.empty((0, self.sequence_length))
        y_test_anom = np.concatenate(y_test_anom) if y_test_anom else np.empty((0, self.sequence_length))

        # Aggiungi dimensione canale (per LSTM)
        X_train = X_train[..., np.newaxis]
        y_train = y_train[..., np.newaxis]
        X_test_norm = X_test_norm[..., np.newaxis]
        y_test_norm = y_test_norm[..., np.newaxis]
        X_test_anom = X_test_anom[..., np.newaxis]
        y_test_anom = y_test_anom[..., np.newaxis]

        return X_train, y_train, X_test_norm, y_test_norm, X_test_anom, y_test_anom

