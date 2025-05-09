# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Listbox, Checkbutton, BooleanVar, Frame, Label, Scrollbar, constants # Import constants
import threading
import traceback
import glob
import math # Aggiunto per isnan/isinf

# Import per TimeSeriesSplit e EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping


try:
    # Opzionale: usa tkcalendar se disponibile per un date picker migliore
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

# --- Constanti ---
RUOTE_LOTTO = ["BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI", "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE"]
NUM_ESTRATTI_LOTTO = 5
MAX_NUMERO_LOTTO = 90 # I numeri vanno da 1 a 90
DEFAULT_CHECK_COLPI = 12 # Default colpi da verificare
DEFAULT_CV_FOLDS = 5 # Default fold per TimeSeriesSplit

# --- Funzioni Globali ---

def set_seed(seed_value=42):
    """Imposta il seme per la riproducibilità dei risultati."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_seed() # Imposta il seed all'avvio

def log_message(message, log_widget, window):
    """Aggiunge un messaggio al widget di log nella GUI in modo sicuro per i thread."""
    # Verifica che i widget esistano ancora prima di chiamare 'after'
    if log_widget and log_widget.winfo_exists() and window and window.winfo_exists():
        try:
            # Passa una copia del messaggio alla lambda
            msg_copy = str(message)
            window.after(10, lambda: _update_log_widget(log_widget, msg_copy))
        except tk.TclError: # Finestra chiusa
             print(f"Log GUI TclError (window likely closed): {message}")
        except Exception as e_after: # Altri errori after
             print(f"Log GUI Errore in window.after: {e_after}")

def _update_log_widget(log_widget, message):
    """Funzione helper per aggiornare il widget di log."""
    # Verifica widget esistenza
    if not log_widget.winfo_exists():
        print(f"Log GUI TclError: Widget non esiste più. Msg: {message}")
        return
    try:
        current_state = log_widget.cget('state')
        log_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, message + "\n") # Message è già stringa
        log_widget.see(tk.END)
        if current_state == tk.DISABLED:
             log_widget.config(state=tk.DISABLED)
    except tk.TclError:
        print(f"Log GUI TclError (update): {message}")
    except Exception as e:
        print(f"Log GUI unexpected error (update): {e}\nMessage: {message}")

# --- Funzioni Specifiche per il Lotto ---

def aggiungi_feature_temporali(data_df):
    """
    Aggiunge feature temporali (sin/cos) a un DataFrame con colonna 'Data'.
    Opera su una copia e restituisce il nuovo DataFrame.
    """
    if 'Data' not in data_df.columns or pd.api.types.infer_dtype(data_df['Data']) not in ['datetime', 'datetime64']:
         print("WARN (aggiungi_feature_temporali): Colonna 'Data' mancante o non di tipo datetime.")
         return data_df

    df = data_df.copy()

    try:
        df['giorno_settimana'] = df['Data'].dt.dayofweek
        df['giorno_mese'] = df['Data'].dt.day
        df['mese'] = df['Data'].dt.month
        min_year = df['Data'].dt.year.min()
        max_year = df['Data'].dt.year.max()
        df['anno_norm'] = (df['Data'].dt.year - min_year) / (max_year - min_year) if max_year > min_year else 0.5

        df['giorno_sett_sin'] = np.sin(2 * np.pi * df['giorno_settimana'] / 7)
        df['giorno_sett_cos'] = np.cos(2 * np.pi * df['giorno_settimana'] / 7)
        df['mese_sin'] = np.sin(2 * np.pi * df['mese'] / 12)
        df['mese_cos'] = np.cos(2 * np.pi * df['mese'] / 12)
        df['giorno_mese_sin'] = np.sin(2 * np.pi * df['giorno_mese'] / 31)
        df['giorno_mese_cos'] = np.cos(2 * np.pi * df['giorno_mese'] / 31)

        df = df.drop(columns=['giorno_settimana', 'giorno_mese', 'mese'])
        # print("DEBUG (aggiungi_feature_temporali): Feature temporali aggiunte.")
        return df

    except Exception as e:
        print(f"ERRORE (aggiungi_feature_temporali): {e}\n{traceback.format_exc()}")
        return data_df # Ritorna originale in caso di errore

def carica_dati_lotto(folder_path, calculation_wheels, start_date=None, end_date=None, log_callback=None):
    """
    Carica i dati del Lotto, COMBINA le ruote di calcolo e restituisce
    il DataFrame finale pronto per l'aggiunta di feature e la creazione di sequenze.
    *** MODIFICATO: Restituisce SOLO il DataFrame combinato e pulito. ***
    """
    all_data = []
    required_cols = 7
    expected_date_format = '%Y/%m/%d'
    final_columns = ['Data', 'Ruota'] + [f'Num{i+1}' for i in range(NUM_ESTRATTI_LOTTO)]

    if not calculation_wheels:
        if log_callback: log_callback("ERRORE (carica_dati_lotto): Nessuna ruota di calcolo selezionata.")
        return pd.DataFrame(columns=final_columns)
    if not folder_path or not os.path.isdir(folder_path):
         if log_callback: log_callback(f"ERRORE (carica_dati_lotto): Cartella dati non valida: '{folder_path}'")
         return pd.DataFrame(columns=final_columns)

    if log_callback: log_callback(f"Inizio caricamento dati per ruote: {', '.join(calculation_wheels)}")

    for wheel_name in calculation_wheels:
        file_path = os.path.join(folder_path, f"{wheel_name}.txt")
        try:
            if not os.path.exists(file_path):
                if log_callback: log_callback(f"ATTENZIONE: File non trovato {file_path}. Salto.")
                continue

            lines = []
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']; file_read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f: lines = f.readlines()
                    file_read_success = True; break
                except: continue
            if not file_read_success:
                 if log_callback: log_callback(f"ERRORE lettura file {file_path}."); continue
            if not lines: continue

            header_skipped = False; data_lines = lines
            if lines:
                try: datetime.strptime(lines[0].strip().split('\t')[0], expected_date_format)
                except: data_lines = lines[1:]; header_skipped = True
            if not data_lines: continue

            wheel_data = []; malformed_lines = 0
            for line in data_lines:
                values = line.strip().split('\t')
                if len(values) >= required_cols: wheel_data.append(values[:required_cols])
                else: malformed_lines += 1
            # if malformed_lines > 0 and log_callback: log_callback(f"ATT ({wheel_name}): {malformed_lines} righe scartate.")
            if not wheel_data: continue

            col_names_read = ['DataStr', 'Sigla', 'Num1S', 'Num2S', 'Num3S', 'Num4S', 'Num5S']
            df_wheel = pd.DataFrame(wheel_data, columns=col_names_read)
            df_wheel['Data'] = pd.to_datetime(df_wheel['DataStr'], format=expected_date_format, errors='coerce')
            df_wheel = df_wheel.dropna(subset=['Data'])
            if df_wheel.empty: continue

            numeri_cols_str = [f'Num{i+1}S' for i in range(NUM_ESTRATTI_LOTTO)]
            numeri_cols_final = [f'Num{i+1}' for i in range(NUM_ESTRATTI_LOTTO)]
            for i in range(NUM_ESTRATTI_LOTTO):
                df_wheel[numeri_cols_final[i]] = pd.to_numeric(df_wheel[numeri_cols_str[i]], errors='coerce')
                df_wheel.loc[~df_wheel[numeri_cols_final[i]].between(1, MAX_NUMERO_LOTTO, inclusive='both') | ~np.isfinite(df_wheel[numeri_cols_final[i]]), numeri_cols_final[i]] = pd.NA
            df_wheel = df_wheel.dropna(subset=numeri_cols_final)
            if df_wheel.empty: continue

            df_wheel = df_wheel[['Data'] + numeri_cols_final].copy()
            df_wheel['Ruota'] = wheel_name
            try: df_wheel[numeri_cols_final] = df_wheel[numeri_cols_final].astype(int)
            except Exception as e_astype:
                 if log_callback: log_callback(f"ERRORE ({wheel_name}): Conversione int fallita: {e_astype}"); continue
            all_data.append(df_wheel)
        except Exception as e:
            if log_callback: log_callback(f"ERRORE CRITICO caricamento/pulizia {wheel_name}: {e}\n{traceback.format_exc()}")
            continue

    if not all_data:
        if log_callback: log_callback("ERRORE: Nessun dato valido caricato."); return pd.DataFrame(columns=final_columns)

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.sort_values(by='Data').reset_index(drop=True)
    if log_callback: log_callback(f"Dati combinati ({len(all_data)} ruote). Righe prima filtro GUI: {len(df_combined)}.")

    original_min_date = df_combined['Data'].min(); original_max_date = df_combined['Data'].max()
    if start_date:
        try: df_combined = df_combined[df_combined['Data'] >= pd.to_datetime(start_date)].copy()
        except Exception as e_start:
            if log_callback: log_callback(f"Errore filtro data inizio ({start_date}): {e_start}")
    if end_date:
         try: df_combined = df_combined[df_combined['Data'] <= pd.to_datetime(end_date)].copy()
         except Exception as e_end:
              if log_callback: log_callback(f"Errore filtro data fine ({end_date}): {e_end}")
    if log_callback:
        date_range_str = f"da {original_min_date.strftime('%Y-%m-%d') if pd.notna(original_min_date) else '?'} a {original_max_date.strftime('%Y-%m-%d') if pd.notna(original_max_date) else '?'}"
        log_callback(f"Righe dopo filtro date GUI ({start_date} - {end_date}): {len(df_combined)} (su dati {date_range_str})")
    if df_combined.empty:
        if log_callback: log_callback("ERRORE: Nessun dato dopo filtro date GUI."); return pd.DataFrame(columns=final_columns)

    if log_callback: log_callback(f"Caricamento completato. Righe finali: {len(df_combined)}.")
    return df_combined # Ritorna SOLO il DataFrame

def prepara_sequenze_lotto(df_input, sequence_length=5, log_callback=None):
    """
    Prepara le sequenze Input (X) e Target (y) per il modello Lotto.
    AGGIUNTO: Aggiunge feature temporali e restituisce i nomi delle feature.
    """
    if df_input is None or df_input.empty:
        if log_callback: log_callback("ERRORE (prep_seq): DataFrame input vuoto.")
        return None, None, None

    # 1. Aggiungi Feature Temporali
    if log_callback: log_callback("Aggiunta feature temporali...")
    df_with_features = aggiungi_feature_temporali(df_input)

    # 2. Definisci Colonne Feature e Target
    numeri_cols = [f'Num{i+1}' for i in range(NUM_ESTRATTI_LOTTO)]
    temporal_cols_present = sorted([col for col in df_with_features.columns if '_sin' in col or '_cos' in col or col=='anno_norm'])
    feature_cols_per_step = numeri_cols + temporal_cols_present
    target_cols = numeri_cols

    missing_cols = [col for col in feature_cols_per_step if col not in df_with_features.columns]
    if missing_cols:
        if log_callback: log_callback(f"ERRORE: Colonne feature mancanti: {missing_cols}"); return None, None, None
    if log_callback: log_callback(f"Colonne Feature/Step ({len(feature_cols_per_step)}): {feature_cols_per_step}")

    # 3. Estrai Valori NumPy
    feature_values = df_with_features[feature_cols_per_step].values.astype(np.float32)
    target_values_orig = df_with_features[target_cols].values.astype(int)

    # 4. Crea Sequenze
    X, y = [], []
    num_estrazioni = len(feature_values)
    if log_callback: log_callback(f"Preparazione sequenze: seq_len={sequence_length}, num_estrazioni={num_estrazioni}.")
    if num_estrazioni <= sequence_length:
        if log_callback: log_callback(f"ERRORE: Estrazioni({num_estrazioni}) <= seq_len({sequence_length})."); return None, None, None

    num_features_final = sequence_length * len(feature_cols_per_step)

    for i in range(num_estrazioni - sequence_length):
        input_sequence = feature_values[i : i + sequence_length]
        target_numbers = target_values_orig[i + sequence_length]

        if np.all((target_numbers >= 1) & (target_numbers <= MAX_NUMERO_LOTTO)):
            input_flattened = input_sequence.flatten()
            if input_flattened.shape[0] != num_features_final: continue # Skip se shape errata

            target_vector = np.zeros(MAX_NUMERO_LOTTO, dtype=int)
            target_vector[target_numbers - 1] = 1 # Mappa 1..90 a 0..89
            X.append(input_flattened)
            y.append(target_vector)
        # else: Log meno verboso per target invalidi

    if not X:
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata."); return None, None, None

    feature_names_flat = [f"{fn}_t-{t}" for t in range(sequence_length, 0, -1) for fn in feature_cols_per_step]
    X_np, y_np = np.array(X), np.array(y)
    if log_callback: log_callback(f"Create {len(X_np)} sequenze. Shape X: {X_np.shape}, Shape y: {y_np.shape}")

    return X_np, y_np, feature_names_flat

def build_model_lotto(input_shape, hidden_layers=[512, 256, 128], loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    """Costruisce il modello Keras per il Lotto."""
    if log_callback: log_callback(f"Costruzione modello: Input={input_shape}, Hidd={hidden_layers}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate:.2f}, L1/L2={l1_reg:.4f}/{l2_reg:.4f}")
    if not isinstance(input_shape, tuple) or len(input_shape) != 1 or not isinstance(input_shape[0], int) or input_shape[0] <= 0:
         if log_callback: log_callback(f"ERRORE: input_shape {input_shape} non valido."); return None
    model = tf.keras.Sequential(name="Modello_Lotto")
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input_Layer"))
    reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None
    if not hidden_layers:
        if log_callback: log_callback("ATT: No hidden layers.")
    else:
        for i, units in enumerate(hidden_layers):
            if not isinstance(units, int) or units <= 0:
                if log_callback: log_callback(f"ERRORE: unità {units} non valida layer {i+1}."); return None
            model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f"Dense_{i+1}"))
            model.add(tf.keras.layers.BatchNormalization(name=f"BN_{i+1}"))
            if dropout_rate > 0: model.add(tf.keras.layers.Dropout(dropout_rate, name=f"Drop_{i+1}"))
    model.add(tf.keras.layers.Dense(MAX_NUMERO_LOTTO, activation='sigmoid', name="Output_Layer"))
    try:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        if log_callback: log_callback("Modello compilato.")
    except Exception as e:
        if log_callback: log_callback(f"ERRORE compilazione: {e}"); return None
    return model

class LogCallback(tf.keras.callbacks.Callback):
    """Callback Keras per loggare epoche nella GUI."""
    def __init__(self, log_callback_func): super().__init__(); self.log_callback_func = log_callback_func; self._is_running = True
    def stop_logging(self): self._is_running = False
    def on_epoch_end(self, epoch, logs=None):
        if not self._is_running or not self.log_callback_func: return
        logs = logs or {}; msg = f"Epoca {epoch+1:03d} - "
        items = [f"{k.replace('_',' ').replace('val ','v_')}: {v:.4f}" for k, v in logs.items()]
        try: self.log_callback_func(msg + ", ".join(items))
        except: pass # Evita crash se log fallisce

def genera_previsione_lotto(model, X_input_last_sequence, num_predictions=5, log_callback=None):
    """
    Genera la previsione dei numeri del Lotto usando il modello addestrato.
    RESTITUISCE I NUMERI ORDINATI PER PROBABILITÀ DECRESCENTE (ATTENDIBILITÀ).
    *** CORRETTO: Indentazione corretta per tutti gli if ***
    """
    if log_callback: log_callback(f"Generazione previsione per {num_predictions} numeri (ord. probabilità)...")
    if model is None:
        if log_callback:
            log_callback("ERRORE (genera_prev): Modello non valido.")
        return None
    if X_input_last_sequence is None or X_input_last_sequence.size == 0:
        if log_callback:
            log_callback("ERRORE (genera_prev): Input sequenza vuoto.")
        return None
    if not (1 <= num_predictions <= MAX_NUMERO_LOTTO):
        if log_callback:
            log_callback(f"ERRORE: num_predictions={num_predictions} non valido (1-{MAX_NUMERO_LOTTO}).")
        return None

    try:
        # Gestione Shape Input
        if X_input_last_sequence.ndim == 1:
             input_reshaped = X_input_last_sequence.reshape(1, -1)
        elif X_input_last_sequence.ndim == 2 and X_input_last_sequence.shape[0] == 1:
             input_reshaped = X_input_last_sequence
        elif X_input_last_sequence.ndim == 2 and X_input_last_sequence.shape[0] > 0:
             input_reshaped = X_input_last_sequence[0].reshape(1, -1)
             if log_callback: log_callback(f"WARN: Usata prima riga da input shape {X_input_last_sequence.shape}")
        else:
             if log_callback:
                 log_callback(f"ERRORE: Shape input non gestita per la previsione: {X_input_last_sequence.shape}")
             return None

        # Verifica Consistenza Shape
        expected_features = model.input_shape[-1]
        if expected_features is not None and input_reshaped.shape[1] != expected_features:
             if log_callback: log_callback(f"ERRORE Shape: Input {input_reshaped.shape[1]} != Modello {expected_features}."); return None

        # Esecuzione Previsione
        probabilities = model.predict(input_reshaped)

        # --- CORREZIONE INDENTAZIONE QUI ---
        if probabilities is None or probabilities.size == 0:
            if log_callback:
                log_callback("ERRORE: model.predict() ha restituito None o array vuoto.")
            return None
        # -----------------------------------

        # Estrazione Vettore Probabilità
        if probabilities.ndim == 2 and probabilities.shape[0] == 1 and probabilities.shape[1] == MAX_NUMERO_LOTTO:
            probs_vector = probabilities[0]
        else:
             if log_callback: log_callback(f"ERRORE: Output shape {probabilities.shape} != (1, {MAX_NUMERO_LOTTO})."); return None

        # Ordinamento Indici e Selezione Numeri
        sorted_indices = np.argsort(probs_vector)
        top_indices_descending_prob = sorted_indices[-num_predictions:][::-1]
        predicted_numbers_by_prob = [index + 1 for index in top_indices_descending_prob]

        if log_callback:
             log_callback(f"Numeri Lotto predetti ({len(predicted_numbers_by_prob)} ord. prob): {predicted_numbers_by_prob}")
             # Log probabilità associate (opzionale)
             # try: probs_dict = {num: f"{probs_vector[num-1]:.4f}" for num in predicted_numbers_by_prob}; log_callback(f"  Prob: {probs_dict}")
             # except: pass

        return predicted_numbers_by_prob
    except Exception as e:
        if log_callback: log_callback(f"ERRORE generazione previsione: {e}\n{traceback.format_exc()}")
        return None

# --- Funzione Principale di Analisi (MODIFICATA PER CV + Features) ---
def analisi_lotto(folder_path, calculation_wheels, game_wheels,
                   start_date, end_date, sequence_length=5,
                   loss_function='binary_crossentropy', optimizer='adam',
                   dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                   hidden_layers_config=[512, 256, 128],
                   max_epochs=100, batch_size=32, patience=15, min_delta=0.0001,
                   num_predictions=5, num_cv_folds=DEFAULT_CV_FOLDS,
                   log_callback=None):
    """
    Analizza i dati del Lotto usando TimeSeriesSplit CV e genera UNA previsione.
    *** CORRETTO: Gestisce feature temporali e input predizione finale ***
    """
    if log_callback:
        log_callback(f"=== Avvio Analisi Lotto (CV={num_cv_folds} folds, Seq={sequence_length}) ===")
        log_callback(f"Ruote Calcolo: {', '.join(calculation_wheels)}")
        log_callback(f"Ruote Gioco: {', '.join(game_wheels)}")
        log_callback(f"Periodo: {start_date} - {end_date}")
        # Log altri parametri...

    # 1. Carica Dati (DataFrame)
    df_combined = None
    try:
        df_combined = carica_dati_lotto(folder_path, calculation_wheels, start_date, end_date, log_callback=log_callback)
        if df_combined is None or df_combined.empty: return None, "Caricamento dati fallito o DataFrame vuoto."
    except Exception as e:
         if log_callback: log_callback(f"ERRORE CRITICO carica_dati: {e}\n{traceback.format_exc()}"); return None, "Caricamento dati fallito (eccezione)"

    # 2. Prepara Sequenze (con Features)
    X, y, feature_names = None, None, None
    df_with_features_for_pred = None # Salva df con feature per predizione
    feature_cols_per_step_used = None # Salva i nomi delle colonne per step
    try:
        if log_callback: log_callback("Preparazione sequenze e feature...")
        df_with_features_for_pred = aggiungi_feature_temporali(df_combined.copy())
        if df_with_features_for_pred.equals(df_combined): log_callback("ATTENZIONE: Nessuna feature temporale aggiunta.")

        # Recupera anche i nomi delle colonne per step usate qui
        X, y, feature_names = prepara_sequenze_lotto(df_with_features_for_pred, sequence_length, log_callback=log_callback)

        if X is None or y is None or len(X) == 0 or feature_names is None:
            return None, "Creazione sequenze (con feature) fallita"

        # Ricava le colonne *per step* dai nomi appiattiti (più sicuro)
        if feature_names:
             num_features_per_step_calc = len(feature_names) // sequence_length
             feature_cols_per_step_used = feature_names[:num_features_per_step_calc]
             # Rimuovi il suffisso '_t-N' per avere i nomi originali
             feature_cols_per_step_used = [name.rsplit('_t-', 1)[0] for name in feature_cols_per_step_used]
             if log_callback: log_callback(f"Colonne per step identificate: {feature_cols_per_step_used}")
        else:
             if log_callback: log_callback("ERRORE: feature_names non restituiti da prepara_sequenze.")
             return None, "Errore nomi feature"

        if log_callback: log_callback(f"Numero totale features per sequenza appiattita: {X.shape[1]}")

    except Exception as e:
         if log_callback: log_callback(f"Errore preparazione sequenze: {e}\n{traceback.format_exc()}"); return None, f"Errore prep sequenze: {e}"

    # 3. Normalizza Input X (Solo Numeri)
    X_scaled = X.astype(np.float32)
    num_numerical_features = sequence_length * NUM_ESTRATTI_LOTTO
    X_scaled[:, :num_numerical_features] /= MAX_NUMERO_LOTTO
    if log_callback: log_callback(f"Scaling X applicato (div / {MAX_NUMERO_LOTTO} su prime {num_numerical_features} colonne).")

    # --- 4. Cross-Validation Setup ---
    if not isinstance(num_cv_folds, int) or num_cv_folds < 2:
        if log_callback: log_callback(f"ATT: Num fold CV ({num_cv_folds}) non valido, imposto {DEFAULT_CV_FOLDS}."); num_cv_folds = DEFAULT_CV_FOLDS
    if len(X_scaled) <= num_cv_folds:
        msg = f"ERRORE: Campioni ({len(X_scaled)}) insuff. per {num_cv_folds} fold CV (min {num_cv_folds + 1})."
        if log_callback: log_callback(msg); return None, msg
    tscv = TimeSeriesSplit(n_splits=num_cv_folds)
    if log_callback: log_callback(f"Configurazione TimeSeriesSplit con {num_cv_folds} fold.")

    # --- 5. Ciclo di Cross-Validation ---
    fold_metrics = []; best_val_loss_cv = float('inf'); best_model_weights = None; all_histories = []
    input_shape_lotto = (X_scaled.shape[1],)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_scaled, y)):
        if log_callback: log_callback(f"\n--- Avvio Fold CV {fold_idx + 1}/{num_cv_folds} ---")
        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        if len(X_train_fold) < 2 or len(X_val_fold) < 1:
            if log_callback: log_callback(f"Fold {fold_idx+1} saltato: Dati insuff."); continue

        model_fold, history_fold, gui_log_callback_fold = None, None, None
        try:
            tf.keras.backend.clear_session()
            model_fold = build_model_lotto(input_shape_lotto, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
            if model_fold is None: raise ValueError("Costruzione modello fold fallita.")

            early_stopping_fold = EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=1)
            gui_log_callback_fold = LogCallback(log_callback)

            if log_callback: log_callback(f"Fold {fold_idx+1}: Addestramento (Train:{len(X_train_fold)}, Val:{len(X_val_fold)})...")
            history_fold = model_fold.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                                          epochs=max_epochs, batch_size=batch_size,
                                          callbacks=[early_stopping_fold, gui_log_callback_fold], verbose=0)
            epochs_ran = len(history_fold.history['loss'])
            if log_callback: log_callback(f"Fold {fold_idx+1}: Training terminato ({epochs_ran} epoche).")

            train_loss = history_fold.history.get('loss', [np.nan])[-1]
            val_loss = history_fold.history.get('val_loss', [np.nan])[-1]
            train_acc = history_fold.history.get('accuracy', [np.nan])[-1]
            val_acc = history_fold.history.get('val_accuracy', [np.nan])[-1]
            if math.isnan(train_loss): raise ValueError("Train Loss NaN nel fold.")
            ratio = val_loss / train_loss if train_loss > 1e-7 and not (math.isnan(val_loss) or math.isinf(val_loss)) else float('inf')

            metrics = {'fold':fold_idx+1, 'train_loss':train_loss, 'val_loss':val_loss, 'train_acc':train_acc, 'val_acc':val_acc, 'ratio':ratio, 'epochs':epochs_ran}
            fold_metrics.append(metrics); all_histories.append(history_fold.history)

            # Formattazione robusta per il log
            train_loss_str = f"{train_loss:.4f}" if not math.isnan(train_loss) else "N/A"
            val_loss_str = f"{val_loss:.4f}" if not math.isnan(val_loss) else "N/A"
            train_acc_str = f"{train_acc:.4f}" if not math.isnan(train_acc) else "N/A"
            val_acc_str = f"{val_acc:.4f}" if not math.isnan(val_acc) else "N/A"
            ratio_str = f"{ratio:.2f}" if ratio != float('inf') else "Inf"
            log_msg = f"Fold {fold_idx+1} Risultati: Loss(T/V)={train_loss_str}/{val_loss_str}, Acc(T/V)={train_acc_str}/{val_acc_str}, Ratio={ratio_str}"
            if log_callback: log_callback(log_msg)

            # Aggiorna best model (pesi)
            if not math.isnan(val_loss) and val_loss < best_val_loss_cv:
                best_val_loss_cv = val_loss
                best_model_weights = model_fold.get_weights()
                if log_callback: log_callback(f"*** Nuovi Best Weights CV (Fold {fold_idx+1}), Val Loss: {best_val_loss_cv:.4f} ***")

        except Exception as e_fold:
             msg = f"ERRORE Fold {fold_idx+1}: {e_fold}"
             if log_callback: log_callback(msg); log_callback(traceback.format_exc())
             if gui_log_callback_fold: gui_log_callback_fold.stop_logging()
             continue

    # --- 6. Valutazione Complessiva CV e Predizione Finale ---
    if best_model_weights is None:
        msg = "ERRORE: Nessun peso valido salvato durante la CV.";
        if log_callback: log_callback(msg); return None, msg

    # Ricostruisci il miglior modello e carica i pesi
    best_model_cv = None
    try:
        tf.keras.backend.clear_session()
        best_model_cv = build_model_lotto(input_shape_lotto, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if best_model_cv is None: raise ValueError("Ricostruzione best model fallita.")
        best_model_cv.set_weights(best_model_weights)
        if log_callback: log_callback("Miglior modello CV ricostruito con pesi caricati.")
    except Exception as e_rebuild:
         msg = f"ERRORE Ricostruzione/Caricamento pesi best model: {e_rebuild}";
         if log_callback: log_callback(msg); return None, msg

    # Calcolo metriche aggregate e attendibilità (con correzione indentazione)
    attendibilita_msg = "Attendibilità N/D (CV)"
    if fold_metrics:
        valid_metrics = [m for m in fold_metrics if not math.isnan(m.get('val_loss', np.nan))]
        if valid_metrics: # Indentazione corretta
            avg_val_loss = np.mean([m['val_loss'] for m in valid_metrics])
            avg_train_loss = np.mean([m['train_loss'] for m in valid_metrics])
            avg_val_acc = np.mean([m.get('val_acc', np.nan) for m in valid_metrics])
            std_val_loss = np.std([m['val_loss'] for m in valid_metrics])
            consistency = std_val_loss / abs(avg_val_loss) if abs(avg_val_loss) > 1e-7 else float('inf')
            valid_ratios = [m['ratio'] for m in valid_metrics if m['ratio'] != float('inf')]
            avg_ratio = np.mean(valid_ratios) if valid_ratios else float('inf')
            if log_callback:
                log_callback(f"\nRisultati CV Aggregati ({len(valid_metrics)}/{num_cv_folds} fold validi):")
                log_callback(f"  Loss Media (Tr/Val): {avg_train_loss:.4f} / {avg_val_loss:.4f}")
                log_callback(f"  Acc Media (Val): {avg_val_acc:.4f}" if not math.isnan(avg_val_acc) else "  Acc Media (Val): N/A")
                log_callback(f"  Consistenza (Std/Mean ValLoss): {consistency:.4f}")
                log_callback(f"  Ratio Medio Val/Train: {avg_ratio:.2f}" if avg_ratio != float('inf') else "  Ratio Medio: Inf/N/A")
            # Calcolo Attendibilità
            attendibilita = "Non Det."; score = 50
            if avg_ratio < 1.2: attendibilita = "Alta"; score = 85
            elif avg_ratio < 1.8: attendibilita = "Media"; score = 65
            else: attendibilita = "Bassa"; score = 40
            if consistency > 0.5: score -= 15
            score = max(10, min(95, score))
            attendibilita_msg = f"Attendibilità CV: {attendibilita} ({score:.0f}/100)"
            if avg_ratio != float('inf'): attendibilita_msg += f" (Avg Ratio:{avg_ratio:.2f})"
            attendibilita_msg += f" (Avg VLoss:{avg_val_loss:.4f})"
        else:
            # Questo else si riferisce a 'if valid_metrics:'
            if log_callback:
                log_callback("ATTENZIONE: Nessun fold CV con Val Loss valida.")
    else:
        # Questo else si riferisce a 'if fold_metrics:'
        if log_callback:
            log_callback("ATTENZIONE: Nessuna metrica dai fold CV.")

    # Prepara input finale e genera previsione con best_model_cv
    numeri_predetti = None
    try:
        if log_callback: log_callback("Preparazione input finale (con feature)...")
        if df_with_features_for_pred is None or len(df_with_features_for_pred) < sequence_length:
            raise ValueError(f"Dati+feat insuff ({len(df_with_features_for_pred) if df_with_features_for_pred is not None else 0}) per ultima seq ({sequence_length}).")

        last_sequence_df = df_with_features_for_pred.iloc[-sequence_length:]

        # --- CORREZIONE QUI: Usa la lista di nomi per step che abbiamo ricavato ---
        if feature_cols_per_step_used is None:
            raise ValueError("Nomi delle colonne per step non disponibili per la predizione finale.")

        # Verifica che le colonne esistano in last_sequence_df
        missing_cols_pred = [col for col in feature_cols_per_step_used if col not in last_sequence_df.columns]
        if missing_cols_pred:
            raise KeyError(f"Colonne mancanti per predizione finale: {missing_cols_pred} in {last_sequence_df.columns.tolist()}")

        # Seleziona le colonne corrette e ottieni i valori
        last_sequence_features = last_sequence_df[feature_cols_per_step_used].values
        # --------------------------------------------------------------------

        input_pred_flattened = last_sequence_features.flatten()
        input_pred_scaled = input_pred_flattened.astype(np.float32)
        # Scala solo la parte numerica (le prime N*seq_len colonne)
        num_numerical_features_pred = sequence_length * NUM_ESTRATTI_LOTTO
        input_pred_scaled[:num_numerical_features_pred] /= MAX_NUMERO_LOTTO

        if log_callback: log_callback(f"Scaling input finale applicato. Shape input: {input_pred_scaled.shape}")

        # Genera previsione
        numeri_predetti = genera_previsione_lotto(
            best_model_cv, input_pred_scaled, num_predictions, log_callback=log_callback
        )
        if numeri_predetti is None: return None, "Generazione previsione finale fallita."
        if log_callback: log_callback(attendibilita_msg) # Log attendibilità CV
        return numeri_predetti, attendibilita_msg

    except Exception as e:
         if log_callback: log_callback(f"Errore previsione finale: {e}\n{traceback.format_exc()}"); return None, f"Errore previsione finale: {e}"

# --- Definizione Classe AppLotto (CON CV Folds Spinbox) ---
class AppLotto:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi e Previsione Lotto (v1.3 - CV + Features)")
        self.root.geometry("950x980")
        self.style = ttk.Style(); self.style.theme_use('vista' if sys.platform == "win32" else 'clam')
        self.main_frame = ttk.Frame(root, padding="10"); self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_game_wheels = []; self.last_analysis_folder = None

        # --- Folder ---
        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Cartella Dati (.txt)", padding="10"); self.folder_frame.pack(fill=tk.X, pady=5)
        self.folder_path_var = tk.StringVar(value=os.getcwd()); self.folder_entry = ttk.Entry(self.folder_frame, textvariable=self.folder_path_var, width=80); self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_folder_button = ttk.Button(self.folder_frame, text="Sfoglia...", command=self.browse_folder); self.browse_folder_button.pack(side=tk.LEFT)

        # --- Wheels ---
        self.wheels_frame = ttk.Frame(self.main_frame); self.wheels_frame.pack(fill=tk.X, pady=5); self.wheels_frame.columnconfigure(0, weight=1); self.wheels_frame.columnconfigure(1, weight=1)
        self.calc_wheels_frame = ttk.LabelFrame(self.wheels_frame, text="Ruote Calcolo", padding="10"); self.calc_wheels_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        self.calc_wheel_vars = {}; self.calc_wheel_checks = {}; rows_calc = (len(RUOTE_LOTTO) + 1) // 2
        for i, wheel in enumerate(RUOTE_LOTTO): var = BooleanVar(value=False); chk = ttk.Checkbutton(self.calc_wheels_frame, text=wheel, variable=var); r, c = i % rows_calc, i // rows_calc; chk.grid(row=r, column=c, padx=5, pady=2, sticky=tk.W); self.calc_wheel_vars[wheel] = var; self.calc_wheel_checks[wheel] = chk
        self.game_wheel_frame = ttk.LabelFrame(self.wheels_frame, text="Ruote Gioco", padding="10"); self.game_wheel_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew"); self.game_wheel_frame.rowconfigure(0, weight=1); self.game_wheel_frame.rowconfigure(1, weight=0); self.game_wheel_frame.columnconfigure(0, weight=1)
        self.listbox_subframe = ttk.Frame(self.game_wheel_frame); self.listbox_subframe.grid(row=0, column=0, sticky='nsew', pady=(0, 3)); self.listbox_subframe.rowconfigure(0, weight=1); self.listbox_subframe.columnconfigure(0, weight=1)
        self.game_wheel_listbox = Listbox(self.listbox_subframe, selectmode=constants.EXTENDED, height=6, exportselection=False); self.game_wheel_scrollbar = Scrollbar(self.listbox_subframe, orient=constants.VERTICAL, command=self.game_wheel_listbox.yview); self.game_wheel_listbox.config(yscrollcommand=self.game_wheel_scrollbar.set)
        for wheel in RUOTE_LOTTO: self.game_wheel_listbox.insert(constants.END, wheel)
        self.game_wheel_scrollbar.grid(row=0, column=1, sticky='ns'); self.game_wheel_listbox.grid(row=0, column=0, sticky='nsew')
        self.multi_select_info_label = ttk.Label(self.game_wheel_frame, text="Ctrl+Click / Shift+Click per selezione multipla", font=('Helvetica', 8, 'italic'), anchor=tk.W); self.multi_select_info_label.grid(row=1, column=0, sticky='ew', padx=5)

        # --- Params ---
        self.params_container = ttk.Frame(self.main_frame); self.params_container.pack(fill=tk.X, pady=5); self.params_container.columnconfigure(0, weight=1); self.params_container.columnconfigure(1, weight=1)
        # Colonna Sinistra (con CV Folds)
        self.data_params_frame = ttk.LabelFrame(self.params_container, text="Parametri Analisi e CV", padding="10"); self.data_params_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        ttk.Label(self.data_params_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        if HAS_TKCALENDAR: default_start = datetime.now() - pd.Timedelta(days=365*2); self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd', year=default_start.year, month=default_start.month, day=default_start.day); self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        else: self.start_date_entry = ttk.Entry(self.data_params_frame, width=12); self.start_date_entry.insert(0, (datetime.now() - pd.Timedelta(days=365*2)).strftime('%Y-%m-%d')); self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        if HAS_TKCALENDAR: self.end_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd'); self.end_date_entry.set_date(datetime.now()); self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        else: self.end_date_entry = ttk.Entry(self.data_params_frame, width=12); self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d')); self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Seq. Input:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.seq_len_var = tk.StringVar(value="5"); self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=2, to=100, textvariable=self.seq_len_var, width=5, wrap=True); self.seq_len_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Num. Predetti:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="5"); self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=1, to=15, increment=1, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly'); self.num_predict_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="CV Folds (>=2):").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_cv_folds_var = tk.StringVar(value=str(DEFAULT_CV_FOLDS)); self.num_cv_folds_spinbox = ttk.Spinbox(self.data_params_frame, from_=2, to=20, increment=1, textvariable=self.num_cv_folds_var, width=5, wrap=True, state='readonly'); self.num_cv_folds_spinbox.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)
        # Colonna Destra (Modello/Training)
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello e Training", padding="10"); self.model_params_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew"); self.model_params_frame.columnconfigure(1, weight=1)
        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W); self.hidden_layers_var = tk.StringVar(value="256, 128"); self.hidden_layers_entry = ttk.Entry(self.model_params_frame, textvariable=self.hidden_layers_var, width=25); self.hidden_layers_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Loss Function:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W); self.loss_var = tk.StringVar(value='binary_crossentropy'); self.loss_combo = ttk.Combobox(self.model_params_frame, textvariable=self.loss_var, width=23, state='readonly', values=['binary_crossentropy', 'categorical_crossentropy', 'mse', 'mae', 'huber_loss']); self.loss_combo.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Optimizer:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W); self.optimizer_var = tk.StringVar(value='adam'); self.optimizer_combo = ttk.Combobox(self.model_params_frame, textvariable=self.optimizer_var, width=23, state='readonly', values=['adam', 'rmsprop', 'sgd', 'adagrad', 'adamw']); self.optimizer_combo.grid(row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Dropout Rate (0-1):").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W); self.dropout_var = tk.StringVar(value="0.5"); self.dropout_spinbox = ttk.Spinbox(self.model_params_frame, from_=0.0, to=1.0, increment=0.05, format="%.2f", textvariable=self.dropout_var, width=7, wrap=True); self.dropout_spinbox.grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="L1 Strength (>=0):").grid(row=4, column=0, padx=5, pady=3, sticky=tk.W); self.l1_var = tk.StringVar(value="0.0"); self.l1_entry = ttk.Entry(self.model_params_frame, textvariable=self.l1_var, width=7); self.l1_entry.grid(row=4, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="L2 Strength (>=0):").grid(row=5, column=0, padx=5, pady=3, sticky=tk.W); self.l2_var = tk.StringVar(value="0.0"); self.l2_entry = ttk.Entry(self.model_params_frame, textvariable=self.l2_var, width=7); self.l2_entry.grid(row=5, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="Max Epoche:").grid(row=6, column=0, padx=5, pady=3, sticky=tk.W); self.epochs_var = tk.StringVar(value="30"); self.epochs_spinbox = ttk.Spinbox(self.model_params_frame, from_=10, to=1000, increment=10, textvariable=self.epochs_var, width=7, wrap=True); self.epochs_spinbox.grid(row=6, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="Batch Size:").grid(row=7, column=0, padx=5, pady=3, sticky=tk.W); self.batch_size_var = tk.StringVar(value="32"); batch_values = [str(2**i) for i in range(3, 10)]; self.batch_size_combo = ttk.Combobox(self.model_params_frame, textvariable=self.batch_size_var, values=batch_values, width=5, state='readonly'); self.batch_size_combo.grid(row=7, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="ES Patience:").grid(row=8, column=0, padx=5, pady=3, sticky=tk.W); self.patience_var = tk.StringVar(value="15"); self.patience_spinbox = ttk.Spinbox(self.model_params_frame, from_=3, to=100, increment=1, textvariable=self.patience_var, width=7, wrap=True); self.patience_spinbox.grid(row=8, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="ES Min Delta:").grid(row=9, column=0, padx=5, pady=3, sticky=tk.W); self.min_delta_var = tk.StringVar(value="0.0005"); self.min_delta_entry = ttk.Entry(self.model_params_frame, textvariable=self.min_delta_var, width=10); self.min_delta_entry.grid(row=9, column=1, padx=5, pady=3, sticky=tk.W)

        # --- Actions ---
        self.action_frame = ttk.Frame(self.main_frame); self.action_frame.pack(pady=10)
        self.run_button = ttk.Button(self.action_frame, text="Avvia Analisi Lotto", command=self.start_analysis_thread); self.run_button.pack(side=tk.LEFT, padx=10)
        self.check_button = ttk.Button(self.action_frame, text="Verifica Previsione (Ruote Gioco)", command=self.start_check_thread, state=tk.DISABLED); self.check_button.pack(side=tk.LEFT, padx=5)
        self.check_all_button = ttk.Button(self.action_frame, text="Verifica Previsione (Tutte le Ruote)", command=self.start_check_all_thread, state=tk.DISABLED); self.check_all_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.action_frame, text="Colpi:").pack(side=tk.LEFT, padx=(10, 2)); self.check_colpi_var = tk.StringVar(value=str(DEFAULT_CHECK_COLPI)); self.check_colpi_spinbox = ttk.Spinbox(self.action_frame, from_=1, to=100, textvariable=self.check_colpi_var, width=5, state='readonly'); self.check_colpi_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # --- Results ---
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Risultato Previsione Lotto", padding="10"); self.results_frame.pack(fill=tk.X, pady=5)
        self.game_wheels_result_var = tk.StringVar(value=""); self.game_wheels_result_label = ttk.Label(self.results_frame, textvariable=self.game_wheels_result_var, font=('Helvetica', 10, 'italic')); self.game_wheels_result_label.pack(pady=2)
        self.result_label_var = tk.StringVar(value="I numeri previsti appariranno qui..."); self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier', 14, 'bold'), foreground='darkblue'); self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value=""); self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 10, 'italic')); self.attendibilita_label.pack(pady=2)

        # --- Log ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10"); self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 9); self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15, width=100, wrap=tk.WORD, state=tk.DISABLED, font=log_font); self.log_text.pack(fill=tk.BOTH, expand=True)

        # Threads
        self.analysis_thread = None; self.check_thread = None; self.check_all_thread = None

    # --- Metodi della Classe ---

    def browse_folder(self):
        foldername = filedialog.askdirectory(title="Seleziona cartella file .txt ruote")
        if foldername:
            self.folder_path_var.set(foldername)
            self.log_message_gui(f"Cartella: {foldername}")
            found_files = glob.glob(os.path.join(foldername, '*.txt'))
            found_wheels = [os.path.basename(f).replace('.txt', '').upper() for f in found_files]
            missing = sorted([r for r in RUOTE_LOTTO if r not in found_wheels])
            if missing: self.log_message_gui(f"ATT: File mancanti per: {', '.join(missing)}")

    def log_message_gui(self, message):
        log_message(message, self.log_text, self.root)

    def set_result(self, numbers, attendibilita, game_wheels):
         self.root.after(0, self._update_result_labels, numbers, attendibilita, game_wheels)

    def _update_result_labels(self, numbers, attendibilita, game_wheels):
        if game_wheels: self.game_wheels_result_var.set(f"Previsione per Ruote: {', '.join(game_wheels)}")
        else: self.game_wheels_result_var.set("")
        if numbers and isinstance(numbers, list): self.result_label_var.set("  ".join(map(lambda x: f"{x:02d}", numbers)))
        else: self.result_label_var.set("Previsione fallita.")
        self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "")

    def set_controls_state(self, state):
        self.root.after(0, self._set_controls_state_tk, state)

    def _set_controls_state_tk(self, state):
        tk_state = tk.NORMAL if state == tk.NORMAL else tk.DISABLED
        widgets = [
            self.browse_folder_button, self.folder_entry, self.game_wheel_listbox,
            self.seq_len_entry, self.num_predict_spinbox, self.num_cv_folds_spinbox, # Aggiunto CV spinbox
            self.run_button, self.loss_combo, self.optimizer_combo, self.dropout_spinbox,
            self.l1_entry, self.l2_entry, self.hidden_layers_entry, self.epochs_spinbox,
            self.batch_size_combo, self.patience_spinbox, self.min_delta_entry,
            self.check_colpi_spinbox
        ]
        for chk in self.calc_wheel_checks.values(): widgets.append(chk)
        if HAS_TKCALENDAR:
            try: self.start_date_entry.configure(state=tk_state); self.end_date_entry.configure(state=tk_state)
            except: pass
        else: widgets.extend([self.start_date_entry, self.end_date_entry])

        for w in widgets:
            try:
                st = tk_state
                if w == self.run_button and tk_state == tk.NORMAL and \
                   ((self.check_thread and self.check_thread.is_alive()) or \
                    (self.check_all_thread and self.check_all_thread.is_alive())):
                    st = tk.DISABLED
                if isinstance(w, (ttk.Combobox, ttk.Spinbox)): w.configure(state='readonly' if st == tk.NORMAL else tk.DISABLED)
                elif hasattr(w, 'configure'): w.configure(state=st)
            except: pass

        check_st = tk.DISABLED; check_all_st = tk.DISABLED
        if tk_state == tk.NORMAL and not (self.analysis_thread and self.analysis_thread.is_alive()):
             if self.last_prediction is not None:
                 check_all_st = tk.NORMAL
                 if self.last_prediction_game_wheels: check_st = tk.NORMAL
        try:
            self.check_button.configure(state=check_st)
            self.check_all_button.configure(state=check_all_st)
        except: pass

    def start_analysis_thread(self):
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("...", "Analisi già in corso.", parent=self.root); return
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("...", "Verifica in corso.", parent=self.root); return
        if self.check_all_thread and self.check_all_thread.is_alive(): messagebox.showwarning("...", "Verifica in corso.", parent=self.root); return

        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi Lotto in corso..."); self.attendibilita_label_var.set(""); self.game_wheels_result_var.set("")
        self.last_prediction = None; self.last_prediction_game_wheels = []; self.last_prediction_end_date = None; self.last_analysis_folder = None

        # Validazione Input GUI
        errors = []; folder_path = self.folder_path_var.get()
        if not folder_path or not os.path.isdir(folder_path): errors.append(f"Cartella dati non valida:\n{folder_path}")
        selected_calc_wheels = [w for w, v in self.calc_wheel_vars.items() if v.get()]
        if not selected_calc_wheels: errors.append("Selezionare Ruota/e di Calcolo.")
        selected_indices = self.game_wheel_listbox.curselection(); selected_game_wheels = [RUOTE_LOTTO[i] for i in selected_indices]
        if not selected_game_wheels: errors.append("Selezionare Ruota/e di Gioco.")
        if not errors:
             missing_files = []
             for wheel in selected_calc_wheels + selected_game_wheels:
                 fpath = os.path.join(folder_path, f"{wheel}.txt")
                 if not os.path.isfile(fpath) and f"{wheel}.txt" not in missing_files: missing_files.append(f"{wheel}.txt")
             if missing_files: errors.append("File mancanti:\n- " + "\n- ".join(missing_files))
        start_date_str, end_date_str = "", ""
        try:
            start_date_str = self.start_date_entry.get_date().strftime('%Y-%m-%d') if HAS_TKCALENDAR else self.start_date_entry.get()
            end_date_str = self.end_date_entry.get_date().strftime('%Y-%m-%d') if HAS_TKCALENDAR else self.end_date_entry.get()
            if datetime.strptime(start_date_str, '%Y-%m-%d') > datetime.strptime(end_date_str, '%Y-%m-%d'): errors.append("Data inizio > Data fine.")
        except: errors.append("Formato date YYYY-MM-DD non valido.")
        params_to_validate = {"Seq. Input": (self.seq_len_var, int, lambda x: 2 <= x <= 100),"Num. Predetti": (self.num_predict_var, int, lambda x: 1 <= x <= 15),"CV Folds": (self.num_cv_folds_var, int, lambda x: 2 <= x <= 20),"Dropout": (self.dropout_var, float, lambda x: 0.0 <= x <= 1.0),"L1": (self.l1_var, float, lambda x: x >= 0),"L2": (self.l2_var, float, lambda x: x >= 0),"Max Epoche": (self.epochs_var, int, lambda x: x > 0),"Batch Size": (self.batch_size_var, int, lambda x: x > 0),"Patience": (self.patience_var, int, lambda x: x >= 0),"Min Delta": (self.min_delta_var, float, lambda x: x >= 0),}
        for name, (var, type_func, validator) in params_to_validate.items():
            try: val = type_func(var.get()); assert validator(val)
            except: errors.append(f"{name} non valido.")
        hidden_layers_config = []
        try:
            layers_str = [x.strip() for x in self.hidden_layers_var.get().split(',') if x.strip()]
            if layers_str: hidden_layers_config = [int(x) for x in layers_str]; assert all(n > 0 for n in hidden_layers_config)
        except: errors.append("Hidden Layers non validi (es. 256,128).")
        if not self.loss_var.get(): errors.append("Selezionare Loss Function.")
        if not self.optimizer_var.get(): errors.append("Selezionare Optimizer.")

        if errors: messagebox.showerror("Errore Parametri", "\n\n".join(errors), parent=self.root); self.result_label_var.set("Errore parametri."); return

        # Recupero parametri validati
        sequence_length = int(self.seq_len_var.get()); num_predictions = int(self.num_predict_var.get())
        num_cv_folds = int(self.num_cv_folds_var.get())
        loss_function = self.loss_var.get(); optimizer = self.optimizer_var.get()
        dropout_rate = float(self.dropout_var.get()); l1_reg = float(self.l1_var.get()); l2_reg = float(self.l2_var.get())
        max_epochs = int(self.epochs_var.get()); batch_size = int(self.batch_size_var.get())
        patience = int(self.patience_var.get()); min_delta = float(self.min_delta_var.get())

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui("=== Avvio Analisi Lotto (con CV) ===")
        self.log_message_gui(f"Param CV: Folds={num_cv_folds}")
        # Log altri parametri...

        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(folder_path, selected_calc_wheels, selected_game_wheels,
                  start_date_str, end_date_str, sequence_length,
                  loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                  hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                  num_predictions, num_cv_folds), # Passa num_cv_folds
            daemon=True
        )
        self.analysis_thread.start()

    def run_analysis(self, folder_path, calculation_wheels, game_wheels,
                     start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions, num_cv_folds): # Riceve num_cv_folds
        numeri_predetti, attendibilita_msg, success = None, "Analisi Lotto fallita", False
        try:
            numeri_predetti, attendibilita_msg = analisi_lotto(
                folder_path=folder_path, calculation_wheels=calculation_wheels, game_wheels=game_wheels,
                start_date=start_date, end_date=end_date, sequence_length=sequence_length,
                loss_function=loss_function, optimizer=optimizer, dropout_rate=dropout_rate,
                l1_reg=l1_reg, l2_reg=l2_reg, hidden_layers_config=hidden_layers_config,
                max_epochs=max_epochs, batch_size=batch_size, patience=patience, min_delta=min_delta,
                num_predictions=num_predictions,
                num_cv_folds=num_cv_folds, # Passa il numero di fold
                log_callback=self.log_message_gui
            )
            success = isinstance(numeri_predetti, list) and len(numeri_predetti) == num_predictions
        except Exception as e:
            self.log_message_gui(f"\nERRORE CRITICO run_analysis: {e}\n{traceback.format_exc()}")
            attendibilita_msg = f"Errore critico: {e}"; success = False
        finally:
            self.log_message_gui("\n=== Analisi Lotto Completata ===")
            self.set_result(numeri_predetti, attendibilita_msg, game_wheels if success else [])
            if success:
                self.last_prediction = numeri_predetti; self.last_prediction_end_date = end_date
                self.last_prediction_game_wheels = game_wheels; self.last_analysis_folder = folder_path
                self.log_message_gui(f"Previsione salvata per ruote {', '.join(game_wheels)} (dati fino a {end_date}).")
            else:
                self.last_prediction = None; self.last_prediction_end_date = None
                self.last_prediction_game_wheels = []; self.last_analysis_folder = None
                self.log_message_gui("Analisi fallita, nessuna previsione salvata.")
            self.set_controls_state(tk.NORMAL)
            self.analysis_thread = None

    def start_check_thread(self):
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("...", "Verifica già in corso.", parent=self.root); return
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("...", "Attendere fine analisi.", parent=self.root); return
        if self.check_all_thread and self.check_all_thread.is_alive(): messagebox.showwarning("...", "Verifica tutte ruote in corso.", parent=self.root); return
        if self.last_prediction is None or not self.last_prediction_game_wheels or self.last_prediction_end_date is None or self.last_analysis_folder is None: messagebox.showinfo("...", "Nessuna previsione valida per ruote gioco.", parent=self.root); return
        try: num_colpi = int(self.check_colpi_var.get()); assert num_colpi > 0
        except: messagebox.showerror("...", "Numero colpi > 0.", parent=self.root); return
        self.set_controls_state(tk.DISABLED); self.log_message_gui(f"\n=== Avvio Verifica (Ruote Gioco) ==="); # Log...
        self.check_thread = threading.Thread(target=self.run_check_results_multi_lotto, args=(self.last_prediction_game_wheels, self.last_prediction, self.last_prediction_end_date, self.last_analysis_folder, num_colpi), daemon=True); self.check_thread.start()

    def start_check_all_thread(self):
        if self.check_all_thread and self.check_all_thread.is_alive(): messagebox.showwarning("...", "Verifica tutte ruote già in corso.", parent=self.root); return
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("...", "Attendere fine analisi.", parent=self.root); return
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("...", "Verifica ruote gioco in corso.", parent=self.root); return
        if self.last_prediction is None or self.last_prediction_end_date is None or self.last_analysis_folder is None: messagebox.showinfo("...", "Nessuna previsione valida.", parent=self.root); return
        try: num_colpi = int(self.check_colpi_var.get()); assert num_colpi > 0
        except: messagebox.showerror("...", "Numero colpi > 0.", parent=self.root); return
        self.set_controls_state(tk.DISABLED); self.log_message_gui(f"\n=== Avvio Verifica (Tutte le Ruote) ==="); # Log...
        self.check_all_thread = threading.Thread(target=self.run_check_results_all_lotto, args=(self.last_prediction, self.last_prediction_end_date, self.last_analysis_folder, num_colpi), daemon=True); self.check_all_thread.start()

    def run_check_results_multi_lotto(self, game_wheels_to_check, prediction_to_check, last_analysis_date_str, folder_path, num_colpi_to_check):
        any_error = False
        for wheel in game_wheels_to_check:
            self.log_message_gui(f"\n--- Verifica Ruota: {wheel} ---"); fpath = os.path.join(folder_path, f"{wheel}.txt")
            try: self.run_check_results_lotto(fpath, prediction_to_check, last_analysis_date_str, wheel, num_colpi_to_check)
            except Exception as e: self.log_message_gui(f"ERRORE verifica {wheel}: {e}\n{traceback.format_exc()}"); any_error = True
        self.log_message_gui("\n=== Verifica Completata (Ruote Selezionate) ==="); self.set_controls_state(tk.NORMAL); self.check_thread = None

    def run_check_results_all_lotto(self, prediction_to_check, last_analysis_date_str, folder_path, num_colpi_to_check):
        any_error = False
        for wheel in RUOTE_LOTTO:
            self.log_message_gui(f"\n--- Verifica Ruota: {wheel} ---"); fpath = os.path.join(folder_path, f"{wheel}.txt")
            try: self.run_check_results_lotto(fpath, prediction_to_check, last_analysis_date_str, wheel, num_colpi_to_check)
            except Exception as e: self.log_message_gui(f"ERRORE verifica {wheel}: {e}\n{traceback.format_exc()}"); any_error = True
        self.log_message_gui("\n=== Verifica Completata (Tutte le Ruote) ==="); self.set_controls_state(tk.NORMAL); self.check_all_thread = None

    def run_check_results_lotto(self, game_wheel_file_path, prediction_to_check, last_analysis_date_str, game_wheel_name, num_colpi_to_check):
        """Verifica singola ruota."""
        try:
            if not os.path.isfile(game_wheel_file_path): self.log_message_gui(f"ERRORE ({game_wheel_name}): File verifica non trovato."); return
            last_date = datetime.strptime(last_analysis_date_str, '%Y-%m-%d'); check_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            # Carica dati SOLO per la ruota da verificare
            df_check = carica_dati_lotto(os.path.dirname(game_wheel_file_path), [game_wheel_name], start_date=check_start_date, log_callback=self.log_message_gui)
            if df_check is None or df_check.empty: self.log_message_gui(f"INFO ({game_wheel_name}): Nessuna estrazione dopo {last_analysis_date_str}."); return
            # Estrai numeri dal DataFrame
            try: numeri_array_check = df_check[[f'Num{i+1}' for i in range(NUM_ESTRATTI_LOTTO)]].values.astype(int)
            except Exception as e_num_extract: self.log_message_gui(f"ERRORE ({game_wheel_name}): Estrazione numeri fallita dai dati successivi: {e_num_extract}"); return
            if len(numeri_array_check) == 0: self.log_message_gui(f"ERRORE ({game_wheel_name}): Array numeri successivi vuoto."); return

            prediction_set = set(prediction_to_check); colpo = 0; found = False; num_avail = len(numeri_array_check); num_run = min(num_colpi_to_check, num_avail)
            if num_avail < num_colpi_to_check: self.log_message_gui(f"INFO ({game_wheel_name}): Solo {num_avail}/{num_colpi_to_check} estraz. trovate.")
            for i in range(num_run):
                colpo += 1; draw_date_str = df_check.iloc[i]['Data'].strftime('%Y-%m-%d'); actual_draw = numeri_array_check[i]; actual_set = set(actual_draw); hits = prediction_set.intersection(actual_set); num_hits = len(hits); hits_log = ""
                if num_hits > 0:
                    found = True; hit_type = "CINQUINA" if num_hits >= 5 else ["AMBATA", "AMBO", "TERNO", "QUATERNA"][num_hits-1]
                    matched_str = ', '.join(map(str, sorted(list(hits)))); hits_log = f" ---> {hit_type} ({num_hits})! [{matched_str}] <---"
                usciti_list = sorted(actual_draw.tolist()); self.log_message_gui(f"Colpo {colpo:02d} ({draw_date_str}) {game_wheel_name}: {usciti_list}{hits_log}")
            if not found: self.log_message_gui(f"\n({game_wheel_name}): Nessun esito nei {num_run} colpi verificati.")
        except ValueError as ve: self.log_message_gui(f"ERRORE ({game_wheel_name}) formato data verifica: {ve}")
        except FileNotFoundError: self.log_message_gui(f"ERRORE ({game_wheel_name}) File non trovato verifica: {game_wheel_file_path}")
        except Exception as e: self.log_message_gui(f"ERRORE CRITICO ({game_wheel_name}) verifica: {e}\n{traceback.format_exc()}")


# --- Funzione di Lancio (Identica a prima) ---
def launch_lotto_analyzer_window(parent_window):
    """Crea e lancia la finestra dell'applicazione Lotto come Toplevel."""
    try:
        lotto_win = tk.Toplevel(parent_window); lotto_win.title("Analisi e Previsione Lotto (CV+Features)"); lotto_win.geometry("950x980")
        app_instance = AppLotto(lotto_win); lotto_win.lift(); lotto_win.focus_force()
    except Exception as e: print(f"ERRORE lancio finestra: {e}\n{traceback.format_exc()}"); messagebox.showerror("Errore Avvio Modulo Lotto", f"Errore:\n{e}", parent=parent_window)

# --- Blocco Esecuzione Standalone (Identico a prima) ---
if __name__ == "__main__":
    print("Esecuzione modulo Lotto Analyzer standalone (con CV e Features)...")
    # Configurazione TF per ridurre log (opzionale)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except AttributeError:
        pass # Ignora se tf.compat.v1 non è disponibile

    root_standalone = tk.Tk()
    app_standalone = AppLotto(root_standalone)
    root_standalone.mainloop()