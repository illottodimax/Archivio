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
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import traceback
import requests
import time
import queue
from collections import Counter # Utile per contare ambi/terni
import itertools # Per generare combinazioni (ambi, terni)

# NUOVO: Import per Feature Engineering e Cross-Validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

DEFAULT_10ELOTTO_CHECK_COLPI = 5
DEFAULT_10ELOTTO_DATA_URL = "https://raw.githubusercontent.com/illottodimax/Archivio/main/it-10elotto-past-draws-archive.txt"
DEFAULT_CV_SPLITS = 5
DEFAULT_SPIA_CHECK_COLPI = 5 


# --- Funzioni Globali (Seed, Log) ---
def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_seed()

def log_message(message, log_widget, window):
    if log_widget and window and window.winfo_exists(): # Aggiunto controllo winfo_exists
        window.after(10, lambda: _update_log_widget(log_widget, message))

def _update_log_widget(log_widget, message):
    try:
        if not log_widget.winfo_exists():
             return
        current_state = log_widget.cget('state')
        log_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, str(message) + "\n")
        log_widget.see(tk.END)
        if current_state == tk.DISABLED:
             log_widget.config(state=tk.DISABLED)
    except tk.TclError: # Silently ignore if widget is destroyed during after() call
        pass
    except Exception as e:
        print(f"Log GUI unexpected error in _update_log_widget: {e}\nMessage: {message}")
        try:
            if log_widget.winfo_exists() and log_widget.cget('state') == tk.NORMAL:
                 log_widget.config(state=tk.DISABLED)
        except: # NOSONAR
            pass


# --- Funzioni Specifiche 10eLotto ---
def carica_dati_10elotto(data_source, start_date=None, end_date=None, log_callback=None):
    lines = []
    is_url = data_source.startswith("http://") or data_source.startswith("https://")
    try:
        if is_url:
            if log_callback: log_callback(f"Caricamento dati 10eLotto da URL: {data_source}")
            try:
                response = requests.get(data_source, timeout=30)
                response.raise_for_status()
                content = response.text
                lines = content.splitlines()
                if log_callback: log_callback(f"Dati scaricati con successo ({len(lines)} righe). Encoding: {response.encoding}")
            except requests.exceptions.RequestException as e_req:
                if log_callback: log_callback(f"ERRORE HTTP: Impossibile scaricare i dati da {data_source} - {e_req}")
                return None, None, None, None
            except Exception as e_url:
                if log_callback: log_callback(f"ERRORE generico download URL: {e_url}")
                return None, None, None, None
        else:
            file_path = data_source
            if log_callback: log_callback(f"Caricamento dati 10eLotto da file locale: {file_path}")
            if not os.path.exists(file_path):
                 if log_callback: log_callback(f"ERRORE: File locale non trovato - {file_path}")
                 return None, None, None, None
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
            file_read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f: lines = f.readlines()
                    file_read_success = True
                    if log_callback: log_callback(f"File locale letto con encoding: {enc}")
                    break
                except UnicodeDecodeError: continue
                except Exception as e:
                    if log_callback: log_callback(f"ERRORE lettura file locale ({enc}): {e}")
                    continue
            if not file_read_success:
                 if log_callback: log_callback("ERRORE: Impossibile leggere il file locale.")
                 return None, None, None, None

        if log_callback: log_callback(f"Lette {len(lines)} righe totali dalla fonte dati.")
        if not lines or len(lines) < 2:
            if log_callback: log_callback("ERRORE: Dati vuoti o solo intestazione.")
            return None, None, None, None

        data_lines = lines[1:]
        data = []; malformed_lines = 0; min_expected_cols = 24 # Data + 20 Num + Vuota + Oro1 + Oro2
        for i, line in enumerate(data_lines):
            values = line.rstrip().split('\t')
            if len(values) >= min_expected_cols: data.append(values)
            else: malformed_lines += 1
        if malformed_lines > 0 and log_callback: log_callback(f"ATTENZIONE: {malformed_lines} righe totali scartate (poche colonne).")
        if not data:
            if log_callback: log_callback("ERRORE: Nessuna riga dati valida trovata dopo il parsing iniziale.")
            return None, None, None, None

        max_cols = max(len(row) for row in data)
        colonne_note = ['Data'] + [f'Num{i+1}' for i in range(20)] + ['ColonnaVuota', 'Oro1', 'Oro2']
        colonne_finali = list(colonne_note)
        if max_cols > len(colonne_note): colonne_finali.extend([f'ExtraCol{i}' for i in range(len(colonne_note), max_cols)])
        
        df = pd.DataFrame(data, columns=colonne_finali[:max_cols])
        if 'ColonnaVuota' in df.columns: df = df.drop(columns=['ColonnaVuota'])

        if 'Data' not in df.columns:
            if log_callback: log_callback("ERRORE: Colonna 'Data' mancante."); return None, None, None, None
        df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d', errors='coerce')
        
        invalid_date_mask = pd.isna(df['Data']) # Identifica NaT
        num_invalid_dates = invalid_date_mask.sum()
        if num_invalid_dates > 0 and log_callback:
            log_callback(f"ATTENZIONE: Trovate {num_invalid_dates} righe con data non valida (NaT). Queste righe verranno escluse se il filtro data utente è attivo o se i numeri non sono validi.")

        df = df.sort_values(by='Data', ascending=True, na_position='first') # NaT all'inizio

        if start_date:
            try: df = df[df['Data'] >= pd.to_datetime(start_date)]
            except Exception as e: 
                if log_callback: log_callback(f"Errore filtro data inizio: {e}")
        if end_date:
             try: df = df[df['Data'] <= pd.to_datetime(end_date)]
             except Exception as e: 
                 if log_callback: log_callback(f"Errore filtro data fine: {e}")
        
        # Rimuovi righe con NaT nella data DOPO il filtro utente, per assicurare che il filtro sia applicato correttamente
        df = df.dropna(subset=['Data'])
        if log_callback: log_callback(f"Righe dopo filtro date utente ({start_date} - {end_date}) e rimozione NaT: {len(df)}")

        if df.empty:
            if log_callback: log_callback("ERRORE: Nessun dato rimasto dopo filtro date e pulizia NaT.");
            return df, None, None, None # Ritorna df vuoto
            
        numeri_cols = [f'Num{i+1}' for i in range(20)]; numeri_array, numeri_oro, numeri_extra = None, None, None
        df_cleaned = df.copy() # Lavora su una copia per estrazione numeri
        if not all(col in df_cleaned.columns for col in numeri_cols):
            if log_callback: log_callback(f"ERRORE: Colonne Num1-20 mancanti nel dataframe filtrato."); 
            return df, None, None, None # Ritorna df originale filtrato (senza numeri_array)

        try:
            for col in numeri_cols: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            rows_b4_num_clean = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=numeri_cols) # Rimuove righe con Num1-20 non numerici
            dropped_num = rows_b4_num_clean - len(df_cleaned)
            if dropped_num > 0 and log_callback: log_callback(f"Scartate {dropped_num} righe (Num1-20 non numerici).")
            
            if not df_cleaned.empty:
                numeri_array = df_cleaned[numeri_cols].values.astype(int)
            else:
                if log_callback: log_callback("ATTENZIONE: Nessuna riga rimasta dopo pulizia Num1-20.")
        except Exception as e: 
            if log_callback: log_callback(f"ERRORE pulizia/estrazione Num1-20: {e}")
        
        # Estrazione Oro e Extra (usa df_cleaned che è allineato con numeri_array)
        if numeri_array is not None:
            if 'Oro1' in df_cleaned.columns and 'Oro2' in df_cleaned.columns:
                try:
                    df_oro_temp = df_cleaned[['Oro1', 'Oro2']].copy() # Lavora su copia per Oro
                    df_oro_temp['Oro1'] = pd.to_numeric(df_oro_temp['Oro1'], errors='coerce')
                    df_oro_temp['Oro2'] = pd.to_numeric(df_oro_temp['Oro2'], errors='coerce')
                    df_oro_temp = df_oro_temp.dropna(subset=['Oro1', 'Oro2'])
                    if not df_oro_temp.empty: numeri_oro = df_oro_temp.values.astype(int)
                except Exception as e: 
                    if log_callback: log_callback(f"Errore pulizia/estrazione Oro: {e}")
            
            extra_cols_present = [col for col in df_cleaned.columns if col.startswith('ExtraCol')]
            if extra_cols_present:
                try:
                    # Prendi solo la prima colonna Extra se ce ne sono multiple (comportamento precedente)
                    extra_vals = pd.to_numeric(df_cleaned[extra_cols_present[0]], errors='coerce')
                    numeri_extra = extra_vals.values # Può contenere NaN
                except Exception as e_ex:
                    if log_callback: log_callback(f"ATTENZIONE: Errore conversione colonna Extra '{extra_cols_present[0]}': {e_ex}")
        
        final_rows_df_orig_filter = len(df) # Righe nel df originale dopo filtro date e pulizia NaT
        final_rows_arr = len(numeri_array) if numeri_array is not None else 0
        if log_callback:
            log_callback(f"Caricamento/Filtraggio completato. Righe df (post-filtro date & NaT): {final_rows_df_orig_filter}")
            log_callback(f"Righe array numeri (Num1-20 validi, allineate con df_cleaned): {final_rows_arr}")

        # Ritorna df (filtrato per data e NaT) e numeri_array (pulito e allineato con df_cleaned)
        # Nota: df e numeri_array potrebbero avere lunghezze diverse se ci sono stati dropna sui numeri.
        # Per l'analisi spia che usa le date, sarà necessario usare df_cleaned['Data'] se si vuole coerenza con numeri_array
        return df, numeri_array, numeri_oro, numeri_extra 
    except Exception as e:
        if log_callback: log_callback(f"Errore grave in carica_dati_10elotto: {e}\n{traceback.format_exc()}");
        return None, None, None, None


# --- Funzioni per Machine Learning (Feature Engineering, Sequenze, Modello, Previsione) ---
def engineer_features(numeri_array, log_callback=None):
    if numeri_array is None or numeri_array.ndim != 2 or numeri_array.shape[1] != 20:
        if log_callback: log_callback("ERRORE (engineer_features): Input numeri_array non valido.")
        return None
    try:
        draw_sum = np.sum(numeri_array, axis=1, keepdims=True)
        draw_mean = np.mean(numeri_array, axis=1, keepdims=True)
        odd_count = np.sum(numeri_array % 2 != 0, axis=1, keepdims=True)
        even_count = 20 - odd_count 
        low_count = np.sum((numeri_array >= 1) & (numeri_array <= 45), axis=1, keepdims=True)
        high_count = 20 - low_count
        engineered_features = np.concatenate([draw_sum, draw_mean, odd_count, even_count, low_count, high_count], axis=1)
        combined_features = np.concatenate([numeri_array, engineered_features], axis=1)
        if log_callback: log_callback(f"Feature Engineering completato. Shape: {combined_features.shape}")
        return combined_features
    except Exception as e:
        if log_callback: log_callback(f"ERRORE Feature Engineering: {e}")
        return None

def prepara_sequenze_per_modello(input_feature_array, target_number_array, sequence_length=5, log_callback=None):
    if input_feature_array is None or target_number_array is None or \
       input_feature_array.ndim != 2 or target_number_array.ndim != 2 or \
       input_feature_array.shape[0] != target_number_array.shape[0] or \
       target_number_array.shape[1] != 20:
        if log_callback: log_callback("ERRORE (prep_seq): Input non validi.")
        return None, None

    n_features = input_feature_array.shape[1]
    X, y = [], []
    num_estrazioni = len(input_feature_array)
    if num_estrazioni <= sequence_length:
        if log_callback: log_callback(f"ERRORE: Estrazioni ({num_estrazioni}) <= seq ({sequence_length}).")
        return None, None

    for i in range(num_estrazioni - sequence_length):
        in_seq = input_feature_array[i : i + sequence_length]
        tgt_extr = target_number_array[i + sequence_length]
        if np.all((tgt_extr >= 1) & (tgt_extr <= 90)):
            target = np.zeros(90, dtype=int)
            target[tgt_extr - 1] = 1 
            X.append(in_seq.flatten())
            y.append(target)
    
    if not X:
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata."); return None, None
    try:
        X_np, y_np = np.array(X), np.array(y)
        if log_callback: log_callback(f"Create {len(X_np)} sequenze. Shape X={X_np.shape}, y={y_np.shape}")
        return X_np, y_np
    except Exception as e:
        if log_callback: log_callback(f"ERRORE conversione NumPy in prep_seq: {e}"); return None, None

def build_model_10elotto(input_shape, hidden_layers=None, loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    if hidden_layers is None: hidden_layers = [256, 128]
    if not isinstance(input_shape, tuple) or len(input_shape) != 1 or not isinstance(input_shape[0], int) or input_shape[0] <= 0:
        if log_callback: log_callback(f"ERRORE build_model: input_shape '{input_shape}' non valido."); return None
    
    model = tf.keras.Sequential(name="Modello_10eLotto")
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input_Layer"))
    reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg + l2_reg > 0 else None
    
    for i, units in enumerate(hidden_layers):
        if not isinstance(units, int) or units <= 0:
            if log_callback: log_callback(f"ERR: Unità layer {i+1} non valida ({units})."); return None
        model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f"Dense_{i+1}"))
        model.add(tf.keras.layers.BatchNormalization(name=f"BN_{i+1}"))
        if 0 < dropout_rate < 1:
            model.add(tf.keras.layers.Dropout(dropout_rate, name=f"Drop_{i+1}_{dropout_rate:.2f}"))
            
    model.add(tf.keras.layers.Dense(90, activation='sigmoid', name="Output_Layer_90_Sigmoid"))
    try:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        if log_callback: log_callback("Modello 10eLotto compilato.")
    except Exception as e:
        if log_callback: log_callback(f"ERR compilazione modello: {e}"); return None
    return model

class LogCallbackKeras(tf.keras.callbacks.Callback): # Rinominato per chiarezza
    def __init__(self, log_callback_func, stop_event=None):
         super().__init__()
         self.log_callback_func = log_callback_func
         self.stop_event = stop_event
    def on_epoch_end(self, epoch, logs=None):
        if self.stop_event and self.stop_event.is_set():
            self.model.stop_training = True
            if self.log_callback_func: self.log_callback_func(f"Epoca {epoch+1}: Stop richiesto, arresto training...")
            return
        if self.log_callback_func:
            logs = logs or {}; msg = f"Epoca {epoch+1:03d} - "
            items = [f"{k.replace('_',' ').replace('val ','v_')}: {v:.4f}" for k, v in logs.items()]
            self.log_callback_func(msg + ", ".join(items))

def genera_previsione_10elotto(model, X_input, num_predictions=10, log_callback=None):
    if model is None or X_input is None or X_input.size == 0 or not (1 <= num_predictions <= 90):
        if log_callback: log_callback("ERR (genera_prev): Input non validi."); return None
    try:
        input_reshaped = X_input.reshape(1, -1) if X_input.ndim == 1 else X_input
        if input_reshaped.ndim != 2 or input_reshaped.shape[0] != 1:
             if log_callback: log_callback(f"ERR: Shape input per predict non valida: {input_reshaped.shape}. Atteso (1, N_features)."); return None
        
        pred_probabilities = model.predict(input_reshaped, verbose=0)
        if pred_probabilities.shape != (1, 90):
             if log_callback: log_callback(f"ERR: Output shape da predict inatteso: {pred_probabilities.shape}. Atteso (1, 90)."); return None

        probs_vector = pred_probabilities[0]
        top_indices_descending_prob = np.argsort(probs_vector)[-num_predictions:][::-1]
        predicted_numbers = [int(index + 1) for index in top_indices_descending_prob]
        if log_callback: log_callback(f"Numeri ML predetti ({len(predicted_numbers)}): {predicted_numbers}")
        return predicted_numbers
    except Exception as e:
        if log_callback: log_callback(f"ERRORE CRITICO generazione previsione ML: {e}\n{traceback.format_exc()}"); return None

# --- Funzione Analisi Principale ML ---
def analisi_10elotto(file_path, start_date, end_date, sequence_length=5,
                     loss_function='binary_crossentropy', optimizer='adam',
                     dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                     hidden_layers_config=None, # Default in build_model
                     max_epochs=30, batch_size=32, patience=15, min_delta=0.0001,
                     num_predictions=10, n_cv_splits=DEFAULT_CV_SPLITS,
                     log_callback=None, stop_event=None):
    if log_callback: log_callback(f"=== Avvio Analisi 10eLotto ML (FE & CV) ===")
    
    _, numeri_array, _, _ = carica_dati_10elotto(file_path, start_date, end_date, log_callback=log_callback)
    if numeri_array is None or len(numeri_array) < sequence_length + 1:
        msg = "Dati numerici insufficienti per analisi ML."; log_callback(msg); return None, msg
    
    combined_features = engineer_features(numeri_array, log_callback=log_callback)
    if combined_features is None: return None, "Feature Engineering fallito."
    
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)
    
    X, y = prepara_sequenze_per_modello(combined_features_scaled, numeri_array, sequence_length, log_callback=log_callback)
    if X is None or y is None or len(X) < n_cv_splits + 1:
        msg = "Creazione sequenze fallita o campioni insuff. per CV."; log_callback(msg); return None, msg

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    fold_val_losses, fold_val_accuracies = [], []
    if log_callback: log_callback(f"\n--- Inizio {n_cv_splits}-Fold TimeSeries Cross-Validation ---")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        if stop_event and stop_event.is_set(): break
        if log_callback: log_callback(f"\n--- Fold {fold+1}/{n_cv_splits} ---")
        X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

        tf.keras.backend.clear_session()
        model_fold = build_model_10elotto((X_train.shape[1],), hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if model_fold is None: continue

        early_stopping_fold = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, restore_best_weights=True)
        keras_log_cb = LogCallbackKeras(log_callback, stop_event)
        
        history_fold = model_fold.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping_fold, keras_log_cb], verbose=0)
        
        if history_fold and 'val_loss' in history_fold.history:
            best_idx = np.argmin(history_fold.history['val_loss'])
            fold_val_losses.append(history_fold.history['val_loss'][best_idx])
            if 'val_accuracy' in history_fold.history: fold_val_accuracies.append(history_fold.history['val_accuracy'][best_idx])
    
    if stop_event and stop_event.is_set(): return None, "Analisi ML Interrotta (CV)"
    
    avg_val_loss = np.mean(fold_val_losses) if fold_val_losses else -1
    avg_val_acc = np.mean(fold_val_accuracies) if fold_val_accuracies else -1
    attendibilita_cv_msg = f"Attendibilità CV: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f} ({len(fold_val_losses)} folds)"
    if log_callback: log_callback(f"\n--- Risultati CV: {attendibilita_cv_msg} ---")

    if log_callback: log_callback("\n--- Addestramento Modello Finale ML ---")
    tf.keras.backend.clear_session()
    final_model = build_model_10elotto((X.shape[1],), hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
    if final_model is None: return None, "Costruzione modello finale ML fallita"
    
    final_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=min_delta, restore_best_weights=True)
    keras_log_cb_final = LogCallbackKeras(log_callback, stop_event)
    history_final = final_model.fit(X, y, epochs=max_epochs, batch_size=batch_size, callbacks=[final_early_stopping, keras_log_cb_final], verbose=0)
    
    if stop_event and stop_event.is_set(): return None, "Analisi ML Interrotta (Final Train)"
    
    final_loss = history_final.history.get('loss', [float('inf')])[-1]
    final_acc = history_final.history.get('accuracy', [-1.0])[-1]
    if log_callback: log_callback(f"Training finale ML completato: Loss={final_loss:.4f}, Acc={final_acc:.4f}")
    
    input_pred_seq_scaled = combined_features_scaled[-sequence_length:]
    input_pred_ready = input_pred_seq_scaled.flatten().reshape(1, -1)
    numeri_predetti = genera_previsione_10elotto(final_model, input_pred_ready, num_predictions, log_callback)
    
    final_attendibilita_msg = f"{attendibilita_cv_msg}. Training finale: Loss={final_loss:.4f}, Acc={final_acc:.4f}"
    return numeri_predetti, final_attendibilita_msg


# --- Funzione Calcolo Numeri Spia (Singoli, Ambi, Terni) ---
def calcola_numeri_spia(numeri_array_storico, date_array_storico, numeri_spia_target,
                        colpi_successivi, top_n_singoli, top_n_ambi, top_n_terni,
                        log_callback=None, stop_event=None):
    if numeri_array_storico is None or len(numeri_array_storico) == 0 or not numeri_spia_target:
        if log_callback: log_callback("ERRORE (Spia): Dati input non validi per calcolo spia.")
        return [], [], [], 0, None, None

    num_estrazioni = len(numeri_array_storico)
    if num_estrazioni <= colpi_successivi:
        if log_callback: log_callback(f"ATT (Spia): Dati insuff. ({num_estrazioni}) per analizzare {colpi_successivi} colpi.")
        return [], [], [], 0, None, None

    frequenze_singoli, frequenze_ambi, frequenze_terni = Counter(), Counter(), Counter()
    occorrenze_spia_trovate = 0
    data_inizio_scan, data_fine_scan = "N/A", "N/A"

    if date_array_storico is not None and len(date_array_storico) == num_estrazioni:
        try:
            valid_dates = date_array_storico[~pd.isna(date_array_storico)]
            if len(valid_dates) > 0:
                data_inizio_scan = pd.to_datetime(valid_dates.min()).strftime('%Y-%m-%d')
                data_fine_scan = pd.to_datetime(valid_dates.max()).strftime('%Y-%m-%d')
        except Exception: pass # Ignora errori nel determinare le date per il log
    if log_callback: log_callback(f"Analisi Spia: Scansione {num_estrazioni} estrazioni (Periodo: {data_inizio_scan} - {data_fine_scan}).")

    numeri_spia_set = set(numeri_spia_target)
    for i in range(num_estrazioni - colpi_successivi):
        if stop_event and stop_event.is_set(): break
        if numeri_spia_set.issubset(set(numeri_array_storico[i])):
            occorrenze_spia_trovate += 1
            for k in range(1, colpi_successivi + 1):
                if stop_event and stop_event.is_set(): break
                idx_succ = i + k
                if idx_succ < num_estrazioni:
                    estrazione_succ_valid = [n for n in numeri_array_storico[idx_succ] if 1 <= n <= 90 and n not in numeri_spia_set]
                    frequenze_singoli.update(estrazione_succ_valid)
                    if len(estrazione_succ_valid) >= 2: frequenze_ambi.update(itertools.combinations(sorted(estrazione_succ_valid), 2))
                    if len(estrazione_succ_valid) >= 3: frequenze_terni.update(itertools.combinations(sorted(estrazione_succ_valid), 3))
        if i > 0 and i % 2000 == 0 and log_callback: log_callback(f"Analisi Spia: Elaborate {i}/{num_estrazioni - colpi_successivi}...")

    if log_callback and not (stop_event and stop_event.is_set()):
         log_callback(f"Analisi Spia: Trovate {occorrenze_spia_trovate} occorrenze spia {numeri_spia_target}.")
    
    return (frequenze_singoli.most_common(top_n_singoli),
            frequenze_ambi.most_common(top_n_ambi),
            frequenze_terni.most_common(top_n_terni),
            occorrenze_spia_trovate, data_inizio_scan, data_fine_scan)


# --- Classe Applicazione GUI ---
class App10eLotto:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi e Previsione 10eLotto (v8.3 - Completo)")
        self.root.geometry("850x1150") 

        self.style = ttk.Style()
        try: 
            if sys.platform == "win32": self.style.theme_use('vista')
            elif sys.platform == "darwin": self.style.theme_use('aqua') # Aggiunto per macOS
            else: self.style.theme_use('clam') 
        except tk.TclError: self.style.theme_use('default')

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Variabili di stato
        self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_date_str = None
        self.last_spia_singoli = None; self.last_spia_ambi = None; self.last_spia_terni = None
        self.last_spia_data_fine_analisi = None; self.last_spia_numeri_input = None

        # --- Input File/URL ---
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Origine Dati Estrazioni", padding="10")
        self.file_frame.pack(fill=tk.X, pady=5)
        self.file_path_var = tk.StringVar(value=DEFAULT_10ELOTTO_DATA_URL)
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=65) # Definito self.file_entry
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_button = ttk.Button(self.file_frame, text="Sfoglia...", command=self.browse_file) # Definito self.browse_button
        self.browse_button.pack(side=tk.LEFT)

        # --- Contenitore Parametri ---
        self.params_container = ttk.Frame(self.main_frame)
        self.params_container.pack(fill=tk.X, pady=5)
        self.params_container.columnconfigure(0, weight=1)
        self.params_container.columnconfigure(1, weight=1)
        
        # --- Colonna Sinistra: Parametri Dati e Previsione ML ---
        self.data_params_frame = ttk.LabelFrame(self.params_container, text="Parametri Dati e Previsione ML", padding="10")
        self.data_params_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        # Date
        ttk.Label(self.data_params_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        default_start_10e = datetime.now() - pd.Timedelta(days=120)
        if HAS_TKCALENDAR: 
            self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd') # Definito self.start_date_entry
            try: self.start_date_entry.set_date(default_start_10e)
            except ValueError: self.start_date_entry.set_date(datetime.now() - pd.Timedelta(days=60))
        else: 
            self.start_date_entry = ttk.Entry(self.data_params_frame, width=12) # Definito self.start_date_entry
            self.start_date_entry.insert(0, default_start_10e.strftime('%Y-%m-%d'))
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(self.data_params_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        if HAS_TKCALENDAR: 
            self.end_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd') # Definito self.end_date_entry
            self.end_date_entry.set_date(datetime.now())
        else: 
            self.end_date_entry = ttk.Entry(self.data_params_frame, width=12) # Definito self.end_date_entry
            self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # SeqLen, NumPredict (ML)
        ttk.Label(self.data_params_frame, text="Seq. Input (ML):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.seq_len_var = tk.StringVar(value="5")
        self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=2, to=50, textvariable=self.seq_len_var, width=5, wrap=True, state='readonly') # Definito self.seq_len_entry
        self.seq_len_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(self.data_params_frame, text="Numeri Prev. (ML):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="5")
        self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=1, to=10, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly') # Definito self.num_predict_spinbox
        self.num_predict_spinbox.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)

        # --- Colonna Destra: Parametri Modello ML e Training ---
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello ML e Training", padding="10")
        self.model_params_frame.grid(row=0, column=1, padx=(5,0), pady=5, sticky="nsew")
        self.model_params_frame.columnconfigure(1, weight=1) # Permette agli Entry di espandersi
        
        _cur_row_mp = 0 

        # Hidden Layers
        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.hidden_layers_var = tk.StringVar(value="256,128")
        self.hidden_layers_entry = ttk.Entry(self.model_params_frame, textvariable=self.hidden_layers_var, width=20) # Definito self.hidden_layers_entry
        self.hidden_layers_entry.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.EW)
        _cur_row_mp += 1

        # Loss Function
        ttk.Label(self.model_params_frame, text="Loss Function:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.loss_var = tk.StringVar(value='binary_crossentropy')
        self.loss_combo = ttk.Combobox(self.model_params_frame, textvariable=self.loss_var, width=18, state='readonly', values=['binary_crossentropy', 'mse', 'mae']) # Definito self.loss_combo
        self.loss_combo.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.EW)
        _cur_row_mp += 1
        
        # Optimizer
        ttk.Label(self.model_params_frame, text="Optimizer:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value='adam')
        self.optimizer_combo = ttk.Combobox(self.model_params_frame, textvariable=self.optimizer_var, width=18, state='readonly', values=['adam', 'rmsprop', 'sgd', 'adagrad']) # Definito self.optimizer_combo
        self.optimizer_combo.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.EW)
        _cur_row_mp += 1

        # Dropout
        ttk.Label(self.model_params_frame, text="Dropout (0-0.8):").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.dropout_var = tk.StringVar(value="0.3")
        self.dropout_spinbox = ttk.Spinbox(self.model_params_frame, from_=0.0, to=0.8, increment=0.05, format="%.2f", textvariable=self.dropout_var, width=5, state='readonly', wrap=True) # Definito self.dropout_spinbox
        self.dropout_spinbox.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1

        # L1 Reg
        ttk.Label(self.model_params_frame, text="L1 Reg:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.l1_var = tk.StringVar(value="0.00")
        self.l1_entry = ttk.Entry(self.model_params_frame, textvariable=self.l1_var, width=6) # Definito self.l1_entry
        self.l1_entry.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1

        # L2 Reg
        ttk.Label(self.model_params_frame, text="L2 Reg:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.l2_var = tk.StringVar(value="0.00")
        self.l2_entry = ttk.Entry(self.model_params_frame, textvariable=self.l2_var, width=6) # Definito self.l2_entry
        self.l2_entry.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1
        
        # Max Epoche
        ttk.Label(self.model_params_frame, text="Max Epoche:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="30")
        self.epochs_spinbox = ttk.Spinbox(self.model_params_frame, from_=10, to=500, increment=10, textvariable=self.epochs_var, width=5, state='readonly', wrap=True) # Definito self.epochs_spinbox
        self.epochs_spinbox.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1

        # Batch Size
        ttk.Label(self.model_params_frame, text="Batch Size:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="32")
        self.batch_size_combo = ttk.Combobox(self.model_params_frame, textvariable=self.batch_size_var, values=[str(2**i) for i in range(4,9)], width=5, state='readonly') # Definito self.batch_size_combo
        self.batch_size_combo.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1

        # ES Patience
        ttk.Label(self.model_params_frame, text="ES Patience:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.patience_var = tk.StringVar(value="10")
        self.patience_spinbox = ttk.Spinbox(self.model_params_frame, from_=3, to=50, textvariable=self.patience_var, width=4, state='readonly', wrap=True) # Definito self.patience_spinbox
        self.patience_spinbox.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1
        
        # ES Min Delta
        ttk.Label(self.model_params_frame, text="ES Min Delta:").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.min_delta_var = tk.StringVar(value="0.0005")
        self.min_delta_entry = ttk.Entry(self.model_params_frame, textvariable=self.min_delta_var, width=8) # Definito self.min_delta_entry
        self.min_delta_entry.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1

        # CV Splits
        ttk.Label(self.model_params_frame, text="CV Splits (>=2):").grid(row=_cur_row_mp, column=0, padx=5, pady=2, sticky=tk.W)
        self.cv_splits_var = tk.StringVar(value=str(DEFAULT_CV_SPLITS))
        self.cv_splits_spinbox = ttk.Spinbox(self.model_params_frame, from_=2, to=10, textvariable=self.cv_splits_var, width=4, state='readonly', wrap=True) # Definito self.cv_splits_spinbox
        self.cv_splits_spinbox.grid(row=_cur_row_mp, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_mp += 1

        # --- Pulsanti Azione ML ---
        self.action_frame = ttk.Frame(self.main_frame)
        self.action_frame.pack(pady=5, fill=tk.X)
        self.run_button = ttk.Button(self.action_frame, text="Avvia Analisi ML", command=self.start_analysis_thread) # Definito self.run_button
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.check_button = ttk.Button(self.action_frame, text="Verifica Prev. ML", command=self.start_check_thread, state=tk.DISABLED) # Definito self.check_button
        self.check_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.action_frame, text="Colpi Verifica ML:").pack(side=tk.LEFT, padx=(10,2))
        self.check_colpi_var = tk.StringVar(value=str(DEFAULT_10ELOTTO_CHECK_COLPI))
        self.check_colpi_spinbox = ttk.Spinbox(self.action_frame, from_=1, to=50, textvariable=self.check_colpi_var, width=4, state='readonly', wrap=True) # Definito self.check_colpi_spinbox
        self.check_colpi_spinbox.pack(side=tk.LEFT)

        # --- Risultati Previsione ML ---
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Risultato Previsione ML", padding="10")
        self.results_frame.pack(fill=tk.X, pady=5)
        self.result_label_var = tk.StringVar(value="I numeri ML appariranno qui...")
        self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier', 14, 'bold'), foreground='darkblue')
        self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value="")
        self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 9, 'italic'))
        self.attendibilita_label.pack(pady=2, fill=tk.X)

        # --- Frame e Controlli Analisi Numeri Spia ---
        self.spia_frame = ttk.LabelFrame(self.main_frame, text="Analisi Numeri Spia (su periodo selezionato)", padding="10")
        self.spia_frame.pack(fill=tk.X, pady=(10,5), padx=0)
        
        spia_params_r1 = ttk.Frame(self.spia_frame) 
        spia_params_r1.pack(fill=tk.X, pady=2)
        ttk.Label(spia_params_r1, text="Numeri Spia (es: 7,23):").pack(side=tk.LEFT, padx=(0,5))
        self.numeri_spia_var = tk.StringVar(value="7,23") 
        self.numeri_spia_entry = ttk.Entry(spia_params_r1, textvariable=self.numeri_spia_var, width=15) # Definito self.numeri_spia_entry
        self.numeri_spia_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_params_r1, text="Colpi An. Post-Spia:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_colpi_var = tk.StringVar(value="3") 
        self.spia_colpi_spinbox = ttk.Spinbox(spia_params_r1, from_=1, to=10, textvariable=self.spia_colpi_var, width=4, state='readonly', wrap=True) # Definito self.spia_colpi_spinbox
        self.spia_colpi_spinbox.pack(side=tk.LEFT, padx=(0,10))
        
        spia_params_r2 = ttk.Frame(self.spia_frame) 
        spia_params_r2.pack(fill=tk.X, pady=2)
        ttk.Label(spia_params_r2, text="Top N Singoli:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_top_n_singoli_var = tk.StringVar(value="10")
        self.spia_top_n_singoli_spinbox = ttk.Spinbox(spia_params_r2, from_=1, to=20, textvariable=self.spia_top_n_singoli_var, width=4, state='readonly', wrap=True) # Definito self.spia_top_n_singoli_spinbox
        self.spia_top_n_singoli_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_params_r2, text="Top N Ambi:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_top_n_ambi_var = tk.StringVar(value="5")
        self.spia_top_n_ambi_spinbox = ttk.Spinbox(spia_params_r2, from_=1, to=10, textvariable=self.spia_top_n_ambi_var, width=4, state='readonly', wrap=True) # Definito self.spia_top_n_ambi_spinbox
        self.spia_top_n_ambi_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_params_r2, text="Top N Terni:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_top_n_terni_var = tk.StringVar(value="3")
        self.spia_top_n_terni_spinbox = ttk.Spinbox(spia_params_r2, from_=1, to=5, textvariable=self.spia_top_n_terni_var, width=4, state='readonly', wrap=True) # Definito self.spia_top_n_terni_spinbox
        self.spia_top_n_terni_spinbox.pack(side=tk.LEFT, padx=(0,10))

        spia_actions = ttk.Frame(self.spia_frame) 
        spia_actions.pack(fill=tk.X, pady=(5,2))
        self.run_spia_button = ttk.Button(spia_actions, text="Avvia Analisi Spie", command=self.start_analisi_spia_thread) # Definito self.run_spia_button
        self.run_spia_button.pack(side=tk.LEFT, padx=(0,10))
        self.check_spia_button = ttk.Button(spia_actions, text="Verifica Ris. Spia", command=self.start_verifica_spia_thread, state=tk.DISABLED) # Definito self.check_spia_button
        self.check_spia_button.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_actions, text="Colpi Verifica Spia:").pack(side=tk.LEFT, padx=(10,2))
        self.check_spia_colpi_var = tk.StringVar(value=str(DEFAULT_SPIA_CHECK_COLPI))
        self.check_spia_colpi_spinbox = ttk.Spinbox(spia_actions, from_=1, to=20, textvariable=self.check_spia_colpi_var, width=4, state='readonly', wrap=True) # Definito self.check_spia_colpi_spinbox
        self.check_spia_colpi_spinbox.pack(side=tk.LEFT)

        # --- Log Area ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 10)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='#E8E8E8', foreground='black')
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Label Ultimo Aggiornamento ---
        self.last_update_label_var = tk.StringVar(value="Ultimo aggiornamento estrazionale: N/A")
        self.last_update_label = ttk.Label(self.main_frame, textvariable=self.last_update_label_var, font=('Helvetica', 9, 'italic'))
        self.last_update_label.pack(pady=(5,0), anchor='w')

        # --- Threading Safety ---
        self.analysis_thread = None; self.check_thread = None; self.spia_thread = None; self.verifica_spia_thread = None
        self._stop_event_analysis = threading.Event(); self._stop_event_check = threading.Event()
        self._stop_event_spia = threading.Event(); self._stop_event_verifica_spia = threading.Event()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def browse_file(self):
        filepath = filedialog.askopenfilename(title="Seleziona file estrazioni (.txt)", filetypes=(("Text files", "*.txt"),("All files", "*.*")))
        if filepath: self.file_path_var.set(filepath); self.log_message_gui(f"File locale selezionato: {filepath}")

    def log_message_gui(self, message): log_message(message, self.log_text, self.root)

    def set_result(self, numbers, attendibilita):
        self.root.after(0, self._update_result_labels, numbers, attendibilita)

    def _update_result_labels(self, numbers, attendibilita):
        try:
            if not self.result_label.winfo_exists(): return # Check widget existence
            if numbers and isinstance(numbers, list) and all(isinstance(n, int) for n in numbers):
                 self.result_label_var.set("  ".join(map(lambda x: f"{x:02d}", numbers)))
                 self.log_message_gui("\n" + "="*20 + " PREVISIONE ML GENERATA " + "="*20)
            else:
                self.result_label_var.set("Previsione ML fallita. Controlla log.")
            self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "N/A")
        except tk.TclError: pass # Widget might be destroyed

    def set_controls_state(self, state): self.root.after(0, self._set_controls_state_tk, state)


    def _set_controls_state_tk(self, state):
        try:
            if not self.root.winfo_exists(): 
                return

            # Lista dei widget da abilitare/disabilitare
            # Assicurati che i nomi qui corrispondano ESATTAMENTE agli attributi self.nome_widget creati in __init__
            widgets_to_toggle = [
                self.browse_button, self.file_entry,
                self.seq_len_entry, self.num_predict_spinbox,
                
                # Parametri del Modello ML
                self.hidden_layers_entry, 
                self.loss_combo, 
                self.optimizer_combo,
                self.dropout_spinbox, 
                self.l1_entry, 
                self.l2_entry,
                self.epochs_spinbox, 
                self.batch_size_combo,
                self.patience_spinbox, 
                self.min_delta_entry, 
                self.cv_splits_spinbox,
                
                # Pulsanti Azione ML
                self.run_button, self.check_button, self.check_colpi_spinbox,
                
                # Widget Spia
                self.numeri_spia_entry, self.spia_colpi_spinbox,
                self.spia_top_n_singoli_spinbox, self.spia_top_n_ambi_spinbox, self.spia_top_n_terni_spinbox,
                self.run_spia_button,
                self.check_spia_button, self.check_spia_colpi_spinbox
            ]

            # Gestione separata per DateEntry di tkcalendar o Entry normali
            date_entries_to_handle = []
            if HAS_TKCALENDAR:
                if hasattr(self, 'start_date_entry') and isinstance(self.start_date_entry, DateEntry):
                    date_entries_to_handle.append(self.start_date_entry)
                if hasattr(self, 'end_date_entry') and isinstance(self.end_date_entry, DateEntry):
                    date_entries_to_handle.append(self.end_date_entry)
            else: # Se non usi tkcalendar, trattali come Entry normali
                if hasattr(self, 'start_date_entry'): widgets_to_toggle.append(self.start_date_entry)
                if hasattr(self, 'end_date_entry'): widgets_to_toggle.append(self.end_date_entry)


            # Determina se un qualsiasi thread è attivo
            is_analysis_running = self.analysis_thread and self.analysis_thread.is_alive()
            is_check_running = self.check_thread and self.check_thread.is_alive()
            is_spia_running = self.spia_thread and self.spia_thread.is_alive()
            is_verifica_spia_running = self.verifica_spia_thread and self.verifica_spia_thread.is_alive()
            
            is_any_thread_running = is_analysis_running or is_check_running or is_spia_running or is_verifica_spia_running

            for widget in widgets_to_toggle:
                if widget is None or not hasattr(widget, 'winfo_exists') or not widget.winfo_exists():
                    # print(f"DEBUG: Widget saltato o non esistente: {widget}") # Utile per debug
                    continue
                
                widget_state_to_set = state # Stato base richiesto (tk.NORMAL o tk.DISABLED)

                # Logica specifica per disabilitare pulsanti se altri thread sono attivi
                if is_any_thread_running and state == tk.NORMAL: # Se un thread è attivo e si sta cercando di abilitare i controlli
                    if widget == self.run_button and (is_check_running or is_spia_running or is_verifica_spia_running):
                         widget_state_to_set = tk.DISABLED
                    elif widget == self.check_button and (is_analysis_running or is_spia_running or is_verifica_spia_running):
                         widget_state_to_set = tk.DISABLED
                    elif widget == self.run_spia_button and (is_analysis_running or is_check_running or is_verifica_spia_running):
                         widget_state_to_set = tk.DISABLED
                    elif widget == self.check_spia_button and (is_analysis_running or is_check_running or is_spia_running):
                         widget_state_to_set = tk.DISABLED
                    # Altrimenti, se un thread è attivo, tutti gli altri controlli (non pulsanti di azione) dovrebbero essere disabilitati
                    elif widget not in [self.run_button, self.check_button, self.run_spia_button, self.check_spia_button]:
                        widget_state_to_set = tk.DISABLED


                # Logica per abilitare/disabilitare pulsanti di VERIFICA basata sulla presenza di risultati
                # Questa logica si applica solo se widget_state_to_set è ancora tk.NORMAL
                if widget == self.check_button and widget_state_to_set == tk.NORMAL:
                     if self.last_prediction is None: # Se non c'è una previsione ML salvata
                         widget_state_to_set = tk.DISABLED
                
                if widget == self.check_spia_button and widget_state_to_set == tk.NORMAL:
                     if not (self.last_spia_singoli or self.last_spia_ambi or self.last_spia_terni): # Se non ci sono risultati spia
                         widget_state_to_set = tk.DISABLED
                
                # Conversione a stato Tkinter effettivo
                target_tk_state = tk.DISABLED # Default
                if widget_state_to_set == tk.NORMAL:
                    if isinstance(widget, (ttk.Combobox, ttk.Spinbox)):
                        target_tk_state = 'readonly'
                    elif isinstance(widget, (ttk.Entry, scrolledtext.ScrolledText)): # Aggiunto ScrolledText per il log
                        target_tk_state = tk.NORMAL
                    elif isinstance(widget, ttk.Button):
                        target_tk_state = tk.NORMAL
                
                try:
                    current_widget_state = widget.cget('state')
                    if str(current_widget_state).lower() != str(target_tk_state).lower():
                        widget.config(state=target_tk_state)
                except (tk.TclError, AttributeError):
                    # self.log_message_gui(f"Attenzione: Impossibile impostare stato per widget {type(widget)}")
                    pass # Ignora se il widget è già distrutto o non ha 'state'

            # Gestione DateEntry di tkcalendar
            for date_widget in date_entries_to_handle:
               if date_widget and hasattr(date_widget, 'winfo_exists') and date_widget.winfo_exists():
                   try:
                       # Le DateEntry si abilitano solo se nessun thread è attivo E lo stato richiesto è NORMAL
                       target_date_state = tk.NORMAL if state == tk.NORMAL and not is_any_thread_running else tk.DISABLED
                       current_date_widget_state = date_widget.cget('state')
                       if str(current_date_widget_state).lower() != str(target_date_state).lower():
                           date_widget.configure(state=target_date_state) # Usa configure per DateEntry
                   except (tk.TclError, AttributeError):
                       # self.log_message_gui(f"Attenzione: Impossibile impostare stato per DateEntry {date_widget}")
                       pass

        except tk.TclError: # Probabile durante chiusura
            pass
        except Exception as e_set_state_generic:
            print(f"Errore imprevisto in _set_controls_state_tk: {e_set_state_generic}\n{traceback.format_exc()}")


    # --- Metodi per Analisi Machine Learning ---
    def start_analysis_thread(self):
        if any(t and t.is_alive() for t in [self.analysis_thread, self.check_thread, self.spia_thread, self.verifica_spia_thread]):
            messagebox.showwarning("Operazione in Corso", "Attendere termine operazione corrente.", parent=self.root); return

        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi ML in corso..."); self.attendibilita_label_var.set("")
        self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_date_str = None
        
        # Recupero e Validazione Parametri (MOLTO SINTETIZZATO - assicurati che la tua logica sia completa)
        try:
            data_source = self.file_path_var.get().strip()
            start_date_str, end_date_str = self._get_spia_dates() # Riutilizza per ML
            if not data_source or not start_date_str: return # Errore già mostrato
            seq_len = int(self.seq_len_var.get()); num_pred = int(self.num_predict_var.get())
            h_layers = [int(x.strip()) for x in self.hidden_layers_var.get().split(',') if x.strip()]
            loss = self.loss_var.get(); opt = self.optimizer_var.get(); drop = float(self.dropout_var.get())
            l1 = float(self.l1_var.get()); l2 = float(self.l2_var.get()); epochs = int(self.epochs_var.get())
            batch = int(self.batch_size_var.get()); pat = int(self.patience_var.get()); m_delta = float(self.min_delta_var.get())
            cv_s = int(self.cv_splits_var.get())
            # Aggiungere validazioni più robuste qui
        except Exception as e_param: messagebox.showerror("Errore Parametri ML", f"Parametri non validi: {e_param}", parent=self.root); return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui("=== Avvio Analisi ML (FE & CV) ===")
        self._stop_event_analysis.clear()
        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(data_source, start_date_str, end_date_str, seq_len, loss, opt, drop, l1, l2,
                  h_layers, epochs, batch, pat, m_delta, num_pred, cv_s, self._stop_event_analysis),
            daemon=True, name="AnalysisThreadML" )
        self.analysis_thread.start()

    def run_analysis(self, data_source, start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions, n_cv_splits, stop_event):
        numeri_predetti, attendibilita_msg = None, "Analisi ML non completata"
        try:
            df_full, _, _, _ = carica_dati_10elotto(data_source, log_callback=None) # Per data max
            if df_full is not None and not df_full.empty: self.root.after(0, self.last_update_label_var.set, f"Ultimo agg.: {df_full['Data'].max().strftime('%Y-%m-%d')}")

            numeri_predetti, attendibilita_msg = analisi_10elotto(
                data_source, start_date, end_date, sequence_length, loss_function, optimizer, 
                dropout_rate, l1_reg, l2_reg, hidden_layers_config, max_epochs, batch_size, 
                patience, min_delta, num_predictions, n_cv_splits, self.log_message_gui, stop_event)
            
            if isinstance(numeri_predetti, list):
                self.last_prediction = numeri_predetti
                self.last_prediction_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                self.last_prediction_date_str = end_date
        except Exception as e:
            self.log_message_gui(f"ERRORE CRITICO run_analysis ML: {e}\n{traceback.format_exc()}")
            attendibilita_msg = f"Errore ML: {e}"
        finally:
            self.set_result(numeri_predetti, attendibilita_msg)
            self.set_controls_state(tk.NORMAL)
            self.root.after(10, self._clear_analysis_thread_ref)

    def _clear_analysis_thread_ref(self): self.analysis_thread = None; self._set_controls_state_tk(tk.NORMAL)

    # --- Metodi per Verifica ML ---
    def start_check_thread(self):
        if any(t and t.is_alive() for t in [self.analysis_thread, self.check_thread, self.spia_thread, self.verifica_spia_thread]):
            messagebox.showwarning("Operazione in Corso", "Attendere.", parent=self.root); return
        if self.last_prediction is None or self.last_prediction_date_str is None:
            messagebox.showinfo("Nessuna Previsione ML", "Eseguire prima un'analisi ML.", parent=self.root); return
        try: num_colpi = int(self.check_colpi_var.get()); assert 1 <= num_colpi <= 50
        except: messagebox.showerror("Errore", "Colpi verifica ML non validi (1-50).", parent=self.root); return
        
        data_source = self.file_path_var.get().strip() # Assicurati che sia valido
        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Previsione ML ({num_colpi} Colpi) ===")
        self._stop_event_check.clear()
        self.check_thread = threading.Thread(target=self.run_check_results, 
            args=(data_source, self.last_prediction, self.last_prediction_date_str, num_colpi, self._stop_event_check), 
            daemon=True, name="CheckThreadML")
        self.check_thread.start()

    def run_check_results(self, data_source, prediction_to_check, last_analysis_date_str, num_colpi_to_check, stop_event):
        try:
            start_verify_date = (datetime.strptime(last_analysis_date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            df_check, arr_check, _, _ = carica_dati_10elotto(data_source, start_date=start_verify_date, log_callback=self.log_message_gui)
            if arr_check is None or len(arr_check) == 0:
                self.log_message_gui(f"Nessuna estrazione ML per verifica dopo {last_analysis_date_str}."); return

            num_to_run = min(num_colpi_to_check, len(arr_check))
            pred_set = set(prediction_to_check); found_total = 0; max_hits = 0
            for i in range(num_to_run):
                if stop_event.is_set(): break
                hits = pred_set.intersection(set(arr_check[i]))
                draw_date = df_check.iloc[i]['Data'].strftime('%Y-%m-%d') if i < len(df_check) else "N/D"
                if hits: found_total+=1; max_hits=max(max_hits, len(hits)); self.log_message_gui(f"Colpo ML {i+1:02d} ({draw_date}): {len(hits)} Punti -> {sorted(list(hits))}")
                else: self.log_message_gui(f"Colpo ML {i+1:02d} ({draw_date}): Nessun risultato.")
            if not stop_event.is_set(): self.log_message_gui(f"Verifica ML: {found_total} colpi con risultati. Max Punti: {max_hits}.")
        except Exception as e: self.log_message_gui(f"ERRORE verifica ML: {e}\n{traceback.format_exc()}")
        finally:
            self.set_controls_state(tk.NORMAL)
            self.root.after(10, self._clear_check_thread_ref)

    def _clear_check_thread_ref(self): self.check_thread = None; self._set_controls_state_tk(tk.NORMAL)


    # --- Metodi per Analisi Numeri Spia ---
    def _get_spia_dates(self): 
        try:
            s_date = self.start_date_entry.get_date().strftime('%Y-%m-%d') if HAS_TKCALENDAR else self.start_date_entry.get()
            e_date = self.end_date_entry.get_date().strftime('%Y-%m-%d') if HAS_TKCALENDAR else self.end_date_entry.get()
            datetime.strptime(s_date, '%Y-%m-%d'); datetime.strptime(e_date, '%Y-%m-%d') # Valida formato
            return s_date, e_date
        except Exception as e: messagebox.showerror("Errore Data", f"Date non valide: {e}", parent=self.root); return None, None

    def _validate_spia_params(self, data_source, numeri_spia_str, start_date, end_date, colpi_an_str, top_s_str, top_a_str, top_t_str):
        errors = {'messages': [], 'parsed': {}}
        if not data_source or (not data_source.startswith("http") and not os.path.exists(data_source)): errors['messages'].append("Sorgente dati non valida.")
        if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'): errors['messages'].append("Data Inizio Spia > Data Fine Spia.")
        try: errors['parsed']['numeri_spia'] = [int(n.strip()) for n in numeri_spia_str.split(',') if n.strip() and 1 <= int(n.strip()) <= 90]; assert errors['parsed']['numeri_spia']
        except: errors['messages'].append("Numeri Spia non validi (1-90, separati da ',').")
        try: errors['parsed']['colpi_analizzare'] = int(colpi_an_str); assert 1 <= errors['parsed']['colpi_analizzare'] <= 10
        except: errors['messages'].append("Colpi Analisi Spia (1-10).")
        try: errors['parsed']['top_n_singoli'] = int(top_s_str); assert 1 <= errors['parsed']['top_n_singoli'] <= 20
        except: errors['messages'].append("Top N Singoli (1-20).")
        try: errors['parsed']['top_n_ambi'] = int(top_a_str); assert 1 <= errors['parsed']['top_n_ambi'] <= 10
        except: errors['messages'].append("Top N Ambi (1-10).")
        try: errors['parsed']['top_n_terni'] = int(top_t_str); assert 1 <= errors['parsed']['top_n_terni'] <= 5
        except: errors['messages'].append("Top N Terni (1-5).")
        return errors if errors['messages'] else None

    # All'interno della classe App10eLotto

    def start_analisi_spia_thread(self):
        if any(t and t.is_alive() for t in [self.spia_thread, self.analysis_thread, self.check_thread, self.verifica_spia_thread]):
            messagebox.showwarning("Operazione in Corso", "Un'altra operazione è già in esecuzione. Attendere.", parent=self.root)
            return

        s_date, e_date = self._get_spia_dates()
        if not s_date: # Se _get_spia_dates ha restituito (None, None) a causa di un errore
            return
        
        data_source_val = self.file_path_var.get().strip()
        numeri_spia_str_val = self.numeri_spia_var.get().strip()
        colpi_an_str_val = self.spia_colpi_var.get()
        top_s_str_val = self.spia_top_n_singoli_var.get()
        top_a_str_val = self.spia_top_n_ambi_var.get()
        top_t_str_val = self.spia_top_n_terni_var.get()

        validation_result = self._validate_spia_params(
            data_source_val, numeri_spia_str_val, 
            s_date, e_date, 
            colpi_an_str_val, top_s_str_val, 
            top_a_str_val, top_t_str_val
        )
        
        if validation_result and validation_result['messages']: # Se ci sono messaggi di errore
            messagebox.showerror("Errore Parametri Analisi Spia", "\n\n".join(validation_result['messages']), parent=self.root)
            return
        
        # Se validation_result è None (nessun errore), allora i parametri sono validi
        # e possiamo procedere a prenderli dai widget e convertirli.
        # Oppure, se validation_result non è None ma messages è vuoto, usiamo i 'parsed'
        
        params_parsed = {}
        if validation_result and 'parsed' in validation_result: # Questo caso non dovrebbe succedere se messages è vuoto
            params_parsed = validation_result['parsed']
        else: # Nessun errore di validazione, quindi _validate_spia_params ha restituito None. Dobbiamo parsare qui.
            try:
                params_parsed['numeri_spia'] = [int(x.strip()) for x in numeri_spia_str_val.split(',') if x.strip()]
                params_parsed['colpi_analizzare'] = int(colpi_an_str_val)
                params_parsed['top_n_singoli'] = int(top_s_str_val)
                params_parsed['top_n_ambi'] = int(top_a_str_val)
                params_parsed['top_n_terni'] = int(top_t_str_val)
                # Aggiungi qui controlli di range se necessario, anche se _validate_spia_params dovrebbe averli già fatti
                if not params_parsed['numeri_spia']: raise ValueError("Lista numeri spia vuota dopo parsing")

            except ValueError as e_parse:
                 messagebox.showerror("Errore Interno", f"Errore conversione parametri spia: {e_parse}", parent=self.root)
                 return

        # Ora usiamo params_parsed
        p = params_parsed # Alias per brevità
        
        self.last_spia_singoli = None
        self.last_spia_ambi = None
        self.last_spia_terni = None
        self.last_spia_data_fine_analisi = None
        self.last_spia_numeri_input = p['numeri_spia'] 
        
        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Analisi Numeri Spia (Spia: {p['numeri_spia']}, Periodo: {s_date}-{e_date}) ===")
        self.log_message_gui(f"Sorgente Dati: {'URL' if data_source_val.startswith('http') else 'File Locale'} ({os.path.basename(data_source_val)})")
        self.log_message_gui(f"Colpi successivi da analizzare: {p['colpi_analizzare']}")
        self.log_message_gui(f"Parametri Top N: Singoli={p['top_n_singoli']}, Ambi={p['top_n_ambi']}, Terni={p['top_n_terni']}")
        self.log_message_gui("-" * 50)

        self._stop_event_spia.clear()
        self.spia_thread = threading.Thread(target=self.run_analisi_spia, 
            args=(data_source_val, s_date, e_date, p['numeri_spia'], p['colpi_analizzare'], 
                  p['top_n_singoli'], p['top_n_ambi'], p['top_n_terni'], self._stop_event_spia), 
            daemon=True, name="SpiaAnalysisThread")
        self.spia_thread.start()

    # All'interno della classe App10eLotto

    def run_analisi_spia(self, data_source, start_date_str, end_date_str, 
                         numeri_spia_list, colpi_an, 
                         top_s, top_a, top_t, stop_event):
        try:
            if stop_event.is_set():
                self.log_message_gui("Analisi Spia: Annullata prima dell'inizio.")
                return

            self.log_message_gui(f"Analisi Spia: Caricamento dati per il periodo {start_date_str} - {end_date_str}...")
            df_spia_original, arr_spia, _, _ = carica_dati_10elotto( # Rinomino df_spia_original per chiarezza
                data_source,
                start_date=start_date_str, 
                end_date=end_date_str,
                log_callback=self.log_message_gui
            )

            if stop_event.is_set():
                self.log_message_gui("Analisi Spia: Annullata dopo caricamento dati.")
                return

            if df_spia_original is None or df_spia_original.empty or arr_spia is None or len(arr_spia) == 0:
                self.log_message_gui(f"ERRORE (Spia): Dati storici insufficienti o non caricati per il periodo {start_date_str} - {end_date_str}.")
                # Assicurati che i risultati spia siano None se i dati non sono caricati
                self.last_spia_singoli = None; self.last_spia_ambi = None; self.last_spia_terni = None
                self.last_spia_data_fine_analisi = None
                return
            
            # Allineamento date per calcola_numeri_spia
            # Dobbiamo usare un DataFrame che sia stato pulito ESATTAMENTE come arr_spia
            # per ottenere le date corrette.
            date_arr_spia_aligned = None
            try:
                # df_spia_original è il df DOPO il filtro date utente, ma PRIMA del dropna sui numeri per creare arr_spia.
                # Dobbiamo replicare la pulizia dei numeri su df_spia_original per allineare le date con arr_spia.
                df_temp_for_alignment = df_spia_original.copy()
                numeri_cols_check = [f'Num{i+1}' for i in range(20)]

                # Verifica se le colonne dei numeri esistono prima di tentare la conversione
                if not all(col in df_temp_for_alignment.columns for col in numeri_cols_check):
                    self.log_message_gui("ERRORE (Spia): Colonne Numeri (Num1-20) mancanti nel DataFrame per allineamento date.")
                    # Gestisci come errore dati, resetta risultati
                    self.last_spia_singoli = None; self.last_spia_ambi = None; self.last_spia_terni = None
                    self.last_spia_data_fine_analisi = None
                    return

                for col in numeri_cols_check:
                    df_temp_for_alignment[col] = pd.to_numeric(df_temp_for_alignment[col], errors='coerce')
                
                df_aligned_with_arr_spia = df_temp_for_alignment.dropna(subset=numeri_cols_check).copy()
                
                if len(df_aligned_with_arr_spia) == len(arr_spia):
                    if 'Data' in df_aligned_with_arr_spia.columns:
                        date_arr_spia_aligned = df_aligned_with_arr_spia['Data'].values
                    else:
                         self.log_message_gui("ATTENZIONE (Spia): Colonna 'Data' non trovata nel DataFrame allineato. Il log del periodo potrebbe essere impreciso.")
                else:
                    self.log_message_gui(f"ERRORE CRITICO (Spia): Disallineamento tra numeri ({len(arr_spia)}) e date ({len(df_aligned_with_arr_spia)}) dopo la pulizia. Impossibile procedere con date precise.")
                    # Procedi senza date precise o gestisci come errore
            except Exception as e_align:
                 self.log_message_gui(f"ERRORE (Spia): Problema durante allineamento date con arr_spia: {e_align}")
                 # Potresti decidere di procedere con date_arr_spia_aligned = None

            self.log_message_gui("Analisi Spia: Inizio calcolo frequenze numeri/ambi/terni spiati...")
            res_s, res_a, res_t, num_occ, d_ini_scan, d_fin_scan = calcola_numeri_spia(
                arr_spia, date_arr_spia_aligned, numeri_spia_list, colpi_an, 
                top_s, top_a, top_t, 
                self.log_message_gui, stop_event
            )

            if stop_event.is_set():
                self.log_message_gui("Analisi Spia: Elaborazione terminata a causa di richiesta di stop.")
                return

            self.log_message_gui("-" * 40 + "\nRISULTATI ANALISI NUMERI SPIA (Spia: " + str(self.last_spia_numeri_input) + "):" + "-" * 40)
            
            # Usa le date iniziali e finali del range fornito dall'utente se quelle scansionate non sono disponibili
            periodo_effettivo_usato_inizio = d_ini_scan if d_ini_scan not in [None, "N/A"] else start_date_str
            periodo_effettivo_usato_fine = d_fin_scan if d_fin_scan not in [None, "N/A"] else end_date_str
            
            self.log_message_gui(f"Periodo analizzato: {periodo_effettivo_usato_inizio} - {periodo_effettivo_usato_fine}")
            self.log_message_gui(f"Occorrenze dei numeri spia (come combinazione) trovate: {num_occ}")
            self.log_message_gui(f"Analizzati {colpi_an} colpi successivi per ogni occorrenza.")

            if num_occ > 0:
                # Salva i risultati per la verifica (SOLO i numeri/ambi/terni, non le frequenze)
                # Converti esplicitamente a int standard Python per evitare problemi di serializzazione o tipo con np.int64
                self.last_spia_singoli = [int(s_val[0]) for s_val in res_s] if res_s else []
                self.last_spia_ambi = [tuple(map(int, a_val[0])) for a_val in res_a] if res_a else []
                self.last_spia_terni = [tuple(map(int, t_val[0])) for t_val in res_t] if res_t else []
                self.last_spia_data_fine_analisi = periodo_effettivo_usato_fine 
                
                if res_s:
                    self.log_message_gui(f"\nTop {len(res_s)} SINGOLI spiati (Numero: Frequenza):")
                    for numero, frequenza in res_s:
                        self.log_message_gui(f"  - N. {int(numero):02d}: uscito {frequenza} volte")
                else:
                    self.log_message_gui("\nNessun singolo spia trovato con frequenza > 0 (esclusi gli spia).")
                
                if res_a:
                    self.log_message_gui(f"\nTop {len(res_a)} AMBI spiati (Ambo: Frequenza):")
                    for ambo_tuple, frequenza in res_a:
                        ambo_str = ", ".join(f"{int(n):02d}" for n in ambo_tuple)
                        self.log_message_gui(f"  - Ambo ({ambo_str}): uscito {frequenza} volte")
                else:
                    self.log_message_gui("\nNessun ambo spia trovato con frequenza > 0 (esclusi gli spia).")

                if res_t:
                    self.log_message_gui(f"\nTop {len(res_t)} TERNI spiati (Terno: Frequenza):")
                    for terno_tuple, frequenza in res_t:
                        terno_str = ", ".join(f"{int(n):02d}" for n in terno_tuple)
                        self.log_message_gui(f"  - Terno ({terno_str}): uscito {frequenza} volte")
                else:
                    self.log_message_gui("\nNessun terno spia trovato con frequenza > 0 (esclusi gli spia).")
                
                if not res_s and not res_a and not res_t: # Se num_occ > 0 ma nessun risultato significativo
                    self.log_message_gui("\nNessun risultato spia (singolo, ambo o terno) significativo trovato nonostante le occorrenze spia.")

            elif not (stop_event and stop_event.is_set()): # num_occ == 0 e non interrotto
                self.log_message_gui(f"Nessuna occorrenza dei numeri spia {self.last_spia_numeri_input} trovata nel periodo selezionato.")
                self.last_spia_singoli = None; self.last_spia_ambi = None; self.last_spia_terni = None
                self.last_spia_data_fine_analisi = None
           
        except Exception as e:
            self.log_message_gui(f"\nERRORE CRITICO durante l'analisi dei Numeri Spia: {e}")
            self.log_message_gui(traceback.format_exc())
            # Resetta i risultati spia in caso di errore grave
            self.last_spia_singoli = None; self.last_spia_ambi = None; self.last_spia_terni = None
            self.last_spia_data_fine_analisi = None
        finally:
            # Log completamento solo se non interrotto e se il thread esiste ancora
            if not (stop_event and stop_event.is_set()) and hasattr(self, 'spia_thread') and self.spia_thread is not None : 
                self.log_message_gui("="*15 + " Analisi Numeri Spia Completata " + "="*15 + "\n")
            
            # Riabilita controlli e pulisci riferimento al thread
            self.set_controls_state(tk.NORMAL) 
            self.root.after(10, self._clear_spia_thread_ref)

    def _clear_spia_thread_ref(self): self.spia_thread = None; self._set_controls_state_tk(tk.NORMAL)

    # --- Metodi per Verifica Risultati Spia ---
    def start_verifica_spia_thread(self):
        if any(t and t.is_alive() for t in [self.analysis_thread, self.check_thread, self.spia_thread, self.verifica_spia_thread]):
            messagebox.showwarning("Operazione in Corso", "Attendere.", parent=self.root); return
        if not (self.last_spia_singoli or self.last_spia_ambi or self.last_spia_terni) or not self.last_spia_data_fine_analisi:
            messagebox.showinfo("Nessun Risultato Spia", "Eseguire prima Analisi Spia.", parent=self.root); return
        try: num_colpi = int(self.check_spia_colpi_var.get()); assert 1 <= num_colpi <= 20
        except: messagebox.showerror("Errore", "Colpi verifica spia non validi (1-20).", parent=self.root); return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Risultati Spia ({num_colpi} Colpi) ===")
        self._stop_event_verifica_spia.clear()
        self.verifica_spia_thread = threading.Thread(target=self.run_verifica_spia_results,
            args=(self.file_path_var.get(), self.last_spia_data_fine_analisi, num_colpi, self._stop_event_verifica_spia),
            daemon=True, name="VerificaSpiaThread")
        self.verifica_spia_thread.start()

    # All'interno della classe App10eLotto

    def run_verifica_spia_results(self, data_source, data_fine_analisi_spia_str, num_colpi_da_verificare, stop_event):
        try:
            try:
                last_date_obj = datetime.strptime(data_fine_analisi_spia_str, '%Y-%m-%d')
                # La verifica inizia dal giorno *successivo* alla fine dell'analisi spia
                check_start_date_obj = last_date_obj + timedelta(days=1)
                check_start_date_str = check_start_date_obj.strftime('%Y-%m-%d')
            except ValueError as ve:
                self.log_message_gui(f"ERRORE formato data fine analisi spia '{data_fine_analisi_spia_str}': {ve}")
                return # Esce se la data non è valida

            if stop_event.is_set():
                self.log_message_gui("Verifica Spia: Annullata prima caricamento dati.")
                return

            self.log_message_gui(f"Verifica Spia: Caricamento dati per verifica da {check_start_date_str} in avanti...")
            # Carica i dati per la verifica, senza data di fine per prendere tutto il disponibile dopo check_start_date_str
            df_check, numeri_array_check, _, _ = carica_dati_10elotto(
                data_source, 
                start_date=check_start_date_str, 
                end_date=None, # Prende tutte le estrazioni successive
                log_callback=self.log_message_gui
            )

            if stop_event.is_set(): 
                self.log_message_gui("Verifica Spia: Annullata dopo caricamento dati.")
                return
            
            if df_check is None or df_check.empty or numeri_array_check is None or len(numeri_array_check) == 0:
                self.log_message_gui(f"Verifica Spia: Nessuna estrazione trovata dopo {data_fine_analisi_spia_str} (a partire da {check_start_date_str}) o caricamento fallito.")
                return
            
            num_estrazioni_disponibili = len(numeri_array_check)
            num_colpi_effettivi_verifica = min(num_colpi_da_verificare, num_estrazioni_disponibili)
            
            if num_colpi_effettivi_verifica == 0 :
                 self.log_message_gui(f"Verifica Spia: Nessuna estrazione disponibile per la verifica dopo {data_fine_analisi_spia_str}.")
                 return

            self.log_message_gui(f"Verifica Spia: Trovate {num_estrazioni_disponibili} estrazioni successive. Verifico le prossime {num_colpi_effettivi_verifica}...");

            # Contatori per i successi
            colpi_con_hit_singoli = 0 # Numero di colpi in cui almeno un singolo spia è uscito
            max_singoli_per_colpo = 0 # Massimo numero di singoli spia usciti in un singolo colpo
            
            colpi_con_hit_ambi = 0    # Numero di colpi in cui almeno un ambo spia è uscito
            max_ambi_per_colpo = 0    # Massimo numero di ambi spia usciti in un singolo colpo
            
            colpi_con_hit_terni = 0   # Numero di colpi in cui almeno un terno spia è uscito
            max_terni_per_colpo = 0   # Massimo numero di terni spia usciti in un singolo colpo

            for i in range(num_colpi_effettivi_verifica):
                if stop_event.is_set():
                    self.log_message_gui(f"Verifica Spia: Interrotta al colpo {i+1}.")
                    break
                
                estrazione_attuale_set = set(numeri_array_check[i])
                # Ottieni la data dell'estrazione attuale per il log
                data_estrazione_attuale_str = "N/D"
                if 'Data' in df_check.columns and i < len(df_check) and pd.notna(df_check.iloc[i]['Data']):
                    try:
                        data_estrazione_attuale_str = pd.to_datetime(df_check.iloc[i]['Data']).strftime('%Y-%m-%d')
                    except Exception: # NOSONAR
                        pass # Lascia N/D se c'è errore nel formato della data specifica
                
                log_colpo_header = f"Colpo Spia {i+1:02d} ({data_estrazione_attuale_str}):"
                found_in_this_draw_overall = False # Flag per vedere se c'è stato ALMENO un hit in questo colpo

                # Verifica Singoli
                if self.last_spia_singoli: # Se ci sono singoli da verificare
                    indovinati_singoli_in_draw = [s for s in self.last_spia_singoli if s in estrazione_attuale_set]
                    if indovinati_singoli_in_draw: # Se almeno un singolo è stato trovato
                        self.log_message_gui(f"{log_colpo_header} SINGOLI: {len(indovinati_singoli_in_draw)}/{len(self.last_spia_singoli)} trovati -> {sorted(indovinati_singoli_in_draw)}")
                        colpi_con_hit_singoli +=1 
                        max_singoli_per_colpo = max(max_singoli_per_colpo, len(indovinati_singoli_in_draw))
                        found_in_this_draw_overall = True
                
                # Verifica Ambi
                if self.last_spia_ambi: # Se ci sono ambi da verificare
                    indovinati_ambi_in_draw = [tuple(sorted(ambo)) for ambo in self.last_spia_ambi if set(ambo).issubset(estrazione_attuale_set)]
                    if indovinati_ambi_in_draw: # Se almeno un ambo è stato trovato
                        self.log_message_gui(f"{log_colpo_header} AMBI: {len(indovinati_ambi_in_draw)}/{len(self.last_spia_ambi)} trovati -> {sorted(indovinati_ambi_in_draw)}")
                        colpi_con_hit_ambi +=1
                        max_ambi_per_colpo = max(max_ambi_per_colpo, len(indovinati_ambi_in_draw))
                        found_in_this_draw_overall = True

                # Verifica Terni
                if self.last_spia_terni: # Se ci sono terni da verificare
                    indovinati_terni_in_draw = [tuple(sorted(terno)) for terno in self.last_spia_terni if set(terno).issubset(estrazione_attuale_set)]
                    if indovinati_terni_in_draw: # Se almeno un terno è stato trovato
                        self.log_message_gui(f"{log_colpo_header} TERNI: {len(indovinati_terni_in_draw)}/{len(self.last_spia_terni)} trovati -> {sorted(indovinati_terni_in_draw)}")
                        colpi_con_hit_terni +=1
                        max_terni_per_colpo = max(max_terni_per_colpo, len(indovinati_terni_in_draw))
                        found_in_this_draw_overall = True
                
                if not found_in_this_draw_overall: # Se non c'è nessun tipo di hit in questa estrazione
                     self.log_message_gui(f"{log_colpo_header} Nessun risultato spia trovato.")


            if not stop_event.is_set(): # Solo se non interrotto
                self.log_message_gui("-" * 40)
                self.log_message_gui(f"Verifica Risultati Spia ({num_colpi_effettivi_verifica} colpi) Riepilogo:")
                if self.last_spia_singoli:
                    self.log_message_gui(f"  SINGOLI: Uscite in {colpi_con_hit_singoli} colpi. Max singoli indovinati per colpo: {max_singoli_per_colpo} (su {len(self.last_spia_singoli)} proposti).")
                if self.last_spia_ambi:
                    self.log_message_gui(f"  AMBI: Uscite in {colpi_con_hit_ambi} colpi. Max ambi indovinati per colpo: {max_ambi_per_colpo} (su {len(self.last_spia_ambi)} proposti).")
                if self.last_spia_terni:
                    self.log_message_gui(f"  TERNI: Uscite in {colpi_con_hit_terni} colpi. Max terni indovinati per colpo: {max_terni_per_colpo} (su {len(self.last_spia_terni)} proposti).")
                
                if not (colpi_con_hit_singoli or colpi_con_hit_ambi or colpi_con_hit_terni):
                    self.log_message_gui("  Nessun risultato spia (singolo, ambo o terno) trovato nei colpi verificati.")

        except Exception as e:
            self.log_message_gui(f"ERRORE CRITICO durante la verifica dei risultati spia: {e}\n{traceback.format_exc()}")
        finally:
            if not stop_event.is_set():
                self.log_message_gui("\n=== Verifica Risultati Spia Completata ===")
            self.set_controls_state(tk.NORMAL)
            self.root.after(10, self._clear_verifica_spia_thread_ref)

    def _clear_verifica_spia_thread_ref(self): self.verifica_spia_thread = None; self._set_controls_state_tk(tk.NORMAL)

    def on_close(self):
        self.log_message_gui("Richiesta chiusura finestra...")
        for event in [self._stop_event_analysis, self._stop_event_check, self._stop_event_spia, self._stop_event_verifica_spia]: event.set()
        
        active_threads = [t for t in [self.analysis_thread, self.check_thread, self.spia_thread, self.verifica_spia_thread] if t and t.is_alive()]
        if active_threads:
            self.log_message_gui(f"Attendo terminazione thread: {[t.name for t in active_threads]} (max 3s)")
            for t in active_threads: t.join(timeout=3.0) # Timeout per thread
        
        if self.root and self.root.winfo_exists(): self.root.destroy()

# --- Funzione di Lancio ---
def launch_10elotto_window(parent_window):
    try:
        lotto_win = tk.Toplevel(parent_window)
        App10eLotto(lotto_win) # Istanzia l'app
        lotto_win.lift(); lotto_win.focus_force()
    except Exception as e:
         messagebox.showerror("Errore Avvio Modulo 10eLotto", f"Errore: {e}\n{traceback.format_exc()}", parent=parent_window)

# --- Blocco Esecuzione Standalone ---
if __name__ == "__main__":
    print("Esecuzione di elotto_module.py in modalità standalone...")
    try:
        if sys.platform == "win32" and hasattr(windll, 'shcore'): from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass # Ignora se non funziona (es. non Windows)
    
    root_standalone = tk.Tk()
    app_standalone = App10eLotto(root_standalone)
    root_standalone.mainloop()