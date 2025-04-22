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
import threading # Già presente
import traceback
import requests
import time      # Aggiunto per il timeout del join
import queue     # Aggiunto per possibile miglioramento log (ma non usato nella soluzione principale)

try:
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

DEFAULT_10ELOTTO_CHECK_COLPI = 1
DEFAULT_10ELOTTO_DATA_URL = "https://raw.githubusercontent.com/illottodimax/Archivio/main/it-10elotto-past-draws-archive.txt"

# --- Funzioni Globali (Seed, Log - INVARIATE) ---
def set_seed(seed_value=42):
    """Imposta il seme per la riproducibilità dei risultati."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_seed()

def log_message(message, log_widget, window):
    """Aggiunge un messaggio al widget di log nella GUI in modo sicuro per i thread."""
    if log_widget and window:
        # Usa after per eseguire l'aggiornamento nel thread principale di Tkinter
        window.after(10, lambda: _update_log_widget(log_widget, message))

def _update_log_widget(log_widget, message):
    """Funzione helper per aggiornare il widget di log."""
    try:
        # Verifica se il widget esiste ancora prima di modificarlo
        if not log_widget.winfo_exists():
             print(f"Log GUI widget destroyed, message lost: {message}")
             return

        current_state = log_widget.cget('state')
        log_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, str(message) + "\n")
        log_widget.see(tk.END)
        # Ripristina lo stato solo se era DISABLED, altrimenti lascialo NORMAL
        # Questo evita problemi se viene chiamato rapidamente più volte
        if current_state == tk.DISABLED:
             log_widget.config(state=tk.DISABLED)
    except tk.TclError as e:
        # Questo può accadere se la finestra viene chiusa nel mezzo di un 'after'
        print(f"Log GUI TclError (likely during shutdown): {e} - Message: {message}")
    except Exception as e:
        print(f"Log GUI unexpected error: {e}\nMessage: {message}")
        # Tentativo di ripristinare lo stato disabilitato in caso di errore
        try:
            if log_widget.winfo_exists() and log_widget.cget('state') == tk.NORMAL:
                 log_widget.config(state=tk.DISABLED)
        except:
            pass # Ignora errori durante il recupero dall'errore


# --- Funzioni Specifiche 10eLotto (INVARIATE) ---
def carica_dati_10elotto(data_source, start_date=None, end_date=None, log_callback=None):
    """
    Carica i dati del 10eLotto da un URL (RAW GitHub) o da un file locale.
    Gestisce la colonna vuota extra.
    """
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
            # Caricamento da file locale
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

        # --- Parsing (Logica quasi invariata, opera su 'lines') ---
        if log_callback: log_callback(f"Lette {len(lines)} righe totali dalla fonte dati.")
        if not lines or len(lines) < 2:
            if log_callback: log_callback("ERRORE: Dati vuoti o solo intestazione.")
            return None, None, None, None

        data_lines = lines[1:]
        data = []; malformed_lines = 0; min_expected_cols = 24 # Data + 20 Num + Vuota + Oro1 + Oro2
        for i, line in enumerate(data_lines):
            # Usa rstrip() per rimuovere solo spazi/newline a destra prima dello split
            values = line.rstrip().split('\t')
            if len(values) >= min_expected_cols:
                data.append(values)
            else:
                malformed_lines += 1
                # Log solo le prime righe scartate per evitare spam
                if malformed_lines < 5 and log_callback: log_callback(f"ATT: Riga {i+2} scartata (campi < {min_expected_cols}, trovati {len(values)}): '{line.strip()}'")
        if malformed_lines > 0 and log_callback: log_callback(f"ATTENZIONE: {malformed_lines} righe totali scartate (poche colonne).")
        if not data:
            if log_callback: log_callback("ERRORE: Nessuna riga dati valida trovata dopo il parsing iniziale.")
            return None, None, None, None

        max_cols = max(len(row) for row in data)
        # Crea nomi colonne dinamicamente ma con i nomi noti per i primi 24/25
        colonne_note = ['Data'] + [f'Num{i+1}' for i in range(20)] + ['ColonnaVuota', 'Oro1', 'Oro2']
        colonne_finali = list(colonne_note) # Copia
        # Aggiungi colonne extra se ce ne sono
        if max_cols > len(colonne_note):
             colonne_finali.extend([f'ExtraCol{i}' for i in range(len(colonne_note), max_cols)])

        df = pd.DataFrame(data, columns=colonne_finali[:max_cols]) # Applica nomi colonne
        if log_callback: log_callback(f"Creato DataFrame shape: {df.shape}, Colonne: {df.columns.tolist()}")

        # Rimuovi la colonna vuota se esiste
        if 'ColonnaVuota' in df.columns:
             df = df.drop(columns=['ColonnaVuota'])
             if log_callback: log_callback("Rimossa 'ColonnaVuota'.")

        # --- Pulizia e filtraggio (logica invariata) ---
        if 'Data' not in df.columns: log_callback("ERRORE: Colonna 'Data' mancante."); return None, None, None, None
        df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d', errors='coerce')
        original_rows = len(df); df = df.dropna(subset=['Data'])
        if original_rows - len(df) > 0 and log_callback: log_callback(f"Rimosse {original_rows - len(df)} righe (data non valida).")
        if df.empty: log_callback("ERRORE: Nessun dato dopo pulizia date."); return df, None, None, None # Ritorna df vuoto
        df = df.sort_values(by='Data', ascending=True) # Ordina per data

        if start_date:
            try: start_dt = pd.to_datetime(start_date); df = df[df['Data'] >= start_dt]
            except Exception as e: log_callback(f"Errore filtro data inizio: {e}")
        if end_date:
             try: end_dt = pd.to_datetime(end_date); df = df[df['Data'] <= end_dt]
             except Exception as e: log_callback(f"Errore filtro data fine: {e}")
        if log_callback: log_callback(f"Righe dopo filtro date ({start_date} - {end_date}): {len(df)}")

        numeri_cols = [f'Num{i+1}' for i in range(20)]; numeri_array, numeri_oro, numeri_extra = None, None, None
        if not df.empty:
            df_cleaned = df.copy() # Lavora su una copia per la pulizia numerica
            if not all(col in df_cleaned.columns for col in numeri_cols): log_callback(f"ERRORE: Colonne Num1-20 mancanti."); return df.copy(), None, None, None
            try:
                for col in numeri_cols: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                rows_b4 = len(df_cleaned); df_cleaned = df_cleaned.dropna(subset=numeri_cols); dropped = rows_b4 - len(df_cleaned)
                if dropped > 0 and log_callback: log_callback(f"Scartate {dropped} righe (Num1-20 non numerici).")
                if not df_cleaned.empty: numeri_array = df_cleaned[numeri_cols].values.astype(int)
                else: log_callback("ATTENZIONE: Nessuna riga dopo pulizia Num1-20.")
            except Exception as e: log_callback(f"ERRORE pulizia Num1-20: {e}")

            # Estrai Oro SOLO se numeri_array è stato estratto (dalle righe valide)
            if numeri_array is not None and 'Oro1' in df_cleaned.columns and 'Oro2' in df_cleaned.columns:
                try:
                    df_cleaned['Oro1'] = pd.to_numeric(df_cleaned['Oro1'], errors='coerce'); df_cleaned['Oro2'] = pd.to_numeric(df_cleaned['Oro2'], errors='coerce')
                    df_cleaned_oro = df_cleaned.dropna(subset=['Oro1', 'Oro2']) # Usa una var temporanea per non perdere righe per Extra
                    if not df_cleaned_oro.empty:
                         numeri_oro = df_cleaned_oro[['Oro1', 'Oro2']].values.astype(int)
                         if log_callback: log_callback(f"Estratto array Oro. Shape: {numeri_oro.shape}")
                    else: numeri_oro = None
                except Exception as e: log_callback(f"Errore pulizia/estrazione Oro: {e}")

            # Estrai Extra dalle righe dove Num1-20 sono validi (df_cleaned)
            if numeri_array is not None and any(col.startswith('ExtraCol') for col in df_cleaned.columns):
                 extra_cols = [col for col in df_cleaned.columns if col.startswith('ExtraCol')]
                 if extra_cols:
                      numeri_extra = df_cleaned[extra_cols[0]].values # Prendi la prima colonna extra
                      if log_callback: log_callback(f"Estratta colonna Extra '{extra_cols[0]}'.")
                 else:
                      numeri_extra = None

        final_rows_df = len(df) # Righe nel df originale dopo filtro date
        final_rows_arr = len(numeri_array) if numeri_array is not None else 0 # Righe nell'array pulito
        if log_callback: log_callback(f"Caricamento/Filtraggio completato. Righe df finali: {final_rows_df}, Righe array numeri: {final_rows_arr}")

        # Ritorna il df originale filtrato per data e gli array numerici puliti
        return df.copy(), numeri_array, numeri_oro, numeri_extra
    except Exception as e:
        if log_callback: log_callback(f"Errore grave carica_dati_10elotto V2: {e}\n{traceback.format_exc()}");
        return None, None, None, None

# --- Funzioni prepara_sequenze, build_model, LogCallback, genera_previsione (INVARIATE) ---
def prepara_sequenze_per_modello(numeri_array, sequence_length=5, log_callback=None):
    if numeri_array is None or len(numeri_array) == 0:
        if log_callback: log_callback("ERRORE (prep_seq): No input array.")
        return None, None
    if numeri_array.ndim != 2 or numeri_array.shape[1] != 20:
        if log_callback: log_callback(f"ERRORE (prep_seq): Array numeri non ha 20 colonne (shape: {numeri_array.shape}).")
        return None, None
    X, y = [], []; num_estrazioni = len(numeri_array)
    if log_callback: log_callback(f"Preparazione sequenze 10eLotto: seq={sequence_length}, estrazioni={num_estrazioni}.")
    if num_estrazioni <= sequence_length:
        if log_callback: log_callback(f"ERRORE: Estrazioni ({num_estrazioni}) <= seq ({sequence_length}). Necessarie {sequence_length + 1}.")
        return None, None
    valid_seq, invalid_tgt = 0, 0
    for i in range(num_estrazioni - sequence_length):
        in_seq = numeri_array[i:i+sequence_length]
        tgt_extr = numeri_array[i+sequence_length]
        mask = (tgt_extr >= 1) & (tgt_extr <= 90)
        if np.all(mask):
             target = np.zeros(90, dtype=int); target[tgt_extr - 1] = 1
             X.append(in_seq.flatten()); y.append(target); valid_seq += 1
        else:
             invalid_tgt += 1
             if invalid_tgt < 5 and log_callback: log_callback(f"ATT: Scartata seq indice {i} (target non valido: {tgt_extr[~mask]})")
    if invalid_tgt > 0 and log_callback: log_callback(f"Scartate {invalid_tgt} sequenze totali (target non valido).")
    if not X:
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata."); return None, None
    if log_callback: log_callback(f"Create {valid_seq} sequenze 10eLotto valide.")
    try:
        X_np = np.array(X); y_np = np.array(y)
        if log_callback: log_callback(f"Shape finale: X={X_np.shape}, y={y_np.shape}")
        return X_np, y_np
    except Exception as e:
        if log_callback: log_callback(f"ERRORE conversione NumPy: {e}"); return None, None

def build_model_10elotto(input_shape, hidden_layers=[512, 256, 128], loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    if not isinstance(input_shape, tuple) or len(input_shape) != 1 or not isinstance(input_shape[0], int) or input_shape[0] <= 0:
        if log_callback: log_callback(f"ERRORE build_model: input_shape '{input_shape}' non valido. Deve essere tipo (num_features,)."); return None
    if log_callback: log_callback(f"Costruzione modello 10eLotto: Input={input_shape}, L={hidden_layers}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")
    model = tf.keras.Sequential(name="Modello_10eLotto")
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input_Layer"))
    reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg + l2_reg > 0 else None
    if not hidden_layers: log_callback("ATT: No hidden layers.")
    else:
        for i, units in enumerate(hidden_layers):
            if not isinstance(units, int) or units <= 0:
                if log_callback: log_callback(f"ERR: Unità layer {i+1} non valida ({units})."); return None
            model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f"Dense_{i+1}"))
            model.add(tf.keras.layers.BatchNormalization(name=f"BN_{i+1}"))
            if dropout_rate > 0:
                actual_dropout = max(0.0, min(dropout_rate, 0.99))
                model.add(tf.keras.layers.Dropout(actual_dropout, name=f"Drop_{i+1}_{actual_dropout:.2f}"))
    model.add(tf.keras.layers.Dense(90, activation='sigmoid', name="Output_Layer_90_Sigmoid"))
    try:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        if log_callback: log_callback("Modello 10eLotto compilato.")
    except Exception as e:
        if log_callback: log_callback(f"ERR compilazione modello 10eLotto: {e}"); return None
    return model

class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_callback_func, stop_event=None): # Aggiunto stop_event opzionale
         super().__init__()
         self.log_callback_func = log_callback_func
         self.stop_event = stop_event # Salva riferimento a stop_event
         self._is_running = True # Mantenuto per logica interna
    def stop_logging(self): self._is_running = False # Potrebbe non essere più necessario se si usa stop_event
    def on_epoch_end(self, epoch, logs=None):
        # Controlla se il thread esterno ha richiesto lo stop
        if self.stop_event and self.stop_event.is_set():
            self.model.stop_training = True # Segnala a Keras di fermarsi
            if self.log_callback_func: self.log_callback_func(f"Epoca {epoch+1}: Richiesta di stop ricevuta, arresto training...")
            return # Esce subito

        if not self._is_running or not self.log_callback_func: return
        logs = logs or {}; msg = f"Epoca {epoch+1:03d} - "; items = [f"{k.replace('_',' ').replace('val ','v_')}: {v:.4f}" for k, v in logs.items()]
        self.log_callback_func(msg + ", ".join(items))
    # Opzionale: potresti aggiungere on_batch_end per controlli più frequenti
    # def on_batch_end(self, batch, logs=None):
    #     if self.stop_event and self.stop_event.is_set():
    #         self.model.stop_training = True

def genera_previsione_10elotto(model, X_input, num_predictions=10, log_callback=None):
    if log_callback: log_callback(f"Generazione previsione 10eLotto per {num_predictions} numeri...")
    if model is None: log_callback("ERR (genera_prev_10eLotto): Modello non valido."); return None
    if X_input is None or X_input.size == 0: log_callback("ERR (genera_prev_10eLotto): Input vuoto."); return None
    if not (1 <= num_predictions <= 90): log_callback(f"ERR (genera_prev_10eLotto): num_predictions={num_predictions} non valido (1-90)."); return None
    try:
        input_reshaped = None
        if X_input.ndim == 1: input_reshaped = X_input.reshape(1, -1)
        elif X_input.ndim == 2 and X_input.shape[0] == 1: input_reshaped = X_input
        else:
            if X_input.ndim >= 2 and X_input.shape[0] > 0:
                input_reshaped = X_input[0].reshape(1, -1)
                if log_callback: log_callback(f"ATT: Ricevuto input 10eLotto shape {X_input.shape}, usata prima riga.")
            else:
                if log_callback: log_callback(f"ERRORE: Shape input 10eLotto non gestita: {X_input.shape}"); return None
        try:
            if hasattr(model, 'input_shape') and model.input_shape is not None: expected_input_features = model.input_shape[-1]
            elif hasattr(model, 'layers') and model.layers and hasattr(model.layers[0], 'input_shape'): expected_input_features = model.layers[0].input_shape[-1]
            else: expected_input_features = None
            if expected_input_features is not None and input_reshaped.shape[1] != expected_input_features:
                log_callback(f"ERRORE Shape Input 10eLotto: Input({input_reshaped.shape[1]}) != Modello({expected_input_features})."); return None
        except Exception as e_shape:
            if log_callback: log_callback(f"ATT: Eccezione verifica input_shape modello 10eLotto: {e_shape}")
        pred_probabilities = model.predict(input_reshaped, verbose=0)
        if pred_probabilities is None or pred_probabilities.size == 0: log_callback("ERR: predict() ha restituito vuoto."); return None
        if pred_probabilities.ndim != 2 or pred_probabilities.shape[0] != 1 or pred_probabilities.shape[1] != 90:
             log_callback(f"ERRORE: Output shape da predict inatteso: {pred_probabilities.shape}. Atteso (1, 90)."); return None
        probs_vector = pred_probabilities[0]
        sorted_indices = np.argsort(probs_vector); top_indices_ascending_prob = sorted_indices[-num_predictions:]
        top_indices_descending_prob = top_indices_ascending_prob[::-1]; predicted_numbers_by_prob = [int(index + 1) for index in top_indices_descending_prob]
        if log_callback: log_callback(f"Numeri 10eLotto predetti ({len(predicted_numbers_by_prob)} ord. per probabilità decr.): {predicted_numbers_by_prob}")
        return predicted_numbers_by_prob
    except Exception as e:
        if log_callback: log_callback(f"ERRORE CRITICO generazione previsione 10eLotto: {e}\n{traceback.format_exc()}"); return None

# --- Funzione Analisi Principale (INVARIATA) ---
def analisi_10elotto(file_path, start_date, end_date, sequence_length=5,
                     loss_function='binary_crossentropy', optimizer='adam',
                     dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                     hidden_layers_config=[512, 256, 128],
                     max_epochs=100, batch_size=32, patience=15, min_delta=0.0001,
                     num_predictions=10,
                     log_callback=None,
                     stop_event=None): # Aggiunto stop_event opzionale
    """
    Analizza i dati del 10 e Lotto (da URL o file) e genera previsioni.
    Aggiunto controllo stop_event.
    """
    if log_callback:
        source_type = "URL" if file_path.startswith("http") else "File Locale"
        source_name = os.path.basename(file_path) if source_type == "File Locale" else file_path
        log_callback(f"=== Avvio Analisi 10eLotto Dettagliata ===")
        log_callback(f"Sorgente: {source_type} ({source_name}), Date: {start_date}-{end_date}, SeqIn: {sequence_length}, NumOut: {num_predictions}")
        log_callback(f"Modello: L={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")
        log_callback(f"Training: Epochs={max_epochs}, Batch={batch_size}, Pat={patience}, MinDelta={min_delta}")
        log_callback(f"---------------------------------")

    # Controllo stop prima di iniziare
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima dell'inizio."); return None, "Analisi annullata"

    # 1. Carica dati
    df, numeri_array, _, _ = carica_dati_10elotto(file_path, start_date, end_date, log_callback=log_callback)
    if df is None: return None, "Caricamento dati fallito (df None)"
    if numeri_array is None :
        msg = "Nessun dato numerico valido (Num1-20) trovato dopo caricamento/pulizia."; log_callback(f"ERRORE: {msg}"); return None, msg
    if len(numeri_array) < sequence_length + 1:
        msg = f"ERRORE: Dati numerici insuff. ({len(numeri_array)}) per seq_len ({sequence_length}). Servono almeno {sequence_length + 1} estrazioni."; log_callback(msg); return None, msg

    # Controllo stop dopo caricamento
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo caricamento dati."); return None, "Analisi annullata"

    # 2. Prepara sequenze
    X, y = None, None
    try:
        X, y = prepara_sequenze_per_modello(numeri_array, sequence_length, log_callback=log_callback)
        if X is None or y is None or len(X) == 0: return None, "Creazione sequenze fallita o nessuna sequenza valida."
        min_samples_for_split = 5; min_samples_for_train = 2
        if len(X) < min_samples_for_train: msg = f"ERRORE: Troppi pochi campioni ({len(X)} < {min_samples_for_train}) per addestrare il modello 10eLotto."; log_callback(msg); return None, msg
        elif len(X) < min_samples_for_split: log_callback(f"ATTENZIONE: Solo {len(X)} campioni disponibili (< {min_samples_for_split}). Training su tutti i dati senza validation.")
    except Exception as e: log_callback(f"Errore preparazione sequenze 10eLotto: {e}\n{traceback.format_exc()}"); return None, f"Errore prep sequenze: {e}"

    # Controllo stop dopo preparazione sequenze
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo preparazione sequenze."); return None, "Analisi annullata"

    # 3. Normalizza Input
    try: X_scaled = X.astype(np.float32) / 90.0; log_callback(f"Input X normalizzato (diviso per 90). Shape: {X_scaled.shape}")
    except Exception as e_scale: log_callback(f"ERRORE normalizzazione input X 10eLotto: {e_scale}"); return None, "Errore normalizzazione dati input"

    # 4. Split train/validation
    X_train, X_val, y_train, y_val = None, None, None, None; split_ratio = 0.8
    if len(X_scaled) >= min_samples_for_split:
        try:
            split_idx = max(1, min(int(split_ratio * len(X_scaled)), len(X_scaled) - 1))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            if len(X_train) == 0 or len(X_val) == 0 or len(y_train) == 0 or len(y_val) == 0:
                 log_callback(f"ERRORE: Split train/val fallito (set vuoti). Fallback a training su tutti i dati.")
                 X_train, y_train = X_scaled, y; X_val, y_val = None, None
            elif log_callback: log_callback(f"Dati divisi: {len(X_train)} train, {len(X_val)} validation.")
        except Exception as e_split: log_callback(f"ERRORE split train/validation 10eLotto: {e_split}. Fallback a training su tutti i dati."); X_train, y_train = X_scaled, y; X_val, y_val = None, None
    else: log_callback(f"INFO: Training 10eLotto su tutti i dati (campioni < {min_samples_for_split}). Nessun set di validazione."); X_train, y_train = X_scaled, y; X_val, y_val = None, None

    # Controllo stop prima del training
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima del training."); return None, "Analisi annullata"

    # 5. Costruisci e addestra il modello
    model, history, tf_log_callback_obj = None, None, None
    final_val_loss, final_train_loss_at_best = float('inf'), float('inf')
    try:
        tf.keras.backend.clear_session()
        if X_train is None or X_train.size == 0 or X_train.ndim != 2 or X_train.shape[1] == 0: log_callback(f"ERRORE CRITICO: Dati training 10eLotto (X_train) non validi. Shape: {X_train.shape if X_train is not None else 'None'}"); return None, "Errore dati training"
        input_shape = (X_train.shape[1],)
        model = build_model_10elotto(input_shape, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if model is None: return None, "Costruzione modello 10eLotto fallita"
        has_validation_data = (X_val is not None and X_val.size > 0 and y_val is not None and y_val.size > 0)
        monitor = 'val_loss' if has_validation_data else 'loss'
        log_callback(f"Monitoraggio EarlyStopping 10eLotto: '{monitor}'")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=1)
        tf_log_callback_obj = LogCallback(log_callback, stop_event) # Passa stop_event a LogCallback
        validation_data_fit = (X_val, y_val) if has_validation_data else None
        log_callback(f"Inizio addestramento modello 10eLotto...")
        history = model.fit(X_train, y_train, validation_data=validation_data_fit, epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping, tf_log_callback_obj], verbose=0)
        # Controllo stop subito dopo il fit (se è stato interrotto dal callback)
        if stop_event and stop_event.is_set(): log_callback("Training interrotto durante l'esecuzione."); return None, "Training interrotto"
        if history and history.history:
            epochs = len(history.history.get('loss', [])); train_loss_hist = history.history.get('loss', []); val_loss_hist = history.history.get('val_loss', [])
            log_callback(f"Addestramento 10eLotto terminato: {epochs} epoche.")
            best_idx = -1
            if val_loss_hist: best_idx = np.argmin(val_loss_hist); final_val_loss = val_loss_hist[best_idx]; final_train_loss_at_best = train_loss_hist[best_idx] if best_idx < len(train_loss_hist) else float('inf')
            elif train_loss_hist: best_idx = np.argmin(train_loss_hist); final_train_loss_at_best = train_loss_hist[best_idx]; final_val_loss = float('inf')
            if best_idx != -1 and log_callback: log_callback(f"Miglior epoca 10eLotto: {best_idx+1}, Val Loss: {final_val_loss:.4f}, Train Loss (epoca): {final_train_loss_at_best:.4f}")
        else: log_callback("ATTENZIONE: Addestramento 10eLotto senza history valida (potrebbe essere stato interrotto).")
    except tf.errors.ResourceExhaustedError as e_mem:
         msg = f"ERRORE Memoria 10eLotto: {e_mem}. Riduci batch size/modello."; log_callback(msg); log_callback(traceback.format_exc()); return None, msg
    except Exception as e:
        # Verifica se l'eccezione è dovuta a un'interruzione richiesta
        if stop_event and stop_event.is_set(): log_callback(f"Eccezione durante training probabilmente dovuta a stop richiesto: {e}"); return None, "Training interrotto"
        msg = f"ERRORE CRITICO addestramento 10eLotto: {e}"; log_callback(msg); log_callback(traceback.format_exc()); return None, msg
    finally:
        # Assicurati che il callback di logging venga fermato se necessario
        # if tf_log_callback_obj: tf_log_callback_obj.stop_logging() # Non più necessario se si usa stop_event
        pass

    # Controllo stop prima della previsione
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima della previsione finale."); return None, "Analisi annullata"

    # 6. Prepara input e genera previsione
    numeri_predetti, attendibilita_msg = None, "Attendibilità Non Determinata"
    try:
        log_callback("Preparazione input previsione finale 10eLotto...")
        if numeri_array is None or len(numeri_array) < sequence_length: log_callback("ERRORE: Dati originali insuff. per input previsione 10eLotto."); return None, "Dati insuff per input previsione"
        input_pred_raw = numeri_array[-sequence_length:]
        input_pred_scaled = input_pred_raw.flatten().astype(np.float32) / 90.0
        numeri_predetti = genera_previsione_10elotto(model, input_pred_scaled, num_predictions, log_callback=log_callback)
        if numeri_predetti is None: return None, "Generazione previsione 10eLotto fallita."
        ratio = float('inf')
        if final_train_loss_at_best > 1e-7 and final_val_loss != float('inf'): ratio = final_val_loss / final_train_loss_at_best
        if ratio < 1.2: attend = "Alta"
        elif ratio < 1.8: attend = "Media"
        elif final_val_loss != float('inf'): attend = "Bassa (overfitting?)"
        elif final_train_loss_at_best != float('inf'): attend = "Non Valutabile (solo training)"
        else: attend = "Non Determinabile"
        attendibilita_msg = f"Attendibilità: {attend}" + (f" (Ratio V/T: {ratio:.2f})" if ratio != float('inf') else "")
        log_callback(attendibilita_msg)
        return numeri_predetti, attendibilita_msg
    except Exception as e:
         log_callback(f"Errore CRITICO previsione finale 10eLotto: {e}\n{traceback.format_exc()}"); return None, f"Errore previsione finale: {e}"
# --- Fine Funzione analisi_10elotto ---


# --- Definizione Classe App10eLotto (MODIFICATA per Threading Safety) ---
class App10eLotto:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi e Previsione 10eLotto (v7.1 - Thread Safe)") # Titolo aggiornato
        self.root.geometry("850x910")

        self.style = ttk.Style()
        try:
            if sys.platform == "win32": self.style.theme_use('vista')
            elif sys.platform == "darwin": self.style.theme_use('aqua')
            else: self.style.theme_use('clam')
        except tk.TclError: self.style.theme_use('default')

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.last_prediction = None
        self.last_prediction_end_date = None
        self.last_prediction_date_str = None

        # --- Input File/URL ---
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Origine Dati Estrazioni 10eLotto (URL Raw GitHub o File Locale .txt)", padding="10")
        self.file_frame.pack(fill=tk.X, pady=5)
        self.file_path_var = tk.StringVar(value=DEFAULT_10ELOTTO_DATA_URL)
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=65)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_button = ttk.Button(self.file_frame, text="Sfoglia Locale...", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)

        # --- Contenitore Parametri ---
        self.params_container = ttk.Frame(self.main_frame)
        self.params_container.pack(fill=tk.X, pady=5)
        self.params_container.columnconfigure(0, weight=1)
        self.params_container.columnconfigure(1, weight=1)

        # --- Colonna Sinistra: Parametri Dati ---
        self.data_params_frame = ttk.LabelFrame(self.params_container, text="Parametri Dati", padding="10")
        self.data_params_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        ttk.Label(self.data_params_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        default_start_10e = datetime.now() - pd.Timedelta(days=90)
        if HAS_TKCALENDAR:
             self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd')
             try: self.start_date_entry.set_date(default_start_10e)
             except ValueError: self.start_date_entry.set_date(datetime.now() - pd.Timedelta(days=30))
             self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        else:
            self.start_date_entry = ttk.Entry(self.data_params_frame, width=12); self.start_date_entry.insert(0, default_start_10e.strftime('%Y-%m-%d'))
            self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        if HAS_TKCALENDAR:
             self.end_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd'); self.end_date_entry.set_date(datetime.now())
             self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        else:
            self.end_date_entry = ttk.Entry(self.data_params_frame, width=12); self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
            self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Seq. Input (Storia):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.seq_len_var = tk.StringVar(value="5")
        self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=1, to=50, textvariable=self.seq_len_var, width=5, wrap=True, state='readonly')
        self.seq_len_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Numeri da Prevedere:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="10")
        self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=1, to=10, increment=1, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly')
        self.num_predict_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # --- Colonna Destra: Parametri Modello e Training ---
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello e Training", padding="10")
        self.model_params_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        self.model_params_frame.columnconfigure(1, weight=1)
        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.hidden_layers_var = tk.StringVar(value="512, 256, 128")
        self.hidden_layers_entry = ttk.Entry(self.model_params_frame, textvariable=self.hidden_layers_var, width=25)
        self.hidden_layers_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Loss Function:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.loss_var = tk.StringVar(value='binary_crossentropy')
        self.loss_combo = ttk.Combobox(self.model_params_frame, textvariable=self.loss_var, width=23, state='readonly', values=['binary_crossentropy', 'mse', 'mae', 'huber_loss'])
        self.loss_combo.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Optimizer:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value='adam')
        self.optimizer_combo = ttk.Combobox(self.model_params_frame, textvariable=self.optimizer_var, width=23, state='readonly', values=['adam', 'rmsprop', 'sgd', 'adagrad', 'adamw'])
        self.optimizer_combo.grid(row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Dropout Rate (0-1):").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.dropout_var = tk.StringVar(value="0.35")
        self.dropout_spinbox = ttk.Spinbox(self.model_params_frame, from_=0.0, to=0.8, increment=0.05, format="%.2f", textvariable=self.dropout_var, width=7, wrap=True, state='readonly')
        self.dropout_spinbox.grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="L1 Strength (>=0):").grid(row=4, column=0, padx=5, pady=3, sticky=tk.W)
        self.l1_var = tk.StringVar(value="0.00")
        self.l1_entry = ttk.Entry(self.model_params_frame, textvariable=self.l1_var, width=7)
        self.l1_entry.grid(row=4, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="L2 Strength (>=0):").grid(row=5, column=0, padx=5, pady=3, sticky=tk.W)
        self.l2_var = tk.StringVar(value="0.00")
        self.l2_entry = ttk.Entry(self.model_params_frame, textvariable=self.l2_var, width=7)
        self.l2_entry.grid(row=5, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="Max Epoche:").grid(row=6, column=0, padx=5, pady=3, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="100")
        self.epochs_spinbox = ttk.Spinbox(self.model_params_frame, from_=10, to=1000, increment=10, textvariable=self.epochs_var, width=7, wrap=True, state='readonly')
        self.epochs_spinbox.grid(row=6, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="Batch Size:").grid(row=7, column=0, padx=5, pady=3, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="32")
        batch_values = [str(2**i) for i in range(3, 9)]
        self.batch_size_combo = ttk.Combobox(self.model_params_frame, textvariable=self.batch_size_var, values=batch_values, width=5, state='readonly')
        self.batch_size_combo.grid(row=7, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="ES Patience:").grid(row=8, column=0, padx=5, pady=3, sticky=tk.W)
        self.patience_var = tk.StringVar(value="15")
        self.patience_spinbox = ttk.Spinbox(self.model_params_frame, from_=3, to=100, increment=1, textvariable=self.patience_var, width=7, wrap=True, state='readonly')
        self.patience_spinbox.grid(row=8, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="ES Min Delta:").grid(row=9, column=0, padx=5, pady=3, sticky=tk.W)
        self.min_delta_var = tk.StringVar(value="0.0001")
        self.min_delta_entry = ttk.Entry(self.model_params_frame, textvariable=self.min_delta_var, width=10)
        self.min_delta_entry.grid(row=9, column=1, padx=5, pady=3, sticky=tk.W)

        # --- Pulsanti Azione ---
        self.action_frame = ttk.Frame(self.main_frame)
        self.action_frame.pack(pady=10)
        self.run_button = ttk.Button(self.action_frame, text="Avvia Analisi e Previsione 10eLotto", command=self.start_analysis_thread)
        self.run_button.pack(side=tk.LEFT, padx=10)
        self.check_button = ttk.Button(self.action_frame, text="Verifica Ultima Previsione", command=self.start_check_thread, state=tk.DISABLED)
        self.check_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.action_frame, text="Colpi da Verificare:").pack(side=tk.LEFT, padx=(10, 2))
        self.check_colpi_var = tk.StringVar(value=str(DEFAULT_10ELOTTO_CHECK_COLPI))
        self.check_colpi_spinbox = ttk.Spinbox(self.action_frame, from_=1, to=50, increment=1, textvariable=self.check_colpi_var, width=4, wrap=True, state='readonly')
        self.check_colpi_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # --- Risultati ---
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Risultato Previsione 10eLotto", padding="10")
        self.results_frame.pack(fill=tk.X, pady=5)
        self.result_label_var = tk.StringVar(value="I numeri previsti appariranno qui...")
        self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier', 14, 'bold'), foreground='darkgreen')
        self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value="")
        self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 10, 'italic'))
        self.attendibilita_label.pack(pady=2)

        # --- Log Area ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 9)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='white', foreground='black')
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Label Ultimo Aggiornamento ---
        self.last_update_label_var = tk.StringVar(value="Ultimo aggiornamento estrazionale: N/A")
        self.last_update_label = ttk.Label(self.main_frame, textvariable=self.last_update_label_var, font=('Helvetica', 10, 'italic'))
        self.last_update_label.pack(pady=5)

        # === Modifiche per Threading Safety ===
        self.analysis_thread = None
        self.check_thread = None
        self._stop_event_analysis = threading.Event() # Evento per fermare l'analisi
        self._stop_event_check = threading.Event()    # Evento per fermare la verifica

        # Intercetta la chiusura della finestra
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # =====================================

    # --- Metodi della Classe ---

    def browse_file(self):
        """Apre una finestra di dialogo per selezionare un file LOCALE."""
        filepath = filedialog.askopenfilename( title="Seleziona file estrazioni 10eLotto Locale (.txt)", filetypes=(("Text files", "*.txt"), ("All files", "*.*")) )
        if filepath:
            self.file_path_var.set(filepath)
            self.log_message_gui(f"File 10eLotto locale selezionato: {filepath}")

    def log_message_gui(self, message):
        """Invia messaggio di log alla GUI in modo sicuro."""
        # Usa la funzione globale che gestisce root.after
        log_message(message, self.log_text, self.root)

    def set_result(self, numbers, attendibilita):
        """Aggiorna le etichette dei risultati in modo sicuro."""
        # Usa after per eseguire l'aggiornamento nel thread principale
        self.root.after(0, self._update_result_labels, numbers, attendibilita)

    def _update_result_labels(self, numbers, attendibilita):
        """Funzione helper eseguita da root.after per aggiornare le etichette."""
        try:
            # Verifica se la finestra/widget esistono ancora
            if not self.root.winfo_exists() or not self.result_label.winfo_exists():
                return # Finestra chiusa, non fare nulla

            if numbers and isinstance(numbers, list) and all(isinstance(n, int) for n in numbers):
                 # RIMUOVI sorted() da qui per mantenere l'ordine di probabilità
                 result_str = "  ".join(map(lambda x: f"{x:02d}", numbers)) # Senza sorted()
                 self.result_label_var.set(result_str)
                 self.log_message_gui("\n" + "="*30 + "\nPREVISIONE 10ELOTTO GENERATA (Ord. Probabilità)\n" + "="*30) # Aggiorna log
            else:
                self.result_label_var.set("Previsione 10eLotto fallita. Controlla i log.")
                log_err = True
                if isinstance(attendibilita, str):
                     if "Attendibilità" in attendibilita or "Errore" in attendibilita or "fallita" in attendibilita or "annullata" in attendibilita: log_err = False # Non loggare errori già noti
                if log_err: self.log_message_gui("\nERRORE: Previsione 10eLotto non ha restituito numeri validi o ha fallito.")
            self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "")
        except tk.TclError as e:
            print(f"TclError in _update_result_labels (likely during shutdown): {e}")
        except Exception as e:
            print(f"Error in _update_result_labels: {e}")


    def set_controls_state(self, state):
        """Imposta lo stato dei controlli in modo sicuro."""
        # Usa after per eseguire l'aggiornamento nel thread principale
        self.root.after(0, lambda: self._set_controls_state_tk(state))

    def _set_controls_state_tk(self, state):
        """Funzione helper eseguita da root.after per impostare lo stato."""
        try:
            # Verifica se la finestra esiste ancora
            if not self.root.winfo_exists():
                return

            widgets_to_toggle = [
                self.browse_button, self.file_entry, self.seq_len_entry, self.num_predict_spinbox,
                self.run_button, self.check_button, self.loss_combo, self.optimizer_combo,
                self.dropout_spinbox, self.l1_entry, self.l2_entry, self.hidden_layers_entry,
                self.epochs_spinbox, self.batch_size_combo, self.patience_spinbox, self.min_delta_entry,
                self.check_colpi_spinbox
            ]
            # Gestione DateEntry/Entry
            date_widgets = []
            if HAS_TKCALENDAR and hasattr(self, 'start_date_entry') and hasattr(self.start_date_entry, 'configure'):
                date_widgets.extend([self.start_date_entry, self.end_date_entry])
            elif not HAS_TKCALENDAR and hasattr(self, 'start_date_entry'):
                 widgets_to_toggle.extend([self.start_date_entry, self.end_date_entry])

            # Applica stato ai widget principali
            for widget in widgets_to_toggle:
                 if widget is None or not widget.winfo_exists(): continue # Salta widget non validi/distrutti
                 widget_state = state
                 # Logica speciale per check_button (abilita solo se c'è previsione)
                 if widget == self.check_button and state == tk.NORMAL:
                      if self.last_prediction is None or not isinstance(self.last_prediction, list):
                          widget_state = tk.DISABLED
                 # Disabilita pulsanti se l'altro thread è attivo
                 is_analysis_running = self.analysis_thread and self.analysis_thread.is_alive()
                 is_check_running = self.check_thread and self.check_thread.is_alive()
                 if widget == self.run_button and state == tk.NORMAL and is_check_running:
                      widget_state = tk.DISABLED
                 if widget == self.check_button and state == tk.NORMAL and is_analysis_running:
                      widget_state = tk.DISABLED

                 # Imposta lo stato effettivo
                 try:
                     current_widget_state = widget.cget('state')
                     target_tk_state = tk.NORMAL if widget_state == tk.NORMAL else tk.DISABLED
                     if isinstance(widget, (ttk.Combobox, ttk.Spinbox)):
                         target_tk_state = 'readonly' if widget_state == tk.NORMAL else tk.DISABLED
                     elif isinstance(widget, ttk.Entry) and not HAS_TKCALENDAR: # Non toccare DateEntry direttamente qui
                         target_tk_state = tk.NORMAL if widget_state == tk.NORMAL else tk.DISABLED

                     # Cambia stato solo se necessario
                     if str(current_widget_state) != str(target_tk_state):
                         widget.config(state=target_tk_state)
                 except (tk.TclError, AttributeError) as e_widget:
                     print(f"Warning: Could not set state for widget {widget}: {e_widget}")
                     pass # Ignora errori se widget non esiste/è distrutto o non ha 'state'

            # Applica stato ai DateEntry (se tkcalendar è usato)
            for date_widget in date_widgets:
                if date_widget is None or not date_widget.winfo_exists(): continue
                try:
                    target_tk_state = tk.NORMAL if state == tk.NORMAL else tk.DISABLED
                    current_widget_state = date_widget.cget('state')
                    if str(current_widget_state) != str(target_tk_state):
                        date_widget.configure(state=target_tk_state)
                except (tk.TclError, AttributeError, Exception) as e_date:
                     print(f"Warning: Could not set state for DateEntry {date_widget}: {e_date}")
                     pass

        except tk.TclError as e:
            print(f"TclError in _set_controls_state_tk (likely during shutdown): {e}")
        except Exception as e:
            print(f"Error setting control states: {e}")


    def start_analysis_thread(self):
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("Analisi in Corso", "Analisi 10eLotto già in esecuzione.", parent=self.root); return
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("Verifica in Corso", "Verifica 10eLotto in corso. Attendi.", parent=self.root); return

        #<editor-fold desc="Recupero e Validazione Parametri Analisi 10eLotto">
        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi 10eLotto in corso..."); self.attendibilita_label_var.set("")
        self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_date_str = None
        self.check_button.config(state=tk.DISABLED)

        data_source = self.file_path_var.get().strip()
        start_date_str, end_date_str = "", ""
        try:
            if HAS_TKCALENDAR and isinstance(self.start_date_entry, DateEntry): start_date_str, end_date_str = self.start_date_entry.get_date().strftime('%Y-%m-%d'), self.end_date_entry.get_date().strftime('%Y-%m-%d')
            else: start_date_str, end_date_str = self.start_date_entry.get(), self.end_date_entry.get()
        except Exception as e: messagebox.showerror("Errore Data", f"Errore recupero date 10eLotto: {e}", parent=self.root); return

        seq_len_str, num_predict_str = self.seq_len_var.get(), self.num_predict_var.get()
        hidden_layers_str, loss_function, optimizer = self.hidden_layers_var.get(), self.loss_var.get(), self.optimizer_var.get()
        dropout_str, l1_str, l2_str = self.dropout_var.get(), self.l1_var.get(), self.l2_var.get()
        epochs_str, batch_size_str, patience_str, min_delta_str = self.epochs_var.get(), self.batch_size_var.get(), self.patience_var.get(), self.min_delta_var.get()

        errors = []; sequence_length, num_predictions = 5, 10; hidden_layers_config = [512, 256, 128]
        dropout_rate, l1_reg, l2_reg = 0.35, 0.0, 0.0; max_epochs, batch_size, patience, min_delta = 100, 32, 15, 0.0001

        if not data_source: errors.append("Specificare un URL Raw GitHub o un percorso file locale per 10eLotto.")
        elif not data_source.startswith("http://") and not data_source.startswith("https://"):
            if not os.path.exists(data_source): errors.append(f"File locale 10eLotto non trovato:\n{data_source}")
            elif not data_source.lower().endswith(".txt"): errors.append("Il file locale 10eLotto dovrebbe essere .txt.")

        try: start_dt, end_dt = datetime.strptime(start_date_str, '%Y-%m-%d'), datetime.strptime(end_date_str, '%Y-%m-%d'); assert start_dt <= end_dt
        except: errors.append("Date 10eLotto non valide o inizio > fine.")
        try: sequence_length = int(seq_len_str); assert 1 <= sequence_length <= 50
        except: errors.append("Seq. Input 10eLotto non valida (1-50).")
        try: num_predictions = int(num_predict_str); assert 1 <= num_predictions <= 10
        except: errors.append("Numeri da Prevedere 10eLotto non validi (1-10).")
        try: hidden_layers_config = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]; assert hidden_layers_config and all(n > 0 for n in hidden_layers_config)
        except: errors.append("Hidden Layers 10eLotto non validi (es. 256,128).")
        if not loss_function: errors.append("Selezionare Loss Function.")
        if not optimizer: errors.append("Selezionare Optimizer.")
        try: dropout_rate = float(dropout_str); assert 0.0 <= dropout_rate <= 0.8
        except: errors.append("Dropout Rate 10eLotto non valido (0.0-0.8).")
        try: l1_reg = float(l1_str); assert l1_reg >= 0
        except: errors.append("L1 Strength 10eLotto non valido (>= 0).")
        try: l2_reg = float(l2_str); assert l2_reg >= 0
        except: errors.append("L2 Strength 10eLotto non valido (>= 0).")
        try: max_epochs = int(epochs_str); assert max_epochs >= 10
        except: errors.append("Max Epoche 10eLotto non valido (>= 10).")
        try: batch_size = int(batch_size_str); assert batch_size > 0 and (batch_size & (batch_size - 1) == 0)
        except: errors.append("Batch Size 10eLotto non valido (potenza di 2 > 0).")
        try: patience = int(patience_str); assert patience >= 3
        except: errors.append("Patience 10eLotto non valida (>= 3).")
        try: min_delta = float(min_delta_str); assert min_delta >= 0
        except: errors.append("Min Delta 10eLotto non valido (>= 0).")

        if errors: messagebox.showerror("Errore Parametri Input 10eLotto", "\n\n".join(errors), parent=self.root); self.result_label_var.set("Errore parametri."); return
        #</editor-fold>

        self.set_controls_state(tk.DISABLED)
        source_type = "URL" if data_source.startswith("http") else "File Locale"
        self.log_message_gui("=== Avvio Analisi 10eLotto ===")
        self.log_message_gui(f"Sorgente Dati: {source_type} ({data_source})")
        self.log_message_gui(f"Param Dati: Date={start_date_str}-{end_date_str}, SeqIn={sequence_length}, NumOut={num_predictions}")
        self.log_message_gui(f"Param Modello: Layers={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate:.2f}, L1={l1_reg:.4f}, L2={l2_reg:.4f}")
        self.log_message_gui(f"Param Training: Epochs={max_epochs}, Batch={batch_size}, Pat={patience}, MinDelta={min_delta:.5f}")
        self.log_message_gui("-" * 40)

        # === Modifica: Resetta l'evento di stop e avvia il thread ===
        self._stop_event_analysis.clear() # Assicurati che l'evento sia resettato
        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=( # Passa anche l'evento di stop
                data_source, start_date_str, end_date_str, sequence_length,
                loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                num_predictions,
                self._stop_event_analysis # <<< Passa l'evento
            ),
            daemon=True, # Lascia daemon=True, on_close gestirà l'attesa
            name="AnalysisThread" # Dà un nome al thread per debug
        )
        self.analysis_thread.start()
        # ==========================================================

    def run_analysis(self, data_source, start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions, stop_event): # <<< Riceve l'evento
        """Esegue l'analisi 10eLotto nel thread, controllando stop_event."""
        numeri_predetti, attendibilita_msg, success = None, "Analisi 10eLotto non completata", False
        last_update_date = "N/A" # Default
        try:
            # Controllo iniziale stop_event
            if stop_event.is_set():
                self.log_message_gui("Analisi annullata prima dell'inizio.")
                attendibilita_msg = "Analisi annullata"
                return # Esce subito

            # --- Carica dati (non richiede controllo stop interno) ---
            df, _, _, _ = carica_dati_10elotto(data_source, start_date=None, end_date=None, log_callback=self.log_message_gui)
            if df is not None and not df.empty:
                last_update_date = df['Data'].max().strftime('%Y-%m-%d')
            # Aggiorna l'etichetta dell'ultimo aggiornamento (sicuro con after)
            self.root.after(0, self.last_update_label_var.set, f"Ultimo aggiornamento estrazionale: {last_update_date}")

            # Controllo stop dopo caricamento iniziale (per data max)
            if stop_event.is_set():
                self.log_message_gui("Analisi annullata dopo caricamento data max.")
                attendibilita_msg = "Analisi annullata"
                return

            # --- Esegui l'analisi vera e propria, passando lo stop_event ---
            numeri_predetti, attendibilita_msg = analisi_10elotto(
                file_path=data_source, start_date=start_date, end_date=end_date,
                sequence_length=sequence_length, loss_function=loss_function,
                optimizer=optimizer, dropout_rate=dropout_rate, l1_reg=l1_reg,
                l2_reg=l2_reg, hidden_layers_config=hidden_layers_config,
                max_epochs=max_epochs, batch_size=batch_size, patience=patience,
                min_delta=min_delta, num_predictions=num_predictions,
                log_callback=self.log_message_gui,
                stop_event=stop_event # <<< Passa l'evento alla funzione core
            )
            # Verifica se l'analisi è stata annullata DALLA funzione analisi_10elotto
            if stop_event.is_set() and numeri_predetti is None:
                 self.log_message_gui("Analisi interrotta durante l'elaborazione.")
                 attendibilita_msg = "Analisi Interrotta" # Sovrascrive messaggio
                 success = False
            else:
                 # Valuta successo normale
                 success = isinstance(numeri_predetti, list) and len(numeri_predetti) == num_predictions and all(isinstance(n, int) for n in numeri_predetti)

        except Exception as e:
            self.log_message_gui(f"\nERRORE CRITICO run_analysis 10eLotto: {e}\n{traceback.format_exc()}")
            attendibilita_msg = f"Errore critico 10eLotto: {e}"; success = False
        finally:
            # Registra il completamento solo se non annullato esplicitamente
            if not stop_event.is_set():
                 self.log_message_gui("\n=== Analisi 10eLotto Completata ===")

            # Aggiorna i risultati e controlli (usando self.set_result che usa after)
            self.set_result(numeri_predetti, attendibilita_msg)

            if success: # Solo se l'analisi ha avuto successo E non è stata annullata
                self.last_prediction = numeri_predetti
                try:
                    self.last_prediction_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                    self.last_prediction_date_str = end_date
                    self.log_message_gui(f"Previsione 10eLotto salvata (dati fino a {end_date}).")
                except ValueError:
                    self.log_message_gui(f"ATTENZIONE: Errore salvataggio data fine 10eLotto ({end_date}). Verifica non possibile."); success = False # Rendi non verificabile
            else: # Se non successo o annullato
                 self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_date_str = None
                 # Non loggare fallimento se è stato annullato o l'errore è già in attendibilita_msg
                 if not stop_event.is_set() and (not attendibilita_msg or not any(kw in attendibilita_msg.lower() for kw in ["errore", "fallita", "annullata", "interrotta"])):
                     self.log_message_gui("Analisi 10eLotto fallita o risultato non valido.")

            # Riabilita i controlli (usa self.set_controls_state che usa after)
            self.set_controls_state(tk.NORMAL)
            # Pulisci il riferimento al thread *dopo* che tutti gli after sono stati schedulati
            self.root.after(10, self._clear_analysis_thread_ref)

    def _clear_analysis_thread_ref(self):
        """Helper per pulire il riferimento al thread nel thread principale."""
        self.analysis_thread = None
        # Potrebbe essere necessario ri-valutare lo stato dei pulsanti qui
        # se la verifica era disabilitata a causa dell'analisi
        self._set_controls_state_tk(tk.NORMAL)


    def start_check_thread(self):
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("Verifica in Corso", "Verifica 10eLotto già in esecuzione.", parent=self.root); return
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("Analisi in Corso", "Attendere fine analisi 10eLotto.", parent=self.root); return
        if self.last_prediction is None or self.last_prediction_end_date is None or not isinstance(self.last_prediction, list) or len(self.last_prediction) == 0:
            messagebox.showinfo("Nessuna Previsione Valida", "Eseguire prima un'analisi 10eLotto con successo.", parent=self.root); return
        if not all(isinstance(n, int) for n in self.last_prediction):
             messagebox.showerror("Errore Previsione Salvata", "Previsione 10eLotto salvata non valida.", parent=self.root)
             self.last_prediction = None; self.set_controls_state(tk.NORMAL); return

        try:
            num_colpi = int(self.check_colpi_var.get()); assert 1 <= num_colpi <= 100
        except: messagebox.showerror("Errore Colpi", "Numero colpi 10eLotto da verificare non valido (1-100).", parent=self.root); return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Previsione 10eLotto ({num_colpi} Colpi Max) ===")
        self.log_message_gui(f"Verifica numeri (Ord. Prob): {self.last_prediction}")
        self.log_message_gui(f"Previsione basata su dati fino al: {self.last_prediction_date_str}")

        data_source_for_check = self.file_path_var.get().strip()
        if not data_source_for_check:
            messagebox.showerror("Errore Sorgente Dati", "La sorgente dati 10eLotto (URL o file) non è specificata per la verifica.", parent=self.root)
            self.set_controls_state(tk.NORMAL); return
        if not data_source_for_check.startswith("http") and not os.path.exists(data_source_for_check):
             messagebox.showerror("Errore File", f"Il file dati locale 10eLotto '{os.path.basename(data_source_for_check)}' non trovato per verifica.", parent=self.root)
             self.set_controls_state(tk.NORMAL); return

        source_type = "URL" if data_source_for_check.startswith("http") else "File Locale"
        self.log_message_gui(f"Usando sorgente dati per verifica: {source_type} ({data_source_for_check})")
        self.log_message_gui("-" * 40)

        # === Modifica: Resetta l'evento di stop e avvia il thread ===
        self._stop_event_check.clear() # Resetta l'evento
        self.check_thread = threading.Thread(
            target=self.run_check_results,
            args=( data_source_for_check, self.last_prediction, self.last_prediction_date_str, num_colpi,
                   self._stop_event_check ), # <<< Passa l'evento
            daemon=True,
            name="CheckThread" # Nome per debug
        )
        self.check_thread.start()
        # =========================================================

    def run_check_results(self, data_source, prediction_to_check, last_analysis_date_str, num_colpi_to_check, stop_event): # <<< Riceve l'evento
        """Carica dati successivi e verifica la previsione, controllando stop_event."""
        try:
            try:
                last_date = datetime.strptime(last_analysis_date_str, '%Y-%m-%d'); check_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            except ValueError as ve: self.log_message_gui(f"ERRORE formato data analisi 10eLotto: {ve}"); return

            # Controllo stop prima di caricare dati
            if stop_event.is_set(): self.log_message_gui("Verifica annullata prima del caricamento dati."); return

            self.log_message_gui(f"Caricamento dati 10eLotto per verifica da {check_start_date}...")
            df_check, numeri_array_check, _, _ = carica_dati_10elotto( data_source, start_date=check_start_date, end_date=None, log_callback=self.log_message_gui )

            # Controllo stop dopo caricamento dati
            if stop_event.is_set(): self.log_message_gui("Verifica annullata dopo caricamento dati."); return

            if df_check is None: self.log_message_gui("ERRORE: Caricamento dati verifica 10eLotto fallito."); return
            if df_check.empty: self.log_message_gui(f"INFO: Nessuna estrazione 10eLotto trovata dopo {last_analysis_date_str}."); return
            if numeri_array_check is None or len(numeri_array_check) == 0: self.log_message_gui(f"ERRORE: Dati 10eLotto trovati post {last_analysis_date_str}, ma estrazione numeri fallita."); return

            num_available = len(numeri_array_check); num_to_run = min(num_colpi_to_check, num_available)
            self.log_message_gui(f"Trovate {num_available} estraz. 10eLotto successive. Verifico le prossime {num_to_run}...");
            prediction_set = set(prediction_to_check);
            self.log_message_gui(f"Previsione da verificare (Set): {prediction_set}"); self.log_message_gui("-" * 40)
            colpo_counter = 0; found_hits_total = 0; highest_score = 0

            for i in range(num_to_run):
                # Controllo stop all'inizio di ogni iterazione del loop
                if stop_event.is_set():
                    self.log_message_gui(f"Verifica interrotta al colpo {colpo_counter + 1}.")
                    break # Esce dal loop

                colpo_counter += 1
                try:
                    row = df_check.iloc[i]; draw_date = row['Data'].strftime('%Y-%m-%d'); actual_draw = numeri_array_check[i]; actual_draw_set = set(actual_draw)
                    hits = prediction_set.intersection(actual_draw_set); num_hits = len(hits)
                    highest_score = max(highest_score, num_hits)

                    if num_hits > 0:
                        found_hits_total += 1
                        hits_str = f" -> Punti: {num_hits}"; matched_str = f" Numeri Indovinati: {sorted(list(hits))}"
                        self.log_message_gui(f"Colpo {colpo_counter:02d} ({draw_date}): {hits_str}{matched_str}")
                    else:
                        self.log_message_gui(f"Colpo {colpo_counter:02d} ({draw_date}): Nessun risultato.")
                except IndexError: self.log_message_gui(f"ERR: Indice {i} fuori range verifica 10eLotto"); break
                except Exception as e_row: self.log_message_gui(f"ERR imprevisto colpo {colpo_counter} 10eLotto: {e_row}")

            # Log finale solo se il loop non è stato interrotto bruscamente
            if not stop_event.is_set():
                 self.log_message_gui("-" * 40)
                 if found_hits_total == 0: self.log_message_gui(f"Nessun colpo vincente nei {num_to_run} colpi 10eLotto verificati.")
                 else: self.log_message_gui(f"Verifica {num_to_run} colpi 10eLotto completata. Trovati {found_hits_total} colpi con risultati. Punteggio massimo: {highest_score} punti.")

        except Exception as e:
            self.log_message_gui(f"ERRORE CRITICO verifica 10eLotto: {e}\n{traceback.format_exc()}")
        finally:
             # Registra completamento solo se non annullato
             if not stop_event.is_set():
                 self.log_message_gui("\n=== Verifica 10eLotto Completata ===")
             # Riabilita controlli e pulisci riferimento al thread
             self.set_controls_state(tk.NORMAL)
             self.root.after(10, self._clear_check_thread_ref)

    def _clear_check_thread_ref(self):
        """Helper per pulire il riferimento al thread nel thread principale."""
        self.check_thread = None
        # Ri-valuta stato controlli
        self._set_controls_state_tk(tk.NORMAL)


    # === Metodo Nuovo: Gestione Chiusura Finestra ===
    def on_close(self):
        """Gestisce la richiesta di chiusura della finestra (pulsante X)."""
        self.log_message_gui("Richiesta chiusura finestra...")

        # 1. Segnala ai thread di fermarsi impostando gli eventi
        self._stop_event_analysis.set()
        self._stop_event_check.set()

        # 2. Attendi che i thread terminino (con un timeout)
        timeout_secs = 3.0 # Attendi massimo 3 secondi per thread
        wait_start = time.time()
        threads_to_wait = []

        # Controlla se i thread esistono e sono vivi *prima* di aggiungerli
        analysis_thread_local = self.analysis_thread # Copia locale per race condition minima
        if analysis_thread_local and analysis_thread_local.is_alive():
            threads_to_wait.append(analysis_thread_local)

        check_thread_local = self.check_thread # Copia locale
        if check_thread_local and check_thread_local.is_alive():
            threads_to_wait.append(check_thread_local)

        if threads_to_wait:
            self.log_message_gui(f"Attendo terminazione thread: {[t.name for t in threads_to_wait]} (max {timeout_secs}s)")
            for thread in threads_to_wait:
                remaining_timeout = max(0.1, timeout_secs - (time.time() - wait_start))
                try:
                    thread.join(timeout=remaining_timeout)
                    if thread.is_alive():
                        self.log_message_gui(f"ATTENZIONE: Timeout attesa {thread.name}. Potrebbe terminare bruscamente.")
                    else:
                         self.log_message_gui(f"Thread {thread.name} terminato correttamente.")
                except Exception as e:
                    self.log_message_gui(f"Errore durante join di {thread.name}: {e}")
        else:
            self.log_message_gui("Nessun thread attivo da attendere.")

        # 3. Distruggi la finestra principale *solo dopo* aver atteso i thread
        self.log_message_gui("Distruzione finestra Tkinter.")
        try:
             # Pulisci riferimenti ai thread PRIMA di distruggere
             self.analysis_thread = None
             self.check_thread = None
             self.root.destroy()
        except tk.TclError as e:
             print(f"TclError durante root.destroy() (potrebbe essere già in chiusura): {e}")
        except Exception as e:
             print(f"Errore imprevisto durante root.destroy(): {e}")
    # ===============================================


# --- Funzione di Lancio (INVARIATA) ---
def launch_10elotto_window(parent_window):
    """Crea e lancia la finestra dell'applicazione 10eLotto come Toplevel."""
    try:
        lotto_win = tk.Toplevel(parent_window)
        # Non impostare geometry qui se App10eLotto la imposta già
        # lotto_win.geometry("850x910")
        app_instance = App10eLotto(lotto_win) # L'init imposta titolo e geometria
        lotto_win.lift()
        lotto_win.focus_force()
    except NameError as ne:
         print(f"ERRORE INTERNO (elotto_module.py): Classe App10eLotto non trovata - {ne}")
         messagebox.showerror("Errore Interno Modulo", "Impossibile avviare modulo 10eLotto: Classe App10eLotto non definita.", parent=parent_window)
    except Exception as e:
         print(f"ERRORE lancio 10eLotto: {e}\n{traceback.format_exc()}")
         messagebox.showerror("Errore Avvio Modulo", f"Errore avvio modulo 10eLotto:\n{e}", parent=parent_window)

# --- Blocco Esecuzione Standalone (INVARIATO) ---
if __name__ == "__main__":
    print("Esecuzione di elotto_module.py in modalità standalone...")
    print("NOTA: Questo script richiede l'installazione della libreria 'requests'.")
    print("Puoi installarla con: pip install requests")
    try:
        if sys.platform == "win32": from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e_dpi: print(f"Nota: impossibile impostare DPI awareness ({e_dpi})")

    root_standalone = tk.Tk()
    app_standalone = App10eLotto(root_standalone)
    root_standalone.mainloop()