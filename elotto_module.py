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
# NUOVO: Default per K-Fold Cross-Validation
DEFAULT_CV_SPLITS = 5

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
             # print(f"Log GUI widget destroyed, message lost: {message}") # Commentato per ridurre output console
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

# --- Funzioni Specifiche 10eLotto (carica_dati_10elotto INVARIATA) ---
def carica_dati_10elotto(data_source, start_date=None, end_date=None, log_callback=None):
    """
    Carica i dati del 10eLotto da un URL (RAW GitHub) o da un file locale.
    Gestisce la colonna vuota extra.
    MODIFICATO: Non rimuove più le righe solo perché la data non è valida (NaT).
               Logga un avviso se vengono trovate date non valide.
    """
    lines = []
    is_url = data_source.startswith("http://") or data_source.startswith("https://")

    try:
        # --- Blocco Caricamento Dati (URL o Locale) - INVARIATO ---
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
        # --- Fine Blocco Caricamento Dati ---

        # --- Blocco Parsing Righe e Creazione DataFrame Iniziale - INVARIATO ---
        if log_callback: log_callback(f"Lette {len(lines)} righe totali dalla fonte dati.")
        if not lines or len(lines) < 2:
            if log_callback: log_callback("ERRORE: Dati vuoti o solo intestazione.")
            return None, None, None, None

        data_lines = lines[1:] # Salta header
        data = []; malformed_lines = 0; min_expected_cols = 24 # Data + 20 Num + Vuota + Oro1 + Oro2
        for i, line in enumerate(data_lines):
            values = line.rstrip().split('\t')
            if len(values) >= min_expected_cols:
                data.append(values)
            else:
                malformed_lines += 1
                if malformed_lines < 5 and log_callback: log_callback(f"ATT: Riga {i+2} scartata (campi < {min_expected_cols}, trovati {len(values)}): '{line.strip()}'")
        if malformed_lines > 0 and log_callback: log_callback(f"ATTENZIONE: {malformed_lines} righe totali scartate (poche colonne).")
        if not data:
            if log_callback: log_callback("ERRORE: Nessuna riga dati valida trovata dopo il parsing iniziale.")
            return None, None, None, None

        max_cols = max(len(row) for row in data)
        colonne_note = ['Data'] + [f'Num{i+1}' for i in range(20)] + ['ColonnaVuota', 'Oro1', 'Oro2']
        colonne_finali = list(colonne_note)
        if max_cols > len(colonne_note):
             colonne_finali.extend([f'ExtraCol{i}' for i in range(len(colonne_note), max_cols)])

        df = pd.DataFrame(data, columns=colonne_finali[:max_cols])
        if log_callback: log_callback(f"Creato DataFrame shape: {df.shape}, Colonne: {df.columns.tolist()}")

        if 'ColonnaVuota' in df.columns:
             df = df.drop(columns=['ColonnaVuota'])
             if log_callback: log_callback("Rimossa 'ColonnaVuota'.")
        # --- Fine Blocco Parsing ---

        # --- Blocco Pulizia Date (MODIFICATO) ---
        if 'Data' not in df.columns:
            log_callback("ERRORE: Colonna 'Data' mancante."); return None, None, None, None

        # Converte in datetime, gli errori diventano NaT (Not a Time)
        df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d', errors='coerce')

        # Identifica e logga le righe con date non valide (NaT) MA NON LE RIMUOVE
        invalid_date_mask = pd.isna(df['Data'])
        num_invalid_dates = invalid_date_mask.sum()

        if num_invalid_dates > 0:
            if log_callback:
                log_callback(f"--- ATTENZIONE: Identificate {num_invalid_dates} righe con data non valida (NaT) ---")
                log_callback(f"    (Queste righe NON sono state rimosse in questa fase e i loro numeri")
                log_callback(f"     potrebbero essere inclusi se i numeri sono validi e se la riga")
                log_callback(f"     non viene esclusa dal filtro date utente successivo)")
                # Loggare le righe specifiche è opzionale ma può essere utile per il debug
                logged_count = 0
                max_log_rows = 5 # Limita output
                invalid_rows_df = df[invalid_date_mask]
                for index, row in invalid_rows_df.iterrows():
                    if logged_count >= max_log_rows:
                        log_callback(f"    (... e altre {num_invalid_dates - logged_count} righe non valide non mostrate)")
                        break
                    log_callback(f"  -> Riga Indice DF: {index}")
                    # log_callback(f"     {row.to_string()}") # Scommenta per vedere l'intera riga
                    logged_count += 1
                log_callback("---------------------------------------------------------------------")

        # Non rimuoviamo più le righe basate su NaT nella data
        # original_rows = len(df); # Non più necessario qui
        # df = df.dropna(subset=['Data']) # <--- RIMOSSO / COMMENTATO

        # Verifica se il DataFrame è vuoto dopo i passaggi iniziali
        if df.empty:
             log_callback("ERRORE: Nessun dato rimasto dopo il parsing iniziale."); return df, None, None, None # Ritorna df vuoto

        # Ordina per data, mettendo eventuali NaT all'inizio (o alla fine se preferisci 'last')
        df = df.sort_values(by='Data', ascending=True, na_position='first')
        # --- Fine Blocco Pulizia Date ---

        # --- Blocco Filtro Date Utente - INVARIATO ---
        # Questo filtro escluderà probabilmente le righe con NaT, il che è ok.
        if start_date:
            try:
                 start_dt = pd.to_datetime(start_date)
                 # Il confronto con NaT risulterà False, quindi le righe NaT verranno escluse qui
                 df = df[df['Data'] >= start_dt]
            except Exception as e: log_callback(f"Errore filtro data inizio: {e}")
        if end_date:
             try:
                 end_dt = pd.to_datetime(end_date)
                 # Il confronto con NaT risulterà False, quindi le righe NaT verranno escluse qui
                 df = df[df['Data'] <= end_dt]
             except Exception as e: log_callback(f"Errore filtro data fine: {e}")

        if log_callback: log_callback(f"Righe dopo filtro date utente ({start_date} - {end_date}): {len(df)}")
        # --- Fine Blocco Filtro Date Utente ---

        # --- Blocco Estrazione Numeri (principali, oro, extra) - INVARIATO ---
        numeri_cols = [f'Num{i+1}' for i in range(20)]; numeri_array, numeri_oro, numeri_extra = None, None, None
        if not df.empty:
            # Lavora su una copia del DataFrame filtrato per data
            df_cleaned = df.copy()

            # Verifica presenza colonne Num1-20
            if not all(col in df_cleaned.columns for col in numeri_cols):
                 log_callback(f"ERRORE: Colonne Num1-20 mancanti nel dataframe filtrato."); return df.copy(), None, None, None # Ritorna df filtrato

            # Prova a convertire i numeri Num1-20 e rimuovi righe dove fallisce
            try:
                for col in numeri_cols: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                rows_b4_num_clean = len(df_cleaned)
                df_cleaned = df_cleaned.dropna(subset=numeri_cols) # Rimuove righe con Num1-20 non numerici
                dropped_num = rows_b4_num_clean - len(df_cleaned)
                if dropped_num > 0 and log_callback: log_callback(f"Scartate {dropped_num} righe (Num1-20 non numerici).")

                # Estrai l'array dei numeri principali SE ci sono righe valide rimaste
                if not df_cleaned.empty:
                    numeri_array = df_cleaned[numeri_cols].values.astype(int)
                else:
                    log_callback("ATTENZIONE: Nessuna riga rimasta dopo pulizia Num1-20.")
            except Exception as e: log_callback(f"ERRORE pulizia/estrazione Num1-20: {e}")

            # Estrai Oro SOLO se numeri_array è stato estratto e dalle righe in df_cleaned
            # Usa df_cleaned perché contiene le righe dove Num1-20 sono validi
            if numeri_array is not None and 'Oro1' in df_cleaned.columns and 'Oro2' in df_cleaned.columns:
                try:
                    # Lavoriamo su una copia specifica per Oro per non alterare df_cleaned per Extra
                    df_oro_temp = df_cleaned.copy()
                    df_oro_temp['Oro1'] = pd.to_numeric(df_oro_temp['Oro1'], errors='coerce')
                    df_oro_temp['Oro2'] = pd.to_numeric(df_oro_temp['Oro2'], errors='coerce')
                    df_oro_temp = df_oro_temp.dropna(subset=['Oro1', 'Oro2'])
                    if not df_oro_temp.empty:
                         numeri_oro = df_oro_temp[['Oro1', 'Oro2']].values.astype(int)
                         if log_callback: log_callback(f"Estratto array Oro. Shape: {numeri_oro.shape}")
                    else:
                        numeri_oro = None
                        if log_callback: log_callback("Nessuna riga valida per estrazione Oro (dopo pulizia Oro).")
                except Exception as e: log_callback(f"Errore pulizia/estrazione Oro: {e}")

            # Estrai Extra SOLO se numeri_array è stato estratto e dalle righe in df_cleaned
            if numeri_array is not None and any(col.startswith('ExtraCol') for col in df_cleaned.columns):
                 extra_cols = [col for col in df_cleaned.columns if col.startswith('ExtraCol')]
                 if extra_cols:
                      first_extra_col = extra_cols[0]
                      try:
                          # Estrai la colonna extra da df_cleaned (dove Num1-20 sono validi)
                          # NON fare dropna qui, così manteniamo la corrispondenza con numeri_array
                          extra_vals = pd.to_numeric(df_cleaned[first_extra_col], errors='coerce')
                          numeri_extra = extra_vals.values # Può contenere NaN se Extra non è numerico
                          if log_callback: log_callback(f"Estratta colonna Extra '{first_extra_col}'. Shape: {numeri_extra.shape}")
                      except Exception as e_ex:
                          log_callback(f"ATTENZIONE: Errore conversione colonna Extra '{first_extra_col}': {e_ex}")
                          numeri_extra = None
                 else:
                      numeri_extra = None
        # --- Fine Blocco Estrazione Numeri ---

        # --- Log Finale e Ritorno ---
        final_rows_df = len(df) # Righe nel df dopo filtro date utente
        final_rows_arr = len(numeri_array) if numeri_array is not None else 0 # Righe nell'array pulito (dopo dropna su numeri)
        if log_callback:
            log_callback(f"Caricamento/Filtraggio completato.")
            log_callback(f"  Righe df dopo filtro date: {final_rows_df}")
            log_callback(f"  Righe array numeri (Num1-20 validi): {final_rows_arr}")

        # Ritorna il df filtrato per data e gli array numerici puliti
        # df.copy() contiene le righe con date valide che rientrano nel range utente
        # numeri_array contiene i Num1-20 dalle righe sopra, solo dove erano numerici validi
        return df.copy(), numeri_array, numeri_oro, numeri_extra
        # --- Fine Log Finale e Ritorno ---

    except Exception as e:
        if log_callback: log_callback(f"Errore grave in carica_dati_10elotto: {e}\n{traceback.format_exc()}");
        return None, None, None, None


# NUOVO: Funzione per Feature Engineering
def engineer_features(numeri_array, log_callback=None):
    """
    Crea features aggiuntive dall'array dei numeri estratti.
    Input: numeri_array (N_draws, 20)
    Output: combined_features (N_draws, 20 + N_new_features)
    """
    if numeri_array is None or numeri_array.ndim != 2 or numeri_array.shape[1] != 20:
        if log_callback: log_callback("ERRORE (engineer_features): Input numeri_array non valido.")
        return None

    if log_callback: log_callback(f"Inizio Feature Engineering su {numeri_array.shape[0]} estrazioni...")

    try:
        # Features base per riga
        draw_sum = np.sum(numeri_array, axis=1, keepdims=True)
        draw_mean = np.mean(numeri_array, axis=1, keepdims=True)
        odd_count = np.sum(numeri_array % 2 != 0, axis=1, keepdims=True)
        even_count = 20 - odd_count # Calcolato da odd_count
        low_count = np.sum((numeri_array >= 1) & (numeri_array <= 45), axis=1, keepdims=True)
        high_count = 20 - low_count # Calcolato da low_count

        # Combina le nuove features
        engineered_features = np.concatenate([
            draw_sum,
            draw_mean,
            odd_count,
            even_count,
            low_count,
            high_count
        ], axis=1)

        # Combina le features ingegnerizzate con i numeri originali
        combined_features = np.concatenate([numeri_array, engineered_features], axis=1)

        if log_callback: log_callback(f"Feature Engineering completato. Shape finale features: {combined_features.shape}")
        return combined_features

    except Exception as e:
        if log_callback: log_callback(f"ERRORE durante Feature Engineering: {e}\n{traceback.format_exc()}")
        return None

# MODIFICATO: Funzione prepara_sequenze per gestire le nuove features
def prepara_sequenze_per_modello(input_feature_array, target_number_array, sequence_length=5, log_callback=None):
    """
    Prepara le sequenze per il modello LSTM/Dense.
    Input:
        input_feature_array: Array con le features (numeri + engineered), shape (N_draws, N_features)
        target_number_array: Array originale dei numeri estratti (usato per il target), shape (N_draws, 20)
        sequence_length: Lunghezza della sequenza di input.
    Output:
        X: Array delle sequenze di input, shape (N_sequences, sequence_length * N_features)
        y: Array target (one-hot encoded), shape (N_sequences, 90)
    """
    if input_feature_array is None or target_number_array is None:
        if log_callback: log_callback("ERRORE (prep_seq): Input array (features o target) mancante.")
        return None, None
    if input_feature_array.ndim != 2 or target_number_array.ndim != 2:
        if log_callback: log_callback("ERRORE (prep_seq): Input arrays devono avere 2 dimensioni.")
        return None, None
    if input_feature_array.shape[0] != target_number_array.shape[0]:
        if log_callback: log_callback("ERRORE (prep_seq): Disallineamento righe tra feature array e target array.")
        return None, None
    if target_number_array.shape[1] != 20:
         if log_callback: log_callback(f"ERRORE (prep_seq): Target number array non ha 20 colonne (shape: {target_number_array.shape}).")
         return None, None

    n_features = input_feature_array.shape[1]
    if log_callback: log_callback(f"Preparazione sequenze: seq_len={sequence_length}, num_features={n_features}")

    X, y = [], []
    num_estrazioni = len(input_feature_array)

    if num_estrazioni <= sequence_length:
        if log_callback: log_callback(f"ERRORE: Estrazioni ({num_estrazioni}) <= seq ({sequence_length}). Necessarie {sequence_length + 1}.")
        return None, None

    valid_seq, invalid_tgt = 0, 0
    for i in range(num_estrazioni - sequence_length):
        # Sequenza di input dalle features combinate
        in_seq = input_feature_array[i : i + sequence_length]

        # Target dalla *successiva* estrazione dell'array *originale*
        tgt_extr = target_number_array[i + sequence_length]

        # Validazione target (1-90)
        mask = (tgt_extr >= 1) & (tgt_extr <= 90)
        if np.all(mask):
            # Crea target one-hot
            target = np.zeros(90, dtype=int)
            target[tgt_extr - 1] = 1 # Indici 0-89 per numeri 1-90

            # Appiattisci la sequenza di input e aggiungi
            X.append(in_seq.flatten())
            y.append(target)
            valid_seq += 1
        else:
            invalid_tgt += 1
            if invalid_tgt < 5 and log_callback: log_callback(f"ATT: Scartata seq indice {i} (target non valido: {tgt_extr[~mask]})")

    if invalid_tgt > 0 and log_callback: log_callback(f"Scartate {invalid_tgt} sequenze totali (target non valido).")
    if not X:
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata."); return None, None

    if log_callback: log_callback(f"Create {valid_seq} sequenze valide.")
    try:
        X_np = np.array(X)
        y_np = np.array(y)
        if log_callback: log_callback(f"Shape finale: X={X_np.shape}, y={y_np.shape}")
        return X_np, y_np
    except Exception as e:
        if log_callback: log_callback(f"ERRORE conversione NumPy in prep_seq: {e}"); return None, None


# --- build_model_10elotto, LogCallback, genera_previsione_10elotto (INVARIATE NELLA LORO LOGICA INTERNA) ---
# Nota: build_model_10elotto riceverà un input_shape diverso, ma la sua struttura non cambia.
# Nota: genera_previsione_10elotto riceverà un X_input con più features, ma la sua logica non cambia.

def build_model_10elotto(input_shape, hidden_layers=[512, 256, 128], loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    """Costruisce il modello Keras. (Logica interna invariata)"""
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
    """Callback Keras per loggare l'output nella GUI. (Invariata)"""
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
    """Genera la previsione usando il modello addestrato. (Logica interna invariata)"""
    if log_callback: log_callback(f"Generazione previsione 10eLotto per {num_predictions} numeri...")
    if model is None: log_callback("ERR (genera_prev_10eLotto): Modello non valido."); return None
    if X_input is None or X_input.size == 0: log_callback("ERR (genera_prev_10eLotto): Input vuoto."); return None
    if not (1 <= num_predictions <= 90): log_callback(f"ERR (genera_prev_10eLotto): num_predictions={num_predictions} non valido (1-90)."); return None
    try:
        input_reshaped = None
        # L'input X_input dovrebbe già essere (1, N_features_flattened) dopo scaling e flatten
        if X_input.ndim == 2 and X_input.shape[0] == 1:
             input_reshaped = X_input
        # Gestiamo anche il caso in cui arrivi un array 1D per errore
        elif X_input.ndim == 1:
             input_reshaped = X_input.reshape(1, -1)
             if log_callback: log_callback(f"ATT: Ricevuto input 1D shape {X_input.shape}, reshaped a {input_reshaped.shape}.")
        else:
            if log_callback: log_callback(f"ERRORE: Shape input 10eLotto non gestita per predict: {X_input.shape}. Atteso (1, N_features_flat)."); return None

        # Verifica shape input vs modello (opzionale ma utile)
        try:
            if hasattr(model, 'input_shape') and model.input_shape is not None: expected_input_features = model.input_shape[-1]
            elif hasattr(model, 'layers') and model.layers and hasattr(model.layers[0], 'input_shape'): expected_input_features = model.layers[0].input_shape[-1]
            else: expected_input_features = None
            if expected_input_features is not None and input_reshaped.shape[1] != expected_input_features:
                log_callback(f"ERRORE Shape Input 10eLotto: Input({input_reshaped.shape[1]}) != Modello({expected_input_features})."); return None
        except Exception as e_shape:
            if log_callback: log_callback(f"ATT: Eccezione verifica input_shape modello 10eLotto: {e_shape}")

        # Previsione
        pred_probabilities = model.predict(input_reshaped, verbose=0)
        if pred_probabilities is None or pred_probabilities.size == 0: log_callback("ERR: predict() ha restituito vuoto."); return None
        if pred_probabilities.ndim != 2 or pred_probabilities.shape[0] != 1 or pred_probabilities.shape[1] != 90:
             log_callback(f"ERRORE: Output shape da predict inatteso: {pred_probabilities.shape}. Atteso (1, 90)."); return None

        # Estrai numeri migliori
        probs_vector = pred_probabilities[0]
        sorted_indices = np.argsort(probs_vector); # Indici ordinati per probabilità crescente
        top_indices_ascending_prob = sorted_indices[-num_predictions:] # Prendi gli ultimi N (prob più alta)
        top_indices_descending_prob = top_indices_ascending_prob[::-1]; # Inverti per avere probabilità decrescente
        predicted_numbers_by_prob = [int(index + 1) for index in top_indices_descending_prob] # +1 per passare da indice 0-89 a numero 1-90

        if log_callback: log_callback(f"Numeri 10eLotto predetti ({len(predicted_numbers_by_prob)} ord. per probabilità decr.): {predicted_numbers_by_prob}")
        return predicted_numbers_by_prob
    except Exception as e:
        if log_callback: log_callback(f"ERRORE CRITICO generazione previsione 10eLotto: {e}\n{traceback.format_exc()}"); return None

# --- Funzione Analisi Principale (MODIFICATA per Feature Engineering e Cross-Validation) ---
def analisi_10elotto(file_path, start_date, end_date, sequence_length=5,
                     loss_function='binary_crossentropy', optimizer='adam',
                     dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                     hidden_layers_config=[512, 256, 128],
                     max_epochs=100, batch_size=32, patience=15, min_delta=0.0001,
                     num_predictions=10,
                     n_cv_splits=DEFAULT_CV_SPLITS, # NUOVO: Numero di fold per CV
                     log_callback=None,
                     stop_event=None):
    """
    Analizza i dati del 10 e Lotto, applica feature engineering,
    usa TimeSeriesSplit cross-validation, addestra un modello finale e genera previsioni.
    """
    if log_callback:
        source_type = "URL" if file_path.startswith("http") else "File Locale"
        source_name = os.path.basename(file_path) if source_type == "File Locale" else file_path
        log_callback(f"=== Avvio Analisi 10eLotto Dettagliata (con FE & CV) ===")
        log_callback(f"Sorgente: {source_type} ({source_name}), Date: {start_date}-{end_date}, SeqIn: {sequence_length}, NumOut: {num_predictions}")
        log_callback(f"Modello: L={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")
        log_callback(f"Training: Epochs={max_epochs}, Batch={batch_size}, CV Splits={n_cv_splits}, Pat={patience}, MinDelta={min_delta}")
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

    # NUOVO: 2. Feature Engineering
    combined_features = engineer_features(numeri_array, log_callback=log_callback)
    if combined_features is None:
        return None, "Feature Engineering fallito."

    # Controllo stop dopo FE
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo Feature Engineering."); return None, "Analisi annullata"

    # NUOVO: 3. Scaling delle Features Combinate (prima di creare le sequenze)
    scaler = StandardScaler()
    try:
        # Adatta lo scaler a tutte le features combinate disponibili
        combined_features_scaled = scaler.fit_transform(combined_features)
        log_callback(f"Features combinate scalate con StandardScaler. Shape: {combined_features_scaled.shape}")
    except Exception as e_scale:
        log_callback(f"ERRORE scaling features combinate: {e_scale}"); return None, "Errore scaling dati input"

    # MODIFICATO: 4. Prepara sequenze usando le features scalate e i numeri target originali
    X, y = None, None
    try:
        # Passa le features scalate per l'input (X) e l'array originale per il target (y)
        X, y = prepara_sequenze_per_modello(combined_features_scaled, numeri_array, sequence_length, log_callback=log_callback)
        if X is None or y is None or len(X) == 0: return None, "Creazione sequenze fallita o nessuna sequenza valida."

        # Verifica minima per CV
        min_samples_for_cv = n_cv_splits + 1
        if len(X) < min_samples_for_cv:
            msg = f"ERRORE: Troppi pochi campioni ({len(X)}) per {n_cv_splits}-Fold CV. Servono almeno {min_samples_for_cv}. Riduci il numero di split o aumenta i dati."; log_callback(msg); return None, msg

    except Exception as e:
        log_callback(f"Errore preparazione sequenze: {e}\n{traceback.format_exc()}"); return None, f"Errore prep sequenze: {e}"

    # Controllo stop dopo preparazione sequenze
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo preparazione sequenze."); return None, "Analisi annullata"

    # MODIFICATO: 5. Cross-Validation con TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    fold_val_losses = []
    fold_val_accuracies = [] # Aggiungiamo anche l'accuracy
    fold_best_epochs = []

    log_callback(f"\n--- Inizio {n_cv_splits}-Fold TimeSeries Cross-Validation ---")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        if stop_event and stop_event.is_set(): log_callback(f"Cross-Validation interrotta prima del fold {fold+1}."); break # Esce dal loop CV

        log_callback(f"\n--- Fold {fold+1}/{n_cv_splits} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        log_callback(f"Train: {len(X_train)} campioni (indici {train_index[0]}-{train_index[-1]}), Validation: {len(X_val)} campioni (indici {val_index[0]}-{val_index[-1]})")

        if len(X_train) == 0 or len(X_val) == 0:
             log_callback(f"ATTENZIONE: Fold {fold+1} saltato (train o validation set vuoto).")
             continue

        # Costruisci e addestra il modello per questo fold
        model_fold, history_fold, tf_log_callback_obj_fold = None, None, None
        try:
            tf.keras.backend.clear_session() # Pulisci sessione per nuovo modello
            input_shape_fold = (X_train.shape[1],)
            model_fold = build_model_10elotto(input_shape_fold, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
            if model_fold is None:
                log_callback(f"ERRORE: Costruzione modello fallita per fold {fold+1}. Salto fold.")
                continue

            monitor = 'val_loss' # Monitoriamo sempre val_loss nei fold
            early_stopping_fold = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=0) # Verbose 0 nel loop
            tf_log_callback_obj_fold = LogCallback(log_callback, stop_event)

            log_callback(f"Inizio addestramento fold {fold+1}...")
            history_fold = model_fold.fit(X_train, y_train, validation_data=(X_val, y_val),
                                          epochs=max_epochs, batch_size=batch_size,
                                          callbacks=[early_stopping_fold, tf_log_callback_obj_fold],
                                          verbose=0)

            # Controllo stop subito dopo il fit del fold
            if stop_event and stop_event.is_set(): log_callback(f"Training fold {fold+1} interrotto."); break # Esce dal loop CV

            if history_fold and history_fold.history and 'val_loss' in history_fold.history:
                best_epoch_idx = np.argmin(history_fold.history['val_loss'])
                best_val_loss = history_fold.history['val_loss'][best_epoch_idx]
                best_val_acc = history_fold.history['val_accuracy'][best_epoch_idx] if 'val_accuracy' in history_fold.history else -1.0

                fold_val_losses.append(best_val_loss)
                fold_val_accuracies.append(best_val_acc)
                fold_best_epochs.append(best_epoch_idx + 1) # Epoche partono da 1

                log_callback(f"Fold {fold+1} completato. Miglior epoca: {best_epoch_idx+1}, Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
            else:
                log_callback(f"ATTENZIONE: History non valida o 'val_loss' mancante per fold {fold+1}. Salto fold.")

        except tf.errors.ResourceExhaustedError as e_mem:
             msg = f"ERRORE Memoria fold {fold+1}: {e_mem}. Riduci batch size/modello."; log_callback(msg); log_callback(traceback.format_exc()); break # Interrompi CV per errore memoria
        except Exception as e_fold:
            # Verifica se l'eccezione è dovuta a stop richiesto
            if stop_event and stop_event.is_set(): log_callback(f"Eccezione fold {fold+1} probabilmente dovuta a stop: {e_fold}"); break # Interrompi CV
            msg = f"ERRORE CRITICO addestramento fold {fold+1}: {e_fold}"; log_callback(msg); log_callback(traceback.format_exc());
            # Potresti decidere di continuare con gli altri fold o interrompere
            # break # Interrompi CV in caso di errore grave
            log_callback("Continuo con il prossimo fold nonostante l'errore...")
            continue # Prova ad andare al prossimo fold

    # Fine loop Cross-Validation
    if stop_event and stop_event.is_set(): log_callback("Cross-Validation interrotta."); return None, "Analisi Interrotta (CV)"

    # Calcola e logga risultati medi CV (se ci sono stati fold completati)
    avg_val_loss, avg_val_acc, avg_epochs = -1.0, -1.0, -1.0
    attendibilita_cv_msg = "Attendibilità CV: Non Determinata (CV incompleta o fallita)"
    if fold_val_losses:
        avg_val_loss = np.mean(fold_val_losses)
        avg_val_acc = np.mean(fold_val_accuracies) if fold_val_accuracies else -1.0
        avg_epochs = np.mean(fold_best_epochs) if fold_best_epochs else -1.0
        log_callback("\n--- Risultati Cross-Validation ---")
        log_callback(f"Loss media (val): {avg_val_loss:.4f}")
        if avg_val_acc >= 0: log_callback(f"Accuracy media (val): {avg_val_acc:.4f}")
        if avg_epochs >= 0: log_callback(f"Epoca media ottimale: {avg_epochs:.1f}")
        attendibilita_cv_msg = f"Attendibilità CV: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f} ({len(fold_val_losses)} folds)"
        log_callback("---------------------------------")
    else:
        log_callback("\nATTENZIONE: Nessun fold di cross-validation completato con successo.")
        return None, "Cross-Validation fallita (nessun fold valido)"

    # Controllo stop prima del training finale
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima del training finale."); return None, "Analisi Interrotta (Pre-Final Train)"

    # MODIFICATO: 6. Addestra il modello finale su TUTTI i dati (X, y)
    # Usiamo i parametri originali ma senza EarlyStopping su validation set,
    # addestriamo per un numero fisso di epoche (es. max_epochs o media epoche da CV).
    # Scelta: Usiamo max_epochs come limite superiore, ma includiamo EarlyStopping
    # monitorando 'loss' (loss di training) per evitare overfitting eccessivo.
    final_model = None
    log_callback("\n--- Addestramento Modello Finale su tutti i dati ---")
    try:
        tf.keras.backend.clear_session()
        input_shape_final = (X.shape[1],)
        final_model = build_model_10elotto(input_shape_final, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if final_model is None: return None, "Costruzione modello finale fallita"

        # Usiamo EarlyStopping sulla loss di training per il modello finale
        # Questo è un compromesso: evita di girare per tutte le max_epochs se la loss smette
        # di migliorare significativamente, anche se non c'è un vero val set qui.
        final_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=1) # Monitor 'loss'
        final_log_callback = LogCallback(log_callback, stop_event)

        log_callback(f"Inizio addestramento finale (max {max_epochs} epoche, ES su 'loss')...")
        history_final = final_model.fit(X, y,
                                        epochs=max_epochs,
                                        batch_size=batch_size,
                                        callbacks=[final_early_stopping, final_log_callback],
                                        verbose=0) # validation_data non specificato

        # Controllo stop subito dopo il fit finale
        if stop_event and stop_event.is_set(): log_callback("Addestramento finale interrotto."); return None, "Analisi Interrotta (Final Train)"

        final_epochs = len(history_final.history.get('loss', []))
        final_loss = history_final.history.get('loss', [float('inf')])[-1]
        final_acc = history_final.history.get('accuracy', [-1.0])[-1]
        log_callback(f"Addestramento finale completato: {final_epochs} epoche, Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")

    except tf.errors.ResourceExhaustedError as e_mem:
         msg = f"ERRORE Memoria Addestramento Finale: {e_mem}."; log_callback(msg); log_callback(traceback.format_exc()); return None, msg
    except Exception as e_final:
        if stop_event and stop_event.is_set(): log_callback(f"Eccezione training finale probabilmente dovuta a stop: {e_final}"); return None, "Analisi Interrotta (Final Train Error)"
        msg = f"ERRORE CRITICO addestramento finale: {e_final}"; log_callback(msg); log_callback(traceback.format_exc()); return None, msg

    # Controllo stop prima della previsione
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima della previsione finale."); return None, "Analisi Interrotta (Pre-Predict)"

    # MODIFICATO: 7. Prepara input e genera previsione con il modello finale
    numeri_predetti = None
    try:
        log_callback("\nPreparazione input per previsione finale...")
        # Prendi le ultime 'sequence_length' righe dalle features COMBINATE e SCALATE
        if len(combined_features_scaled) < sequence_length:
             log_callback("ERRORE: Dati scalati insufficienti per input previsione."); return None, "Dati insuff per input previsione"

        input_pred_seq_scaled = combined_features_scaled[-sequence_length:]
        # Appiattisci la sequenza
        input_pred_flat_scaled = input_pred_seq_scaled.flatten()
        # Reshape per il modello (1 campione, N features appiattite)
        input_pred_ready = input_pred_flat_scaled.reshape(1, -1)

        log_callback(f"Input previsione pronto (shape: {input_pred_ready.shape}). Generazione...")
        numeri_predetti = genera_previsione_10elotto(final_model, input_pred_ready, num_predictions, log_callback=log_callback)

        if numeri_predetti is None: return None, "Generazione previsione finale fallita."

        # Combina l'attendibilità CV con quella del training finale
        final_attendibilita_msg = f"{attendibilita_cv_msg}. Training finale: Loss={final_loss:.4f}, Acc={final_acc:.4f}"
        log_callback(f"\nAttendibilità combinata: {final_attendibilita_msg}")

        return numeri_predetti, final_attendibilita_msg

    except Exception as e_pred:
         log_callback(f"Errore CRITICO previsione finale: {e_pred}\n{traceback.format_exc()}"); return None, f"Errore previsione finale: {e_pred}"
# --- Fine Funzione analisi_10elotto ---


# --- Definizione Classe App10eLotto (MODIFICATA leggermente per CV) ---
class App10eLotto:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi e Previsione 10eLotto (v8.0 - FE & CV)") # Titolo aggiornato
        self.root.geometry("850x950") # Leggermente più alta per CV param

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
        self.data_params_frame = ttk.LabelFrame(self.params_container, text="Parametri Dati e Previsione", padding="10")
        self.data_params_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        ttk.Label(self.data_params_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        default_start_10e = datetime.now() - pd.Timedelta(days=120) # Aumentato default per avere più dati
        if HAS_TKCALENDAR:
             self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd')
             try: self.start_date_entry.set_date(default_start_10e)
             except ValueError: self.start_date_entry.set_date(datetime.now() - pd.Timedelta(days=60))
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
        self.seq_len_var = tk.StringVar(value="5") # Aumentato leggermente default
        self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=2, to=50, textvariable=self.seq_len_var, width=5, wrap=True, state='readonly')
        self.seq_len_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Numeri da Prevedere:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="5")
        self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=1, to=10, increment=1, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly')
        self.num_predict_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # --- Colonna Destra: Parametri Modello e Training ---
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello e Training", padding="10")
        self.model_params_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        self.model_params_frame.columnconfigure(1, weight=1) # Permetti all'entry di espandersi
        current_row = 0 # Usa un contatore per le righe

        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=current_row, column=0, padx=5, pady=3, sticky=tk.W)
        self.hidden_layers_var = tk.StringVar(value="256, 128") # Ridotto default per FE
        self.hidden_layers_entry = ttk.Entry(self.model_params_frame, textvariable=self.hidden_layers_var, width=25)
        self.hidden_layers_entry.grid(row=current_row, column=1, columnspan=2, padx=5, pady=3, sticky=tk.EW); current_row += 1 # columnspan=2

        ttk.Label(self.model_params_frame, text="Loss Function:").grid(row=current_row, column=0, padx=5, pady=3, sticky=tk.W)
        self.loss_var = tk.StringVar(value='binary_crossentropy')
        self.loss_combo = ttk.Combobox(self.model_params_frame, textvariable=self.loss_var, width=23, state='readonly', values=['binary_crossentropy', 'mse', 'mae', 'huber_loss'])
        self.loss_combo.grid(row=current_row, column=1, columnspan=2, padx=5, pady=3, sticky=tk.EW); current_row += 1

        ttk.Label(self.model_params_frame, text="Optimizer:").grid(row=current_row, column=0, padx=5, pady=3, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value='adam')
        self.optimizer_combo = ttk.Combobox(self.model_params_frame, textvariable=self.optimizer_var, width=23, state='readonly', values=['adam', 'rmsprop', 'sgd', 'adagrad', 'adamw'])
        self.optimizer_combo.grid(row=current_row, column=1, columnspan=2, padx=5, pady=3, sticky=tk.EW); current_row += 1

        # Parametri su stessa riga se possibile
        param_frame_1 = ttk.Frame(self.model_params_frame); param_frame_1.grid(row=current_row, column=0, columnspan=3, sticky=tk.EW); current_row += 1
        ttk.Label(param_frame_1, text="Dropout:").pack(side=tk.LEFT, padx=(5,2))
        self.dropout_var = tk.StringVar(value="0.50") # Leggermente ridotto
        self.dropout_spinbox = ttk.Spinbox(param_frame_1, from_=0.0, to=0.8, increment=0.05, format="%.2f", textvariable=self.dropout_var, width=5, wrap=True, state='readonly')
        self.dropout_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(param_frame_1, text="L1:").pack(side=tk.LEFT, padx=(5,2))
        self.l1_var = tk.StringVar(value="0.00")
        self.l1_entry = ttk.Entry(param_frame_1, textvariable=self.l1_var, width=6)
        self.l1_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(param_frame_1, text="L2:").pack(side=tk.LEFT, padx=(5,2))
        self.l2_var = tk.StringVar(value="0.00")
        self.l2_entry = ttk.Entry(param_frame_1, textvariable=self.l2_var, width=6)
        self.l2_entry.pack(side=tk.LEFT, padx=(0,5))

        param_frame_2 = ttk.Frame(self.model_params_frame); param_frame_2.grid(row=current_row, column=0, columnspan=3, sticky=tk.EW); current_row += 1
        ttk.Label(param_frame_2, text="Max Epoche:").pack(side=tk.LEFT, padx=(5,2))
        self.epochs_var = tk.StringVar(value="30") # Ridotto default
        self.epochs_spinbox = ttk.Spinbox(param_frame_2, from_=10, to=500, increment=10, textvariable=self.epochs_var, width=5, wrap=True, state='readonly')
        self.epochs_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(param_frame_2, text="Batch Size:").pack(side=tk.LEFT, padx=(5,2))
        self.batch_size_var = tk.StringVar(value="32") # Aumentato default
        batch_values = [str(2**i) for i in range(4, 10)] # 16 a 512
        self.batch_size_combo = ttk.Combobox(param_frame_2, textvariable=self.batch_size_var, values=batch_values, width=5, state='readonly')
        self.batch_size_combo.pack(side=tk.LEFT, padx=(0,10))

        param_frame_3 = ttk.Frame(self.model_params_frame); param_frame_3.grid(row=current_row, column=0, columnspan=3, sticky=tk.EW); current_row += 1
        ttk.Label(param_frame_3, text="ES Patience:").pack(side=tk.LEFT, padx=(5,2))
        self.patience_var = tk.StringVar(value="15") # Ridotto default
        self.patience_spinbox = ttk.Spinbox(param_frame_3, from_=3, to=50, increment=1, textvariable=self.patience_var, width=4, wrap=True, state='readonly')
        self.patience_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(param_frame_3, text="ES Min Delta:").pack(side=tk.LEFT, padx=(5,2))
        self.min_delta_var = tk.StringVar(value="0.0005") # Aumentato leggermente
        self.min_delta_entry = ttk.Entry(param_frame_3, textvariable=self.min_delta_var, width=8)
        self.min_delta_entry.pack(side=tk.LEFT, padx=(0,10))

        # NUOVO: Parametro CV Splits
        ttk.Label(self.model_params_frame, text="CV Splits (>=2):").grid(row=current_row, column=0, padx=5, pady=3, sticky=tk.W)
        self.cv_splits_var = tk.StringVar(value=str(DEFAULT_CV_SPLITS))
        self.cv_splits_spinbox = ttk.Spinbox(self.model_params_frame, from_=2, to=20, increment=1, textvariable=self.cv_splits_var, width=5, wrap=True, state='readonly')
        self.cv_splits_spinbox.grid(row=current_row, column=1, padx=5, pady=3, sticky=tk.W); current_row += 1


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
        self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier', 14, 'bold'), foreground='darkblue') # Colore cambiato
        self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value="")
        self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 9, 'italic')) # Font leggermente ridotto
        self.attendibilita_label.pack(pady=2, fill=tk.X) # Fill X per andare a capo

        # --- Log Area ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 9)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=18, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='#f0f0f0', foreground='black') # Sfondo leggermente grigio
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Label Ultimo Aggiornamento ---
        self.last_update_label_var = tk.StringVar(value="Ultimo aggiornamento estrazionale: N/A")
        self.last_update_label = ttk.Label(self.main_frame, textvariable=self.last_update_label_var, font=('Helvetica', 9, 'italic'))
        self.last_update_label.pack(pady=(5,0), anchor='w') # Allineato a sx

        # --- Threading Safety (Invariato) ---
        self.analysis_thread = None
        self.check_thread = None
        self._stop_event_analysis = threading.Event() # Evento per fermare l'analisi
        self._stop_event_check = threading.Event()    # Evento per fermare la verifica

        # Intercetta la chiusura della finestra
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)


    # --- Metodi Classe (browse_file, log_message_gui, set_result, _update_result_labels INVARIATI) ---

    def browse_file(self):
        """Apre una finestra di dialogo per selezionare un file LOCALE."""
        filepath = filedialog.askopenfilename( title="Seleziona file estrazioni 10eLotto Locale (.txt)", filetypes=(("Text files", "*.txt"), ("All files", "*.*")) )
        if filepath:
            self.file_path_var.set(filepath)
            self.log_message_gui(f"File 10eLotto locale selezionato: {filepath}")

    def log_message_gui(self, message):
        """Invia messaggio di log alla GUI in modo sicuro."""
        log_message(message, self.log_text, self.root)

    def set_result(self, numbers, attendibilita):
        """Aggiorna le etichette dei risultati in modo sicuro."""
        self.root.after(0, self._update_result_labels, numbers, attendibilita)

    def _update_result_labels(self, numbers, attendibilita):
        """Funzione helper eseguita da root.after per aggiornare le etichette."""
        try:
            if not self.root.winfo_exists() or not self.result_label.winfo_exists(): return

            if numbers and isinstance(numbers, list) and all(isinstance(n, int) for n in numbers):
                 result_str = "  ".join(map(lambda x: f"{x:02d}", numbers)) # Mantiene ordine prob.
                 self.result_label_var.set(result_str)
                 self.log_message_gui("\n" + "="*30 + "\nPREVISIONE 10ELOTTO GENERATA (Ord. Probabilità)\n" + "="*30)
            else:
                self.result_label_var.set("Previsione 10eLotto fallita. Controlla i log.")
                log_err = True
                if isinstance(attendibilita, str):
                     if any(kw in attendibilita.lower() for kw in ["attendibilità", "errore", "fallita", "annullata", "interrotta"]): log_err = False
                if log_err: self.log_message_gui("\nERRORE: Previsione 10eLotto non ha restituito numeri validi o ha fallito.")
            # Mostra sempre il messaggio di attendibilità/errore
            self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "Nessuna informazione sull'attendibilità.")

        except tk.TclError as e: print(f"TclError in _update_result_labels (likely during shutdown): {e}")
        except Exception as e: print(f"Error in _update_result_labels: {e}")


    # --- Metodo set_controls_state (MODIFICATO per aggiungere cv_splits_spinbox) ---
    def set_controls_state(self, state):
        """Imposta lo stato dei controlli in modo sicuro."""
        self.root.after(0, lambda: self._set_controls_state_tk(state))

    def _set_controls_state_tk(self, state):
        """Funzione helper eseguita da root.after per impostare lo stato."""
        try:
            if not self.root.winfo_exists(): return

            widgets_to_toggle = [
                self.browse_button, self.file_entry, self.seq_len_entry, self.num_predict_spinbox,
                self.run_button, self.check_button, self.loss_combo, self.optimizer_combo,
                self.dropout_spinbox, self.l1_entry, self.l2_entry, self.hidden_layers_entry,
                self.epochs_spinbox, self.batch_size_combo, self.patience_spinbox, self.min_delta_entry,
                self.check_colpi_spinbox,
                self.cv_splits_spinbox # NUOVO: Aggiunto spinbox CV
            ]
            # Gestione DateEntry/Entry
            date_widgets = []
            if HAS_TKCALENDAR and hasattr(self, 'start_date_entry') and hasattr(self.start_date_entry, 'configure'):
                date_widgets.extend([self.start_date_entry, self.end_date_entry])
            elif not HAS_TKCALENDAR and hasattr(self, 'start_date_entry'):
                 widgets_to_toggle.extend([self.start_date_entry, self.end_date_entry])

            # Applica stato ai widget principali
            for widget in widgets_to_toggle:
                 if widget is None or not widget.winfo_exists(): continue
                 widget_state = state
                 # Logica speciale per check_button
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
                     # Default a tk.DISABLED se lo stato richiesto non è tk.NORMAL
                     target_tk_state = tk.DISABLED
                     if widget_state == tk.NORMAL:
                         if isinstance(widget, (ttk.Combobox, ttk.Spinbox)):
                             target_tk_state = 'readonly'
                         elif isinstance(widget, ttk.Entry):
                             target_tk_state = tk.NORMAL
                         elif isinstance(widget, ttk.Button):
                             target_tk_state = tk.NORMAL
                         # Altri tipi di widget potrebbero avere stati diversi

                     # Cambia stato solo se necessario
                     if str(current_widget_state).lower() != str(target_tk_state).lower():
                         widget.config(state=target_tk_state)
                 except (tk.TclError, AttributeError) as e_widget:
                     print(f"Warning: Could not set state for widget {widget}: {e_widget}")
                     pass

            # Applica stato ai DateEntry (se tkcalendar è usato)
            for date_widget in date_widgets:
                if date_widget is None or not date_widget.winfo_exists(): continue
                try:
                    target_tk_state = tk.NORMAL if state == tk.NORMAL else tk.DISABLED
                    current_widget_state = date_widget.cget('state')
                    if str(current_widget_state).lower() != str(target_tk_state).lower():
                        date_widget.configure(state=target_tk_state)
                except (tk.TclError, AttributeError, Exception) as e_date:
                     print(f"Warning: Could not set state for DateEntry {date_widget}: {e_date}")
                     pass

        except tk.TclError as e: print(f"TclError in _set_controls_state_tk (likely during shutdown): {e}")
        except Exception as e: print(f"Error setting control states: {e}")


    # --- Metodo start_analysis_thread (MODIFICATO per leggere n_cv_splits) ---
    def start_analysis_thread(self):
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("Analisi in Corso", "Analisi 10eLotto già in esecuzione.", parent=self.root); return
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("Verifica in Corso", "Verifica 10eLotto in corso. Attendi.", parent=self.root); return

        #<editor-fold desc="Recupero e Validazione Parametri Analisi 10eLotto">
        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi 10eLotto in corso..."); self.attendibilita_label_var.set("")
        self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_date_str = None
        self.check_button.config(state=tk.DISABLED) # Disabilita subito

        # Recupero valori dai widget
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
        cv_splits_str = self.cv_splits_var.get() # NUOVO: Leggi CV splits

        # Validazione e conversione
        errors = []; sequence_length, num_predictions = 7, 10; hidden_layers_config = [256, 128]
        dropout_rate, l1_reg, l2_reg = 0.30, 0.0, 0.0; max_epochs, batch_size, patience, min_delta = 80, 64, 10, 0.0005
        n_cv_splits = DEFAULT_CV_SPLITS # NUOVO: Default CV splits

        if not data_source: errors.append("Specificare un URL Raw GitHub o un percorso file locale per 10eLotto.")
        elif not data_source.startswith(("http://", "https://")):
            if not os.path.exists(data_source): errors.append(f"File locale 10eLotto non trovato:\n{data_source}")
            elif not data_source.lower().endswith(".txt"): errors.append("Il file locale 10eLotto dovrebbe essere .txt.")

        try: start_dt = datetime.strptime(start_date_str, '%Y-%m-%d'); end_dt = datetime.strptime(end_date_str, '%Y-%m-%d'); assert start_dt <= end_dt
        except: errors.append("Date 10eLotto non valide o inizio > fine.")
        try: sequence_length = int(seq_len_str); assert 2 <= sequence_length <= 50 # Min 2
        except: errors.append("Seq. Input 10eLotto non valida (2-50).")
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
        try: batch_size = int(batch_size_str); assert batch_size >= 8 and (batch_size & (batch_size - 1) == 0) # Potenza di 2 >= 8
        except: errors.append("Batch Size 10eLotto non valido (potenza di 2 >= 8).")
        try: patience = int(patience_str); assert patience >= 3
        except: errors.append("Patience 10eLotto non valida (>= 3).")
        try: min_delta = float(min_delta_str); assert min_delta >= 0
        except: errors.append("Min Delta 10eLotto non valido (>= 0).")
        # NUOVO: Validazione CV splits
        try: n_cv_splits = int(cv_splits_str); assert 2 <= n_cv_splits <= 20
        except: errors.append(f"Numero CV Splits non valido (intero 2-20).")


        if errors: messagebox.showerror("Errore Parametri Input 10eLotto", "\n\n".join(errors), parent=self.root); self.result_label_var.set("Errore parametri."); return
        #</editor-fold>

        self.set_controls_state(tk.DISABLED)
        source_type = "URL" if data_source.startswith("http") else "File Locale"
        self.log_message_gui("=== Avvio Analisi 10eLotto (FE & CV) ===")
        self.log_message_gui(f"Sorgente Dati: {source_type} ({data_source})")
        self.log_message_gui(f"Param Dati: Date={start_date_str}-{end_date_str}, SeqIn={sequence_length}, NumOut={num_predictions}")
        self.log_message_gui(f"Param Modello: Layers={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate:.2f}, L1={l1_reg:.4f}, L2={l2_reg:.4f}")
        self.log_message_gui(f"Param Training: Epochs={max_epochs}, Batch={batch_size}, CV Splits={n_cv_splits}, Pat={patience}, MinDelta={min_delta:.5f}") # Log CV Splits
        self.log_message_gui("-" * 40)

        # Resetta evento di stop e avvia thread
        self._stop_event_analysis.clear()
        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=( # Passa tutti i parametri, incluso n_cv_splits e stop_event
                data_source, start_date_str, end_date_str, sequence_length,
                loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                num_predictions, n_cv_splits, # Passa n_cv_splits
                self._stop_event_analysis # Passa l'evento
            ),
            daemon=True,
            name="AnalysisThread"
        )
        self.analysis_thread.start()

    # --- Metodo run_analysis (MODIFICATO per passare n_cv_splits) ---
    def run_analysis(self, data_source, start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions, n_cv_splits, # Riceve n_cv_splits
                     stop_event):
        """Esegue l'analisi 10eLotto nel thread, controllando stop_event."""
        numeri_predetti, attendibilita_msg, success = None, "Analisi 10eLotto non completata", False
        last_update_date = "N/A"
        try:
            if stop_event.is_set():
                self.log_message_gui("Analisi annullata prima dell'inizio.")
                attendibilita_msg = "Analisi annullata"
                return

            # --- Carica dati per data max ---
            df_full, _, _, _ = carica_dati_10elotto(data_source, start_date=None, end_date=None, log_callback=None) # Carica silenzioso
            if df_full is not None and not df_full.empty:
                last_update_date = df_full['Data'].max().strftime('%Y-%m-%d')
            self.root.after(0, self.last_update_label_var.set, f"Ultimo aggiornamento estrazionale: {last_update_date}")

            if stop_event.is_set():
                self.log_message_gui("Analisi annullata dopo caricamento data max.")
                attendibilita_msg = "Analisi annullata"
                return

            # --- Esegui l'analisi vera e propria con CV ---
            numeri_predetti, attendibilita_msg = analisi_10elotto(
                file_path=data_source, start_date=start_date, end_date=end_date,
                sequence_length=sequence_length, loss_function=loss_function,
                optimizer=optimizer, dropout_rate=dropout_rate, l1_reg=l1_reg,
                l2_reg=l2_reg, hidden_layers_config=hidden_layers_config,
                max_epochs=max_epochs, batch_size=batch_size, patience=patience,
                min_delta=min_delta, num_predictions=num_predictions,
                n_cv_splits=n_cv_splits, # Passa n_cv_splits
                log_callback=self.log_message_gui,
                stop_event=stop_event # Passa l'evento
            )

            # Verifica se annullato DALLA funzione analisi_10elotto
            if stop_event.is_set() and numeri_predetti is None:
                 self.log_message_gui("Analisi interrotta durante l'elaborazione.")
                 # attendibilita_msg dovrebbe già essere impostato da analisi_10elotto
                 if not attendibilita_msg or "interrotta" not in attendibilita_msg.lower():
                     attendibilita_msg = "Analisi Interrotta" # Messaggio di fallback
                 success = False
            else:
                 # Valuta successo normale
                 success = isinstance(numeri_predetti, list) and len(numeri_predetti) == num_predictions and all(isinstance(n, int) for n in numeri_predetti)

        except Exception as e:
            self.log_message_gui(f"\nERRORE CRITICO run_analysis 10eLotto: {e}\n{traceback.format_exc()}")
            attendibilita_msg = f"Errore critico 10eLotto: {e}"; success = False
        finally:
            # Log completamento solo se non annullato
            if not stop_event.is_set():
                 self.log_message_gui("\n=== Analisi 10eLotto (FE & CV) Completata ===")

            # Aggiorna risultati e controlli
            self.set_result(numeri_predetti, attendibilita_msg)

            if success: # Solo se successo E non annullato
                self.last_prediction = numeri_predetti
                try:
                    self.last_prediction_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                    self.last_prediction_date_str = end_date
                    self.log_message_gui(f"Previsione 10eLotto salvata (dati fino a {end_date}).")
                except ValueError:
                    self.log_message_gui(f"ATTENZIONE: Errore salvataggio data fine 10eLotto ({end_date}). Verifica non possibile."); success = False
            else:
                 self.last_prediction = None; self.last_prediction_end_date = None; self.last_prediction_date_str = None
                 if not stop_event.is_set() and (not attendibilita_msg or not any(kw in attendibilita_msg.lower() for kw in ["errore", "fallita", "annullata", "interrotta"])):
                     self.log_message_gui("Analisi 10eLotto fallita o risultato non valido.")

            # Riabilita controlli e pulisci ref thread
            self.set_controls_state(tk.NORMAL)
            self.root.after(10, self._clear_analysis_thread_ref) # Usa helper


    # --- Metodi _clear_analysis_thread_ref, start_check_thread, run_check_results, _clear_check_thread_ref, on_close (INVARIATI) ---
    def _clear_analysis_thread_ref(self):
        """Helper per pulire il riferimento al thread nel thread principale."""
        self.analysis_thread = None
        # Ri-valuta stato controlli
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

        # Resetta evento di stop e avvia thread
        self._stop_event_check.clear()
        self.check_thread = threading.Thread(
            target=self.run_check_results,
            args=( data_source_for_check, self.last_prediction, self.last_prediction_date_str, num_colpi,
                   self._stop_event_check ), # Passa l'evento
            daemon=True,
            name="CheckThread"
        )
        self.check_thread.start()

    def run_check_results(self, data_source, prediction_to_check, last_analysis_date_str, num_colpi_to_check, stop_event):
        """Carica dati successivi e verifica la previsione, controllando stop_event."""
        try:
            try:
                last_date = datetime.strptime(last_analysis_date_str, '%Y-%m-%d'); check_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            except ValueError as ve: self.log_message_gui(f"ERRORE formato data analisi 10eLotto: {ve}"); return

            if stop_event.is_set(): self.log_message_gui("Verifica annullata prima del caricamento dati."); return

            self.log_message_gui(f"Caricamento dati 10eLotto per verifica da {check_start_date}...")
            df_check, numeri_array_check, _, _ = carica_dati_10elotto( data_source, start_date=check_start_date, end_date=None, log_callback=self.log_message_gui )

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
                if stop_event.is_set():
                    self.log_message_gui(f"Verifica interrotta al colpo {colpo_counter + 1}.")
                    break

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

            if not stop_event.is_set():
                 self.log_message_gui("-" * 40)
                 if found_hits_total == 0: self.log_message_gui(f"Nessun colpo vincente nei {num_to_run} colpi 10eLotto verificati.")
                 else: self.log_message_gui(f"Verifica {num_to_run} colpi 10eLotto completata. Trovati {found_hits_total} colpi con risultati. Punteggio massimo: {highest_score} punti.")

        except Exception as e:
            self.log_message_gui(f"ERRORE CRITICO verifica 10eLotto: {e}\n{traceback.format_exc()}")
        finally:
             if not stop_event.is_set():
                 self.log_message_gui("\n=== Verifica 10eLotto Completata ===")
             self.set_controls_state(tk.NORMAL)
             self.root.after(10, self._clear_check_thread_ref) # Usa helper

    def _clear_check_thread_ref(self):
        """Helper per pulire il riferimento al thread nel thread principale."""
        self.check_thread = None
        self._set_controls_state_tk(tk.NORMAL)

    def on_close(self):
        """Gestisce la richiesta di chiusura della finestra (pulsante X)."""
        self.log_message_gui("Richiesta chiusura finestra...")

        # 1. Segnala stop
        self._stop_event_analysis.set()
        self._stop_event_check.set()

        # 2. Attendi thread (con timeout)
        timeout_secs = 3.0
        wait_start = time.time()
        threads_to_wait = []
        analysis_thread_local = self.analysis_thread
        if analysis_thread_local and analysis_thread_local.is_alive(): threads_to_wait.append(analysis_thread_local)
        check_thread_local = self.check_thread
        if check_thread_local and check_thread_local.is_alive(): threads_to_wait.append(check_thread_local)

        if threads_to_wait:
            self.log_message_gui(f"Attendo terminazione thread: {[t.name for t in threads_to_wait]} (max {timeout_secs}s)")
            for thread in threads_to_wait:
                remaining_timeout = max(0.1, timeout_secs - (time.time() - wait_start))
                try:
                    thread.join(timeout=remaining_timeout)
                    log_level = "ATTENZIONE" if thread.is_alive() else "INFO"
                    status = "non terminato (timeout)" if thread.is_alive() else "terminato"
                    self.log_message_gui(f"{log_level}: Thread {thread.name} {status}.")
                except Exception as e:
                    self.log_message_gui(f"Errore durante join di {thread.name}: {e}")
        else:
            self.log_message_gui("Nessun thread attivo da attendere.")

        # 3. Distruggi finestra
        self.log_message_gui("Distruzione finestra Tkinter.")
        try:
             self.analysis_thread = None
             self.check_thread = None
             self.root.destroy()
        except tk.TclError as e: print(f"TclError durante root.destroy(): {e}")
        except Exception as e: print(f"Errore imprevisto durante root.destroy(): {e}")

# --- Funzione di Lancio (INVARIATA) ---
def launch_10elotto_window(parent_window):
    """Crea e lancia la finestra dell'applicazione 10eLotto come Toplevel."""
    try:
        lotto_win = tk.Toplevel(parent_window)
        app_instance = App10eLotto(lotto_win)
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
    print("NOTA: Questo script richiede l'installazione delle librerie 'requests', 'tensorflow', 'pandas', 'numpy', 'scikit-learn'.")
    print("      Potrebbe richiedere anche 'tkcalendar' (opzionale).")
    print("Puoi installarle con: pip install requests tensorflow pandas numpy scikit-learn tkcalendar")
    try:
        if sys.platform == "win32": from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e_dpi: print(f"Nota: impossibile impostare DPI awareness ({e_dpi})")

    root_standalone = tk.Tk()
    app_standalone = App10eLotto(root_standalone)
    root_standalone.mainloop()