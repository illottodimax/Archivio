# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
from tensorflow.keras import regularizers
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from collections import Counter
import threading
import traceback
import requests  # Richiede: pip install requests

# NUOVO: Import per FE e CV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import time # Per on_close

# Opzionale, ma consigliato per GUI migliori: pip install tkcalendar
try:
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

DEFAULT_SUPERENALOTTO_CHECK_COLPI = 5
DEFAULT_SUPERENALOTTO_DATA_URL = "https://raw.githubusercontent.com/illottodimax/Archivio/main/it-superenalotto-past-draws-archive.txt"
# NUOVO: Default per K-Fold Cross-Validation
DEFAULT_SUPERENALOTTO_CV_SPLITS = 5

# --- Funzioni Globali (Seed, Log - INVARIATE) ---
def set_seed(seed_value=42):
    """Imposta il seme per la riproducibilità dei risultati."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_seed() # Imposta il seed all'avvio

def log_message(message, log_widget, window):
    """Aggiunge un messaggio al widget di log nella GUI in modo sicuro per i thread."""
    if log_widget and window:
        try:
            # Usa after per assicurare l'esecuzione nel thread principale della GUI
            window.after(10, lambda: _update_log_widget(log_widget, message))
        except tk.TclError: # Finestra potrebbe essere stata distrutta
            print(f"Log GUI TclError (Window destroyed?): {message}")


def _update_log_widget(log_widget, message):
    """Funzione helper per aggiornare il widget di log."""
    try:
        # Verifica se il widget esiste ancora prima di modificarlo
        if not log_widget.winfo_exists():
             # print(f"Log GUI widget destroyed, message lost: {message}") # Commentato
             return

        # Abilita temporaneamente il widget per l'inserimento
        current_state = log_widget.cget('state')
        log_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, str(message) + "\n")
        log_widget.see(tk.END) # Scrolla fino alla fine
        # Ripristina lo stato originale (di solito DISABLED per impedire input utente)
        if current_state == tk.DISABLED:
             log_widget.config(state=tk.DISABLED)
    except tk.TclError as e:
        # Fallback se la GUI non è disponibile (es. chiusura finestra nel mezzo di 'after')
        print(f"Log GUI TclError (likely during shutdown): {e} - Message: {message}")
    except Exception as e:
        # Gestisci altri errori imprevisti
        print(f"Log GUI unexpected error: {e}\nMessage: {message}")
        # Tentativo di ripristino stato disabilitato
        try:
            if log_widget.winfo_exists() and log_widget.cget('state') == tk.NORMAL:
                 log_widget.config(state=tk.DISABLED)
        except: pass

def carica_dati_superenalotto(data_source, start_date=None, end_date=None, log_callback=None):
    """
    Carica i dati del SuperEnalotto da un URL (RAW GitHub) o da un file locale.
    Formato atteso: YYYY-MM-DD N1 N2 N3 N4 N5 N6 '' JJ SS
    Restituisce: DataFrame pulito, array numeri principali, array jolly, array superstar, data ultimo aggiornamento.
    (Versione con correzione SyntaxError nel blocco except)
    """
    lines = []
    is_url = data_source.startswith("http://") or data_source.startswith("https://")

    try:
        if is_url:
            if log_callback: log_callback(f"Tentativo caricamento dati SuperEnalotto da URL: {data_source}")
            try:
                response = requests.get(data_source, timeout=30)
                response.raise_for_status()
                content = response.text
                lines = content.splitlines()
                if log_callback: log_callback(f"Dati scaricati con successo ({len(lines)} righe). Encoding presunto: {response.encoding}")
            except requests.exceptions.Timeout:
                 if log_callback: log_callback(f"ERRORE HTTP Timeout: Impossibile scaricare i dati da {data_source} in 30 secondi.")
                 return None, None, None, None, None
            except requests.exceptions.RequestException as e_req:
                if log_callback: log_callback(f"ERRORE HTTP: Impossibile scaricare i dati da {data_source} - {e_req}")
                return None, None, None, None, None
            except Exception as e_url:
                if log_callback: log_callback(f"ERRORE generico durante il download da URL: {e_url}")
                return None, None, None, None, None
        else:
            # Caricamento da file locale
            file_path = data_source
            if log_callback: log_callback(f"Tentativo caricamento dati SuperEnalotto da file locale: {file_path}")
            if not os.path.exists(file_path):
                 if log_callback: log_callback(f"ERRORE: File locale non trovato - {file_path}")
                 return None, None, None, None, None
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
            file_read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f: lines = f.readlines()
                    file_read_success = True
                    if log_callback: log_callback(f"File locale letto con successo usando encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    if log_callback: log_callback(f"Info: Encoding {enc} fallito, provo il prossimo.")
                    continue
                except Exception as e_file:
                    if log_callback: log_callback(f"ERRORE durante lettura file locale con encoding {enc}: {e_file}")
                    continue
            if not file_read_success:
                 if log_callback: log_callback("ERRORE CRITICO: Impossibile leggere il file locale con gli encoding noti.")
                 return None, None, None, None, None

        # --- Parsing delle righe ---
        if log_callback: log_callback(f"Inizio parsing di {len(lines)} righe...")
        if not lines or len(lines) < 2:
            if log_callback: log_callback("ERRORE: Dati vuoti o solo intestazione.")
            return None, None, None, None, None

        data_lines = lines[1:]
        processed_data = []
        malformed_lines_count = 0
        processed_lines_count = 0
        min_expected_fields = 10

        for i, line in enumerate(data_lines):
            values = line.strip().split('\t')
            if len(values) >= min_expected_fields:
                try:
                    date_val = values[0].strip()
                    num_vals_str = [v.strip() for v in values[1:7]] # Num1-Num6
                    jolly_val_str = values[8].strip() # Jolly
                    superstar_val_str = values[9].strip() # SuperStar
                    datetime.strptime(date_val, '%Y-%m-%d')
                    if not all(n.isdigit() for n in num_vals_str): raise ValueError("Num1-6 non numerici")
                    if not jolly_val_str.isdigit(): raise ValueError("Jolly non numerico")
                    if not superstar_val_str.isdigit(): raise ValueError("SuperStar non numerico")
                    num_vals = [int(n) for n in num_vals_str]
                    jolly_val = int(jolly_val_str)
                    superstar_val = int(superstar_val_str)
                    if not all(1 <= n <= 90 for n in num_vals + [jolly_val, superstar_val]):
                        raise ValueError("Numeri fuori range (1-90)")
                    clean_row = [date_val] + num_vals_str + [jolly_val_str, superstar_val_str]
                    processed_data.append(clean_row)
                    processed_lines_count += 1
                except (IndexError, ValueError, TypeError) as e_parse:
                    malformed_lines_count += 1
                    if malformed_lines_count <= 5 and log_callback: log_callback(f"ATT: Riga {i+2} scartata (Parse Err: {e_parse}). Val: '{line.strip()}'")
                    elif malformed_lines_count == 6 and log_callback: log_callback("ATT: Ulteriori errori parsing non loggati.")
            else:
                malformed_lines_count += 1
                if malformed_lines_count <= 5 and log_callback: log_callback(f"ATT: Riga {i+2} scartata (Campi < {min_expected_fields}, trovati {len(values)}). Val: '{line.strip()}'")
                elif malformed_lines_count == 6 and log_callback: log_callback("ATT: Ulteriori errori campi insuff. non loggati.")

        if malformed_lines_count > 0 and log_callback: log_callback(f"ATTENZIONE: {malformed_lines_count} righe totali scartate durante il parsing.")
        if not processed_data:
            if log_callback: log_callback("ERRORE: Nessuna riga dati valida trovata dopo il parsing.")
            return None, None, None, None, None
        else:
            if log_callback: log_callback(f"Parsing completato: {processed_lines_count} righe elaborate.")

        # --- Crea DataFrame ---
        colonne_superenalotto = ['Data'] + [f'Num{i+1}' for i in range(6)] + ['Jolly', 'SuperStar']
        try:
            df = pd.DataFrame(processed_data, columns=colonne_superenalotto)
            if log_callback: log_callback(f"Creato DataFrame. Shape iniziale: {df.shape}")
        except Exception as e_df:
             if log_callback: log_callback(f"ERRORE CRITICO creazione DataFrame: {e_df}.");
             return None, None, None, None, None

        # --- Pulizia Tipi e Date ---
        try:
            df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d', errors='coerce')
            rows_before_na = len(df); df = df.dropna(subset=['Data']); rows_after_na = len(df)
            if rows_before_na > rows_after_na and log_callback: log_callback(f"Rimosse {rows_before_na - rows_after_na} righe con date non valide.")
            if df.empty:
                if log_callback: log_callback("ERRORE: Nessun dato valido rimasto dopo pulizia date.")
                return df.copy(), None, None, None, None
            df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)
            if log_callback: log_callback(f"DataFrame ordinato per data. Righe valide: {len(df)}")

            num_cols = [f'Num{i+1}' for i in range(6)] + ['Jolly', 'SuperStar']
            rows_before_num = len(df)
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].astype('Int64', errors='ignore')

            df_cleaned = df.dropna(subset=num_cols[:6]) # Num1-6 obbligatori
            rows_after_num = len(df_cleaned)
            if rows_before_num > rows_after_num and log_callback: log_callback(f"Rimosse {rows_before_num - rows_after_num} righe con Num1-6 non validi.")
            if df_cleaned.empty:
                 if log_callback: log_callback("ERRORE: Nessun dato valido rimasto dopo pulizia numeri principali.")
                 return df_cleaned.copy(), None, None, None, None

            rows_before_range = len(df_cleaned)
            for col in num_cols:
                if df_cleaned[col].notna().any():
                    df_cleaned = df_cleaned[df_cleaned[col].isna() | ((df_cleaned[col] >= 1) & (df_cleaned[col] <= 90))]
            rows_after_range = len(df_cleaned)
            if rows_before_range > rows_after_range and log_callback: log_callback(f"Rimosse {rows_before_range - rows_after_range} righe con numeri fuori range (1-90).")
            if df_cleaned.empty:
                 if log_callback: log_callback("ERRORE: Nessun dato valido rimasto dopo controllo range 1-90.")
                 return df_cleaned.copy(), None, None, None, None
            if log_callback: log_callback(f"Pulizia tipi e range completata. Righe finali per l'analisi: {len(df_cleaned)}")
        except Exception as e_clean:
            if log_callback: log_callback(f"ERRORE durante pulizia tipi/date/numeri: {e_clean}\n{traceback.format_exc()}")
            return None, None, None, None, None

        # --- Filtraggio Date Utente ---
        df_filtered = df_cleaned.copy()
        rows_before_filter = len(df_filtered)
        if log_callback: log_callback(f"Applicazione filtro date utente: Start={start_date}, End={end_date}")
        if start_date:
            try:
                df_filtered = df_filtered[df_filtered['Data'] >= pd.to_datetime(start_date)]
            except Exception as e_start:
                # CORREZIONE: Indentazione corretta
                if log_callback:
                    log_callback(f"Errore filtro data inizio (verrà ignorato): {e_start}")
        if end_date:
             try:
                 df_filtered = df_filtered[df_filtered['Data'] <= pd.to_datetime(end_date)]
             except Exception as e_end:
                 # CORREZIONE: Indentazione corretta
                 if log_callback:
                     log_callback(f"Errore filtro data fine (verrà ignorato): {e_end}")

        rows_after_filter = len(df_filtered)
        if log_callback: log_callback(f"Righe dopo filtro date utente: {rows_after_filter} (rimosse {rows_before_filter - rows_after_filter})")
        if df_filtered.empty:
            if log_callback: log_callback("INFO: Nessuna riga rimasta nel DataFrame dopo il filtro per data.")
            # Determina last_update_date dal df PRIMA del filtro se possibile
            last_update_before_filter = df_cleaned['Data'].max() if not df_cleaned.empty else None
            return df_filtered.copy(), None, None, None, last_update_before_filter # Restituisce data prima del filtro

                # --- Estrazione Array Finali ---
        numeri_principali_cols = [f'Num{i+1}' for i in range(6)]
        numeri_principali_array, numeri_jolly, numeri_superstar = None, None, None
        try:
            numeri_principali_array = df_filtered[numeri_principali_cols].values.astype(int)
            if log_callback: log_callback(f"Estratto array numeri principali. Shape: {numeri_principali_array.shape}")

            # Estrazione Jolly con correzione sintassi
            if 'Jolly' in df_filtered.columns and df_filtered['Jolly'].notna().any():
                numeri_jolly = df_filtered.dropna(subset=['Jolly'])['Jolly'].values.astype(int)
                if log_callback: log_callback(f"Estratto array Jolly. Shape: {numeri_jolly.shape}")
            else:
                # CORREZIONE: Indentazione corretta
                if log_callback:
                     log_callback("Colonna Jolly non presente/vuota nel set filtrato.")

            # Estrazione SuperStar con correzione sintassi
            if 'SuperStar' in df_filtered.columns and df_filtered['SuperStar'].notna().any():
                numeri_superstar = df_filtered.dropna(subset=['SuperStar'])['SuperStar'].values.astype(int)
                if log_callback: log_callback(f"Estratto array SuperStar. Shape: {numeri_superstar.shape}")
            else:
                 # CORREZIONE: Indentazione corretta
                 if log_callback:
                      log_callback("Colonna SuperStar non presente/vuota nel set filtrato.")

        except Exception as e_extract:
            if log_callback: log_callback(f"ERRORE durante estrazione array finali: {e_extract}")
            # In caso di errore qui, restituiamo ciò che abbiamo (df filtrato) e None per gli array
            # Ottieni comunque la data max dal df filtrato prima dell'errore di estrazione, se possibile
            last_update_date_on_error = df_filtered['Data'].max() if not df_filtered.empty else None
            return df_filtered.copy(), None, None, None, last_update_date_on_error

        last_update_date = df_filtered['Data'].max() if not df_filtered.empty else None
        return df_filtered.copy(), numeri_principali_array, numeri_jolly, numeri_superstar, last_update_date

    except Exception as e_main:
        if log_callback: log_callback(f"Errore GRAVE non gestito in carica_dati_superenalotto: {e_main}\n{traceback.format_exc()}")
        return None, None, None, None, None

# NUOVO: Funzione per Feature Engineering Superenalotto
def engineer_features_superenalotto(numeri_principali_array, log_callback=None):
    """
    Crea features aggiuntive dall'array dei 6 numeri principali.
    Input: numeri_principali_array (N_draws, 6)
    Output: combined_features (N_draws, 6 + N_new_features)
    """
    if numeri_principali_array is None or numeri_principali_array.ndim != 2 or numeri_principali_array.shape[1] != 6:
        if log_callback: log_callback("ERRORE (engineer_features_se): Input numeri_principali_array non valido.")
        return None

    if log_callback: log_callback(f"Inizio Feature Engineering SuperEnalotto su {numeri_principali_array.shape[0]} estrazioni...")

    try:
        # Features base per riga
        draw_sum = np.sum(numeri_principali_array, axis=1, keepdims=True)
        draw_mean = np.mean(numeri_principali_array, axis=1, keepdims=True)
        odd_count = np.sum(numeri_principali_array % 2 != 0, axis=1, keepdims=True)
        even_count = 6 - odd_count
        low_count = np.sum((numeri_principali_array >= 1) & (numeri_principali_array <= 45), axis=1, keepdims=True)
        high_count = 6 - low_count
        # range_val = np.max(numeri_principali_array, axis=1, keepdims=True) - np.min(numeri_principali_array, axis=1, keepdims=True) # Range può essere interessante

        # Combina le nuove features
        engineered_features = np.concatenate([
            draw_sum,
            draw_mean,
            odd_count,
            even_count,
            low_count,
            high_count
            # range_val
        ], axis=1)

        # Combina le features ingegnerizzate con i numeri originali
        combined_features = np.concatenate([numeri_principali_array, engineered_features], axis=1)

        if log_callback: log_callback(f"Feature Engineering SuperEnalotto completato. Shape finale features: {combined_features.shape}")
        return combined_features.astype(np.float32) # Converti in float per scaler

    except Exception as e:
        if log_callback: log_callback(f"ERRORE durante Feature Engineering SuperEnalotto: {e}\n{traceback.format_exc()}")
        return None


# MODIFICATO: Preparazione Sequenze per Modello Superenalotto
def prepara_sequenze_per_modello_superenalotto(input_feature_array, target_number_array, sequence_length=5, log_callback=None):
    """
    Prepara le sequenze per il modello SuperEnalotto usando features combinate.
    Input:
        input_feature_array: Array con le features (numeri + engineered), shape (N_draws, N_features). ASSUME GIÀ SCALATO.
        target_number_array: Array originale dei numeri estratti (usato per il target), shape (N_draws, 6)
        sequence_length: Lunghezza della sequenza di input.
    Output:
        X: Array delle sequenze di input, shape (N_sequences, sequence_length * N_features)
        y: Array target (multi-hot encoded, 6 numeri su 90), shape (N_sequences, 90)
    """
    if log_callback: log_callback(f"Avvio preparazione sequenze SuperEnalotto (SeqLen={sequence_length})...")

    # Validazione input
    if input_feature_array is None or target_number_array is None:
        if log_callback: log_callback("ERRORE (prep_seq_se): Input array (features o target) mancante.")
        return None, None
    if input_feature_array.ndim != 2 or target_number_array.ndim != 2:
        if log_callback: log_callback("ERRORE (prep_seq_se): Input arrays devono avere 2 dimensioni.")
        return None, None
    if input_feature_array.shape[0] != target_number_array.shape[0]:
        if log_callback: log_callback("ERRORE (prep_seq_se): Disallineamento righe tra feature array e target array.")
        return None, None
    if target_number_array.shape[1] != 6:
         if log_callback: log_callback(f"ERRORE (prep_seq_se): Target number array non ha 6 colonne (shape: {target_number_array.shape}).")
         return None, None

    n_features = input_feature_array.shape[1]
    num_estrazioni = len(input_feature_array)
    if log_callback: log_callback(f"Numero estrazioni disponibili: {num_estrazioni}, Num Features per estrazione: {n_features}")

    if num_estrazioni <= sequence_length:
        msg = f"ERRORE: Estrazioni insuff. ({num_estrazioni}) per seq_len ({sequence_length}). Servono {sequence_length + 1}."
        if log_callback: log_callback(msg)
        return None, None

    X, y = [], []
    valid_sequences_count = 0
    invalid_target_count = 0

    for i in range(num_estrazioni - sequence_length):
        # Sequenza di input dalle features combinate (e scalate)
        input_seq = input_feature_array[i : i + sequence_length]
        # Target dall'array originale dei numeri
        target_extraction = target_number_array[i + sequence_length]

        # Validazione target
        if len(target_extraction) == 6 and np.all((target_extraction >= 1) & (target_extraction <= 90)):
            # Crea target multi-hot (90 elementi, 6 attivi)
            target_vector = np.zeros(90, dtype=np.int8)
            target_vector[target_extraction - 1] = 1 # Indici 0-89 per numeri 1-90

            # Appiattisci input e aggiungi
            X.append(input_seq.flatten())
            y.append(target_vector)
            valid_sequences_count += 1
        else:
            invalid_target_count += 1
            if invalid_target_count <= 5 and log_callback: log_callback(f"ATT (prep_seq_se): Scartata seq indice {i}. Target non valido: {target_extraction}")
            elif invalid_target_count == 6 and log_callback: log_callback("ATT (prep_seq_se): Ulteriori errori target non loggati.")

    if invalid_target_count > 0 and log_callback: log_callback(f"ATTENZIONE: Scartate {invalid_target_count} sequenze totali (target non valido).")
    if not X:
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata."); return None, None

    if log_callback: log_callback(f"Create {valid_sequences_count} sequenze Input/Target valide.")
    try:
        # X contiene già float32 dallo scaler, y è int8
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.int8)
        if log_callback: log_callback(f"Array NumPy creati. Shape X: {X_np.shape}, Shape y: {y_np.shape}")
        return X_np, y_np
    except Exception as e_np_conv:
        if log_callback: log_callback(f"ERRORE durante conversione finale a NumPy array: {e_np_conv}")
        return None, None


# --- Costruzione Modello Keras (build_model_superenalotto INVARIATA NELLA LOGICA) ---
# Riceverà un input_shape diverso (N features * seq_len), ma la struttura resta uguale.
def build_model_superenalotto(input_shape, hidden_layers=[512, 256, 128], loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    """
    Costruisce il modello Keras (DNN) per SuperEnalotto.
    (Logica interna invariata, ma riceve shape aggiornato)
    """
    if log_callback: log_callback(f"Costruzione modello SuperEnalotto: InputShape={input_shape}, Layers={hidden_layers}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")

    # Validazioni (omesse per brevità, sono uguali a prima)
    if not isinstance(input_shape, tuple) or len(input_shape) != 1 or input_shape[0] <= 0: raise ValueError(f"Invalid input_shape {input_shape}")
    if not isinstance(hidden_layers, list) or not all(isinstance(u, int) and u > 0 for u in hidden_layers): raise ValueError("Invalid hidden_layers")
    # ... altre validazioni ...

    model = tf.keras.Sequential(name="Modello_SuperEnalotto_DNN_FE")
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input_Layer"))
    kernel_regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None

    if not hidden_layers:
        if log_callback: log_callback("ATTENZIONE: Nessun hidden layer specificato.")
    else:
        for i, units in enumerate(hidden_layers):
            layer_num = i + 1
            model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=kernel_regularizer, name=f"Dense_{layer_num}_{units}"))
            model.add(tf.keras.layers.BatchNormalization(name=f"BatchNorm_{layer_num}"))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate, name=f"Dropout_{layer_num}_{dropout_rate:.2f}"))

    model.add(tf.keras.layers.Dense(90, activation='sigmoid', name="Output_Layer_90_Sigmoid"))

    try:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy']) # Accuracy qui è approssimativa
        if log_callback: log_callback(f"Modello SuperEnalotto compilato (Optimizer: {optimizer}, Loss: {loss_function}).")
    except Exception as e_generic_compile:
        if log_callback: log_callback(f"ERRORE generico compilazione modello: {e_generic_compile}")
        return None

    if log_callback:
        try:
            stringlist = []; model.summary(print_fn=lambda x: stringlist.append(x))
            log_callback("Riepilogo Modello SuperEnalotto:\n" + "\n".join(stringlist))
        except Exception as e_summary: log_callback(f"ATT: Impossibile generare riepilogo modello: {e_summary}")

    return model

def calcola_numeri_spia_superenalotto(numeri_array_storico, date_array_storico, numeri_spia_target,
                                     colpi_successivi, top_n_singoli, top_n_ambi, top_n_terni,
                                     log_callback=None, stop_event=None):
    """
    Analizza lo storico del SuperEnalotto per trovare singoli, ambi e terni 
    che seguono i numeri spia.
    Args:
        numeri_array_storico (np.array): Array (N_draws, 6) delle estrazioni storiche (solo i 6 principali).
    Restituisce: (list_singoli, list_ambi, list_terni, occorrenze_spia, data_inizio, data_fine)
    """
    if numeri_array_storico is None or len(numeri_array_storico) == 0:
        if log_callback: log_callback("ERRORE (Spia SE): Dati storici SuperEnalotto mancanti.")
        return [], [], [], 0, None, None
    if not numeri_spia_target:
        if log_callback: log_callback("ERRORE (Spia SE): Nessun numero spia fornito.")
        return [], [], [], 0, None, None
    if numeri_array_storico.shape[1] != 6:
        if log_callback: log_callback(f"ERRORE (Spia SE): L'array storico deve avere 6 colonne, ne ha {numeri_array_storico.shape[1]}.")
        return [], [], [], 0, None, None


    num_estrazioni = len(numeri_array_storico)
    if num_estrazioni <= colpi_successivi:
        if log_callback: log_callback(f"ATT (Spia SE): Dati insuff. ({num_estrazioni} estr.) per analizzare {colpi_successivi} colpi.")
        return [], [], [], 0, None, None

    frequenze_singoli = Counter()
    frequenze_ambi = Counter()
    frequenze_terni = Counter()
    occorrenze_spia_trovate = 0
    
    data_inizio_scan, data_fine_scan = "N/A", "N/A"
    if date_array_storico is not None and len(date_array_storico) == num_estrazioni:
        try:
            valid_dates = date_array_storico[~pd.isna(date_array_storico)]
            if len(valid_dates) > 0:
                data_inizio_scan = pd.to_datetime(valid_dates.min()).strftime('%Y-%m-%d')
                data_fine_scan = pd.to_datetime(valid_dates.max()).strftime('%Y-%m-%d')
        except Exception: pass 
    if log_callback: log_callback(f"Analisi Spia SE: Scansione {num_estrazioni} estrazioni (Periodo: {data_inizio_scan} - {data_fine_scan}).")

    numeri_spia_set = set(numeri_spia_target)

    for i in range(num_estrazioni - colpi_successivi):
        if stop_event and stop_event.is_set(): break
        
        estrazione_corrente_set = set(numeri_array_storico[i])
        if numeri_spia_set.issubset(estrazione_corrente_set): # Se tutti i numeri spia sono presenti
            occorrenze_spia_trovate += 1
            for k in range(1, colpi_successivi + 1):
                if stop_event and stop_event.is_set(): break
                idx_succ = i + k
                if idx_succ < num_estrazioni:
                    estrazione_succ = numeri_array_storico[idx_succ]
                    # Numeri validi (1-90) nell'estrazione successiva, escludendo i numeri spia stessi
                    estrazione_succ_valid_numeri = [n for n in estrazione_succ if 1 <= n <= 90 and n not in numeri_spia_set]
                    
                    frequenze_singoli.update(estrazione_succ_valid_numeri)
                    if len(estrazione_succ_valid_numeri) >= 2:
                        frequenze_ambi.update(itertools.combinations(sorted(estrazione_succ_valid_numeri), 2))
                    if len(estrazione_succ_valid_numeri) >= 3:
                        frequenze_terni.update(itertools.combinations(sorted(estrazione_succ_valid_numeri), 3))
        
        if i > 0 and i % 1000 == 0 and log_callback: log_callback(f"Analisi Spia SE: Elaborate {i}/{num_estrazioni - colpi_successivi}...")

    if log_callback and not (stop_event and stop_event.is_set()):
         log_callback(f"Analisi Spia SE: Trovate {occorrenze_spia_trovate} occorrenze dei numeri spia {numeri_spia_target}.")
    
    return (frequenze_singoli.most_common(top_n_singoli),
            frequenze_ambi.most_common(top_n_ambi),
            frequenze_terni.most_common(top_n_terni),
            occorrenze_spia_trovate, data_inizio_scan, data_fine_scan)


# --- Callback per Logging Epoche (LogCallback INVARIATA) ---
class LogCallback(tf.keras.callbacks.Callback):
    """Callback Keras per inviare i log delle epoche alla funzione di log della GUI."""
    def __init__(self, log_callback_func, stop_event=None): # Aggiunto stop_event opzionale
         super().__init__()
         self.log_callback_func = log_callback_func
         self.stop_event = stop_event # Salva riferimento a stop_event

    def on_epoch_end(self, epoch, logs=None):
        """Chiamato alla fine di ogni epoca."""
        # Controlla stop event
        if self.stop_event and self.stop_event.is_set():
            self.model.stop_training = True # Segnala a Keras di fermarsi
            if self.log_callback_func: self.log_callback_func(f"Epoca {epoch+1}: Richiesta di stop ricevuta, arresto training...")
            return

        if not self.log_callback_func: return
        logs = logs or {}; msg = f"Epoca {epoch+1:03d} - "
        log_items = [f"{k.replace('_',' ').replace('val ','V_')}: {v:.5f}" for k, v in logs.items()]
        msg += ", ".join(log_items)
        self.log_callback_func(msg) # Invia il messaggio


# --- Generazione Previsione (genera_previsione_superenalotto INVARIATA NELLA LOGICA) ---
# Riceverà X_input con più features (già scalato), ma la logica resta uguale.
def genera_previsione_superenalotto(model, X_input_scaled, num_predictions=6, log_callback=None):
    """
    Genera la previsione dei numeri del SuperEnalotto usando il modello addestrato.
    Input: X_input_scaled (1, sequence_length * N_features) già scalato.
    Output: Lista di dizionari [{'number': n, 'probability': p}, ...]
    (Logica interna invariata)
    """
    if log_callback: log_callback(f"Avvio generazione previsione SuperEnalotto per {num_predictions} numeri...")
    if model is None:
        if log_callback: log_callback("ERRORE (genera_prev_se): Modello non fornito."); return None
    if X_input_scaled is None or X_input_scaled.size == 0:
        if log_callback: log_callback("ERRORE (genera_prev_se): Input scalato vuoto."); return None

    # Assicura input 2D (1, N_features_flat)
    if X_input_scaled.ndim == 1: X_input_reshaped = X_input_scaled.reshape(1, -1)
    elif X_input_scaled.ndim == 2 and X_input_scaled.shape[0] == 1: X_input_reshaped = X_input_scaled
    else:
        if log_callback: log_callback(f"ERRORE (genera_prev_se): Shape input scalato non gestita: {X_input_scaled.shape}. Atteso 1D o (1, features).")
        return None
    if log_callback: log_callback(f"Input per predict preparato con shape: {X_input_reshaped.shape}")

    # Validazione shape vs modello (opzionale)
    try:
        expected_features = model.input_shape[-1]
        if expected_features is not None and X_input_reshaped.shape[1] != expected_features:
            msg = f"ERRORE Shape Input: Input ({X_input_reshaped.shape[1]}) != Modello ({expected_features})."
            if log_callback: log_callback(msg); return None
    except Exception as e_shape_check:
         if log_callback: log_callback(f"ATT (genera_prev_se): Impossibile verificare shape input modello ({e_shape_check}).")

    if not isinstance(num_predictions, int) or not (1 <= num_predictions <= 90):
        msg = f"ERRORE (genera_prev_se): num_predictions={num_predictions} non valido (1-90)."
        if log_callback: log_callback(msg); return None

    try:
        pred_probabilities = model.predict(X_input_reshaped, verbose=0)
        if pred_probabilities is None or pred_probabilities.size == 0:
            if log_callback: log_callback("ERRORE (genera_prev_se): model.predict() ha restituito vuoto."); return None
        if pred_probabilities.ndim != 2 or pred_probabilities.shape[0] != 1 or pred_probabilities.shape[1] != 90:
             msg = f"ERRORE (genera_prev_se): Output shape da predict inatteso: {pred_probabilities.shape}. Atteso (1, 90)."
             if log_callback: log_callback(msg); return None

        probs_vector = pred_probabilities[0] # Shape (90,)
        sorted_indices = np.argsort(probs_vector)
        top_n_indices = sorted_indices[-num_predictions:]

        predicted_results = [{"number": int(index + 1), "probability": float(probs_vector[index])}
                             for index in top_n_indices]

        if log_callback:
            results_sorted_by_prob_desc = sorted(predicted_results, key=lambda x: x['probability'], reverse=True)
            log_probs = [f"{res['number']:02d} (p={res['probability']:.5f})" for res in results_sorted_by_prob_desc]
            log_callback(f"Top {num_predictions} numeri predetti (ord. prob. decr.):\n  " + "\n  ".join(log_probs))

        return predicted_results # Lista di dizionari

    except Exception as e_predict:
        if log_callback: log_callback(f"ERRORE CRITICO durante generazione previsione SuperEnalotto: {e_predict}\n{traceback.format_exc()}")
        return None


# --- Funzione Principale di Analisi (MODIFICATA per FE & CV) ---
def analisi_superenalotto(file_path, start_date, end_date, sequence_length=5,
                          loss_function='binary_crossentropy', optimizer='adam',
                          dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                          hidden_layers_config=[512, 256, 128],
                          max_epochs=100, batch_size=32, patience=15, min_delta=0.0001,
                          num_predictions=6,
                          n_cv_splits=DEFAULT_SUPERENALOTTO_CV_SPLITS, # NUOVO: Parametro CV
                          log_callback=None,
                          stop_event=None): # NUOVO: Parametro Stop Event
    """
    Analizza i dati del SuperEnalotto con Feature Engineering e Cross-Validation.
    Restituisce: previsione (lista dict), messaggio attendibilità, data ultimo aggiornamento.
    """
    # --- Logging Iniziale ---
    if log_callback:
        source_type = "URL" if file_path.startswith("http") else "File"
        source_name = os.path.basename(file_path) if source_type == "File" else file_path
        log_callback(f"\n=== Avvio Analisi SuperEnalotto (FE & CV) ===")
        log_callback(f"Sorgente: {source_type} ({source_name}), Periodo: {start_date} -> {end_date}")
        log_callback(f"Seq/Pred: SeqLen={sequence_length}, NumPred={num_predictions}, CV Splits={n_cv_splits}")
        log_callback(f"Modello: HL={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")
        log_callback(f"Training: Epochs={max_epochs}, Batch={batch_size}, Pat={patience}, MinDelta={min_delta}")
        log_callback("-" * 40)

    # Controllo stop iniziale
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima dell'inizio."); return None, "Analisi annullata", None

    # 1. Carica e Preprocessa i Dati
    df, numeri_principali_array, _, _, last_update_date = carica_dati_superenalotto(
        file_path, start_date, end_date, log_callback=log_callback
    )
    if df is None: return None, "Errore critico caricamento dati.", None
    if df.empty or numeri_principali_array is None or len(numeri_principali_array) == 0:
        msg = "Nessun dato valido trovato per analisi."
        if log_callback: log_callback(f"INFO: {msg}"); return None, msg, last_update_date
    if len(numeri_principali_array) < sequence_length + 1:
        msg = f"ERRORE: Dati insuff. ({len(numeri_principali_array)}) per seq_len ({sequence_length}). Servono {sequence_length + 1}."
        if log_callback: log_callback(msg); return None, msg, last_update_date

    last_update_str = last_update_date.strftime('%Y-%m-%d') if last_update_date else 'N/D'
    if log_callback: log_callback(f"Dati caricati. Estrazioni: {len(numeri_principali_array)}. Ultimo agg: {last_update_str}")

    # Controllo stop dopo caricamento
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo caricamento dati."); return None, "Analisi annullata", last_update_date

    # 2. Feature Engineering
    combined_features = engineer_features_superenalotto(numeri_principali_array, log_callback=log_callback)
    if combined_features is None:
        return None, "Feature Engineering fallito.", last_update_date

    # Controllo stop dopo FE
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo Feature Engineering."); return None, "Analisi annullata", last_update_date

    # 3. Prepara Sequenze (Input dalle Combined Features, Target dai Numeri Originali)
    # Nota: combined_features non è ancora scalato qui. Lo scalerà la CV/train finale.
    X, y = None, None
    try:
        # Passiamo combined_features (non scalato) e numeri_principali_array (per target)
        # La funzione prepara_sequenze ora assume che X sarà scalato *dopo*
        X, y = prepara_sequenze_per_modello_superenalotto(combined_features, numeri_principali_array, sequence_length, log_callback=log_callback)
        if X is None or y is None or len(X) == 0:
            return None, "Creazione sequenze Input/Target fallita.", last_update_date

        # Verifica minima per CV
        min_samples_for_cv = n_cv_splits + 1
        if len(X) < min_samples_for_cv:
            msg = f"ERRORE: Campioni insuff. ({len(X)}) per {n_cv_splits}-Fold CV (min {min_samples_for_cv}). Riduci n_splits o aumenta dati."; log_callback(msg); return None, msg, last_update_date

    except Exception as e_prep:
         if log_callback: log_callback(f"ERRORE CRITICO preparazione sequenze: {e_prep}\n{traceback.format_exc()}")
         return None, f"Errore preparazione sequenze: {e_prep}", last_update_date

    # Controllo stop dopo prep sequenze
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata dopo preparazione sequenze."); return None, "Analisi annullata", last_update_date

    # 4. Cross-Validation con TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    fold_val_losses = []
    fold_val_accuracies = []
    fold_best_epochs = []
    log_callback(f"\n--- Inizio {n_cv_splits}-Fold TimeSeries Cross-Validation ---")

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        fold_num = fold + 1
        if stop_event and stop_event.is_set(): log_callback(f"CV Interrotta prima del fold {fold_num}."); break

        log_callback(f"\n--- Fold {fold_num}/{n_cv_splits} ---")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        log_callback(f"Train: {len(X_train_fold)} campioni (idx {train_index[0]}-{train_index[-1]}), Validation: {len(X_val_fold)} campioni (idx {val_index[0]}-{val_index[-1]})")

        if len(X_train_fold) == 0 or len(X_val_fold) == 0:
             log_callback(f"ATT: Fold {fold_num} saltato (set vuoto)."); continue

        # -> 4a. Scale Fold Data (Fit on Train, Transform Train & Val)
        scaler_fold = StandardScaler()
        try:
            X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
            X_val_fold_scaled = scaler_fold.transform(X_val_fold)
            log_callback(f"Dati Fold {fold_num} scalati (StandardScaler fit su train).")
        except Exception as e_scale_fold:
            log_callback(f"ERRORE scaling fold {fold_num}: {e_scale_fold}. Salto fold."); continue

        # -> 4b. Build & Train Fold Model
        model_fold, history_fold = None, None
        gui_log_callback_fold = LogCallback(log_callback, stop_event) # Passa stop_event
        try:
            tf.keras.backend.clear_session(); set_seed()
            input_shape_fold = (X_train_fold_scaled.shape[1],)
            model_fold = build_model_superenalotto(input_shape_fold, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
            if model_fold is None: log_callback(f"ERRORE: Costruzione modello fold {fold_num} fallita."); continue

            monitor = 'val_loss'; patience_fold = max(5, patience // 2) # Patience ridotta per CV
            early_stopping_fold = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience_fold, min_delta=min_delta, restore_best_weights=True, verbose=0)
            log_callback(f"Inizio addestramento fold {fold_num} (Patience={patience_fold})...")
            history_fold = model_fold.fit(X_train_fold_scaled, y_train_fold, validation_data=(X_val_fold_scaled, y_val_fold),
                                          epochs=max_epochs, batch_size=batch_size,
                                          callbacks=[early_stopping_fold, gui_log_callback_fold], verbose=0)

            # Controllo stop dopo fit fold
            if stop_event and stop_event.is_set(): log_callback(f"Training fold {fold_num} interrotto."); break

            if history_fold and history_fold.history and 'val_loss' in history_fold.history:
                best_epoch_idx = np.argmin(history_fold.history['val_loss'])
                best_val_loss = history_fold.history['val_loss'][best_epoch_idx]
                best_val_acc = history_fold.history['val_accuracy'][best_epoch_idx] if 'val_accuracy' in history_fold.history else -1.0
                fold_val_losses.append(best_val_loss)
                fold_val_accuracies.append(best_val_acc)
                fold_best_epochs.append(best_epoch_idx + 1)
                log_callback(f"Fold {fold_num} OK. Miglior epoca: {best_epoch_idx+1}, Val Loss: {best_val_loss:.5f}, Val Acc: {best_val_acc:.5f}")
            else: log_callback(f"ATT: History fold {fold_num} invalida o 'val_loss' mancante.")

        except tf.errors.ResourceExhaustedError as e_mem:
             msg = f"ERRORE Memoria fold {fold_num}: {e_mem}."; log_callback(msg); log_callback(traceback.format_exc()); break # Stop CV
        except Exception as e_fold:
            if stop_event and stop_event.is_set(): log_callback(f"Eccezione fold {fold_num} (prob. da stop): {e_fold}"); break
            msg = f"ERRORE CRITICO addestramento fold {fold_num}: {e_fold}"; log_callback(msg); log_callback(traceback.format_exc());
            # break # Considera interrompere CV
            log_callback("Continuo con il prossimo fold...")
            continue
        finally:
             pass # gui_log_callback_fold non ha bisogno di stop manuale se stop_event è usato

    # --- Fine Loop CV ---
    if stop_event and stop_event.is_set(): log_callback("Cross-Validation interrotta."); return None, "Analisi Interrotta (CV)", last_update_date

    avg_val_loss, avg_val_acc, avg_epochs = -1.0, -1.0, -1.0
    attendibilita_cv_msg = "Attendibilità CV: Non Determinata (CV incompleta o fallita)"
    if fold_val_losses:
        avg_val_loss = np.mean(fold_val_losses)
        avg_val_acc = np.mean(fold_val_accuracies) if fold_val_accuracies and all(v >= 0 for v in fold_val_accuracies) else -1.0
        avg_epochs = np.mean(fold_best_epochs) if fold_best_epochs else -1.0
        log_callback("\n--- Risultati Cross-Validation ---")
        log_callback(f"Loss media (val): {avg_val_loss:.5f}")
        if avg_val_acc >= 0: log_callback(f"Accuracy media (val): {avg_val_acc:.5f}")
        if avg_epochs >= 0: log_callback(f"Epoca media ottimale: {avg_epochs:.1f}")
        attendibilita_cv_msg = f"Attendibilità CV: Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.4f} ({len(fold_val_losses)} folds)"
        log_callback("---------------------------------")
    else:
        log_callback("\nATTENZIONE: Nessun fold CV completato con successo.")
        return None, "Cross-Validation fallita (nessun fold valido)", last_update_date

    # Controllo stop prima del training finale
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima del training finale."); return None, "Analisi Interrotta (Pre-Final)", last_update_date

    # 5. Addestra Modello Finale su TUTTI i dati (X, y)
    final_model, history_final = None, None
    final_loss = float('inf'); final_acc = -1.0; final_epochs_run = 0
    final_scaler = StandardScaler() # Scaler per i dati finali
    log_callback("\n--- Addestramento Modello Finale su tutti i dati ---")
    gui_log_callback_final = LogCallback(log_callback, stop_event) # Passa stop_event
    try:
        # -> 5a. Scale ALL Data using final_scaler
        X_scaled_final = final_scaler.fit_transform(X)
        log_callback(f"Dati finali (X) scalati per training. Shape: {X_scaled_final.shape}")

        # -> 5b. Build & Train Final Model
        tf.keras.backend.clear_session(); set_seed()
        input_shape_final = (X_scaled_final.shape[1],)
        final_model = build_model_superenalotto(input_shape_final, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if final_model is None: return None, "Costruzione modello finale fallita.", last_update_date

        # Usiamo EarlyStopping sulla loss di training
        final_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, min_delta=min_delta, restore_best_weights=True, verbose=1)
        log_callback(f"Inizio addestramento finale (max {max_epochs} epoche, ES su 'loss', Patience={patience})...")
        history_final = final_model.fit(X_scaled_final, y, epochs=max_epochs, batch_size=batch_size,
                                        callbacks=[final_early_stopping, gui_log_callback_final], verbose=0)

        # Controllo stop dopo fit finale
        if stop_event and stop_event.is_set(): log_callback("Addestramento finale interrotto."); return None, "Analisi Interrotta (Final Train)", last_update_date

        if history_final and history_final.history and 'loss' in history_final.history:
             final_epochs_run = len(history_final.history['loss'])
             best_final_epoch_idx = np.argmin(history_final.history['loss']) # Indice della loss minima
             final_loss = history_final.history['loss'][best_final_epoch_idx]
             final_acc = history_final.history['accuracy'][best_final_epoch_idx] if 'accuracy' in history_final.history else -1.0
             log_callback(f"Addestramento finale OK: {final_epochs_run} epoche (migliore @ {best_final_epoch_idx+1}). Loss: {final_loss:.5f}, Acc: {final_acc:.5f}")
        else:
             log_callback("ATT: History addestramento finale non valida o 'loss' mancante.")

    except tf.errors.ResourceExhaustedError as e_mem:
         msg = f"ERRORE Memoria Training Finale: {e_mem}."; log_callback(msg); log_callback(traceback.format_exc()); return None, msg, last_update_date
    except Exception as e_final:
        if stop_event and stop_event.is_set(): log_callback(f"Eccezione training finale (prob. da stop): {e_final}"); return None, "Analisi Interrotta (Final Train Err)", last_update_date
        msg = f"ERRORE CRITICO addestramento finale: {e_final}"; log_callback(msg); log_callback(traceback.format_exc()); return None, msg, last_update_date
    finally:
         pass # gui_log_callback_final non necessita stop manuale

    # Controllo stop prima della previsione
    if stop_event and stop_event.is_set(): log_callback("Analisi annullata prima della previsione finale."); return None, "Analisi Interrotta (Pre-Predict)", last_update_date

    # 6. Prepara Input per Previsione Finale e Genera
    previsione_completa = None
    attendibilita_msg = "Attendibilità Non Determinata"
    try:
        log_callback("\n--- Preparazione Input per Previsione Finale ---")
        # Prendi le ultime 'sequence_length' righe dall'array originale
        if len(numeri_principali_array) < sequence_length:
            msg = f"ERRORE: Dati originali insuff. ({len(numeri_principali_array)}) per input previsione (richiesti {sequence_length})."
            log_callback(msg); return None, msg, last_update_date

        last_sequence_raw = numeri_principali_array[-sequence_length:]

        # Applica Feature Engineering alla sequenza raw
        last_sequence_engineered = engineer_features_superenalotto(last_sequence_raw, log_callback=None) # Log non necessario qui
        if last_sequence_engineered is None:
             return None, "Errore Feature Engineering per input previsione.", last_update_date

        # Appiattisci la sequenza con features
        last_sequence_flat = last_sequence_engineered.flatten()

        # Scala usando lo scaler adattato sui dati di training finali (final_scaler)
        input_pred_scaled = final_scaler.transform(last_sequence_flat.reshape(1, -1))
        if log_callback: log_callback(f"Input finale per previsione scalato (shape {input_pred_scaled.shape}). Generazione...")

        # Genera previsione con il modello finale
        previsione_completa = genera_previsione_superenalotto(final_model, input_pred_scaled, num_predictions, log_callback=log_callback)

        if previsione_completa is None:
            return None, "Generazione previsione finale fallita.", last_update_date

        # Costruisci messaggio attendibilità combinato
        attendibilita_msg_final_train = f"Training finale ({final_epochs_run} ep): Loss={final_loss:.5f}, Acc={final_acc:.5f}"
        attendibilita_msg = f"{attendibilita_cv_msg}. {attendibilita_msg_final_train}"
        log_callback(f"\nAttendibilità Combinata: {attendibilita_msg}")

        return previsione_completa, attendibilita_msg, last_update_date

    except Exception as e_final_pred:
         if log_callback: log_callback(f"Errore CRITICO fase previsione finale: {e_final_pred}\n{traceback.format_exc()}")
         return None, f"Errore critico previsione finale: {e_final_pred}", last_update_date
# --- Fine Funzione Analisi ---


# --- Definizione Classe GUI AppSuperEnalotto (MODIFICATA per CV) ---
class AppSuperEnalotto:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi e Previsione SuperEnalotto (v1.5 - Spia Integrata)") # Aggiorna versione se vuoi
        self.root.geometry("850x1180") # Altezza leggermente aumentata per la sezione spia

        self.style = ttk.Style()
        try: 
            if sys.platform == "win32": self.style.theme_use('vista')
            elif sys.platform == "darwin": self.style.theme_use('aqua')
            else: self.style.theme_use('clam') 
        except tk.TclError: self.style.theme_use('default')

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Variabili di stato per Previsione ML
        self.last_prediction_numbers = None
        self.last_prediction_full = None # Per conservare anche le probabilità
        self.last_prediction_end_date = None
        self.last_prediction_date_str = None

        # Variabili di Stato per Spia SuperEnalotto
        self.last_spia_se_singoli = None
        self.last_spia_se_ambi = None
        self.last_spia_se_terni = None
        self.last_spia_se_data_fine_analisi = None
        self.last_spia_se_numeri_input = None

        # --- Input File/URL ---
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Origine Dati Estrazioni (URL Raw GitHub o File Locale .txt)", padding="10")
        self.file_frame.pack(fill=tk.X, pady=5)
        self.file_path_var = tk.StringVar(value=DEFAULT_SUPERENALOTTO_DATA_URL)
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, width=65)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_button = ttk.Button(self.file_frame, text="Sfoglia Locale...", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT)

        # --- Contenitore Parametri ---
        self.params_container = ttk.Frame(self.main_frame)
        self.params_container.pack(fill=tk.X, pady=5)
        self.params_container.columnconfigure(0, weight=1) # Colonna sinistra
        self.params_container.columnconfigure(1, weight=1) # Colonna destra

        # --- Colonna Sinistra: Parametri Dati e Previsione ML ---
        self.data_params_frame = ttk.LabelFrame(self.params_container, text="Parametri Dati e Previsione ML", padding="10")
        self.data_params_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        
        _cur_row_data_ml = 0 # Contatore per le righe in questo frame
        
        ttk.Label(self.data_params_frame, text="Data Inizio:").grid(row=_cur_row_data_ml, column=0, padx=5, pady=2, sticky=tk.W)
        # MODIFICA QUI LA DATA DI DEFAULT PER L'INIZIO
        default_start_ml = datetime(2025, 1, 1) # Impostato al 1 Gennaio 2025
        
        if HAS_TKCALENDAR:
            self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd', show_weeknumbers=False, locale='it_IT',
                                              year=default_start_ml.year, month=default_start_ml.month, day=default_start_ml.day) # Imposta direttamente nel costruttore
            # try: 
            #     self.start_date_entry.set_date(default_start_ml) # Alternativa se il costruttore non funziona come previsto
            # except ValueError: 
            #     self.start_date_entry.set_date(datetime.now()) # Fallback generico
        else:
            self.start_date_entry_var = tk.StringVar(value=default_start_ml.strftime('%Y-%m-%d')) 
            self.start_date_entry = ttk.Entry(self.data_params_frame, textvariable=self.start_date_entry_var, width=12)
        self.start_date_entry.grid(row=_cur_row_data_ml, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_data_ml += 1
        
        ttk.Label(self.data_params_frame, text="Data Fine:").grid(row=_cur_row_data_ml, column=0, padx=5, pady=2, sticky=tk.W)
        default_end_ml = datetime.now() # Lasciamo la data di fine a "oggi" per default
        if HAS_TKCALENDAR:
            self.end_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd', show_weeknumbers=False, locale='it_IT',
                                            year=default_end_ml.year, month=default_end_ml.month, day=default_end_ml.day)
            # self.end_date_entry.set_date(default_end_ml) # Alternativa
        else:
            self.end_date_entry_var = tk.StringVar(value=default_end_ml.strftime('%Y-%m-%d')) 
            self.end_date_entry = ttk.Entry(self.data_params_frame, textvariable=self.end_date_entry_var, width=12)
        self.end_date_entry.grid(row=_cur_row_data_ml, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_data_ml += 1

        ttk.Label(self.data_params_frame, text="Seq. Input (Storia):").grid(row=_cur_row_data_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.seq_len_var = tk.StringVar(value="12") 
        self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=3, to=50, increment=1, textvariable=self.seq_len_var, width=5, wrap=True, state='readonly')
        self.seq_len_entry.grid(row=_cur_row_data_ml, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_data_ml += 1

        ttk.Label(self.data_params_frame, text="Numeri da Prevedere:").grid(row=_cur_row_data_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="6")
        self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=6, to=15, increment=1, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly')
        self.num_predict_spinbox.grid(row=_cur_row_data_ml, column=1, padx=5, pady=2, sticky=tk.W)
        _cur_row_data_ml += 1

        # --- Colonna Destra: Parametri Modello ML e Training ---
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello ML e Training", padding="10")
        self.model_params_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        self.model_params_frame.columnconfigure(1, weight=1) 
        
        _cur_row_model_ml = 0 

        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.hidden_layers_var = tk.StringVar(value="128, 64") 
        self.hidden_layers_entry = ttk.Entry(self.model_params_frame, textvariable=self.hidden_layers_var, width=25)
        self.hidden_layers_entry.grid(row=_cur_row_model_ml, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="Loss Function:").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.loss_var = tk.StringVar(value='binary_crossentropy')
        self.loss_combo = ttk.Combobox(self.model_params_frame, textvariable=self.loss_var, width=23, state='readonly', values=['binary_crossentropy', 'mse', 'mae', 'huber_loss'])
        self.loss_combo.grid(row=_cur_row_model_ml, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="Optimizer:").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value='adam')
        self.optimizer_combo = ttk.Combobox(self.model_params_frame, textvariable=self.optimizer_var, width=23, state='readonly', values=['adam', 'rmsprop', 'sgd', 'adagrad', 'adamw'])
        self.optimizer_combo.grid(row=_cur_row_model_ml, column=1, columnspan=2, padx=5, pady=2, sticky=tk.EW); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="Dropout Rate (0-1):").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.dropout_var = tk.StringVar(value="0.25") 
        self.dropout_spinbox = ttk.Spinbox(self.model_params_frame, from_=0.0, to=0.8, increment=0.05, format="%.2f", textvariable=self.dropout_var, width=7, wrap=True, state='readonly')
        self.dropout_spinbox.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="L1 Strength (>=0):").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.l1_var = tk.StringVar(value="0.00")
        self.l1_entry = ttk.Entry(self.model_params_frame, textvariable=self.l1_var, width=7)
        self.l1_entry.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="L2 Strength (>=0):").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.l2_var = tk.StringVar(value="0.00")
        self.l2_entry = ttk.Entry(self.model_params_frame, textvariable=self.l2_var, width=7)
        self.l2_entry.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="Max Epoche:").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="100") 
        self.epochs_spinbox = ttk.Spinbox(self.model_params_frame, from_=20, to=500, increment=10, textvariable=self.epochs_var, width=7, wrap=True, state='readonly')
        self.epochs_spinbox.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="Batch Size:").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="128") 
        self.batch_size_combo = ttk.Combobox(self.model_params_frame, textvariable=self.batch_size_var, values=[str(2**i) for i in range(4, 10)], width=5, state='readonly')
        self.batch_size_combo.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="ES Patience:").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.patience_var = tk.StringVar(value="15") 
        self.patience_spinbox = ttk.Spinbox(self.model_params_frame, from_=5, to=50, increment=1, textvariable=self.patience_var, width=7, wrap=True, state='readonly')
        self.patience_spinbox.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="ES Min Delta:").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.min_delta_var = tk.StringVar(value="0.0001")
        self.min_delta_entry = ttk.Entry(self.model_params_frame, textvariable=self.min_delta_var, width=10)
        self.min_delta_entry.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        ttk.Label(self.model_params_frame, text="CV Splits (>=2):").grid(row=_cur_row_model_ml, column=0, padx=5, pady=2, sticky=tk.W)
        self.cv_splits_var = tk.StringVar(value=str(DEFAULT_SUPERENALOTTO_CV_SPLITS))
        self.cv_splits_spinbox = ttk.Spinbox(self.model_params_frame, from_=2, to=20, increment=1, textvariable=self.cv_splits_var, width=5, wrap=True, state='readonly')
        self.cv_splits_spinbox.grid(row=_cur_row_model_ml, column=1, padx=5, pady=2, sticky=tk.W); _cur_row_model_ml += 1

        # --- Pulsanti Azione ML ---
        # Se action_frame è già stato definito prima, non ridefinirlo. 
        # Altrimenti, questo è il posto corretto.
        self.action_frame = ttk.Frame(self.main_frame) 
        self.action_frame.pack(pady=5, fill=tk.X)
        self.run_button = ttk.Button(self.action_frame, text="Avvia Analisi ML", command=self.start_analysis_thread)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(self.action_frame, text="Ferma Analisi ML", command=self.stop_analysis_thread, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.check_button = ttk.Button(self.action_frame, text="Verifica Prev. ML", command=self.start_check_thread, state=tk.DISABLED)
        self.check_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.action_frame, text="Colpi Verifica ML:").pack(side=tk.LEFT, padx=(10, 2))
        self.check_colpi_var = tk.StringVar(value=str(DEFAULT_SUPERENALOTTO_CHECK_COLPI))
        self.check_colpi_spinbox = ttk.Spinbox(self.action_frame, from_=1, to=100, increment=1, textvariable=self.check_colpi_var, width=4, wrap=True, state='readonly')
        self.check_colpi_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # --- Risultati Previsione ML ---
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Risultato Previsione SuperEnalotto (ML)", padding="10")
        self.results_frame.pack(fill=tk.X, pady=5)
        self.result_label_var = tk.StringVar(value="I numeri ML previsti appariranno qui...")
        self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier New', 16, 'bold'), foreground='#000080')
        self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value="")
        self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 9, 'italic'))
        self.attendibilita_label.pack(pady=2, fill=tk.X)

        # --- Data Ultimo Aggiornamento Dati Usati (per ML) ---
        self.last_update_frame = ttk.LabelFrame(self.main_frame, text="Dati Utilizzati nell'Ultima Analisi ML", padding="5")
        self.last_update_frame.pack(fill=tk.X, pady=5)
        self.last_update_label_var = tk.StringVar(value="Data ultimo aggiornamento usato (ML) apparirà qui...")
        self.last_update_label = ttk.Label(self.last_update_frame, textvariable=self.last_update_label_var, font=('Helvetica', 9))
        self.last_update_label.pack(pady=3, anchor='w')

        # --- Frame e Controlli per Analisi Numeri Spia SuperEnalotto ---
        self.spia_se_frame = ttk.LabelFrame(self.main_frame, text="Analisi Numeri Spia SuperEnalotto (su periodo date sopra)", padding="10")
        self.spia_se_frame.pack(fill=tk.X, pady=(10,5), padx=0)

        spia_se_params_r1 = ttk.Frame(self.spia_se_frame) 
        spia_se_params_r1.pack(fill=tk.X, pady=2)
        ttk.Label(spia_se_params_r1, text="Numeri Spia (es: 7,23):").pack(side=tk.LEFT, padx=(0,5))
        self.numeri_spia_se_var = tk.StringVar(value="7,23") 
        self.numeri_spia_se_entry = ttk.Entry(spia_se_params_r1, textvariable=self.numeri_spia_se_var, width=15)
        self.numeri_spia_se_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_se_params_r1, text="Colpi An. Post-Spia:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_se_colpi_var = tk.StringVar(value="3") 
        self.spia_se_colpi_spinbox = ttk.Spinbox(spia_se_params_r1, from_=1, to=10, textvariable=self.spia_se_colpi_var, width=4, state='readonly', wrap=True)
        self.spia_se_colpi_spinbox.pack(side=tk.LEFT, padx=(0,10))

        spia_se_params_r2 = ttk.Frame(self.spia_se_frame) 
        spia_se_params_r2.pack(fill=tk.X, pady=2)
        ttk.Label(spia_se_params_r2, text="Top N Singoli:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_se_top_n_singoli_var = tk.StringVar(value="6")
        self.spia_se_top_n_singoli_spinbox = ttk.Spinbox(spia_se_params_r2, from_=1, to=15, textvariable=self.spia_se_top_n_singoli_var, width=4, state='readonly', wrap=True)
        self.spia_se_top_n_singoli_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_se_params_r2, text="Top N Ambi:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_se_top_n_ambi_var = tk.StringVar(value="5")
        self.spia_se_top_n_ambi_spinbox = ttk.Spinbox(spia_se_params_r2, from_=1, to=10, textvariable=self.spia_se_top_n_ambi_var, width=4, state='readonly', wrap=True)
        self.spia_se_top_n_ambi_spinbox.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_se_params_r2, text="Top N Terni:").pack(side=tk.LEFT, padx=(0,5))
        self.spia_se_top_n_terni_var = tk.StringVar(value="3")
        self.spia_se_top_n_terni_spinbox = ttk.Spinbox(spia_se_params_r2, from_=1, to=5, textvariable=self.spia_se_top_n_terni_var, width=4, state='readonly', wrap=True)
        self.spia_se_top_n_terni_spinbox.pack(side=tk.LEFT, padx=(0,10))

        spia_se_actions = ttk.Frame(self.spia_se_frame) 
        spia_se_actions.pack(fill=tk.X, pady=(5,2))
        self.run_spia_se_button = ttk.Button(spia_se_actions, text="Avvia Analisi Spie SE", command=self.start_analisi_spia_superenalotto_thread)
        self.run_spia_se_button.pack(side=tk.LEFT, padx=(0,10))
        self.check_spia_se_button = ttk.Button(spia_se_actions, text="Verifica Ris. Spia SE", command=self.start_verifica_spia_superenalotto_thread, state=tk.DISABLED)
        self.check_spia_se_button.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(spia_se_actions, text="Colpi Verifica Spia SE:").pack(side=tk.LEFT, padx=(10,2))
        self.check_spia_se_colpi_var = tk.StringVar(value=str(DEFAULT_SUPERENALOTTO_CHECK_COLPI))
        self.check_spia_se_colpi_spinbox = ttk.Spinbox(spia_se_actions, from_=1, to=20, textvariable=self.check_spia_se_colpi_var, width=4, state='readonly', wrap=True)
        self.check_spia_se_colpi_spinbox.pack(side=tk.LEFT)
        
        # --- Log Area ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monaco", 9)
        try:
            self.log_text = scrolledtext.ScrolledText(self.log_frame, height=12, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='#f5f5f5', foreground='black')
        except tk.TclError: 
             log_font = ("Courier New", 9) 
             self.log_text = scrolledtext.ScrolledText(self.log_frame, height=12, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='#f5f5f5', foreground='black')
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Threading Safety ---
        self.analysis_thread = None; self.check_thread = None
        self.spia_se_thread = None; self.verifica_spia_se_thread = None 
        
        self._stop_event_analysis = threading.Event()
        self._stop_event_check = threading.Event()
        self._stop_event_spia_se = threading.Event() 
        self._stop_event_verifica_spia_se = threading.Event() 

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def browse_file(self):
        """Apre una finestra di dialogo per selezionare un file LOCALE."""
        filepath = filedialog.askopenfilename(
            title="Seleziona file estrazioni SuperEnalotto Locale (.txt)",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            parent=self.root
        )
        if filepath: self.file_path_var.set(filepath); self.log_message_gui(f"Selezionato file locale: {filepath}")

    def log_message_gui(self, message):
        """Wrapper per inviare messaggi al widget di log dalla GUI."""
        log_message(message, self.log_text, self.root)

    def set_result(self, prediction_data, attendibilita):
         """Aggiorna i label dei risultati nella GUI."""
         self.root.after(0, self._update_result_labels, prediction_data, attendibilita)

    def _update_result_labels(self, prediction_data, attendibilita):
        """Funzione helper eseguita nel thread GUI per aggiornare le etichette."""
        try:
            if not self.root.winfo_exists() or not self.result_label.winfo_exists(): return

            if (isinstance(prediction_data, list) and prediction_data and
                    all(isinstance(item, dict) and 'number' in item and 'probability' in item for item in prediction_data)):
                sorted_by_number = sorted(prediction_data, key=lambda x: x['number'])
                result_str = "  ".join([f"{item['number']:02d}" for item in sorted_by_number])
                self.result_label_var.set(result_str)

                log_nums_only = sorted([item['number'] for item in prediction_data])
                self.log_message_gui("\n" + "="*35 + "\nPREVISIONE SUPERENALOTTO GENERATA (ML)\n" + "="*35)
                self.log_message_gui("Numeri e probabilità stimate (ord. prob. decr.):")
                sorted_by_prob_desc = sorted(prediction_data, key=lambda x: x['probability'], reverse=True)
                for item in sorted_by_prob_desc:
                     self.log_message_gui(f"  - N: {item['number']:02d} (P: {item['probability']:.6f})")
                self.log_message_gui(f"Numeri finali selezionati (ordinati): {log_nums_only}")
                self.log_message_gui("="*35)
            else:
                self.result_label_var.set("Previsione fallita o non valida.")
                log_err = True
                if isinstance(attendibilita, str) and any(kw in attendibilita.lower() for kw in ["errore", "fallit", "insuff", "invalid", "nessun"]): log_err = False
                if log_err: self.log_message_gui("\nERRORE: Previsione non ha restituito dati validi.")

            self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "Attendibilità non disponibile.")
        except tk.TclError as e: print(f"TclError in _update_result_labels (shutdown?): {e}")
        except Exception as e: print(f"Error in _update_result_labels: {e}")

    # --- Metodo set_controls_state (MODIFICATO per aggiungere cv_splits e stop_button) ---
    def set_controls_state(self, state):
        """Abilita o disabilita i controlli della GUI."""
        self.root.after(10, lambda: self._set_controls_state_tk(state))

# 2. Modifica _set_controls_state_tk
    def _set_controls_state_tk(self, state):
        try:
            if not self.root.winfo_exists(): return

            is_analysis_running = self.analysis_thread and self.analysis_thread.is_alive()
            is_check_running = self.check_thread and self.check_thread.is_alive()
            # NUOVO: controlli per thread spia SE
            is_spia_se_running = self.spia_se_thread and self.spia_se_thread.is_alive()
            is_verifica_spia_se_running = self.verifica_spia_se_thread and self.verifica_spia_se_thread.is_alive()
            
            is_any_ml_action_running = is_analysis_running or is_check_running
            is_any_spia_se_action_running = is_spia_se_running or is_verifica_spia_se_running
            is_any_thread_running = is_any_ml_action_running or is_any_spia_se_action_running

            target_state_general = tk.DISABLED if is_any_thread_running else state

            # Aggiungi i nuovi widget spia SE alla lista
            widgets_to_toggle = [
                self.browse_button, self.file_entry,
                self.seq_len_entry, self.num_predict_spinbox,
                self.hidden_layers_entry, self.loss_combo, self.optimizer_combo,
                self.dropout_spinbox, self.l1_entry, self.l2_entry,
                self.epochs_spinbox, self.batch_size_combo,
                self.patience_spinbox, self.min_delta_entry,
                self.cv_splits_spinbox, 
                self.run_button, self.stop_button, # Assumendo che stop_button esista
                self.check_colpi_spinbox, self.check_button,
                # NUOVI WIDGET SPIA SE
                self.numeri_spia_se_entry, self.spia_se_colpi_spinbox,
                self.spia_se_top_n_singoli_spinbox, self.spia_se_top_n_ambi_spinbox, self.spia_se_top_n_terni_spinbox,
                self.run_spia_se_button, self.check_spia_se_button, self.check_spia_se_colpi_spinbox
            ]
            # ... (logica per DateEntry come prima) ...
            date_entries_to_handle = [] # Popolala come nella versione precedente
            if HAS_TKCALENDAR:
                if hasattr(self, 'start_date_entry'): date_entries_to_handle.append(self.start_date_entry)
                if hasattr(self, 'end_date_entry'): date_entries_to_handle.append(self.end_date_entry)
            else:
                if hasattr(self, 'start_date_entry'): widgets_to_toggle.append(self.start_date_entry) # Se ttk.Entry
                if hasattr(self, 'end_date_entry'): widgets_to_toggle.append(self.end_date_entry)   # Se ttk.Entry


            for widget in widgets_to_toggle:
                if widget is None or not hasattr(widget, 'winfo_exists') or not widget.winfo_exists(): continue
                
                current_widget_target_state = target_state_general

                # Disabilita pulsanti di un gruppo se l'altro è attivo
                if state == tk.NORMAL: # Solo se si sta cercando di abilitare
                    if widget in [self.run_button, self.check_button] and is_any_spia_se_action_running:
                        current_widget_target_state = tk.DISABLED
                    if widget in [self.run_spia_se_button, self.check_spia_se_button] and is_any_ml_action_running:
                        current_widget_target_state = tk.DISABLED
                
                # Logica specifica per pulsanti di verifica
                if widget == self.check_button and current_widget_target_state == tk.NORMAL and self.last_prediction_numbers is None:
                    current_widget_target_state = tk.DISABLED
                if widget == self.check_spia_se_button and current_widget_target_state == tk.NORMAL and not (self.last_spia_se_singoli or self.last_spia_se_ambi or self.last_spia_se_terni):
                    current_widget_target_state = tk.DISABLED
                
                # Stato bottone Stop ML
                if widget == self.stop_button: # Assumendo che self.stop_button esista
                    current_widget_target_state = tk.NORMAL if is_analysis_running else tk.DISABLED


                # Applica stato effettivo
                tk_effective_state = tk.DISABLED
                if current_widget_target_state == tk.NORMAL:
                    if isinstance(widget, (ttk.Combobox, ttk.Spinbox)): tk_effective_state = 'readonly'
                    elif isinstance(widget, (ttk.Entry, scrolledtext.ScrolledText)): tk_effective_state = tk.NORMAL
                    elif isinstance(widget, ttk.Button): tk_effective_state = tk.NORMAL
                
                try:
                    if str(widget.cget('state')).lower() != str(tk_effective_state).lower():
                        widget.config(state=tk_effective_state)
                except (tk.TclError, AttributeError): pass
            
            # Gestione DateEntry tkcalendar
            for date_widget in date_entries_to_handle:
                if date_widget and hasattr(date_widget, 'winfo_exists') and date_widget.winfo_exists():
                    target_date_state = tk.NORMAL if target_state_general == tk.NORMAL else tk.DISABLED
                    if str(date_widget.cget('state')).lower() != str(target_date_state).lower():
                        date_widget.configure(state=target_date_state)
        except Exception as e: print(f"Errore _set_controls_state_tk SE: {e}\n{traceback.format_exc()}")


    def _get_spia_se_dates(self): # Helper per le date, può essere lo stesso del 10eLotto se usi gli stessi widget
        # Riusa _get_spia_dates se i widget delle date sono condivisi, 
        # altrimenti crea una versione specifica se hai DateEntry separati per le spie SE.
        # Per ora assumo che riusi gli stessi widget di data dell'analisi ML.
        try:
            s_date = self.start_date_entry.get_date().strftime('%Y-%m-%d') if HAS_TKCALENDAR else self.start_date_entry_var.get() # Adatta se usi var separate per Entry
            e_date = self.end_date_entry.get_date().strftime('%Y-%m-%d') if HAS_TKCALENDAR else self.end_date_entry_var.get() # Adatta
            datetime.strptime(s_date, '%Y-%m-%d'); datetime.strptime(e_date, '%Y-%m-%d') 
            return s_date, e_date
        except Exception as e: messagebox.showerror("Errore Data (Spia SE)", f"Date non valide: {e}", parent=self.root); return None, None

    def _validate_spia_se_params(self, data_source, numeri_spia_str, start_date, end_date, colpi_an_str, top_s_str, top_a_str, top_t_str):
        # Questa funzione può essere quasi identica a _validate_spia_params del 10eLotto
        # Cambia solo i messaggi di errore se vuoi specificare "Spia SE"
        errors = {'messages': [], 'parsed': {}}
        if not data_source or (not data_source.startswith("http") and not os.path.exists(data_source)): errors['messages'].append("Sorgente dati SE non valida.")
        try:
            if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'): errors['messages'].append("Data Inizio Spia SE > Data Fine Spia SE.")
        except ValueError: errors['messages'].append("Formato date Spia SE non valido.")

        try: 
            parsed_nums = [int(n.strip()) for n in numeri_spia_str.split(',') if n.strip() and 1 <= int(n.strip()) <= 90]
            if not parsed_nums: raise ValueError()
            errors['parsed']['numeri_spia'] = parsed_nums
        except: errors['messages'].append("Numeri Spia SE non validi (1-90, separati da ',').")
        
        try: errors['parsed']['colpi_analizzare'] = int(colpi_an_str); assert 1 <= errors['parsed']['colpi_analizzare'] <= 10
        except: errors['messages'].append("Colpi Analisi Spia SE (1-10).")
        try: errors['parsed']['top_n_singoli'] = int(top_s_str); assert 1 <= errors['parsed']['top_n_singoli'] <= 15 # Adattato per SE
        except: errors['messages'].append("Top N Singoli Spia SE (1-15).")
        try: errors['parsed']['top_n_ambi'] = int(top_a_str); assert 1 <= errors['parsed']['top_n_ambi'] <= 10
        except: errors['messages'].append("Top N Ambi Spia SE (1-10).")
        try: errors['parsed']['top_n_terni'] = int(top_t_str); assert 1 <= errors['parsed']['top_n_terni'] <= 5
        except: errors['messages'].append("Top N Terni Spia SE (1-5).")
        return errors if errors['messages'] else None


    # All'interno della classe AppSuperEnalotto

    def start_analisi_spia_superenalotto_thread(self):
        if any(t and t.is_alive() for t in [self.analysis_thread, self.check_thread, self.spia_se_thread, self.verifica_spia_se_thread]):
            messagebox.showwarning("Operazione in Corso", "Attendere il termine dell'operazione corrente.", parent=self.root)
            return

        s_date, e_date = self._get_spia_se_dates() # Assumiamo che questo recuperi le date correttamente
        if not s_date or not e_date: # Se _get_spia_se_dates ha restituito (None, None) o una delle due è None
            # Il messaggio di errore dovrebbe essere già stato mostrato da _get_spia_se_dates
            return 
        
        # Recupera i valori stringa dai widget
        data_source_val = self.file_path_var.get().strip()
        numeri_spia_str_val = self.numeri_spia_se_var.get().strip()
        colpi_an_str_val = self.spia_se_colpi_var.get()
        top_s_str_val = self.spia_se_top_n_singoli_var.get()
        top_a_str_val = self.spia_se_top_n_ambi_var.get()
        top_t_str_val = self.spia_se_top_n_terni_var.get()

        validation_errors = self._validate_spia_se_params( # Rinomino per chiarezza
            data_source_val, numeri_spia_str_val, 
            s_date, e_date, 
            colpi_an_str_val, top_s_str_val, 
            top_a_str_val, top_t_str_val
        )
        
        if validation_errors: # Se _validate_spia_se_params ha restituito un dizionario (cioè ci sono errori)
            messagebox.showerror("Errore Parametri Spia SE", "\n\n".join(validation_errors['messages']), parent=self.root)
            return
        
        # Se siamo qui, validation_errors è None, significa che non ci sono stati errori di validazione.
        # Quindi, possiamo procedere a parsare i parametri con sicurezza.
        try:
            p = { # Dizionario per i parametri parsati
                'numeri_spia': [int(x.strip()) for x in numeri_spia_str_val.split(',') if x.strip()],
                'colpi_analizzare': int(colpi_an_str_val),
                'top_n_singoli': int(top_s_str_val),
                'top_n_ambi': int(top_a_str_val),
                'top_n_terni': int(top_t_str_val)
            }
            # Ulteriore controllo post-parsing (anche se _validate_spia_se_params dovrebbe averli coperti)
            if not p['numeri_spia'] or not all(1 <= n <= 90 for n in p['numeri_spia']):
                raise ValueError("Numeri spia non validi dopo parsing.")
            if not (1 <= p['colpi_analizzare'] <= 10): raise ValueError("Colpi analizzare fuori range.")
            # ... Aggiungi controlli simili per top_n se necessario ...

        except ValueError as e_parse:
             messagebox.showerror("Errore Conversione Parametri", f"Errore durante la conversione dei parametri spia SE: {e_parse}", parent=self.root)
             return

        # Resetta i risultati precedenti dell'analisi spia SE
        self.last_spia_se_singoli = None
        self.last_spia_se_ambi = None
        self.last_spia_se_terni = None
        self.last_spia_se_data_fine_analisi = None
        self.last_spia_se_numeri_input = p['numeri_spia'] # Salva i numeri spia usati per questa analisi
        
        self.set_controls_state(tk.DISABLED) # Disabilita i controlli
        self.log_message_gui(f"\n=== Avvio Analisi Numeri Spia SuperEnalotto (Spia: {p['numeri_spia']}, Periodo: {s_date}-{e_date}) ===")
        self.log_message_gui(f"Sorgente Dati: {'URL' if data_source_val.startswith('http') else 'File Locale'} ({os.path.basename(data_source_val)})")
        self.log_message_gui(f"Colpi successivi da analizzare: {p['colpi_analizzare']}")
        self.log_message_gui(f"Parametri Top N: Singoli={p['top_n_singoli']}, Ambi={p['top_n_ambi']}, Terni={p['top_n_terni']}")
        self.log_message_gui("-" * 50)

        self._stop_event_spia_se.clear() # Resetta l'evento di stop
        self.spia_se_thread = threading.Thread(target=self.run_analisi_spia_superenalotto, 
            args=(data_source_val, s_date, e_date, p['numeri_spia'], p['colpi_analizzare'], 
                  p['top_n_singoli'], p['top_n_ambi'], p['top_n_terni'], self._stop_event_spia_se), 
            daemon=True, name="SpiaSuperEnalottoThread")
        self.spia_se_thread.start()

    def run_analisi_spia_superenalotto(self, data_source, start_date_str, end_date_str, 
                                     numeri_spia_list, colpi_an, 
                                     top_s, top_a, top_t, stop_event):
        try:
            if stop_event.is_set():
                self.log_message_gui("Analisi Spia SE: Annullata prima dell'inizio.")
                return

            self.log_message_gui(f"Analisi Spia SE: Caricamento dati per il periodo {start_date_str} - {end_date_str}...")
            df_spia_original, arr_spia_se_main, _, _, _ = carica_dati_superenalotto(
                data_source,
                start_date=start_date_str, 
                end_date=end_date_str,
                log_callback=self.log_message_gui
            )

            if stop_event.is_set():
                self.log_message_gui("Analisi Spia SE: Annullata dopo caricamento dati.")
                return

            if df_spia_original is None or df_spia_original.empty or arr_spia_se_main is None or len(arr_spia_se_main) == 0:
                self.log_message_gui(f"ERRORE (Spia SE): Dati storici insufficienti o non caricati per il periodo {start_date_str} - {end_date_str}.")
                self.last_spia_se_singoli = None; self.last_spia_se_ambi = None; self.last_spia_se_terni = None
                self.last_spia_se_data_fine_analisi = None
                return
            
            date_arr_spia_se_aligned = None
            try:
                df_temp_for_alignment = df_spia_original.copy()
                numeri_cols_check = [f'Num{i+1}' for i in range(6)] # 6 numeri per SuperEnalotto

                if not all(col in df_temp_for_alignment.columns for col in numeri_cols_check):
                    self.log_message_gui("ERRORE (Spia SE): Colonne Numeri (Num1-6) mancanti nel DataFrame per allineamento date.")
                    self.last_spia_se_singoli = None; self.last_spia_se_ambi = None; self.last_spia_se_terni = None
                    self.last_spia_se_data_fine_analisi = None
                    return

                for col in numeri_cols_check:
                    df_temp_for_alignment[col] = pd.to_numeric(df_temp_for_alignment[col], errors='coerce')
                
                df_aligned_with_arr_spia = df_temp_for_alignment.dropna(subset=numeri_cols_check).copy()
                
                if len(df_aligned_with_arr_spia) == len(arr_spia_se_main):
                    if 'Data' in df_aligned_with_arr_spia.columns:
                        date_arr_spia_se_aligned = df_aligned_with_arr_spia['Data'].values
                    else:
                         self.log_message_gui("ATTENZIONE (Spia SE): Colonna 'Data' non trovata nel DataFrame allineato.")
                else:
                    self.log_message_gui(f"ATTENZIONE (Spia SE): Disallineamento tra numeri ({len(arr_spia_se_main)}) e date ({len(df_aligned_with_arr_spia)}) dopo pulizia. Il log del periodo potrebbe usare le date di input.")
            except Exception as e_align:
                 self.log_message_gui(f"ERRORE (Spia SE): Problema durante allineamento date: {e_align}")


            self.log_message_gui("Analisi Spia SE: Inizio calcolo frequenze numeri/ambi/terni spiati...")
            res_s, res_a, res_t, num_occ, d_ini_scan, d_fin_scan = calcola_numeri_spia_superenalotto(
                arr_spia_se_main, date_arr_spia_se_aligned, numeri_spia_list, colpi_an, 
                top_s, top_a, top_t, 
                self.log_message_gui, stop_event
            )

            if stop_event.is_set():
                self.log_message_gui("Analisi Spia SE: Elaborazione terminata a causa di richiesta di stop.")
                return

            self.log_message_gui("-" * 40 + "\nRISULTATI ANALISI SPIA SUPERENALOTTO (Spia: " + str(self.last_spia_se_numeri_input) + "):" + "-" * 40)
            
            periodo_effettivo_usato_inizio = d_ini_scan if d_ini_scan not in [None, "N/A"] else start_date_str
            periodo_effettivo_usato_fine = d_fin_scan if d_fin_scan not in [None, "N/A"] else end_date_str
            
            self.log_message_gui(f"Periodo analizzato: {periodo_effettivo_usato_inizio} - {periodo_effettivo_usato_fine}")
            self.log_message_gui(f"Occorrenze dei numeri spia (come combinazione) trovate: {num_occ}")
            self.log_message_gui(f"Analizzati {colpi_an} colpi successivi per ogni occorrenza.")

            if num_occ > 0:
                self.last_spia_se_singoli = [int(s_val[0]) for s_val in res_s] if res_s else []
                self.last_spia_se_ambi = [tuple(map(int, a_val[0])) for a_val in res_a] if res_a else []
                self.last_spia_se_terni = [tuple(map(int, t_val[0])) for t_val in res_t] if res_t else []
                self.last_spia_se_data_fine_analisi = periodo_effettivo_usato_fine 
                
                if res_s:
                    self.log_message_gui(f"\nTop {len(res_s)} SINGOLI spiati:")
                    for numero, frequenza in res_s:
                        self.log_message_gui(f"  - SINGOLO: {int(numero):02d} (Freq: {frequenza})")
                else:
                    self.log_message_gui("\nNessun singolo spia SE significativo trovato.")
                
                if res_a:
                    self.log_message_gui(f"\nTop {len(res_a)} AMBI spiati:")
                    for ambo_tuple, frequenza in res_a:
                        ambo_str = ", ".join(f"{int(n):02d}" for n in ambo_tuple)
                        self.log_message_gui(f"  - AMBO: {ambo_str} (Freq: {frequenza})")
                else:
                    self.log_message_gui("\nNessun ambo spia SE significativo trovato.")

                if res_t:
                    self.log_message_gui(f"\nTop {len(res_t)} TERNI spiati:")
                    for terno_tuple, frequenza in res_t:
                        terno_str = ", ".join(f"{int(n):02d}" for n in terno_tuple)
                        self.log_message_gui(f"  - TERNO: {terno_str} (Freq: {frequenza})")
                else:
                    self.log_message_gui("\nNessun terno spia SE significativo trovato.")
                
                if not res_s and not res_a and not res_t:
                    self.log_message_gui("\nNessun risultato spia (singolo, ambo o terno) significativo trovato nonostante le occorrenze spia.")

            elif not (stop_event and stop_event.is_set()): 
                self.log_message_gui(f"Nessuna occorrenza dei numeri spia {self.last_spia_se_numeri_input} trovata nel periodo selezionato.")
                self.last_spia_se_singoli = None; self.last_spia_se_ambi = None; self.last_spia_se_terni = None
                self.last_spia_se_data_fine_analisi = None
           
        except Exception as e:
            self.log_message_gui(f"\nERRORE CRITICO durante l'analisi dei Numeri Spia SE: {e}")
            self.log_message_gui(traceback.format_exc())
            self.last_spia_se_singoli = None; self.last_spia_se_ambi = None; self.last_spia_se_terni = None
            self.last_spia_se_data_fine_analisi = None
        finally:
            if not (stop_event and stop_event.is_set()) and hasattr(self, 'spia_se_thread') and self.spia_se_thread is not None : 
                self.log_message_gui("="*15 + " Analisi Numeri Spia SuperEnalotto Completata " + "="*15 + "\n")
            
            self.set_controls_state(tk.NORMAL) 
            self.root.after(10, self._clear_spia_se_thread_ref)

    def _clear_spia_se_thread_ref(self): self.spia_se_thread = None; self._set_controls_state_tk(tk.NORMAL)

    def start_verifica_spia_superenalotto_thread(self):
        # Simile a start_verifica_spia_thread del 10eLotto, ma usa variabili _se_
        if any(t and t.is_alive() for t in [self.analysis_thread, self.check_thread, self.spia_se_thread, self.verifica_spia_se_thread]):
            messagebox.showwarning("Operazione in Corso", "Attendere.", parent=self.root); return
        if not (self.last_spia_se_singoli or self.last_spia_se_ambi or self.last_spia_se_terni) or not self.last_spia_se_data_fine_analisi:
            messagebox.showinfo("Nessun Risultato Spia SE", "Eseguire prima Analisi Spia SuperEnalotto.", parent=self.root); return
        try: num_colpi = int(self.check_spia_se_colpi_var.get()); assert 1 <= num_colpi <= 20
        except: messagebox.showerror("Errore", "Colpi verifica spia SE non validi (1-20).", parent=self.root); return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Risultati Spia SuperEnalotto ({num_colpi} Colpi) ===")
        self._stop_event_verifica_spia_se.clear()
        self.verifica_spia_se_thread = threading.Thread(target=self.run_verifica_spia_superenalotto_results,
            args=(self.file_path_var.get(), self.last_spia_se_data_fine_analisi, num_colpi, self._stop_event_verifica_spia_se),
            daemon=True, name="VerificaSpiaSEThread")
        self.verifica_spia_se_thread.start()

    # All'interno della classe AppSuperEnalotto

    def run_verifica_spia_superenalotto_results(self, data_source, data_fine_an_spia_se, num_colpi_ver, stop_event):
        try:
            try:
                last_date_obj = datetime.strptime(data_fine_an_spia_se, '%Y-%m-%d')
                check_start_date_obj = last_date_obj + timedelta(days=1)
                check_start_date_str = check_start_date_obj.strftime('%Y-%m-%d')
            except ValueError as ve:
                self.log_message_gui(f"ERRORE formato data fine analisi spia SE '{data_fine_an_spia_se}': {ve}")
                return

            if stop_event.is_set():
                self.log_message_gui("Verifica Spia SE: Annullata prima caricamento dati.")
                return

            self.log_message_gui(f"Verifica Spia SE: Caricamento dati per verifica da {check_start_date_str} in avanti...")
            df_chk, arr_chk_main, _, _, _ = carica_dati_superenalotto(
                data_source, 
                start_date=check_start_date_str, 
                end_date=None, 
                log_callback=self.log_message_gui
            )
            
            if stop_event.is_set(): 
                self.log_message_gui("Verifica Spia SE: Annullata dopo caricamento dati.")
                return
            
            if df_chk is None or df_chk.empty or arr_chk_main is None or len(arr_chk_main) == 0: 
                self.log_message_gui(f"Verifica Spia SE: Nessuna estrazione trovata dopo {data_fine_an_spia_se} (a partire da {check_start_date_str}) o caricamento fallito.")
                return

            num_estrazioni_disponibili = len(arr_chk_main)
            num_colpi_effettivi_verifica = min(num_colpi_ver, num_estrazioni_disponibili)
            
            if num_colpi_effettivi_verifica == 0 :
                 self.log_message_gui(f"Verifica Spia SE: Nessuna estrazione disponibile per la verifica dopo {data_fine_an_spia_se}.")
                 return

            self.log_message_gui(f"Verifica Spia SE: Trovate {num_estrazioni_disponibili} estrazioni successive. Verifico le prossime {num_colpi_effettivi_verifica}...");

            # Contatori per i successi
            colpi_con_hit_singoli = 0; max_singoli_indovinati_per_colpo = 0
            colpi_con_hit_ambi = 0;    max_ambi_indovinati_per_colpo = 0
            colpi_con_hit_terni = 0;   max_terni_indovinati_per_colpo = 0

            for i in range(num_colpi_effettivi_verifica):
                if stop_event.is_set():
                    self.log_message_gui(f"Verifica Spia SE: Interrotta al colpo {i+1}.")
                    break
                
                estrazione_attuale_set = set(arr_chk_main[i])
                data_estrazione_attuale_str = "N/D"
                if 'Data' in df_chk.columns and i < len(df_chk) and pd.notna(df_chk.iloc[i]['Data']):
                    try: data_estrazione_attuale_str = pd.to_datetime(df_chk.iloc[i]['Data']).strftime('%Y-%m-%d')
                    except Exception: pass
                
                log_colpo_header = f"Colpo Spia SE {i+1:02d} ({data_estrazione_attuale_str}):"
                found_in_this_draw_overall = False

                # Verifica Singoli
                if self.last_spia_se_singoli:
                    indovinati_singoli_in_draw = [s for s in self.last_spia_se_singoli if s in estrazione_attuale_set]
                    if indovinati_singoli_in_draw:
                        self.log_message_gui(f"{log_colpo_header} SINGOLI: {len(indovinati_singoli_in_draw)}/{len(self.last_spia_se_singoli)} trovati -> {sorted(indovinati_singoli_in_draw)}")
                        colpi_con_hit_singoli +=1 
                        max_singoli_indovinati_per_colpo = max(max_singoli_indovinati_per_colpo, len(indovinati_singoli_in_draw))
                        found_in_this_draw_overall = True
                
                # Verifica Ambi
                if self.last_spia_se_ambi:
                    indovinati_ambi_in_draw = [tuple(sorted(ambo)) for ambo in self.last_spia_se_ambi if set(ambo).issubset(estrazione_attuale_set)]
                    if indovinati_ambi_in_draw:
                        self.log_message_gui(f"{log_colpo_header} AMBI: {len(indovinati_ambi_in_draw)}/{len(self.last_spia_se_ambi)} trovati -> {sorted(indovinati_ambi_in_draw)}")
                        colpi_con_hit_ambi +=1
                        max_ambi_indovinati_per_colpo = max(max_ambi_indovinati_per_colpo, len(indovinati_ambi_in_draw))
                        found_in_this_draw_overall = True

                # Verifica Terni
                if self.last_spia_se_terni:
                    indovinati_terni_in_draw = [tuple(sorted(terno)) for terno in self.last_spia_se_terni if set(terno).issubset(estrazione_attuale_set)]
                    if indovinati_terni_in_draw:
                        self.log_message_gui(f"{log_colpo_header} TERNI: {len(indovinati_terni_in_draw)}/{len(self.last_spia_se_terni)} trovati -> {sorted(indovinati_terni_in_draw)}")
                        colpi_con_hit_terni +=1
                        max_terni_indovinati_per_colpo = max(max_terni_indovinati_per_colpo, len(indovinati_terni_in_draw))
                        found_in_this_draw_overall = True
                
                if not found_in_this_draw_overall: 
                     self.log_message_gui(f"{log_colpo_header} Nessun risultato spia SE trovato.")

            if not stop_event.is_set(): 
                self.log_message_gui("-" * 40)
                self.log_message_gui(f"Verifica Risultati Spia SE ({num_colpi_effettivi_verifica} colpi) Riepilogo:")
                if self.last_spia_se_singoli:
                    self.log_message_gui(f"  SINGOLI: Uscite in {colpi_con_hit_singoli} colpi. Max singoli indovinati per colpo: {max_singoli_indovinati_per_colpo} (su {len(self.last_spia_se_singoli)} proposti).")
                if self.last_spia_se_ambi:
                    self.log_message_gui(f"  AMBI: Uscite in {colpi_con_hit_ambi} colpi. Max ambi indovinati per colpo: {max_ambi_indovinati_per_colpo} (su {len(self.last_spia_se_ambi)} proposti).")
                if self.last_spia_se_terni:
                    self.log_message_gui(f"  TERNI: Uscite in {colpi_con_hit_terni} colpi. Max terni indovinati per colpo: {max_terni_indovinati_per_colpo} (su {len(self.last_spia_se_terni)} proposti).")
                
                if not (colpi_con_hit_singoli or colpi_con_hit_ambi or col_con_hit_terni): # Corretto qui
                    self.log_message_gui("  Nessun risultato spia (singolo, ambo o terno) trovato nei colpi verificati.")

        except Exception as e:
            self.log_message_gui(f"ERRORE CRITICO durante la verifica dei risultati spia SE: {e}\n{traceback.format_exc()}")
        finally:
            if not stop_event.is_set():
                self.log_message_gui("\n=== Verifica Risultati Spia SuperEnalotto Completata ===")
            self.set_controls_state(tk.NORMAL)
            self.root.after(10, self._clear_verifica_spia_se_thread_ref)

    def _clear_verifica_spia_se_thread_ref(self): self.verifica_spia_se_thread = None; self._set_controls_state_tk(tk.NORMAL)

# 4. Modifica on_close
    def on_close(self):
        self.log_message_gui("Richiesta chiusura finestra SuperEnalotto...")
        # Segnala stop a TUTTI i thread
        self._stop_event_analysis.set()
        self._stop_event_check.set()
        self._stop_event_spia_se.set() # NUOVO
        self._stop_event_verifica_spia_se.set() # NUOVO
        
        active_threads = [t for t in [
            self.analysis_thread, self.check_thread, 
            self.spia_se_thread, self.verifica_spia_se_thread # NUOVI
        ] if t and t.is_alive()]
        
        # ... (logica di join dei thread e destroy come prima) ...
        if active_threads:
            self.log_message_gui(f"Attendo terminazione thread SE: {[t.name for t in active_threads]} (max 3s)")
            for t in active_threads: t.join(timeout=3.0)
        
        if self.root and self.root.winfo_exists(): self.root.destroy()


    # --- Metodo start_analysis_thread (MODIFICATO per leggere n_cv_splits) ---
    def start_analysis_thread(self):
        """Avvia il thread per l'analisi e la previsione."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Analisi in Corso", "Analisi SuperEnalotto già in esecuzione.", parent=self.root); return
        if self.check_thread and self.check_thread.is_alive():
            messagebox.showwarning("Verifica in Corso", "Verifica in corso. Attendere.", parent=self.root); return

        #<editor-fold desc="Recupero e Validazione Parametri GUI">
        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi in corso..."); self.attendibilita_label_var.set("")
        self.last_update_label_var.set("Data ultimo aggiornamento ...");
        self.last_prediction_numbers = None; self.last_prediction_full = None
        self.last_prediction_end_date = None; self.last_prediction_date_str = None
        self.check_button.config(state=tk.DISABLED)

        # Recupero valori
        data_source = self.file_path_var.get().strip()
        start_date_str, end_date_str = "", ""
        try:
            if HAS_TKCALENDAR and isinstance(self.start_date_entry, DateEntry): start_date_str, end_date_str = self.start_date_entry.get_date().strftime('%Y-%m-%d'), self.end_date_entry.get_date().strftime('%Y-%m-%d')
            else: start_date_str, end_date_str = self.start_date_entry_var.get(), self.end_date_entry_var.get() # Usa le var per Entry
        except Exception as e_date_get: messagebox.showerror("Errore Date", f"Errore lettura date: {e_date_get}", parent=self.root); return

        seq_len_str, num_predict_str = self.seq_len_var.get(), self.num_predict_var.get()
        hidden_layers_str, loss_function, optimizer = self.hidden_layers_var.get(), self.loss_var.get(), self.optimizer_var.get()
        dropout_str, l1_str, l2_str = self.dropout_var.get(), self.l1_var.get(), self.l2_var.get()
        epochs_str, batch_size_str, patience_str, min_delta_str = self.epochs_var.get(), self.batch_size_var.get(), self.patience_var.get(), self.min_delta_var.get()
        cv_splits_str = self.cv_splits_var.get() # NUOVO: Leggi CV splits

        # Validazione e conversione
        errors = []; sequence_length, num_predictions = 12, 6; hidden_layers_config = [128, 64]
        dropout_rate, l1_reg, l2_reg = 0.25, 0.0, 0.0; max_epochs, batch_size, patience, min_delta = 100, 128, 15, 0.0001
        n_cv_splits = DEFAULT_SUPERENALOTTO_CV_SPLITS # NUOVO: Default CV

        if not data_source: errors.append("- Specificare URL o percorso file.")
        elif not data_source.startswith(("http://", "https://")) and not os.path.exists(data_source): errors.append(f"- File locale non trovato: {data_source}")
        try: start_dt, end_dt = datetime.strptime(start_date_str, '%Y-%m-%d'), datetime.strptime(end_date_str, '%Y-%m-%d'); assert start_dt <= end_dt
        except: errors.append("- Date non valide (YYYY-MM-DD) o inizio > fine.")
        try: sequence_length = int(seq_len_str); assert 3 <= sequence_length <= 50
        except: errors.append("- Seq. Input non valida (3-50).")
        try: num_predictions = int(num_predict_str); assert 6 <= num_predictions <= 15
        except: errors.append("- Numeri da Prevedere non validi (6-15).")
        try: layers_str = [x.strip() for x in hidden_layers_str.split(',') if x.strip()]; assert layers_str; hidden_layers_config = [int(x) for x in layers_str]; assert all(n>0 for n in hidden_layers_config)
        except: errors.append("- Hidden Layers non validi (es. 128,64).")
        if not loss_function: errors.append("- Selezionare Loss Function.")
        if not optimizer: errors.append("- Selezionare Optimizer.")
        try: dropout_rate = float(dropout_str); assert 0.0 <= dropout_rate < 1.0
        except: errors.append("- Dropout Rate non valido (0.0 - 0.99).")
        try: l1_reg = float(l1_str); assert l1_reg >= 0
        except: errors.append("- L1 Strength non valido (>= 0).")
        try: l2_reg = float(l2_str); assert l2_reg >= 0
        except: errors.append("- L2 Strength non valido (>= 0).")
        try: max_epochs = int(epochs_str); assert max_epochs >= 10
        except: errors.append("- Max Epoche non valido (>= 10).")
        try: batch_size = int(batch_size_str); assert batch_size > 0 and (batch_size & (batch_size-1) == 0)
        except: errors.append("- Batch Size non valido (potenza di 2 > 0).")
        try: patience = int(patience_str); assert patience >= 3
        except: errors.append("- ES Patience non valida (>= 3).")
        try: min_delta = float(min_delta_str); assert min_delta >= 0
        except: errors.append("- ES Min Delta non valido (>= 0).")
        # NUOVO: Validazione CV splits
        try: n_cv_splits = int(cv_splits_str); assert 2 <= n_cv_splits <= 20
        except: errors.append(f"- Numero CV Splits non valido (2-20).")

        if errors:
            messagebox.showerror("Errore Parametri Input", "Correggere i seguenti errori:\n\n" + "\n".join(errors), parent=self.root)
            self.result_label_var.set("Errore parametri."); return
        #</editor-fold>

        self.set_controls_state(tk.DISABLED) # Disabilita input, abilita stop
        self.log_message_gui("=== Avvio Analisi SuperEnalotto (FE & CV - Thread) ===")
        self.log_message_gui(f"Sorgente: {data_source}")
        self.log_message_gui(f"Periodo: {start_date_str} - {end_date_str}")
        self.log_message_gui(f"Params: Seq={sequence_length}, Pred={num_predictions}, CV={n_cv_splits}")
        self.log_message_gui(f"Modello: HL={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate:.2f}, L1={l1_reg:.4f}, L2={l2_reg:.4f}")
        self.log_message_gui(f"Training: Epochs={max_epochs}, Batch={batch_size}, Pat={patience}, MinDelta={min_delta:.6f}")
        self.log_message_gui("-" * 40)

        # Resetta evento stop e avvia thread
        self._stop_event_analysis.clear()
        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=( # Passa tutti i parametri, incluso n_cv_splits e stop_event
                data_source, start_date_str, end_date_str, sequence_length,
                loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                num_predictions, n_cv_splits, # Passa n_cv_splits
                self._stop_event_analysis # <<< Passa l'evento
            ),
            daemon=True,
            name="SuperEnalottoAnalysisThread"
        )
        self.analysis_thread.start()
        # Riabilita/disabilita controlli dopo l'avvio del thread (principalmente per il bottone Stop)
        self.set_controls_state(tk.NORMAL) # Chiamata fittizia per aggiornare stati


    # --- Metodo run_analysis (MODIFICATO per passare n_cv_splits e stop_event) ---
    def run_analysis(self, data_source, start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions, n_cv_splits, # Riceve n_cv_splits
                     stop_event): # Riceve stop_event
        """Funzione eseguita nel thread secondario per analisi con FE e CV."""
        self.last_prediction_numbers = None; self.last_prediction_full = None
        self.last_prediction_end_date = None; self.last_prediction_date_str = None
        analysis_success = False; final_attendibilita_msg = "Analisi non completata."
        final_last_update_date = None; previsione_completa_result = None

        try:
            # Esegui l'analisi vera e propria passando stop_event
            previsione_completa_result, final_attendibilita_msg, final_last_update_date = analisi_superenalotto(
                file_path=data_source, start_date=start_date, end_date=end_date,
                sequence_length=sequence_length, loss_function=loss_function,
                optimizer=optimizer, dropout_rate=dropout_rate, l1_reg=l1_reg,
                l2_reg=l2_reg, hidden_layers_config=hidden_layers_config,
                max_epochs=max_epochs, batch_size=batch_size, patience=patience,
                min_delta=min_delta, num_predictions=num_predictions,
                n_cv_splits=n_cv_splits, # Passa n_cv_splits
                log_callback=self.log_message_gui,
                stop_event=stop_event # <<< Passa l'evento
            )

            # Verifica se l'analisi è stata interrotta DALLA funzione analisi_superenalotto
            if stop_event.is_set() and previsione_completa_result is None:
                 self.log_message_gui("Analisi interrotta durante l'elaborazione.")
                 if not final_attendibilita_msg or "interrotta" not in final_attendibilita_msg.lower():
                     final_attendibilita_msg = "Analisi Interrotta"
                 analysis_success = False
            else:
                 # Valuta successo normale
                 analysis_success = (isinstance(previsione_completa_result, list) and previsione_completa_result and
                                     len(previsione_completa_result) == num_predictions and
                                     all(isinstance(item, dict) and 'number' in item for item in previsione_completa_result))

            # Aggiorna variabili interne e log solo se successo e non interrotto
            if analysis_success:
                self.last_prediction_full = previsione_completa_result
                self.last_prediction_numbers = sorted([item['number'] for item in previsione_completa_result])
                try:
                    self.last_prediction_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                    self.last_prediction_date_str = end_date
                    self.log_message_gui(f"Previsione valida generata e salvata (dati fino al {end_date}).")
                except ValueError:
                    self.log_message_gui(f"ATT: Errore formato data fine ({end_date}) post-analisi."); self.last_prediction_end_date = None; self.last_prediction_date_str = None
            elif not stop_event.is_set(): # Log fallimento solo se non interrotto
                self.log_message_gui(f"Analisi completata ma senza previsione valida. Msg: {final_attendibilita_msg}")

            # Aggiorna label data ultimo aggiornamento
            last_update_str = final_last_update_date.strftime('%Y-%m-%d') if final_last_update_date else "N/D"
            self.root.after(0, lambda: self.last_update_label_var.set(f"Dati analizzati fino al: {last_update_str}"))

            # Aggiorna risultato GUI
            self.set_result(previsione_completa_result, final_attendibilita_msg)

        except Exception as e_run:
            self.log_message_gui(f"\nERRORE CRITICO run_analysis: {e_run}\n{traceback.format_exc()}")
            final_attendibilita_msg = f"Errore critico: {e_run}"
            self.set_result(None, final_attendibilita_msg)
            analysis_success = False
        finally:
             # Log completamento solo se non interrotto
             if not stop_event.is_set():
                 self.log_message_gui("\n=== Analisi SuperEnalotto (FE & CV - Thread) Completata ===")
             # Riabilita controlli e pulisci ref thread
             self.set_controls_state(tk.NORMAL) # Riabilita/Disabilita controlli
             self.root.after(10, self._clear_analysis_thread_ref) # Usa helper per pulire ref

    # NUOVO: Metodo per fermare il thread di analisi
    def stop_analysis_thread(self):
        """Imposta l'evento per fermare il thread di analisi in corso."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.log_message_gui("\n!!! Richiesta di interruzione analisi... !!!")
            self._stop_event_analysis.set()
            # Il bottone Stop verrà disabilitato automaticamente da set_controls_state
            # quando il thread terminerà e is_analysis_running diventerà False.
            # Potremmo disabilitarlo subito qui per feedback immediato, ma attendiamo
            # che set_controls_state faccia il suo corso per coerenza.
        else:
            self.log_message_gui("Nessuna analisi da interrompere.")

    # NUOVO: Helper per pulire riferimento thread analisi
    def _clear_analysis_thread_ref(self):
        """Helper per pulire il riferimento al thread nel thread principale."""
        self.analysis_thread = None
        # Potrebbe essere necessario ri-valutare lo stato dei pulsanti qui
        # se la verifica era disabilitata a causa dell'analisi.
        # Richiamiamo set_controls_state per sicurezza.
        self.set_controls_state(tk.NORMAL)

    # --- Metodi start_check_thread, run_check_results (INVARIATI) ---
    def start_check_thread(self):
        """Avvia il thread per la verifica dell'ultima previsione."""
        if self.check_thread and self.check_thread.is_alive(): messagebox.showwarning("Verifica in Corso", "Verifica già in esecuzione.", parent=self.root); return
        if self.analysis_thread and self.analysis_thread.is_alive(): messagebox.showwarning("Analisi in Corso", "Attendere fine analisi.", parent=self.root); return
        if not self.last_prediction_numbers or not self.last_prediction_end_date or not self.last_prediction_date_str:
            messagebox.showinfo("Nessuna Previsione", "Nessuna previsione valida per verifica.", parent=self.root); return
        if not isinstance(self.last_prediction_numbers, list) or not all(isinstance(n, int) for n in self.last_prediction_numbers):
            messagebox.showerror("Errore Previsione", "Dati previsione salvata corrotti.", parent=self.root); self.last_prediction_numbers = None; self.set_controls_state(tk.NORMAL); return

        try: num_colpi_to_check = int(self.check_colpi_var.get()); assert 1 <= num_colpi_to_check <= 100
        except: messagebox.showerror("Errore Input", "Numero colpi verifica non valido (1-100).", parent=self.root); return

        data_source_for_check = self.file_path_var.get().strip()
        if not data_source_for_check: messagebox.showerror("Errore Sorgente Dati", "Specificare sorgente dati per verifica.", parent=self.root); return
        if not data_source_for_check.startswith(("http://", "https://")) and not os.path.exists(data_source_for_check):
            messagebox.showerror("Errore File", f"File dati locale '{os.path.basename(data_source_for_check)}' non trovato per verifica.", parent=self.root); return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Previsione ({num_colpi_to_check} Colpi Max) ===")
        self.log_message_gui(f"Previsione Numeri: {self.last_prediction_numbers}")
        self.log_message_gui(f"Basata su dati fino a: {self.last_prediction_date_str}")
        self.log_message_gui(f"Sorgente verifica: {data_source_for_check}")
        self.log_message_gui("-" * 40)

        # Resetta evento stop check (anche se non c'è bottone) e avvia thread
        self._stop_event_check.clear()
        self.check_thread = threading.Thread(
            target=self.run_check_results,
            args=(data_source_for_check, self.last_prediction_numbers,
                  self.last_prediction_date_str, num_colpi_to_check,
                  self._stop_event_check), # Passa evento stop
            daemon=True,
            name="SuperEnalottoCheckThread"
        )
        self.check_thread.start()
        self.set_controls_state(tk.NORMAL) # Aggiorna stati (es. disabilita Run)

    def run_check_results(self, data_source, prediction_numbers_to_check, last_analysis_date_str, num_colpi_to_check, stop_event):
        """Esegue la verifica nel thread, controllando stop_event."""
        try:
            try: last_date_obj = datetime.strptime(last_analysis_date_str, '%Y-%m-%d'); check_start_date_str = (last_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
            except ValueError as ve_date: self.log_message_gui(f"ERRORE CRITICO formato data analisi: {ve_date}. Verifica annullata."); return

            if stop_event.is_set(): self.log_message_gui("Verifica annullata prima caricamento dati."); return

            self.log_message_gui(f"Caricamento dati verifica (da {check_start_date_str})...")
            df_check, numeri_principali_check, _, _, _ = carica_dati_superenalotto(data_source, start_date=check_start_date_str, end_date=None, log_callback=self.log_message_gui)

            if stop_event.is_set(): self.log_message_gui("Verifica annullata dopo caricamento dati."); return

            if df_check is None: self.log_message_gui("ERRORE: Caricamento dati verifica fallito."); return
            if df_check.empty: self.log_message_gui(f"INFO: Nessuna estrazione trovata dopo {last_analysis_date_str}."); return
            if numeri_principali_check is None or len(numeri_principali_check) == 0: self.log_message_gui(f"ERRORE: Dati trovati post {last_analysis_date_str}, ma estrazione numeri fallita."); return

            num_disp = len(numeri_principali_check); num_eff = min(num_colpi_to_check, num_disp)
            self.log_message_gui(f"Trovate {num_disp} estrazioni. Verifico prossime {num_eff}...");
            prediction_set = set(prediction_numbers_to_check)
            self.log_message_gui(f"Numeri previsti (Set): {prediction_set}"); self.log_message_gui("-" * 40)

            colpo = 0; found_hit = False; max_score = 0
            for i in range(num_eff):
                if stop_event.is_set(): self.log_message_gui(f"Verifica interrotta al colpo {colpo + 1}."); break
                colpo += 1
                try:
                    row = df_check.iloc[i]; date_str = row['Data'].strftime('%Y-%m-%d'); actual = numeri_principali_check[i]; actual_set = set(actual)
                    hits = prediction_set.intersection(actual_set); n_hits = len(hits); max_score = max(max_score, n_hits)
                    log_line = f"Colpo {colpo:02d}/{num_eff:02d} ({date_str}): {sorted(list(actual_set))} -> "
                    if n_hits > 0: found_hit = True; pts = f"{n_hits} punti"; log_line += f"*** {pts}! ({sorted(list(hits))}) ***"
                    else: log_line += "Nessun risultato."
                    self.log_message_gui(log_line)
                except Exception as e_row: self.log_message_gui(f"ERR colpo {colpo} ({date_str}): {e_row}"); continue

            if not stop_event.is_set():
                 self.log_message_gui("-" * 40)
                 if not found_hit: self.log_message_gui(f"Nessun risultato nei {num_eff} colpi verificati.")
                 else: self.log_message_gui(f"Verifica completata. Punteggio massimo: {max_score} punti.")

        except Exception as e_check: self.log_message_gui(f"ERRORE CRITICO verifica: {e_check}\n{traceback.format_exc()}")
        finally:
             if not stop_event.is_set(): self.log_message_gui("\n=== Verifica SuperEnalotto (Thread) Completata ===")
             self.set_controls_state(tk.NORMAL) # Riabilita/Disabilita
             self.root.after(10, self._clear_check_thread_ref) # Pulisci ref

    # NUOVO: Helper per pulire riferimento thread verifica
    def _clear_check_thread_ref(self):
        """Helper per pulire il riferimento al thread nel thread principale."""
        self.check_thread = None
        self.set_controls_state(tk.NORMAL) # Ricalcola stati

    # --- Metodo on_close (MODIFICATO per gestire stop events) ---
    def on_close(self):
        """Gestisce la richiesta di chiusura della finestra."""
        self.log_message_gui("Richiesta chiusura finestra...")

        # 1. Segnala ai thread di fermarsi
        self._stop_event_analysis.set()
        self._stop_event_check.set()

        # 2. Attendi terminazione thread (con timeout)
        timeout_secs = 3.0
        wait_start = time.time()
        threads_to_wait = []
        analysis_thread_local = self.analysis_thread
        if analysis_thread_local and analysis_thread_local.is_alive(): threads_to_wait.append(analysis_thread_local)
        check_thread_local = self.check_thread
        if check_thread_local and check_thread_local.is_alive(): threads_to_wait.append(check_thread_local)

        if threads_to_wait:
            self.log_message_gui(f"Attendo terminazione thread: {[t.name for t in threads_to_wait]} (max {timeout_secs:.1f}s)")
            for thread in threads_to_wait:
                remaining_timeout = max(0.1, timeout_secs - (time.time() - wait_start))
                try:
                    thread.join(timeout=remaining_timeout)
                    status = "non terminato (timeout)" if thread.is_alive() else "terminato"
                    log_level = "ATTENZIONE" if thread.is_alive() else "INFO"
                    self.log_message_gui(f"{log_level}: Thread {thread.name} {status}.")
                except Exception as e: self.log_message_gui(f"Errore durante join di {thread.name}: {e}")
        else:
            self.log_message_gui("Nessun thread attivo da attendere.")

        # 3. Distruggi finestra
        self.log_message_gui("Distruzione finestra Tkinter.")
        try:
             self.analysis_thread = None; self.check_thread = None # Pulisci ref
             self.root.destroy()
        except tk.TclError as e: print(f"TclError durante root.destroy() (normale se già distrutta): {e}")
        except Exception as e: print(f"Errore imprevisto durante root.destroy(): {e}")

# --- Fine Classe GUI ---


# --- Funzione di Lancio (INVARIATA NELLA LOGICA) ---
def launch_superenalotto_window(parent_window=None):
    """Crea e lancia la finestra dell'applicazione SuperEnalotto."""
    try:
        win_root = tk.Toplevel(parent_window) if parent_window else tk.Tk()
        app_instance = AppSuperEnalotto(win_root) # Init imposta titolo/geo
        # Centra finestra (opzionale)
        win_root.update_idletasks()
        w = win_root.winfo_width(); h = win_root.winfo_height()
        ws = win_root.winfo_screenwidth(); hs = win_root.winfo_screenheight()
        x = (ws // 2) - (w // 2); y = (hs // 2) - (h // 2)
        win_root.geometry(f'+{x}+{y}')
        win_root.lift()
        win_root.focus_force()
        if not parent_window: win_root.mainloop()
    except Exception as e_launch:
        print(f"ERRORE CRITICO lancio finestra SuperEnalotto: {e_launch}\n{traceback.format_exc()}")
        try: messagebox.showerror("Errore Avvio Applicazione", f"Errore critico:\n{e_launch}", parent=parent_window)
        except: pass

# --- Blocco Esecuzione Standalone (INVARIATO) ---
if __name__ == "__main__":
    print("Esecuzione Modulo SuperEnalotto ML Predictor (FE & CV) standalone...")
    print("-" * 60)
    print("Requisiti: tensorflow, pandas, numpy, requests, scikit-learn")
    print("Opzionale (GUI): tkcalendar")
    print("Installazione: pip install tensorflow pandas numpy requests scikit-learn tkcalendar")
    print("-" * 60)
    try:
        if sys.platform == "win32": from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1); print("INFO: DPI awareness impostato (Win).")
    except Exception as e_dpi: print(f"Nota: Impossibile impostare DPI awareness: {e_dpi}")

    launch_superenalotto_window(parent_window=None)
    print("\nFinestra SuperEnalotto chiusa. Programma terminato.")