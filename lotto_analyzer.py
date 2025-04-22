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

try:
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

# --- Constanti ---
RUOTE_LOTTO = ["BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI", "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE"]
NUM_ESTRATTI_LOTTO = 5
MAX_NUMERO_LOTTO = 90 # I numeri vanno da 1 a 90
DEFAULT_CHECK_COLPI = 18 # Default colpi da verificare

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
    if log_widget and window:
        # Usa after per eseguire l'aggiornamento nel thread principale di Tkinter
        try:
            window.after(10, lambda: _update_log_widget(log_widget, message))
        except tk.TclError: # Finestra chiusa
             print(f"Log GUI TclError (window likely closed): {message}")


def _update_log_widget(log_widget, message):
    """Funzione helper per aggiornare il widget di log."""
    try:
        # Salva lo stato corrente prima di modificarlo
        current_state = log_widget.cget('state')
        log_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, str(message) + "\n") # Assicura sia stringa
        log_widget.see(tk.END) # Scrolla alla fine
        # Ripristina lo stato precedente solo se era DISABLED
        if current_state == tk.DISABLED:
             log_widget.config(state=tk.DISABLED)
    except tk.TclError:
        # Può succedere se la finestra viene chiusa mentre after() è in coda
        print(f"Log GUI TclError: {message}") # Log su console come fallback
    except Exception as e:
        # Gestisci altri errori imprevisti
        print(f"Log GUI unexpected error: {e}\nMessage: {message}")


# --- Funzioni Specifiche per il Lotto ---

def carica_dati_lotto(folder_path, calculation_wheels, start_date=None, end_date=None, log_callback=None):
    """
    Carica i dati del Lotto dai file specificati per le ruote di calcolo.
    Gestisce il formato YYYY/MM/DD e la colonna extra con la sigla ruota.
    Rileva l'header controllando il formato data della prima colonna.
    CORRETTO: Usa inclusive='both' per pd.Series.between.
    Combina i dati in un unico DataFrame ordinato per data.
    """
    all_data = []
    required_cols = 7
    expected_date_format = '%Y/%m/%d'

    if not calculation_wheels:
        if log_callback: log_callback("ERRORE: Nessuna ruota di calcolo selezionata.")
        return None, None

    if log_callback: log_callback(f"Inizio caricamento dati Lotto per ruote: {', '.join(calculation_wheels)}")
    if log_callback: log_callback(f"Cartella dati: {folder_path}")

    for wheel_name in calculation_wheels:
        file_path = os.path.join(folder_path, f"{wheel_name}.txt")
        if log_callback: log_callback(f"--- Caricamento Ruota: {wheel_name} ({os.path.basename(file_path)}) ---")
        try:
            if not os.path.exists(file_path):
                if log_callback: log_callback(f"ATTENZIONE: File non trovato per ruota {wheel_name} - {file_path}. Salto.")
                continue

            lines = []
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
            file_read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        lines = f.readlines()
                    file_read_success = True
                    # if log_callback: log_callback(f"File letto con encoding: {enc}") # Log meno verboso
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if log_callback: log_callback(f"ERRORE imprevisto lettura file ({enc}): {e}")
                    continue

            if not file_read_success:
                 if log_callback: log_callback(f"ERRORE: Impossibile leggere il file {file_path} con encoding noti.")
                 continue

            if not lines:
                if log_callback: log_callback("ATTENZIONE: File vuoto.")
                continue

            header_skipped = False
            data_lines = lines
            if lines:
                first_line_content = lines[0].strip()
                if first_line_content:
                    first_col = first_line_content.split('\t')[0]
                    try:
                        datetime.strptime(first_col, expected_date_format)
                    except ValueError:
                        # if log_callback: log_callback(f"Prima riga non corrisponde al formato data {expected_date_format}, considerata header e saltata.") # Log meno verboso
                        data_lines = lines[1:]
                        header_skipped = True
                    except Exception as e_parse:
                         if log_callback: log_callback(f"Errore inatteso parsing data prima riga '{first_col}': {e_parse}. Riga saltata.")
                         data_lines = lines[1:]
                         header_skipped = True
                else:
                     data_lines = lines[1:]
                     header_skipped = True

            if not data_lines:
                 if log_callback: log_callback(f"ATTENZIONE: Nessuna riga dati trovata dopo eventuale salto header ({wheel_name}).")
                 continue

            wheel_data = []
            malformed_lines = 0
            line_num_offset = 1 if header_skipped else 0
            for i, line in enumerate(data_lines):
                values = line.strip().split('\t')
                if len(values) >= required_cols:
                    row_data = values[:required_cols]
                    wheel_data.append(row_data)
                else:
                    malformed_lines += 1

            if malformed_lines > 0 and log_callback:
                 log_callback(f"ATTENZIONE ({wheel_name}): {malformed_lines} righe scartate (numero colonne errato).")

            if not wheel_data:
                if log_callback: log_callback(f"ATTENZIONE: Nessuna riga valida trovata per {wheel_name} dopo controllo colonne.")
                continue

            col_names_read = ['DataStr', 'SiglaRuotaFile', 'Num1Str', 'Num2Str', 'Num3Str', 'Num4Str', 'Num5Str']
            df_wheel = pd.DataFrame(wheel_data, columns=col_names_read)

            original_rows = len(df_wheel)
            df_wheel['Data'] = pd.to_datetime(df_wheel['DataStr'], format=expected_date_format, errors='coerce')
            df_wheel = df_wheel.dropna(subset=['Data'])
            dropped_rows = original_rows - len(df_wheel)
            if dropped_rows > 0 and log_callback:
                log_callback(f"({wheel_name}): Rimosse {dropped_rows} righe (data non valida o formato non {expected_date_format}).")

            if df_wheel.empty:
                if log_callback: log_callback(f"ATTENZIONE: Nessuna data valida per {wheel_name} dopo la conversione.")
                continue

            numeri_cols_str = [f'Num{i+1}Str' for i in range(NUM_ESTRATTI_LOTTO)]
            numeri_cols_final = [f'Num{i+1}' for i in range(NUM_ESTRATTI_LOTTO)]
            rows_b4_num_conv = len(df_wheel)
            invalid_num_count = 0

            for i in range(NUM_ESTRATTI_LOTTO):
                col_str = numeri_cols_str[i]
                col_final = numeri_cols_final[i]
                df_wheel[col_final] = pd.to_numeric(df_wheel[col_str], errors='coerce')
                invalid_mask = ~df_wheel[col_final].between(1, MAX_NUMERO_LOTTO, inclusive='both') # Corretto
                invalid_num_count += invalid_mask.sum()
                df_wheel.loc[invalid_mask, col_final] = pd.NA

            df_wheel = df_wheel.dropna(subset=numeri_cols_final)
            dropped_num_rows = rows_b4_num_conv - len(df_wheel)
            if dropped_num_rows > 0 and log_callback:
                 log_callback(f"({wheel_name}): Scartate {dropped_num_rows} righe a causa di numeri non validi (non numerici o fuori range 1-{MAX_NUMERO_LOTTO}). ({invalid_num_count} numeri individuali problematici rilevati)")

            if df_wheel.empty:
                 if log_callback: log_callback(f"ATTENZIONE: Nessuna riga valida per {wheel_name} dopo pulizia numeri.")
                 continue

            df_wheel = df_wheel[['Data'] + numeri_cols_final]
            df_wheel['Ruota'] = wheel_name
            all_data.append(df_wheel)
            # if log_callback: log_callback(f"Dati validi caricati e puliti per {wheel_name}: {len(df_wheel)} estrazioni.") # Log meno verboso

        except FileNotFoundError:
             if log_callback: log_callback(f"ERRORE: File non trovato - {file_path}")
             continue
        except Exception as e:
            if log_callback: log_callback(f"ERRORE grave durante caricamento ruota {wheel_name}: {e}\n{traceback.format_exc()}")
            continue

    if not all_data:
        if log_callback: log_callback("ERRORE: Nessun dato valido caricato da nessuna delle ruote selezionate.")
        return None, None

    df_combined = pd.concat(all_data, ignore_index=True)
    if log_callback: log_callback(f"Dati combinati da {len(all_data)} ruote. Totale righe prima di sort/filter: {len(df_combined)}.")

    df_combined = df_combined.sort_values(by='Data').reset_index(drop=True)

    rows_before_date_filter = len(df_combined)
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            df_combined = df_combined[df_combined['Data'] >= start_dt]
        except Exception as e:
            if log_callback: log_callback(f"Errore filtro data inizio ({start_date}): {e}")
    if end_date:
         try:
            end_dt = pd.to_datetime(end_date)
            df_combined = df_combined[df_combined['Data'] <= end_dt]
         except Exception as e:
            if log_callback: log_callback(f"Errore filtro data fine ({end_date}): {e}")

    rows_after_date_filter = len(df_combined)
    if log_callback:
        log_callback(f"Righe dopo filtro date GUI ({start_date} - {end_date}): {rows_after_date_filter} (rimosse {rows_before_date_filter - rows_after_date_filter})")

    if df_combined.empty:
        if log_callback: log_callback("ERRORE: Nessun dato rimasto dopo il filtro per data della GUI.")
        return df_combined.copy(), None

    numeri_cols_final = [f'Num{i+1}' for i in range(NUM_ESTRATTI_LOTTO)]
    numeri_array = None
    try:
        if all(col in df_combined.columns for col in numeri_cols_final):
            numeri_array = df_combined[numeri_cols_final].values.astype(int)
        else:
             missing_cols = [col for col in numeri_cols_final if col not in df_combined.columns]
             if log_callback: log_callback(f"ERRORE: Colonne numeri mancanti nel DataFrame combinato finale: {missing_cols}")
             return df_combined.copy(), None
    except Exception as e:
        if log_callback: log_callback(f"ERRORE estrazione finale numeri_array dai dati combinati: {e}")
        return df_combined.copy(), None

    if log_callback: log_callback(f"Caricamento/Filtraggio Lotto completato. Righe finali nel DataFrame: {len(df_combined)}. Shape array numeri: {numeri_array.shape if numeri_array is not None else 'None'}")
    return df_combined.copy(), numeri_array


def prepara_sequenze_lotto(numeri_array, sequence_length=5, log_callback=None):
    """Prepara le sequenze Input (X) e Target (y) per il modello Lotto."""
    if numeri_array is None or len(numeri_array) == 0:
        if log_callback: log_callback("ERRORE (prep_seq_lotto): Array numeri input vuoto.")
        return None, None

    if numeri_array.shape[1] != NUM_ESTRATTI_LOTTO:
         if log_callback: log_callback(f"ERRORE (prep_seq_lotto): L'array numeri non ha {NUM_ESTRATTI_LOTTO} colonne (shape: {numeri_array.shape})")
         return None, None

    X, y = [], []
    num_estrazioni = len(numeri_array)
    if log_callback: log_callback(f"Preparazione sequenze Lotto: seq_len={sequence_length}, num_estrazioni={num_estrazioni}.")

    if num_estrazioni <= sequence_length:
        if log_callback: log_callback(f"ERRORE: Estrazioni ({num_estrazioni}) <= seq_len ({sequence_length}). Impossibile creare sequenze.")
        return None, None

    valid_seq_count = 0
    invalid_target_count = 0

    for i in range(num_estrazioni - sequence_length):
        input_sequence_rows = numeri_array[i : i + sequence_length]
        target_extraction = numeri_array[i + sequence_length]

        if np.all((target_extraction >= 1) & (target_extraction <= MAX_NUMERO_LOTTO)):
            input_flattened = input_sequence_rows.flatten()
            target_vector = np.zeros(MAX_NUMERO_LOTTO, dtype=int)
            target_vector[target_extraction - 1] = 1

            X.append(input_flattened)
            y.append(target_vector)
            valid_seq_count += 1
        else:
            invalid_target_count += 1

    if invalid_target_count > 0 and log_callback:
        log_callback(f"ATTENZIONE: Scartate {invalid_target_count} sequenze a causa di numeri target non validi (fuori da 1-{MAX_NUMERO_LOTTO}).")

    if not X:
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata.")
        return None, None

    if log_callback: log_callback(f"Create {valid_seq_count} sequenze Lotto valide.")
    return np.array(X), np.array(y)


def build_model_lotto(input_shape, hidden_layers=[512, 256, 128], loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    """Costruisce il modello Keras per il Lotto."""
    if log_callback: log_callback(f"Costruzione modello Lotto: Input Shape={input_shape}, Hidden Layers={hidden_layers}, Loss={loss_function}, Optimizer={optimizer}, Dropout={dropout_rate}, L1={l1_reg}, L2={l2_reg}")

    if not isinstance(input_shape, tuple) or len(input_shape) != 1 or not isinstance(input_shape[0], int) or input_shape[0] <= 0:
         if log_callback: log_callback(f"ERRORE: input_shape non valido: {input_shape}. Deve essere una tupla con un intero positivo (es. (50,)).")
         return None

    model = tf.keras.Sequential(name="Modello_Lotto")
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input_Layer"))
    reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg) if l1_reg > 0 or l2_reg > 0 else None

    if not hidden_layers:
        if log_callback: log_callback("ATTENZIONE: Nessun hidden layer specificato.")
    else:
        for i, units in enumerate(hidden_layers):
            layer_name_base = f"Layer_{i+1}"
            if not isinstance(units, int) or units <= 0:
                if log_callback: log_callback(f"ERRORE: Numero di unità non valido ({units}) per hidden layer {i+1}.")
                return None
            model.add(tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=reg, name=f"{layer_name_base}_Dense"))
            model.add(tf.keras.layers.BatchNormalization(name=f"{layer_name_base}_BN"))
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate, name=f"{layer_name_base}_Dropout"))

    model.add(tf.keras.layers.Dense(MAX_NUMERO_LOTTO, activation='sigmoid', name="Output_Layer"))

    try:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        if log_callback: log_callback("Modello Lotto compilato con successo.")
    except Exception as e:
        if log_callback: log_callback(f"ERRORE durante la compilazione del modello Lotto: {e}")
        return None

    return model


class LogCallback(tf.keras.callbacks.Callback):
    """Callback Keras per loggare le epoche nella GUI Tkinter."""
    def __init__(self, log_callback_func):
        super().__init__()
        self.log_callback_func = log_callback_func
        self._is_running = True

    def stop_logging(self):
        self._is_running = False

    def on_epoch_end(self, epoch, logs=None):
        if not self._is_running or not self.log_callback_func:
            return
        logs = logs or {}
        msg = f"Epoca {epoch+1:03d} - "
        items = [f"{k.replace('_',' ').replace('val ','v_')}: {v:.4f}" for k, v in logs.items()]
        # Chiama la funzione di log della GUI (che usa window.after)
        self.log_callback_func(msg + ", ".join(items))

def genera_previsione_lotto(model, X_input_last_sequence, num_predictions=5, log_callback=None):
    """
    Genera la previsione dei numeri del Lotto usando il modello addestrato.
    RESTITUISCE I NUMERI ORDINATI PER PROBABILITÀ DECRESCENTE (ATTENDIBILITÀ).
    """
    if log_callback: log_callback(f"Generazione previsione Lotto per {num_predictions} numeri (ordinati per probabilità)...") # Messaggio aggiornato

    if model is None:
        if log_callback: log_callback("ERRORE (genera_prev_lotto): Modello non valido.")
        return None
    if X_input_last_sequence is None or X_input_last_sequence.size == 0:
        if log_callback: log_callback("ERRORE (genera_prev_lotto): Input (ultima sequenza) vuoto.")
        return None
    if not (1 <= num_predictions <= MAX_NUMERO_LOTTO):
        if log_callback: log_callback(f"ERRORE: num_predictions ({num_predictions}) non valido (deve essere tra 1 e {MAX_NUMERO_LOTTO}).")
        return None

    try:
        # --- Gestione Shape Input (come prima) ---
        if X_input_last_sequence.ndim == 1:
             input_reshaped = X_input_last_sequence.reshape(1, -1)
        elif X_input_last_sequence.ndim == 2 and X_input_last_sequence.shape[0] == 1:
             input_reshaped = X_input_last_sequence
        else:
             # Prova a gestire il caso di più righe prendendo la prima
             if X_input_last_sequence.ndim == 2 and X_input_last_sequence.shape[0] > 0:
                 input_reshaped = X_input_last_sequence[0].reshape(1, -1) # Prendi la prima riga
                 if log_callback: log_callback(f"ATTENZIONE: Ricevuto input con shape {X_input_last_sequence.shape}, usata la prima riga per la previsione.")
             else:
                if log_callback: log_callback(f"ERRORE: Shape input non gestita per la previsione: {X_input_last_sequence.shape}")
                return None

        # --- Verifica Consistenza Shape Input/Modello (come prima) ---
        expected_input_features = model.input_shape[-1]
        if input_reshaped.shape[1] != expected_input_features:
            if log_callback: log_callback(f"ERRORE Shape: Input per predict ({input_reshaped.shape[1]} features) non corrisponde all'input del modello ({expected_input_features} features).")
            return None

        # --- Esecuzione Previsione (come prima) ---
        probabilities = model.predict(input_reshaped)

        if probabilities is None or probabilities.size == 0:
            if log_callback: log_callback("ERRORE: model.predict() ha restituito None o array vuoto.")
            return None

        # --- Estrazione Vettore Probabilità (come prima) ---
        if probabilities.ndim == 2 and probabilities.shape[0] == 1 and probabilities.shape[1] == MAX_NUMERO_LOTTO:
            probs_vector = probabilities[0] # Array NumPy di 90 probabilità (o score)
        else:
            if log_callback: log_callback(f"ERRORE: Output shape inatteso da predict: {probabilities.shape}. Atteso (1, {MAX_NUMERO_LOTTO}).")
            return None

        # --- MODIFICA CHIAVE QUI ---
        # 1. Trova gli indici dei numeri ordinati per probabilità (dal meno probabile al più probabile)
        #    Esempio: se probs_vector[10] è la probabilità più alta, l'ultimo elemento di sorted_indices sarà 10.
        #             se probs_vector[5] è la più bassa, il primo elemento sarà 5.
        sorted_indices = np.argsort(probs_vector)

        # 2. Prendi gli ultimi 'num_predictions' indici (quelli con probabilità più alta)
        #    Questi indici sono ancora ordinati dalla probabilità più bassa (tra i top) alla più alta.
        #    Esempio (num_pred=3): Se i 3 più probabili hanno indici 10, 25, 60 (in ordine di probabilità),
        #                         top_indices_ascending_prob sarà [60, 25, 10] (assumendo altre probabilità minori)
        #                         NO, argsort ordina dal più piccolo al più grande, quindi sarà [indice_terzo, indice_secondo, indice_primo]
        #                         Esempio concreto: probs=[0.1, 0.8, 0.5, 0.9, 0.2] -> argsort=[0, 4, 2, 1, 3]
        #                         Se num_pred=3 -> [-3:] -> [2, 1, 3] (indici di 0.5, 0.8, 0.9)
        top_indices_ascending_prob = sorted_indices[-num_predictions:]

        # 3. Inverti l'ordine di questi indici per averli dal più probabile al meno probabile
        #    Esempio: da [2, 1, 3] a [3, 1, 2]
        top_indices_descending_prob = top_indices_ascending_prob[::-1]

        # 4. Converti gli indici (0-89) in numeri Lotto (1-90) mantenendo questo ordine
        #    Esempio: da [3, 1, 2] a [3+1, 1+1, 2+1] -> [4, 2, 3]
        predicted_numbers_by_prob = [index + 1 for index in top_indices_descending_prob]
        # --- FINE MODIFICA CHIAVE ---

        if log_callback:
            # Logga i numeri nell'ordine di probabilità decrescente (come richiesto)
            log_callback(f"Numeri Lotto predetti ({len(predicted_numbers_by_prob)} ord. per probabilità decr.): {predicted_numbers_by_prob}")
            # Opzionale: Loggare anche le probabilità per verifica
            try:
                probs_dict_ordered = {num: f"{probs_vector[num-1]:.4f}" for num in predicted_numbers_by_prob}
                log_callback(f"    Probabilità associate (Top {num_predictions}): {probs_dict_ordered}")
            except IndexError:
                log_callback("    (Errore nel recuperare le probabilità associate per il log)")


        # Ritorna la lista ordinata per probabilità
        return predicted_numbers_by_prob

    except Exception as e:
        if log_callback: log_callback(f"ERRORE durante generazione previsione Lotto: {e}\n{traceback.format_exc()}")
        return None

def analisi_lotto(folder_path, calculation_wheels, game_wheels, # Accetta lista game_wheels
                   start_date, end_date, sequence_length=5,
                   loss_function='binary_crossentropy', optimizer='adam',
                   dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                   hidden_layers_config=[512, 256, 128],
                   max_epochs=100, batch_size=32, patience=15, min_delta=0.0001,
                   num_predictions=5,
                   log_callback=None):
    """
    Analizza i dati del Lotto e genera UNA previsione basata sulle ruote di calcolo.
    La lista game_wheels è usata per logging e viene restituita con la previsione.
    """
    if log_callback:
        log_callback(f"=== Avvio Analisi Lotto ===")
        log_callback(f"Cartella Dati: {folder_path}")
        log_callback(f"Ruote Calcolo: {', '.join(calculation_wheels)}")
        log_callback(f"Ruote Gioco (per verifica): {', '.join(game_wheels)}") # Log modificato
        log_callback(f"Periodo: {start_date} - {end_date}")
        log_callback(f"Parametri Seq/Pred: SeqIn={sequence_length}, NumOut={num_predictions}")
        log_callback(f"Modello: L={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")
        log_callback(f"Training: Epochs={max_epochs}, Batch={batch_size}, Pat={patience}, MinDelta={min_delta}")
        log_callback(f"---------------------------------")

    # 1. Carica dati dalle RUOTE DI CALCOLO
    df_combined, numeri_array = None, None
    try:
        df_combined, numeri_array = carica_dati_lotto(
            folder_path, calculation_wheels, start_date, end_date, log_callback=log_callback
        )
    except Exception as e:
         if log_callback: log_callback(f"ERRORE CRITICO durante carica_dati_lotto: {e}\n{traceback.format_exc()}")
         return None, "Caricamento dati fallito (eccezione)"

    if df_combined is None: return None, "Caricamento dati Lotto fallito (df None)"
    if numeri_array is None :
        msg = "Nessun dato numerico valido trovato dopo caricamento/pulizia dalle ruote di calcolo" if df_combined.empty else "Errore estrazione/pulizia numeri dalle ruote di calcolo"
        if log_callback: log_callback(f"ERRORE: {msg}")
        return None, msg

    if len(numeri_array) < sequence_length + 1:
        msg = f"ERRORE: Dati numerici combinati insuff. ({len(numeri_array)}) per seq_len ({sequence_length}+1)"
        if log_callback: log_callback(msg)
        return None, msg

    # 2. Prepara sequenze Lotto
    X, y = None, None
    try:
        X, y = prepara_sequenze_lotto(numeri_array, sequence_length, log_callback=log_callback)
        if X is None or y is None or len(X) == 0:
            return None, "Creazione sequenze Lotto fallita"

        min_samples_for_split = 5
        if len(X) < min_samples_for_split:
            if log_callback: log_callback(f"ATTENZIONE: Solo {len(X)} campioni disponibili (< {min_samples_for_split}). L'addestramento potrebbe essere inaffidabile.")
            if len(X) < 2:
                msg = f"ERRORE: Troppi pochi campioni ({len(X)} < 2) per l'addestramento."
                if log_callback: log_callback(msg)
                return None, msg

    except Exception as e:
         if log_callback: log_callback(f"Errore preparazione sequenze Lotto: {e}\n{traceback.format_exc()}")
         return None, f"Errore prep sequenze Lotto: {e}"

    # 3. Normalizza Input (X)
    X_scaled = X.astype(np.float32) / MAX_NUMERO_LOTTO

    # 4. Split train/validation
    X_train, X_val, y_train, y_val = [], [], [], []
    split_ratio = 0.8
    if len(X_scaled) >= min_samples_for_split:
        split_idx = max(1, min(int(split_ratio * len(X_scaled)), len(X_scaled) - 1))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        if len(X_train)==0 or len(X_val)==0:
            if log_callback: log_callback("ERRORE: Split fallito nonostante controllo iniziale.")
            return None, "Split Train/Validation fallito"
        if log_callback: log_callback(f"Split Dati: {len(X_train)} training, {len(X_val)} validation")
    else:
        if log_callback: log_callback(f"ATTENZIONE: Training su tutti i {len(X_scaled)} campioni (troppo pochi per validation split)")
        X_train, y_train = X_scaled, y
        X_val, y_val = None, None

    # 5. Costruisci e addestra il modello Lotto
    model, history, gui_log_callback = None, None, None
    final_val_loss, final_train_loss_at_best = float('inf'), float('inf')
    try:
        tf.keras.backend.clear_session()
        if X_train is None or X_train.size == 0 or X_train.shape[1] == 0:
            if log_callback: log_callback("ERRORE: Dati di training (X_train) non validi o vuoti prima della costruzione del modello.")
            return None, "Errore dati training"

        input_shape_lotto = (X_train.shape[1],)

        model = build_model_lotto(input_shape_lotto, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if model is None:
            return None, "Costruzione modello Lotto fallita"

        monitor_metric = 'val_loss' if (X_val is not None and len(X_val) > 0) else 'loss'
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        )
        gui_log_callback = LogCallback(log_callback)
        validation_data = (X_val, y_val) if (X_val is not None and len(X_val) > 0) else None

        if log_callback: log_callback(f"Inizio addestramento modello Lotto (monitor='{monitor_metric}')...")
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, gui_log_callback],
            verbose=0
        )

        if history and history.history:
            epochs_ran = len(history.history.get('loss', []))
            if log_callback: log_callback(f"Addestramento terminato dopo {epochs_ran} epoche.")

            train_loss_hist = history.history.get('loss', [])
            val_loss_hist = history.history.get('val_loss', [])

            best_epoch_idx = -1
            if val_loss_hist:
                best_epoch_idx = np.argmin(val_loss_hist)
                final_val_loss = val_loss_hist[best_epoch_idx]
                if best_epoch_idx < len(train_loss_hist):
                    final_train_loss_at_best = train_loss_hist[best_epoch_idx]
                else:
                    final_train_loss_at_best = float('inf')
            elif train_loss_hist:
                 best_epoch_idx = np.argmin(train_loss_hist)
                 final_train_loss_at_best = train_loss_hist[best_epoch_idx]
                 final_val_loss = float('inf')

            if best_epoch_idx != -1 and log_callback:
                log_msg = f"Miglior Epoca (secondo '{monitor_metric}'): {best_epoch_idx+1}"
                if final_val_loss != float('inf'):
                    log_msg += f", Val Loss: {final_val_loss:.4f}"
                if final_train_loss_at_best != float('inf'):
                     log_msg += f", Train Loss (a quella epoca): {final_train_loss_at_best:.4f}"
                log_callback(log_msg)
        else:
             if log_callback: log_callback("ATTENZIONE: L'addestramento non ha restituito una history valida.")

    except Exception as e:
        msg = f"ERRORE durante l'addestramento del modello Lotto: {e}"
        if log_callback: log_callback(msg); log_callback(traceback.format_exc())
        if gui_log_callback: gui_log_callback.stop_logging()
        return None, msg

    # 6. Prepara input per la previsione finale e genera
    numeri_predetti, attendibilita_msg = None, "Attendibilità N/D"
    try:
        if log_callback: log_callback("Preparazione input per la previsione finale Lotto...")
        if numeri_array is None or len(numeri_array) < sequence_length:
            if log_callback: log_callback("ERRORE: Dati originali (numeri_array) insufficienti per creare l'ultima sequenza per la previsione.")
            return None, "Dati insuff. per input previsione"

        last_sequence_raw = numeri_array[-sequence_length:]
        input_pred_flattened = last_sequence_raw.flatten()
        input_pred_scaled = input_pred_flattened.astype(np.float32) / MAX_NUMERO_LOTTO

        numeri_predetti = genera_previsione_lotto(
            model, input_pred_scaled, num_predictions, log_callback=log_callback
        )

        if numeri_predetti is None:
            return None, "Generazione previsione Lotto fallita."

        ratio = float('inf')
        if final_train_loss_at_best > 1e-7 and final_val_loss != float('inf'):
            ratio = final_val_loss / final_train_loss_at_best

        attendibilita = "Non Determinabile"
        if final_val_loss == float('inf') and final_train_loss_at_best != float('inf'):
             attendibilita = "Non Valutabile (solo training)"
        elif ratio < 1.2: attendibilita = "Alta"
        elif ratio < 1.8: attendibilita = "Media"
        elif final_val_loss != float('inf'):
             attendibilita = "Bassa (possibile overfitting)"

        attendibilita_msg = f"Attendibilità: {attendibilita}"
        if ratio != float('inf'):
            attendibilita_msg += f" (ratio V/T: {ratio:.2f})"
        elif final_val_loss == float('inf') and final_train_loss_at_best != float('inf'):
             attendibilita_msg += f" (Train Loss: {final_train_loss_at_best:.4f})"

        if log_callback: log_callback(attendibilita_msg)
        # Ritorna la previsione unica e l'attendibilità. La classe AppLotto gestirà la lista game_wheels.
        return numeri_predetti, attendibilita_msg

    except Exception as e:
         if log_callback: log_callback(f"Errore durante la fase di previsione finale Lotto: {e}\n{traceback.format_exc()}")
         return None, f"Errore previsione finale Lotto: {e}"
# --- Fine Funzione analisi_lotto ---

# --- Definizione Classe AppLotto COMPLETA E CORRETTA CON LABEL INFO ---
class AppLotto:
    def __init__(self, root):
        self.root = root
        # Aggiornato titolo per riflettere la modifica
        self.root.title("Analisi e Previsione Lotto (v1.2 - Info Selezione Multipla)")
        self.root.geometry("950x980") # Altezza per ospitare label aggiuntiva

        self.style = ttk.Style()
        try:
            if sys.platform == "win32": self.style.theme_use('vista')
            else: self.style.theme_use('clam')
        except tk.TclError: self.style.theme_use('default')

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Memorizzazione ultima previsione e ruote di gioco (plurale)
        self.last_prediction = None
        self.last_prediction_end_date = None
        self.last_prediction_game_wheels = [] # Ora una lista
        self.last_analysis_folder = None

        # --- Input Cartella Dati ---
        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Cartella Dati Estrazioni Lotto (.txt)", padding="10")
        self.folder_frame.pack(fill=tk.X, pady=5)
        self.folder_path_var = tk.StringVar(value=os.getcwd())
        self.folder_entry = ttk.Entry(self.folder_frame, textvariable=self.folder_path_var, width=80)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.browse_folder_button = ttk.Button(self.folder_frame, text="Sfoglia Cartella...", command=self.browse_folder)
        self.browse_folder_button.pack(side=tk.LEFT)

        # --- Selezione Ruote ---
        self.wheels_frame = ttk.Frame(self.main_frame)
        self.wheels_frame.pack(fill=tk.X, pady=5)
        self.wheels_frame.columnconfigure(0, weight=1)
        self.wheels_frame.columnconfigure(1, weight=1)

        # Frame Ruote di Calcolo (Sinistra)
        self.calc_wheels_frame = ttk.LabelFrame(self.wheels_frame, text="Ruote di Calcolo (per Addestramento)", padding="10")
        self.calc_wheels_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")

        # Lista Checkbutton per Ruote di Calcolo
        self.calc_wheel_vars = {}
        self.calc_wheel_checks = {}
        rows = (len(RUOTE_LOTTO) + 1) // 2
        for i, wheel in enumerate(RUOTE_LOTTO):
            var = BooleanVar(value=False) # Non selezionate di default
            chk = ttk.Checkbutton(self.calc_wheels_frame, text=wheel, variable=var)
            r, c = i % rows, i // rows
            chk.grid(row=r, column=c, padx=5, pady=2, sticky=tk.W)
            self.calc_wheel_vars[wheel] = var
            self.calc_wheel_checks[wheel] = chk

        # --- Frame Ruota di Gioco (Destra) con Listbox e istruzioni (MODIFICATO) ---
        self.game_wheel_frame = ttk.LabelFrame(self.wheels_frame, text="Ruote di Gioco (per Previsione/Verifica)", padding="10")
        self.game_wheel_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        # Configura righe/colonne del frame per posizionare correttamente listbox e label
        self.game_wheel_frame.rowconfigure(0, weight=1) # Riga per Listbox/Scrollbar si espande
        self.game_wheel_frame.rowconfigure(1, weight=0) # Riga per Label info non si espande
        self.game_wheel_frame.columnconfigure(0, weight=1) # Colonna unica si espande

        # Sotto-frame per Listbox e Scrollbar
        self.listbox_subframe = ttk.Frame(self.game_wheel_frame)
        # Usa grid per posizionare il subframe, sticky lo fa espandere, pady lascia spazio sotto
        self.listbox_subframe.grid(row=0, column=0, sticky='nsew', pady=(0, 3))
        self.listbox_subframe.rowconfigure(0, weight=1)
        self.listbox_subframe.columnconfigure(0, weight=1)

        self.game_wheel_listbox = Listbox(
            self.listbox_subframe, # Mettere nel sotto-frame
            selectmode=constants.EXTENDED, # Permette selezione multipla
            height=6, # Altezza della listbox
            exportselection=False
        )
        self.game_wheel_scrollbar = Scrollbar(
            self.listbox_subframe, # Mettere nel sotto-frame
            orient=constants.VERTICAL,
            command=self.game_wheel_listbox.yview
        )
        self.game_wheel_listbox.config(yscrollcommand=self.game_wheel_scrollbar.set)

        for wheel in RUOTE_LOTTO:
            self.game_wheel_listbox.insert(constants.END, wheel)

        # Posiziona Listbox e Scrollbar dentro il sotto-frame usando grid
        self.game_wheel_scrollbar.grid(row=0, column=1, sticky='ns')
        self.game_wheel_listbox.grid(row=0, column=0, sticky='nsew')

        # Etichetta informativa per la selezione multipla
        self.multi_select_info_label = ttk.Label(
            self.game_wheel_frame, # Aggiungi al frame delle ruote di gioco
            text="Per selezione multipla: Ctrl+Click / Shift+Click",
            font=('Helvetica', 8, 'italic'), # Font piccolo e corsivo
            anchor=tk.W # Allinea testo a sinistra (West)
        )
        # Posiziona l'etichetta usando grid nella riga sottostante al listbox_subframe
        self.multi_select_info_label.grid(row=1, column=0, sticky='ew', padx=5)
        # --- FINE SEZIONE MODIFICATA RUOTE GIOCO ---


        # --- Contenitore Parametri ---
        self.params_container = ttk.Frame(self.main_frame)
        self.params_container.pack(fill=tk.X, pady=5)
        self.params_container.columnconfigure(0, weight=1)
        self.params_container.columnconfigure(1, weight=1)

        # --- Colonna Sinistra: Parametri Dati ---
        self.data_params_frame = ttk.LabelFrame(self.params_container, text="Parametri Analisi", padding="10")
        self.data_params_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        ttk.Label(self.data_params_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        if HAS_TKCALENDAR:
             default_start = datetime.now() - pd.Timedelta(days=365*2)
             self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd', year=default_start.year, month=default_start.month, day=default_start.day)
             self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        else:
            self.start_date_entry = ttk.Entry(self.data_params_frame, width=12); self.start_date_entry.insert(0, (datetime.now() - pd.Timedelta(days=365*2)).strftime('%Y-%m-%d'))
            self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        if HAS_TKCALENDAR:
             self.end_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd'); self.end_date_entry.set_date(datetime.now())
             self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        else:
            self.end_date_entry = ttk.Entry(self.data_params_frame, width=12); self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
            self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Seq. Input (Storia):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.seq_len_var = tk.StringVar(value="10")
        self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=2, to=100, textvariable=self.seq_len_var, width=5, wrap=True)
        self.seq_len_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.data_params_frame, text="Numeri da Prevedere:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="5")
        self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=1, to=15, increment=1, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly')
        self.num_predict_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # --- Colonna Destra: Parametri Modello e Training ---
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello e Training", padding="10")
        self.model_params_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        self.model_params_frame.columnconfigure(1, weight=1)
        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.hidden_layers_var = tk.StringVar(value="256, 128")
        self.hidden_layers_entry = ttk.Entry(self.model_params_frame, textvariable=self.hidden_layers_var, width=25)
        self.hidden_layers_entry.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Loss Function:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.loss_var = tk.StringVar(value='binary_crossentropy')
        self.loss_combo = ttk.Combobox(self.model_params_frame, textvariable=self.loss_var, width=23, state='readonly', values=['binary_crossentropy', 'categorical_crossentropy', 'mse', 'mae', 'huber_loss'])
        self.loss_combo.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Optimizer:").grid(row=2, column=0, padx=5, pady=3, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value='adam')
        self.optimizer_combo = ttk.Combobox(self.model_params_frame, textvariable=self.optimizer_var, width=23, state='readonly', values=['adam', 'rmsprop', 'sgd', 'adagrad', 'adamw'])
        self.optimizer_combo.grid(row=2, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Label(self.model_params_frame, text="Dropout Rate (0-1):").grid(row=3, column=0, padx=5, pady=3, sticky=tk.W)
        self.dropout_var = tk.StringVar(value="0.3")
        self.dropout_spinbox = ttk.Spinbox(self.model_params_frame, from_=0.0, to=1.0, increment=0.05, format="%.2f", textvariable=self.dropout_var, width=7, wrap=True)
        self.dropout_spinbox.grid(row=3, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="L1 Strength (>=0):").grid(row=4, column=0, padx=5, pady=3, sticky=tk.W)
        self.l1_var = tk.StringVar(value="0.0")
        self.l1_entry = ttk.Entry(self.model_params_frame, textvariable=self.l1_var, width=7)
        self.l1_entry.grid(row=4, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="L2 Strength (>=0):").grid(row=5, column=0, padx=5, pady=3, sticky=tk.W)
        self.l2_var = tk.StringVar(value="0.0")
        self.l2_entry = ttk.Entry(self.model_params_frame, textvariable=self.l2_var, width=7)
        self.l2_entry.grid(row=5, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="Max Epoche:").grid(row=6, column=0, padx=5, pady=3, sticky=tk.W)
        self.epochs_var = tk.StringVar(value="100")
        self.epochs_spinbox = ttk.Spinbox(self.model_params_frame, from_=10, to=1000, increment=10, textvariable=self.epochs_var, width=7, wrap=True)
        self.epochs_spinbox.grid(row=6, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="Batch Size:").grid(row=7, column=0, padx=5, pady=3, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="32")
        batch_values = [str(2**i) for i in range(3, 10)]
        self.batch_size_combo = ttk.Combobox(self.model_params_frame, textvariable=self.batch_size_var, values=batch_values, width=5, state='readonly')
        self.batch_size_combo.grid(row=7, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="ES Patience:").grid(row=8, column=0, padx=5, pady=3, sticky=tk.W)
        self.patience_var = tk.StringVar(value="15")
        self.patience_spinbox = ttk.Spinbox(self.model_params_frame, from_=3, to=100, increment=1, textvariable=self.patience_var, width=7, wrap=True)
        self.patience_spinbox.grid(row=8, column=1, padx=5, pady=3, sticky=tk.W)
        ttk.Label(self.model_params_frame, text="ES Min Delta:").grid(row=9, column=0, padx=5, pady=3, sticky=tk.W)
        self.min_delta_var = tk.StringVar(value="0.0001")
        self.min_delta_entry = ttk.Entry(self.model_params_frame, textvariable=self.min_delta_var, width=10)
        self.min_delta_entry.grid(row=9, column=1, padx=5, pady=3, sticky=tk.W)

        # --- Pulsanti Azione ---
        self.action_frame = ttk.Frame(self.main_frame)
        self.action_frame.pack(pady=10)
        self.run_button = ttk.Button(self.action_frame, text="Avvia Analisi Lotto", command=self.start_analysis_thread)
        self.run_button.pack(side=tk.LEFT, padx=10)
        self.check_button = ttk.Button(self.action_frame, text="Verifica Ultima Previsione Lotto", command=self.start_check_thread, state=tk.DISABLED)
        # ---> MODIFICA LA RIGA SEGUENTE (cambia solo padx=5) <---
        self.check_button.pack(side=tk.LEFT, padx=5) # Padding ridotto
        # ---> AGGIUNGI LE RIGHE SEGUENTI (per Label e Spinbox) <---
        ttk.Label(self.action_frame, text="Colpi da Verificare:").pack(side=tk.LEFT, padx=(10, 2)) # Label
        self.check_colpi_var = tk.StringVar(value=str(DEFAULT_CHECK_COLPI)) # Variabile
        self.check_colpi_spinbox = ttk.Spinbox(self.action_frame, from_=1, to=100, increment=1, textvariable=self.check_colpi_var, width=5, wrap=True, state='readonly') # Spinbox
        self.check_colpi_spinbox.pack(side=tk.LEFT, padx=(0, 10)) # Posizionamento

        # --- Risultati ---
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Risultato Previsione Lotto", padding="10")
        self.results_frame.pack(fill=tk.X, pady=5)
        self.game_wheels_result_var = tk.StringVar(value="") # Plurale
        self.game_wheels_result_label = ttk.Label(self.results_frame, textvariable=self.game_wheels_result_var, font=('Helvetica', 10, 'italic'))
        self.game_wheels_result_label.pack(pady=2)
        self.result_label_var = tk.StringVar(value="I numeri previsti appariranno qui...")
        self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier', 14, 'bold'), foreground='darkblue')
        self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value="")
        self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 10, 'italic'))
        self.attendibilita_label.pack(pady=2)

        # --- Log Area ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monospace", 9)
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15, width=100, wrap=tk.WORD, state=tk.DISABLED, font=log_font)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Thread attivi
        self.analysis_thread = None
        self.check_thread = None

    # --- Metodi della Classe ---

    def browse_folder(self):
        foldername = filedialog.askdirectory(title="Seleziona la cartella contenente i file .txt delle ruote")
        if foldername:
            self.folder_path_var.set(foldername)
            self.log_message_gui(f"Cartella dati selezionata: {foldername}")
            found_files = glob.glob(os.path.join(foldername, '*.txt'))
            found_wheels = [os.path.basename(f).replace('.txt', '').upper() for f in found_files]
            missing = [r for r in RUOTE_LOTTO if r not in found_wheels]
            if missing:
                 self.log_message_gui(f"ATTENZIONE: File non trovati per le ruote: {', '.join(missing)}")

    def log_message_gui(self, message):
        log_message(message, self.log_text, self.root)

    def set_result(self, numbers, attendibilita, game_wheels): # game_wheels è una lista
         self.root.after(0, self._update_result_labels, numbers, attendibilita, game_wheels)

    def _update_result_labels(self, numbers, attendibilita, game_wheels): # game_wheels è una lista
        if game_wheels:
             wheels_str = ", ".join(game_wheels)
             self.game_wheels_result_var.set(f"Previsione per Ruote: {wheels_str}")
        else:
             self.game_wheels_result_var.set("")

        if numbers and isinstance(numbers, list):
             result_str = "  ".join(map(lambda x: f"{x:02d}", numbers))
             self.result_label_var.set(result_str)
        else:
            self.result_label_var.set("Previsione fallita. Controlla i log.")

        self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "")

    def set_controls_state(self, state):
        self.root.after(0, self._set_controls_state_tk, state)

    def _set_controls_state_tk(self, state):
        tk_state = tk.NORMAL if state == tk.NORMAL else tk.DISABLED

        widgets_to_toggle = [
            self.browse_folder_button, self.folder_entry,
            self.game_wheel_listbox, self.game_wheel_scrollbar,
            self.seq_len_entry, self.num_predict_spinbox,
            self.run_button,
            self.loss_combo, self.optimizer_combo,
            self.dropout_spinbox, self.l1_entry, self.l2_entry,
            self.hidden_layers_entry, self.epochs_spinbox,
            self.batch_size_combo, self.patience_spinbox,
            self.min_delta_entry,
            self.check_colpi_spinbox
        ]

        for chk in self.calc_wheel_checks.values():
            widgets_to_toggle.append(chk)

        if HAS_TKCALENDAR:
            try: self.start_date_entry.configure(state=tk_state)
            except: pass
            try: self.end_date_entry.configure(state=tk_state)
            except: pass
        else:
            widgets_to_toggle.extend([self.start_date_entry, self.end_date_entry])

        for widget in widgets_to_toggle:
            try:
                current_widget_state = tk_state
                if widget == self.run_button:
                    if tk_state == tk.NORMAL and self.check_thread and self.check_thread.is_alive():
                        current_widget_state = tk.DISABLED

                if isinstance(widget, (ttk.Combobox, ttk.Spinbox)) and hasattr(widget, 'configure'):
                     widget.configure(state='readonly' if current_widget_state == tk.NORMAL else tk.DISABLED)
                elif hasattr(widget, 'configure'):
                     widget.configure(state=current_widget_state)
            except (tk.TclError, AttributeError):
                 pass

        check_button_state = tk.DISABLED
        if tk_state == tk.NORMAL:
            if self.last_prediction is not None and self.last_prediction_game_wheels:
                if not (self.analysis_thread and self.analysis_thread.is_alive()):
                    check_button_state = tk.NORMAL
        try:
             self.check_button.configure(state=check_button_state)
        except (tk.TclError, AttributeError):
             pass

    def start_analysis_thread(self):
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Analisi in Corso", "Un'analisi è già in esecuzione.", parent=self.root)
            return
        if self.check_thread and self.check_thread.is_alive():
            messagebox.showwarning("Verifica in Corso", "Una verifica è in corso. Attendere il completamento.", parent=self.root)
            return

        self.log_text.config(state=tk.NORMAL); self.log_text.delete('1.0', tk.END); self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi Lotto in corso..."); self.attendibilita_label_var.set(""); self.game_wheels_result_var.set("")
        self.last_prediction = None
        self.last_prediction_game_wheels = []
        self.last_prediction_end_date = None
        self.last_analysis_folder = None

        errors = []
        folder_path = self.folder_path_var.get()
        if not folder_path or not os.path.isdir(folder_path):
            errors.append(f"Cartella dati non valida o inesistente:\n{folder_path}")

        selected_calc_wheels = [wheel for wheel, var in self.calc_wheel_vars.items() if var.get()]
        if not selected_calc_wheels:
            errors.append("Selezionare almeno una Ruota di Calcolo.")

        selected_indices = self.game_wheel_listbox.curselection()
        selected_game_wheels = [RUOTE_LOTTO[i] for i in selected_indices]
        if not selected_game_wheels:
             errors.append("Selezionare almeno una Ruota di Gioco dalla lista.")

        missing_files = []
        if not errors:
             for wheel in selected_calc_wheels:
                 fpath = os.path.join(folder_path, f"{wheel}.txt")
                 if not os.path.isfile(fpath):
                     missing_files.append(f"{wheel}.txt (Calcolo)")
             for wheel in selected_game_wheels:
                  fpath = os.path.join(folder_path, f"{wheel}.txt")
                  if not os.path.isfile(fpath) and f"{wheel}.txt (Gioco)" not in missing_files:
                     missing_files.append(f"{wheel}.txt (Gioco)")
        if missing_files:
             errors.append("File mancanti nella cartella selezionata:\n- " + "\n- ".join(missing_files))

        start_date_str, end_date_str = "", ""
        try:
            if HAS_TKCALENDAR:
                start_date_str = self.start_date_entry.get_date().strftime('%Y-%m-%d')
                end_date_str = self.end_date_entry.get_date().strftime('%Y-%m-%d')
            else:
                start_date_str = self.start_date_entry.get()
                end_date_str = self.end_date_entry.get()
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_dt > end_dt: errors.append("Data inizio non può essere successiva a Data fine.")
        except ValueError:
            errors.append("Formato date non valido (usare YYYY-MM-DD).")
        except Exception as e:
            errors.append(f"Errore recupero/validazione date: {e}")

        seq_len_str = self.seq_len_var.get()
        num_predict_str = self.num_predict_var.get()
        hidden_layers_str = self.hidden_layers_var.get()
        loss_function = self.loss_var.get()
        optimizer = self.optimizer_var.get()
        dropout_str = self.dropout_var.get()
        l1_str = self.l1_var.get(); l2_str = self.l2_var.get()
        epochs_str = self.epochs_var.get(); batch_size_str = self.batch_size_var.get()
        patience_str = self.patience_var.get(); min_delta_str = self.min_delta_var.get()

        sequence_length, num_predictions = 10, 5
        hidden_layers_config = [256, 128]
        dropout_rate, l1_reg, l2_reg = 0.3, 0.0, 0.0
        max_epochs, batch_size, patience, min_delta = 100, 32, 15, 0.0001

        try: sequence_length = int(seq_len_str); assert 2 <= sequence_length <= 100
        except: errors.append("Seq. Input non valida (2-100).")
        try: num_predictions = int(num_predict_str); assert 1 <= num_predictions <= 15
        except: errors.append(f"Numeri da Prevedere non validi (1-15).")
        try:
            layers_str = [x.strip() for x in hidden_layers_str.split(',') if x.strip()]
            if not layers_str: hidden_layers_config = []
            else: hidden_layers_config = [int(x) for x in layers_str]; assert all(n > 0 for n in hidden_layers_config)
        except: errors.append("Hidden Layers non validi (es. 256,128 o vuoto).")
        if not loss_function: errors.append("Selezionare Loss Function.")
        if not optimizer: errors.append("Selezionare Optimizer.")
        try: dropout_rate = float(dropout_str); assert 0.0 <= dropout_rate <= 1.0
        except: errors.append("Dropout Rate non valido (0.0-1.0).")
        try: l1_reg = float(l1_str); assert l1_reg >= 0
        except: errors.append("L1 Strength non valido (>= 0).")
        try: l2_reg = float(l2_str); assert l2_reg >= 0
        except: errors.append("L2 Strength non valido (>= 0).")
        try: max_epochs = int(epochs_str); assert max_epochs > 0
        except: errors.append("Max Epoche non valido (> 0).")
        try: batch_size = int(batch_size_str); assert batch_size > 0
        except: errors.append("Batch Size non valido (> 0).")
        try: patience = int(patience_str); assert patience >= 0
        except: errors.append("ES Patience non valida (>= 0).")
        try: min_delta = float(min_delta_str); assert min_delta >= 0
        except: errors.append("ES Min Delta non valido (>= 0).")

        if errors:
            messagebox.showerror("Errore Parametri Input Lotto", "\n\n".join(errors), parent=self.root)
            self.result_label_var.set("Errore parametri.")
            return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui("=== Avvio Analisi Lotto ===")
        self.log_message_gui(f"Cartella: {folder_path}")
        self.log_message_gui(f"Ruote Calcolo: {', '.join(selected_calc_wheels)}")
        self.log_message_gui(f"Ruote Gioco: {', '.join(selected_game_wheels)}")
        self.log_message_gui(f"Periodo: {start_date_str} - {end_date_str}")
        self.log_message_gui(f"Param Seq/Pred: SeqIn={sequence_length}, NumOut={num_predictions}")
        self.log_message_gui(f"Param Modello: Layers={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate:.2f}, L1={l1_reg:.4f}, L2={l2_reg:.4f}")
        self.log_message_gui(f"Param Training: Epochs={max_epochs}, Batch={batch_size}, Pat={patience}, MinDelta={min_delta:.5f}")

        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(folder_path, selected_calc_wheels, selected_game_wheels,
                  start_date_str, end_date_str, sequence_length,
                  loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                  hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                  num_predictions),
            daemon=True
        )
        self.analysis_thread.start()

    def run_analysis(self, folder_path, calculation_wheels, game_wheels,
                     start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions):
        numeri_predetti, attendibilita_msg, success = None, "Analisi Lotto fallita", False
        try:
            numeri_predetti, attendibilita_msg = analisi_lotto(
                folder_path=folder_path,
                calculation_wheels=calculation_wheels,
                game_wheels=game_wheels, # Passa la lista
                start_date=start_date, end_date=end_date,
                sequence_length=sequence_length,
                loss_function=loss_function, optimizer=optimizer,
                dropout_rate=dropout_rate, l1_reg=l1_reg, l2_reg=l2_reg,
                hidden_layers_config=hidden_layers_config,
                max_epochs=max_epochs, batch_size=batch_size,
                patience=patience, min_delta=min_delta,
                num_predictions=num_predictions,
                log_callback=self.log_message_gui
            )
            success = isinstance(numeri_predetti, list) and len(numeri_predetti) == num_predictions

        except Exception as e:
            self.log_message_gui(f"\nERRORE CRITICO DURANTE ANALISI LOTTO: {e}\n{traceback.format_exc()}")
            attendibilita_msg = f"Errore critico Lotto: {e}"
            success = False

        finally:
            self.log_message_gui("\n=== Analisi Lotto Completata ===")
            self.set_result(numeri_predetti, attendibilita_msg, game_wheels if success else [])

            if success:
                self.last_prediction = numeri_predetti
                self.last_prediction_end_date = end_date
                self.last_prediction_game_wheels = game_wheels # Salva la lista
                self.last_analysis_folder = folder_path
                self.log_message_gui(f"Previsione Lotto salvata per ruote {', '.join(game_wheels)} (basata su dati fino a {end_date}).")
            else:
                self.last_prediction = None
                self.last_prediction_end_date = None
                self.last_prediction_game_wheels = []
                self.last_analysis_folder = None
                self.log_message_gui("Analisi fallita, nessuna previsione salvata.")

            self.set_controls_state(tk.NORMAL)
            self.analysis_thread = None

    def start_check_thread(self):
        if self.check_thread and self.check_thread.is_alive():
            messagebox.showwarning("Verifica in Corso", "Una verifica è già in esecuzione.", parent=self.root)
            return
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Analisi in Corso", "Attendere il termine dell'analisi prima di verificare.", parent=self.root)
            return
        if self.last_prediction is None or not self.last_prediction_game_wheels or self.last_prediction_end_date is None or self.last_analysis_folder is None:
            messagebox.showinfo("Nessuna Previsione Valida", "Eseguire prima un'analisi Lotto con successo selezionando almeno una ruota di gioco.", parent=self.root)
            return
        try:
            num_colpi = int(self.check_colpi_var.get())
            if num_colpi <= 0:
                messagebox.showerror("Errore Colpi", "Numero colpi da verificare deve essere almeno 1.", parent=self.root)
                return
        except (ValueError, tk.TclError):
            messagebox.showerror("Errore Colpi", "Inserire un numero valido per i colpi da verificare.", parent=self.root)
            return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Previsione Lotto (Multi-Ruota) ===")
        self.log_message_gui(f"Ruote da Verificare: {', '.join(self.last_prediction_game_wheels)}")
        self.log_message_gui(f"Numeri Predetti: {self.last_prediction}")
        self.log_message_gui(f"Verifica a partire dal giorno dopo: {self.last_prediction_end_date}")
        self.log_message_gui(f"Cartella Dati: {self.last_analysis_folder}")
        self.log_message_gui("-" * 40)

        self.check_thread = threading.Thread(
            target=self.run_check_results_multi_lotto, # CHIAMA IL METODO CORRETTO
            args=(self.last_prediction_game_wheels, self.last_prediction, self.last_prediction_end_date, self.last_analysis_folder, num_colpi),
            daemon=True
        )
        self.check_thread.start()

    # --- METODO CHE GESTISCE IL LOOP SULLE RUOTE ---
    def run_check_results_multi_lotto(self, game_wheels_to_check, prediction_to_check, last_analysis_date_str, folder_path, num_colpi_to_check):
        """ Esegue la verifica su ogni ruota di gioco specificata. """
        any_error = False
        for game_wheel_name in game_wheels_to_check:
            self.log_message_gui(f"\n--- Inizio Verifica per Ruota: {game_wheel_name} ---")
            game_wheel_file_path = os.path.join(folder_path, f"{game_wheel_name}.txt")
            try:
                # Chiama la funzione originale che verifica una singola ruota
                self.run_check_results_lotto( # Passa i parametri corretti
                    game_wheel_file_path,
                    prediction_to_check,
                    last_analysis_date_str,
                    game_wheel_name,
                    num_colpi_to_check # Passa num_colpi qui!
                )
            except Exception as e:
                self.log_message_gui(f"ERRORE CRITICO durante verifica ruota {game_wheel_name}: {e}\n{traceback.format_exc()}")
                any_error = True
            # self.log_message_gui(f"--- Fine Verifica per Ruota: {game_wheel_name} ---") # Log meno verboso

        self.log_message_gui("\n=== Verifica Lotto Completata (Tutte le ruote selezionate) ===")
        if any_error:
             self.log_message_gui("ATTENZIONE: Si sono verificati errori durante la verifica di alcune ruote (vedi log sopra).")

        self.set_controls_state(tk.NORMAL)
        self.check_thread = None

    # --- METODO CHE VERIFICA UNA SINGOLA RUOTA ---
    def run_check_results_lotto(self, game_wheel_file_path, prediction_to_check, last_analysis_date_str, game_wheel_name, num_colpi_to_check):
        """Carica i dati successivi SOLO per la ruota di gioco specificata e verifica la previsione.
           CORRETTO: Converte l'array numeri in lista Python per un log pulito.
        """
        try:
            if not os.path.isfile(game_wheel_file_path):
                 self.log_message_gui(f"ERRORE ({game_wheel_name}): File non trovato: {game_wheel_file_path}")
                 return

            last_date = datetime.strptime(last_analysis_date_str, '%Y-%m-%d')
            check_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

            df_check, numeri_array_check = carica_dati_lotto(
                os.path.dirname(game_wheel_file_path),
                [game_wheel_name],
                start_date=check_start_date,
                end_date=None,
                log_callback=self.log_message_gui
            )

            if df_check is None: return
            if df_check.empty:
                self.log_message_gui(f"INFO ({game_wheel_name}): Nessuna estrazione trovata dopo il {last_analysis_date_str}.")
                return
            if numeri_array_check is None or len(numeri_array_check) == 0:
                self.log_message_gui(f"ERRORE ({game_wheel_name}): Dati trovati dopo {last_analysis_date_str}, ma estrazione numeri fallita.")
                return

            prediction_set = set(prediction_to_check)
            colpo_counter = 0
            found_hits_this_wheel = False
            num_available = len(numeri_array_check)
            num_to_run = min(num_colpi_to_check, num_available)

            if num_available < num_colpi_to_check:
                 # Usa log_message_gui se vuoi essere consistente
                 self.log_message_gui(f"INFO ({game_wheel_name}): Trovate solo {num_available} estraz. successive (richiesti {num_colpi_to_check}). Verifico le {num_available}.")

            for i in range(num_to_run): # Cicla solo per i colpi da eseguire
                colpo_counter += 1
                draw_date_dt = df_check.iloc[i]['Data']
                draw_date_str = draw_date_dt.strftime('%Y-%m-%d')
                actual_draw = numeri_array_check[i]
                actual_draw_set = set(actual_draw)
                hits = prediction_set.intersection(actual_draw_set)
                num_hits = len(hits)

                hits_log_str = ""
                if num_hits > 0:
                    found_hits_this_wheel = True
                    if num_hits == 1: hit_type = "AMBATA"
                    elif num_hits == 2: hit_type = "AMBO"
                    elif num_hits == 3: hit_type = "TERNO"
                    elif num_hits == 4: hit_type = "QUATERNA"
                    elif num_hits >= 5: hit_type = "CINQUINA"
                    else: hit_type = f"{num_hits} Punti"
                    matched_numbers_str = ', '.join(map(str, sorted(list(hits))))
                    hits_log_str = f" ---> {hit_type} ({num_hits} Punti)! Numeri: [{matched_numbers_str}] <---"

                usciti_list = sorted(actual_draw.tolist())
                self.log_message_gui(f"Colpo {colpo_counter:02d} ({draw_date_str}) - Ruota {game_wheel_name}: Usciti {usciti_list}{hits_log_str}")


            if not found_hits_this_wheel:
                 self.log_message_gui(f"\n({game_wheel_name}): Nessun esito positivo trovato nei {num_to_run} colpi verificati.")

        except ValueError as ve:
            self.log_message_gui(f"ERRORE ({game_wheel_name}) formato data durante preparazione verifica: {ve}")
        except FileNotFoundError:
            self.log_message_gui(f"ERRORE ({game_wheel_name}) File non trovato durante verifica: {game_wheel_file_path}")
        except Exception as e:
            self.log_message_gui(f"ERRORE CRITICO ({game_wheel_name}) durante verifica: {e}\n{traceback.format_exc()}")


# --- FUNZIONE DI LANCIO PER IMPORTAZIONE ---

def launch_lotto_analyzer_window(parent_window):
    """
    Crea e lancia la finestra dell'applicazione Lotto come Toplevel.

    Args:
        parent_window: La finestra principale (Tk o Toplevel) da cui viene chiamata.
                       Può essere la 'root' del tuo Empathx.py.
    """
    try:
        # Crea una nuova finestra Toplevel (secondaria) legata alla finestra genitore
        lotto_win = tk.Toplevel(parent_window)
        # Imposta il titolo e le dimensioni desiderate per la finestra del Lotto
        lotto_win.title("Analisi e Previsione Lotto")
        lotto_win.geometry("950x980") # Usa le dimensioni definite in AppLotto

        # Impedisce il ridimensionamento se desiderato (opzionale)
        # lotto_win.resizable(False, False)

        # Crea l'istanza dell'app Lotto, passando la nuova finestra Toplevel
        # come 'root' per la classe AppLotto
        app_instance = AppLotto(lotto_win) # Assumendo che la classe sia AppLotto

        # Porta la finestra del Lotto in primo piano rispetto alle altre
        lotto_win.lift()
        # Imposta il focus sulla nuova finestra
        lotto_win.focus_force()

        # Blocca l'interazione con la finestra genitore (opzionale, rende "modale")
        # Se vuoi che l'utente debba chiudere la finestra Lotto prima di tornare a Empathx,
        # decommenta le righe seguenti:
        # lotto_win.grab_set()
        # parent_window.wait_window(lotto_win) # Attende la chiusura della Toplevel

    except Exception as e:
         print(f"ERRORE imprevisto durante il lancio della finestra Lotto: {e}\n{traceback.format_exc()}")
         # Mostra un messaggio di errore all'utente nella finestra genitore
         messagebox.showerror("Errore Avvio Modulo Lotto",
                              f"Si è verificato un errore nell'avvio del modulo Lotto:\n{e}",
                              parent=parent_window) # Mostra l'errore relativo alla finestra genitore
# --- FINE FUNZIONE DI LANCIO ---


if __name__ == "__main__":
    print("Esecuzione modulo Lotto in modalità standalone...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except AttributeError:
        pass # Ignora se non applicabile

    root_standalone = tk.Tk()
    app_standalone = AppLotto(root_standalone)
    root_standalone.mainloop()