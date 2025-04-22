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
import requests  # Richiede: pip install requests

# Opzionale, ma consigliato per GUI migliori: pip install tkcalendar
try:
    from tkcalendar import DateEntry
    HAS_TKCALENDAR = True
except ImportError:
    HAS_TKCALENDAR = False

DEFAULT_SUPERENALOTTO_CHECK_COLPI = 5
# URL RAW del file su GitHub (assicurati che sia ancora valido)
DEFAULT_SUPERENALOTTO_DATA_URL = "https://raw.githubusercontent.com/illottodimax/Archivio/main/it-superenalotto-past-draws-archive.txt"

# --- Funzioni Globali (Seed, Log) ---
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
        # Usa after per assicurare l'esecuzione nel thread principale della GUI
        window.after(10, lambda: _update_log_widget(log_widget, message))

def _update_log_widget(log_widget, message):
    """Funzione helper per aggiornare il widget di log."""
    try:
        # Abilita temporaneamente il widget per l'inserimento
        current_state = log_widget.cget('state')
        log_widget.config(state=tk.NORMAL)
        log_widget.insert(tk.END, str(message) + "\n")
        log_widget.see(tk.END) # Scrolla fino alla fine
        # Ripristina lo stato originale (di solito DISABLED per impedire input utente)
        log_widget.config(state=tk.DISABLED) # Lo rimettiamo disabilitato
    except tk.TclError:
        # Fallback se la GUI non è disponibile (es. chiusura finestra)
        print(f"Log GUI TclError: {message}")
    except Exception as e:
        # Gestisci altri errori imprevisti
        print(f"Log GUI unexpected error: {e}\nMessage: {message}")

# --- Caricamento Dati (gestisce URL e File Locale) ---
def carica_dati_superenalotto(data_source, start_date=None, end_date=None, log_callback=None):
    """
    Carica i dati del SuperEnalotto da un URL (RAW GitHub) o da un file locale.
    Formato atteso:
    Header: Ignorato
    Data:   YYYY-MM-DD N1 N2 N3 N4 N5 N6 '' JJ SS  (10 campi totali separati da TAB)
    Restituisce: DataFrame pulito, array numeri principali, array jolly, array superstar, data ultimo aggiornamento.
    """
    lines = []
    is_url = data_source.startswith("http://") or data_source.startswith("https://")

    try:
        if is_url:
            if log_callback: log_callback(f"Tentativo caricamento dati SuperEnalotto da URL: {data_source}")
            try:
                response = requests.get(data_source, timeout=30)  # Timeout di 30 secondi
                response.raise_for_status()  # Solleva un errore per status HTTP >= 400
                content = response.text # requests gestisce la decodifica iniziale
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
            # Prova diversi encoding comuni
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
            file_read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f: lines = f.readlines()
                    file_read_success = True
                    if log_callback: log_callback(f"File locale letto con successo usando encoding: {enc}")
                    break # Esce dal loop se la lettura ha successo
                except UnicodeDecodeError:
                    if log_callback: log_callback(f"Info: Encoding {enc} fallito, provo il prossimo.")
                    continue
                except Exception as e_file:
                    if log_callback: log_callback(f"ERRORE durante lettura file locale con encoding {enc}: {e_file}")
                    # Non ritornare subito, prova altri encoding
                    continue
            if not file_read_success:
                 if log_callback: log_callback("ERRORE CRITICO: Impossibile leggere il file locale con gli encoding noti.")
                 return None, None, None, None, None

        # --- Parsing delle righe ---
        if log_callback: log_callback(f"Inizio parsing di {len(lines)} righe...")
        if not lines or len(lines) < 2: # Assumendo almeno 1 riga header + 1 riga dati
            if log_callback: log_callback("ERRORE: Dati vuoti o solo intestazione.")
            return None, None, None, None, None

        data_lines = lines[1:] # Salta l'header (prima riga)
        processed_data = []
        malformed_lines_count = 0
        processed_lines_count = 0
        min_expected_fields = 10 # YYYY-MM-DD N1 N2 N3 N4 N5 N6 '' JJ SS

        for i, line in enumerate(data_lines):
            # Rimuovi spazi bianchi inizio/fine e splitta per TAB
            values = line.strip().split('\t')

            if len(values) >= min_expected_fields:
                try:
                    # Estrai i campi richiesti (ignora campo vuoto indice 7)
                    date_val = values[0].strip()
                    num_vals_str = [v.strip() for v in values[1:7]] # Num1-Num6
                    jolly_val_str = values[8].strip() # Jolly
                    superstar_val_str = values[9].strip() # SuperStar

                    # Validazione minimale: data sembra data, numeri sono numeri
                    datetime.strptime(date_val, '%Y-%m-%d') # Lancia errore se formato non valido
                    if not all(n.isdigit() for n in num_vals_str): raise ValueError("Num1-6 non numerici")
                    if not jolly_val_str.isdigit(): raise ValueError("Jolly non numerico")
                    if not superstar_val_str.isdigit(): raise ValueError("SuperStar non numerico")

                    # Conversione a tipi corretti (verrà ricontrollata da pandas)
                    num_vals = [int(n) for n in num_vals_str]
                    jolly_val = int(jolly_val_str)
                    superstar_val = int(superstar_val_str)

                    # Ulteriore controllo range (opzionale qui, pandas lo farà comunque)
                    if not all(1 <= n <= 90 for n in num_vals + [jolly_val, superstar_val]):
                        raise ValueError("Numeri fuori range (1-90)")

                    # Aggiungi la riga processata (mantenendo stringhe per flessibilità con pandas)
                    clean_row = [date_val] + num_vals_str + [jolly_val_str, superstar_val_str]
                    processed_data.append(clean_row)
                    processed_lines_count += 1

                except (IndexError, ValueError, TypeError) as e_parse:
                    malformed_lines_count += 1
                    # Logga solo i primi errori per non inondare il log
                    if malformed_lines_count <= 5 and log_callback:
                        log_callback(f"ATTENZIONE: Riga {i+2} scartata (Errore parsing: {e_parse}). Valori: '{line.strip()}'")
                    elif malformed_lines_count == 6 and log_callback:
                        log_callback("ATTENZIONE: Ulteriori errori di parsing non verranno loggati singolarmente.")
            else:
                malformed_lines_count += 1
                if malformed_lines_count <= 5 and log_callback:
                    log_callback(f"ATTENZIONE: Riga {i+2} scartata (Campi < {min_expected_fields}, trovati {len(values)}). Valori: '{line.strip()}'")
                elif malformed_lines_count == 6 and log_callback:
                        log_callback("ATTENZIONE: Ulteriori errori di campi insufficienti non verranno loggati.")

        if malformed_lines_count > 0 and log_callback:
            log_callback(f"ATTENZIONE: {malformed_lines_count} righe totali scartate durante il parsing.")

        if not processed_data:
            if log_callback: log_callback("ERRORE: Nessuna riga dati valida trovata dopo il parsing.")
            return None, None, None, None, None
        else:
            if log_callback: log_callback(f"Parsing completato: {processed_lines_count} righe elaborate con successo.")

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
            # Converti Data
            df['Data'] = pd.to_datetime(df['Data'], format='%Y-%m-%d', errors='coerce')
            rows_before_date_na = len(df)
            df = df.dropna(subset=['Data'])
            rows_after_date_na = len(df)
            if rows_before_date_na > rows_after_date_na and log_callback:
                log_callback(f"Rimosse {rows_before_date_na - rows_after_date_na} righe con date non valide.")
            if df.empty:
                if log_callback: log_callback("ERRORE: Nessun dato valido rimasto dopo pulizia date.")
                return df.copy(), None, None, None, None # Restituisci df vuoto
            df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)
            if log_callback: log_callback(f"DataFrame ordinato per data. Righe valide: {len(df)}")

            # Converti Numeri (da Num1 a SuperStar)
            num_cols = [f'Num{i+1}' for i in range(6)] + ['Jolly', 'SuperStar']
            rows_before_num_conv = len(df)
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Converte in float, NaN se errore
                df[col] = df[col].astype('Int64', errors='ignore') # Prova a convertire in Int64 (che supporta NA)

            # Rimuovi righe dove i numeri principali sono NA
            df_cleaned = df.dropna(subset=num_cols[:6]) # Solo Num1-6 sono obbligatori per il modello base
            rows_after_num_conv = len(df_cleaned)
            if rows_before_num_conv > rows_after_num_conv and log_callback:
                 log_callback(f"Rimosse {rows_before_num_conv - rows_after_num_conv} righe con Num1-6 non validi.")

            if df_cleaned.empty:
                 if log_callback: log_callback("ERRORE: Nessun dato valido rimasto dopo pulizia numeri principali.")
                 return df_cleaned.copy(), None, None, None, None

            # Controllo range 1-90 (opzionale, ma buona pratica)
            rows_before_range_check = len(df_cleaned)
            for col in num_cols:
                # Applica il check solo se la colonna non è tutta NA (es. Jolly/SS se non usati)
                if df_cleaned[col].notna().any():
                    df_cleaned = df_cleaned[df_cleaned[col].isna() | ((df_cleaned[col] >= 1) & (df_cleaned[col] <= 90))]
            rows_after_range_check = len(df_cleaned)
            if rows_before_range_check > rows_after_range_check and log_callback:
                 log_callback(f"Rimosse {rows_before_range_check - rows_after_range_check} righe con numeri fuori range (1-90).")

            if df_cleaned.empty:
                 if log_callback: log_callback("ERRORE: Nessun dato valido rimasto dopo controllo range 1-90.")
                 return df_cleaned.copy(), None, None, None, None

            if log_callback: log_callback(f"Pulizia tipi e range completata. Righe finali per l'analisi: {len(df_cleaned)}")

        except Exception as e_clean:
            if log_callback: log_callback(f"ERRORE durante pulizia tipi/date/numeri: {e_clean}\n{traceback.format_exc()}")
            return None, None, None, None, None

        # --- Filtraggio Date Utente ---
        df_filtered = df_cleaned.copy() # Lavora sulla copia pulita
        rows_before_filter = len(df_filtered)
        if log_callback: log_callback(f"Applicazione filtro date utente: Start={start_date}, End={end_date}")
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df_filtered = df_filtered[df_filtered['Data'] >= start_dt]
            except Exception as e_start:
                # <<<--- CORREZIONE QUI ---<<<
                if log_callback:
                    log_callback(f"Errore filtro data inizio (verrà ignorato): {e_start}")
        if end_date:
             try:
                 end_dt = pd.to_datetime(end_date)
                 df_filtered = df_filtered[df_filtered['Data'] <= end_dt]
             except Exception as e_end:
                 # <<<--- CORREZIONE QUI ---<<<
                 if log_callback:
                     log_callback(f"Errore filtro data fine (verrà ignorato): {e_end}")

        rows_after_filter = len(df_filtered)
        if log_callback:
            if rows_before_filter == rows_after_filter: log_callback("Filtro date non ha rimosso righe (o date non specificate/invalide).")
            else: log_callback(f"Righe dopo filtro date utente ({start_date} - {end_date}): {rows_after_filter} (rimosse {rows_before_filter - rows_after_filter})")

        if df_filtered.empty:
            if log_callback: log_callback("INFO: Nessuna riga rimasta nel DataFrame dopo il filtro per data specificato.")
            # Restituiamo comunque il DF (vuoto) e None per gli array
            return df_filtered.copy(), None, None, None, None

        # --- Estrazione Array Finali dal DataFrame Filtrato ---
        numeri_principali_cols = [f'Num{i+1}' for i in range(6)]
        numeri_principali_array, numeri_jolly, numeri_superstar = None, None, None

        try:
            # Estrai Num1-6 (sappiamo già che non sono NA qui)
            numeri_principali_array = df_filtered[numeri_principali_cols].values.astype(int)
            if log_callback: log_callback(f"Estratto array numeri principali. Shape: {numeri_principali_array.shape}")

            # Estrai Jolly e SuperStar solo se le colonne esistono e hanno valori non NA
            if 'Jolly' in df_filtered.columns and df_filtered['Jolly'].notna().any():
                numeri_jolly = df_filtered.dropna(subset=['Jolly'])['Jolly'].values.astype(int)
                if log_callback: log_callback(f"Estratto array Jolly. Shape: {numeri_jolly.shape}")
            elif log_callback: log_callback("Colonna Jolly non presente o vuota nel set filtrato.")

            if 'SuperStar' in df_filtered.columns and df_filtered['SuperStar'].notna().any():
                numeri_superstar = df_filtered.dropna(subset=['SuperStar'])['SuperStar'].values.astype(int)
                if log_callback: log_callback(f"Estratto array SuperStar. Shape: {numeri_superstar.shape}")
            elif log_callback: log_callback("Colonna SuperStar non presente o vuota nel set filtrato.")

        except Exception as e_extract:
            if log_callback: log_callback(f"ERRORE durante estrazione array finali: {e_extract}")
            # In caso di errore qui, restituiamo ciò che abbiamo (df filtrato) e None per gli array
            return df_filtered.copy(), None, None, None, None

        # Ottieni la data dell'ultima estrazione NEL SET DI DATI FILTRATO
        last_update_date = df_filtered['Data'].max() if not df_filtered.empty else None

        return df_filtered.copy(), numeri_principali_array, numeri_jolly, numeri_superstar, last_update_date

    except Exception as e_main:
        # Errore generale non catturato prima
        if log_callback: log_callback(f"Errore GRAVE non gestito in carica_dati_superenalotto: {e_main}\n{traceback.format_exc()}")
        return None, None, None, None, None

# --- Preparazione Sequenze per Modello ---
def prepara_sequenze_per_modello_superenalotto(numeri_principali_array, sequence_length=5, log_callback=None):
    """
    Prepara le sequenze di input (X) e target (y) per il modello SuperEnalotto.
    X: sequenze di 'sequence_length' estrazioni (6 numeri appiattiti).
    y: la successiva estrazione (6 numeri) in formato multi-hot (vettore di 90 con 1 per i numeri estratti).
    """
    if log_callback: log_callback(f"Avvio preparazione sequenze SuperEnalotto (SeqLen={sequence_length})...")
    if numeri_principali_array is None or len(numeri_principali_array) == 0:
        if log_callback: log_callback("ERRORE (prep_seq): Array numeri principali vuoto o None.")
        return None, None
    if numeri_principali_array.ndim != 2 or numeri_principali_array.shape[1] != 6:
        if log_callback: log_callback(f"ERRORE (prep_seq): Array numeri principali non ha 6 colonne (shape: {numeri_principali_array.shape}).")
        return None, None

    num_estrazioni = len(numeri_principali_array)
    if log_callback: log_callback(f"Numero estrazioni disponibili per sequenze: {num_estrazioni}.")

    # Devono esserci abbastanza estrazioni per creare almeno una sequenza input + target
    if num_estrazioni <= sequence_length:
        msg = f"ERRORE: Estrazioni insufficienti ({num_estrazioni}) per creare sequenze di lunghezza {sequence_length}. Servono almeno {sequence_length + 1} estrazioni."
        if log_callback: log_callback(msg)
        return None, None

    X, y = [], []
    valid_sequences_count = 0
    invalid_target_count = 0

    # Itera fino a num_estrazioni - sequence_length per avere sempre un target disponibile
    for i in range(num_estrazioni - sequence_length):
        # Estrai la sequenza di input (gli ultimi 'sequence_length' elementi)
        input_seq = numeri_principali_array[i : i + sequence_length]
        # Estrai l'estrazione target (l'elemento successivo alla sequenza)
        target_extraction = numeri_principali_array[i + sequence_length]

        # Validazione del target: assicurati che siano 6 numeri validi (1-90)
        if len(target_extraction) == 6 and np.all((target_extraction >= 1) & (target_extraction <= 90)):
            # Crea il vettore target multi-hot (lunghezza 90)
            target_vector = np.zeros(90, dtype=np.int8) # Usiamo int8 per risparmiare memoria
            # Imposta a 1 le posizioni corrispondenti ai numeri estratti (indice = numero - 1)
            target_vector[target_extraction - 1] = 1

            # Appiattisci la sequenza di input e aggiungila a X
            X.append(input_seq.flatten())
            # Aggiungi il vettore target a y
            y.append(target_vector)
            valid_sequences_count += 1
        else:
            invalid_target_count += 1
            # Logga solo i primi errori per evitare spam
            if invalid_target_count <= 5 and log_callback:
                log_callback(f"ATTENZIONE (prep_seq): Scartata sequenza con indice iniziale {i}. Target non valido: {target_extraction}")
            elif invalid_target_count == 6 and log_callback:
                log_callback("ATTENZIONE (prep_seq): Ulteriori errori target non verranno loggati.")

    if invalid_target_count > 0 and log_callback:
        log_callback(f"ATTENZIONE: Scartate {invalid_target_count} sequenze totali a causa di target non validi.")

    if not X: # Se la lista X è vuota dopo il ciclo
        if log_callback: log_callback("ERRORE: Nessuna sequenza valida creata. Controllare i dati di input.")
        return None, None

    if log_callback: log_callback(f"Create {valid_sequences_count} sequenze Input/Target valide.")

    # Converti le liste X e y in array NumPy
    try:
        X_np = np.array(X, dtype=np.int32) # Usiamo int32 per i numeri da 1 a 90
        y_np = np.array(y, dtype=np.int8) # y è già 0/1, int8 va bene
        if log_callback: log_callback(f"Array NumPy creati. Shape X: {X_np.shape}, Shape y: {y_np.shape}")
        return X_np, y_np
    except Exception as e_np_conv:
        if log_callback: log_callback(f"ERRORE durante conversione finale a NumPy array: {e_np_conv}")
        return None, None

# --- Costruzione Modello Keras ---
def build_model_superenalotto(input_shape, hidden_layers=[512, 256, 128], loss_function='binary_crossentropy', optimizer='adam', dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0, log_callback=None):
    """
    Costruisce il modello Keras (rete neurale densa) per SuperEnalotto.
    Input shape è (sequence_length * 6). Output shape è (90,).
    """
    if log_callback: log_callback(f"Costruzione modello SuperEnalotto: InputShape={input_shape}, Layers={hidden_layers}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")

    # Validazione input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) != 1 or not isinstance(input_shape[0], int) or input_shape[0] <= 0:
         msg = f"ERRORE: input_shape non valido: {input_shape}. Deve essere una tupla con un intero positivo (es. (30,))."
         if log_callback: log_callback(msg)
         raise ValueError(msg) # Solleva errore per interrompere

    # Validazione hidden_layers
    if not isinstance(hidden_layers, list) or not all(isinstance(units, int) and units > 0 for units in hidden_layers):
        msg = f"ERRORE: hidden_layers non validi: {hidden_layers}. Deve essere una lista di interi positivi (es. [512, 256])."
        if log_callback: log_callback(msg)
        raise ValueError(msg) # Solleva errore

    # Validazione dropout
    if not isinstance(dropout_rate, (float, int)) or not (0.0 <= dropout_rate < 1.0): # Dropout 1.0 non ha senso
        msg = f"ERRORE: dropout_rate non valido: {dropout_rate}. Deve essere un float tra 0.0 (incluso) e 1.0 (escluso)."
        if log_callback: log_callback(msg)
        raise ValueError(msg) # Solleva errore

    # Validazione regolarizzazione
    if not isinstance(l1_reg, (float, int)) or l1_reg < 0.0:
        msg = f"ERRORE: l1_reg non valido: {l1_reg}. Deve essere un float >= 0.0."
        if log_callback: log_callback(msg)
        raise ValueError(msg) # Solleva errore
    if not isinstance(l2_reg, (float, int)) or l2_reg < 0.0:
        msg = f"ERRORE: l2_reg non valido: {l2_reg}. Deve essere un float >= 0.0."
        if log_callback: log_callback(msg)
        raise ValueError(msg) # Solleva errore

    model = tf.keras.Sequential(name="Modello_SuperEnalotto_DNN")

    # Input Layer (implicito specificando input_shape nel primo Dense o esplicito con Input)
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input_Layer"))

    # Kernel Regularizer (L1/L2)
    kernel_regularizer = None
    if l1_reg > 0 or l2_reg > 0:
        kernel_regularizer = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        if log_callback: log_callback(f"Applicata regolarizzazione L1={l1_reg:.4f}, L2={l2_reg:.4f}")

    # Hidden Layers
    if not hidden_layers:
        if log_callback: log_callback("ATTENZIONE: Nessun hidden layer specificato. Il modello avrà solo Input -> Output.")
    else:
        for i, units in enumerate(hidden_layers):
            layer_num = i + 1
            # Dense Layer
            model.add(tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=kernel_regularizer,
                name=f"Dense_{layer_num}_{units}"
            ))
            # Batch Normalization (spesso utile dopo Dense e prima di Dropout)
            model.add(tf.keras.layers.BatchNormalization(name=f"BatchNorm_{layer_num}"))
            # Dropout (se specificato)
            if dropout_rate > 0:
                model.add(tf.keras.layers.Dropout(dropout_rate, name=f"Dropout_{layer_num}_{dropout_rate:.2f}"))

    # Output Layer
    # 90 neuroni (uno per ogni possibile numero da 1 a 90)
    # Attivazione Sigmoid: produce output tra 0 e 1 per ogni neurone, interpretabile
    #                    come probabilità (per problemi multi-label come questo).
    #                    Adatta per binary_crossentropy loss.
    model.add(tf.keras.layers.Dense(90, activation='sigmoid', name="Output_Layer_90_Sigmoid"))

    # Compilazione del modello
    try:
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy']) # Accuracy qui misura la frazione di etichette predette correttamente (0 o 1)
                                            # Potrebbe non essere la metrica più intuitiva per il lotto.
                                            # Potresti considerare metriche custom o focalizzarti sulla loss.
        if log_callback: log_callback(f"Modello SuperEnalotto compilato con successo (Optimizer: {optimizer}, Loss: {loss_function}).")
    except ValueError as e_compile:
        # Errore comune se optimizer o loss non sono riconosciuti da Keras
         if log_callback: log_callback(f"ERRORE durante la compilazione del modello: {e_compile}. Verificare nomi optimizer/loss.")
         return None # Restituisce None se la compilazione fallisce
    except Exception as e_generic_compile:
        if log_callback: log_callback(f"ERRORE generico durante la compilazione del modello: {e_generic_compile}")
        return None

    # Stampa riepilogo modello nel log
    if log_callback:
        try:
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            summary_str = "\n".join(stringlist)
            log_callback("Riepilogo Modello SuperEnalotto:\n" + summary_str)
        except Exception as e_summary:
            log_callback(f"ATTENZIONE: Impossibile generare riepilogo modello: {e_summary}")

    return model

# --- Callback per Logging Epoche nella GUI ---
class LogCallback(tf.keras.callbacks.Callback):
    """Callback Keras per inviare i log delle epoche alla funzione di log della GUI."""
    def __init__(self, log_callback_func):
        super().__init__()
        self.log_callback_func = log_callback_func
        self._is_running = True # Flag per fermare il logging se necessario

    def stop_logging(self):
        """Imposta il flag per fermare l'invio di log."""
        self._is_running = False

    def on_epoch_end(self, epoch, logs=None):
        """Chiamato alla fine di ogni epoca."""
        # Se il logging è stato fermato o non c'è funzione di callback, non fare nulla
        if not self._is_running or not self.log_callback_func:
            return

        logs = logs or {} # Assicura che logs sia un dizionario
        # Formatta il messaggio di log per l'epoca
        msg = f"Epoca {epoch+1:03d} - "
        log_items = []
        # Formatta ogni metrica nel dizionario logs
        for k, v in logs.items():
            # Sostituisci underscore e 'val ' per rendere più leggibile
            metric_name = k.replace('_', ' ').replace('val ', 'V_')
            log_items.append(f"{metric_name}: {v:.5f}") # Aumentata precisione a 5 decimali
        msg += ", ".join(log_items)

        # Invia il messaggio alla funzione di log della GUI
        self.log_callback_func(msg)

# --- Generazione Previsione ---
def genera_previsione_superenalotto(model, X_input, num_predictions=6, log_callback=None):
    """
    Genera la previsione dei numeri del SuperEnalotto usando il modello addestrato.
    Prende l'ultima sequenza (già normalizzata), fa la previsione e restituisce
    una LISTA DI DIZIONARI, ognuno con 'number' e 'probability',
    per i 'num_predictions' numeri con la probabilità più alta stimata dal modello.
    La lista restituita NON è ordinata per probabilità, ma gli elementi contengono le probabilità.
    """
    if log_callback: log_callback(f"Avvio generazione previsione SuperEnalotto per {num_predictions} numeri...")
    if model is None:
        if log_callback: log_callback("ERRORE (genera_prev): Modello non fornito o non valido.")
        return None
    if X_input is None or X_input.size == 0:
        if log_callback: log_callback("ERRORE (genera_prev): Input (X_input) vuoto o non valido.")
        return None
    # Assicurati che l'input sia 2D (anche se è una sola sequenza)
    if X_input.ndim == 1:
        X_input_reshaped = X_input.reshape(1, -1) # Reshape (features,) in (1, features)
    elif X_input.ndim == 2 and X_input.shape[0] == 1:
        X_input_reshaped = X_input # Già nel formato corretto (1, features)
    elif X_input.ndim >= 2 and X_input.shape[0] > 1:
         if log_callback: log_callback(f"ATTENZIONE (genera_prev): Ricevuto input con shape {X_input.shape}. Verrà usata solo la prima riga per la previsione.")
         X_input_reshaped = X_input[0].reshape(1, -1) # Prendi solo la prima riga/sequenza
    else:
        if log_callback: log_callback(f"ERRORE (genera_prev): Shape input non gestita: {X_input.shape}. Atteso 1D o 2D.")
        return None

    if log_callback: log_callback(f"Input per predict preparato con shape: {X_input_reshaped.shape}")

    # Validazione coerenza shape input con modello
    try:
        expected_input_features = model.input_shape[-1]
        if expected_input_features is not None and X_input_reshaped.shape[1] != expected_input_features:
            msg = f"ERRORE Shape Input: Dimensione input ({X_input_reshaped.shape[1]}) != Dimensione attesa dal modello ({expected_input_features})."
            if log_callback: log_callback(msg)
            return None
    except Exception as e_shape_check:
         # Non critico se non riusciamo a verificarlo, ma logghiamo l'attenzione
         if log_callback: log_callback(f"ATTENZIONE (genera_prev): Impossibile verificare shape input modello ({e_shape_check}). Procedo comunque.")

    # Validazione num_predictions
    if not isinstance(num_predictions, int) or not (1 <= num_predictions <= 90):
        msg = f"ERRORE (genera_prev): num_predictions={num_predictions} non valido. Deve essere un intero tra 1 e 90."
        if log_callback: log_callback(msg)
        return None

    try:
        # Esegui la previsione
        # verbose=0 per evitare output di Keras nella console/log
        pred_probabilities = model.predict(X_input_reshaped, verbose=0)

        # Verifica l'output della previsione
        if pred_probabilities is None or pred_probabilities.size == 0:
            if log_callback: log_callback("ERRORE (genera_prev): model.predict() ha restituito None o un array vuoto.")
            return None
        # L'output dovrebbe essere (1, 90) perché abbiamo fornito 1 campione di input
        if pred_probabilities.ndim != 2 or pred_probabilities.shape[0] != 1 or pred_probabilities.shape[1] != 90:
             msg = f"ERRORE (genera_prev): Output shape da predict inatteso: {pred_probabilities.shape}. Atteso (1, 90)."
             if log_callback: log_callback(msg)
             return None

        # Estrai il vettore di 90 probabilità
        probs_vector = pred_probabilities[0] # Shape (90,)

        # Trova gli INDICI dei numeri con le probabilità più alte
        # np.argsort restituisce gli indici che ordinerebbero l'array (dal più basso al più alto)
        sorted_indices = np.argsort(probs_vector)

        # Prendi gli ultimi 'num_predictions' indici (quelli delle probabilità più alte)
        top_n_indices = sorted_indices[-num_predictions:]

        # Crea la lista di risultati (dizionari)
        predicted_results = []
        for index in top_n_indices:
            number = int(index + 1) # Il numero è indice + 1
            probability = float(probs_vector[index]) # La probabilità associata a quell'indice
            predicted_results.append({"number": number, "probability": probability})

        # Log dettagliato delle previsioni (opzionale: ordina per probabilità per il log)
        if log_callback:
            results_sorted_by_prob_desc = sorted(predicted_results, key=lambda x: x['probability'], reverse=True)
            log_probs = [f"{res['number']:02d} (p={res['probability']:.5f})" for res in results_sorted_by_prob_desc]
            log_callback(f"Top {num_predictions} numeri predetti (ord. per probabilità decrescente):\n  " + "\n  ".join(log_probs))

        # Restituisci la lista di dizionari (non necessariamente ordinata)
        # L'ordinamento per numero avverrà nella GUI se necessario
        return predicted_results

    except Exception as e_predict:
        if log_callback: log_callback(f"ERRORE CRITICO durante la generazione della previsione: {e_predict}\n{traceback.format_exc()}")
        return None

# --- Funzione Principale di Analisi ---
def analisi_superenalotto(file_path, start_date, end_date, sequence_length=5,
                          loss_function='binary_crossentropy', optimizer='adam',
                          dropout_rate=0.3, l1_reg=0.0, l2_reg=0.0,
                          hidden_layers_config=[512, 256, 128],
                          max_epochs=100, batch_size=32, patience=15, min_delta=0.0001,
                          num_predictions=6,
                          log_callback=None):
    """
    Analizza i dati del SuperEnalotto, addestra il modello e genera previsioni.
    'file_path' può essere un URL (raw) o un percorso di file locale.
    Restituisce:
        - Una lista di dizionari [{'number': n, 'probability': p}, ...] per i numeri predetti, o None in caso di errore.
        - Un messaggio sull'attendibilità/stato.
        - La data dell'ultimo aggiornamento dei dati usati.
    """
    # --- Logging Iniziale ---
    if log_callback:
        source_type = "URL" if file_path.startswith("http") else "File"
        source_name = os.path.basename(file_path) if source_type == "File" else file_path
        log_callback(f"\n=== Avvio Analisi SuperEnalotto (vCorretta) ===")
        log_callback(f"Sorgente: {source_type} ({source_name})")
        log_callback(f"Periodo Dati: {start_date} -> {end_date}")
        log_callback(f"Parametri Seq/Pred: SeqLen={sequence_length}, NumPred={num_predictions}")
        log_callback(f"Modello: HiddenL={hidden_layers_config}, Loss={loss_function}, Opt={optimizer}, Drop={dropout_rate}, L1={l1_reg}, L2={l2_reg}")
        log_callback(f"Training: MaxEpochs={max_epochs}, Batch={batch_size}, Patience={patience}, MinDelta={min_delta}")
        log_callback("-" * 40)

    # 1. Carica e Preprocessa i Dati
    df, numeri_principali_array, _, _, last_update_date = carica_dati_superenalotto(
        file_path, start_date, end_date, log_callback=log_callback
    )

    # Verifica caricamento dati
    if df is None:
        return None, "Errore critico durante il caricamento dei dati.", None
    if df.empty:
         msg = "Nessun dato trovato per il periodo specificato o dopo la pulizia iniziale."
         if log_callback: log_callback(f"INFO: {msg}")
         # Restituiamo la data dell'ultimo aggiornamento anche se il df filtrato è vuoto,
         # se la funzione carica_dati l'ha determinata prima del filtro.
         return None, msg, last_update_date
    if numeri_principali_array is None or len(numeri_principali_array) == 0:
        msg = "Dati caricati, ma nessun array valido di numeri principali estratto."
        if log_callback: log_callback(f"ERRORE: {msg}")
        return None, msg, last_update_date
    if len(numeri_principali_array) < sequence_length + 1:
        msg = f"ERRORE: Dati numerici insufficienti ({len(numeri_principali_array)}) per creare sequenze di lunghezza {sequence_length}. Servono almeno {sequence_length + 1} estrazioni."
        if log_callback: log_callback(msg)
        return None, msg, last_update_date

    if log_callback: log_callback(f"Dati caricati e validati. Numero estrazioni per sequenze: {len(numeri_principali_array)}. Ultimo aggiornamento: {last_update_date.strftime('%Y-%m-%d') if last_update_date else 'N/D'}")

    # 2. Prepara Sequenze
    X, y = None, None
    try:
        X, y = prepara_sequenze_per_modello_superenalotto(numeri_principali_array, sequence_length, log_callback=log_callback)
        if X is None or y is None or len(X) == 0:
            return None, "Creazione sequenze Input/Target fallita o nessuna sequenza valida generata.", last_update_date
    except Exception as e_prep:
         if log_callback: log_callback(f"ERRORE CRITICO durante preparazione sequenze: {e_prep}\n{traceback.format_exc()}")
         return None, f"Errore preparazione sequenze: {e_prep}", last_update_date

    # 3. Normalizza Input (X)
    try:
        X_scaled = X.astype(np.float32) / 90.0
        if log_callback: log_callback(f"Input X normalizzato (diviso per 90). Shape: {X_scaled.shape}, Range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    except Exception as e_scale:
        if log_callback: log_callback(f"ERRORE durante normalizzazione input X: {e_scale}")
        return None, "Errore normalizzazione dati input", last_update_date

    # 4. Split Train/Validation (opzionale ma raccomandato)
    X_train, X_val, y_train, y_val = None, None, None, None
    split_ratio = 0.8 # 80% train, 20% validation
    min_samples_for_split = 10 # Numero minimo di campioni per tentare uno split sensato

    if len(X_scaled) >= min_samples_for_split:
        try:
            split_idx = int(len(X_scaled) * split_ratio)
            split_idx = max(1, min(split_idx, len(X_scaled) - 1))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            if log_callback: log_callback(f"Dati divisi: {len(X_train)} train ({split_ratio*100:.0f}%), {len(X_val)} validation ({(1-split_ratio)*100:.0f}%).")

            if len(X_train)==0 or len(y_train)==0 or len(X_val)==0 or len(y_val)==0:
                if log_callback: log_callback("ATTENZIONE: Split ha prodotto set vuoti. Fallback a training su tutti i dati.")
                X_train, y_train = X_scaled, y
                X_val, y_val = None, None
                validation_data_fit = None
                monitor_metric = 'loss'
            else:
                 validation_data_fit = (X_val, y_val)
                 monitor_metric = 'val_loss'

        except Exception as e_split:
             if log_callback: log_callback(f"ERRORE durante lo split train/validation: {e_split}. Fallback a training su tutti i dati.")
             X_train, y_train = X_scaled, y
             X_val, y_val = None, None
             validation_data_fit = None
             monitor_metric = 'loss'
    else:
        if log_callback: log_callback(f"INFO: Campioni insufficienti ({len(X_scaled)} < {min_samples_for_split}) per split. Training su tutti i dati.")
        X_train, y_train = X_scaled, y
        X_val, y_val = None, None
        validation_data_fit = None
        monitor_metric = 'loss'

    # 5. Costruisci e Addestra il Modello
    model, history, gui_log_callback_instance = None, None, None
    final_loss_for_attendibilita = float('inf')
    best_epoch_number = -1

    try:
        tf.keras.backend.clear_session()
        set_seed()

        if X_train is None or X_train.size == 0 or X_train.ndim != 2 or X_train.shape[1] == 0:
            if log_callback: log_callback(f"ERRORE CRITICO: Dati di training (X_train) non validi prima della costruzione del modello. Shape: {X_train.shape if X_train is not None else 'None'}")
            return None, "Errore: dati di training invalidi.", last_update_date

        input_shape_model = (X_train.shape[1],)
        model = build_model_superenalotto(input_shape_model, hidden_layers_config, loss_function, optimizer, dropout_rate, l1_reg, l2_reg, log_callback)
        if model is None:
            return None, "Costruzione modello SuperEnalotto fallita.", last_update_date

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric, patience=patience, min_delta=min_delta,
            restore_best_weights=True, verbose=1
        )
        gui_log_callback_instance = LogCallback(log_callback)
        callbacks_list = [early_stopping, gui_log_callback_instance]

        if log_callback: log_callback(f"\n--- Inizio Addestramento Modello (Monitor: '{monitor_metric}', Patience: {patience}) ---")

        history = model.fit(
            X_train, y_train, validation_data=validation_data_fit,
            epochs=max_epochs, batch_size=batch_size,
            callbacks=callbacks_list, verbose=0
        )

        if history and history.history:
            epochs_run = len(history.history.get('loss', []))
            if log_callback: log_callback(f"--- Addestramento Terminato (Epoche eseguite: {epochs_run}) ---")

            train_loss_hist = history.history.get('loss', [])
            val_loss_hist = history.history.get('val_loss', [])

            if val_loss_hist:
                best_epoch_idx = np.argmin(val_loss_hist)
                final_loss_for_attendibilita = val_loss_hist[best_epoch_idx]
                best_epoch_number = best_epoch_idx + 1
                train_loss_at_best_val = train_loss_hist[best_epoch_idx] if best_epoch_idx < len(train_loss_hist) else float('inf')
                if log_callback:
                    log_callback(f"Miglior epoca (basata su val_loss): {best_epoch_number}")
                    log_callback(f"  - Val Loss Minima: {final_loss_for_attendibilita:.5f}")
                    log_callback(f"  - Train Loss (in quell'epoca): {train_loss_at_best_val:.5f}")
            elif train_loss_hist:
                 best_epoch_idx = np.argmin(train_loss_hist)
                 final_loss_for_attendibilita = train_loss_hist[best_epoch_idx]
                 best_epoch_number = best_epoch_idx + 1
                 if log_callback:
                     log_callback(f"Miglior epoca (basata su train_loss): {best_epoch_number}")
                     log_callback(f"  - Train Loss Minima: {final_loss_for_attendibilita:.5f} (Nessuna validazione)")
            else:
                 if log_callback: log_callback("ATTENZIONE: Nessuna history di loss trovata dopo l'addestramento.")
                 final_loss_for_attendibilita = float('inf')
                 best_epoch_number = epochs_run
        else:
             if log_callback: log_callback("ATTENZIONE: Oggetto History non valido o vuoto restituito da model.fit(). Impossibile determinare la migliore epoca o loss.")
             final_loss_for_attendibilita = float('inf')
             best_epoch_number = max_epochs

    except tf.errors.ResourceExhaustedError as e_mem:
         msg = f"ERRORE OOM: Memoria insufficiente (GPU?). Riduci batch size, lunghezza sequenza o complessità modello. Dettagli: {e_mem}"
         if log_callback: log_callback(msg); log_callback(traceback.format_exc())
         if gui_log_callback_instance: gui_log_callback_instance.stop_logging()
         return None, msg, last_update_date
    except Exception as e_train:
        msg = f"ERRORE CRITICO durante addestramento: {e_train}"
        if log_callback: log_callback(msg); log_callback(traceback.format_exc())
        if gui_log_callback_instance: gui_log_callback_instance.stop_logging()
        return None, msg, last_update_date
    finally:
         if gui_log_callback_instance:
             gui_log_callback_instance.stop_logging()

    # 6. Prepara Input e Genera Previsione Finale
    previsione_completa = None
    attendibilita_msg = "Attendibilità Non Determinata"
    try:
        if log_callback: log_callback("\n--- Preparazione Input per Previsione Finale ---")
        if numeri_principali_array is None or len(numeri_principali_array) < sequence_length:
            needed = sequence_length
            available = len(numeri_principali_array) if numeri_principali_array is not None else 0
            msg = f"ERRORE: Dati originali insufficienti ({available}) per creare l'input finale per la previsione (richieste {needed} estrazioni)."
            if log_callback: log_callback(msg)
            return None, msg, last_update_date

        input_pred_raw = numeri_principali_array[-sequence_length:]
        input_pred_flat = input_pred_raw.flatten()
        input_pred_scaled = input_pred_flat.astype(np.float32) / 90.0
        if log_callback:
            log_callback(f"Ultima sequenza raw (shape {input_pred_raw.shape}): {input_pred_raw.tolist()}")
            log_callback(f"Input finale per previsione (shape {input_pred_scaled.shape}) pronto.")

        previsione_completa = genera_previsione_superenalotto(
            model, input_pred_scaled, num_predictions, log_callback=log_callback
        )

        if previsione_completa is None:
            return None, "Generazione previsione finale fallita.", last_update_date

        attendibilita_livello = "Bassa"
        loss_threshold_alta = 0.10
        loss_threshold_media = 0.25

        if final_loss_for_attendibilita == float('inf'):
             attendibilita_livello = "Indeterminata (loss non disponibile)"
        elif monitor_metric == 'val_loss':
             if final_loss_for_attendibilita < loss_threshold_alta: attendibilita_livello = "Alta (Val Loss Bassa)"
             elif final_loss_for_attendibilita < loss_threshold_media: attendibilita_livello = "Media (Val Loss Moderata)"
             else: attendibilita_livello = "Bassa (Val Loss Alta)"
        else: # monitor_metric == 'loss'
             if final_loss_for_attendibilita < loss_threshold_alta: attendibilita_livello = "Potenzialmente Alta (Train Loss Bassa, no val.)"
             elif final_loss_for_attendibilita < loss_threshold_media: attendibilita_livello = "Potenzialmente Media (Train Loss Moderata, no val.)"
             else: attendibilita_livello = "Incertezza Alta (Train Loss Alta, no val.)"

        attendibilita_msg = f"Attendibilità Stimata: {attendibilita_livello} (Loss Finale {monitor_metric}: {final_loss_for_attendibilita:.5f} @ Epoca {best_epoch_number})"
        if log_callback: log_callback(attendibilita_msg)

        return previsione_completa, attendibilita_msg, last_update_date

    except Exception as e_final_pred:
         if log_callback: log_callback(f"Errore CRITICO durante fase di previsione finale: {e_final_pred}\n{traceback.format_exc()}")
         return None, f"Errore critico previsione finale: {e_final_pred}", last_update_date
# --- Fine Funzione Analisi ---


# --- Definizione Classe GUI AppSuperEnalotto ---
class AppSuperEnalotto:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi e Previsione SuperEnalotto ML (v1.3.1 - Corretto)")
        self.root.geometry("850x950")

        self.style = ttk.Style()
        try:
            if sys.platform == "win32": self.style.theme_use('vista')
            elif sys.platform == "darwin": self.style.theme_use('aqua')
            else: self.style.theme_use('clam')
        except tk.TclError:
            self.style.theme_use('default')

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.last_prediction_numbers = None
        self.last_prediction_full = None
        self.last_prediction_end_date = None
        self.last_prediction_date_str = None

        # --- Input File/URL ---
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Origine Dati Estrazioni (URL Raw GitHub o Percorso File Locale .txt)", padding="10")
        self.file_frame.pack(fill=tk.X, pady=5)
        self.file_path_var = tk.StringVar(value=DEFAULT_SUPERENALOTTO_DATA_URL)
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
        default_start = datetime.now() - pd.Timedelta(days=1095)
        if HAS_TKCALENDAR:
             self.start_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd', show_weeknumbers=False, locale='it_IT')
             try: self.start_date_entry.set_date(default_start)
             except ValueError: self.start_date_entry.set_date(datetime.now() - pd.Timedelta(days=365))
             self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        else:
            self.start_date_entry_var = tk.StringVar(value=default_start.strftime('%Y-%m-%d'))
            self.start_date_entry = ttk.Entry(self.data_params_frame, textvariable=self.start_date_entry_var, width=12)
            self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Label(self.data_params_frame, text="(yyyy-mm-dd)").grid(row=0, column=2, padx=2, pady=5, sticky=tk.W)

        ttk.Label(self.data_params_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        default_end = datetime.now()
        if HAS_TKCALENDAR:
             self.end_date_entry = DateEntry(self.data_params_frame, width=12, date_pattern='yyyy-mm-dd', show_weeknumbers=False, locale='it_IT')
             self.end_date_entry.set_date(default_end)
             self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        else:
            self.end_date_entry_var = tk.StringVar(value=default_end.strftime('%Y-%m-%d'))
            self.end_date_entry = ttk.Entry(self.data_params_frame, textvariable=self.end_date_entry_var, width=12)
            self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            ttk.Label(self.data_params_frame, text="(yyyy-mm-dd)").grid(row=1, column=2, padx=2, pady=5, sticky=tk.W)

        ttk.Label(self.data_params_frame, text="Seq. Input (Storia):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.seq_len_var = tk.StringVar(value="10")
        self.seq_len_entry = ttk.Spinbox(self.data_params_frame, from_=3, to=50, increment=1, textvariable=self.seq_len_var, width=5, wrap=True, state='readonly')
        self.seq_len_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(self.data_params_frame, text="Numeri da Prevedere:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_predict_var = tk.StringVar(value="6")
        self.num_predict_spinbox = ttk.Spinbox(self.data_params_frame, from_=6, to=15, increment=1, textvariable=self.num_predict_var, width=5, wrap=True, state='readonly')
        self.num_predict_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # --- Colonna Destra: Parametri Modello e Training ---
        self.model_params_frame = ttk.LabelFrame(self.params_container, text="Configurazione Modello e Training", padding="10")
        self.model_params_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        self.model_params_frame.columnconfigure(1, weight=1)

        ttk.Label(self.model_params_frame, text="Hidden Layers (n,n,..):").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.hidden_layers_var = tk.StringVar(value="256, 128, 64")
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
        self.dropout_var = tk.StringVar(value="0.30")
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
        self.epochs_var = tk.StringVar(value="150")
        self.epochs_spinbox = ttk.Spinbox(self.model_params_frame, from_=20, to=1000, increment=10, textvariable=self.epochs_var, width=7, wrap=True, state='readonly')
        self.epochs_spinbox.grid(row=6, column=1, padx=5, pady=3, sticky=tk.W)

        ttk.Label(self.model_params_frame, text="Batch Size:").grid(row=7, column=0, padx=5, pady=3, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="64")
        batch_values = [str(2**i) for i in range(3, 9)]
        self.batch_size_combo = ttk.Combobox(self.model_params_frame, textvariable=self.batch_size_var, values=batch_values, width=5, state='readonly')
        self.batch_size_combo.grid(row=7, column=1, padx=5, pady=3, sticky=tk.W)

        ttk.Label(self.model_params_frame, text="ES Patience:").grid(row=8, column=0, padx=5, pady=3, sticky=tk.W)
        self.patience_var = tk.StringVar(value="20")
        self.patience_spinbox = ttk.Spinbox(self.model_params_frame, from_=5, to=100, increment=1, textvariable=self.patience_var, width=7, wrap=True, state='readonly')
        self.patience_spinbox.grid(row=8, column=1, padx=5, pady=3, sticky=tk.W)

        ttk.Label(self.model_params_frame, text="ES Min Delta:").grid(row=9, column=0, padx=5, pady=3, sticky=tk.W)
        self.min_delta_var = tk.StringVar(value="0.0001")
        self.min_delta_entry = ttk.Entry(self.model_params_frame, textvariable=self.min_delta_var, width=10)
        self.min_delta_entry.grid(row=9, column=1, padx=5, pady=3, sticky=tk.W)

        # --- Pulsanti Azione ---
        self.action_frame = ttk.Frame(self.main_frame)
        self.action_frame.pack(pady=10)
        self.run_button = ttk.Button(self.action_frame, text="Avvia Analisi e Previsione", command=self.start_analysis_thread)
        self.run_button.pack(side=tk.LEFT, padx=10)
        self.check_button = ttk.Button(self.action_frame, text="Verifica Ultima Previsione", command=self.start_check_thread, state=tk.DISABLED)
        self.check_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.action_frame, text="Colpi da Verificare:").pack(side=tk.LEFT, padx=(10, 2))
        self.check_colpi_var = tk.StringVar(value=str(DEFAULT_SUPERENALOTTO_CHECK_COLPI))
        self.check_colpi_spinbox = ttk.Spinbox(self.action_frame, from_=1, to=100, increment=1, textvariable=self.check_colpi_var, width=4, wrap=True, state='readonly')
        self.check_colpi_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        # --- Risultati Previsione ---
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Risultato Previsione SuperEnalotto (Numeri più probabili secondo il modello)", padding="10")
        self.results_frame.pack(fill=tk.X, pady=5)
        self.result_label_var = tk.StringVar(value="I numeri previsti appariranno qui...")
        self.result_label = ttk.Label(self.results_frame, textvariable=self.result_label_var, font=('Courier New', 16, 'bold'), foreground='darkblue')
        self.result_label.pack(pady=5)
        self.attendibilita_label_var = tk.StringVar(value="")
        self.attendibilita_label = ttk.Label(self.results_frame, textvariable=self.attendibilita_label_var, font=('Helvetica', 9, 'italic'))
        self.attendibilita_label.pack(pady=2)

        # --- Data Ultimo Aggiornamento Dati Usati ---
        self.last_update_frame = ttk.LabelFrame(self.main_frame, text="Dati Utilizzati nell'Ultima Analisi", padding="5")
        self.last_update_frame.pack(fill=tk.X, pady=5)
        self.last_update_label_var = tk.StringVar(value="Data ultimo aggiornamento usato apparirà qui...")
        self.last_update_label = ttk.Label(self.last_update_frame, textvariable=self.last_update_label_var, font=('Helvetica', 9))
        self.last_update_label.pack(pady=3)

        # --- Log Area ---
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Elaborazione", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        log_font = ("Consolas", 9) if sys.platform == "win32" else ("Monaco", 9)
        try:
            self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='#f0f0f0', foreground='black')
        except tk.TclError:
             log_font = ("Courier New", 9)
             self.log_text = scrolledtext.ScrolledText(self.log_frame, height=15, width=90, wrap=tk.WORD, state=tk.DISABLED, font=log_font, background='#f0f0f0', foreground='black')
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Threading
        self.analysis_thread = None
        self.check_thread = None

    # --- Metodi della Classe GUI ---

    def browse_file(self):
        """Apre una finestra di dialogo per selezionare un file LOCALE."""
        filepath = filedialog.askopenfilename(
            title="Seleziona file estrazioni SuperEnalotto Locale (.txt)",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
            parent=self.root
        )
        if filepath:
            self.file_path_var.set(filepath)
            self.log_message_gui(f"Selezionato file locale: {filepath}")

    def log_message_gui(self, message):
        """Wrapper per inviare messaggi al widget di log dalla GUI."""
        log_message(message, self.log_text, self.root)

    def set_result(self, prediction_data, attendibilita):
         """
         Aggiorna i label dei risultati nella GUI.
         prediction_data: Lista di dizionari [{'number': n, 'probability': p}] o None.
         attendibilita: Stringa messaggio.
         """
         self.root.after(0, self._update_result_labels, prediction_data, attendibilita)

    def _update_result_labels(self, prediction_data, attendibilita):
        """Funzione helper eseguita nel thread GUI per aggiornare le etichette."""
        if (isinstance(prediction_data, list) and prediction_data and
                all(isinstance(item, dict) and 'number' in item and 'probability' in item for item in prediction_data)):

            sorted_by_number = sorted(prediction_data, key=lambda x: x['number'])
            result_str = "  ".join([f"{item['number']:02d}" for item in sorted_by_number])
            self.result_label_var.set(result_str)

            log_nums_only = sorted([item['number'] for item in prediction_data])
            self.log_message_gui("\n" + "="*35 + "\nPREVISIONE SUPERENALOTTO GENERATA (ML)\n" + "="*35)
            self.log_message_gui("Numeri e probabilità stimate dal modello (ordinati per probabilità decrescente):")
            sorted_by_prob_desc = sorted(prediction_data, key=lambda x: x['probability'], reverse=True)
            for item in sorted_by_prob_desc:
                 self.log_message_gui(f"  - Numero: {item['number']:02d} (Prob: {item['probability']:.6f})")
            self.log_message_gui(f"Numeri finali selezionati (ordinati): {log_nums_only}")
            self.log_message_gui("="*35)

        else:
            self.result_label_var.set("Previsione fallita o non valida.")
            log_err = True
            if isinstance(attendibilita, str) and \
               any(keyword in attendibilita.lower() for keyword in ["errore", "fallit", "insufficienti", "invalido", "nessun"]):
                 log_err = False
            if log_err: self.log_message_gui("\nERRORE: La previsione non ha restituito dati validi.")

        self.attendibilita_label_var.set(str(attendibilita) if attendibilita else "Attendibilità non disponibile.")

    def set_controls_state(self, state):
        """Abilita o disabilita i controlli della GUI (tk.NORMAL o tk.DISABLED)."""
        self.root.after(10, lambda: self._set_controls_state_tk(state))

    def _set_controls_state_tk(self, state):
        """Funzione helper eseguita nel thread GUI per modificare lo stato dei widget."""
        is_running = (self.analysis_thread and self.analysis_thread.is_alive()) or \
                     (self.check_thread and self.check_thread.is_alive())
        actual_state = tk.DISABLED if is_running else tk.NORMAL

        widgets_to_toggle = [
            self.browse_button, self.file_entry,
            self.seq_len_entry, self.num_predict_spinbox,
            self.hidden_layers_entry, self.loss_combo, self.optimizer_combo,
            self.dropout_spinbox, self.l1_entry, self.l2_entry,
            self.epochs_spinbox, self.batch_size_combo,
            self.patience_spinbox, self.min_delta_entry,
            self.run_button, self.check_colpi_spinbox, self.check_button
        ]

        if HAS_TKCALENDAR and hasattr(self.start_date_entry, 'configure'):
             widgets_to_toggle.extend([self.start_date_entry, self.end_date_entry])
        elif not HAS_TKCALENDAR:
             widgets_to_toggle.extend([self.start_date_entry, self.end_date_entry])

        for widget in widgets_to_toggle:
            widget_state = actual_state

            if widget == self.check_button:
                if actual_state == tk.NORMAL and self.last_prediction_numbers:
                    widget_state = tk.NORMAL
                else:
                    widget_state = tk.DISABLED

            if widget == self.run_button and actual_state == tk.DISABLED:
                 widget_state = tk.DISABLED

            try:
                current_widget_type = widget.winfo_class()
                if current_widget_type in ('TCombobox', 'TSpinbox'):
                    widget.config(state='readonly' if widget_state == tk.NORMAL else tk.DISABLED)
                elif current_widget_type == 'DateEntry' and HAS_TKCALENDAR:
                    widget.config(state=widget_state)
                else:
                    widget.config(state=widget_state)
            except (tk.TclError, AttributeError):
                 pass

    def start_analysis_thread(self):
        """Avvia il thread per l'analisi e la previsione."""
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Analisi in Corso", "Un'analisi SuperEnalotto è già in esecuzione.", parent=self.root)
            return
        if self.check_thread and self.check_thread.is_alive():
            messagebox.showwarning("Verifica in Corso", "Una verifica è in corso. Attendere la fine prima di avviare una nuova analisi.", parent=self.root)
            return

        #<editor-fold desc="Recupero e Validazione Parametri GUI">
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.result_label_var.set("Analisi in corso...")
        self.attendibilita_label_var.set("")
        self.last_update_label_var.set("Data ultimo aggiornamento apparirà qui...")
        self.last_prediction_numbers = None
        self.last_prediction_full = None
        self.last_prediction_end_date = None
        self.last_prediction_date_str = None
        self.check_button.config(state=tk.DISABLED)

        data_source = self.file_path_var.get().strip()
        start_date_str, end_date_str = "", ""
        try:
            if HAS_TKCALENDAR and isinstance(self.start_date_entry, DateEntry):
                start_date_str = self.start_date_entry.get_date().strftime('%Y-%m-%d')
                end_date_str = self.end_date_entry.get_date().strftime('%Y-%m-%d')
            else:
                start_date_str = self.start_date_entry_var.get()
                end_date_str = self.end_date_entry_var.get()
        except AttributeError:
             start_date_str = self.start_date_entry.get()
             end_date_str = self.end_date_entry.get()
        except Exception as e_date_get:
            messagebox.showerror("Errore Lettura Date", f"Impossibile leggere le date dai widget: {e_date_get}", parent=self.root)
            return

        seq_len_str = self.seq_len_var.get()
        num_predict_str = self.num_predict_var.get()
        hidden_layers_str = self.hidden_layers_var.get()
        loss_function = self.loss_var.get()
        optimizer = self.optimizer_var.get()
        dropout_str = self.dropout_var.get()
        l1_str = self.l1_var.get()
        l2_str = self.l2_var.get()
        epochs_str = self.epochs_var.get()
        batch_size_str = self.batch_size_var.get()
        patience_str = self.patience_var.get()
        min_delta_str = self.min_delta_var.get()

        errors = []
        sequence_length, num_predictions = 10, 6
        hidden_layers_config = [256, 128, 64]
        dropout_rate, l1_reg, l2_reg = 0.3, 0.0, 0.0
        max_epochs, batch_size, patience, min_delta = 150, 64, 20, 0.0001

        if not data_source:
             errors.append("- Specificare un URL Raw GitHub valido o un percorso file locale.")
        elif not data_source.startswith(("http://", "https://")):
            if not os.path.exists(data_source):
                errors.append(f"- File locale non trovato: {data_source}")
            elif not data_source.lower().endswith(".txt"):
                 errors.append("- Il file locale dovrebbe avere estensione .txt.")

        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_dt > end_dt: errors.append("- Data inizio deve essere <= Data fine.")
        except ValueError: errors.append("- Formato data non valido (richiesto YYYY-MM-DD).")

        try:
            sequence_length = int(seq_len_str)
            if not (3 <= sequence_length <= 50): raise ValueError()
        except: errors.append("- Seq. Input deve essere un intero tra 3 e 50.")
        try:
            num_predictions = int(num_predict_str)
            if not (6 <= num_predictions <= 15): raise ValueError()
        except: errors.append("- Numeri da Prevedere deve essere un intero tra 6 e 15.")
        try:
            layers_str_list = [x.strip() for x in hidden_layers_str.split(',') if x.strip()]
            if not layers_str_list: raise ValueError("Lista vuota")
            hidden_layers_config = [int(x) for x in layers_str_list]
            if not all(n > 0 for n in hidden_layers_config): raise ValueError("Unità non positive")
        except: errors.append("- Hidden Layers non validi (es. '256, 128'). Usare numeri interi positivi separati da virgola.")
        if not loss_function: errors.append("- Selezionare una Loss Function.")
        if not optimizer: errors.append("- Selezionare un Optimizer.")
        try:
            dropout_rate = float(dropout_str)
            if not (0.0 <= dropout_rate < 1.0): raise ValueError()
        except: errors.append("- Dropout Rate deve essere un numero tra 0.0 (incluso) e 1.0 (escluso).")
        try:
            l1_reg = float(l1_str)
            if l1_reg < 0: raise ValueError()
        except: errors.append("- L1 Strength deve essere un numero >= 0.")
        try:
            l2_reg = float(l2_str)
            if l2_reg < 0: raise ValueError()
        except: errors.append("- L2 Strength deve essere un numero >= 0.")
        try:
            max_epochs = int(epochs_str)
            if max_epochs < 10: raise ValueError()
        except: errors.append("- Max Epoche deve essere un intero >= 10.")
        try:
            batch_size = int(batch_size_str)
            if batch_size <= 0: raise ValueError()
        except: errors.append("- Batch Size deve essere un intero positivo.")
        try:
            patience = int(patience_str)
            if patience < 1: raise ValueError()
        except: errors.append("- ES Patience deve essere un intero >= 1.")
        try:
            min_delta = float(min_delta_str)
            if min_delta < 0: raise ValueError()
        except: errors.append("- ES Min Delta deve essere un numero >= 0.")

        if errors:
            error_message = "Correggere i seguenti errori nei parametri:\n\n" + "\n".join(errors)
            messagebox.showerror("Errore Parametri Input", error_message, parent=self.root)
            self.result_label_var.set("Errore nei parametri di input.")
            return
        #</editor-fold>

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui("=== Avvio Analisi SuperEnalotto (Thread) ===")
        self.log_message_gui(f"Sorgente Dati: {data_source}")
        self.log_message_gui(f"Periodo: {start_date_str} - {end_date_str}")
        self.log_message_gui(f"SeqLen: {sequence_length}, NumPred: {num_predictions}")
        self.log_message_gui(f"Layers: {hidden_layers_config}, Loss: {loss_function}, Opt: {optimizer}")
        self.log_message_gui(f"Reg/Drop: L1={l1_reg:.4f}, L2={l2_reg:.4f}, Drop={dropout_rate:.2f}")
        self.log_message_gui(f"Training: Epochs={max_epochs}, Batch={batch_size}, Patience={patience}, MinDelta={min_delta:.6f}")
        self.log_message_gui("-" * 40)

        self.analysis_thread = threading.Thread(
            target=self.run_analysis,
            args=(
                data_source, start_date_str, end_date_str, sequence_length,
                loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                num_predictions
            ),
            daemon=True
        )
        self.analysis_thread.start()

    def run_analysis(self, data_source, start_date, end_date, sequence_length,
                     loss_function, optimizer, dropout_rate, l1_reg, l2_reg,
                     hidden_layers_config, max_epochs, batch_size, patience, min_delta,
                     num_predictions):
        """
        Funzione eseguita nel thread secondario per effettuare l'analisi.
        """
        self.last_prediction_numbers = None
        self.last_prediction_full = None
        self.last_prediction_end_date = None
        self.last_prediction_date_str = None
        analysis_success = False
        final_attendibilita_msg = "Analisi non completata."
        final_last_update_date = None
        previsione_completa_result = None # Variabile per tenere il risultato

        try:
            previsione_completa_result, final_attendibilita_msg, final_last_update_date = analisi_superenalotto(
                file_path=data_source, start_date=start_date, end_date=end_date,
                sequence_length=sequence_length, loss_function=loss_function,
                optimizer=optimizer, dropout_rate=dropout_rate, l1_reg=l1_reg,
                l2_reg=l2_reg, hidden_layers_config=hidden_layers_config,
                max_epochs=max_epochs, batch_size=batch_size, patience=patience,
                min_delta=min_delta, num_predictions=num_predictions,
                log_callback=self.log_message_gui
            )

            if (isinstance(previsione_completa_result, list) and previsione_completa_result and
                    len(previsione_completa_result) == num_predictions and
                    all(isinstance(item, dict) and 'number' in item and 'probability' in item for item in previsione_completa_result)):
                analysis_success = True
                self.last_prediction_full = previsione_completa_result
                self.last_prediction_numbers = sorted([item['number'] for item in previsione_completa_result])

                try:
                    self.last_prediction_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                    self.last_prediction_date_str = end_date
                    self.log_message_gui(f"Previsione valida generata e salvata (basata su dati fino al {end_date}). Pronta per verifica.")
                except ValueError:
                    self.log_message_gui(f"ATTENZIONE: Errore nel formato data fine ({end_date}) dopo l'analisi. La verifica potrebbe non funzionare.")
                    self.last_prediction_end_date = None
                    self.last_prediction_date_str = None
            else:
                self.log_message_gui(f"Analisi completata ma non ha prodotto una previsione valida. Messaggio: {final_attendibilita_msg}")

            if final_last_update_date is not None:
                last_update_str = final_last_update_date.strftime('%Y-%m-%d')
                self.root.after(0, lambda: self.last_update_label_var.set(f"Dati analizzati fino al: {last_update_str}"))
            else:
                 self.root.after(0, lambda: self.last_update_label_var.set("Data ultimo aggiornamento non disponibile."))

            # Passa previsione_completa_result a set_result
            self.set_result(previsione_completa_result, final_attendibilita_msg)

        except Exception as e_run:
            self.log_message_gui(f"\nERRORE CRITICO nel thread run_analysis: {e_run}\n{traceback.format_exc()}")
            final_attendibilita_msg = f"Errore critico: {e_run}"
            self.set_result(None, final_attendibilita_msg)
            analysis_success = False
        finally:
            self.log_message_gui("\n=== Analisi SuperEnalotto (Thread) Completata ===")
            self.set_controls_state(tk.NORMAL)
            self.analysis_thread = None


    def start_check_thread(self):
        """Avvia il thread per la verifica dell'ultima previsione."""
        if self.check_thread and self.check_thread.is_alive():
            messagebox.showwarning("Verifica in Corso", "Una verifica SuperEnalotto è già in esecuzione.", parent=self.root)
            return
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Analisi in Corso", "Attendere la fine dell'analisi prima di avviare la verifica.", parent=self.root)
            return

        if not self.last_prediction_numbers or not self.last_prediction_end_date or not self.last_prediction_date_str:
            messagebox.showinfo("Nessuna Previsione", "Nessuna previsione valida disponibile per la verifica. Eseguire prima un'analisi con successo.", parent=self.root)
            return
        if not isinstance(self.last_prediction_numbers, list) or not all(isinstance(n, int) for n in self.last_prediction_numbers):
            messagebox.showerror("Errore Previsione", "I dati della previsione salvata sembrano corrotti.", parent=self.root)
            self.last_prediction_numbers = None
            self.set_controls_state(tk.NORMAL)
            return

        try:
            num_colpi_to_check = int(self.check_colpi_var.get())
            if not (1 <= num_colpi_to_check <= 100): raise ValueError()
        except:
            messagebox.showerror("Errore Input", "Numero colpi da verificare non valido (deve essere un intero tra 1 e 100).", parent=self.root)
            return

        data_source_for_check = self.file_path_var.get().strip()
        if not data_source_for_check:
            messagebox.showerror("Errore Sorgente Dati", "Specificare la sorgente dati (URL o file) per poter effettuare la verifica.", parent=self.root)
            return
        if not data_source_for_check.startswith(("http://", "https://")) and not os.path.exists(data_source_for_check):
            messagebox.showerror("Errore File", f"Il file dati locale specificato ('{os.path.basename(data_source_for_check)}') non è stato trovato per la verifica.", parent=self.root)
            return

        self.set_controls_state(tk.DISABLED)
        self.log_message_gui(f"\n=== Avvio Verifica Previsione SuperEnalotto ({num_colpi_to_check} Colpi Max) ===")
        self.log_message_gui(f"Previsione da verificare (numeri): {self.last_prediction_numbers}")
        self.log_message_gui(f"Previsione basata su dati fino al: {self.last_prediction_date_str}")
        self.log_message_gui(f"Sorgente dati per verifica: {data_source_for_check}")
        self.log_message_gui("-" * 40)

        self.check_thread = threading.Thread(
            target=self.run_check_results,
            args=(data_source_for_check,
                  self.last_prediction_numbers,
                  self.last_prediction_date_str,
                  num_colpi_to_check),
            daemon=True
        )
        self.check_thread.start()

    def run_check_results(self, data_source, prediction_numbers_to_check, last_analysis_date_str, num_colpi_to_check):
        """
        Esegue la verifica nel thread.
        """
        try:
            try:
                last_date_obj = datetime.strptime(last_analysis_date_str, '%Y-%m-%d')
                check_start_date = last_date_obj + timedelta(days=1)
                check_start_date_str = check_start_date.strftime('%Y-%m-%d')
            except ValueError as ve_date:
                self.log_message_gui(f"ERRORE CRITICO: Formato data analisi non valido ({last_analysis_date_str}): {ve_date}. Impossibile procedere con la verifica.")
                return

            self.log_message_gui(f"Caricamento dati SuperEnalotto per verifica (da {check_start_date_str} in poi)...")

            df_check, numeri_principali_check, _, _, _ = carica_dati_superenalotto(
                data_source, start_date=check_start_date_str, end_date=None,
                log_callback=self.log_message_gui
            )

            if df_check is None:
                self.log_message_gui("ERRORE: Caricamento dati per la verifica fallito.")
                return
            if df_check.empty:
                self.log_message_gui(f"INFO: Nessuna estrazione trovata nel file/URL dopo la data {last_analysis_date_str}.")
                return
            if numeri_principali_check is None or len(numeri_principali_check) == 0:
                self.log_message_gui(f"ERRORE: Dati trovati dopo {last_analysis_date_str}, ma impossibile estrarre i numeri principali per la verifica.")
                return

            num_estrazioni_disponibili = len(numeri_principali_check)
            num_colpi_effettivi = min(num_colpi_to_check, num_estrazioni_disponibili)

            self.log_message_gui(f"Trovate {num_estrazioni_disponibili} estrazioni successive. Verifico le prossime {num_colpi_effettivi}...")
            prediction_set = set(prediction_numbers_to_check)
            self.log_message_gui(f"Numeri previsti (Set): {prediction_set}")
            self.log_message_gui("-" * 40)

            colpo_counter = 0
            found_any_hit = False
            highest_score = 0

            for i in range(num_colpi_effettivi):
                colpo_counter += 1
                try:
                    draw_row = df_check.iloc[i]
                    draw_date_str = draw_row['Data'].strftime('%Y-%m-%d')
                    actual_draw_numbers = numeri_principali_check[i]
                    actual_draw_set = set(actual_draw_numbers)
                    hits = prediction_set.intersection(actual_draw_set)
                    num_hits = len(hits)
                    highest_score = max(highest_score, num_hits)

                    log_line = f"Colpo {colpo_counter:02d}/{num_colpi_effettivi:02d} ({draw_date_str}): Estrazione={sorted(list(actual_draw_set))} -> "
                    if num_hits > 0:
                        found_any_hit = True
                        hits_sorted = sorted(list(hits))
                        points_str = f"{num_hits} punti" if num_hits != 1 else "1 punto"
                        log_line += f"*** {points_str}! Numeri indovinati: {hits_sorted} ***"
                        self.log_message_gui(log_line)
                    else:
                        log_line += "Nessun risultato."
                        self.log_message_gui(log_line)

                except IndexError:
                    self.log_message_gui(f"ERRORE INTERNO: Indice {i} fuori range durante la verifica (Estrazioni disp: {len(numeri_principali_check)}).")
                    break
                except KeyError as ke:
                    self.log_message_gui(f"ERRORE Dati: Colonna '{ke}' mancante nel DataFrame di verifica al colpo {colpo_counter}.")
                    continue
                except Exception as e_row_check:
                    self.log_message_gui(f"ERRORE imprevisto durante l'analisi del colpo {colpo_counter} ({draw_date_str}): {e_row_check}")
                    continue

            self.log_message_gui("-" * 40)
            if not found_any_hit:
                self.log_message_gui(f"Nessun numero della previsione è stato estratto nei {num_colpi_effettivi} colpi verificati.")
            else:
                self.log_message_gui(f"Verifica completata. Punteggio massimo ottenuto: {highest_score} punti su 6.")

        except Exception as e_check_main:
            self.log_message_gui(f"ERRORE CRITICO durante la verifica SuperEnalotto: {e_check_main}\n{traceback.format_exc()}")
        finally:
            self.log_message_gui("\n=== Verifica SuperEnalotto (Thread) Completata ===")
            self.set_controls_state(tk.NORMAL)
            self.check_thread = None

# --- Fine Classe GUI ---


# --- Funzione di Lancio ---
def launch_superenalotto_window(parent_window=None):
    """
    Crea e lancia la finestra dell'applicazione SuperEnalotto.
    """
    try:
        if parent_window:
            superenalotto_win = tk.Toplevel(parent_window)
            superenalotto_win.transient(parent_window)
        else:
            superenalotto_win = tk.Tk()

        superenalotto_win.title("SuperEnalotto ML Predictor (Corretto)")
        superenalotto_win.geometry("850x950")

        app_instance = AppSuperEnalotto(superenalotto_win)

        superenalotto_win.update_idletasks()
        x = (superenalotto_win.winfo_screenwidth() // 2) - (superenalotto_win.winfo_width() // 2)
        y = (superenalotto_win.winfo_screenheight() // 2) - (superenalotto_win.winfo_height() // 2)
        superenalotto_win.geometry(f'+{x}+{y}')

        superenalotto_win.lift()
        superenalotto_win.focus_force()

        if not parent_window:
            superenalotto_win.mainloop()

    except Exception as e_launch:
        print(f"ERRORE CRITICO durante il lancio della finestra SuperEnalotto: {e_launch}\n{traceback.format_exc()}")
        try:
            messagebox.showerror("Errore Avvio Applicazione", f"Errore critico durante l'avvio:\n{e_launch}", parent=parent_window)
        except:
            pass

# --- Blocco Esecuzione Standalone ---
if __name__ == "__main__":
    print("Esecuzione Modulo SuperEnalotto ML Predictor in modalità standalone...")
    print("-" * 60)
    print("Requisiti: tensorflow, pandas, numpy, requests")
    print("Opzionale (per calendario): tkcalendar")
    print("Installazione: pip install tensorflow pandas numpy requests tkcalendar")
    print("-" * 60)

    try:
        if sys.platform == "win32":
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
            print("INFO: DPI awareness impostato per Windows.")
    except Exception as e_dpi:
        print(f"Nota: Impossibile impostare DPI awareness: {e_dpi}")

    launch_superenalotto_window(parent_window=None)

    print("\nFinestra SuperEnalotto chiusa. Programma terminato.")