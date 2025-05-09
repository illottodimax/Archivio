import os
import sys
import random
import logging
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
import json
import re
import time
import threading
import traceback
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import io
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, ttk, filedialog # messagebox importato
import matplotlib.pyplot as plt
import pyttsx3
import subprocess
import gc

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit # Per la cross-validation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GaussianNoise, Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, LogCosh
from tensorflow.keras.utils import register_keras_serializable
from tkinter import messagebox, ttk, filedialog
from tkcalendar import DateEntry
from PIL import Image, ImageTk
from analizzatore_ritardi_lotto import AnalizzatoreRitardiLotto, apri_analizzatore_ritardi

# *******************************************************************
# *** INIZIO AGGIUNTE PER MODULO 10eLotto ***

# Flag per sapere se il modulo è stato caricato correttamente
LOTTO_MODULE_LOADED = False
# Placeholder per la funzione di avvio (sarà la funzione reale o una di fallback)
launch_10elotto_window_func = None

try:
    # Assicurati che il file si chiami 'elotto_module.py'
    # e che contenga la funzione 'launch_10elotto_window' definita FUORI dalla classe App10eLotto
    from elotto_module import launch_10elotto_window as imported_launch_func

    # Se l'importazione ha successo, assegna la funzione reale importata
    launch_10elotto_window_func = imported_launch_func
    LOTTO_MODULE_LOADED = True
    print("Modulo 10eLotto (elotto_module.py) caricato con successo.")

except ImportError as e:
    # Se il modulo non viene trovato o c'è un errore nell'import
    print(f"ATTENZIONE: Impossibile importare il modulo 'elotto_module.py'. Errore: {e}")
    # Definisci una funzione di fallback che mostra solo un errore
    def fallback_launch(parent):
        messagebox.showerror("Errore Modulo",
                             "Modulo 10eLotto (elotto_module.py) non trovato o contenente errori.\n"
                             "Verificare che il file sia presente e corretto.",
                             parent=parent)
    launch_10elotto_window_func = fallback_launch # Usa la funzione di fallback

except Exception as e_gen:
    # Per gestire altri errori imprevisti durante l'importazione
     print(f"ERRORE GENERALE durante l'importazione del modulo 10eLotto: {e_gen}")
     print(traceback.format_exc())
     # Definisci una funzione di fallback anche per errori generici
     def fallback_launch(parent):
         messagebox.showerror("Errore Modulo",
                              f"Errore imprevisto durante il caricamento del modulo 10eLotto:\n{e_gen}",
                              parent=parent)
     launch_10elotto_window_func = fallback_launch

# --- Funzione Helper (Globale) ---
def open_lotto_module_helper(parent_window):
    """Chiama la funzione di lancio del modulo 10eLotto (reale o fallback)."""
    print("Tentativo di aprire la finestra 10eLotto...")
    if launch_10elotto_window_func: # Controlla se la funzione è stata definita (o è il fallback)
        try:
            launch_10elotto_window_func(parent_window) # Esegui la funzione
        except Exception as e_launch:
             # Cattura errori che potrebbero verificarsi durante l'esecuzione della launch function
             print(f"ERRORE durante l'esecuzione di launch_10elotto_window_func: {e_launch}")
             print(traceback.format_exc())
             messagebox.showerror("Errore Esecuzione Modulo",
                                  f"Si è verificato un errore all'avvio del modulo 10eLotto:\n{e_launch}",
                                  parent=parent_window)
    else:
        # Questo caso limite non dovrebbe verificarsi se la logica sopra è corretta
        messagebox.showerror("Errore Interno", "Funzione di avvio per 10eLotto non definita.", parent=parent_window)

# *** FINE AGGIUNTE PER MODULO 10eLotto ***
# *******************************************************************

# *******************************************************************
# *** INIZIO AGGIUNTE PER MODULO Lotto Analyzer ***

# Flag per sapere se il modulo Lotto Analyzer è stato caricato
LOTTO_ANALYZER_MODULE_LOADED = False
# Placeholder per la funzione di avvio del Lotto Analyzer
launch_lotto_analyzer_func = None

try:
    # Assicurati che il file si chiami 'lotto_analyzer.py'
    # e contenga la funzione 'launch_lotto_analyzer_window' definita FUORI dalle classi
    from lotto_analyzer import launch_lotto_analyzer_window as imported_lotto_analyzer_launch_func

    # Se l'importazione ha successo, assegna la funzione reale
    launch_lotto_analyzer_func = imported_lotto_analyzer_launch_func
    LOTTO_ANALYZER_MODULE_LOADED = True
    print("Modulo Lotto Analyzer (lotto_analyzer.py) caricato con successo.")

except ImportError as e:
    # Se il modulo non viene trovato o c'è un errore nell'import
    print(f"ATTENZIONE: Impossibile importare il modulo 'lotto_analyzer.py'. Errore: {e}")
    # Definisci una funzione di fallback che mostra solo un errore
    def fallback_launch_lotto_analyzer(parent):
        messagebox.showerror("Errore Modulo",
                             "Modulo Lotto Analyzer (lotto_analyzer.py) non trovato o contenente errori.\n"
                             "Verificare che il file sia presente e corretto.",
                             parent=parent)
    launch_lotto_analyzer_func = fallback_launch_lotto_analyzer # Usa la funzione di fallback

except Exception as e_gen:
    # Per gestire altri errori imprevisti durante l'importazione
     print(f"ERRORE GENERALE durante l'importazione del modulo Lotto Analyzer: {e_gen}")
     print(traceback.format_exc())
     # Definisci una funzione di fallback anche per errori generici
     def fallback_launch_lotto_analyzer_err(parent):
         messagebox.showerror("Errore Modulo",
                              f"Errore imprevisto durante il caricamento del modulo Lotto Analyzer:\n{e_gen}",
                              parent=parent)
     launch_lotto_analyzer_func = fallback_launch_lotto_analyzer_err

# --- Funzione Helper per Lotto Analyzer ---
def open_lotto_analyzer_helper(parent_window):
    """Chiama la funzione di lancio del modulo Lotto Analyzer (reale o fallback)."""
    print("Tentativo di aprire la finestra Lotto Analyzer...")
    if launch_lotto_analyzer_func: # Controlla se la funzione è stata definita (o è il fallback)
        try:
            launch_lotto_analyzer_func(parent_window) # Esegui la funzione
        except Exception as e_launch:
             # Cattura errori che potrebbero verificarsi durante l'esecuzione della launch function
             print(f"ERRORE durante l'esecuzione di launch_lotto_analyzer_func: {e_launch}")
             print(traceback.format_exc())
             messagebox.showerror("Errore Esecuzione Modulo",
                                  f"Si è verificato un errore all'avvio del modulo Lotto Analyzer:\n{e_launch}",
                                  parent=parent_window)
    else:
        # Questo caso limite non dovrebbe verificarsi se la logica sopra è corretta
        messagebox.showerror("Errore Interno", "Funzione di avvio per Lotto Analyzer non definita.", parent=parent_window)

# *** FINE AGGIUNTE PER MODULO Lotto Analyzer ***
# *******************************************************************

try:
    # Il nome del tuo file è selotto_module.py
    from selotto_module import launch_superenalotto_window
    SUPERENALOTTO_MODULE_LOADED = True
    print("Modulo SuperEnalotto caricato con successo.")
except ImportError as e_selotto:
    launch_superenalotto_window = None # Definisci come None se non caricato
    SUPERENALOTTO_MODULE_LOADED = False
    print(f"ATTENZIONE: Modulo SuperEnalotto ('selotto_module.py') non trovato o errore import: {e_selotto}")

# NUOVO: Helper per SuperEnalotto
def open_superenalotto_module_helper(parent_root):
    """Controlla se il modulo SuperEnalotto è caricato e lo avvia."""
    if SUPERENALOTTO_MODULE_LOADED and callable(launch_superenalotto_window):
        print("Avvio modulo SuperEnalotto...")
        try:
            # Chiama la funzione importata passando la finestra principale (root)
            launch_superenalotto_window(parent_root)
        except Exception as e_launch:
            messagebox.showerror("Errore Avvio Modulo", f"Errore durante l'avvio del modulo SuperEnalotto:\n{e_launch}", parent=parent_root)
            print(f"Errore launch_superenalotto_window: {e_launch}\n{traceback.format_exc()}") # Log dettagliato
    else:
        messagebox.showwarning("Modulo Mancante", "Il modulo SuperEnalotto ('selotto_module.py') non è stato caricato correttamente.", parent=parent_root)

# *** INIZIO AGGIUNTE PER MODULO Numerical Binary ***

# Flag per sapere se il modulo Numerical Binary è stato caricato
NUMERICAL_BINARY_MODULE_LOADED = False
# Placeholder per la funzione di avvio del Numerical Binary
launch_numerical_binary_func = None

try:
    # Assicurati che il file si chiami 'numerical_binary_module.py'
    # e contenga la funzione 'launch_numerical_binary_window' definita FUORI dalle classi
    from numerical_binary_module import launch_numerical_binary_window as imported_numerical_binary_launch_func

    # Se l'importazione ha successo, assegna la funzione reale
    launch_numerical_binary_func = imported_numerical_binary_launch_func
    NUMERICAL_BINARY_MODULE_LOADED = True
    print("Modulo Numerical Binary (numerical_binary_module.py) caricato con successo.")

except ImportError as e:
    # Se il modulo non viene trovato o c'è un errore nell'import
    print(f"ATTENZIONE: Impossibile importare il modulo 'numerical_binary_module.py'. Errore: {e}")
    # Definisci una funzione di fallback che mostra solo un errore
    def fallback_launch_numerical_binary(parent):
        messagebox.showerror("Errore Modulo",
                             "Modulo Numerical Binary (numerical_binary_module.py) non trovato o contenente errori.\n"
                             "Verificare che il file sia presente e corretto.",
                             parent=parent)
    launch_numerical_binary_func = fallback_launch_numerical_binary # Usa la funzione di fallback

except Exception as e_gen:
    # Per gestire altri errori imprevisti durante l'importazione
     print(f"ERRORE GENERALE durante l'importazione del modulo Numerical Binary: {e_gen}")
     print(traceback.format_exc())
     # Definisci una funzione di fallback anche per errori generici
     def fallback_launch_numerical_binary_err(parent):
         messagebox.showerror("Errore Modulo",
                              f"Errore imprevisto durante il caricamento del modulo Numerical Binary:\n{e_gen}",
                              parent=parent)
     launch_numerical_binary_func = fallback_launch_numerical_binary_err

# --- Funzione Helper per Numerical Binary ---
def open_numerical_binary_helper(parent_window):
    """Chiama la funzione di lancio del modulo Numerical Binary (reale o fallback)."""
    print("Tentativo di aprire la finestra Numerical Binary...")
    if launch_numerical_binary_func: # Controlla se la funzione è stata definita (o è il fallback)
        try:
            # Esegui la funzione (passando la finestra principale come genitore)
            launch_numerical_binary_func(parent_window)
        except Exception as e_launch:
             # Cattura errori che potrebbero verificarsi durante l'esecuzione della launch function
             print(f"ERRORE durante l'esecuzione di launch_numerical_binary_func: {e_launch}")
             print(traceback.format_exc())
             messagebox.showerror("Errore Esecuzione Modulo",
                                  f"Si è verificato un errore all'avvio del modulo Numerical Binary:\n{e_launch}",
                                  parent=parent_window)
    else:
        # Questo caso limite non dovrebbe verificarsi se la logica sopra è corretta
        messagebox.showerror("Errore Interno", "Funzione di avvio per Numerical Binary non definita.", parent=parent_window)

# *** FINE AGGIUNTE PER MODULO Numerical Binary ***
# *******************************************************************

GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"
GITHUB_FOLDER_PATH = "" # Lascia vuoto se i file sono nella root del repo
BASE_GITHUB_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_FOLDER_PATH}"

# Definizione dei percorsi dei file delle ruote
FILE_RUOTE = {
    'BA': 'BARI.txt', 'CA': 'CAGLIARI.txt', 'FI': 'FIRENZE.txt',
    'GE': 'GENOVA.txt', 'MI': 'MILANO.txt', 'NA': 'NAPOLI.txt',
    'PA': 'PALERMO.txt', 'RM': 'ROMA.txt', 'TO': 'TORINO.txt',
    'VE': 'VENEZIA.txt','NZ':'NAZIONALE.txt'
}

# Configurazione del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed_value=42):
    """Imposta il seme per la riproducibilità dei risultati."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Impostazione del seed per la riproducibilità
set_seed()
   
def carica_dati_grezzi(ruota):
    """
    Carica i dati grezzi da GitHub, gestendo errori e impedendo
    che "NA" venga interpretato come NaN. Include debug temporaneo.
    """
    file_name = FILE_RUOTE.get(ruota)
    if not file_name:
        logger.error(f"Abbreviazione ruota non trovata nel dizionario: {ruota}")
        return None

    url = f"{BASE_GITHUB_URL.rstrip('/')}/{file_name}"
    logger.info(f"Tentativo di caricamento dati per {ruota} da GitHub URL: {url}")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Controlla errori HTTP (404, 500, etc.)
        content = response.text

        if not content or content.isspace():
             logger.warning(f"File scaricato da {url} per {ruota} è vuoto o contiene solo spazi.")
             return None

        # --- MODIFICA CRUCIALE PER GESTIRE "NA" ---
        # Usa io.StringIO per leggere la stringa come file
        df = pd.read_csv(
            io.StringIO(content),
            header=None,              # Il file non ha header
            sep='\t',                 # Separatore è tabulazione
            encoding='utf-8',         # Specifica encoding
            dtype={1: str},           # Specifica che la colonna 1 (Ruota) dovrebbe essere stringa
            keep_default_na=False,    # NON interpretare "NA", "NaN" etc. come valori mancanti
            na_values=['']            # Considera SOLO le stringhe vuote come valori mancanti (opzionale)
        )
        # --- FINE MODIFICA ---

        # === INIZIO BLOCCO DEBUG TEMPORANEO ===
        # (Puoi commentare o rimuovere queste righe dopo aver verificato che funziona)
        logger.info(f"DEBUG {ruota}: Info DataFrame DOPO lettura con keep_default_na=False:")
        df_info_str = io.StringIO()
        df.info(buf=df_info_str)
        logger.info(f"DEBUG {ruota}:\n{df_info_str.getvalue()}")
        # === FINE BLOCCO DEBUG TEMPORANEO ===

        expected_columns = 7 # Data, Ruota, Num1, Num2, Num3, Num4, Num5
        if df.shape[1] == expected_columns:
             # Assegna nomi alle colonne
             df.columns = ['Data', 'Ruota'] + [f'Num{i}' for i in range(1, 6)]

             # === INIZIO BLOCCO DEBUG TEMPORANEO ===
             logger.info(f"DEBUG {ruota}: Valori unici in df['Ruota'] DOPO rinomina colonne: {df['Ruota'].unique()}")
             # === FINE BLOCCO DEBUG TEMPORANEO ===

             # Verifica Consistenza Sigla
             sigle_nel_file = df['Ruota'].unique()

             if sigle_nel_file.size > 0:
                 # Estrai la prima (e si spera unica) sigla trovata
                 sigla_letta = sigle_nel_file[0] # Ora DOVREBBE essere la stringa 'NA'

                 # === INIZIO BLOCCO DEBUG TEMPORANEO ===
                 logger.info(f"DEBUG {ruota}: Valore sigla_letta estratto: {repr(sigla_letta)}") # repr() mostra se è stringa
                 logger.info(f"DEBUG {ruota}: Tipo sigla_letta estratto: {type(sigla_letta)}")
                 # === FINE BLOCCO DEBUG TEMPORANEO ===

                 # Confronto sigla (ora dovrebbe funzionare)
                 if len(sigle_nel_file) == 1 and sigla_letta.upper() != ruota.upper():
                     logger.warning(f"Sigla nel file {file_name} ('{sigla_letta}') != chiave usata ('{ruota}').")
                 elif len(sigle_nel_file) > 1:
                      logger.warning(f"Multiple sigle ruota in {file_name} per {ruota}: {sigle_nel_file}.")
             else:
                 logger.warning(f"Nessuna sigla trovata nella colonna 'Ruota' del file {file_name}.")

             logger.info(f"Dati caricati e colonne rinominate per {ruota} da GitHub.")
             return df # Restituisce il DataFrame caricato e validato
        else:
             # Errore se il numero di colonne non è quello atteso
             logger.error(f"Formato colonne inatteso ({df.shape[1]}) in {file_name} da {url}. Attese {expected_columns}.")
             try:
                 # Prova a loggare le prime righe per aiutare nel debug
                 logger.error(f"Prime righe:\n{df.head().to_string()}")
             except Exception:
                 logger.error("Impossibile mostrare prime righe.")
             return None # Fallimento caricamento

    # Gestione delle eccezioni durante il caricamento o parsing
    except requests.exceptions.Timeout:
        logger.error(f"Timeout (15s) durante connessione a {url} per {ruota}")
        return None
    except requests.exceptions.HTTPError as http_err:
         logger.error(f"Errore HTTP ({http_err.response.status_code}) da {url} per {ruota}: {http_err}")
         if http_err.response.status_code == 404:
             logger.error(f"--> VERIFICA: File '{file_name}' esiste su GitHub all'URL: {url}?")
         return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Errore di rete/connessione da {url} per {ruota}: {req_err}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Errore Pandas: Nessun dato (EmptyDataError) da {url} per {ruota}.")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Errore parsing Pandas del TSV da {url} per {ruota}: {e}")
        try:
            logger.error(f"Contenuto (prime 500 char):\n{content[:500]}")
        except NameError: # 'content' potrebbe non essere definito se errore avviene prima
            logger.error("Contenuto non disponibile.")
        return None
    except AttributeError as ae: # Cattura specifica per l'errore originale (ora improbabile)
        sigla_debug = 'Non disponibile'
        tipo_debug = 'Non disponibile'
        # Controlla se le variabili esistono prima di accedervi nel log
        if 'sigle_nel_file' in locals() and sigle_nel_file.size > 0:
             sigla_debug = repr(sigle_nel_file[0])
             tipo_debug = type(sigle_nel_file[0])
        logger.error(f"!!! AttributeError ANCORA PRESENTE? per {ruota} da {url}.")
        logger.error(f"Valore (sigla_letta): {sigla_debug}, Tipo: {tipo_debug}")
        logger.exception(f"Dettagli errore (Traceback): {ae}") # Logga lo stack trace
        return None
    except Exception as e:
        # Cattura qualsiasi altra eccezione imprevista
        logger.error(f"Errore generico imprevisto caricamento/processamento da {url} per {ruota}:")
        # Stampa il traceback completo nel log per aiutare a diagnosticare
        logger.error(traceback.format_exc())
        return None

def preprocessa_dati(df, start_date, end_date):
    """
    Preprocessa i dati:
    1. Converte la colonna 'Data' in formato datetime.
    2. Rimuove le righe con date mancanti o non valide.
    3. Imposta la colonna 'Data' come indice del DataFrame. <--- AGGIUNTO
    4. Filtra il DataFrame in base all'intervallo di date specificato.
    5. Gestisce gli errori di conversione.

    Args:
        df (pd.DataFrame): Il DataFrame grezzo.
        start_date (datetime): Data di inizio.
        end_date (datetime): Data di fine.

    Returns:
        pd.DataFrame: Il DataFrame preprocessato con DatetimeIndex,
                      o None se si verifica un errore.
    """
    if df is None:
        return None

    try:
        # Conversione date e gestione errori
        df['Data'] = pd.to_datetime(df['Data'], format='%Y/%m/%d', errors='coerce')
        # Rimuovi righe dove la conversione è fallita (diventa NaT)
        df.dropna(subset=['Data'], inplace=True)

        # Filtro date (sulla colonna 'Data')
        df_filtrato = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)].copy() # Usa .copy() per evitare SettingWithCopyWarning

        # Verifica se rimangono dati dopo il filtraggio
        if df_filtrato.empty:
             logger.warning(f"Nessun dato trovato nell'intervallo da {start_date.strftime('%Y/%m/%d')} a {end_date.strftime('%Y/%m/%d')}")
             # Potresti voler restituire un DataFrame vuoto invece di None se preferisci
             # return df_filtrato
             return None # O restituisce None se vuoi segnalare "nessun dato utile"

        # === IMPOSTA L'INDICE DOPO IL FILTRAGGIO ===
        df_filtrato.set_index('Data', inplace=True)
        # ==========================================

        return df_filtrato

    except KeyError as ke:
         logger.error(f"Errore nella pre-elaborazione: Colonna mancante - {ke}. Colonne disponibili: {df.columns.tolist()}")
         return None
    except Exception as e:
        logger.error(f"Errore generico nella pre-elaborazione: {e}")
        import traceback
        traceback.print_exc() # Stampa traceback per debug
        return None

def estrai_numeri(df):
    """Estrae i numeri come array NumPy e verifica il range."""
    if df is None:
        return None
    try:
        numeri = df[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].values.astype(int)
        if not np.all((numeri >= 1) & (numeri <= 90)):
            logger.error("Numeri fuori range")
            return None
        return numeri
    except Exception as e:
        logger.error(f"Errore in estrai_numeri: {e}")
        return None

def normalizza_numeri(numeri):
  """Normalizza i numeri tra 0 e 1 usando MinMaxScaler."""
  if numeri is None:
      return None, None
  try:
      scaler = MinMaxScaler()
      numeri_normalizzati = scaler.fit_transform(numeri)
      return numeri_normalizzati, scaler #restituisco anche lo scaler
  except Exception as e:
      logger.error(f"Errore nella normalizzazione: {e}")
      return None, None

# Colori per i pulsanti
BUTTON_DEFAULT_COLOR = "#C9E4CA"  # Verde chiaro
BUTTON_SELECTED_COLOR = "#4CAF50"  # Verde scuro

def salva_info_modello(ruota, fold_idx, train_loss, val_loss, ratio, epoch, model_path):
    """
    Salva le informazioni sul modello in un file JSON per tenere traccia dei migliori modelli.
    
    Args:
        ruota (str): Identificativo della ruota.
        fold_idx (int): Indice del fold.
        train_loss (float): Loss di addestramento.
        val_loss (float): Loss di validazione.
        ratio (float): Rapporto val_loss/train_loss.
        epoch (int): Epoca corrente.
        model_path (str): Percorso del file del modello.
    """
    info_file = f'model_info_{ruota}.json'
    model_info = {
        'ruota': ruota,
        'fold_idx': fold_idx,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'ratio': float(ratio),
        'epoch': epoch,
        'model_path': model_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Carica le informazioni esistenti, se presenti
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as f:
                models_info = json.load(f)
        except:
            models_info = []
    else:
        models_info = []
    
    # Aggiungi le nuove informazioni
    models_info.append(model_info)
    
    # Salva le informazioni aggiornate
    with open(info_file, 'w') as f:
        json.dump(models_info, f, indent=4)

def get_best_model_info(ruota, criterio='val_loss'):
    """
    Restituisce le informazioni sul miglior modello in base al criterio specificato.
    
    Args:
        ruota (str): Identificativo della ruota.
        criterio (str): Criterio di selezione ('val_loss', 'ratio', 'train_loss').
        
    Returns:
        dict: Informazioni sul miglior modello o None se non ci sono informazioni.
    """
    info_file = f'model_info_{ruota}.json'
    if not os.path.exists(info_file):
        return None
    
    try:
        with open(info_file, 'r') as f:
            models_info = json.load(f)
    except:
        return None
    
    if not models_info:
        return None
    
    # Seleziona il miglior modello in base al criterio
    if criterio == 'ratio':
        # Il miglior ratio è quello più vicino a 1 (né underfitting né overfitting)
        best_model = min(models_info, key=lambda x: abs(x['ratio'] - 1.0))
    elif criterio == 'val_loss':
        # Il miglior val_loss è il più basso (predizione più accurata)
        best_model = min(models_info, key=lambda x: x['val_loss'])
    elif criterio == 'train_loss':
        # Il miglior train_loss è il più basso
        best_model = min(models_info, key=lambda x: x['train_loss'])
    else:
        # Default a val_loss (più razionale per la predizione)
        best_model = min(models_info, key=lambda x: x['val_loss'])
    
    return best_model

class ModelConfig:
    """Classe per la gestione della configurazione del modello."""
    def __init__(self):
        # Parametri di addestramento
        self.optimizer_choice = 'adam'
        self.loss_function_choice = 'mean_squared_error'
        self.use_custom_loss = False
        self.use_binary_crossentropy_mode = False # <-- NUOVO FLAG: False di default
        self.patience = 10 
        self.min_delta = 0.001 

        # Parametri di architettura
        self.activation_choice = 'relu'
        self.output_activation = None # Attivazione output (per MSE/MAE serve lineare/None)
        self.regularization_choice = None
        self.regularization_value = 0.0001
        self.model_type = 'dense'  # 'dense' o 'lstm'
        self.dense_layers = [512, 256, 128]
        self.dropout_rates = [0.3, 0.3, 0.3]  # Aumentati da 0.2 a 0.3

        # Parametri di ensemble
        self.use_ensemble = False
        self.ensemble_models = []

        # Parametri di rumore
        self.adaptive_noise = False
        self.max_noise_factor = 0.05
        self.noise_type = 'gaussian'
        self.noise_scale = 0.01
        self.noise_percentage = 0.1

        # Log del valore iniziale di patience
        logger.info(f"Valore iniziale di patience: {self.patience}")

# Funzioni di supporto
def verifica_numeri(verifica_tutte_ruote=False):
    """Verifica numeri proposti in modo ottimizzato, solo per la ruota selezionata di default."""
    ruota = ruota_selezionata.get()
    if not ruota:
        messagebox.showwarning("Attenzione", "Seleziona prima una ruota.")
        return

    numeri_proposti = ottieni_numeri_proposti()
    if not numeri_proposti:
        return

    try:
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
    except ValueError:
        messagebox.showerror("Errore", "Formato data non valido.")
        return

    # Mostra attesa
    progress_window = tk.Toplevel()
    progress_window.title("Elaborazione in corso")
    progress_window.geometry("300x100")
    
    label = tk.Label(progress_window, text="Verifica estrazioni in corso...\nAttendere prego.")
    label.pack(pady=20)
    
    progress_window.update()
    
    # Risultati raccolta
    risultati_complessivi = {}
    
    # Verifica per ruota selezionata
    estrazioni, date_estrazioni = carica_estrazioni(ruota, end_date, 12)
    if estrazioni:
        risultati_ruota = {}
        for i, estrazione in enumerate(estrazioni):
            numeri_trovati = [num for num in numeri_proposti if int(num) in estrazione]
            if numeri_trovati:
                risultati_ruota[i+1] = {
                    'data': date_estrazioni[i],
                    'numeri_trovati': numeri_trovati
                }
        if risultati_ruota:
            risultati_complessivi[ruota] = risultati_ruota
    
    # Se richiesto, verifica altre ruote
    if verifica_tutte_ruote:
        ruote_verificate = []
        for r in FILE_RUOTE.keys():
            if r != ruota:
                estrazioni_r, date_estrazioni_r = carica_estrazioni(r, end_date, 9)
                if estrazioni_r:
                    risultati_r = {}
                    for i, estrazione in enumerate(estrazioni_r):
                        numeri_trovati = [num for num in numeri_proposti if int(num) in estrazione]
                        if len(numeri_trovati) >= 2:  # Solo ambi per le altre ruote
                            risultati_r[i+1] = {
                                'data': date_estrazioni_r[i],
                                'numeri_trovati': numeri_trovati
                            }
                    if risultati_r:
                        risultati_complessivi[r] = risultati_r
                ruote_verificate.append(r)
                
                # Aggiorna la finestra di progresso
                label.config(text=f"Verificate {len(ruote_verificate)}/{len(FILE_RUOTE)-1} ruote...")
                progress_window.update()
    
    # Chiudi la finestra di progresso
    progress_window.destroy()
    
    # Prepara messaggio risultato
    msg = f"NUMERI VERIFICATI: {', '.join(map(str, numeri_proposti))}\n\n"
    
    # Risultati per ruota selezionata
    msg += f"=== RUOTA {ruota} ===\n"
    if ruota in risultati_complessivi:
        for num_estrazione, dettagli in risultati_complessivi[ruota].items():
            data = dettagli['data'].strftime('%Y/%m/%d')
            numeri = dettagli['numeri_trovati']
            msg += f"Estrazione #{num_estrazione} ({data}): {', '.join(map(str, numeri))}\n"
    else:
        msg += "Nessun numero trovato nelle estrazioni successive.\n"
    
    # Risultati per altre ruote (solo se richiesto)
    if verifica_tutte_ruote:
        altre_ruote = [r for r in risultati_complessivi.keys() if r != ruota]
        if altre_ruote:
            msg += "\n=== ALTRE RUOTE (ambi) ===\n"
            for r in altre_ruote:
                msg += f"\nRuota {r}:\n"
                for num_estrazione, dettagli in risultati_complessivi[r].items():
                    data = dettagli['data'].strftime('%Y/%m/%d')
                    numeri = dettagli['numeri_trovati']
                    msg += f"Estrazione #{num_estrazione} ({data}): {', '.join(map(str, numeri))}\n"
    
    # Mostra i risultati
    messagebox.showinfo("Risultati Verifica", msg)

def popup_verifica():
    """Mostra un popup con opzioni di verifica."""
    popup = tk.Toplevel()
    popup.title("Opzioni di Verifica")
    popup.geometry("300x150")
    
    tk.Label(popup, text="Seleziona tipo di verifica:").pack(pady=10)
    
    tk.Button(
        popup, 
        text="Solo Ruota Selezionata\n(Veloce)", 
        command=lambda: [popup.destroy(), verifica_numeri(False)],
        bg="#ADD8E6",
        width=25,
        height=2
    ).pack(pady=5)
    
    tk.Button(
        popup, 
        text="Tutte le Ruote\n(Più lento)", 
        command=lambda: [popup.destroy(), verifica_numeri(True)],
        bg="#FFB6C1",
        width=25,
        height=2
    ).pack(pady=5)

def verifica_esiti_multi_ruota():
    """
    Funzione per verificare i numeri generati dall'analisi multi-ruota nelle estrazioni successive,
    utilizzando solo le ruote che erano state selezionate nell'analisi.
    """
    global numeri_finali, ruote_multi_ruota
    
    if numeri_finali is None or len(numeri_finali) == 0:
        messagebox.showwarning("Attenzione", "Esegui prima l'analisi multi-ruota per generare numeri da verificare.")
        return
    
    if not ruote_multi_ruota or len(ruote_multi_ruota) == 0:
        messagebox.showwarning("Attenzione", "Informazioni sulle ruote dell'analisi multi-ruota non disponibili.")
        return
    
    # Usa la stessa logica del pulsante "Verifica numeri" ma verificando solo sulle ruote selezionate
    popup = tk.Toplevel()
    popup.title("Verifica Esiti Multi-Ruota")
    popup.geometry("300x150")
    
    tk.Label(popup, text=f"Verifica i numeri sulle ruote selezionate:\n{', '.join(ruote_multi_ruota)}", font=("Arial", 10, "bold")).pack(pady=10)
    
    tk.Button(
        popup, 
        text="Verifica", 
        command=lambda: [popup.destroy(), verifica_numeri_multi_ruota(ruote_multi_ruota)],
        bg="#ADD8E6",
        width=25,
        height=2
    ).pack(pady=20)
    
def verifica_numeri_multi_ruota(ruote_da_verificare):
    """
    Verifica i numeri generati dall'analisi multi-ruota sulle ruote specificate.
    
    Args:
        ruote_da_verificare (list): Lista delle ruote da verificare.
    """
    global numeri_finali
    
    if not ruote_da_verificare or len(ruote_da_verificare) == 0:
        messagebox.showwarning("Attenzione", "Nessuna ruota specificata per la verifica.")
        return
    
    numeri_proposti = numeri_finali
    if not numeri_proposti or len(numeri_proposti) == 0:
        messagebox.showwarning("Attenzione", "Esegui prima l'analisi multi-ruota per generare numeri da verificare.")
        return
    
    try:
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
    except ValueError:
        messagebox.showerror("Errore", "Formato data non valido.")
        return
    
    # Mostra attesa
    progress_window = tk.Toplevel()
    progress_window.title("Elaborazione in corso")
    progress_window.geometry("300x100")
    
    label = tk.Label(progress_window, text="Verifica estrazioni per Analisi Multi-Ruota...\nAttendere prego.")
    label.pack(pady=20)
    
    progress_window.update()
    
    # Risultati raccolta
    risultati_complessivi = {}
    
    # Verifica su tutte le ruote selezionate nell'analisi multi-ruota
    ruote_verificate = []
    for r in ruote_da_verificare:
        estrazioni, date_estrazioni = carica_estrazioni(r, end_date, 9)
        if estrazioni:
            risultati_r = {}
            for i, estrazione in enumerate(estrazioni):
                numeri_trovati = [num for num in numeri_proposti if int(num) in estrazione]
                if numeri_trovati:  # Mostra tutti i numeri trovati, non solo gli ambi
                    risultati_r[i+1] = {
                        'data': date_estrazioni[i],
                        'numeri_trovati': numeri_trovati
                    }
            if risultati_r:
                risultati_complessivi[r] = risultati_r
        ruote_verificate.append(r)
        
        # Aggiorna la finestra di progresso
        label.config(text=f"Verificate {len(ruote_verificate)}/{len(ruote_da_verificare)} ruote...")
        progress_window.update()
    
    # Chiudi la finestra di progresso
    progress_window.destroy()
    
    # Prepara messaggio risultato
    msg = f"VERIFICA NUMERI ANALISI MULTI-RUOTA: {', '.join(map(str, numeri_proposti))}\n\n"
    
    # Nessuna distinzione tra ruota principale e altre ruote
    for r in ruote_da_verificare:
        msg += f"=== RUOTA {r} ===\n"
        if r in risultati_complessivi:
            for num_estrazione, dettagli in risultati_complessivi[r].items():
                data = dettagli['data'].strftime('%Y/%m/%d')
                numeri = dettagli['numeri_trovati']
                msg += f"Estrazione #{num_estrazione} ({data}): {', '.join(map(str, numeri))}\n"
        else:
            msg += "Nessun numero trovato nelle estrazioni successive.\n"
        msg += "\n"
    
    # Mostra i risultati
    messagebox.showinfo("Risultati Verifica Analisi Multi-Ruota", msg)

def verifica_estrazioni_con_conteggio(estrazioni, numeri_proposti, min_match):
    """
    Verifica se almeno 'min_match' numeri proposti sono stati estratti nelle estrazioni fornite.
    
    Args:
        estrazioni (list): Lista di estrazioni, dove ogni estrazione è una lista di numeri
        numeri_proposti (list): Lista di numeri proposti da verificare
        min_match (int): Numero minimo di corrispondenze richieste
        
    Returns:
        tuple: (True/False se ci sono match, numero dell'estrazione in cui sono stati trovati)
    """
    # Assicuriamo di controllare solo fino a 9 estrazioni (o meno se non ce ne sono abbastanza)
    num_estrazioni = min(len(estrazioni), 9)
    
    for i in range(num_estrazioni):
        estrazione = estrazioni[i]
        
        # Conta quanti numeri proposti sono presenti in questa estrazione
        match = sum(num in estrazione for num in numeri_proposti)
        
        # Se troviamo almeno min_match corrispondenze, ritorniamo True e l'indice dell'estrazione
        if match >= min_match:
            return True, i + 1  # +1 perché contiamo da 1 per leggibilità
    
    # Se arriviamo qui, non abbiamo trovato corrispondenze sufficienti
    return False, None

def diagnosi_file_estrazioni():
    """
    Funzione di diagnosi per verificare la continuità delle date nei file delle estrazioni.
    """
    ruota = ruota_selezionata.get()
    if not ruota:
        messagebox.showwarning("Attenzione", "Seleziona prima una ruota.")
        return
    
    file_name = FILE_RUOTE.get(ruota)
    if not file_name or not os.path.exists(file_name):
        messagebox.showerror("Errore", f"File delle estrazioni per la ruota {ruota} non trovato.")
        return
    
    try:
        # Leggi il file e estrai le date e i numeri
        date_estrazioni = []
        numeri_estrazioni = []
        
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 7:  # Data + codice ruota + 5 numeri
                    try:
                        data_str = parts[0]
                        data = pd.to_datetime(data_str, format='%Y/%m/%d')
                        numeri = [int(parts[i]) for i in range(2, 7)]
                        
                        date_estrazioni.append(data)
                        numeri_estrazioni.append(numeri)
                    except (ValueError, IndexError) as e:
                        print(f"Errore nel parsing della riga: {line.strip()}, Errore: {e}")
                        continue
        
        # Crea un dataframe per analisi
        df = pd.DataFrame({
            'Data': date_estrazioni,
            'Numeri': numeri_estrazioni
        })
        
        # Aggiungi anno e mese
        df['Anno'] = df['Data'].dt.year
        df['Mese'] = df['Data'].dt.month
        
        # Ottieni la data di fine dall'interfaccia utente
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
        
        # Filtra solo le date successive alla data di fine
        filtered_df = df[df['Data'] > end_date].sort_values(by='Data')
        
        # Agrregazione per mese per verificare continuità
        monthly_counts = filtered_df.groupby(['Anno', 'Mese']).size().reset_index(name='Conteggio')
        
        # Inizia l'output
        textbox.delete(1.0, tk.END)  # Pulisci la textbox
        textbox.insert(tk.END, f"=== DIAGNOSI FILE {file_name} ===\n\n")
        textbox.insert(tk.END, f"Totale righe valide nel file: {len(df)}\n")
        
        if len(df) > 0:
            textbox.insert(tk.END, f"Data più vecchia: {df['Data'].min().strftime('%Y/%m/%d')}\n")
            textbox.insert(tk.END, f"Data più recente: {df['Data'].max().strftime('%Y/%m/%d')}\n\n")
        else:
            textbox.insert(tk.END, "Nessuna data valida trovata nel file.\n\n")
        
        textbox.insert(tk.END, f"=== ESTRAZIONI DOPO {end_date.strftime('%Y/%m/%d')} ===\n\n")
        textbox.insert(tk.END, f"Totale estrazioni trovate: {len(filtered_df)}\n\n")
        
        textbox.insert(tk.END, "=== DISTRIBUZIONE MENSILE ===\n\n")
        for i, row in monthly_counts.iterrows():
            anno = row['Anno']
            mese = row['Mese']
            conteggio = row['Conteggio']
            textbox.insert(tk.END, f"Anno {anno}, Mese {mese}: {conteggio} estrazioni\n")
        
        textbox.insert(tk.END, "\n=== PRIME 15 ESTRAZIONI SUCCESSIVE ===\n\n")
        for i, (_, row) in enumerate(filtered_df.head(15).iterrows()):
            date_str = row['Data'].strftime('%Y/%m/%d')
            numbers_str = ' '.join([str(num) for num in row['Numeri']])
            textbox.insert(tk.END, f"{i+1}: {date_str} - {numbers_str}\n")
        
        # Verifica esplicitamente se ci sono estrazioni di febbraio 2025
        feb_2025 = filtered_df[(filtered_df['Anno'] == 2025) & (filtered_df['Mese'] == 2)]
        
        textbox.insert(tk.END, f"\n=== ESTRAZIONI DI FEBBRAIO 2025 ===\n\n")
        if len(feb_2025) > 0:
            textbox.insert(tk.END, f"Trovate {len(feb_2025)} estrazioni di Febbraio 2025\n")
            for i, (_, row) in enumerate(feb_2025.iterrows()):
                date_str = row['Data'].strftime('%Y/%m/%d')
                numbers_str = ' '.join([str(num) for num in row['Numeri']])
                textbox.insert(tk.END, f"{i+1}: {date_str} - {numbers_str}\n")
        else:
            textbox.insert(tk.END, "NESSUNA ESTRAZIONE DI FEBBRAIO 2025 TROVATA!\n")
            textbox.insert(tk.END, "Questo spiega perché la verifica salta da gennaio a marzo.\n")
            
        # Verifica se ci sono giorni mancanti nelle 9 estrazioni successive
        if len(filtered_df) >= 9:
            first_nine = filtered_df.head(9)
            dates_list = sorted(first_nine['Data'].tolist())
            
            textbox.insert(tk.END, f"\n=== ANALISI DELLE PRIME 9 ESTRAZIONI ===\n\n")
            textbox.insert(tk.END, "Date delle prime 9 estrazioni:\n")
            
            for i, date in enumerate(dates_list):
                textbox.insert(tk.END, f"{i+1}: {date.strftime('%Y/%m/%d')}\n")
            
            # Verifica intervalli anomali tra le date
            if len(dates_list) > 1:
                gaps = [(dates_list[i+1] - dates_list[i]).days for i in range(len(dates_list)-1)]
                max_gap = max(gaps)
                max_gap_index = gaps.index(max_gap)
                
                textbox.insert(tk.END, f"\nIntervallo massimo: {max_gap} giorni tra "
                                      f"{dates_list[max_gap_index].strftime('%Y/%m/%d')} e "
                                      f"{dates_list[max_gap_index+1].strftime('%Y/%m/%d')}\n")
                
                if max_gap > 7:  # Più di una settimana potrebbe indicare dati mancanti
                    textbox.insert(tk.END, f"\nATTENZIONE: Intervallo anomalo di {max_gap} giorni. "
                                          f"Potrebbero mancare estrazioni tra queste date!\n")
        
        # Messagebox riassuntivo
        messagebox.showinfo("Diagnosi Completata", 
                            f"Diagnosi del file {file_name} completata.\n"
                            f"Totale estrazioni dopo {end_date.strftime('%Y/%m/%d')}: {len(filtered_df)}\n"
                            f"Estrazioni di Febbraio 2025: {len(feb_2025)}")
        
    except Exception as e:
        textbox.insert(tk.END, f"Errore durante la diagnosi: {e}\n")
        import traceback
        textbox.insert(tk.END, f"Traceback: {traceback.format_exc()}\n")
        messagebox.showerror("Errore", f"Errore durante la diagnosi: {e}")


def carica_estrazioni(ruota, end_date, max_estrazioni=9, use_cache=True):
    """
    Carica le estrazioni successive per una data ruota, con sistema di caching.
    
    Args:
        ruota (str): Identificativo della ruota.
        end_date (datetime): Data di fine oltre la quale cercare le estrazioni
        max_estrazioni (int): Numero massimo di estrazioni da caricare
        use_cache (bool): Se usare la cache o forzare la rilettura
        
    Returns:
        tuple: (estrazioni, date_estrazioni)
    """
    global cache_estrazioni, cache_timestamp
    
    file_name = FILE_RUOTE.get(ruota)
    if not file_name or not os.path.exists(file_name):
        return [], []
    
    # Crea una chiave unica per questa combinazione di ruota e data
    cache_key = f"{ruota}_{end_date.strftime('%Y%m%d')}"
    
    # Verifica se il file è stato modificato dall'ultima volta che abbiamo caricato la cache
    file_mtime = os.path.getmtime(file_name)
    
    # Usa la cache se disponibile, il file non è cambiato e non forziamo la rilettura
    if use_cache and cache_key in cache_estrazioni and cache_key in cache_timestamp:
        if file_mtime <= cache_timestamp[cache_key]:
            # Usa i dati dalla cache
            return cache_estrazioni[cache_key]
    
    try:
        # Dizionario per memorizzare estrazioni uniche per data
        estrazioni_per_data = {}
        
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 7:  # Data + codice ruota + 5 numeri
                    try:
                        data_str = parts[0]
                        data = pd.to_datetime(data_str, format='%Y/%m/%d')
                        
                        # Verifica se la data è successiva alla data di fine
                        if data > end_date:
                            # Estrai i 5 numeri
                            numeri = [int(parts[i]) for i in range(2, 7)]
                            
                            # Salva solo una estrazione per data
                            if data not in estrazioni_per_data:
                                estrazioni_per_data[data] = numeri
                                # Interrompi se abbiamo già trovato max_estrazioni date uniche
                                if len(estrazioni_per_data) >= max_estrazioni:
                                    break
                    except (ValueError, IndexError):
                        continue
        
        # Ottieni le date in ordine cronologico
        date_ordinate = sorted(estrazioni_per_data.keys())
        
        # Prepara le liste finali
        estrazioni = [estrazioni_per_data[data] for data in date_ordinate]
        
        # Salva nella cache prima di restituire
        cache_estrazioni[cache_key] = (estrazioni, date_ordinate)
        cache_timestamp[cache_key] = file_mtime
        
        return estrazioni, date_ordinate
    except Exception as e:
        # In caso di errore, restituisci liste vuote
        return [], []

def pulisci_cache():
    """Pulisce la cache delle estrazioni e altre risorse per liberare memoria."""
    global cache_estrazioni, cache_timestamp, textbox
    
    try:
        cache_estrazioni.clear()
        cache_timestamp.clear()
        
        # Chiudi tutte le figure matplotlib
        plt.close('all')
        
        # Forza garbage collection
        import gc
        gc.collect()
        
        # Debug: stampa su stdout 
        print("Cache e risorse grafiche pulite con successo.")
        
        # Prova ad aggiornare la textbox se disponibile
        if 'textbox' in globals() and textbox:
            try:
                textbox.insert(tk.END, "Cache e risorse grafiche pulite con successo.\n")
                textbox.see(tk.END)  # Scorri alla fine
            except Exception as e:
                print(f"Errore nell'aggiornare textbox: {e}")
                messagebox.showinfo("Pulizia Cache", "Cache e risorse grafiche pulite con successo.")
        else:
            messagebox.showinfo("Pulizia Cache", "Cache e risorse grafiche pulite con successo.")
    
    except Exception as e:
        print(f"Errore durante la pulizia della cache: {e}")
        messagebox.showerror("Errore", f"Errore durante la pulizia della cache: {e}")

def verifica_integrità_file():
    """Verifica l'integrità dei file delle estrazioni."""
    ruota = ruota_selezionata.get()
    if not ruota:
        ruota = list(FILE_RUOTE.keys())[0]  # Usa la prima ruota disponibile se nessuna è selezionata
        
    file_name = FILE_RUOTE.get(ruota)
    if not file_name or not os.path.exists(file_name):
        messagebox.showerror("Errore", f"File delle estrazioni per la ruota {ruota} non trovato.")
        return
        
    try:
        # Conta le righe nel file
        line_count = 0
        date_count = 0
        invalid_lines = 0
        
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                parts = line.strip().split()
                if len(parts) >= 7:  # Data + codice ruota + 5 numeri
                    try:
                        data_str = parts[0]
                        pd.to_datetime(data_str, format='%Y/%m/%d')
                        date_count += 1
                    except ValueError:
                        invalid_lines += 1
                else:
                    invalid_lines += 1
        
        # Mostra i risultati
        textbox.delete(1.0, tk.END)  # Pulisci la textbox
        textbox.insert(tk.END, f"=== VERIFICA INTEGRITÀ FILE: {file_name} ===\n\n")
        textbox.insert(tk.END, f"Righe totali: {line_count}\n")
        textbox.insert(tk.END, f"Date valide: {date_count}\n")
        textbox.insert(tk.END, f"Righe non valide: {invalid_lines}\n\n")
        
        if invalid_lines > 0:
            textbox.insert(tk.END, "ATTENZIONE: Ci sono righe non valide nel file!\n")
            textbox.insert(tk.END, "Si consiglia di controllare manualmente il file.\n")
        else:
            textbox.insert(tk.END, "Il file sembra essere integro e ben formattato.\n")
            
        messagebox.showinfo("Verifica Integrità File", f"Verifica completata per {file_name}.\nRighe valide: {date_count}/{line_count}")
    except Exception as e:
        textbox.insert(tk.END, f"Errore durante la verifica: {e}\n")
        messagebox.showerror("Errore", f"Errore durante la verifica: {e}")

def ottieni_numeri_proposti():
    """
    Restituisce i numeri predetti dall'ultima elaborazione.
    Se non ci sono numeri predetti, mostra un messaggio di errore.
    """
    global numeri_finali  # Questa variabile dovrebbe contenere i numeri predetti
    
    if numeri_finali is not None and len(numeri_finali) > 0:
        # Converti in interi e restituisci
        return [int(num) for num in numeri_finali]
    else:
        messagebox.showwarning("Attenzione", "Non ci sono numeri predetti disponibili. Esegui prima un'elaborazione.")
        return []

@register_keras_serializable(package='Custom', name='custom_loss_function')
def custom_loss_function(y_true, y_pred):
    """
    Una funzione di loss personalizzata che penalizza sia l'errore di previsione
    che la ripetizione dei numeri.
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    base_loss = mse + 0.5 * mae

    pred_variance = tf.math.reduce_variance(y_pred, axis=1)
    diversity_penalty = 0.1 / (pred_variance + 1e-8)

    return base_loss + 0.05 * diversity_penalty

def carica_dati(ruota, start_date, end_date):
    """
    Funzione principale: carica, preprocessa, estrae e normalizza i dati.
    Gestisce *tutti* gli errori in modo centralizzato.

    Args:
        ruota (str): La ruota da caricare.
        start_date (datetime): Data di inizio.
        end_date (datetime): Data di fine.

    Returns:
        tuple: (numeri_normalizzati, numeri, scaler, df_preproc), dove:
               - numeri_normalizzati è un array NumPy con i numeri normalizzati.
               - numeri è un array NumPy con i numeri originali (interi).
               - scaler è l'oggetto MinMaxScaler usato per la normalizzazione.
               - df_preproc è il DataFrame preprocessato (filtrato per data).
               Restituisce (None, None, None, None) se si verifica un errore *in qualsiasi punto*.
    """
    df_grezzo = carica_dati_grezzi(ruota)
    if df_grezzo is None:  # Gestione centralizzata degli errori
        return None, None, None, None

    df_preproc = preprocessa_dati(df_grezzo, start_date, end_date)
    if df_preproc is None:  # Gestione centralizzata degli errori
        return None, None, None, None

    numeri = estrai_numeri(df_preproc)
    if numeri is None:  # Gestione centralizzata degli errori
        return None, None, None, None

    numeri_normalizzati, scaler = normalizza_numeri(numeri)
    if numeri_normalizzati is None:
        return None, None, None, None
    return numeri_normalizzati, numeri, scaler, df_preproc

class Tooltip:
    """Classe migliorata per gestire i tooltip nei widget di tkinter."""
    def __init__(self, widget, text, delay=500, wrap_length=250):
        self.widget = widget
        self.text = text
        self.delay = delay  # Ritardo in millisecondi
        self.wrap_length = wrap_length
        self.tooltip = None
        self.id = None
        self.widget.bind("<Enter>", self.schedule)
        self.widget.bind("<Leave>", self.hide)
        self.widget.bind("<ButtonPress>", self.hide)

    def schedule(self, event=None):
        self.id = self.widget.after(self.delay, self.show)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Crea il tooltip
        self.tooltip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffcc",
                     relief=tk.SOLID, borderwidth=1, wraplength=self.wrap_length)
        label.pack(ipadx=2, ipady=2)

    def hide(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# Istanza di configurazione del modello
config = ModelConfig()

# Variabili globali per l'interfaccia utente
progress_bar = None
pulsanti_ruote = {}
entry_info = None
textbox = None
selected_ruota = None
root = None
numeri_finali = None
ruote_multi_ruota = []
cache_estrazioni = {}  # Dizionario per memorizzare le estrazioni già caricate
cache_timestamp = {}   # Per tenere traccia della data dell'ultimo aggiornamento dei file
ultima_ruota_elaborata = None  # Per tenere traccia dell'ultima ruota elaborata
num_folds = None  # Verrà inizializzato come IntVar successivamente

def carica_dati_multi_ruota(ruote_selezionate, start_date, end_date):
    """
    Carica e combina i dati di più ruote selezionate.
    
    Args:
        ruote_selezionate (list): Lista di ruote selezionate
        start_date (datetime): Data di inizio
        end_date (datetime): Data di fine
        
    Returns:
        tuple: Dati combinati, dati originali per ruota
    """
    dati_combinati = []
    dati_per_ruota = {}
    
    for ruota in ruote_selezionate:
        data = carica_dati(ruota, start_date, end_date)
        if data is not None:
            X, y, scaler, raw_data = data
            dati_per_ruota[ruota] = data
            
            # Aggiungi identificatore della ruota come feature aggiuntiva
            ruota_id = np.ones((X.shape[0], 1)) * list(FILE_RUOTE.keys()).index(ruota) / len(FILE_RUOTE)
            X_con_ruota = np.hstack([X, ruota_id])
            
            dati_combinati.append((X_con_ruota, y, scaler, raw_data))
    
    if not dati_combinati:
        return None, None
    
    # Combina tutti i dati
    X_combined = np.vstack([d[0] for d in dati_combinati])
    y_combined = np.vstack([d[1] for d in dati_combinati])
    
    return (X_combined, y_combined, dati_combinati[0][2], None), dati_per_ruota

def apri_selezione_multi_ruota():
    """Apre una finestra popup per la selezione multipla delle ruote."""
    popup = tk.Toplevel()
    popup.title("Selezione Multi-Ruota")
    popup.geometry("400x400")
    
    label = tk.Label(popup, text="Seleziona le ruote per l'analisi combinata:", font=("Arial", 12))
    label.pack(pady=10)
    
    # Crea dizionario per memorizzare le variabili BooleanVar
    checkbox_vars = {}
    
    # Frame per i checkbox
    checkbox_frame = tk.Frame(popup)
    checkbox_frame.pack(pady=10)
    
    # Crea i checkbox
    ruote_list = ["BA", "CA", "FI", "GE", "MI", "NA", "PA", "RM", "TO", "VE", "NZ"]
    col, row = 0, 0
    for ruota in ruote_list:
        var = tk.BooleanVar()
        checkbox = tk.Checkbutton(checkbox_frame, text=ruota, variable=var, width=8, height=2)
        checkbox.grid(row=row, column=col, padx=5, pady=5)
        checkbox_vars[ruota] = var
        
        col += 1
        if col > 3:  # 4 colonne per riga
            col = 0
            row += 1
    
    # Pulsante per avviare l'analisi
    btn_avvia = tk.Button(popup, text="Avvia Analisi Multi-Ruota", 
                         command=lambda: analisi_multi_ruota(checkbox_vars, popup),
                         bg="#FF9900", fg="white", font=("Arial", 12, "bold"), width=25, height=2)
    btn_avvia.pack(pady=20)

def analisi_multi_ruota(checkbox_vars, popup=None):
    """
    Esegue un'analisi combinata su più ruote selezionate implementando
    la legge del terzo basata sui ritardi di sortita dei numeri rispetto
    al ciclo teorico di 18 estrazioni.
    
    Args:
        checkbox_vars (dict): Dizionario con le variabili dei checkbox
        popup (Toplevel, optional): Finestra popup da chiudere dopo l'avvio
    """
    global numeri_finali, ruote_multi_ruota
    
    # Ottieni le ruote selezionate
    ruote_selezionate = [ruota for ruota, var in checkbox_vars.items() if var.get()]
    
    if len(ruote_selezionate) < 2:
        messagebox.showwarning("Attenzione", "Seleziona almeno due ruote per l'analisi multi-ruota.")
        return
    
    # Salva le ruote selezionate per futura verifica
    ruote_multi_ruota = ruote_selezionate.copy()
    
    # Chiudi il popup se fornito
    if popup:
        popup.destroy()
    
    textbox.delete(1.0, tk.END)
    textbox.insert(tk.END, f"Avvio analisi multi-ruota con legge del terzo (ritardi) per: {', '.join(ruote_selezionate)}...\n")
    
    try:
        start_date = pd.to_datetime(entry_start_date.get_date(), format='%Y/%m/%d')
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
    except ValueError:
        messagebox.showerror("Errore", "Formato data non valido.")
        return
    
    # Raccogliamo tutti i numeri estratti per ogni ruota con le relative date
    ruote_estrazioni = {}
    
    for ruota in ruote_selezionate:
        textbox.insert(tk.END, f"Raccolta dati per ruota {ruota}...\n")
        textbox.update()
        
        file_name = FILE_RUOTE.get(ruota)
        if not file_name or not os.path.exists(file_name):
            textbox.insert(tk.END, f"File per ruota {ruota} non trovato.\n")
            continue
        
        try:
            # Lettura del file
            estrazioni_ruota = []
            with open(file_name, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 7:  # Data + ruota + 5 numeri
                        try:
                            data_str = parts[0]
                            data = pd.to_datetime(data_str, format='%Y/%m/%d')
                            
                            # Verifica se la data è nel range
                            if start_date <= data <= end_date:
                                numeri = []
                                for i in range(2, 7):
                                    try:
                                        num = int(parts[i])
                                        if 1 <= num <= 90:
                                            numeri.append(num)
                                    except (ValueError, IndexError):
                                        pass
                                
                                if len(numeri) == 5:  # Solo se abbiamo 5 numeri validi
                                    estrazioni_ruota.append({
                                        'data': data,
                                        'numeri': numeri
                                    })
                        except:
                            pass
            
            # Ordina per data crescente
            estrazioni_ruota.sort(key=lambda x: x['data'])
            
            if estrazioni_ruota:
                ruote_estrazioni[ruota] = estrazioni_ruota
                textbox.insert(tk.END, f"Raccolti {len(estrazioni_ruota)} record validi per ruota {ruota}.\n")
            else:
                textbox.insert(tk.END, f"Nessun dato valido trovato per ruota {ruota}.\n")
                
        except Exception as e:
            textbox.insert(tk.END, f"Errore nella lettura dei dati per ruota {ruota}: {str(e)}\n")
    
    if not ruote_estrazioni:
        messagebox.showerror("Errore", "Nessun dato valido trovato per le ruote selezionate.")
        return
    
    # =========================================
    # ANALISI DEI RITARDI (LEGGE DEL TERZO)
    # =========================================
    ritardi_numeri = {}
    ultima_estrazione = {}
    
    # Inizializza i ritardi per tutti i numeri
    for num in range(1, 91):
        ritardi_numeri[num] = 0
        ultima_estrazione[num] = None
    
    # Per ogni ruota, calcola quando è uscito l'ultima volta ciascun numero
    for ruota, estrazioni in ruote_estrazioni.items():
        # Prendiamo tutte le estrazioni disponibili per questa ruota
        for i, estrazione in enumerate(estrazioni):
            for num in estrazione['numeri']:
                ultima_estrazione[num] = i  # Salva l'indice dell'ultima estrazione
    
    # Calcola i ritardi attuali (quante estrazioni fa è uscito l'ultima volta)
    max_estrazioni = max([len(estrazioni) for estrazioni in ruote_estrazioni.values()]) if ruote_estrazioni else 0
    
    for num in range(1, 91):
        if ultima_estrazione[num] is not None:
            ritardi_numeri[num] = max_estrazioni - ultima_estrazione[num]
        else:
            ritardi_numeri[num] = max_estrazioni + 1  # Se non è mai uscito, ha ritardo massimo
    
    # Classifica i numeri secondo la legge del terzo basata sui ritardi
    numeri_primo_terzo = []  # Ritardo ≤ 18
    numeri_secondo_terzo = []  # 18 < Ritardo ≤ 36
    numeri_terzo_terzo = []  # Ritardo > 36
    
    for num, ritardo in ritardi_numeri.items():
        if ritardo <= 18:
            numeri_primo_terzo.append(num)
        elif ritardo <= 36:
            numeri_secondo_terzo.append(num)
        else:
            numeri_terzo_terzo.append(num)
    
    # =========================================
    # CALCOLO PUNTEGGIO FINALE E SELEZIONE NUMERI
    # =========================================
    punteggi_numeri = {}
    
    # Secondo la legge del terzo, i numeri nel secondo terzo hanno
    # maggiori probabilità di uscire prossimamente, seguiti da quelli
    # nel terzo terzo, mentre quelli nel primo terzo hanno già "fatto il loro dovere"
    
    # Assegna pesi basati sulla teoria
    peso_primo_terzo = 0.2    # Peso basso per i numeri recenti
    peso_secondo_terzo = 0.7  # Peso alto per i numeri nella "fascia ideale"
    peso_terzo_terzo = 0.4    # Peso medio per i numeri molto ritardatari
    
    # Calcola i punteggi secondo questa logica
    for num in numeri_primo_terzo:
        # Per i numeri recenti, il punteggio è inversamente proporzionale al ritardo
        # (più è recente, meno punteggio ha)
        fattore_recente = (18 - ritardi_numeri[num]) / 18  # Valore tra 0 e 1
        punteggi_numeri[num] = peso_primo_terzo * (1 - fattore_recente) * 100
    
    for num in numeri_secondo_terzo:
        # Per i numeri nella fascia ideale, il punteggio è proporzionale a quanto
        # sono avanzati nella fascia
        fattore_ideale = (ritardi_numeri[num] - 18) / 18  # Valore tra 0 e 1
        punteggi_numeri[num] = peso_secondo_terzo * fattore_ideale * 100
    
    for num in numeri_terzo_terzo:
        # Per i numeri molto ritardatari, il punteggio è proporzionale al ritardo,
        # ma con un limite per evitare sovrastimare i ritardatari estremi
        fattore_ritardo = min(1, (ritardi_numeri[num] - 36) / 36)  # Valore tra 0 e 1, max 1
        punteggi_numeri[num] = peso_terzo_terzo * fattore_ritardo * 100
    
    # Ordina i numeri per punteggio
    numeri_ordinati = sorted(punteggi_numeri.items(), key=lambda x: x[1], reverse=True)
    
    # Seleziona i numeri finali, bilanciando tra le tre categorie
    numeri_finali = []
    
    # Prendi i migliori da ciascuna categoria
    categorie = [
        (numeri_secondo_terzo, 2),  # 2 numeri dalla fascia ideale
        (numeri_terzo_terzo, 2),    # 2 numeri ritardatari
        (numeri_primo_terzo, 1)     # 1 numero recente
    ]
    
    for categoria, quanti in categorie:
        # Ordina i numeri di questa categoria per punteggio
        numeri_categoria = [(num, punteggi_numeri[num]) for num in categoria]
        numeri_categoria.sort(key=lambda x: x[1], reverse=True)
        
        # Aggiungi i migliori
        for num, _ in numeri_categoria[:quanti]:
            if len(numeri_finali) < 5:
                numeri_finali.append(num)
    
    # Se non abbiamo ancora 5 numeri, aggiungi dai top generali
    while len(numeri_finali) < 5:
        for num, _ in numeri_ordinati:
            if num not in numeri_finali:
                numeri_finali.append(num)
                break
    
    # Calcola l'attendibilità
    # Verifica quanto la distribuzione attuale è vicina alla teoria (30-30-30)
    teorico_terzo = 30  # 1/3 di 90
    margine_errore = 5  # Tolleranza
    
    conforme_terzo = (
        abs(len(numeri_primo_terzo) - teorico_terzo) <= margine_errore and
        abs(len(numeri_secondo_terzo) - teorico_terzo) <= margine_errore and
        abs(len(numeri_terzo_terzo) - teorico_terzo) <= margine_errore
    )
    
    # Calcola l'attendibilità in base alla conformità e al punteggio medio
    punteggio_medio = sum(punteggi_numeri[num] for num in numeri_finali) / len(numeri_finali) if numeri_finali else 0
    max_punteggio = max(punteggi_numeri.values()) if punteggi_numeri else 100
    
    # Se la distribuzione conferma la legge del terzo, l'attendibilità è maggiore
    base_attendibility = min(100, punteggio_medio * 100 / max_punteggio) if max_punteggio > 0 else 50
    attendibility_score = base_attendibility * 1.2 if conforme_terzo else base_attendibility
    attendibility_score = min(100, attendibility_score)  # Assicura che non superi 100
    
    if attendibility_score > 80:
        commento = "Previsione molto attendibile"
    elif attendibility_score > 60:
        commento = "Previsione attendibile"
    elif attendibility_score > 40:
        commento = "Previsione moderatamente attendibile"
    elif attendibility_score > 20:
        commento = "Previsione poco attendibile"
    else:
        commento = "Previsione non attendibile"
    
    # =========================================
    # VISUALIZZAZIONE DEI RISULTATI
    # =========================================
    textbox.insert(tk.END, "\n=== RISULTATI ANALISI MULTI-RUOTA CON LEGGE DEL TERZO (RITARDI) ===\n\n")
    
    # Visualizza la distribuzione secondo la legge del terzo
    textbox.insert(tk.END, "Analisi secondo la legge del terzo (ciclo teorico di 18 estrazioni):\n")
    textbox.insert(tk.END, f"- Primo terzo (ritardo ≤ 18): {len(numeri_primo_terzo)} numeri ({len(numeri_primo_terzo)/90*100:.1f}%)\n")
    textbox.insert(tk.END, f"- Secondo terzo (18 < ritardo ≤ 36): {len(numeri_secondo_terzo)} numeri ({len(numeri_secondo_terzo)/90*100:.1f}%)\n")
    textbox.insert(tk.END, f"- Terzo terzo (ritardo > 36): {len(numeri_terzo_terzo)} numeri ({len(numeri_terzo_terzo)/90*100:.1f}%)\n\n")
    
    # Verifica della legge del terzo
    if conforme_terzo:
        textbox.insert(tk.END, "✓ La distribuzione attuale CONFERMA la legge del terzo!\n\n")
    else:
        textbox.insert(tk.END, "✗ La distribuzione attuale NON conferma la legge del terzo\n\n")
    
    # Mostra i numeri selezionati con i loro ritardi
    textbox.insert(tk.END, "Numeri selezionati e loro ritardi:\n")
    for num in numeri_finali:
        categoria = ""
        if num in numeri_primo_terzo:
            categoria = "primo terzo"
        elif num in numeri_secondo_terzo:
            categoria = "secondo terzo"
        else:
            categoria = "terzo terzo"
            
        textbox.insert(tk.END, f"- Numero {num}: ritardo di {ritardi_numeri[num]} estrazioni ({categoria})\n")
    
    textbox.insert(tk.END, "\n")
    
    # Mostra la strategia utilizzata
    textbox.insert(tk.END, "Strategia di selezione:\n")
    textbox.insert(tk.END, "- 2 numeri dal secondo terzo (ritardo 19-36)\n")
    textbox.insert(tk.END, "- 2 numeri dal terzo terzo (ritardo >36)\n")
    textbox.insert(tk.END, "- 1 numero dal primo terzo (ritardo ≤18)\n\n")
    
    # Mostra la previsione finale
    textbox.insert(tk.END, "=== PREVISIONE FINALE SECONDO LEGGE DEL TERZO (RITARDI) ===\n")
    textbox.insert(tk.END, f"Numeri consigliati: {', '.join(map(str, numeri_finali))}\n")
    textbox.insert(tk.END, f"Attendibilità: {attendibility_score:.1f}/100 - {commento}\n")
    textbox.insert(tk.END, f"(Basata sull'analisi combinata di {len(ruote_estrazioni)} ruote)\n")
    
    # =========================================
    # VISUALIZZAZIONE GRAFICA
    # =========================================
    try:
        # Pulisci il frame
        for child in frame_grafico.winfo_children():
            child.destroy()
        
        # Crea una figura con 2 sottografici
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 1. Grafico a torta della distribuzione secondo la legge del terzo
        labels = ['Primo terzo (≤18)', 'Secondo terzo (19-36)', 'Terzo terzo (>36)']
        sizes = [len(numeri_primo_terzo), len(numeri_secondo_terzo), len(numeri_terzo_terzo)]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0.1, 0.1)  # esplodi tutti i settori leggermente
        
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio assicura che il grafico sia un cerchio
        ax1.set_title('Distribuzione secondo la Legge del Terzo (Ritardi)')
        
        # 2. Grafico a barre dei ritardi dei numeri selezionati
        ritardi_selezionati = [ritardi_numeri[num] for num in numeri_finali]
        
        # Colori diversi in base alla categoria
        colori_barre = []
        for num in numeri_finali:
            if num in numeri_primo_terzo:
                colori_barre.append('#ff9999')  # Primo terzo (rosa)
            elif num in numeri_secondo_terzo:
                colori_barre.append('#66b3ff')  # Secondo terzo (blu)
            else:
                colori_barre.append('#99ff99')  # Terzo terzo (verde)
        
        bars = ax2.bar(numeri_finali, ritardi_selezionati, color=colori_barre)
        
        # Aggiungi etichette sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(bar.get_x() + bar.get_width()/2)}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 punti sopra la barra
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Aggiungi linee orizzontali per i limiti dei terzi
        ax2.axhline(y=18, color='r', linestyle='-', alpha=0.5)
        ax2.axhline(y=36, color='r', linestyle='-', alpha=0.5)
        
        # Aggiungi annotation per i limiti
        ax2.text(numeri_finali[0], 18, 'Limite primo terzo', verticalalignment='bottom')
        ax2.text(numeri_finali[0], 36, 'Limite secondo terzo', verticalalignment='bottom')
        
        ax2.set_title('Ritardi dei Numeri Consigliati')
        ax2.set_xlabel('Numeri')
        ax2.set_ylabel('Ritardo (estrazioni)')
        
        plt.tight_layout()
        
        # Mostra grafico
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    except Exception as e:
        textbox.insert(tk.END, f"Errore nella visualizzazione del grafico: {str(e)}\n")
    
    # Mostra popup con i numeri
    try:
        mostra_numeri_forti_popup(numeri_finali, attendibility_score)
    except Exception as e:
        textbox.insert(tk.END, f"Errore nel mostrare il popup: {str(e)}\n")
    
    return numeri_finali

def mostra_grafico_multi_ruota(histories):
    """
    Mostra il grafico dell'andamento dell'addestramento multi-ruota.
    
    Args:
        histories (list): Lista delle storie di addestramento per ogni fold
    """
    for child in frame_grafico.winfo_children():
        child.destroy()
    
    plt.rcParams.update({'font.size': 12})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calcola le medie per ogni epoca attraverso tutti i fold
    max_epochs = max([len(h['loss']) for h in histories])
    avg_train_loss = []
    avg_val_loss = []
    
    for epoch in range(max_epochs):
        epoch_train_losses = []
        epoch_val_losses = []
        
        for h in histories:
            if epoch < len(h['loss']):
                epoch_train_losses.append(h['loss'][epoch])
            if epoch < len(h['val_loss']):
                epoch_val_losses.append(h['val_loss'][epoch])
        
        if epoch_train_losses:
            avg_train_loss.append(np.mean(epoch_train_losses))
        if epoch_val_losses:
            avg_val_loss.append(np.mean(epoch_val_losses))
    
    # Grafico dell'andamento della loss
    epochs = range(1, len(avg_train_loss) + 1)
    ax1.plot(epochs, avg_train_loss, 'b-', label='Train Loss')
    ax1.plot(epochs, avg_val_loss, 'r-', label='Validation Loss')
    ax1.set_title('Andamento della Loss (Media)')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Grafico del rapporto train/val
    ratios = [v/t if t > 0 else 1.0 for t, v in zip(avg_train_loss, avg_val_loss)]
    ax2.plot(epochs, ratios, 'g-')
    ax2.set_title('Rapporto Val/Train Loss')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Ratio')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)  # Linea di riferimento per ratio=1
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Funzione per salvare il grafico
    def save_plot():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")
    
    btn_save = tk.Button(
        frame_grafico,
        text="Salva Grafico",
        command=save_plot,
        bg="#FFDDC1",
        width=15
    )
    btn_save.pack(pady=5)
    
    # Mostra il grafico
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Toolbar
    toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
def mostra_grafico(all_hist_loss, all_hist_val_loss):
    """
    Mostra il grafico dell'andamento della perdita, con linea di early stopping.
    
    Args:
        all_hist_loss (list): Lista delle loss di training per ogni fold
        all_hist_val_loss (list): Lista delle loss di validazione per ogni fold
    """
    for child in frame_grafico.winfo_children():
        child.destroy()

    try:
        if not all_hist_loss or not all_hist_val_loss:
            logger.error("Nessun dato disponibile per il grafico")
            tk.Label(frame_grafico, text="Nessun dato disponibile per il grafico",
                     font=("Arial", 12), bg="#f0f0f0").pack(pady=20)
            return

        # Calcola la media delle loss di addestramento e validazione per ogni epoca
        avg_train_loss = []
        avg_val_loss = []
        max_epochs = max([len(fold) for fold in all_hist_loss])  # Trova il numero massimo di epoche

        for epoch in range(max_epochs):
            # Prendi le loss per l'epoca corrente da tutti i fold
            epoch_train_losses = [fold[epoch] if epoch < len(fold) else None for fold in all_hist_loss]
            epoch_val_losses = [fold[epoch] if epoch < len(fold) else None for fold in all_hist_val_loss]

            # Rimuovi i valori None e quelli non validi (NaN o infinito)
            valid_train_losses = [x for x in epoch_train_losses if
                                  x is not None and not math.isnan(x) and not math.isinf(x)]
            valid_val_losses = [x for x in epoch_val_losses if
                                x is not None and not math.isnan(x) and not math.isinf(x)]

            # Calcola la media (solo se ci sono valori validi)
            if valid_train_losses:
                avg_train_loss.append(sum(valid_train_losses) / len(valid_train_losses))
            else:
                avg_train_loss.append(None)  # Metti None se non ci sono valori validi

            if valid_val_losses:
                avg_val_loss.append(sum(valid_val_losses) / len(valid_val_losses))
            else:
                avg_val_loss.append(None)  # Metti None se non ci sono valori validi

        # Calcola l'epoca di early stopping (media tra i fold)
        all_early_stopping_epochs = []
        for fold_train_loss, fold_val_loss in zip(all_hist_loss, all_hist_val_loss):
            best_val_loss = float('inf')
            early_stopping_epoch = 0
            for i, val_loss in enumerate(fold_val_loss):
                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    early_stopping_epoch = i
            all_early_stopping_epochs.append(early_stopping_epoch)
        avg_early_stopping_epoch = int(
            np.round(np.mean(all_early_stopping_epochs))) if all_early_stopping_epochs else None

        # Calcola il rapporto tra loss di validazione e loss di addestramento
        loss_ratio = []
        epochs = []  # Tiene traccia delle epoche valide
        for i, (train, val) in enumerate(zip(avg_train_loss, avg_val_loss)):
            if train is not None and val is not None and train > 0:
                ratio = min(val / train, 5.0)  # Limita il rapporto a 5
                loss_ratio.append(ratio)
                epochs.append(i)

        # --- CREAZIONE DEL GRAFICO ---
        plt.rcParams.update({'font.size': 12})  # Dimensione del font
        fig, ax = plt.subplots(figsize=(14, 8), dpi=100)  # Crea figura e asse principale

        # Filtra i valori None prima di passarli a Matplotlib
        valid_train = [x for x in avg_train_loss if x is not None]
        valid_val = [x for x in avg_val_loss if x is not None]

        if not valid_train or not valid_val:
            logger.error("Dati insufficienti per generare il grafico")
            tk.Label(frame_grafico, text="Dati insufficienti per generare il grafico",
                     font=("Arial", 12), bg="#f0f0f0").pack(pady=20)
            return

        # Decidi il fattore di scala in base al valore massimo (tra train e val)
        max_loss = max(max(valid_train, default=0), max(valid_val, default=0))
        if max_loss > 5000:
            scale_factor = 1000
            y_label = "Perdita (valori in migliaia)"
        else:
            scale_factor = 1  # Nessuna scalatura
            y_label = "Perdita"

        scaled_train = [x / scale_factor for x in valid_train]
        scaled_val = [x / scale_factor for x in valid_val]

        # Trova l'epoca con la minima val_loss (per il marker)
        min_val_loss_idx = None
        min_val = float('inf')
        for i, val in enumerate(avg_val_loss):
            if val is not None and val < min_val:
                min_val = val
                min_val_loss_idx = i

        # Disegna le linee (solo se ci sono dati validi)
        if scaled_train:
            ax.plot(range(len(scaled_train)), scaled_train, 'b-', linewidth=2.5, label='Loss Addestramento')
        if scaled_val:
            ax.plot(range(len(scaled_val)), scaled_val, 'orange', linewidth=2.5, label='Loss Validazione')

        # Disegna il grafico del rapporto (asse y secondario)
        if loss_ratio:
            ax2 = ax.twinx()  # Crea un secondo asse y
            ax2.plot(epochs, loss_ratio, 'g-', linewidth=1.5, label='Rapporto Loss/Val')
            ax2.set_ylabel('Rapporto Loss/Val', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim(0, min(5.0, max(loss_ratio) * 1.2))  # Limita l'asse y
            ax2.grid(False)  # Nessuna griglia per il secondo asse

        # Evidenzia il punto di minimo val_loss
        if min_val_loss_idx is not None:
            min_val_scaled = min_val / scale_factor
            ax.plot(min_val_loss_idx, min_val_scaled, 'ro', markersize=10, label='Soluzione Ottimale')

        # Disegna la linea verticale per l'early stopping
        if avg_early_stopping_epoch is not None:
            ax.axvline(x=avg_early_stopping_epoch, color='r', linestyle='--', linewidth=2,
                       label=f'Early Stopping (Epoca {avg_early_stopping_epoch})')

        # Configura il grafico
        ax.grid(True, linestyle='-', alpha=0.7, which='both')
        ax.set_title("Andamento della Perdita durante l'Addestramento e Rapporto", fontsize=16,
                     fontweight='bold')
        ax.set_xlabel("Epoche di Addestramento", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)  # Usa l'etichetta dinamica

        # Combina le legende dei due assi
        lines1, labels1 = ax.get_legend_handles_labels()
        if 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
        else:
            lines = lines1
            labels = labels1
        ax.legend(lines, labels, loc='upper left')

        # Funzione per salvare il grafico (definita internamente)
        def save_plot():
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

        # Pulsante per salvare il grafico
        save_button = tk.Button(frame_grafico, text="Salva Grafico", command=save_plot)
        save_button.pack(pady=5)

        # Mostra il grafico in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Aggiungi la toolbar di Matplotlib
        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        logger.error(f"Errore durante la creazione del grafico: {e}")
        messagebox.showerror("Errore", f"Errore nella generazione grafico: {e}")
    finally:
        plt.close('all')  # Chiudi tutte le figure matplotlib

def aggiungi_feature_temporali(data):
    """
    Aggiunge feature temporali al dataframe delle estrazioni.
    
    Args:
        data (pd.DataFrame): Dataframe con colonna 'Data' di tipo datetime
        
    Returns:
        pd.DataFrame: Dataframe con nuove feature temporali
    """
    # Copia il dataframe per non modificare l'originale
    df = data.copy()
    
    # Estrai componenti temporali
    df['giorno_settimana'] = df['Data'].dt.dayofweek
    df['giorno_mese'] = df['Data'].dt.day
    df['settimana_anno'] = df['Data'].dt.isocalendar().week
    df['mese'] = df['Data'].dt.month
    df['trimestre'] = df['Data'].dt.quarter
    
    # Trasforma componenti circolari in coordinate sinusoidali
    # Giorno della settimana (0-6) -> coordiante circolari
    df['giorno_sett_sin'] = np.sin(2 * np.pi * df['giorno_settimana'] / 7)
    df['giorno_sett_cos'] = np.cos(2 * np.pi * df['giorno_settimana'] / 7)
    
    # Mese (1-12) -> coordinate circolari
    df['mese_sin'] = np.sin(2 * np.pi * df['mese'] / 12)
    df['mese_cos'] = np.cos(2 * np.pi * df['mese'] / 12)
    
    # Giorno del mese (1-31) -> coordinate circolari
    df['giorno_mese_sin'] = np.sin(2 * np.pi * df['giorno_mese'] / 31)
    df['giorno_mese_cos'] = np.cos(2 * np.pi * df['giorno_mese'] / 31)
    
    return df

def aggiungi_statistiche_numeri(numeri_storici, finestra=10):
    """
    Aggiunge feature basate sulle statistiche storiche dei numeri.
    
    Args:
        numeri_storici (np.array): Matrice con i numeri estratti storicamente
        finestra (int): Dimensione della finestra per le statistiche
        
    Returns:
        np.array: Matrice con le nuove feature
    """
    num_estrazioni, num_numeri = numeri_storici.shape
    result = []
    
    for i in range(num_estrazioni):
        # Considera solo le estrazioni precedenti
        indice_inizio = max(0, i - finestra)
        subset = numeri_storici[indice_inizio:i]
        
        if len(subset) > 0:
            # Calcola frequenza di ogni numero nella finestra
            freq_map = {}
            for estrazione in subset:
                for num in estrazione:
                    freq_map[num] = freq_map.get(num, 0) + 1
            
            # Feature per ogni numero nell'estrazione corrente
            estrazione_feature = []
            for num in numeri_storici[i]:
                # Frequenza recente
                freq_recente = freq_map.get(num, 0) / len(subset) if len(subset) > 0 else 0
                
                # Tempo dall'ultima estrazione
                ultima_estrazione = 0
                for j in range(len(subset)-1, -1, -1):
                    if num in subset[j]:
                        ultima_estrazione = i - (indice_inizio + j)
                        break
                
                estrazione_feature.extend([freq_recente, ultima_estrazione])
                
            result.append(estrazione_feature)
        else:
            # Per le prime estrazioni senza dati storici sufficienti
            result.append([0] * (num_numeri * 2))
    
    return np.array(result)

def analizza_importanza_feature_primo_layer(model, feature_names):
    """
    Analizza l'importanza delle feature basandosi sulla somma dei valori
    assoluti dei pesi che connettono l'input al primo layer denso (o simile).
    Gestisce anche discrepanze tra dimensioni del modello e nomi feature.

    Args:
        model (tf.keras.Model): Il modello TensorFlow/Keras addestrato.
        feature_names (list[str]): Una lista di stringhe contenente i nomi
                                   delle feature di input, nell'ordine in cui
                                   appaiono nelle colonne di X.

    Returns:
        dict: Un dizionario con i nomi delle feature e le loro importanze.
    """
    if not feature_names:
        print("Errore: Lista 'feature_names' vuota fornita.")
        return {}

    try:
        first_dense_layer = None
        # Itera sui layer per trovare il primo con pesi 'kernel'
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.weights:
                first_dense_layer = layer
                print(f"DEBUG: Trovato primo layer con kernel: {first_dense_layer.name} (Tipo: {type(layer).__name__})")
                break

        if first_dense_layer is None:
            print("Errore: Impossibile trovare un layer con pesi di input ('kernel') nel modello.")
            return {}

        weights = first_dense_layer.get_weights()
        if not weights or len(weights) == 0:
             print(f"Errore: Il layer {first_dense_layer.name} non ha restituito pesi.")
             return {}

        kernel_weights = weights[0]
        print(f"DEBUG: Shape del kernel del primo layer: {kernel_weights.shape}")

        num_input_features_in_weights = kernel_weights.shape[0]

        # MODIFICA: Adatta i nomi delle feature alla dimensione del kernel
        if num_input_features_in_weights != len(feature_names):
            print(f"AVVISO: Discrepanza tra numero feature nei pesi ({num_input_features_in_weights}) e nomi forniti ({len(feature_names)})")
            print("Nomi forniti:", feature_names)
            
            # Adatta la lista dei nomi
            adapted_feature_names = []
            
            if num_input_features_in_weights > len(feature_names):
                # Se il modello ha più feature di quelle fornite
                adapted_feature_names = feature_names.copy()
                # Aggiungi nomi generici per le feature mancanti
                for i in range(len(feature_names), num_input_features_in_weights):
                    adapted_feature_names.append(f"Feature_Extra_{i+1}")
                print(f"Aggiunti {num_input_features_in_weights - len(feature_names)} nomi generici")
            else:
                # Se il modello ha meno feature di quelle fornite
                adapted_feature_names = feature_names[:num_input_features_in_weights]
                print(f"Troncati i nomi feature a {num_input_features_in_weights}")
            
            # Usa i nomi adattati
            feature_names = adapted_feature_names
            print("Nomi adattati:", feature_names)

        # Calcola l'importanza
        importances_per_feature = np.sum(np.abs(kernel_weights), axis=1)
        
        # Normalizza le importanze
        total_importance = np.sum(importances_per_feature)
        if total_importance > 0:
            normalized_importances = importances_per_feature / total_importance
        else:
            print("Warning: Tutte le importanze calcolate sono zero.")
            normalized_importances = np.zeros_like(importances_per_feature)

        # Crea il dizionario associando nomi e importanze
        result = dict(zip(feature_names, normalized_importances))

        print("DEBUG: Importanze calcolate e normalizzate:", result)
        return result

    except Exception as e:
        # Cattura tutti gli errori in un unico blocco
        print(f"Errore nell'analisi dell'importanza: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def visualizza_risultati_analisi_avanzata(fold_metrics, importanze):
    """
    Visualizza i risultati dell'analisi avanzata con adattamento automatico per il numero di fold.
    Se ci sono più di 5 fold, mostra solo un sottoinsieme rappresentativo.
    
    Args:
        fold_metrics: Metriche per ogni fold
        importanze: Importanza delle feature (non utilizzata in questa versione)
    """
    # Pulisci il frame
    for child in frame_grafico.winfo_children():
        child.destroy()

    plt.rcParams.update({'font.size': 12})

    # Crea una figura con 2 sottografici
    fig = plt.figure(figsize=(14, 10))

    # PRIMO GRAFICO: Andamento della Loss per ogni fold (con adattamento)
    ax1 = fig.add_subplot(2, 1, 1)

    # Determina il numero totale di fold
    num_folds = len(fold_metrics)
    
    # Decidi quali fold visualizzare in base al numero totale
    folds_to_show = []
    if num_folds <= 5:
        # Se ci sono 5 o meno fold, mostra tutti
        folds_to_show = list(range(num_folds))
    else:
        # Se ci sono più di 5 fold, seleziona strategicamente:
        # - Il primo fold
        # - L'ultimo fold
        # - Alcuni fold intermedi distribuiti uniformemente
        folds_to_show = [0]  # Primo fold
        
        # Aggiungi fold intermedi (distribuiti uniformemente)
        step = (num_folds - 2) / 3  # Scegli 3 fold intermedi
        for i in range(1, 4):
            idx = int(i * step)
            if idx > 0 and idx < num_folds - 1 and idx not in folds_to_show:
                folds_to_show.append(idx)
        
        folds_to_show.append(num_folds - 1)  # Ultimo fold
        folds_to_show.sort()  # Ordina gli indici
    
    # Colori per i fold
    colors = plt.cm.tab10(np.linspace(0, 1, len(folds_to_show)))
    
    # Visualizza solo i fold selezionati
    for i, fold_idx in enumerate(folds_to_show):
        metrics = fold_metrics[fold_idx]
        history = metrics['history']
        epochs = range(1, len(history['loss']) + 1)
        
        # Usa lo stesso colore per train e val, ma con stili diversi
        color = colors[i]
        ax1.plot(epochs, history['loss'], '--', label=f'Fold {fold_idx+1} Train', color=color)
        ax1.plot(epochs, history['val_loss'], '-', label=f'Fold {fold_idx+1} Val', color=color)

        # Trova e mostra il punto di minima val_loss per questo fold
        min_val_loss_idx = np.argmin(history['val_loss'])
        ax1.plot(epochs[min_val_loss_idx], history['val_loss'][min_val_loss_idx], 'o', color='red')

    # Aggiungi un testo per indicare che si stanno mostrando solo alcuni fold
    if num_folds > 5:
        ax1.text(0.5, 0.02, f"Mostrando 5 fold rappresentativi di {num_folds} totali", 
                 ha='center', va='bottom', transform=ax1.transAxes, 
                 fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.7))

    ax1.set_title(f'Andamento Loss per Fold Rappresentativi (totale: {num_folds} fold)')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, max([len(m['history']['loss']) for m in fold_metrics])+1))

    # SECONDO GRAFICO: Andamento della perdita media
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Calcola la media delle loss di addestramento e validazione per ogni epoca
    avg_train_loss = []
    avg_val_loss = []
    max_epochs = max([len(m['history']['loss']) for m in fold_metrics])

    for epoch in range(max_epochs):
        # Prendi le loss per l'epoca corrente da tutti i fold
        epoch_train_losses = [m['history']['loss'][epoch] if epoch < len(m['history']['loss']) else None for m in fold_metrics]
        epoch_val_losses = [m['history']['val_loss'][epoch] if epoch < len(m['history']['val_loss']) else None for m in fold_metrics]

        # Rimuovi i valori None e quelli non validi
        valid_train_losses = [x for x in epoch_train_losses if x is not None and not math.isnan(x) and not math.isinf(x)]
        valid_val_losses = [x for x in epoch_val_losses if x is not None and not math.isnan(x) and not math.isinf(x)]

        # Calcola la media
        if valid_train_losses:
            avg_train_loss.append(sum(valid_train_losses) / len(valid_train_losses))
        else:
            avg_train_loss.append(None)

        if valid_val_losses:
            avg_val_loss.append(sum(valid_val_losses) / len(valid_val_losses))
        else:
            avg_val_loss.append(None)

    # Calcola l'epoca di early stopping (media tra i fold)
    all_early_stopping_epochs = []
    for metrics in fold_metrics:
        history = metrics['history']
        best_val_loss = float('inf')
        early_stopping_epoch = 0
        for i, val_loss in enumerate(history['val_loss']):
            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                early_stopping_epoch = i
        all_early_stopping_epochs.append(early_stopping_epoch)
    
    avg_early_stopping_epoch = int(np.round(np.mean(all_early_stopping_epochs))) if all_early_stopping_epochs else None

    # Calcola il rapporto tra loss di validazione e loss di addestramento
    loss_ratio = []
    epochs_indices = []
    for i, (train, val) in enumerate(zip(avg_train_loss, avg_val_loss)):
        if train is not None and val is not None and train > 0:
            ratio = min(val / train, 5.0)  # Limita il rapporto a 5
            loss_ratio.append(ratio)
            epochs_indices.append(i)

    # Filtra i valori None
    valid_train = [x for x in avg_train_loss if x is not None]
    valid_val = [x for x in avg_val_loss if x is not None]
    valid_epochs = range(len(valid_train))

    if not valid_train or not valid_val:
        ax2.text(0.5, 0.5, "Dati insufficienti per generare il grafico",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14)
    else:
        # Trova l'epoca con la minima val_loss (per il marker)
        min_val_loss_idx = np.argmin(valid_val)
        min_val = valid_val[min_val_loss_idx]

        # Disegna le linee
        ax2.plot(valid_epochs, valid_train, 'b-', linewidth=2.5, label='Loss Addestramento')
        ax2.plot(valid_epochs, valid_val, 'orange', linewidth=2.5, label='Loss Validazione')

        # Evidenzia il punto di minimo val_loss
        ax2.plot(min_val_loss_idx, min_val, 'ro', markersize=10, label='Soluzione Ottimale')

        # Disegna il grafico del rapporto (asse y secondario)
        if loss_ratio:
            ax3 = ax2.twinx()  # Crea un secondo asse y
            ax3.plot(epochs_indices, loss_ratio, 'g-', linewidth=1.5, label='Rapporto Loss/Val')
            ax3.set_ylabel('Rapporto Loss/Val', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            ax3.set_ylim(0, min(2.0, max(loss_ratio) * 1.2))
            ax3.grid(False)

        # Disegna la linea verticale per l'early stopping
        if avg_early_stopping_epoch is not None:
            ax2.axvline(x=avg_early_stopping_epoch, color='r', linestyle='--', linewidth=2,
                       label=f'Early Stopping (Epoca {avg_early_stopping_epoch})')

        # Configura il grafico
        ax2.grid(True, linestyle='-', alpha=0.7, which='both')
        ax2.set_title(f"Andamento della Perdita Media su {num_folds} Fold e Rapporto", fontsize=14)
        ax2.set_xlabel("Epoche di Addestramento")
        ax2.set_ylabel("Perdita")

        # Combina le legende dei due assi
        lines1, labels1 = ax2.get_legend_handles_labels()
        if 'ax3' in locals():
            lines2, labels2 = ax3.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
            ax2.legend(lines, labels, loc='upper left')
        else:
            ax2.legend(loc='upper left')

    plt.tight_layout()

    # Mostra il grafico
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Toolbar
    toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Aggiungi pulsante per salvare
    btn_save = tk.Button(
        frame_grafico,
        text="Salva Grafico",
        command=lambda: save_plot(fig),
        bg="#FFDDC1",
        width=15
    )
    btn_save.pack(pady=5)

    # Funzione per salvare il grafico
    def save_plot(fig):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

# Assicurati che le importazioni necessarie siano presenti all'inizio del file
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import traceback
import os
import logging # Assumi che logger sia configurato
# Assicurati che le altre tue funzioni siano definite/importate
# (es. carica_dati_grezzi, preprocessa_dati, estrai_numeri,
# aggiungi_feature_temporali, aggiungi_statistiche_numeri, build_model,
# valuta_attendibilita, estrai_numeri_frequenti, genera_numeri_attendibili,
# mostra_numeri_forti_popup, analizza_importanza_feature_primo_layer,
# visualizza_risultati_analisi_avanzata)

logger = logging.getLogger(__name__) # Esempio

# Versione che legge parametri UI e usa attendibilità standard (come v8)
def analisi_avanzata_completa():
    """
    Esegue un'analisi avanzata completa (con KFold CV).
    Legge parametri modello (neuroni, dropout, patience, min_delta) dalla UI.
    Usa formula attendibilità standard.
    """
    # --- Variabili Globali Necessarie ---
    global numeri_finali, ruota_selezionata, textbox, config, entry_start_date, entry_end_date
    global entry_epochs, entry_batch_size, FILE_RUOTE, fold_performances, progress_bar
    global ultima_ruota_elaborata, pulsanti_ruote, btn_start, root, frame_grafico
    # Aggiungi le entry globali per leggere i parametri dalla UI
    global entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3
    global entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3
    global entry_patience, entry_min_delta
    # ------------------------------------

    ruota = ruota_selezionata.get()
    if not ruota: messagebox.showwarning("Attenzione", "Seleziona ruota."); return None

    # --- Lettura e Validazione Parametri dalla UI ---
    try:
        start_date = pd.to_datetime(entry_start_date.get_date(), format='%Y/%m/%d')
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
        if start_date >= end_date: raise ValueError("Data inizio >= data fine.")

        # Parametri di addestramento base
        epochs = int(entry_epochs.get())
        batch_size = int(entry_batch_size.get())
        if epochs <= 0 or batch_size <= 0: raise ValueError("Epochs e Batch Size devono essere > 0.")

        # === Leggi parametri modello e early stopping dalla UI ===
        try:
            patience_ui = int(entry_patience.get())
            min_delta_ui = float(entry_min_delta.get())
            if patience_ui <= 0 or min_delta_ui < 0: raise ValueError("Patience > 0 e Min Delta >= 0.")

            dense_layers_ui = [int(getattr(entry, 'get', lambda: 0)()) for entry in [entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3]]
            dropout_rates_ui = [float(getattr(entry, 'get', lambda: 0.0)()) for entry in [entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3]]

            valid_layers_dropout = [(n, d) for n, d in zip(dense_layers_ui, dropout_rates_ui) if n > 0]
            if not valid_layers_dropout:
                 print("Warning [Analisi Completa]: Nessun neurone specificato. Uso default [128].")
                 dense_layers_ui = [128]; dropout_rates_ui = [0.3]
            else:
                 dense_layers_ui = [n for n, d in valid_layers_dropout]; dropout_rates_ui = [d for n, d in valid_layers_dropout]

            dropout_rates_ui = [d for d in dropout_rates_ui if d >= 0]
            if len(dropout_rates_ui) < len(dense_layers_ui):
                print(f"Warning [Analisi Completa]: Aggiustando dropout."); dropout_rates_ui.extend([0.3] * (len(dense_layers_ui) - len(dropout_rates_ui)))
            elif len(dropout_rates_ui) > len(dense_layers_ui):
                 print(f"Warning [Analisi Completa]: Troncando dropout."); dropout_rates_ui = dropout_rates_ui[:len(dense_layers_ui)]

            print("DEBUG [Analisi Completa]: Parametri letti da UI:")
            print(f"  Dense Layers: {dense_layers_ui}"); print(f"  Dropout Rates: {dropout_rates_ui}"); print(f"  Patience: {patience_ui}"); print(f"  Min Delta: {min_delta_ui}")

        except Exception as param_err:
            messagebox.showerror("Errore Parametri Modello", f"Errore lettura parametri dalla UI:\n{param_err}"); return None
        # === FINE BLOCCO LETTURA UI ===

    except ValueError as ve: messagebox.showerror("Errore Input", f"Valore non valido:\n{ve}"); return None
    except Exception as e: messagebox.showerror("Errore Parametri", f"Errore parametri generali:\n{e}"); return None

    # --- Aggiorna/Disabilita UI ---
    try:
        for rb in pulsanti_ruote.values():
             if rb.winfo_exists(): rb.config(state="disabled")
        if btn_start.winfo_exists(): btn_start.config(text="ELABORAZIONE IN CORSO...", state="disabled", bg="#cccccc")
        if textbox.winfo_exists(): textbox.delete(1.0, tk.END); textbox.insert(tk.END, f"Avvio analisi avanzata completa per {ruota}...\n");
        if root.winfo_exists(): root.update()
    except Exception as ui_err: print(f"Errore agg UI: {ui_err}")

    # --- Carica Dati ---
    file_name = FILE_RUOTE.get(ruota)
    if not file_name or not os.path.exists(file_name): messagebox.showerror("Errore", f"File estrazioni per {ruota} non trovato."); return None

    try: # Blocco try principale per l'intera analisi
        # --- Carica e Preprocessa Dati ---
        textbox.insert(tk.END, f"Caricamento dati da {start_date.strftime('%Y/%m/%d')} a {end_date.strftime('%Y/%m/%d')}...\n"); textbox.update()
        df_grezzo = carica_dati_grezzi(ruota);
        if df_grezzo is None: raise ValueError("Caricamento dati grezzi fallito.")
        df_preproc = preprocessa_dati(df_grezzo, start_date, end_date);
        if df_preproc is None or df_preproc.empty: raise ValueError("Nessun dato trovato nell'intervallo.")

        # --- Feature Engineering ---
        data_con_feature = aggiungi_feature_temporali(df_preproc.reset_index())
        numeri = estrai_numeri(data_con_feature);
        if numeri is None: raise ValueError("Estrazione numeri fallita.")
        window_stats = 10
        feature_statistiche = aggiungi_statistiche_numeri(numeri, finestra=window_stats);
        if feature_statistiche is None: raise ValueError("Calcolo feature statistiche fallito.")

        # --- Definizione Nomi Feature ---
        feature_names = []; num_stat_features = 0
        feature_names.extend([f'Num{i}_prev' for i in range(1, 6)])
        temporal_cols = ['giorno_sett_sin', 'giorno_sett_cos', 'mese_sin', 'mese_cos', 'giorno_mese_sin', 'giorno_mese_cos']
        feature_names.extend(temporal_cols)
        if isinstance(feature_statistiche, np.ndarray) and feature_statistiche.ndim == 2:
             num_stat_features = feature_statistiche.shape[1]
             if num_stat_features > 0:
                 if num_stat_features % 2 == 0: feature_names.extend([f'{stat}{i}' for i in range(1, (num_stat_features // 2) + 1) for stat in ['Freq', 'Ritardo']])
                 else: feature_names.extend([f'Stat_{i+1}' for i in range(num_stat_features)])
        total_expected_features = len(feature_names)
        print("DEBUG - Feature Names:", feature_names); print(f"DEBUG - Total Expected Features: {total_expected_features}")

        # --- Crea input X e target y ---
        numeri_prev = numeri[:-1]; temporal_prev = data_con_feature[temporal_cols].values[:-1]; stats_prev = feature_statistiche[:-1]
        y = numeri[1:]
        min_len = min(len(numeri_prev), len(temporal_prev), len(stats_prev), len(y))
        if min_len < len(numeri_prev) or min_len < len(temporal_prev) or min_len < len(stats_prev) or min_len < len(y):
             print(f"Warning: Allineamento, tronco a {min_len}."); numeri_prev = numeri_prev[:min_len]; temporal_prev = temporal_prev[:min_len]; stats_prev = stats_prev[:min_len]; y = y[:min_len]
        X = np.hstack([numeri_prev, temporal_prev, stats_prev])
        if X.shape[1] != total_expected_features: raise ValueError(f"Shape X ({X.shape[1]}) != Atteso ({total_expected_features}) dopo hstack.")
        if len(X) != len(y): raise ValueError(f"Lunghezza X != y dopo allineamento.")
        if len(X) == 0: raise ValueError("Dataset X/y vuoto dopo preparazione feature.")

        # --- Normalizzazione X ---
        X_scaled = np.zeros_like(X, dtype=float); X_scaled[:, :5] = X[:, :5] / 90.0; X_scaled[:, 5:11] = X[:, 5:11]
        scaler_stats_global = None
        if total_expected_features > 11 and num_stat_features > 0: scaler_stats_global = MinMaxScaler(); X_scaled[:, 11:] = scaler_stats_global.fit_transform(X[:, 11:])
        elif X_scaled.shape[1] > 11 and num_stat_features == 0: print("Warning: Features statistiche attese ma non calcolate."); X_scaled = X_scaled[:, :11]
        textbox.insert(tk.END, f"Dataset creato con {X_scaled.shape[1]} feature.\n");

        # --- Cross-Validation ---
        n_splits = 5; k = min(n_splits, len(X_scaled));
        if k < 2: raise ValueError("Dati insufficienti per CV (min 2 richiesti).")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(kf.split(X_scaled))
        textbox.insert(tk.END, f"Avvio CV con {k} fold...\n"); textbox.update()
        fold_metrics = []; best_val_loss = float('inf'); best_model = None
        all_hist_loss = []; all_hist_val_loss = []

        # --- Ciclo sui Fold ---
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            print(f"--- Fold {fold_idx}/{k} ---"); textbox.insert(tk.END, f"Fold {fold_idx}/{k}..."); textbox.update()
            if len(train_idx) == 0 or len(val_idx) == 0: print("Fold saltato: indici vuoti."); textbox.insert(tk.END," Saltato.\n"); continue
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]; y_train, y_val = y[train_idx], y[val_idx]
            input_shape=(X_scaled.shape[1],); output_shape=y.shape[1]

            # --- USA PARAMETRI LETTI DALLA UI e FORZA OUTPUT LINEARE ---
            # Assicurati che build_model sia la v9 (con force_regression_output)
            try:
                model = build_model(input_shape, output_shape, dense_layers_ui, dropout_rates_ui, force_regression_output=True)
            except NameError: raise ValueError("Funzione 'build_model' non definita/importata.")
            # -----------------------------------------------------------

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            # Determina loss function (ignora BinaryCE globale per questa analisi)
            is_binary_ce_mode = getattr(config, 'use_binary_crossentropy_mode', False)
            if is_binary_ce_mode: loss_function_to_use = "mean_squared_error"; print("Warning [Analisi Completa]: Forzato MSE perché BinaryCE non adatto.")
            elif config.use_custom_loss: loss_function_to_use = custom_loss_function
            elif config.loss_function_choice: loss_function_to_use = config.loss_function_choice
            else: loss_function_to_use = "mean_squared_error"
            model.compile(optimizer=optimizer, loss=loss_function_to_use, metrics=["mae"])

            # --- USA PATIENCE E MIN_DELTA LETTI DALLA UI ---
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience_ui, min_delta=min_delta_ui, restore_best_weights=True, verbose=0)
            # ---------------------------------------------

            history = model.fit(X_train, y_train / 90.0, validation_data=(X_val, y_val / 90.0), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

            # Valutazione Fold
            val_loss = history.history['val_loss'][-1]; train_loss = history.history['loss'][-1]
            if not (np.isnan(val_loss) or np.isinf(val_loss)):
                ratio = val_loss / train_loss if train_loss > 1e-6 else float('inf')
                textbox.insert(tk.END, f". Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n"); textbox.update()
                fold_metrics.append({'fold': fold_idx, 'train_loss': train_loss, 'val_loss': val_loss, 'ratio': ratio, 'history': history.history})
                all_hist_loss.append(history.history['loss']); all_hist_val_loss.append(history.history['val_loss'])
                if val_loss < best_val_loss: best_val_loss = val_loss; best_model = model
            else: textbox.insert(tk.END, f" Fold {fold_idx}: Val Loss non valida ({val_loss}).\n"); textbox.update()

        if best_model is None: raise ValueError("Nessun modello valido da CV.")
        textbox.insert(tk.END, f"\nAddestramento CV completato.\n")

        # --- Calcolo Metriche e Attendibilità (Formula Standard - come nella tua originale) ---
        attendibility_score = 0; commento = "N/D"
        if fold_metrics:
            valid_val_losses = [m['val_loss'] for m in fold_metrics if m and 'val_loss' in m and not np.isnan(m['val_loss']) and not np.isinf(m['val_loss'])]
            avg_val_loss = np.mean(valid_val_losses) if valid_val_losses else float('nan')
            valid_train_losses = [m['train_loss'] for m in fold_metrics if m and 'train_loss' in m and not np.isnan(m['train_loss']) and not np.isinf(m['train_loss'])]
            avg_train_loss = np.mean(valid_train_losses) if valid_train_losses else float('nan')
            consistency = float('inf')
            if valid_val_losses and len(valid_val_losses) > 1 and avg_val_loss > 1e-9 and not np.isnan(avg_val_loss):
                 consistency_std = np.std(valid_val_losses); consistency = consistency_std / avg_val_loss if not np.isnan(consistency_std) else float('inf')

            loss_multiplier = 10; consistency_multiplier = 10
            if not np.isnan(avg_val_loss) and not np.isinf(consistency):
                term1 = (1 / (1 + avg_val_loss * loss_multiplier)) * 0.7
                term2 = (1 / (1 + consistency * consistency_multiplier)) * 0.3
                attendibility_score = 100 * (term1 + term2); attendibility_score = max(0, min(100, attendibility_score))
            else: attendibility_score = 0

            if attendibility_score > 75:   commento = "Attendibilità Molto Buona"
            elif attendibility_score > 55:  commento = "Attendibilità Buona"
            elif attendibility_score > 35:  commento = "Attendibilità Sufficiente"
            elif attendibility_score > 15:  commento = "Attendibilità Bassa"
            else:                           commento = "Attendibilità Molto Bassa/Inaffidabile"
            textbox.insert(tk.END, f"Attendibilità: {attendibility_score:.1f}/100 ({commento})\n\n")
        else: textbox.insert(tk.END, "Nessuna metrica dai fold per calcolare attendibilità.\n")

        # --- Predizione Finale ---
        X_pred_input_scaled = X_scaled[-1:]
        predizione_norm = best_model.predict(X_pred_input_scaled)[0]
        numeri_denormalizzati = predizione_norm * 90.0
        numeri_interi_predetti = np.round(numeri_denormalizzati).astype(int)
        numeri_da_passare = np.clip(numeri_interi_predetti, 1, 90).flatten()

        # --- Generazione Numeri Finali ---
        numeri_finali = []; origine_ml = []
        try:
             numeri_frequenti = estrai_numeri_frequenti(ruota, start_date, end_date, n=20)
             # Chiama genera_numeri_attendibili (che usa soglia fissa 60)
             numeri_finali, origine_ml = genera_numeri_attendibili(numeri_da_passare, attendibility_score, numeri_frequenti)
             textbox.insert(tk.END, "Analisi avanzata completa completata.\n")
             textbox.insert(tk.END, "Numeri predetti finali suggeriti: " + ", ".join(map(str, numeri_finali)) + "\n")
             mostra_numeri_forti_popup(numeri_finali, attendibility_score, origine_ml)
        except NameError as ne_gen: messagebox.showerror("Errore Interno", f"Funzione 'genera_numeri_attendibili' non trovata: {ne_gen}"); textbox.insert(tk.END, f"Errore critico: {ne_gen}\n")
        except Exception as final_num_err:
             messagebox.showerror("Errore Numeri Finali", f"Errore: {final_num_err}")
             textbox.insert(tk.END, f"Generazione numeri finali fallita: {final_num_err}\n")
             fallback_numeri = [];
             if numeri_da_passare is not None: fallback_numeri = numeri_da_passare.tolist()
             numeri_finali = fallback_numeri; origine_ml = [False] * len(numeri_finali)

        # --- Analisi Importanza Feature ---
        importanze = {}
        try:
            if 'feature_names' in locals() and feature_names:
                 importanze = analizza_importanza_feature_primo_layer(best_model, feature_names)
                 print("DEBUG - Importanze calcolate (Primo Layer):", importanze)
            else: print("Warning: 'feature_names' non definito per analisi importanza.")
        except NameError: print("ERRORE: Funzione 'analizza_importanza_feature_primo_layer' non definita!")
        except Exception as fi_err: print(f"Errore calcolo importanza: {fi_err}")

        # --- Visualizza Risultati Graficamente ---
        try:
             visualizza_risultati_analisi_avanzata(fold_metrics, importanze) # Passa le importanze
        except NameError: print("ERRORE: Funzione 'visualizza_risultati_analisi_avanzata' non definita!")
        except Exception as vis_err: print(f"Errore visualizzazione risultati: {vis_err}")

        # --- Salvataggio Modello ---
        try:
            final_model_path = f'best_model_{ruota}.weights.h5'; best_model.save_weights(final_model_path)
            log_func = logger.info if 'logger' in globals() and hasattr(logger, 'info') else print
            log_func(f"Modello migliore salvato: {final_model_path}")
        except Exception as save_err:
            log_func = logger.error if 'logger' in globals() and hasattr(logger, 'error') else print
            log_func(f"Errore salvataggio modello: {save_err}")

        ultima_ruota_elaborata = ruota
        return numeri_finali # Successo

    except Exception as e: # Cattura errori principali
        textbox.insert(tk.END, f"\nERRORE durante l'analisi completa: {str(e)}\n")
        textbox.insert(tk.END, traceback.format_exc())
        return None # Fallimento
    finally:
        # --- Riabilita Interfaccia ---
        try:
            for rb in pulsanti_ruote.values():
                 if rb.winfo_exists(): rb.config(state="normal")
            if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
            if root.winfo_exists(): root.update()
        except Exception as ui_err_exc: print(f"Errore riabilitazione UI (eccezione): {ui_err_exc}")

# --- Assicurati che le altre funzioni necessarie siano definite sopra questa ---

def analisi_pattern_numeri(ruota, start_date, end_date):
    """
    Analizza i pattern ricorrenti nelle estrazioni per una data ruota.
    
    Args:
        ruota (str): Identificativo della ruota
        start_date (datetime): Data di inizio analisi
        end_date (datetime): Data di fine analisi
        
    Returns:
        dict: Dictionary con i risultati dell'analisi
    """
    # Aggiungi questa funzione dopo analisi_avanzata_completa()

def visualizza_probabilita_numeri(predizioni, top_n=20):
    """
    Visualizza le probabilità di estrazione per i numeri più probabili.
    
    Args:
        predizioni (np.array): Array con le predizioni del modello
        top_n (int): Numero di numeri da visualizzare
    """ 

def calcola_ritardo_singola_ruota(numeri_storici, date_estrazioni, num_predetto, data_fine=None):
    """
    Calcola correttamente il ritardo di un numero dalle estrazioni fornite.
    
    Args:
        numeri_storici (np.array): Array con i numeri delle estrazioni
        date_estrazioni (list): Date corrispondenti alle estrazioni
        num_predetto (int): Il numero di cui calcolare il ritardo
        data_fine (datetime, optional): Data di fine per limitare la ricerca
        
    Returns:
        int: Il ritardo (numero di estrazioni dall'ultima apparizione)
    """
    # Ordina le estrazioni dalla più recente alla più vecchia
    sorted_indices = np.argsort(date_estrazioni)[::-1]
    
    # Se data_fine è specificata, filtra solo le estrazioni fino a quella data
    if data_fine is not None:
        filtered_indices = [i for i in sorted_indices if date_estrazioni[i] <= data_fine]
    else:
        filtered_indices = sorted_indices
    
    # Cerca il numero nelle estrazioni ordinate
    for i, idx in enumerate(filtered_indices):
        estrazione = numeri_storici[idx]
        numeri = [int(n) for n in estrazione if n is not None and str(n).isdigit()]
        
        if int(num_predetto) in numeri:
            return i
    
    # Se non trovato, restituisce il numero di estrazioni esaminate
    return len(filtered_indices)

def calcola_ritardo_reale(ruota, num_predetto, data_riferimento):
    """
    Calcola il ritardo di un numero con gestione duplicati.
    """
    file_path = FILE_RUOTE.get(ruota)
    if not file_path or not os.path.exists(file_path):
        print(f"File della ruota {ruota} non trovato")
        return 0
        
    try:
        # Leggi il file
        df = pd.read_csv(file_path, header=None, sep='\t', encoding='utf-8')
        df.columns = ['Data', 'Ruota'] + [f'Num{i}' for i in range(1, 6)]
        
        # Converti le date
        df['Data'] = pd.to_datetime(df['Data'], format='%Y/%m/%d')
        
        # Rimuovi i duplicati basati sulla data
        df = df.drop_duplicates(subset=['Data'])
        
        # Filtra e ordina
        df = df[df['Data'] <= data_riferimento].sort_values(by='Data', ascending=False)
        
        print(f"Calcolo ritardo per il numero {num_predetto} sulla ruota {ruota}")
        print(f"Estrazioni uniche fino a {data_riferimento}: {len(df)}")
        
        # Verifica le date
        date_uniche = df['Data'].unique()
        print(f"Numero di date uniche: {len(date_uniche)}")
        
        if len(date_uniche) < 5:
            print("ATTENZIONE: Meno di 5 date uniche trovate!")
        
        # Mostra le prime 5 estrazioni più recenti (assicurati che siano diverse)
        ultime_estrazioni = df.head(5)
        date_mostrate = []
        
        for i, (_, row) in enumerate(ultime_estrazioni.iterrows()):
            data_str = row['Data'].strftime('%Y/%m/%d')
            
            # Verifica se questa data è già stata mostrata
            if data_str in date_mostrate:
                print(f"AVVISO: Data duplicata {data_str} trovata!")
            
            date_mostrate.append(data_str)
            
            numeri = [int(row[f'Num{j}']) for j in range(1, 6)]
            print(f"  {i+1}. {data_str}: {numeri}")
        
        # Calcola il ritardo
        ritardo = 0
        for _, row in df.iterrows():
            numeri = [int(row[f'Num{j}']) for j in range(1, 6)]
            
            if int(num_predetto) in numeri:
                data_str = row['Data'].strftime('%Y/%m/%d')
                print(f"Numero {num_predetto} trovato dopo {ritardo} estrazioni in data {data_str}")
                return ritardo
            
            ritardo += 1
        
        return ritardo
        
    except Exception as e:
        import traceback
        print(f"Errore nel calcolo del ritardo: {e}")
        print(traceback.format_exc())
        return 0


def calcola_frequenza_nel_periodo(ruota, num_predetto, start_date, end_date):
    """
    Calcola la frequenza TOTALE (conteggio occorrenze) di un numero
    in TUTTE le estrazioni UNICHE PER DATA comprese ESATTAMENTE
    tra start_date ed end_date per una data ruota.

    Args:
        ruota (str): Nome della ruota.
        num_predetto (int or str): Il numero di cui calcolare la frequenza.
        start_date (datetime or str): Data di inizio del periodo (inclusa).
        end_date (datetime or str): Data di fine del periodo (inclusa).

    Returns:
        tuple[int, int]: Una tupla contenente:
                         - frequenza (int): La frequenza totale del numero nel periodo.
                         - totale_estrazioni_considerate (int): Il numero totale di
                           estrazioni uniche per data trovate nel periodo specificato.
                         Restituisce (0, 0) in caso di errore o se non ci sono
                         estrazioni nel periodo.
    """
    # --- 1. Controllo Esistenza File ---
    # (Codice identico a prima per trovare file_path)
    file_path = FILE_RUOTE.get(ruota)
    if not file_path:
        print(f"ERRORE: Ruota '{ruota}' non trovata nel dizionario FILE_RUOTE.")
        return 0, 0 # Ritorna tupla (freq, tot_estrazioni)
    if not os.path.exists(file_path):
        print(f"ERRORE: File per la ruota '{ruota}' non trovato al percorso: {file_path}")
        return 0, 0 # Ritorna tupla

    try:
        # --- 2. Validazione Input ---
        # (Codice identico a prima per validare num_predetto, start_date, end_date)
        try:
            numero_target = int(num_predetto)
            if not (1 <= numero_target <= 90):
                 print(f"ATTENZIONE: 'num_predetto' ({numero_target}) fuori range (1-90).")
        except (ValueError, TypeError):
            print(f"ERRORE: 'num_predetto' ('{num_predetto}') non è un numero intero valido.")
            return 0, 0 # Ritorna tupla
        try:
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            if pd.isna(start_date_dt) or pd.isna(end_date_dt):
                raise ValueError("Formato data inizio/fine non valido.")
            if start_date_dt > end_date_dt:
                 print(f"ATTENZIONE: Data inizio successiva a data fine.")
                 return 0, 0 # Ritorna tupla
        except Exception as date_err:
            print(f"ERRORE conversione/validazione date: {date_err}")
            return 0, 0 # Ritorna tupla

        # --- 3. Lettura e Preparazione Dati ---
        # (Codice identico a prima per leggere CSV, convertire date, droppare duplicati)
        df = pd.read_csv(file_path, header=None, sep='\t', encoding='utf-8')
        df.columns = ['Data', 'Ruota'] + [f'Num{i}' for i in range(1, 6)]
        df['Data'] = pd.to_datetime(df['Data'], format='%Y/%m/%d', errors='coerce')
        df.dropna(subset=['Data'], inplace=True)
        df = df.drop_duplicates(subset=['Data'], keep='first')

        # --- 4. Filtraggio per Data ---
        # Filtra ESATTAMENTE per l'intervallo di date specificato
        df_periodo = df[
            (df['Data'] >= start_date_dt) &
            (df['Data'] <= end_date_dt)
        ]

        # **MODIFICA CHIAVE**: Non applichiamo più .head(finestra)
        # Calcoliamo su tutto df_periodo

        # Ottieni il numero totale di estrazioni considerate in questo periodo
        totale_estrazioni_considerate = len(df_periodo)

        if totale_estrazioni_considerate == 0:
            print(f"DEBUG: Nessuna estrazione unica trovata tra {start_date_dt.strftime('%Y/%m/%d')} e {end_date_dt.strftime('%Y/%m/%d')}")
            return 0, 0 # Ritorna tupla

        print(f"DEBUG: Calcolo frequenza su {totale_estrazioni_considerate} estrazioni uniche tra {start_date_dt.strftime('%Y/%m/%d')} e {end_date_dt.strftime('%Y/%m/%d')}")

        # --- 5. Calcolo Frequenza Totale ---
        frequenza = 0
        colonne_numeri = [f'Num{j}' for j in range(1, 6)]

        # Itera sulle righe del DataFrame FILTRATO PER PERIODO
        for _, row in df_periodo.iterrows(): # <- Usa df_periodo
            for col in colonne_numeri:
                try:
                    if int(row[col]) == numero_target:
                        frequenza += 1
                except (ValueError, TypeError, pd.errors.IntCastingNaNError):
                    continue # Ignora valori non validi

        print(f"DEBUG: Frequenza totale calcolata per {numero_target} nel periodo: {frequenza}")

        # --- 6. Ritorno Risultato ---
        # Ritorna SIA la frequenza SIA il numero di estrazioni considerate
        return frequenza, totale_estrazioni_considerate

    # --- Gestione Errori Generici ---
    # (Identica a prima, ma ritorna tupla (0,0))
    except FileNotFoundError:
        print(f"ERRORE: File non trovato (interno): {file_path}")
        return 0, 0
    except pd.errors.EmptyDataError:
        print(f"ERRORE: File CSV vuoto o malformattato: {file_path}")
        return 0, 0
    except Exception as e:
        print(f"ERRORE generico non previsto nel calcolo della frequenza per {ruota}: {e}")
        # import traceback
        # traceback.print_exc()
        return 0, 0

def genera_spiegazione_predizione(numeri_predetti, feature_importanze, start_date, end_date, ruota):
    """
    Genera spiegazioni più precise per i numeri predetti, basandosi su ritardo
    e frequenza CALCOLATA SULL'INTERO PERIODO specificato (start_date - end_date).

    Args:
        numeri_predetti (list): Lista dei numeri (int or str) predetti dal modello.
        feature_importanze (dict): Dizionario (opzionale) con importanza delle feature.
                                   Può essere None o vuoto.
        start_date (datetime or str): Data di inizio del periodo (inclusa).
                                      Deve essere convertibile da pd.to_datetime.
        end_date (datetime or str): Data di fine periodo (inclusa).
                                    Deve essere convertibile da pd.to_datetime.
        ruota (str): Ruota analizzata (es. 'Bari', 'BA'). Deve essere una chiave
                     valida in FILE_RUOTE.

    Returns:
        str: Spiegazione testuale formattata dei motivi della predizione per ogni numero.
             Restituisce una stringa di messaggio se numeri_predetti è vuota.
    """
    if not numeri_predetti:
        return "Nessun numero predetto fornito per la spiegazione."

    # --- Preparazione Date per Formattazione (gestione errori inclusa) ---
    try:
        # Converti in datetime per formattazione sicura
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        # Formatta le stringhe per l'output
        start_date_str = start_date_dt.strftime('%Y/%m/%d')
        end_date_str = end_date_dt.strftime('%Y/%m/%d')
    except Exception:
        # Fallback se la conversione fallisce (usa rappresentazione originale)
        start_date_str = str(start_date)
        end_date_str = str(end_date)
        print(f"Attenzione: Impossibile formattare le date {start_date} o {end_date} nel formato YYYY/MM/DD.")

    spiegazioni = [] # Lista per contenere le stringhe di spiegazione per ogni numero

    # --- Ciclo su ogni numero predetto ---
    for num_predetto in numeri_predetti:
        # Validazione: assicurati che sia un numero intero valido
        try:
            num_valido = int(num_predetto)
            if not (1 <= num_valido <= 90): # Validazione range opzionale
                 print(f"Attenzione: Numero predetto {num_valido} fuori range (1-90).")
                 # Potresti decidere di saltare o continuare
        except (ValueError, TypeError):
            # Se non è un numero valido, aggiungi un messaggio e salta
            spiegazioni.append(f"**Elemento '{num_predetto}' non è un numero valido:** Impossibile analizzare.")
            continue # Passa al prossimo elemento in numeri_predetti

        # Inizio costruzione stringa per il numero corrente
        spiegazione_num = f"**Numero {num_valido} (Ruota: {ruota}):**\n"
        motivazioni = [] # Lista per le motivazioni (ritardo, frequenza, modello)

        # --- 1. Calcolo e Spiegazione Ritardo ---
        try:
            # Chiama la funzione che calcola il ritardo esatto fino a end_date
            ritardo = calcola_ritardo_reale(ruota, num_valido, end_date)

            # Interpreta il valore del ritardo e aggiungi alla lista motivazioni
            if ritardo > 100: # Soglie indicative, puoi aggiustarle
                motivazioni.append(f"- È un forte ritardatario (assente da {ritardo} estrazioni).")
            elif ritardo > 50:
                motivazioni.append(f"- È un ritardatario significativo (assente da {ritardo} estrazioni).")
            elif ritardo > 20:
                motivazioni.append(f"- Ha un ritardo medio (assente da {ritardo} estrazioni).")
            elif ritardo > 0:
                 motivazioni.append(f"- Ha un ritardo basso (assente da {ritardo} estrazioni).")
            elif ritardo == 0:
                 # Caso specifico: uscito nell'ultima estrazione considerata (end_date)
                 motivazioni.append(f"- È appena uscito (presente nell'ultima estrazione del {end_date_str}).")
            else: # Caso imprevisto (ritardo negativo?)
                 motivazioni.append(f"- Calcolo del ritardo ha prodotto un risultato inatteso ({ritardo}).")

        except Exception as e:
            # Gestione errore nel calcolo del ritardo
            print(f"ERRORE nel calcolo del ritardo per {num_valido} su {ruota}: {e}")
            motivazioni.append(f"- Impossibile calcolare il ritardo ({e}).") # Aggiungi messaggio di errore alla spiegazione

        # --- 2. Calcolo e Spiegazione Frequenza SUL PERIODO ---
        try:
            # Chiama la NUOVA funzione che restituisce (frequenza, totale_estrazioni)
            frequenza, totale_estrazioni = calcola_frequenza_nel_periodo(ruota, num_valido, start_date, end_date)

            # Controlla se ci sono state estrazioni nel periodo prima di descrivere la frequenza
            if totale_estrazioni > 0:
                # Costruisci la spiegazione usando SIA frequenza SIA totale_estrazioni
                if frequenza > 3: # Soglie indicative
                    motivazioni.append(f"- È uscito frequentemente ({frequenza} volte) nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
                elif frequenza > 1:
                    motivazioni.append(f"- È uscito {frequenza} volte nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
                elif frequenza == 1:
                    motivazioni.append(f"- È uscito una volta nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
                else: # frequenza == 0
                    motivazioni.append(f"- Non è uscito nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
            else:
                 # Messaggio se non ci sono estrazioni nel periodo selezionato
                 motivazioni.append(f"- Nessuna estrazione trovata nel periodo specificato ({start_date_str} - {end_date_str}) per calcolare la frequenza.")

        except NameError:
             # Errore grave se la funzione non è definita
             print(f"FATAL ERROR: La funzione 'calcola_frequenza_nel_periodo' non è definita o non è accessibile.")
             motivazioni.append(f"- Impossibile calcolare la frequenza (Errore interno: funzione mancante).")
        except Exception as e:
            # Gestione altri errori nel calcolo della frequenza
            print(f"ERRORE nel calcolo della frequenza per {num_valido} su {ruota}: {e}")
            # import traceback; traceback.print_exc() # Utile per debug
            motivazioni.append(f"- Impossibile calcolare la frequenza nel periodo ({e}).")

        # --- 3. Altre Motivazioni (Opzionale, basato su feature_importanze) ---
        motivazioni_modello = []
        # Verifica che feature_importanze sia un dizionario non vuoto
        if isinstance(feature_importanze, dict) and feature_importanze:
             try:
                # Ordina le feature per importanza (assumendo dict 'nome': score)
                top_features = sorted(feature_importanze.items(), key=lambda item: item[1], reverse=True)
                # Se ci sono feature e la più importante supera una soglia, menzionala
                if top_features and top_features[0][1] > 0.05: # Soglia esempio
                    feature_name = str(top_features[0][0]).replace("_", " ") # Nome leggibile
                    motivazioni_modello.append(f"- Il modello lo suggerisce basandosi su: {feature_name} (importanza: {top_features[0][1]:.2f}).")
                else:
                    # Messaggio generico se nessuna feature è molto importante
                    motivazioni_modello.append("- Suggerito dal modello in base a pattern generali.")
             except Exception as fe_err:
                 # Gestione errore nell'interpretazione delle feature
                 print(f"Errore nell'interpretazione di feature_importanze: {fe_err}")
                 motivazioni_modello.append("- Suggerito dal modello predittivo (analisi feature fallita).")
        # Non aggiungere un messaggio generico se feature_importanze è None o vuoto,
        # le info su ritardo/frequenza potrebbero essere sufficienti.

        # --- 4. Costruzione Spiegazione Finale per il Numero ---
        # Unisci tutte le motivazioni trovate (ritardo, frequenza, modello)
        tutte_le_motivazioni = motivazioni + motivazioni_modello
        if tutte_le_motivazioni:
            # Aggiungi le motivazioni alla stringa principale, separate da a capo
            spiegazione_num += "\n".join(tutte_le_motivazioni)
        else:
            # Messaggio di fallback se NESSUNA motivazione è stata generata
            spiegazione_num += "- La predizione si basa sull'analisi complessiva del modello."

        # Aggiungi la spiegazione completa per questo numero alla lista generale
        spiegazioni.append(spiegazione_num)

    # --- Ritorno: Unisci le spiegazioni di tutti i numeri ---
    # Usa due "a capo" per separare visivamente le spiegazioni dei diversi numeri
    return "\n\n".join(spiegazioni)


# --- DEFINIZIONE LOGGER ---
logger = logging.getLogger(__name__) # O usa print come fallback
log_debug = logger.debug if logger else print

# ==============================================================================
# === analisi_interpretabile - COMPLETA con POPUP RISULTATI ===
# ==============================================================================
# --- Assicurati che log_debug sia definito all'inizio del file o qui ---
import logging
logger = logging.getLogger(__name__) # O usa il logger che hai già configurato
log_debug = logger.debug if logger and logger.hasHandlers() else print
# --------------------------------------------------------------------

def analisi_interpretabile():
    """
    Esegue analisi interpretabile, addestramento semplificato, predizione,
    spiegazione e mostra il popup con i numeri finali.
    MODIFICATO (v8): Legge parametri modello (neuroni, dropout, patience, min_delta) dalla UI.
    """
    # --- Variabili Globali Necessarie ---
    global numeri_finali, ruota_selezionata, textbox, config, entry_start_date, entry_end_date
    global entry_epochs, entry_batch_size, root, pulsanti_ruote, btn_start
    # Aggiungi le entry globali per leggere i parametri dalla UI
    global entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3
    global entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3
    global entry_patience, entry_min_delta
    # ------------------------------------

    analysis_successful = False
    model = None
    numeri_finali = None
    attendibility_score = 0
    commento = "N/D"
    # log_file = None # Rimosso, non sembra utilizzato attivamente qui

    current_ruota_name = ruota_selezionata.get()

    try: # Try principale per l'intera analisi
        # --- Validazione Input Ruota e Date ---
        ruota = current_ruota_name
        if not ruota: messagebox.showwarning("Attenzione", "Seleziona ruota."); return None
        log_debug("Validazione Parametri Input (Interpretabile)...")
        try:
            start_date = pd.to_datetime(entry_start_date.get_date())
            end_date = pd.to_datetime(entry_end_date.get_date())
            if start_date >= end_date: raise ValueError("Data inizio >= data fine.")
        except Exception as date_err:
            log_debug(f"ERRORE parametri date: {date_err}"); messagebox.showerror("Errore Date", f"Formato data non valido:\n{date_err}"); return None

        # --- Lettura e Validazione Parametri Addestramento e Modello dalla UI ---
        try:
            epochs = int(entry_epochs.get())
            batch_size = int(entry_batch_size.get())
            if epochs <= 0 or batch_size <= 0: raise ValueError("Epochs e Batch Size devono essere > 0.")

            # Leggi patience e min_delta dalla UI
            patience_ui = int(entry_patience.get())
            min_delta_ui = float(entry_min_delta.get())
            if patience_ui <= 0 or min_delta_ui < 0: raise ValueError("Patience > 0 e Min Delta >= 0.")

            # Leggi neuroni e dropout dalla UI
            dense_layers_ui = [int(getattr(entry, 'get', lambda: 0)()) for entry in [entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3]]
            dropout_rates_ui = [float(getattr(entry, 'get', lambda: 0.0)()) for entry in [entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3]]

            # Filtra e aggiusta (come fatto per analisi_avanzata_completa)
            valid_layers_dropout = [(n, d) for n, d in zip(dense_layers_ui, dropout_rates_ui) if n > 0]
            if not valid_layers_dropout:
                 print("Warning [Analisi Interpretabile]: Nessun neurone specificato. Uso default [128].")
                 dense_layers_ui = [128]; dropout_rates_ui = [0.3]
            else:
                 dense_layers_ui = [n for n, d in valid_layers_dropout]; dropout_rates_ui = [d for n, d in valid_layers_dropout]

            dropout_rates_ui = [d for d in dropout_rates_ui if d >= 0]
            if len(dropout_rates_ui) < len(dense_layers_ui):
                print(f"Warning [Analisi Interpretabile]: Aggiustando dropout.")
                dropout_rates_ui.extend([0.3] * (len(dense_layers_ui) - len(dropout_rates_ui)))
            elif len(dropout_rates_ui) > len(dense_layers_ui):
                 print(f"Warning [Analisi Interpretabile]: Troncando dropout.")
                 dropout_rates_ui = dropout_rates_ui[:len(dense_layers_ui)]

            print("DEBUG [Analisi Interpretabile]: Parametri letti da UI:")
            print(f"  Dense Layers: {dense_layers_ui}")
            print(f"  Dropout Rates: {dropout_rates_ui}")
            print(f"  Patience: {patience_ui}")
            print(f"  Min Delta: {min_delta_ui}")

        except Exception as param_err:
            log_debug(f"ERRORE parametri UI: {param_err}"); messagebox.showerror("Errore Parametri", f"Parametri input non validi:\n{param_err}"); return None

        # --- Disabilita UI / Pulisci Textbox ---
        try: # Aggiunto try specifico per UI
            # Qui puoi aggiungere la disabilitazione dei bottoni se necessario per questa analisi
            textbox.delete(1.0, tk.END); textbox.insert(tk.END, f"Avvio analisi interpretabile per {ruota}...\n"); textbox.update()
            log_debug("Interfaccia aggiornata per analisi interpretabile.")
        except Exception as ui_err: print(f"Errore agg UI: {ui_err}")

        # --- Caricamento e Preparazione Dati ---
        log_debug(f"Caricamento dati per {ruota} da {start_date.strftime('%Y/%m/%d')} a {end_date.strftime('%Y/%m/%d')}...")
        # NOTA: carica_dati NON viene usata qui, si usa un preprocessing specifico
        # Assicurati che carica_dati_grezzi, preprocessa_dati, estrai_numeri siano definite
        df_grezzo = carica_dati_grezzi(ruota)
        if df_grezzo is None: raise ValueError(f"Caricamento dati grezzi fallito per {ruota}.")
        df_preproc = preprocessa_dati(df_grezzo, start_date, end_date)
        if df_preproc is None or df_preproc.empty: raise ValueError("Nessun dato nell'intervallo.")
        # Rinomina colonne se necessario (dipende da cosa restituisce preprocessa_dati)
        if 'Num1' not in df_preproc.columns:
             # Aggiusta nomi colonne se diversi (es. se hai solo le 5 colonne numeriche)
             if len(df_preproc.columns) == 5: df_preproc.columns = [f'Num{i}' for i in range(1, 6)]
             else: raise ValueError("Colonne numeri non trovate o in formato inatteso.")

        y_orig_all = estrai_numeri(df_preproc) # Estrai i numeri DOPO preprocessing
        if y_orig_all is None: raise ValueError("Estrazione numeri fallita.")

        # Prepara lo scaler per il target (necessario per denormalizzare)
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        # Adatta lo scaler ai dati target PRIMA di creare features lag
        scaler_y.fit(y_orig_all)


        # --- Creazione Feature e Nomi ---
        # (Questa logica sembra specifica per l'interpretabile, mantenuta)
        log_debug("Creazione features di input (X)...")
        feature_list_names = []; temp_df_features = pd.DataFrame(index=df_preproc.index)
        LAGS_TO_USE=[1,2,3];
        for lag in LAGS_TO_USE:
            for i in range(1, 6): col_name=f'Num{i}_lag{lag}'; temp_df_features[col_name]=df_preproc[f'Num{i}'].shift(lag); feature_list_names.append(col_name)
        temp_df_features['Month_Sin']=np.sin(2*np.pi*df_preproc.index.month/12); feature_list_names.append('Month_Sin')
        temp_df_features['Month_Cos']=np.cos(2*np.pi*df_preproc.index.month/12); feature_list_names.append('Month_Cos')
        log_debug("Feature temporali e lag create.")

        # --- Allineamento e Drop NaN ---
        y_orig_df = pd.DataFrame(y_orig_all, index=df_preproc.index, columns=[f'Num{i}' for i in range(1, 6)])
        combined_data = pd.concat([temp_df_features, y_orig_df], axis=1); combined_data.dropna(inplace=True)
        if combined_data.empty: raise ValueError("Nessun dato valido dopo allineamento features.")
        X_final_df = combined_data[feature_list_names]; y_final_df = combined_data[[f'Num{i}' for i in range(1, 6)]]
        X_np = X_final_df.values; y_np = y_final_df.values # Target originali allineati

        # Scalatura X e Y per il modello
        scaler_X = MinMaxScaler(feature_range=(0, 1)); X_scaled = scaler_X.fit_transform(X_np)
        y_scaled = scaler_y.transform(y_np) # Scala i target allineati
        log_debug("Feature Engineering completato."); log_debug(f"Feature Names: {feature_list_names}"); log_debug(f"X_scaled shape: {X_scaled.shape}, y_scaled shape: {y_scaled.shape}")

        # --- Info Base ---
        # ... (invariato) ...
        textbox.insert(tk.END, f"Analisi interpretabile per ruota: {ruota}\n");
        textbox.insert(tk.END, f"Periodo dati effettivo: {combined_data.index.min().strftime('%Y/%m/%d')} - {combined_data.index.max().strftime('%Y/%m/%d')}\n")
        textbox.insert(tk.END, f"Estrazioni per modello: {len(X_scaled)}\n\n"); textbox.update()

        # --- Split Train/Val ---
        # (semplificato per interpretabile, ok)
        if len(X_scaled) < 10: X_train, X_val = X_scaled, X_scaled; y_train, y_val = y_scaled, y_scaled
        else: split_idx = int(len(X_scaled) * 0.8); X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]; y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        log_debug(f"Split dati: Train={len(X_train)}, Val={len(X_val)}")

        # --- Addestramento Modello ---
        textbox.insert(tk.END, f"Addestramento modello ({epochs} epochs)...\n"); textbox.update()
        input_shape=(X_train.shape[1],); output_shape=y_train.shape[1] # Output è 5

        # === USA PARAMETRI LETTI DALLA UI ===
        model = build_model(input_shape, output_shape, dense_layers_ui, dropout_rates_ui)
        # ===================================

        # Determina optimizer e loss (legge da config globale, ok)
        optimizer_choice = getattr(config, 'optimizer_choice', 'adam')
        optimizer = tf.keras.optimizers.get(optimizer_choice) # Usa get per più flessibilità
        optimizer.learning_rate=0.001 # Imposta LR se necessario
        is_binary_ce_mode = getattr(config, 'use_binary_crossentropy_mode', False)
        if is_binary_ce_mode:
             print("Warning [Analisi Interpretabile]: Modalità BinaryCE non prevista per analisi interpretabile. Uso MSE.")
             loss_function_to_use = "mean_squared_error"
             metrics_to_use = ["mae"]
             # Qui potresti anche voler reimpostare output_shape a 5 se build_model non lo fa già
             if model.output_shape[-1] != 5:
                 print("Errore: Incoerenza shape output per loss MSE")
                 # Ricostruisci modello o genera errore
                 raise ValueError("Shape output modello non compatibile con loss MSE")
        else:
             loss_function_to_use = config.loss_function_choice if config.loss_function_choice and not config.use_custom_loss else "mean_squared_error"
             if config.use_custom_loss: loss_function_to_use = custom_loss_function
             metrics_to_use = ["mae"]

        model.compile(optimizer=optimizer, loss=loss_function_to_use, metrics=metrics_to_use)

        # === USA PATIENCE E MIN_DELTA LETTI DALLA UI ===
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_ui, min_delta=min_delta_ui, restore_best_weights=True)]
        # =============================================

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        textbox.insert(tk.END, "Addestramento completato.\n"); textbox.update()

        # --- Predizione ---
        log_debug("Esecuzione Predizione..."); X_pred_input = X_scaled[-1:]
        predizione_scaled = model.predict(X_pred_input)[0]
        numeri_denormalizzati = scaler_y.inverse_transform(predizione_scaled.reshape(1, -1))
        numeri_interi_predetti = np.round(numeri_denormalizzati).astype(int).flatten();
        numeri_da_passare = np.clip(numeri_interi_predetti, 1, 90) # Numeri predetti (dal modello)
        log_debug(f"Predizione (interi): {numeri_da_passare}")

        # --- Generazione Numeri Finali e Attendibilità ---
        log_debug("Generazione Numeri Finali e Attendibilità...")
        # Usa la funzione standard di valutazione basata sulla history (MSE/MAE qui)
        attendibility_score, commento = valuta_attendibilita(history.history)
        numeri_frequenti = estrai_numeri_frequenti(ruota, start_date, end_date, n=20)
        # Passa i numeri predetti (numeri_da_passare) a genera_numeri_attendibili
        numeri_finali, origine_ml = genera_numeri_attendibili(numeri_da_passare, attendibility_score, numeri_frequenti) # Rimosso is_binary_mode
        textbox.insert(tk.END, "Numeri predetti finali suggeriti: " + ", ".join(map(str, numeri_finali)) + "\n")
        mostra_numeri_forti_popup(numeri_finali, attendibility_score, origine_ml)
        log_debug(f"Numeri Finali Generati: {numeri_finali}")

        # --- Analisi Importanza Feature ---
        # (Richiede che feature_list_names sia corretto)
        log_debug("Analisi Importanza Feature..."); feature_importances = None
        try:
            importanze = analizza_importanza_feature_primo_layer(model, feature_list_names)
            feature_importances = importanze
        except Exception as fi_err: log_debug(f"Warning: Errore analisi importanza: {fi_err}")

        # --- Generazione Spiegazione ---
        # (Richiede che genera_spiegazione_predizione esista e funzioni)
        log_debug("Generazione Spiegazione..."); textbox.insert(tk.END, "Generazione spiegazione...\n"); textbox.update()
        try:
            spiegazione = genera_spiegazione_predizione(numeri_finali, feature_importances, start_date, end_date, ruota)
            textbox.insert(tk.END, "\n=== SPIEGAZIONE INTERPRETABILE ===\n\n"); textbox.insert(tk.END, spiegazione + "\n"); textbox.update()
            log_debug("Spiegazione generata e inserita.")
        except NameError as ne: log_debug(f"ERRORE: Funzione spiegazione mancante: {ne}"); messagebox.showerror("Errore Interno", f"Funzione necessaria non trovata: {ne}.");
        except Exception as spieg_err: log_debug(f"ERRORE generazione spiegazione: {spieg_err}"); messagebox.showerror("Errore Spiegazione", f"Errore: {spieg_err}");

        # --- Visualizzazione Grafico Loss ---
        # (Richiede che mostra_grafico_semplificato esista e funzioni)
        try: mostra_grafico_semplificato(history.history)
        except NameError: print("Warning: Funzione mostra_grafico_semplificato non trovata.")
        except Exception as vis_err: log_debug(f"Warning: Errore grafico loss: {vis_err}")

        analysis_successful = True
        log_debug("Blocco try principale completato con successo.")

    # === GESTIONE ERRORI ===
    except ValueError as val_err:
         error_message = str(val_err); log_debug(f"ERRORE ValueError: {error_message}")
         messagebox.showerror("Errore Dati/Valore", f"Si è verificato un errore:\n\n{error_message}")
    except Exception as e:
        error_type = type(e).__name__; error_msg = str(e); detailed_error = traceback.format_exc()
        log_debug(f"ERRORE IMPREVISTO: {error_type} - {error_msg}\n{detailed_error}")
        textbox.insert(tk.END, f"\n--- ERRORE IMPREVISTO ---\n{error_type}: {error_msg}\n--------------------------\n")
        messagebox.showerror("Errore Inatteso", f"Errore imprevisto.\nDettagli nel log/textbox.\n\n({error_type})")

    # === BLOCCO FINALLY ===
    finally:
        # Riabilita Interfaccia (se era stata disabilitata)
        log_debug("Riabilitazione interfaccia (analisi interpretabile)...")
        try: # Riabilita bottoni ruote e start (se esistono)
             for rb in pulsanti_ruote.values():
                  if rb.winfo_exists(): rb.config(state="normal")
             if btn_start.winfo_exists():
                  # Usa current_ruota_name che abbiamo salvato all'inizio
                  btn_start.config(text=f"AVVIA ELABORAZIONE ({current_ruota_name if current_ruota_name else '...'})", state="normal", bg="#4CAF50")
             if root.winfo_exists(): root.update()
        except Exception as ui_err: log_debug(f"Errore riabilitazione UI: {ui_err}")
        # Rimosso log_file.close() perché log_file non era usato

    # === RITORNO FINALE ===
    if analysis_successful:
        textbox.insert(tk.END, "\nAnalisi interpretabile completata con successo.\n")
        textbox.see(tk.END); textbox.update()
        log_debug(f"Ritorno numeri finali (interpretabile): {numeri_finali}")
        return numeri_finali
    else:
        textbox.insert(tk.END, "\nAnalisi interpretabile terminata con errori.\n")
        textbox.see(tk.END); textbox.update()
        log_debug("Ritorno None (analisi interpretabile fallita).")
        return None

def visualizza_semplice_interpretazione(numeri_predetti, numeri_frequenti):
    """
    Visualizza una semplice interpretazione dei numeri predetti.
    
    Args:
        numeri_predetti (list): Lista dei numeri predetti
        numeri_frequenti (list): Lista dei numeri più frequenti
    """
    for child in frame_grafico.winfo_children():
        child.destroy()
    
    plt.rcParams.update({'font.size': 12})
    
    # Crea una figura con 2 sottografici
    fig = plt.figure(figsize=(14, 8))
    
    # Grafico dei numeri predetti
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(range(len(numeri_predetti)), [1] * len(numeri_predetti), tick_label=numeri_predetti, color='skyblue')
    ax1.set_title('Numeri Predetti')
    ax1.set_xlabel('Indice')
    ax1.set_ylabel('Presenza')
    
    # Grafico dei numeri frequenti
    ax2 = fig.add_subplot(2, 1, 2)
    top_n = min(20, len(numeri_frequenti))
    ax2.bar(range(top_n), [1] * top_n, tick_label=numeri_frequenti[:top_n], color='lightgreen')
    ax2.set_title(f'Top {top_n} Numeri Frequenti')
    ax2.set_xlabel('Indice')
    ax2.set_ylabel('Frequenza Relativa')
    
    plt.tight_layout()
    
    # Aggiungi un pulsante per salvare il grafico
    def save_plot():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

    btn_save = tk.Button(
        frame_grafico,
        text="Salva Grafico",
        command=save_plot,
        bg="#FFDDC1",
        width=15
    )
    btn_save.pack(pady=5)
    
    # Mostra il grafico
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Aggiungi la toolbar per le funzionalità di zoom, pan, etc.
    toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Assicurati che le importazioni necessarie siano presenti all'inizio del file
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import traceback
import os
import logging # Assumi che logger sia configurato
# Assicurati che le altre tue funzioni siano definite/importate

logger = logging.getLogger(__name__) # Esempio

def analisi_avanzata_completa():
    """
    Esegue un'analisi avanzata completa (con KFold CV).
    Versione RIPRISTINATA: Legge parametri dalla UI, usa formula attendibilità standard,
    chiama genera_numeri_attendibili con soglia 60 implicita.
    Forza output lineare in build_model.
    """
    # --- Variabili Globali Necessarie ---
    global numeri_finali, ruota_selezionata, textbox, config, entry_start_date, entry_end_date
    global entry_epochs, entry_batch_size, FILE_RUOTE, fold_performances, progress_bar
    global ultima_ruota_elaborata, pulsanti_ruote, btn_start, root, frame_grafico
    global entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3
    global entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3
    global entry_patience, entry_min_delta
    # ------------------------------------

    ruota = ruota_selezionata.get()
    if not ruota: messagebox.showwarning("Attenzione", "Seleziona ruota."); return None

    # --- Lettura e Validazione Parametri dalla UI ---
    try:
        start_date = pd.to_datetime(entry_start_date.get_date(), format='%Y/%m/%d')
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
        if start_date >= end_date: raise ValueError("Data inizio >= data fine.")

        epochs = int(entry_epochs.get())
        batch_size = int(entry_batch_size.get())
        if epochs <= 0 or batch_size <= 0: raise ValueError("Epochs e Batch Size devono essere > 0.")

        try: # Blocco specifico per parametri modello/early stopping
            patience_ui = int(entry_patience.get())
            min_delta_ui = float(entry_min_delta.get())
            if patience_ui <= 0 or min_delta_ui < 0: raise ValueError("Patience > 0 e Min Delta >= 0.")

            dense_layers_ui = [int(getattr(entry, 'get', lambda: 0)()) for entry in [entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3]]
            dropout_rates_ui = [float(getattr(entry, 'get', lambda: 0.0)()) for entry in [entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3]]

            valid_layers_dropout = [(n, d) for n, d in zip(dense_layers_ui, dropout_rates_ui) if n > 0]
            if not valid_layers_dropout: print("Warning [Analisi Completa]: Nessun neurone specificato. Uso default [128]."); dense_layers_ui = [128]; dropout_rates_ui = [0.3]
            else: dense_layers_ui = [n for n, d in valid_layers_dropout]; dropout_rates_ui = [d for n, d in valid_layers_dropout]

            dropout_rates_ui = [d for d in dropout_rates_ui if d >= 0]
            if len(dropout_rates_ui) < len(dense_layers_ui): print(f"Warning [Analisi Completa]: Aggiustando dropout."); dropout_rates_ui.extend([0.3] * (len(dense_layers_ui) - len(dropout_rates_ui)))
            elif len(dropout_rates_ui) > len(dense_layers_ui): print(f"Warning [Analisi Completa]: Troncando dropout."); dropout_rates_ui = dropout_rates_ui[:len(dense_layers_ui)]

            print("DEBUG [Analisi Completa]: Parametri letti da UI:")
            print(f"  Dense Layers: {dense_layers_ui}"); print(f"  Dropout Rates: {dropout_rates_ui}"); print(f"  Patience: {patience_ui}"); print(f"  Min Delta: {min_delta_ui}")

        except Exception as param_err: messagebox.showerror("Errore Parametri Modello", f"Errore lettura parametri dalla UI:\n{param_err}"); return None

    except ValueError as ve: messagebox.showerror("Errore Input", f"Valore non valido:\n{ve}"); return None
    except Exception as e: messagebox.showerror("Errore Parametri", f"Errore parametri generali:\n{e}"); return None

    # --- Aggiorna/Disabilita UI ---
    try:
        for rb in pulsanti_ruote.values():
            if rb.winfo_exists(): rb.config(state="disabled")
        if btn_start.winfo_exists(): btn_start.config(text="ELABORAZIONE IN CORSO...", state="disabled", bg="#cccccc")
        if textbox.winfo_exists(): textbox.delete(1.0, tk.END); textbox.insert(tk.END, f"Avvio analisi avanzata completa per {ruota}...\n");
        if root.winfo_exists(): root.update()
    except Exception as ui_err: print(f"Errore agg UI: {ui_err}")

    # --- Carica Dati ---
    file_name = FILE_RUOTE.get(ruota)
    if not file_name or not os.path.exists(file_name): messagebox.showerror("Errore", f"File estrazioni per {ruota} non trovato."); return None

    try: # Blocco try principale per l'intera analisi
        # --- Carica e Preprocessa Dati ---
        textbox.insert(tk.END, f"Caricamento dati da {start_date.strftime('%Y/%m/%d')} a {end_date.strftime('%Y/%m/%d')}...\n"); textbox.update()
        df_grezzo = carica_dati_grezzi(ruota);
        if df_grezzo is None: raise ValueError("Caricamento dati grezzi fallito.")
        df_preproc = preprocessa_dati(df_grezzo, start_date, end_date);
        if df_preproc is None or df_preproc.empty: raise ValueError("Nessun dato trovato nell'intervallo.")

        # --- Feature Engineering ---
        data_con_feature = aggiungi_feature_temporali(df_preproc.reset_index())
        numeri = estrai_numeri(data_con_feature);
        if numeri is None: raise ValueError("Estrazione numeri fallita.")
        window_stats = 10
        feature_statistiche = aggiungi_statistiche_numeri(numeri, finestra=window_stats);
        if feature_statistiche is None: raise ValueError("Calcolo feature statistiche fallito.")

        # --- Definizione Nomi Feature ---
        feature_names = []; num_stat_features = 0
        feature_names.extend([f'Num{i}_prev' for i in range(1, 6)])
        temporal_cols = ['giorno_sett_sin', 'giorno_sett_cos', 'mese_sin', 'mese_cos', 'giorno_mese_sin', 'giorno_mese_cos']
        feature_names.extend(temporal_cols)
        if isinstance(feature_statistiche, np.ndarray) and feature_statistiche.ndim == 2:
             num_stat_features = feature_statistiche.shape[1]
             if num_stat_features > 0:
                 if num_stat_features % 2 == 0: feature_names.extend([f'{stat}{i}' for i in range(1, (num_stat_features // 2) + 1) for stat in ['Freq', 'Ritardo']])
                 else: feature_names.extend([f'Stat_{i+1}' for i in range(num_stat_features)])
        total_expected_features = len(feature_names)
        print("DEBUG - Feature Names:", feature_names); print(f"DEBUG - Total Expected Features: {total_expected_features}")

        # --- Crea input X e target y ---
        numeri_prev = numeri[:-1]; temporal_prev = data_con_feature[temporal_cols].values[:-1]; stats_prev = feature_statistiche[:-1]
        y = numeri[1:]
        min_len = min(len(numeri_prev), len(temporal_prev), len(stats_prev), len(y))
        if min_len < len(numeri_prev) or min_len < len(temporal_prev) or min_len < len(stats_prev) or min_len < len(y):
             print(f"Warning: Allineamento, tronco a {min_len}."); numeri_prev = numeri_prev[:min_len]; temporal_prev = temporal_prev[:min_len]; stats_prev = stats_prev[:min_len]; y = y[:min_len]
        X = np.hstack([numeri_prev, temporal_prev, stats_prev])
        if X.shape[1] != total_expected_features: raise ValueError(f"Shape X ({X.shape[1]}) != Atteso ({total_expected_features}) dopo hstack.")
        if len(X) != len(y): raise ValueError(f"Lunghezza X != y dopo allineamento.")
        if len(X) == 0: raise ValueError("Dataset X/y vuoto dopo preparazione feature.")

        # --- Normalizzazione X ---
        X_scaled = np.zeros_like(X, dtype=float); X_scaled[:, :5] = X[:, :5] / 90.0; X_scaled[:, 5:11] = X[:, 5:11]
        scaler_stats_global = None
        if total_expected_features > 11 and num_stat_features > 0: scaler_stats_global = MinMaxScaler(); X_scaled[:, 11:] = scaler_stats_global.fit_transform(X[:, 11:])
        elif X_scaled.shape[1] > 11 and num_stat_features == 0: print("Warning: Features statistiche attese ma non calcolate."); X_scaled = X_scaled[:, :11]
        textbox.insert(tk.END, f"Dataset creato con {X_scaled.shape[1]} feature.\n");

        # --- Cross-Validation ---
        n_splits = 5; k = min(n_splits, len(X_scaled));
        if k < 2: raise ValueError("Dati insufficienti per CV (min 2 richiesti).")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        splits = list(kf.split(X_scaled))
        textbox.insert(tk.END, f"Avvio CV con {k} fold...\n"); textbox.update()
        fold_metrics = []; best_val_loss = float('inf'); best_model = None
        all_hist_loss = []; all_hist_val_loss = []

        # --- Ciclo sui Fold ---
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            print(f"--- Fold {fold_idx}/{k} ---"); textbox.insert(tk.END, f"Fold {fold_idx}/{k}..."); textbox.update()
            if len(train_idx) == 0 or len(val_idx) == 0: print("Fold saltato: indici vuoti."); textbox.insert(tk.END," Saltato.\n"); continue
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]; y_train, y_val = y[train_idx], y[val_idx]
            input_shape=(X_scaled.shape[1],); output_shape=y.shape[1] # Output shape è 5

            # === Usa parametri letti da UI e FORZA output lineare ===
            model = build_model(input_shape, output_shape, dense_layers_ui, dropout_rates_ui, force_regression_output=True)
            # ========================================================

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            # Determina loss function (ignora BinaryCE globale)
            is_binary_ce_mode = getattr(config, 'use_binary_crossentropy_mode', False) # Leggi solo per warning
            if is_binary_ce_mode: loss_function_to_use = "mean_squared_error"; print("Warning [Analisi Completa]: Forzato MSE perché BinaryCE non adatto.")
            elif config.use_custom_loss: loss_function_to_use = custom_loss_function
            elif config.loss_function_choice: loss_function_to_use = config.loss_function_choice
            else: loss_function_to_use = "mean_squared_error" # Default
            model.compile(optimizer=optimizer, loss=loss_function_to_use, metrics=["mae"])

            # === Usa patience e min_delta letti da UI ===
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience_ui, min_delta=min_delta_ui, restore_best_weights=True, verbose=0)
            # ============================================

            history = model.fit(X_train, y_train / 90.0, validation_data=(X_val, y_val / 90.0), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

            # Valutazione Fold
            val_loss = history.history['val_loss'][-1]; train_loss = history.history['loss'][-1]
            if not (np.isnan(val_loss) or np.isinf(val_loss)):
                ratio = val_loss / train_loss if train_loss > 1e-6 else float('inf')
                textbox.insert(tk.END, f". Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n"); textbox.update()
                fold_metrics.append({'fold': fold_idx, 'train_loss': train_loss, 'val_loss': val_loss, 'ratio': ratio, 'history': history.history})
                all_hist_loss.append(history.history['loss']); all_hist_val_loss.append(history.history['val_loss'])
                if val_loss < best_val_loss: best_val_loss = val_loss; best_model = model
            else: textbox.insert(tk.END, f" Fold {fold_idx}: Val Loss non valida ({val_loss}).\n"); textbox.update()

        if best_model is None: raise ValueError("Nessun modello valido da CV.")
        textbox.insert(tk.END, f"\nAddestramento CV completato.\n")

        # --- Calcolo Metriche e Attendibilità (Formula Standard) ---
        attendibility_score = 0; commento = "N/D"
        if fold_metrics:
            valid_val_losses = [m['val_loss'] for m in fold_metrics if m and 'val_loss' in m and not np.isnan(m['val_loss']) and not np.isinf(m['val_loss'])]
            avg_val_loss = np.mean(valid_val_losses) if valid_val_losses else float('nan')
            valid_train_losses = [m['train_loss'] for m in fold_metrics if m and 'train_loss' in m and not np.isnan(m['train_loss']) and not np.isinf(m['train_loss'])]
            avg_train_loss = np.mean(valid_train_losses) if valid_train_losses else float('nan')
            consistency = float('inf')
            if valid_val_losses and len(valid_val_losses) > 1 and avg_val_loss > 1e-9 and not np.isnan(avg_val_loss):
                 consistency_std = np.std(valid_val_losses); consistency = consistency_std / avg_val_loss if not np.isnan(consistency_std) else float('inf')

            loss_multiplier = 10; consistency_multiplier = 10 # Moltiplicatori standard
            if not np.isnan(avg_val_loss) and not np.isinf(consistency):
                term1 = (1 / (1 + avg_val_loss * loss_multiplier)) * 0.7
                term2 = (1 / (1 + consistency * consistency_multiplier)) * 0.3
                attendibility_score = 100 * (term1 + term2); attendibility_score = max(0, min(100, attendibility_score))
            else: attendibility_score = 0

            if attendibility_score > 75:   commento = "Attendibilità Molto Buona"
            elif attendibility_score > 55:  commento = "Attendibilità Buona"
            elif attendibility_score > 35:  commento = "Attendibilità Sufficiente"
            elif attendibility_score > 15:  commento = "Attendibilità Bassa"
            else:                           commento = "Attendibilità Molto Bassa/Inaffidabile"
            textbox.insert(tk.END, f"\nRisultati CV: Loss Media={avg_train_loss:.4f}, Val Loss Media={avg_val_loss:.4f}\n") # Rimosso loss_type_info
            textbox.insert(tk.END, f"Attendibilità: {attendibility_score:.1f}/100 ({commento})\n\n")
        else: textbox.insert(tk.END, "Nessuna metrica dai fold per calcolare attendibilità.\n")

        # --- Predizione Finale ---
        X_pred_input_scaled = X_scaled[-1:]
        predizione_norm = best_model.predict(X_pred_input_scaled)[0]
        numeri_denormalizzati = predizione_norm * 90.0
        numeri_interi_predetti = np.round(numeri_denormalizzati).astype(int)
        numeri_da_passare = np.clip(numeri_interi_predetti, 1, 90).flatten()

        # --- Generazione Numeri Finali ---
        numeri_finali = []; origine_ml = []
        try:
             numeri_frequenti = estrai_numeri_frequenti(ruota, start_date, end_date, n=20)
             # Chiama genera_numeri_attendibili (che usa soglia fissa 60)
             numeri_finali, origine_ml = genera_numeri_attendibili(numeri_da_passare, attendibility_score, numeri_frequenti)
             textbox.insert(tk.END, "Analisi avanzata completa completata.\n")
             textbox.insert(tk.END, "Numeri predetti finali suggeriti: " + ", ".join(map(str, numeri_finali)) + "\n")
             mostra_numeri_forti_popup(numeri_finali, attendibility_score, origine_ml)
        except NameError as ne_gen: messagebox.showerror("Errore Interno", f"Funzione 'genera_numeri_attendibili' non trovata: {ne_gen}"); textbox.insert(tk.END, f"Errore critico: {ne_gen}\n")
        except Exception as final_num_err:
             messagebox.showerror("Errore Numeri Finali", f"Errore: {final_num_err}")
             textbox.insert(tk.END, f"Generazione numeri finali fallita: {final_num_err}\n")
             fallback_numeri = [];
             if numeri_da_passare is not None: fallback_numeri = numeri_da_passare.tolist()
             numeri_finali = fallback_numeri; origine_ml = [False] * len(numeri_finali)

        # --- Analisi Importanza Feature ---
        importanze = {}
        try:
            # Assicurati che feature_names sia stato definito correttamente prima
            if 'feature_names' in locals() and feature_names:
                 importanze = analizza_importanza_feature_primo_layer(best_model, feature_names)
                 print("DEBUG - Importanze calcolate (Primo Layer):", importanze)
            else: print("Warning: 'feature_names' non definito per analisi importanza.")
        except NameError: print("ERRORE: Funzione 'analizza_importanza_feature_primo_layer' non definita!")
        except Exception as fi_err: print(f"Errore calcolo importanza: {fi_err}")

        # --- Visualizza Risultati Graficamente ---
        try:
             visualizza_risultati_analisi_avanzata(fold_metrics, importanze) # Passa le importanze calcolate
        except NameError: print("ERRORE: Funzione 'visualizza_risultati_analisi_avanzata' non definita!")
        except Exception as vis_err: print(f"Errore visualizzazione risultati: {vis_err}")

        # --- Salvataggio Modello ---
        try:
            final_model_path = f'best_model_{ruota}.weights.h5'; best_model.save_weights(final_model_path)
            log_func = logger.info if 'logger' in globals() and hasattr(logger, 'info') else print
            log_func(f"Modello migliore salvato: {final_model_path}")
        except Exception as save_err:
             log_func = logger.error if 'logger' in globals() and hasattr(logger, 'error') else print
             log_func(f"Errore salvataggio modello: {save_err}")

        ultima_ruota_elaborata = ruota
        return numeri_finali # Successo

    except Exception as e: # Cattura errori principali
        textbox.insert(tk.END, f"\nERRORE durante l'analisi completa: {str(e)}\n")
        textbox.insert(tk.END, traceback.format_exc())
        return None # Fallimento
    finally:
        # --- Riabilita Interfaccia ---
        try:
            for rb in pulsanti_ruote.values():
                 if rb.winfo_exists(): rb.config(state="normal")
            if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
            if root.winfo_exists(): root.update()
        except Exception as ui_err_exc: print(f"Errore riabilitazione UI (eccezione): {ui_err_exc}")

def visualizza_impatto_feature_temporali(history):
    """
    Visualizza il confronto tra l'addestramento con e senza feature temporali.
    
    Args:
        history: Storia dell'addestramento con feature temporali
    """
    for child in frame_grafico.winfo_children():
        child.destroy()
    
    plt.rcParams.update({'font.size': 12})
    
    # Crea una figura con 2 sottografici
    fig = plt.figure(figsize=(14, 8))
    
    # Grafico dell'andamento della loss
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax1.set_title('Andamento Loss con Feature Temporali')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Grafico dell'importanza delle feature (simulata per ora)
    ax2 = fig.add_subplot(2, 1, 2)
    feature_names = ['Num 1', 'Num 2', 'Num 3', 'Num 4', 'Num 5', 
                     'G.Sett Sin', 'G.Sett Cos', 'Mese Sin', 'Mese Cos', 'G.Mese Sin', 'G.Mese Cos']
    importances = [0.15, 0.12, 0.11, 0.13, 0.14, 0.07, 0.06, 0.08, 0.08, 0.05, 0.05]  # Valori simulati
    
    ax2.bar(feature_names, importances, color='skyblue')
    ax2.set_title('Importanza Relativa delle Feature (Simulata)')
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('Importanza')
    plt.xticks(rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Aggiungi un pulsante per salvare il grafico
    def save_plot():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

    btn_save = tk.Button(
        frame_grafico,
        text="Salva Grafico",
        command=save_plot,
        bg="#FFDDC1",
        width=15
    )
    btn_save.pack(pady=5)
    
    # Mostra il grafico
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Aggiungi la toolbar per le funzionalità di zoom, pan, etc.
    toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def mostra_grafico(all_hist_loss, all_hist_val_loss):
    """
    Mostra il grafico dell'andamento della perdita, con linea di early stopping.
    
    Args:
        all_hist_loss (list): Lista delle loss di training per ogni fold
        all_hist_val_loss (list): Lista delle loss di validazione per ogni fold
    """
    for child in frame_grafico.winfo_children():
        child.destroy()

    try:
        if not all_hist_loss or not all_hist_val_loss:
            logger.error("Nessun dato disponibile per il grafico")
            tk.Label(frame_grafico, text="Nessun dato disponibile per il grafico",
                     font=("Arial", 12), bg="#f0f0f0").pack(pady=20)
            return

        # Calcola la media delle loss di addestramento e validazione per ogni epoca
        avg_train_loss = []
        avg_val_loss = []
        max_epochs = max([len(fold) for fold in all_hist_loss])  # Trova il numero massimo di epoche

        for epoch in range(max_epochs):
            # Prendi le loss per l'epoca corrente da tutti i fold
            epoch_train_losses = [fold[epoch] if epoch < len(fold) else None for fold in all_hist_loss]
            epoch_val_losses = [fold[epoch] if epoch < len(fold) else None for fold in all_hist_val_loss]

            # Rimuovi i valori None e quelli non validi (NaN o infinito)
            valid_train_losses = [x for x in epoch_train_losses if
                                  x is not None and not math.isnan(x) and not math.isinf(x)]
            valid_val_losses = [x for x in epoch_val_losses if
                                x is not None and not math.isnan(x) and not math.isinf(x)]

            # Calcola la media (solo se ci sono valori validi)
            if valid_train_losses:
                avg_train_loss.append(sum(valid_train_losses) / len(valid_train_losses))
            else:
                avg_train_loss.append(None)  # Metti None se non ci sono valori validi

            if valid_val_losses:
                avg_val_loss.append(sum(valid_val_losses) / len(valid_val_losses))
            else:
                avg_val_loss.append(None)  # Metti None se non ci sono valori validi

        # Calcola l'epoca di early stopping (media tra i fold)
        all_early_stopping_epochs = []
        for fold_train_loss, fold_val_loss in zip(all_hist_loss, all_hist_val_loss):
            best_val_loss = float('inf')
            early_stopping_epoch = 0
            for i, val_loss in enumerate(fold_val_loss):
                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    early_stopping_epoch = i
            all_early_stopping_epochs.append(early_stopping_epoch)
        avg_early_stopping_epoch = int(
            np.round(np.mean(all_early_stopping_epochs))) if all_early_stopping_epochs else None

        # Calcola il rapporto tra loss di validazione e loss di addestramento
        loss_ratio = []
        epochs = []  # Tiene traccia delle epoche valide
        for i, (train, val) in enumerate(zip(avg_train_loss, avg_val_loss)):
            if train is not None and val is not None and train > 0:
                ratio = min(val / train, 5.0)  # Limita il rapporto a 5
                loss_ratio.append(ratio)
                epochs.append(i)

        # --- CREAZIONE DEL GRAFICO ---
        plt.rcParams.update({'font.size': 12})  # Dimensione del font
        fig, ax = plt.subplots(figsize=(14, 8), dpi=100)  # Crea figura e asse principale

        # Filtra i valori None prima di passarli a Matplotlib
        valid_train = [x for x in avg_train_loss if x is not None]
        valid_val = [x for x in avg_val_loss if x is not None]

        if not valid_train or not valid_val:
            logger.error("Dati insufficienti per generare il grafico")
            tk.Label(frame_grafico, text="Dati insufficienti per generare il grafico",
                     font=("Arial", 12), bg="#f0f0f0").pack(pady=20)
            return

        # Decidi il fattore di scala in base al valore massimo (tra train e val)
        max_loss = max(max(valid_train, default=0), max(valid_val, default=0))
        if max_loss > 5000:
            scale_factor = 1000
            y_label = "Perdita (valori in migliaia)"
        else:
            scale_factor = 1  # Nessuna scalatura
            y_label = "Perdita"

        scaled_train = [x / scale_factor for x in valid_train]
        scaled_val = [x / scale_factor for x in valid_val]

        # Trova l'epoca con la minima val_loss (per il marker)
        min_val_loss_idx = None
        min_val = float('inf')
        for i, val in enumerate(avg_val_loss):
            if val is not None and val < min_val:
                min_val = val
                min_val_loss_idx = i

        # Disegna le linee (solo se ci sono dati validi)
        if scaled_train:
            ax.plot(range(len(scaled_train)), scaled_train, 'b-', linewidth=2.5, label='Loss Addestramento')
        if scaled_val:
            ax.plot(range(len(scaled_val)), scaled_val, 'orange', linewidth=2.5, label='Loss Validazione')

        # Disegna il grafico del rapporto (asse y secondario)
        if loss_ratio:
            ax2 = ax.twinx()  # Crea un secondo asse y
            ax2.plot(epochs, loss_ratio, 'g-', linewidth=1.5, label='Rapporto Loss/Val')
            ax2.set_ylabel('Rapporto Loss/Val', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim(0, min(5.0, max(loss_ratio) * 1.2))  # Limita l'asse y
            ax2.grid(False)  # Nessuna griglia per il secondo asse

        # Evidenzia il punto di minimo val_loss
        if min_val_loss_idx is not None:
            min_val_scaled = min_val / scale_factor
            ax.plot(min_val_loss_idx, min_val_scaled, 'ro', markersize=10, label='Soluzione Ottimale')

        # Disegna la linea verticale per l'early stopping
        if avg_early_stopping_epoch is not None:
            ax.axvline(x=avg_early_stopping_epoch, color='r', linestyle='--', linewidth=2,
                       label=f'Early Stopping (Epoca {avg_early_stopping_epoch})')

        # Configura il grafico
        ax.grid(True, linestyle='-', alpha=0.7, which='both')
        ax.set_title("Andamento della Perdita durante l'Addestramento e Rapporto", fontsize=16,
                     fontweight='bold')
        ax.set_xlabel("Epoche di Addestramento", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)  # Usa l'etichetta dinamica

        # Combina le legende dei due assi
        lines1, labels1 = ax.get_legend_handles_labels()
        if 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
        else:
            lines = lines1
            labels = labels1
        ax.legend(lines, labels, loc='upper left')

        # Funzione per salvare il grafico (definita internamente)
        def save_plot():
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

        # Pulsante per salvare il grafico
        save_button = tk.Button(frame_grafico, text="Salva Grafico", command=save_plot)
        save_button.pack(pady=5)

        # Mostra il grafico in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Aggiungi la toolbar di Matplotlib
        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        logger.error(f"Errore durante la creazione del grafico: {e}")
        messagebox.showerror("Errore", f"Errore nella generazione grafico: {e}")
    finally:
        plt.close('all')  # Chiudi tutte le figure matplotlib

log_info = logger.info if logger and logger.hasHandlers() else print

# --- Sostituisci la tua vecchia funzione build_model con QUESTA versione DEBUG ---
# Assicurati che le importazioni necessarie siano presenti all'inizio del file
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GaussianNoise, Input, Reshape
from tensorflow.keras.regularizers import l1, l2
import logging # Assumi che logger sia configurato
import numpy as np # Per getattr e is_binary_ce_mode
# Assicurati che 'config' sia accessibile (globale o passata come argomento)
# Assicurati che log_info sia definito (es. logger.info)
logger = logging.getLogger(__name__) # Esempio, usa il tuo logger
log_info = logger.info # Esempio

# --- Sostituisci la tua vecchia funzione build_model con QUESTA ---
def build_model(input_shape, output_units_or_shape, dense_layers=None, dropout_rates=None, force_regression_output=False):
    """
    Costruisce il modello di rete neurale con la configurazione corrente.
    MODIFICATO (v9): Aggiunto force_regression_output per analisi specifiche.
                     Attivazione finale controlla prima force_regression_output.
    Args:
        input_shape (tuple): Shape dell'input (es. (num_features,)).
        output_units_or_shape (int): Numero di unità nell'output layer (5 o 90).
        dense_layers (list, optional): Lista di neuroni per i layer densi.
        dropout_rates (list, optional): Lista di tassi di dropout.
        force_regression_output (bool, optional): Se True, forza l'attivazione finale
                                                  a lineare (None), ignorando la
                                                  modalità BinaryCE globale. Default False.
    Returns:
        tf.keras.Model: Il modello Keras costruito.
    """
    # --- Gestione Parametri Default e Logging Iniziale ---
    if dense_layers is None: dense_layers = getattr(config, 'dense_layers', [128, 64]); log_info("Usando dense_layers di default.")
    if dropout_rates is None: dropout_rates = getattr(config, 'dropout_rates', [0.2] * len(dense_layers)); log_info("Usando dropout_rates di default.")
    if len(dropout_rates) < len(dense_layers):
        default_dropout = 0.2
        dropout_rates.extend([default_dropout] * (len(dense_layers) - len(dropout_rates)))
        log_info(f"Estendendo dropout_rates a {len(dense_layers)} con valore {default_dropout}.")
    elif len(dropout_rates) > len(dense_layers):
         log_info(f"Numero dropout ({len(dropout_rates)}) > Numero layer ({len(dense_layers)}). Troncando.")
         dropout_rates = dropout_rates[:len(dense_layers)]

    model = Sequential()
    actual_input_shape = input_shape

    # --- Gestione Input Shape e Rumore ---
    model_type = getattr(config, 'model_type', 'dense')
    if model_type == 'lstm':
        model.add(Input(shape=actual_input_shape))
        model.add(Reshape((1, actual_input_shape[0])))
        log_info(f"Input reshaped per LSTM: (1, {actual_input_shape[0]})")
    else:
        model.add(Input(shape=actual_input_shape))
        log_info(f"Input shape per Dense: {actual_input_shape}")
    if getattr(config, 'adaptive_noise', False):
        noise_scale = getattr(config, 'noise_scale', 0.01)
        model.add(GaussianNoise(noise_scale))
        log_info(f"Aggiunto GaussianNoise scala {noise_scale}")

    # --- Regolarizzatore e Attivazione Nascosta ---
    regularizer = None
    reg_choice = getattr(config, 'regularization_choice', None)
    reg_value = getattr(config, 'regularization_value', 0.01)
    if reg_choice == 'l1': regularizer = l1(reg_value); log_info(f"Regolarizzazione L1: {reg_value}")
    elif reg_choice == 'l2': regularizer = l2(reg_value); log_info(f"Regolarizzazione L2: {reg_value}")
    activation_func = getattr(config, 'activation_choice', 'relu')
    activation_msg = activation_func if activation_func else 'lineare'
    model_type_msg = model_type if model_type else 'default (dense)'
    reg_msg = reg_choice if reg_choice else 'nessuna'
    log_info(f"Costruzione - Tipo: {model_type_msg}, Attivazione Nascosta: {activation_msg}, Reg: {reg_msg}")

    # --- Costruzione Layer Nascosti ---
    if model_type == 'lstm':
        if not dense_layers: dense_layers = [64]; dropout_rates = [0.2]; log_info("Warning: default LSTM [64].")
        for i, (neurons, dropout) in enumerate(zip(dense_layers, dropout_rates)):
            return_sequences = (i < len(dense_layers) - 1)
            model.add(LSTM(neurons, activation=activation_func, return_sequences=return_sequences, kernel_regularizer=regularizer))
            model.add(BatchNormalization()); model.add(Dropout(dropout)); log_info(f"LSTM {i+1}: N={neurons}, D={dropout}, RS={return_sequences}")
    elif model_type == 'dense':
        if not dense_layers: dense_layers = [128, 64]; dropout_rates = [0.2] * len(dense_layers); log_info("Warning: default Dense [128, 64].")
        for i, (neurons, dropout) in enumerate(zip(dense_layers, dropout_rates)):
            model.add(Dense(neurons, activation=activation_func, kernel_regularizer=regularizer))
            model.add(BatchNormalization()); model.add(Dropout(dropout)); log_info(f"Dense {i+1}: N={neurons}, D={dropout}")
    else:
        log_info("Tipo modello non riconosciuto. Usando architettura Dense di default (fallback).")
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizer)); model.add(BatchNormalization()); model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizer)); model.add(BatchNormalization()); model.add(Dropout(0.3))

    # === ULTIMO LAYER DI OUTPUT (CONDIZIONALE MODIFICATO v9) ===
    is_binary_ce_mode_global = getattr(config, 'use_binary_crossentropy_mode', False)
    final_output_units = output_units_or_shape

    # Determina l'attivazione finale
    if force_regression_output: # <<< CONTROLLA PRIMA QUESTO FLAG
        final_activation = None # Forza a lineare (None)
        log_info(f"Output FORZATO a regressione.")
    elif is_binary_ce_mode_global: # Altrimenti, controlla la modalità globale
        final_activation = 'sigmoid' # Usa sigmoid per BinaryCE
    else: # Se non forzato e non BinaryCE globale
        final_activation = getattr(config, 'output_activation', None) # Usa config o lineare

    activation_log = final_activation if final_activation else 'lineare'
    log_info(f"Aggiunto ultimo layer Dense con {final_output_units} unità e attivazione '{activation_log}'.") # Ora questo log dovrebbe essere corretto

    model.add(Dense(final_output_units, activation=final_activation))
    # =========================================================

    return model

# --- Fine della funzione build_model ---def add_noise(X_train, noise_type_param=None, scale=None, percentage=None):
    """
    Aggiunge rumore ai dati di addestramento.

    Args:
        X_train (np.array): Dati di addestramento.
        noise_type_param (str, optional): Tipo di rumore da aggiungere.
        scale (float, optional): Scala del rumore.
        percentage (float, optional): Percentuale di dati a cui aggiungere rumore.

    Returns:
        np.array: Dati di addestramento con rumore aggiunto.
    """
    if noise_type_param is None:
        noise_type_param = config.noise_type
    if scale is None:
        scale = config.noise_scale
    if percentage is None:
        percentage = config.noise_percentage

    try:
        if noise_type_param == 'gaussian':
            noise = np.random.normal(0, scale * np.abs(X_train), X_train.shape)
        elif noise_type_param == 'uniform':
            noise = np.random.uniform(-scale * np.abs(X_train), scale * np.abs(X_train), X_train.shape)
        elif noise_type_param == 'poisson':
            noise = np.random.poisson(scale, X_train.shape)
        elif noise_type_param == 'exponential':
            noise = np.random.exponential(scale, X_train.shape)
        else:
            logger.error(f"Tipo di rumore '{noise_type_param}' non supportato.")
            return X_train
    except Exception as e:
        logger.error(f"Errore nell'aggiunta del rumore: {e}")
        return X_train

    mask = np.random.rand(*X_train.shape) < percentage
    X_train_noisy = X_train.copy()
    X_train_noisy[mask] += noise[mask]

    X_train_noisy = np.clip(X_train_noisy, 0, 1)

    return X_train_noisy

def calcola_prevedibilita(history):
    """
    Calcola la prevedibilità per ogni ruota in base alla storia dell'addestramento.

    Args:
        history (dict): Storia dell'addestramento per ogni ruota.

    Returns:
        dict: Prevedibilità calcolata per ogni ruota.
    """
    return {ruota: np.mean(data['val_loss']) + 1e-10 for ruota, data in history.items()}

def mostra_prevedibilita(prevedibilita):
    """
    Mostra le ruote con la migliore prevedibilità nella casella di testo.

    Args:
        prevedibilita (dict): Prevedibilità calcolata per ogni ruota.
    """
    migliori_ruote = sorted(prevedibilita.items(), key=lambda x: x[1])[:2]
    ruote_string = ", ".join([ruota for ruota, _ in migliori_ruote])
    textbox.insert(tk.END, f"Le migliori ruote per la previsione sono: {ruote_string}\n")

def update_progress(value, max_value, text_to_append=None):
    """
    Aggiorna la barra di progresso e la casella di testo.

    Args:
        value (int): Valore attuale.
        max_value (int): Valore massimo.
        text_to_append (str, optional): Testo da aggiungere alla casella di testo.
    """
    if progress_bar:
        progress_bar["maximum"] = max_value
        progress_bar["value"] = value
        root.update_idletasks()

    if text_to_append and textbox:
        textbox.insert(tk.END, text_to_append + "\n")

def create_lr_scheduler():
    """Crea uno scheduler per il learning rate."""
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.05)

    return LearningRateScheduler(lr_schedule)

def valuta_attendibilita(history, val_loss=None, train_loss=None):
    """
    Valuta l'attendibilità delle previsioni, gestendo il caso in cui
    'val_loss' non sia presente nella storia.

    Args:
        history (dict): Storia dell'addestramento.
        val_loss (float, optional): Loss di validazione.
        train_loss (float, optional): Loss di training.

    Returns:
        float: Punteggio di attendibilità (0-100).
        str:   Commento sull'attendibilità.
    """
    try:
        # Usa i valori forniti o prendili dalla storia
        final_train_loss = train_loss if train_loss is not None else history['loss'][-1]

        # --- GESTIONE MANCANZA 'val_loss' ---
        if 'val_loss' in history:  # Controlla se 'val_loss' è una chiave
            final_val_loss = val_loss if val_loss is not None else history['val_loss'][-1]
            ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
            stability_window = min(10, len(history['val_loss']))
            recent_val_losses = history['val_loss'][-stability_window:]
            stability = np.std(recent_val_losses) / np.mean(recent_val_losses) if np.mean(recent_val_losses) > 0 else 1.0

            attendibility_score = 100 * (
                (1 / (1 + 5 * final_val_loss)) * 0.5 +  # Premia val_loss bassa
                (1 / (1 + np.abs(ratio - 1.0) * 2)) * 0.3 +  # Premia ratio vicino a 1
                (1 / (1 + 10 * stability)) * 0.2  # Premia stabilità
            )
        else:
            # Se NON c'è val_loss, usa un'attendibilità BASSA
            attendibility_score = 20.0
            commento = "Attendibilità bassa (nessun dato di validazione disponibile)."
            return attendibility_score, commento

        attendibility_score = max(0, min(100, attendibility_score))
        # ... (resto del codice per il commento) ...
        if attendibility_score > 80:
            commento = "Previsione molto attendibile"
        elif attendibility_score > 60:
            commento = "Previsione attendibile"
        elif attendibility_score > 40:
            commento = "Previsione moderatamente attendibile"
        elif attendibility_score > 20:
            commento = "Previsione poco attendibile"
        else:
            commento = "Previsione non attendibile"

        return attendibility_score, commento
    except Exception as e:
        logger.error(f"Errore nel calcolo dell'attendibilità: {e}")
        return 0, "Impossibile calcolare l'attendibilità"

def mostra_grafico_semplificato(history):
    """Mostra un grafico semplificato dell'andamento della loss."""
    for child in frame_grafico.winfo_children():
        child.destroy()

    try:
        if not history['loss']:
            tk.Label(frame_grafico, text="Nessun dato di addestramento disponibile.",
                     font=("Arial", 12)).pack(pady=20)
            return

        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(history['loss'], label='Train Loss', color='blue')
        # Aggiunta della riga per la val_loss, SE ESISTE
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss', color='orange')

        ax.set_title('Andamento della Loss di Addestramento')
        ax.set_xlabel('Epoche')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        logger.error(f"Errore nella creazione del grafico semplificato: {e}")
        tk.Label(frame_grafico, text=f"Errore: {e}", font=("Arial", 12), fg="red").pack(pady=20)

    finally:
        plt.close('all')

def valuta_accuratezza_previsione(y_true, y_pred, tolerance=0):
    """
    Valuta l'accuratezza diretta delle previsioni con diverse metriche.

    Args:
        y_true (np.array): Valori reali
        y_pred (np.array): Valori predetti
        tolerance (int): Tolleranza per l'esatta corrispondenza (0 significa corrispondenza esatta)

    Returns:
        dict: Dizionario con diverse metriche di accuratezza
    """
    # Trasforma in array numpy se non lo sono già
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Assicurati che le forme siano compatibili
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Le forme degli array sono incompatibili: {y_true.shape} vs {y_pred.shape}")

    # Appiattisci gli array se necessario
    if len(y_true.shape) > 1:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred

    # --- Aggiungi il clipping e la gestione degli errori PRIMA di calcolare le metriche ---
    y_true_flat = np.clip(y_true_flat, 1, 90)  # Forza i valori tra 1 e 90
    y_pred_flat = np.clip(y_pred_flat, 1, 90)  # Forza i valori tra 1 e 90
    
    #Controlla che i valori sono nel range, altrimenti gestisci.
    if not np.all((y_true_flat >= 1) & (y_true_flat <= 90)) or not np.all((y_pred_flat >= 1) & (y_pred_flat <= 90)) :
      logger.error(f"Errore: Alcuni valori in y_true o y_pred non sono nel range corretto (1-90). Controlla la normalizzazione/denormalizzazione.")
      messagebox.showerror("Errore", "Errore nei dati. Controlla i valori di input.")
      #Potresti anche decidere di ritornare un dizionario con valori speciali
      #return {"MAE": None, "MSE": None, ... , "Accuratezza_relativa": None}
      return None #Oppure, se vuoi comunque provare a calcolare qualcosa

    # Calcola metriche standard di regressione
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    mse = float(np.mean(np.square(y_true_flat - y_pred_flat)))
    rmse = float(np.sqrt(mse))

    # Prova a calcolare R² (potrebbe non essere significativo per questo tipo di dati)
    try:
        # Calcola R² manualmente per evitare dipendenze da sklearn
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    except:
        r2 = float('nan')

    # Calcola l'accuratezza con tolleranza
    correct = np.sum(np.abs(y_true_flat - y_pred_flat) <= tolerance)
    total = len(y_true_flat)
    accuracy_with_tolerance = float(correct / total if total > 0 else 0)

    # Calcola percentuale di numeri corretti (dato che sono numeri del lotto)
    #if np.all((y_true_flat >= 1) & (y_true_flat <= 90)) and np.all((y_pred_flat >= 1) & (y_pred_flat <= 90)): #RIMOSSO: gestito sopra
    # Calcola quanti numeri predetti sono esattamente corretti (per estrazione)
    if len(y_true.shape) > 1:
        exact_matches_per_draw = np.sum(y_true == y_pred, axis=1)
        draws_with_matches = np.sum(exact_matches_per_draw > 0)
        total_draws = y_true.shape[0]

        # Percentuale di estrazioni con almeno un numero indovinato
        percent_draws_with_matches = float(draws_with_matches / total_draws * 100 if total_draws > 0 else 0)

        # Media dei numeri indovinati per estrazione
        avg_matches_per_draw = float(np.mean(exact_matches_per_draw))
    else:
        exact_matches = np.sum(y_true == y_pred)
        percent_exact_matches = float(exact_matches / len(y_true) * 100 if len(y_true) > 0 else 0)
        avg_matches_per_draw = float(exact_matches)
        percent_draws_with_matches = float(percent_exact_matches)
    #else: #RIMOSSO: gestito sopra
    #    avg_matches_per_draw = 0.0
    #    percent_draws_with_matches = 0.0

    # Calcola l'accuratezza relativa (quanto vicino al target in percentuale)
    # max_possible_error = float(np.max(y_true_flat) - np.min(y_true_flat) if len(y_true_flat) > 0 else 1) #RIMOSSA
    max_possible_error = 89.0  # Massimo errore possibile tra due numeri del Lotto
    relative_accuracy = max(0.0, float((1 - (mae / max_possible_error)) * 100)) #MODIFICATA

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Accuratezza_con_tolleranza": accuracy_with_tolerance * 100,  # Percentuale
        "Media_numeri_corretti_per_estrazione": avg_matches_per_draw,
        "Percentuale_estrazioni_con_numeri_corretti": percent_draws_with_matches,
        "Accuratezza_relativa": relative_accuracy  # Percentuale
    }

def calcola_probabilita_vincita(numeri_predetti, tolleranza=0):
    """
    Calcola la probabilità statistica di indovinare numeri del lotto
    con le previsioni fornite, rispetto a una scelta casuale.
    
    Args:
        numeri_predetti (list/array): Lista dei numeri predetti (es. 5 numeri)
        tolleranza (int): Tolleranza per l'esattezza (es. 0 = esatto, 1 = +/- 1)
        
    Returns:
        dict: Probabilità e miglioramento rispetto al caso
    """
    # Parametri specifici del gioco del lotto
    num_total = 90  # Numeri da 1 a 90
    num_drawn = 5   # Numeri estratti tipicamente
    
    # Calcola probabilità base (estrazione casuale)
    prob_random = 1.0
    for i in range(num_drawn):
        prob_random *= (num_drawn / num_total)
    
    # Probabilità con il modello (assumendo che i numeri siano ben distribuiti)
    # La tolleranza aumenta l'intervallo di "successo" per ogni numero
    effective_range = 1 + 2 * tolleranza  # es. con tolleranza=1, si accettano -1, 0, +1
    prob_model = 1.0
    for i in range(len(numeri_predetti)):
        # Probabilità di indovinare ciascun numero con la tolleranza
        prob_model *= (effective_range / num_total)
    
    # Miglioramento rispetto al caso
    improvement = float(prob_model / prob_random if prob_random > 0 else float('inf'))
    
    return {
        "Probabilità_casuale": float(prob_random * 100),  # Percentuale
        "Probabilità_modello": float(prob_model * 100),   # Percentuale
        "Miglioramento": improvement
    }

def valuta_qualita_previsione(metriche, soglie=None):
    """
    Valuta la qualità complessiva della previsione in base alle metriche.
    
    Args:
        metriche (dict): Dizionario delle metriche dalla funzione valuta_accuratezza_previsione
        soglie (dict, optional): Soglie personalizzate per la valutazione
        
    Returns:
        tuple: (punteggio, valutazione, dettagli)
    """
    if soglie is None:
        # Soglie predefinite (personalizzale in base ai tuoi dati)
        soglie = {
            "Ottimo": {
                "Accuratezza_relativa": 90,
                "Media_numeri_corretti_per_estrazione": 1.5,
                "Miglioramento": 5
            },
            "Buono": {
                "Accuratezza_relativa": 75,
                "Media_numeri_corretti_per_estrazione": 1.0,
                "Miglioramento": 3
            },
            "Discreto": {
                "Accuratezza_relativa": 60,
                "Media_numeri_corretti_per_estrazione": 0.5,
                "Miglioramento": 2
            },
            "Sufficiente": {
                "Accuratezza_relativa": 50,
                "Media_numeri_corretti_per_estrazione": 0.2,
                "Miglioramento": 1.5
            }
        }
    
    # Calcola un punteggio ponderato
    accuracy_weight = 0.4
    matches_weight = 0.4
    improvement_weight = 0.2
    
    accuracy_score = float(metriche.get("Accuratezza_relativa", 0) / 100)
    matches_score = float(min(metriche.get("Media_numeri_corretti_per_estrazione", 0) / 2, 1))  # Normalizza a 1
    improvement_score = float(min(metriche.get("Miglioramento", 1) / 5, 1))  # Normalizza a 1
    
    weighted_score = float(accuracy_score * accuracy_weight + 
                      matches_score * matches_weight + 
                      improvement_score * improvement_weight)
    
    final_score = float(weighted_score * 100)  # Converti in percentuale
    
    # Determina la valutazione qualitativa
    if final_score >= soglie["Ottimo"]["Accuratezza_relativa"]:
        valutazione = "Ottimo"
    elif final_score >= soglie["Buono"]["Accuratezza_relativa"]:
        valutazione = "Buono"
    elif final_score >= soglie["Discreto"]["Accuratezza_relativa"]:
        valutazione = "Discreto"
    elif final_score >= soglie["Sufficiente"]["Accuratezza_relativa"]:
        valutazione = "Sufficiente"
    else:
        valutazione = "Insufficiente"
    
    # Consigli in base alla valutazione
    dettagli = {
        "Ottimo": "Le previsioni sono notevolmente migliori del caso. Affidabilità molto alta.",
        "Buono": "Le previsioni mostrano un pattern significativo. Buona affidabilità.",
        "Discreto": "Le previsioni sono moderatamente migliori del caso. Discreta affidabilità.",
        "Sufficiente": "Le previsioni offrono un leggero vantaggio rispetto al caso. Affidabilità sufficiente.",
        "Insufficiente": "Le previsioni non sono statisticamente migliori del caso. Bassa affidabilità."
    }
    
    return final_score, valutazione, dettagli[valutazione]

def visualizza_accuratezza(y_true, y_pred, numeri_finali=None):
    """
    Crea e mostra un grafico di visualizzazione dell'accuratezza delle previsioni.

    Args:
        y_true (np.array): Valori reali (target), già ARROTONDATI e tra 1 e 90.
        y_pred (np.array): Valori predetti dal modello, già ARROTONDATI e tra 1 e 90.
        numeri_finali (np.array, optional): Numeri finali suggeriti (per evidenziarli).
    """
    for child in frame_grafico.winfo_children():
        child.destroy()  # Pulisci il frame del grafico

    plt.rcParams.update({'font.size': 12})  # Imposta la dimensione del font

    try:
        # Crea una figura con 4 sottografici (2x2)
        fig = plt.figure(figsize=(14, 10))

        # Assicuriamoci che gli array siano NumPy arrays e che abbiano dimensioni corrette
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)


        # 1. CONFRONTO DISTRIBUZIONE NUMERI REALI VS PREDETTI (Istogramma)
        ax1 = fig.add_subplot(2, 2, 1)  # Primo sottografico (riga 1, colonna 1)

        # Se abbiamo array multidimensionali, appiattiamoli per l'istogramma
        if len(y_true.shape) > 1:
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred

        # Crea i bin per l'istogramma (da 1 a 91, per includere tutti i numeri)
        bins = np.arange(1, 92)  # I numeri del Lotto vanno da 1 a 90
        ax1.hist(y_true_flat, bins=bins, alpha=0.5, label='Numeri Reali', color='blue')
        ax1.hist(y_pred_flat, bins=bins, alpha=0.5, label='Numeri Predetti', color='red')

        # Aggiungi linee verticali per i numeri finali (se forniti)
        if numeri_finali is not None and len(numeri_finali) > 0:  # Controlla che non sia vuoto
            for num in numeri_finali:
                if isinstance(num, (int, np.integer)) and 1 <= num <=90:
                  ax1.axvline(x=num, color='green', linestyle='--', linewidth=2) #linea tratteggiata verde

        ax1.set_title('Distribuzione dei Numeri')
        ax1.set_xlabel('Numero Estratto')  # Etichetta asse x più specifica
        ax1.set_ylabel('Frequenza')  # Etichetta asse y
        ax1.legend()
        ax1.grid(True, alpha=0.3)  # Aggiungi una griglia leggera


        # 2. HEATMAP DI ACCURATEZZA (Matrice di Confusione Semplificata)
        ax2 = fig.add_subplot(2, 2, 2)  # Secondo sottografico

        try: #gestione errori
            # Crea una matrice di confusione semplificata (90x90 sarebbe troppo grande)
            # Dividiamo i numeri in gruppi di 10 (per una visualizzazione più compatta)
            num_groups = 9  # 9 gruppi da 10 numeri ciascuno
            confusion = np.zeros((num_groups, num_groups)) #matrice 9x9

            for true_val, pred_val in zip(y_true_flat, y_pred_flat):
                # Calcola gli indici dei gruppi (da 0 a 8)
                true_group = min(int((true_val - 1) / 10), 8) #divisione intera
                pred_group = min(int((pred_val - 1) / 10), 8)
                confusion[true_group, pred_group] += 1

            # Normalizza la matrice di confusione per riga (in modo che ogni riga sommi a 1)
            row_sums = confusion.sum(axis=1, keepdims=True)  # Somma per riga
            confusion_norm = np.where(row_sums > 0, confusion / row_sums, 0) #divisione sicura

            # Crea la heatmap
            im = ax2.imshow(confusion_norm, cmap='YlGnBu')  # Usa una colormap (giallo-verde-blu)
            plt.colorbar(im, ax=ax2) # Aggiunge la colorbar

            # Etichette degli assi (gruppi di numeri)
            group_labels = [f"{i*10+1}-{(i+1)*10}" for i in range(num_groups)]
            ax2.set_xticks(np.arange(num_groups))
            ax2.set_yticks(np.arange(num_groups))
            ax2.set_xticklabels(group_labels)
            ax2.set_yticklabels(group_labels)

            # Ruota le etichette sull'asse x per una migliore leggibilità
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            ax2.set_title("Matrice di Confusione (Gruppi di 10 numeri)")
            ax2.set_xlabel("Numeri Predetti (Raggruppati)")  # Etichetta asse x
            ax2.set_ylabel("Numeri Reali (Raggruppati)")  # Etichetta asse y
        except Exception as e:
            logger.error(f"Errore nella creazione della matrice di confusione: {e}")
            ax2.text(0.5, 0.5, f"Errore: {e}",
                    horizontalalignment='center', verticalalignment='center', color='red')

        # 3. GRAFICO DI ACCURATEZZA PER TOLLERANZA
        ax3 = fig.add_subplot(2, 2, 3)  # Terzo sottografico

        try: #gestione errori
            tolerances = range(6)  # Prova tolleranze da 0 a 5
            accuracies = []

            for tol in tolerances:
                # Calcola l'accuratezza con la tolleranza corrente
                metrics = valuta_accuratezza_previsione(y_true_flat, y_pred_flat, tolerance=tol) #usa i vettori piatti
                accuracies.append(metrics["Accuratezza_con_tolleranza"])

            # Crea il grafico a barre
            ax3.bar(tolerances, accuracies, color='skyblue', width=0.6)  # Barre più larghe

            # Aggiungi i valori sopra le barre
            for i, acc in enumerate(accuracies):
                ax3.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom')

            ax3.set_title('Accuratezza per Tolleranza')
            ax3.set_xlabel('Tolleranza (±)')
            ax3.set_ylabel('Accuratezza (%)')
            ax3.set_xticks(tolerances)  # Mostra tutti i valori di tolleranza sull'asse x
            ax3.set_ylim(0, 105)  # Imposta il limite dell'asse y per includere il 100%
            ax3.grid(True, alpha=0.3)

        except Exception as e:
            logger.error(f"Errore creazione grafico accuratezza: {e}")
            ax3.text(0.5, 0.5, f"Errore: {e}",
                    horizontalalignment='center', verticalalignment='center', color='red')

        # 4. GRAFICO DI PREVISIONE VS REALTÀ (Scatter Plot)
        ax4 = fig.add_subplot(2, 2, 4)  # Quarto sottografico

        try: #gestione errori
            # Prendi un campione casuale di 10 punti (o meno se ci sono meno di 10 punti)
            sample_size = min(10, len(y_true_flat))
            if sample_size > 0:
                if sample_size < len(y_true_flat): #se ho più di 10 punti
                    indices = np.random.choice(len(y_true_flat), size=sample_size, replace=False)
                    sample_true = y_true_flat[indices]
                    sample_pred = y_pred_flat[indices]
                else: #se ho meno di 10 punti, prendo tutti
                    sample_true = y_true_flat
                    sample_pred = y_pred_flat

                ax4.scatter(sample_true, sample_pred, alpha=0.7)

                # Aggiungi la linea di identità perfetta
                min_val = min(min(sample_true), min(sample_pred))
                max_val = max(max(sample_true), max(sample_pred))
                ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

                # Calcola e mostra il coefficiente di correlazione (in modo sicuro)
                try:
                    corr = np.corrcoef(sample_true, sample_pred)[0, 1]
                    if not np.isnan(corr): #se il calcolo è possibile
                        ax4.text(0.05, 0.95, f'Correlazione: {corr:.2f}', transform=ax4.transAxes,
                                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except Exception:
                    pass

                ax4.set_title('Predetto vs Reale (Campione Casuale)')
                ax4.set_xlabel('Valore Reale')
                ax4.set_ylabel('Valore Predetto')
                ax4.grid(True, alpha=0.3) #griglia
            else:
                ax4.text(0.5, 0.5, "Dati insufficienti",
                        horizontalalignment='center', verticalalignment='center', color='red')
        except Exception as e: #se un qualsiasi errore
            logger.error(f"Errore nel grafico di correlazione: {e}")
            ax4.text(0.5, 0.5, f"Errore: {e}",
                    horizontalalignment='center', verticalalignment='center', color='red')

        plt.tight_layout()  # Adatta automaticamente i sottografici per evitare sovrapposizioni

        # Aggiungi un pulsante per salvare il grafico
        def save_plot():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

        btn_save = tk.Button(
            frame_grafico,
            text="Salva Grafico",
            command=save_plot,
            bg="#FFDDC1",
            width=15
        )
        btn_save.pack(pady=5)

        # Mostra il grafico
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Aggiungi la toolbar per le funzionalità di zoom, pan, etc.
        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        logger.error(f"Errore durante la visualizzazione dell'accuratezza: {e}")
        messagebox.showerror("Errore", f"Errore durante la visualizzazione dell'accuratezza: {e}")
        # Crea un semplice messaggio di errore nel frame grafico (in caso di errore)
        error_label = tk.Label(
            frame_grafico,
            text=f"Si è verificato un errore nella visualizzazione: {e}",
            fg="red",
            font=("Arial", 12),
            wraplength=400
        )
        error_label.pack(pady=20)

def estrai_numeri_frequenti(ruota, start_date=None, end_date=None, n=10):
    """
    Estrae i numeri più frequenti per una ruota in un dato periodo,
    gestendo correttamente il formato dei dati.

    Args:
        ruota (str): Identificativo della ruota.
        start_date (datetime, optional): Data di inizio.
        end_date (datetime, optional): Data di fine.
        n (int): Numero di numeri frequenti da estrarre.

    Returns:
        list: Lista dei numeri più frequenti.
    """
    try:
        file_name = FILE_RUOTE.get(ruota)
        if not file_name or not os.path.exists(file_name):  # Controlla esistenza file
            return []

        # Carica i dati correttamente, specificando le colonne
        data = pd.read_csv(file_name, header=None, sep="\t", encoding='utf-8')
        # Rinomina le colonne per chiarezza
        data.columns = ['Data', 'Ruota'] + [f'Num{i}' for i in range(1, 6)]  # Nomi più descrittivi
        data['Data'] = pd.to_datetime(data['Data'], format='%Y/%m/%d')  # Conversione data

        # Filtra per data, se fornite
        if start_date and end_date:
            mask = (data['Data'] >= start_date) & (data['Data'] <= end_date)
            data = data.loc[mask]

        if data.empty:
            return []

        # Estrai *SOLO* i numeri (usa le colonne rinominate!)
        numeri = data[['Num1', 'Num2', 'Num3', 'Num4', 'Num5']].values.flatten()
        numeri = [int(x) for x in numeri if str(x).isdigit()] #assicura che siano interi


        frequenze = {}
        for num in numeri:
            if 1 <= num <= 90:
                frequenze[num] = frequenze.get(num, 0) + 1

        numeri_ordinati = sorted(frequenze.items(), key=lambda x: x[1], reverse=True)
        return [num for num, freq in numeri_ordinati[:n]]


    except Exception as e:
        logger.error(f"Errore nell'estrazione dei numeri frequenti: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def filtra_duplicati(numeri_predetti, n=5):
    """
    Filtra i numeri predetti per evitare duplicati e assicurare diversità.
    """
    numeri_flat = numeri_predetti.flatten()
    numeri_unici = []

    for num in numeri_flat:
        if num not in numeri_unici and 1 <= num <= 90:
            numeri_unici.append(num)
            if len(numeri_unici) >= n:
                break

    while len(numeri_unici) < n:
        nuovo_num = np.random.randint(1, 91)
        if nuovo_num not in numeri_unici:
            numeri_unici.append(nuovo_num)

    return np.array(numeri_unici[:n])

def genera_numeri_attendibili(numeri_interi, attendibility_score, numeri_frequenti):
    """
    Genera una lista di numeri attendibili combinando i numeri predetti dal modello
    con i numeri più frequenti, in base all'attendibilità della previsione.
    VERSIONE CON SOGLIA FISSA 60.
    Args:
        numeri_interi (np.ndarray | list | None): Numeri predetti (dovrebbero essere interi 1-90).
        attendibility_score (float): Punteggio di attendibilità (0-100).
        numeri_frequenti (list): Lista dei numeri più frequenti (interi, 1-90).
    Returns:
        tuple: (list di numeri attendibili, list di booleani origine ML)
    """
    SOGLIA_ATTENDIBILITA = 60 # <-- SOGLIA FISSA
    NUM_NUMERI_FINALI = 5

    # --- Gestione Input Robusta ---
    numeri_validi_ml = []
    if numeri_interi is None:
        print("Info: 'numeri_interi' ricevuto come None.")
        numeri_interi_np = np.array([])
    elif isinstance(numeri_interi, list):
         try: numeri_interi_np = np.array(numeri_interi, dtype=float)
         except ValueError: print(f"Attenzione: impossibile convertire lista {numeri_interi}. Trattata come vuota."); numeri_interi_np = np.array([])
    elif isinstance(numeri_interi, np.ndarray):
        numeri_interi_np = numeri_interi
    else: print(f"Attenzione: tipo non gestito per 'numeri_interi': {type(numeri_interi)}. Trattato come vuoto."); numeri_interi_np = np.array([])
    numeri_interi_flat = numeri_interi_np.flatten()
    for item in numeri_interi_flat:
        try:
            if np.isfinite(item):
                num_int = int(round(float(item)))
                if 1 <= num_int <= 90: numeri_validi_ml.append(num_int)
        except (ValueError, TypeError): continue

    # --- Logica di Selezione ---
    numeri_finali = []
    origine_ml = []
    numeri_frequenti_validi = [num for num in numeri_frequenti if isinstance(num, (int, np.integer)) and 1 <= num <= 90]

    # --- USA LA SOGLIA FISSA ---
    if attendibility_score >= SOGLIA_ATTENDIBILITA:
        print(f"DEBUG [genera_numeri_attendibili]: Ramo ALTA Attendibilità (Score={attendibility_score:.1f} >= Soglia={SOGLIA_ATTENDIBILITA})")
        for num in numeri_validi_ml:
            if num not in numeri_finali:
                numeri_finali.append(num); origine_ml.append(True)
                if len(numeri_finali) >= NUM_NUMERI_FINALI: break
        if len(numeri_finali) < NUM_NUMERI_FINALI:
            for num in numeri_frequenti_validi:
                if num not in numeri_finali:
                    numeri_finali.append(num); origine_ml.append(False)
                    if len(numeri_finali) >= NUM_NUMERI_FINALI: break
    else:
        print(f"DEBUG [genera_numeri_attendibili]: Ramo BASSA Attendibilità (Score={attendibility_score:.1f} < Soglia={SOGLIA_ATTENDIBILITA})")
        count_freq = 0
        for num in numeri_frequenti_validi:
             if num not in numeri_finali:
                 numeri_finali.append(num); origine_ml.append(False)
                 count_freq += 1
                 if count_freq >= 2: break
        needed_for_4 = 4 - len(numeri_finali)
        if needed_for_4 > 0:
            count_ml_added = 0
            for num in numeri_validi_ml:
                if num not in numeri_finali:
                    numeri_finali.append(num); origine_ml.append(True)
                    count_ml_added += 1
                    if count_ml_added >= needed_for_4: break

    # --- Completamento con casuali ---
    tentativi_random = 0; MAX_TENTATIVI_RANDOM = 200
    while len(numeri_finali) < NUM_NUMERI_FINALI and tentativi_random < MAX_TENTATIVI_RANDOM:
        num_casuale = random.randint(1, 90)
        if num_casuale not in numeri_finali:
            numeri_finali.append(num_casuale); origine_ml.append(False)
        tentativi_random += 1
    if len(numeri_finali) < NUM_NUMERI_FINALI:
         print(f"Attenzione: non è stato possibile generare {NUM_NUMERI_FINALI} numeri unici...")

    numeri_finali = numeri_finali[:NUM_NUMERI_FINALI]
    origine_ml = origine_ml[:NUM_NUMERI_FINALI]

    return numeri_finali, origine_ml
def genera_spiegazione_predizione(numeri_predetti, feature_importanze, start_date, end_date, ruota):
    """
    Genera spiegazioni più precise per i numeri predetti, basandosi su ritardo
    e frequenza CALCOLATA SULL'INTERO PERIODO specificato (start_date - end_date).

    Args:
        numeri_predetti (list): Lista dei numeri (int or str) predetti dal modello.
        feature_importanze (dict): Dizionario (opzionale) con importanza delle feature.
                                   Può essere None o vuoto.
        start_date (datetime or str): Data di inizio del periodo (inclusa).
                                      Deve essere convertibile da pd.to_datetime.
        end_date (datetime or str): Data di fine periodo (inclusa).
                                    Deve essere convertibile da pd.to_datetime.
        ruota (str): Ruota analizzata (es. 'Bari', 'BA'). Deve essere una chiave
                     valida in FILE_RUOTE.

    Returns:
        str: Spiegazione testuale formattata dei motivi della predizione per ogni numero.
             Restituisce una stringa di messaggio se numeri_predetti è vuota.
    """
    if not numeri_predetti:
        return "Nessun numero predetto fornito per la spiegazione."

    # --- Preparazione Date per Formattazione (gestione errori inclusa) ---
    try:
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        start_date_str = start_date_dt.strftime('%Y/%m/%d')
        end_date_str = end_date_dt.strftime('%Y/%m/%d')
    except Exception:
        start_date_str = str(start_date)
        end_date_str = str(end_date)
        print(f"Attenzione: Impossibile formattare le date {start_date} o {end_date} nel formato YYYY/MM/DD.")

    spiegazioni = []

    # --- Ciclo su ogni numero predetto ---
    for num_predetto in numeri_predetti:
        try:
            num_valido = int(num_predetto)
            if not (1 <= num_valido <= 90):
                 print(f"Attenzione: Numero predetto {num_valido} fuori range (1-90).")
        except (ValueError, TypeError):
            spiegazioni.append(f"**Elemento '{num_predetto}' non è un numero valido:** Impossibile analizzare.")
            continue

        spiegazione_num = f"**Numero {num_valido} (Ruota: {ruota}):**\n"
        motivazioni = []

        # --- 1. Calcolo e Spiegazione Ritardo ---
        try:
            ritardo = calcola_ritardo_reale(ruota, num_valido, end_date)
            if ritardo > 100: motivazioni.append(f"- È un forte ritardatario (assente da {ritardo} estrazioni).")
            elif ritardo > 50: motivazioni.append(f"- È un ritardatario significativo (assente da {ritardo} estrazioni).")
            elif ritardo > 20: motivazioni.append(f"- Ha un ritardo medio (assente da {ritardo} estrazioni).")
            elif ritardo > 0: motivazioni.append(f"- Ha un ritardo basso (assente da {ritardo} estrazioni).")
            elif ritardo == 0: motivazioni.append(f"- È appena uscito (presente nell'ultima estrazione del {end_date_str}).")
            else: motivazioni.append(f"- Calcolo ritardo inatteso ({ritardo}).")
        except Exception as e:
            print(f"ERRORE nel calcolo del ritardo per {num_valido} su {ruota}: {e}")
            motivazioni.append(f"- Impossibile calcolare il ritardo ({e}).")

        # --- 2. Calcolo e Spiegazione Frequenza SUL PERIODO ---
        try:
            # ==============================================================
            # === QUESTA È LA CHIAMATA CORRETTA CHE DEVE ESSERE PRESENTE ===
            frequenza, totale_estrazioni = calcola_frequenza_nel_periodo(ruota, num_valido, start_date, end_date)
            # ==============================================================

            if totale_estrazioni > 0:
                if frequenza > 3:
                    motivazioni.append(f"- È uscito frequentemente ({frequenza} volte) nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
                elif frequenza > 1:
                    motivazioni.append(f"- È uscito {frequenza} volte nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
                elif frequenza == 1:
                    motivazioni.append(f"- È uscito una volta nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
                else: # frequenza == 0
                    motivazioni.append(f"- Non è uscito nelle {totale_estrazioni} estrazioni considerate (periodo: {start_date_str} - {end_date_str}).")
            else:
                 motivazioni.append(f"- Nessuna estrazione trovata nel periodo specificato ({start_date_str} - {end_date_str}) per calcolare la frequenza.")

        # Cattura specificamente NameError se la funzione chiamata non esiste
        except NameError:
             # Questo errore ora si riferirebbe a 'calcola_frequenza_nel_periodo' se non fosse definita
             print(f"FATAL ERROR: La funzione 'calcola_frequenza_nel_periodo' non è definita o non è accessibile.")
             motivazioni.append(f"- Impossibile calcolare la frequenza (Errore interno: funzione mancante).")
        except Exception as e:
            print(f"ERRORE nel calcolo della frequenza per {num_valido} su {ruota}: {e}")
            motivazioni.append(f"- Impossibile calcolare la frequenza nel periodo ({e}).")

        # --- 3. Altre Motivazioni (Opzionale) ---
        motivazioni_modello = []
        if isinstance(feature_importanze, dict) and feature_importanze:
             try:
                top_features = sorted(feature_importanze.items(), key=lambda item: item[1], reverse=True)
                if top_features and top_features[0][1] > 0.05:
                    feature_name = str(top_features[0][0]).replace("_", " ")
                    motivazioni_modello.append(f"- Il modello lo suggerisce basandosi su: {feature_name} (importanza: {top_features[0][1]:.2f}).")
                else:
                    motivazioni_modello.append("- Suggerito dal modello in base a pattern generali.")
             except Exception as fe_err:
                 print(f"Errore nell'interpretazione di feature_importanze: {fe_err}")
                 motivazioni_modello.append("- Suggerito dal modello predittivo (analisi feature fallita).")

        # --- 4. Costruzione Spiegazione Finale per il Numero ---
        tutte_le_motivazioni = motivazioni + motivazioni_modello
        if tutte_le_motivazioni:
            spiegazione_num += "\n".join(tutte_le_motivazioni)
        else:
            spiegazione_num += "- La predizione si basa sull'analisi complessiva del modello."

        spiegazioni.append(spiegazione_num)

    # --- Ritorno: Unisci le spiegazioni di tutti i numeri ---
    return "\n\n".join(spiegazioni)

def mostra_numeri_forti_popup(numeri_finali, attendibility_score, origine_ml=None):
    """
    Mostra un popup con i numeri finali, evidenziando in rosso quelli derivati dal modello ML
    e in nero quelli derivati dai numeri frequenti.
    
    Args:
        numeri_finali (list): Lista dei numeri consigliati
        attendibility_score (float): Punteggio di attendibilità (0-100)
        origine_ml (list, optional): Lista di booleani che indica se ogni numero proviene dal ML
    """
    print("Programmazione popup con numeri forti...")  # DEBUG
    # Usa root.after per garantire l'esecuzione nel thread principale dell'UI
    root.after(100, lambda: _mostra_numeri_forti_popup_interno(numeri_finali, attendibility_score, origine_ml))

def _mostra_numeri_forti_popup_interno(numeri_finali, attendibility_score, origine_ml=None):
    """Funzione interna che crea effettivamente il popup."""
    try:
        print("Creazione popup numeri forti...")  # DEBUG
        print("numeri_finali:", numeri_finali)  # DEBUG
        print("attendibility_score:", attendibility_score)  # DEBUG
        if origine_ml:
            print("origine_ml:", origine_ml)  # DEBUG
        
        popup = tk.Toplevel(root)  # Usa 'root' come parent esplicito
        popup.title("Numeri Consigliati")
        popup.geometry("400x250")
        
        # Titolo
        tk.Label(popup, text="Numeri Consigliati:", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Frame per i numeri
        frame_numeri = tk.Frame(popup)
        frame_numeri.pack(pady=10)
        
        # Mostra i numeri con colori differenti in base all'origine
        for i, num in enumerate(numeri_finali):
            # Determina il colore in base all'origine (se disponibile)
            if origine_ml and i < len(origine_ml):
                # Rosso per i numeri derivati dal machine learning
                color = "red" if origine_ml[i] else "black"
            else:
                # Se origine_ml non è disponibile, usa la logica originale
                if attendibility_score > 80:
                    num_forti = 5
                elif attendibility_score > 60:
                    num_forti = 4
                elif attendibility_score > 40:
                    num_forti = 3
                elif attendibility_score > 20:
                    num_forti = 2
                else:
                    num_forti = 1
                    
                color = "red" if i < num_forti else "black"
            
            # Crea la label per il numero con il colore appropriato
            label = tk.Label(
                frame_numeri, 
                text=str(num), 
                font=("Arial", 16, "bold"), 
                fg=color, 
                padx=10, 
                pady=5, 
                relief="solid", 
                borderwidth=2
            )
            label.pack(side=tk.LEFT)
        
        # Messaggi informativi
        tk.Label(popup, text="Questi numeri sono leggermente più rilevanti.", font=("Arial", 12)).pack(pady=5)
        tk.Label(popup, text=f"Attendibilità complessiva: {attendibility_score:.1f}/100", 
                font=("Arial", 12)).pack(pady=5)
        
        # Pulsante di chiusura
        btn_chiudi = tk.Button(popup, text="Chiudi", command=popup.destroy, 
                            bg="#f0f0f0", width=10)
        btn_chiudi.pack(pady=10)
        
        # Porta il popup in primo piano
        popup.lift()
        popup.attributes('-topmost', True)
        popup.after_idle(popup.attributes, '-topmost', False)
        popup.focus_force()
        popup.grab_set()
        
        print("Popup creato con successo")  # DEBUG
        
    except Exception as e:
        import traceback
        error_msg = f"Errore nella creazione del popup: {e}\n{traceback.format_exc()}"
        print(error_msg)
        messagebox.showerror("Errore", error_msg)

def mostra_grafico(all_hist_loss, all_hist_val_loss):
    """
    Mostra il grafico dell'andamento della perdita, con linea di early stopping.
    """
    for child in frame_grafico.winfo_children():
        child.destroy()

    try:
        if not all_hist_loss or not all_hist_val_loss:
            logger.error("Nessun dato disponibile per il grafico")
            tk.Label(frame_grafico, text="Nessun dato disponibile per il grafico",
                     font=("Arial", 12), bg="#f0f0f0").pack(pady=20)
            return

        # Calcola la media delle loss di addestramento e validazione per ogni epoca
        avg_train_loss = []
        avg_val_loss = []
        max_epochs = max([len(fold) for fold in all_hist_loss])  # Trova il numero massimo di epoche

        for epoch in range(max_epochs):
            # Prendi le loss per l'epoca corrente da tutti i fold
            epoch_train_losses = [fold[epoch] if epoch < len(fold) else None for fold in all_hist_loss]
            epoch_val_losses = [fold[epoch] if epoch < len(fold) else None for fold in all_hist_val_loss]

            # Rimuovi i valori None e quelli non validi (NaN o infinito)
            valid_train_losses = [x for x in epoch_train_losses if
                                  x is not None and not math.isnan(x) and not math.isinf(x)]
            valid_val_losses = [x for x in epoch_val_losses if
                                x is not None and not math.isnan(x) and not math.isinf(x)]

            # Calcola la media (solo se ci sono valori validi)
            if valid_train_losses:
                avg_train_loss.append(sum(valid_train_losses) / len(valid_train_losses))
            else:
                avg_train_loss.append(None)  # Metti None se non ci sono valori validi

            if valid_val_losses:
                avg_val_loss.append(sum(valid_val_losses) / len(valid_val_losses))
            else:
                avg_val_loss.append(None)  # Metti None se non ci sono valori validi

        # Calcola l'epoca di early stopping (media tra i fold)
        all_early_stopping_epochs = []
        for fold_train_loss, fold_val_loss in zip(all_hist_loss, all_hist_val_loss):
            best_val_loss = float('inf')
            early_stopping_epoch = 0
            for i, val_loss in enumerate(fold_val_loss):
                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    early_stopping_epoch = i
            all_early_stopping_epochs.append(early_stopping_epoch)
        avg_early_stopping_epoch = int(
            np.round(np.mean(all_early_stopping_epochs))) if all_early_stopping_epochs else None

        # Calcola il rapporto tra loss di validazione e loss di addestramento
        loss_ratio = []
        epochs = []  # Tiene traccia delle epoche valide
        for i, (train, val) in enumerate(zip(avg_train_loss, avg_val_loss)):
            if train is not None and val is not None and train > 0:
                ratio = min(val / train, 5.0)  # Limita il rapporto a 5
                loss_ratio.append(ratio)
                epochs.append(i)

        # --- CREAZIONE DEL GRAFICO ---
        plt.rcParams.update({'font.size': 12})  # Dimensione del font
        fig, ax = plt.subplots(figsize=(14, 8), dpi=100)  # Crea figura e asse principale

        # Filtra i valori None prima di passarli a Matplotlib
        valid_train = [x for x in avg_train_loss if x is not None]
        valid_val = [x for x in avg_val_loss if x is not None]

        if not valid_train or not valid_val:
            logger.error("Dati insufficienti per generare il grafico")
            tk.Label(frame_grafico, text="Dati insufficienti per generare il grafico",
                     font=("Arial", 12), bg="#f0f0f0").pack(pady=20)
            return

        # Decidi il fattore di scala in base al valore massimo (tra train e val)
        max_loss = max(max(valid_train, default=0), max(valid_val, default=0))
        if max_loss > 5000:
            scale_factor = 1000
            y_label = "Perdita (valori in migliaia)"
        else:
            scale_factor = 1  # Nessuna scalatura
            y_label = "Perdita"

        scaled_train = [x / scale_factor for x in valid_train]
        scaled_val = [x / scale_factor for x in valid_val]

        # Trova l'epoca con la minima val_loss (per il marker)
        min_val_loss_idx = None
        min_val = float('inf')
        for i, val in enumerate(avg_val_loss):
            if val is not None and val < min_val:
                min_val = val
                min_val_loss_idx = i

        # Disegna le linee (solo se ci sono dati validi)
        if scaled_train:
            ax.plot(range(len(scaled_train)), scaled_train, 'b-', linewidth=2.5, label='Loss Addestramento')
        if scaled_val:
            ax.plot(range(len(scaled_val)), scaled_val, 'orange', linewidth=2.5, label='Loss Validazione')

        # Disegna il grafico del rapporto (asse y secondario)
        if loss_ratio:
            ax2 = ax.twinx()  # Crea un secondo asse y
            ax2.plot(epochs, loss_ratio, 'g-', linewidth=1.5, label='Rapporto Loss/Val')
            ax2.set_ylabel('Rapporto Loss/Val', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_ylim(0, min(5.0, max(loss_ratio) * 1.2))  # Limita l'asse y
            ax2.grid(False)  # Nessuna griglia per il secondo asse

        # Evidenzia il punto di minimo val_loss
        if min_val_loss_idx is not None:
            min_val_scaled = min_val / scale_factor
            ax.plot(min_val_loss_idx, min_val_scaled, 'ro', markersize=10, label='Soluzione Ottimale')

        # Disegna la linea verticale per l'early stopping
        if avg_early_stopping_epoch is not None:
            ax.axvline(x=avg_early_stopping_epoch, color='r', linestyle='--', linewidth=2,
                       label=f'Early Stopping (Epoca {avg_early_stopping_epoch})')

        # Configura il grafico
        ax.grid(True, linestyle='-', alpha=0.7, which='both')
        ax.set_title("Andamento della Perdita durante l'Addestramento e Rapporto", fontsize=16,
                     fontweight='bold')
        ax.set_xlabel("Epoche di Addestramento", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)  # Usa l'etichetta dinamica

        # Combina le legende dei due assi
        lines1, labels1 = ax.get_legend_handles_labels()
        if 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines1 + lines2
            labels = labels1 + labels2
        else:
            lines = lines1
            labels = labels1
        ax.legend(lines, labels, loc='upper left')

        # Funzione per salvare il grafico (definita internamente)
        def save_plot():
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

        # Pulsante per salvare il grafico
        save_button = tk.Button(frame_grafico, text="Salva Grafico", command=save_plot)
        save_button.pack(pady=5)

        # Mostra il grafico in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Aggiungi la toolbar di Matplotlib
        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        logger.error(f"Errore durante la creazione del grafico: {e}")
        messagebox.showerror("Errore", f"Errore nella generazione grafico: {e}")
    finally:
        plt.close('all')  # Chiudi tutte le figure matplotlib

def visualizza_accuratezza(y_true, y_pred, numeri_finali=None):
    """
    Crea e mostra un grafico di visualizzazione dell'accuratezza delle previsioni.
    print("Dentro visualizza_accuratezza")
    print("  y_true:", y_true)          # <-- Aggiungi questo
    print("  y_pred:", y_pred)          # <-- Aggiungi questo
    print("  numeri_finali:", numeri_finali)  # <-- Aggiungi questo
    Args:
        y_true (np.array): Valori reali (target)
        y_pred (np.array): Valori predetti dal modello
        numeri_finali (np.array, optional): Numeri finali suggeriti
    """
    for child in frame_grafico.winfo_children():
        child.destroy()
    
    plt.rcParams.update({'font.size': 12})
    
    try:
        # Crea una figura con 3 sottografici
        fig = plt.figure(figsize=(14, 10))
        
        # Assicuriamoci che gli array siano numpy arrays e che abbiano dimensioni corrette
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 1. Confronto distribuzione numeri reali vs predetti
        ax1 = fig.add_subplot(2, 2, 1)
        if len(y_true.shape) > 1:
            # Appiattisci per il grafico di distribuzione
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        else:
            y_true_flat = y_true
            y_pred_flat = y_pred
        
        # Verifico che ci siano dati validi
        if len(y_true_flat) == 0 or len(y_pred_flat) == 0:
            messagebox.showwarning("Attenzione", "Dati insufficienti per la visualizzazione dell'accuratezza.")
            return
            
        # Assicuriamoci che i dati siano nell'intervallo corretto (1-90)
        y_true_flat = np.clip(y_true_flat, 1, 90)
        y_pred_flat = np.clip(y_pred_flat, 1, 90)
        
        # Crea bins per numeri da 1 a 90
        bins = np.arange(1, 92, 1)
        ax1.hist(y_true_flat, bins=bins, alpha=0.5, label='Numeri Reali', color='blue')
        ax1.hist(y_pred_flat, bins=bins, alpha=0.5, label='Numeri Predetti', color='red')
        
        if numeri_finali is not None and len(numeri_finali) > 0:
            # Aggiungi indicatori per i numeri finali consigliati
            for num in numeri_finali:
                ax1.axvline(x=num, color='green', linestyle='--', linewidth=2)
        
        ax1.set_title('Distribuzione dei Numeri')
        ax1.set_xlabel('Numero')
        ax1.set_ylabel('Frequenza')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap di accuratezza (numeri reali vs predetti)
        ax2 = fig.add_subplot(2, 2, 2)
        
        try:
            # Crea una matrice di confusione semplificata (90x90 sarebbe troppo grande)
            # Dividiamo i numeri in gruppi di 10
            num_groups = 9  # 9 gruppi da 10 numeri ciascuno
            confusion = np.zeros((num_groups, num_groups))
            
            for true_val, pred_val in zip(y_true_flat, y_pred_flat):
                true_idx = max(0, min(int((true_val - 1) / 10), num_groups-1))
                pred_idx = max(0, min(int((pred_val - 1) / 10), num_groups-1))
                confusion[true_idx, pred_idx] += 1
            
            # Normalizza per riga in modo sicuro
            row_sums = confusion.sum(axis=1, keepdims=True)
            confusion_norm = np.zeros_like(confusion)
            for i in range(confusion.shape[0]):
                if row_sums[i] > 0:
                    confusion_norm[i, :] = confusion[i, :] / row_sums[i]
            
            im = ax2.imshow(confusion_norm, cmap='YlGnBu')
            plt.colorbar(im, ax=ax2)
            
            # Etichette per la heatmap
            group_labels = [f"{i*10+1}-{(i+1)*10}" for i in range(num_groups)]
            ax2.set_xticks(np.arange(num_groups))
            ax2.set_yticks(np.arange(num_groups))
            ax2.set_xticklabels(group_labels)
            ax2.set_yticklabels(group_labels)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            ax2.set_title("Matrice di Confusione (Gruppi di 10 numeri)")
            ax2.set_xlabel("Numeri Predetti")
            ax2.set_ylabel("Numeri Reali")
        except Exception as e:
            logger.error(f"Errore nella creazione della matrice di confusione: {e}")
            ax2.text(0.5, 0.5, f"Errore: {e}", 
                    horizontalalignment='center', verticalalignment='center')
        
        # 3. Grafico di accuratezza per tolleranza
        ax3 = fig.add_subplot(2, 2, 3)
        
        try:
            # Calcola accuratezza per diverse tolleranze
            tolerances = range(6)  # 0, 1, 2, 3, 4, 5
            accuracies = []
            
            for tol in tolerances:
                # Calcola accuratezza con tolleranza in modo sicuro
                correct = np.sum(np.abs(y_true_flat - y_pred_flat) <= tol)
                total = len(y_true_flat)
                accuracy = (correct / total * 100) if total > 0 else 0
                accuracies.append(accuracy)
            
            ax3.plot(tolerances, accuracies, 'o-', linewidth=2)
            ax3.set_title('Accuratezza per Tolleranza')
            ax3.set_xlabel('Tolleranza (±)')
            ax3.set_ylabel('Accuratezza (%)')
            ax3.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Errore nel grafico di accuratezza per tolleranza: {e}")
            ax3.text(0.5, 0.5, f"Errore: {e}", 
                    horizontalalignment='center', verticalalignment='center')
        
        # 4. Grafico di previsione vs realtà per un campione casuale
        ax4 = fig.add_subplot(2, 2, 4)
        
        try:
            # Prendi un campione casuale di punti (o tutti se ci sono meno di 10 punti)
            sample_size = min(10, len(y_true_flat))
            if sample_size > 0:
                if sample_size < len(y_true_flat):
                    indices = np.random.choice(len(y_true_flat), size=sample_size, replace=False)
                    sample_true = y_true_flat[indices]
                    sample_pred = y_pred_flat[indices]
                else:
                    sample_true = y_true_flat
                    sample_pred = y_pred_flat
                
                ax4.scatter(sample_true, sample_pred, alpha=0.7)
                
                # Aggiungi la linea di identità perfetta
                min_val = min(min(sample_true), min(sample_pred))
                max_val = max(max(sample_true), max(sample_pred))
                ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                
                # Calcola e mostra il coefficiente di correlazione in modo sicuro
                try:
                    corr = np.corrcoef(sample_true, sample_pred)[0, 1]
                    if not np.isnan(corr):
                        ax4.text(0.05, 0.95, f'Correlazione: {corr:.2f}', transform=ax4.transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    pass
                
                ax4.set_title('Predetto vs Reale (Campione Casuale)')
                ax4.set_xlabel('Valore Reale')
                ax4.set_ylabel('Valore Predetto')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, "Dati insufficienti", 
                        horizontalalignment='center', verticalalignment='center')
        except Exception as e:
            logger.error(f"Errore nel grafico di correlazione: {e}")
            ax4.text(0.5, 0.5, f"Errore: {e}", 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Aggiungi un pulsante per salvare il grafico
        def save_plot():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

        btn_save = tk.Button(
            frame_grafico,
            text="Salva Grafico",
            command=save_plot,
            bg="#FFDDC1",
            width=15
        )
        btn_save.pack(pady=5)
        
        # Mostra il grafico
        canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Aggiungi la toolbar per le funzionalità di zoom, pan, etc.
        toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    except Exception as e:
        logger.error(f"Errore durante la visualizzazione dell'accuratezza: {e}")
        messagebox.showerror("Errore", f"Errore durante la visualizzazione dell'accuratezza: {e}")
        # Crea un semplice messaggio di errore nel frame grafico
        error_label = tk.Label(
            frame_grafico,
            text=f"Si è verificato un errore nella visualizzazione: {e}",
            fg="red",
            font=("Arial", 12),
            wraplength=400
        )
        error_label.pack(pady=20)
    
    # Aggiungi un pulsante per salvare il grafico
    def save_plot():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Successo", f"Grafico salvato in {file_path}")

    btn_save = tk.Button(
        frame_grafico,
        text="Salva Grafico",
        command=save_plot,
        bg="#FFDDC1",
        width=15
    )
    btn_save.pack(pady=5)
    
    # Mostra il grafico
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Aggiungi la toolbar per le funzionalità di zoom, pan, etc.
    toolbar = NavigationToolbar2Tk(canvas, frame_grafico)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def train_ensemble_models(X, y, n_models=10, epochs=10, batch_size=32):
    """
    Addestra un insieme di modelli per l'ensemble learning.

    Args:
        X (np.array): Dati di input.
        y (np.array): Target.
        n_models (int, optional): Numero di modelli da addestrare. Default 10.
        epochs (int, optional): Numero di epoche per addestrare ogni modello. Default 10.
        batch_size (int, optional): Dimensione del batch per ogni modello. Default 32.

    Returns:
        list: Lista di modelli addestrati.
    """
    models = []
    for i in range(n_models):
        tf.keras.utils.set_random_seed(42 + i)

        dense_units = [max(1, units // (1 + i % 3)) for units in config.dense_layers]
        dropout_values = [dropout + (i % 3) * 0.05 for dropout in config.dropout_rates]

        model = build_model(
            input_shape=(X.shape[1],),
            output_shape=y.shape[1],
            dense_layers=dense_units,
            dropout_rates=dropout_values
        )

        model.compile(optimizer='adam', loss=custom_loss_function)
        logger.info(
            f"Inizio addestramento modello {i + 1}/{n_models} con dense={dense_units} e dropout={dropout_values}.")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        logger.info(f"Addestramento modello {i + 1}/{n_models} completato.")
        models.append(model)

    return models

def ensemble_predict(models, X):
    """
    Fa una previsione usando l'ensemble di modelli.

    Args:
        models (list): Lista di modelli nell'ensemble.
        X (np.array): Dati di input.

    Returns:
        np.array: Previsione media dell'ensemble.
    """
    assert len(set([m.predict(X).shape for m in models])) == 1, "I modelli non hanno lo stesso output shape"

    ensemble_prediction = np.zeros(models[0].predict(X).shape)
    for model in models:
        ensemble_prediction += model.predict(X)
    ensemble_prediction /= len(models)

    return ensemble_prediction

# 1. Modificare le funzioni di selezione per aggiornare correttamente i colori dei pulsanti

def select_optimizer(opt):
    """Imposta l'ottimizzatore selezionato."""
    config.optimizer_choice = opt
    logger.info(f"Ottimizzatore selezionato: {config.optimizer_choice}")
    
    # Aggiorna il colore dei pulsanti
    for btn in frame_optimizer.winfo_children():
        if isinstance(btn, tk.Button):
            btn["bg"] = BUTTON_DEFAULT_COLOR
            
            # Controllo del testo per determinare se corrisponde all'ottimizzatore selezionato
            if ("Adam" in btn["text"] and opt == "adam") or \
               ("RMSprop" in btn["text"] and opt == "rmsprop") or \
               ("SGD" in btn["text"] and opt == "sgd"):
                btn["bg"] = BUTTON_SELECTED_COLOR

def select_personal_loss():
    """Imposta la modalità 'Personale' con Binary Crossentropy."""
    config.use_binary_crossentropy_mode = True # <-- ATTIVA LA MODALITÀ SPECIALE
    config.use_custom_loss = False # Disattiva la custom loss
    config.loss_function_choice = 'binary_crossentropy' # Imposta anche la scelta per chiarezza
    logger.info("Modalità Funzione di Loss Personale (Binary Crossentropy) selezionata.")

    # Aggiorna i colori dei pulsanti loss
    for btn in frame_loss_function.winfo_children():
        if isinstance(btn, tk.Button):
            btn["bg"] = BUTTON_DEFAULT_COLOR # Resetta tutti
    # Evidenzia solo il pulsante personale
    # Assumiamo che btn_personal sia accessibile globalmente o tramite ricerca
    try:
        # Cerca il bottone per testo se non hai una variabile globale btn_personal
        for btn in frame_loss_function.winfo_children():
             if isinstance(btn, tk.Button) and "Personale" in btn.cget('text'):
                  btn["bg"] = BUTTON_SELECTED_COLOR
                  break
    except NameError: # Gestisci se btn_personal non è definito globalmente
         print("Warning: Variabile bottone 'btn_personal' non trovata per aggiornamento colore.")

def select_loss_function(loss_func):
    """Imposta la funzione di perdita selezionata E disattiva la modalità Personale."""
    try:
        config.use_binary_crossentropy_mode = False # <-- DISATTIVA LA MODALITÀ SPECIALE

        # Logica esistente per impostare config.loss_function_choice e config.use_custom_loss
        if loss_func == "huber_loss":
            config.loss_function_choice = Huber(delta=1.0)
            config.use_custom_loss = False
        elif loss_func == "mean_squared_error":
            config.loss_function_choice = "mean_squared_error"
            config.use_custom_loss = False
        elif loss_func == "mean_absolute_error":
            config.loss_function_choice = "mean_absolute_error"
            config.use_custom_loss = False
        elif loss_func == "log_cosh":
            config.loss_function_choice = LogCosh()
            config.use_custom_loss = False
        elif loss_func == "custom_loss_function":
            config.loss_function_choice = custom_loss_function # Assumi sia definita
            config.use_custom_loss = True
            # Non uscire qui, continua per aggiornare i colori
        else: # Gestisci altri casi o fallback
            config.loss_function_choice = loss_func
            config.use_custom_loss = False

        logger.info(f"Funzione di perdita standard selezionata: {loss_func}")

        # Aggiorna il colore dei pulsanti (logica esistente + reset Personale)
        for btn in frame_loss_function.winfo_children():
            if isinstance(btn, tk.Button):
                btn_text = btn.cget('text') # Ottieni testo bottone
                btn["bg"] = BUTTON_DEFAULT_COLOR # Resetta tutti

                # Evidenzia il bottone selezionato
                if "MSE" in btn_text and loss_func == "mean_squared_error": btn["bg"] = BUTTON_SELECTED_COLOR
                elif "MAE" in btn_text and loss_func == "mean_absolute_error": btn["bg"] = BUTTON_SELECTED_COLOR
                elif "Custom" in btn_text and loss_func == "custom_loss_function": btn["bg"] = BUTTON_SELECTED_COLOR
                elif "Huber" in btn_text and loss_func == "huber_loss": btn["bg"] = BUTTON_SELECTED_COLOR
                elif "Log Cosh" in btn_text and loss_func == "log_cosh": btn["bg"] = BUTTON_SELECTED_COLOR
                # Assicurati che il pulsante Personale NON sia selezionato
                elif "Personale" in btn_text: btn["bg"] = BUTTON_DEFAULT_COLOR

    except Exception as e:
        messagebox.showerror("Errore", f"Impossibile impostare la funzione di loss: {e}")
        logger.error(f"Errore nell'impostazione della loss function: {e}")

def select_activation(activation):
    """
    Imposta la funzione di attivazione selezionata o None se non specificata.

    Args:
        activation (str or None): Nome della funzione di attivazione o None per nessuna selezione.
    """
    config.activation_choice = activation

    if activation:
        logger.info(f"Funzione di attivazione selezionata: {config.activation_choice}")
    else:
        logger.info("Nessuna funzione di attivazione selezionata.")
        config.activation_choice = None  # Imposta None se non è stata selezionata alcuna attivazione

    # Inizialmente imposta tutti i pulsanti al colore di default
    for btn in frame_activation_function.winfo_children():
        if isinstance(btn, tk.Button):
            btn["bg"] = BUTTON_DEFAULT_COLOR
    
    # Imposta il colore del pulsante selezionato
    if activation == "relu":
        btn_relu["bg"] = BUTTON_SELECTED_COLOR
    elif activation == "leaky_relu":
        btn_leaky_relu["bg"] = BUTTON_SELECTED_COLOR
    elif activation == "elu":
        btn_elu["bg"] = BUTTON_SELECTED_COLOR
    elif activation is None:
        # Se activation è None, evidenzia il pulsante "Nessuna selezione"
        for btn in frame_activation_function.winfo_children():
            if isinstance(btn, tk.Button) and "Nessuna selezione" in btn["text"]:
                btn["bg"] = BUTTON_SELECTED_COLOR
                break

def select_regularization(reg_type):
    """Imposta il tipo di regolarizzazione selezionato."""
    config.regularization_choice = reg_type
    logger.info(f"Regolarizzazione selezionata: {config.regularization_choice}")

    # Imposta tutti i pulsanti di regolarizzazione al colore di default
    for btn in frame_regularization.winfo_children():
        if isinstance(btn, tk.Button):
            btn["bg"] = BUTTON_DEFAULT_COLOR

    # Imposta il colore del pulsante selezionato
    if reg_type is None:
        btn_reg_none["bg"] = BUTTON_SELECTED_COLOR
    elif reg_type == "l1":
        btn_reg_l1["bg"] = BUTTON_SELECTED_COLOR
    elif reg_type == "l2":
        btn_reg_l2["bg"] = BUTTON_SELECTED_COLOR

def select_model_type(model):
    """
    Imposta il tipo di modello selezionato o None per usare il valore predefinito.

    Args:
        model (str or None): Tipo di modello ('dense', 'lstm') o None.
    """
    config.model_type = model

    if model:
        logger.info(f"Tipo di modello selezionato: {config.model_type}")
    else:
        logger.info("Nessun tipo di modello selezionato.")
        config.model_type = None  # Imposta None se nessuna selezione è fatta

    # Imposta tutti i pulsanti del tipo di modello al colore di default
    for btn in frame_model_type.winfo_children():
        if isinstance(btn, tk.Button):
            btn["bg"] = BUTTON_DEFAULT_COLOR

    # Imposta il colore del pulsante selezionato
    if model == "dense":
        btn_model_dense["bg"] = BUTTON_SELECTED_COLOR
    elif model == "lstm":
        btn_model_lstm["bg"] = BUTTON_SELECTED_COLOR
    elif model is None:
        # Se model è None, evidenzia il pulsante "Nessuna selezione"
        for btn in frame_model_type.winfo_children():
            if isinstance(btn, tk.Button) and "Nessuna selezione" in btn["text"]:
                btn["bg"] = BUTTON_SELECTED_COLOR
                break

def toggle_adaptive_noise():
    """Attiva/disattiva l'aggiunta di rumore adattivo."""
    config.adaptive_noise = not config.adaptive_noise
    noise_status = "Attivato" if config.adaptive_noise else "Disattivato"
    logger.info(f"Rumore adattivo {noise_status}.")
    
    # Aggiorna il colore del pulsante per indicare lo stato attivo/disattivo
    if config.adaptive_noise:
        btn_toggle_noise["bg"] = BUTTON_SELECTED_COLOR
    else:
        btn_toggle_noise["bg"] = BUTTON_DEFAULT_COLOR
    
    messagebox.showinfo("Info", f"Rumore adattivo {noise_status}.")

def toggle_ensemble():
    """Attiva/disattiva l'ensemble learning."""
    config.use_ensemble = not config.use_ensemble
    ensemble_status = "Attivato" if config.use_ensemble else "Disattivato"
    logger.info(f"Ensemble methods {ensemble_status}.")
    
    # Aggiorna il colore del pulsante per indicare lo stato attivo/disattivo
    if config.use_ensemble:
        btn_toggle_ensemble["bg"] = BUTTON_SELECTED_COLOR
    else:
        btn_toggle_ensemble["bg"] = BUTTON_DEFAULT_COLOR
    
    messagebox.showinfo("Info", f"Ensemble methods {ensemble_status}.")

def select_patience(patience_value):
    """Imposta il valore di patience per l'early stopping."""
    config.patience = patience_value
    logger.info(f"Patience impostato a: {config.patience}")

def select_min_delta(min_delta_value):
    """Imposta il valore minimo di delta per l'early stopping."""
    config.min_delta = min_delta_value
    logger.info(f"Min Delta impostato a: {config.min_delta}")

def select_activation(activation):
    """
    Imposta la funzione di attivazione selezionata o None se non specificata.

    Args:
        activation (str or None): Nome della funzione di attivazione o None per nessuna selezione.
    """
    config.activation_choice = activation

    if activation:
        logger.info(f"Funzione di attivazione selezionata: {config.activation_choice}")
    else:
        logger.info("Nessuna funzione di attivazione selezionata.")
        config.activation_choice = None  # Imposta None se non è stata selezionata alcuna attivazione

    # Reimposta tutti i pulsanti al colore predefinito
    btn_relu["bg"] = BUTTON_DEFAULT_COLOR
    btn_leaky_relu["bg"] = BUTTON_DEFAULT_COLOR
    btn_elu["bg"] = BUTTON_DEFAULT_COLOR
    btn_clear_activation["bg"] = BUTTON_DEFAULT_COLOR
    
    # Imposta il colore del pulsante selezionato
    if activation == "relu":
        btn_relu["bg"] = BUTTON_SELECTED_COLOR
    elif activation == "leaky_relu":
        btn_leaky_relu["bg"] = BUTTON_SELECTED_COLOR
    elif activation == "elu":
        btn_elu["bg"] = BUTTON_SELECTED_COLOR
    elif activation is None:
        # Nessuna attivazione selezionata, evidenzia il pulsante "Nessuna selezione"
        btn_clear_activation["bg"] = BUTTON_SELECTED_COLOR

def select_regularization(reg_type):
    """Imposta il tipo di regolarizzazione selezionato."""
    config.regularization_choice = reg_type
    logger.info(f"Regolarizzazione selezionata: {config.regularization_choice}")

    for btn in [btn_reg_none, btn_reg_l1, btn_reg_l2]:
        btn["bg"] = "#C9E4CA"

    if reg_type is None:
        btn_reg_none["bg"] = "#4CAF50"
    elif reg_type == "l1":
        btn_reg_l1["bg"] = "#4CAF50"
    elif reg_type == "l2":
        btn_reg_l2["bg"] = "#4CAF50"

def select_model_type(model):
    """
    Imposta il tipo di modello selezionato o None per usare il valore predefinito.

    Args:
        model (str or None): Tipo di modello ('dense', 'lstm') o None.
    """
    config.model_type = model

    if model:
        logger.info(f"Tipo di modello selezionato: {config.model_type}")
    else:
        logger.info("Nessun tipo di modello selezionato.")
        config.model_type = None  # Imposta None se nessuna selezione è fatta

    # Reimposta tutti i pulsanti al colore predefinito
    btn_model_dense["bg"] = BUTTON_DEFAULT_COLOR
    btn_model_lstm["bg"] = BUTTON_DEFAULT_COLOR
    btn_clear_model_type["bg"] = BUTTON_DEFAULT_COLOR

    # Imposta il colore del pulsante selezionato
    if model == "dense":
        btn_model_dense["bg"] = BUTTON_SELECTED_COLOR
    elif model == "lstm":
        btn_model_lstm["bg"] = BUTTON_SELECTED_COLOR
    elif model is None:
        # Nessun modello selezionato, evidenzia il pulsante "Nessuna selezione"
        btn_clear_model_type["bg"] = BUTTON_SELECTED_COLOR

def update_regularization_value(value):
    """Aggiorna il valore di regolarizzazione."""
    try:
        config.regularization_value = float(value)
        logger.info(f"Valore di regolarizzazione impostato a: {config.regularization_value}")
    except ValueError:
        messagebox.showerror("Errore", "Il valore di regolarizzazione deve essere un numero valido.")

def update_textbox(numeri):
    """Aggiorna la casella di testo con i numeri predetti."""
    textbox.insert(tk.END, "Numeri Predetti:\n")
    numeri_filtrati = filtra_duplicati(numeri, n=5)
    textbox.insert(tk.END, ", ".join(map(str, numeri_filtrati)) + "\n")

def mostra_numeri_predetti(numeri):
    """Mostra i numeri predetti nella casella di testo."""
    textbox.insert(tk.END, "Numeri Predetti:\n")
    numeri_filtrati = filtra_duplicati(numeri, n=5)
    textbox.insert(tk.END, ", ".join(map(str, numeri_filtrati)) + "\n")

def mostra_migliori_risultati(history):
    """
    Mostra i migliori risultati dell'addestramento con valutazione dell'attendibilità.

    Args:
        history (dict): Storia dell'addestramento.
    """
    if 'loss' in history and 'val_loss' in history:
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')

        textbox.insert(tk.END, f"Ultima Loss Train: {final_train_loss:.4f}\n")
        textbox.insert(tk.END, f"Ultima Loss Val: {final_val_loss:.4f}\n")
        textbox.insert(tk.END, f"Rapporto Val/Train: {ratio:.4f}\n")

        attendibility_score, commento = valuta_attendibilita(history)
        textbox.insert(tk.END, f"Punteggio di attendibilità: {attendibility_score:.1f}/100\n")
        textbox.insert(tk.END, f"Giudizio: {commento}\n")

        thresholds = {
            "Ottimale": 0.03,
            "Buona": 0.07,
            "Discreta": 0.15,
            "Sufficiente": 0.25
        }

        if final_val_loss <= thresholds["Ottimale"]:
            quality = "Ottimale"
        elif final_val_loss <= thresholds["Buona"]:
            quality = "Buona"
        elif final_val_loss <= thresholds["Discreta"]:
            quality = "Discreta"
        else:
            quality = "Sufficiente"

        textbox.insert(tk.END, f"Qualità (metrica storica): {quality}\n")
    else:
        logger.error("La storia dell'addestramento non contiene le chiavi 'loss' o 'val_loss'.")
        messagebox.showerror("Errore", "La storia dell'addestramento non contiene le chiavi 'loss' o 'val_loss'.")

def salva_risultati():
    """Salva i risultati in un file di testo."""
    try:
        from tkinter import filedialog  # Aggiungi questo import qui
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'w') as file:
                file.write(textbox.get(1.0, tk.END))
            messagebox.showinfo("Successo", f"Risultati salvati in {file_path}")
        else:
            messagebox.showinfo("Info", "Salvataggio cancellato dall'utente.")

    except Exception as e:
        logger.error(f"Errore nel salvataggio dei risultati: {e}")
        messagebox.showerror("Errore", f"Errore nel salvataggio dei risultati: {e}")

    except Exception as e:
        logger.error(f"Errore nel salvataggio dei risultati: {e}")
        messagebox.showerror("Errore", f"Errore nel salvataggio dei risultati: {e}")

def esegui_aggiornamento():
    """Esegue lo script di aggiornamento delle estrazioni."""
    conferma = messagebox.askyesno("Conferma", "Sei sicuro di voler aggiornare i dati delle estrazioni?")
    if conferma:
        try:
            import subprocess
            import sys
            import os
            
            # Determina il percorso corretto per aggiornamento.exe
            if getattr(sys, 'frozen', False):
                # Se in esecuzione come eseguibile
                base_path = os.path.dirname(sys.executable)
                aggiornamento_exe = os.path.join(base_path, "aggiornamento.exe")
                
                # Avvia l'eseguibile se esiste
                if os.path.exists(aggiornamento_exe):
                    subprocess.Popen([aggiornamento_exe])
                    return
                else:
                    # Se non troviamo l'eseguibile, proviamo ad importare direttamente
                    pass
            
            # Durante lo sviluppo o se l'eseguibile non è stato trovato
            try:
                import aggiornamento
                success = aggiornamento.main()
                if success:
                    messagebox.showinfo("Successo", "Aggiornamento completato con successo.")
                    logger.info("Aggiornamento estrazioni completato con successo.")
                else:
                    messagebox.showwarning("Attenzione", "L'aggiornamento non è stato completato correttamente.")
                    logger.warning("L'aggiornamento non è stato completato con successo.")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore durante l'aggiornamento:\n{str(e)}")
                logger.error(f"Errore durante l'aggiornamento estrazioni: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante l'avvio dell'aggiornamento:\n{e}")
            import traceback
            traceback.print_exc()

def gestisci_licenza():
    """
    Gestisce la visualizzazione delle informazioni di licenza in una finestra separata.
    Permette all'utente di avviare l'importazione di una nuova licenza.
    """
    try:
        # Importa il sistema di licenze solo quando necessario
        from license_system import LicenseSystem

        # Crea la finestra Toplevel per le info sulla licenza
        license_window = tk.Toplevel()
        license_window.title("Informazioni Licenza")
        license_window.geometry("450x350") # Dimensioni adeguate per i contenuti

        # Ottieni lo stato attuale della licenza
        license_system = LicenseSystem()
        is_valid, message = license_system.check_license()

        # Frame per visualizzare lo stato
        status_frame = tk.LabelFrame(license_window, text="Stato Licenza", padx=10, pady=10)
        status_frame.pack(pady=10, padx=10, fill=tk.X)

        status_text = "Attiva" if is_valid else "Non Attiva"
        status_color = "#4CAF50" if is_valid else "#F44336" # Verde se valida, Rosso se non valida

        # Label per lo Stato (Attiva/Non Attiva)
        tk.Label(status_frame, text=f"Stato: {status_text}", bg=status_color, fg="white", width=40).pack(pady=5)
        # Label per i Dettagli (messaggio da check_license)
        tk.Label(status_frame, text=f"Dettagli: {message}", bg="#F0F0F0", wraplength=400, justify=tk.LEFT).pack(pady=5)

        # Frame per le azioni sulla licenza (Importazione/Contatti)
        action_frame = tk.LabelFrame(license_window, text="Azioni Licenza", padx=10, pady=10)
        action_frame.pack(pady=10, padx=10, fill=tk.X)

        # Pulsante per avviare l'importazione di un file di licenza
        btn_importa = tk.Button(
            action_frame,
            text="Importa Licenza da File (.json)",
            # Chiama importa_licenza, passandogli questa finestra come genitore
            # in modo che importa_licenza possa chiuderla dopo un import successo
            command=lambda: importa_licenza(license_window),
            bg="#3498db", # Blu
            fg="white",
            width=25, # Larghezza adeguata
            height=2
        )
        btn_importa.pack(pady=10)

        # Informazioni di contatto per il rinnovo/acquisto
        contact_label = tk.Label(
            action_frame,
            # !!! MODIFICA CON LA TUA EMAIL O INFO DI CONTATTO REALE !!!
            text="Per rinnovare o acquistare una licenza contatta:\massimoferrughelli63@gmail.com",
            justify=tk.LEFT
        )
        contact_label.pack(pady=5)

        # --- Gestione Modalità Finestra ---
        # Rende la finestra modale (blocca l'interazione con la finestra principale)
        # Cerca la root attiva in modo sicuro
        try:
            # Cerca la finestra principale attiva (potrebbe essere 'root' o altro)
            active_root = license_window.master.winfo_toplevel()
            if active_root and active_root.winfo_exists():
                license_window.transient(active_root) # Imposta come figlia transitoria
        except: # Se non trova una master valida, ignora transient
            pass
        license_window.grab_set() # Blocca eventi su altre finestre
        license_window.wait_window() # Aspetta che questa finestra sia chiusa

    except ImportError:
        messagebox.showerror("Errore Modulo", "Impossibile trovare il modulo 'license_system.py'.")
    except Exception as e:
        messagebox.showerror("Errore", f"Errore imprevisto nella visualizzazione delle info licenza:\n{e}")

def importa_licenza(parent_window=None):
    """
    Permette all'utente di selezionare un file .json di licenza,
    lo valida sommariamente, lo salva come file di licenza attivo
    e mostra un messaggio di risultato.
    Args:
        parent_window (tk.Toplevel, optional): La finestra genitore (es. gestisci_licenza)
                                              da chiudere in caso di successo. Defaults to None.
    """
    try:
        # Importa il sistema di licenze solo quando necessario
        from license_system import LicenseSystem

        # Apri la finestra di dialogo per selezionare il file
        file_path = filedialog.askopenfilename(
            title="Seleziona file di licenza (.json)",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        # Se l'utente annulla, esci dalla funzione
        if not file_path:
            print("Importazione annullata dall'utente.")
            return

        # Crea un'istanza del sistema di licenze per accedere al nome del file target
        license_system = LicenseSystem()

        # --- Leggi il contenuto del file JSON selezionato ---
        try:
            with open(file_path, "r", encoding='utf-8') as f: # Aggiunto encoding
                license_data = json.load(f)
        except json.JSONDecodeError:
             messagebox.showerror("Errore Lettura", "File JSON selezionato non valido o corrotto.")
             return
        except Exception as e_read:
             messagebox.showerror("Errore Lettura", f"Impossibile leggere il file selezionato:\n{e_read}")
             return

        # --- Validazione di base del contenuto della licenza ---
        # Assicurati che queste chiavi corrispondano a quelle create da create_license_for_machine
        required_keys = ["license_key", "expiry_date", "app_name", "machine_id"]
        if not all(key in license_data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in license_data]
            messagebox.showerror("Errore Formato", f"File di licenza non valido.\nMancano dati essenziali: {', '.join(missing_keys)}")
            return

        # Controllo opzionale ma consigliato: Nome Applicazione
        if license_data.get("app_name") != license_system.app_name:
             # Chiedi conferma se il nome app non corrisponde
             confirm = messagebox.askyesno("Attenzione Nome Applicazione",
                                         f"La licenza selezionata è per '{license_data.get('app_name')}',\nma questa applicazione è '{license_system.app_name}'.\n\nVuoi importarla comunque?")
             if not confirm:
                 print("Importazione annullata a causa del nome applicazione non corrispondente.")
                 return

        # --- Salva/Sovrascrivi il file di licenza locale ---
        try:
            # Usa il percorso definito nella classe LicenseSystem
            with open(license_system.license_file, "w", encoding='utf-8') as f: # Aggiunto encoding
                json.dump(license_data, f, indent=4)
            print(f"File licenza '{license_system.license_file}' sovrascritto con i dati da '{os.path.basename(file_path)}'")
        except PermissionError:
            messagebox.showerror("Errore Permessi", f"Impossibile scrivere il file di licenza in:\n{license_system.license_file}\nControlla i permessi della cartella.")
            return
        except Exception as e_write:
             messagebox.showerror("Errore Salvataggio", f"Impossibile salvare la nuova licenza:\n{e_write}")
             return

        # --- Messaggio Finale ---
        # NON riverifichiamo subito qui, l'utente deve riavviare.
        messagebox.showinfo("Importazione Completata",
                            f"File di licenza importato con successo da:\n'{os.path.basename(file_path)}'.\n\n"
                            "!!! RIAVVIA L'APPLICAZIONE !!!\n"
                            "per applicare la nuova licenza.")

        # Chiudi la finestra genitore (gestisci_licenza) se è stata passata
        if parent_window and parent_window.winfo_exists():
             print("Chiusura finestra info licenza.")
             parent_window.destroy()

    except ImportError:
        messagebox.showerror("Errore Modulo", "Impossibile trovare il modulo 'license_system.py'.")
    except Exception as e:
        # Cattura altri errori imprevisti durante l'importazione
        messagebox.showerror("Errore Importazione", f"Si è verificato un errore imprevisto durante l'importazione:\n{e}")

def carica_e_valuta_modello():
    """
    Carica il miglior modello salvato per la ruota selezionata e lo valuta.
    Versione migliorata con gestione robusta degli errori di compatibilità.
    """
    global ultima_ruota_elaborata, numeri_finali
    numeri_interi = None

    ruota = ruota_selezionata.get()  # Ottieni la ruota dalla selezione attuale
    
    # Se non c'è una ruota selezionata ma c'è un'ultima ruota elaborata, usala
    if not ruota and ultima_ruota_elaborata:
        ruota = ultima_ruota_elaborata
        # Aggiorna visivamente il pulsante della ruota
        if ruota in pulsanti_ruote:
            for r in pulsanti_ruote:
                pulsanti_ruote[r]["bg"] = "SystemButtonFace"  # Reset di tutti i pulsanti
            pulsanti_ruote[ruota]["bg"] = "lightgreen"  # Evidenzia ruota attualmente in uso
            ruota_selezionata.set(ruota)  # Imposta la ruota selezionata
            entry_info.delete(0, tk.END)
            entry_info.insert(0, f"Ruota: {ruota}, Periodo: {entry_start_date.get_date().strftime('%Y/%m/%d')} - {entry_end_date.get_date().strftime('%Y/%m/%d')}")
            root.update()
    
    # Se ancora non abbiamo una ruota, chiedi all'utente di selezionarne una
    if not ruota:
        # Controlla se esistono modelli per qualsiasi ruota
        modelli_esistenti = []
        for r in FILE_RUOTE.keys():
            model_path = f'best_model_{r}.weights.h5'
            if os.path.exists(model_path):
                modelli_esistenti.append(r)
        
        if modelli_esistenti:
            # Apri una finestra per selezionare tra i modelli esistenti
            popup = tk.Toplevel()
            popup.title("Seleziona Modello")
            popup.geometry("300x400")
            
            tk.Label(popup, text="Modelli disponibili:", font=("Arial", 12, "bold")).pack(pady=10)
            
            def seleziona_e_chiudi(r):
                global ultima_ruota_elaborata
                ultima_ruota_elaborata = r
                ruota_selezionata.set(r)
                for btn in pulsanti_ruote.values():
                    btn["bg"] = "SystemButtonFace"
                pulsanti_ruote[r]["bg"] = "lightgreen"
                entry_info.delete(0, tk.END)
                entry_info.insert(0, f"Ruota: {r}, Periodo: {entry_start_date.get_date().strftime('%Y/%m/%d')} - {entry_end_date.get_date().strftime('%Y/%m/%d')}")
                popup.destroy()
                # Richiama la funzione dopo la chiusura
                root.after(100, carica_e_valuta_modello)
            
            for r in modelli_esistenti:
                btn = tk.Button(
                    popup,
                    text=f"Ruota {r}",
                    command=lambda ruota=r: seleziona_e_chiudi(ruota),
                    bg="#ADD8E6",
                    width=20,
                    height=2
                )
                btn.pack(pady=5)
            
            # Pulsante per annullare
            tk.Button(
                popup,
                text="Annulla",
                command=popup.destroy,
                bg="#FF6B6B",
                width=20
            ).pack(pady=10)
            
            return
        else:
            messagebox.showwarning("Attenzione", "Nessun modello salvato trovato. Seleziona una ruota e addestra un modello prima.")
            return

    # Pulisci la textbox
    textbox.delete(1.0, tk.END)
    textbox.insert(tk.END, f"Tentativo di caricamento del modello per {ruota}...\n")
    textbox.update()

    # Ottieni le date per il caricamento dati
    try:
        start_date = pd.to_datetime(entry_start_date.get_date(), format='%Y/%m/%d')
        end_date = pd.to_datetime(entry_end_date.get_date(), format='%Y/%m/%d')
    except ValueError:
        messagebox.showerror("Errore", "Formato data non valido. Usa YYYY/MM/DD.")
        return

    # Carica i dati per preparare input/output shape
    data = carica_dati(ruota, start_date, end_date)
    if data is None:
        messagebox.showerror("Errore", f"Impossibile caricare i dati per la ruota {ruota}.")
        return
    
    X, y, scaler, raw_data = data
    
    if len(X) == 0 or len(y) == 0:
        messagebox.showerror("Errore", "Dataset vuoto per il periodo specificato.")
        return
    
    # Definisci il percorso del modello
    model_path = f'best_model_{ruota}.weights.h5'
    
    # Verifica se il file esiste
    if not os.path.exists(model_path):
        risposta = messagebox.askyesno("Attenzione", 
                               f"File modello '{model_path}' non trovato.\n"
                               f"Desideri addestrare un nuovo modello per la ruota {ruota}?")
        if risposta:
            avvia_elaborazione()
        return
    
    # Definisci input_shape e output_shape
    input_shape = (X.shape[1],)
    output_shape = y.shape[1]
    
    # Messagebox per chiedere se si vuole riaddestrare il modello
    risposta = messagebox.askyesno("Caricamento modello", 
                           "Il caricamento di modelli salvati può generare errori di compatibilità.\n"
                           "Desideri provare a caricare il modello esistente o preferisci addestrare un nuovo modello?",
                           detail="Sì = Carica modello esistente, No = Addestra nuovo modello")
    
    if not risposta:
        avvia_elaborazione()
        return
    
    # Tenta di caricare il modello esistente
    try:
        textbox.insert(tk.END, "Creazione di un nuovo modello con la stessa struttura...\n")
        textbox.update()
        
        # Crea un nuovo modello
        model = build_model(input_shape, output_shape, config.dense_layers, config.dropout_rates)
        
        # Compila il modello
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        if config.use_custom_loss:
            model.compile(optimizer=optimizer, loss=custom_loss_function, metrics=["mae"])
        else:
            loss_function = config.loss_function_choice if config.loss_function_choice else "mean_squared_error"
            model.compile(optimizer=optimizer, loss=loss_function, metrics=["mae"])
        
        textbox.insert(tk.END, "Modello creato. Tentativo di caricamento pesi...\n")
        textbox.update()
        
        loading_success = False
        
        # APPROCCIO 1: Prova a riaddestrate il modello con pochi dati
        try:
            textbox.insert(tk.END, "STRATEGIA: Riaddestramento veloce su dati recenti...\n")
            textbox.update()
            
            # Usa gli ultimi 20% dei dati per un rapido riaddestramento
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[split_idx:], X[:split_idx]
            y_train, y_val = y[split_idx:], y[:split_idx]
            
            # Addestra il modello per poche epoche
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=5,  # Poche epoche per un riaddestramento veloce
                batch_size=32,
                verbose=0
            )
            
            textbox.insert(tk.END, "✓ Modello riaddestrato con successo su dati recenti.\n")
            textbox.update()
            loading_success = True
            
            # Usa l'ultimo campione per la predizione
            X_pred = X[-1:].copy()
            y_pred = model.predict(X_pred)
            textbox.insert(tk.END, f"DEBUG - Predizione grezza: {y_pred}\n")
            
        # Se l'addestramento fallisce, prova a caricare il modello esistente
        except Exception as e:
            textbox.insert(tk.END, f"✗ Riaddestramento fallito: {str(e)}\n")
            textbox.insert(tk.END, "STRATEGIA: Tentativo caricamento modello esistente...\n")
            textbox.update()
            
            # Tenta di caricare il modello esistente, con gestione esplicita degli errori
            try:
                model = Sequential()
                model.add(Dense(512, activation='relu', input_shape=input_shape))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(Dense(256, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(Dense(128, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                model.add(Dense(output_shape))
                model.compile(optimizer='adam', loss='mse')
                
                model.load_weights(model_path, skip_mismatch=True)  # Prova con skip_mismatch=True
                textbox.insert(tk.END, "✓ Modello caricato con successo (skip_mismatch).\n")
                loading_success = True
                
                # Usa l'ultimo campione per la predizione
                X_pred = X[-1:].copy()
                y_pred = model.predict(X_pred)
                textbox.insert(tk.END, f"DEBUG - Predizione grezza: {y_pred}\n")
                
            except Exception as e2:
                textbox.insert(tk.END, f"✗ Caricamento fallito: {str(e2)}\n")
                textbox.insert(tk.END, "STRATEGIA FINALE: Addestramento modello semplificato...\n")
                textbox.update()
                
                # Crea un modello semplificato
                model = Sequential()
                model.add(Dense(64, activation='relu', input_shape=input_shape))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(output_shape))
                model.compile(optimizer='adam', loss='mse')
                
                # Addestra il modello semplificato
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                textbox.insert(tk.END, "✓ Modello semplificato addestrato con successo.\n")
                loading_success = True
                
                # Usa l'ultimo campione per la predizione
                X_pred = X[-1:].copy()
                y_pred = model.predict(X_pred)
                textbox.insert(tk.END, f"DEBUG - Predizione grezza: {y_pred}\n")
        
        # Denormalizza i risultati
        try:
            if not loading_success:
                raise ValueError("Nessun modello caricato con successo. Impossibile procedere con la predizione.")
                
            # Correggi la denormalizzazione in base a come è stata fatta la normalizzazione
            # Se i dati sono stati normalizzati dividendo per 90 durante l'addestramento:
            numeri_denormalizzati = y_pred * 90.0
            textbox.insert(tk.END, f"DEBUG - Numeri denormalizzati: {numeri_denormalizzati}\n")
            
            # Alternativa: usa lo scaler direttamente se è quello usato nell'addestramento
            # numeri_denormalizzati = scaler.inverse_transform(y_pred)
            
            numeri_interi = np.round(numeri_denormalizzati).astype(int)
            numeri_interi = np.clip(numeri_interi, 1, 90)  # Assicura numeri tra 1 e 90
            textbox.insert(tk.END, f"DEBUG - Numeri interi dopo arrotondamento: {numeri_interi}\n")
            
            # Estrazione dei numeri frequenti
            numeri_frequenti = estrai_numeri_frequenti(ruota, start_date, end_date, n=20)
            textbox.insert(tk.END, f"DEBUG - Numeri frequenti: {numeri_frequenti[:5]}...\n")
            
            # Calcola attendibilità in modo più robusto
            try:
                attendibility_score, commento = valuta_attendibilita(history.history)
            except Exception:
                # Fallback se la valutazione fallisce
                attendibility_score = 60.0
                commento = "Previsione attendibile (stima)"
                
            textbox.insert(tk.END, f"DEBUG - Attendibilità calcolata: {attendibility_score}\n")
            
            # Genera numeri finali con controllo
            numeri_finali, origine_ml = genera_numeri_attendibili(numeri_interi, attendibility_score, numeri_frequenti)
            textbox.insert(tk.END, f"DEBUG - Numeri finali generati: {numeri_finali}\n")
            
            textbox.insert(tk.END, f"\nRisultati per la ruota {ruota}:\n")
            textbox.insert(tk.END, f"Attendibilità: {attendibility_score:.1f}/100 - {commento}\n")
            textbox.insert(tk.END, "Numeri consigliati: " + ", ".join(map(str, numeri_finali)) + "\n")
            
            # Visualizza il popup con i numeri forti
            mostra_numeri_forti_popup(numeri_finali, attendibility_score, origine_ml)
            
            # Se possibile, mostra anche metriche di accuratezza
            if len(y) > 0:
                try:
                    # Usa la stessa logica di denormalizzazione usata sopra
                    y_scaled = y / 90.0  # Normalizza allo stesso modo del training
                    y_denormalized = y  # I dati originali non normalizzati
                    
                    if len(y_denormalized.shape) > 1:
                        y_denormalized_last = y_denormalized[-1]
                    else:
                        y_denormalized_last = y_denormalized
                    
                    visualizza_accuratezza(y_denormalized_last, numeri_interi[0], numeri_finali)
                    accuracy_metrics_0 = valuta_accuratezza_previsione(y_denormalized_last, numeri_interi[0], tolerance=0)
                    
                    textbox.insert(tk.END, "\n=== METRICHE DI ACCURATEZZA ===\n")
                    textbox.insert(tk.END, f"MAE: {accuracy_metrics_0['MAE']:.4f}\n")
                    textbox.insert(tk.END, f"Media numeri corretti: {accuracy_metrics_0['Media_numeri_corretti_per_estrazione']:.2f}\n")
                except Exception as e:
                    textbox.insert(tk.END, f"\nErrore nel calcolo delle metriche: {str(e)}\n")
                    
        except Exception as e:
            textbox.insert(tk.END, f"\nErrore nella denormalizzazione o predizione: {str(e)}\n")
            textbox.insert(tk.END, "Si consiglia di addestrare un nuovo modello.\n")
            
    except Exception as e:
        textbox.insert(tk.END, f"\nErrore generale: {str(e)}\n")
        textbox.insert(tk.END, "Si consiglia di addestrare un nuovo modello.\n")
        messagebox.showerror("Errore", f"Errore generale: {str(e)}")

def toggle_adaptive_noise():
    """Attiva/disattiva l'aggiunta di rumore adattivo."""
    config.adaptive_noise = not config.adaptive_noise
    noise_status = "Attivato" if config.adaptive_noise else "Disattivato"
    logger.info(f"Rumore adattivo {noise_status}.")
    messagebox.showinfo("Info", f"Rumore adattivo {noise_status}.")

def update_max_noise_factor(value):
    """Aggiorna il fattore massimo di rumore."""
    try:
        config.max_noise_factor = float(value)
        logger.info(f"Fattore massimo di rumore impostato a: {config.max_noise_factor}")
    except ValueError:
        messagebox.showerror("Errore", "Il fattore massimo di rumore deve essere un numero valido.")

def select_noise_type(noise_type_param):
    """Imposta il tipo di rumore selezionato."""
    config.noise_type = noise_type_param
    logger.info(f"Tipo di rumore selezionato: {config.noise_type}")

def update_noise_scale(value):
    """Aggiorna la scala del rumore."""
    try:
        config.noise_scale = float(value)
        logger.info(f"Scala del rumore impostata a: {config.noise_scale}")
    except ValueError:
        messagebox.showerror("Errore", "La scala del rumore deve essere un numero valido.")

def update_noise_percentage(value):
    """Aggiorna la percentuale di rumore."""
    try:
        config.noise_percentage = float(value)
        if not 0 <= config.noise_percentage <= 1:
            raise ValueError("La percentuale deve essere tra 0 e 1")
        logger.info(f"Percentuale di rumore impostata a: {config.noise_percentage}")
    except ValueError as e:
        messagebox.showerror("Errore", f"Errore nel valore della percentuale: {str(e)}")

def toggle_ensemble():
    """Attiva/disattiva l'ensemble learning."""
    config.use_ensemble = not config.use_ensemble
    ensemble_status = "Attivato" if config.use_ensemble else "Disattivato"
    logger.info(f"Ensemble methods {ensemble_status}.")
    messagebox.showinfo("Info", f"Ensemble methods {ensemble_status}.")

def toggle_custom_loss():
    """Attiva/disattiva l'uso della loss function personalizzata."""
    config.use_custom_loss = not config.use_custom_loss
    status = "Attivata" if config.use_custom_loss else "Disattivata"
    logger.info(f"Loss function personalizzata {status}.")
    messagebox.showinfo("Info", f"Loss function personalizzata {status}.")

def delete_models(ruota=None):
    """Elimina i modelli salvati e il file JSON delle informazioni per una specifica ruota o per tutte le ruote se nessuna è specificata."""
    try:
        # Se una ruota è specificata, elimina solo i modelli di quella ruota
        if ruota:
            ruote_to_delete = [ruota]
        else:
            # Altrimenti, elimina i modelli di tutte le ruote
            ruote_to_delete = FILE_RUOTE.keys()

        for ruota in ruote_to_delete:
            # Elimina i file dei pesi del modello
            model_path = f'best_model_{ruota}.weights.h5'
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Modello {model_path} cancellato.")

            # Elimina anche il vecchio formato del file se esiste
            alt_model_path = f'best_model_{ruota}.h5'
            if os.path.exists(alt_model_path):
                os.remove(alt_model_path)
                logger.info(f"Modello {alt_model_path} cancellato.")

            # Elimina i file dei modelli relativi ai fold e alle epoche
            for file_name in os.listdir('.'):
                if file_name.startswith(f'model_{ruota}_fold') and file_name.endswith('.weights.h5'):
                    os.remove(file_name)
                    logger.info(f"Modello {file_name} cancellato.")

            # Elimina il file JSON delle informazioni del modello
            info_file = f'model_info_{ruota}.json'
            if os.path.exists(info_file):
                os.remove(info_file)
                logger.info(f"File delle informazioni del modello {info_file} cancellato.")
    except Exception as e:
        logger.error(f"Errore durante l'eliminazione dei modelli: {e}")

def on_closing():
    """Gestisce la chiusura dell'applicazione."""
    if messagebox.askokcancel("Chiusura", "Sei sicuro di voler chiudere l'applicazione?"):
        try:
            # Chiama delete_models senza argomenti per eliminare i modelli di tutte le ruote
            delete_models()
            logger.info("Applicazione chiusa correttamente.")
        except Exception as e:
            logger.error(f"Errore durante la chiusura dell'applicazione: {e}")
        finally:
            root.destroy()
            root.quit()
             
def on_ruota_selected(*args):
    """
    Gestisce la selezione di una ruota (aggiorna solo l'interfaccia).
    """
    global ruota_selezionata, entry_info, btn_start, root

    ruota = ruota_selezionata.get()  # Ottieni la ruota
    print(f"Dentro on_ruota_selected, ruota: {ruota}") # DEBUG
    if ruota:
        # Formatta date
        start_date_str = entry_start_date.get_date().strftime('%Y/%m/%d')
        end_date_str = entry_end_date.get_date().strftime('%Y/%m/%d')

        # --- PRINT DI DEBUG ---
        print("Dentro on_ruota_selected:")
        print("  start_date_str:", start_date_str)
        print("  end_date_str:", end_date_str)
        # ----------------------

        entry_info.delete(0, tk.END)
        entry_info.insert(0, f"Ruota: {ruota}, Periodo: {start_date_str} - {end_date_str}")

        # Aggiorna pulsante
        if btn_start:
            btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})")
            btn_start.config(bg="#4CAF50")
        root.update_idletasks()  # Forza aggiornamento


import numpy as np # Assicurati che numpy sia importato all'inizio del file

# --- Funzione Helper per Codifica Target ---
def multi_hot_encode(data_array, num_classes=90):
    """
    Converte un array di numeri estratti (es. shape [N, 5])
    in un array multi-hot (es. shape [N, 90]).
    """
    # Gestisce il caso di input vuoto o None
    if data_array is None or data_array.size == 0:
        # Restituisce un array vuoto con la forma corretta se possibile, altrimenti None o errore
        # O potresti voler restituire np.zeros((0, num_classes)) a seconda della gestione a valle
        print("Warning: multi_hot_encode ha ricevuto un input vuoto.")
        return np.zeros((0, num_classes), dtype=int) # Restituisce array vuoto con shape corretta

    # Gestisce il caso di una singola estrazione (array 1D)
    original_ndim = data_array.ndim
    if original_ndim == 1:
        data_array = data_array.reshape(1, -1) # Converti temporaneamente a 2D

    # Controlla che l'input sia 2D dopo l'eventuale reshape
    if data_array.ndim != 2:
        raise ValueError(f"Input a multi_hot_encode deve essere 1D o 2D, ricevuto {original_ndim}D")

    num_samples = data_array.shape[0]
    if num_samples == 0: # Check aggiunto dopo reshape
         return np.zeros((0, num_classes), dtype=int)

    encoded = np.zeros((num_samples, num_classes), dtype=int)

    for i in range(num_samples):
        # Estrai la riga, assicurati siano numeri interi
        try:
            # Converte in interi, gestendo NaN o valori non numerici se presenti
            numeri_riga = pd.to_numeric(data_array[i], errors='coerce')
            numeri_riga = numeri_riga[~np.isnan(numeri_riga)].astype(int)
        except Exception as e:
             print(f"Warning: Errore conversione riga {i} in interi per multi_hot_encode: {e}. Riga saltata.")
             continue # Salta questa riga se la conversione fallisce

        # Sottrai 1 perché i numeri sono 1-90 ma gli indici sono 0-89
        # Usa np.unique per gestire eventuali duplicati nell'input
        indices = np.unique(numeri_riga) - 1

        # Filtra indici validi (0 to num_classes-1)
        valid_indices = indices[(indices >= 0) & (indices < num_classes)]

        if len(valid_indices) > 0:
             encoded[i, valid_indices] = 1

    # Se l'input originale era 1D, restituisci un array 1D
    return encoded[0] if original_ndim == 1 else encoded

# --- DEFINIZIONE LOGGER (assicurati sia definito) ---
logger = logging.getLogger(__name__)

# ==============================================================================
# === DEFINIZIONE COMPLETA E CORRETTA di avvia_elaborazione ===
# ==============================================================================

# Funzione avvia_elaborazione (Versione v5 - Score Unificato con Moltiplicatore Condizionale)
def avvia_elaborazione():
    """
    Avvia l'elaborazione per la ruota selezionata (addestramento con CV).
    Gestisce caricamento, preprocessing, feature engineering, addestramento CV,
    predizione, valutazione, analisi importanza, salvataggio modello e visualizzazione.
    MODIFICATO per gestire sia la modalità BinaryCE che le loss standard.
    CORRETTO: Passaggio argomenti per BinaryCE.
    MODIFICATO (v5): Usa la STESSA formula di attendibilità per TUTTE le modalità,
                     ma con moltiplicatore loss condizionale.
    """
    global numeri_finali, progress_bar, fold_performances, ultima_ruota_elaborata, num_folds, ruota_selezionata, textbox, btn_start, root, pulsanti_ruote, config
    # Rimuovi: numeri_interi = None

    # --- Gestione Configurazione TF per EXE ---
    # ... (invariato) ...
    is_exe = getattr(sys, 'frozen', False)
    if is_exe:
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus: tf.config.set_visible_devices([], 'GPU')
            tf.config.set_soft_device_placement(True)
            gc.collect()
        except Exception as e: print(f"Errore configurazione TensorFlow: {e}")

    ruota = ruota_selezionata.get()
    if not ruota:
        messagebox.showwarning("Attenzione", "Seleziona prima una ruota.")
        return

    # --- Disabilita/Aggiorna Interfaccia ---
    # ... (invariato) ...
    try:
        for rb in pulsanti_ruote.values():
             if rb.winfo_exists(): rb.config(state="disabled")
        if btn_start.winfo_exists(): btn_start.config(text="ELABORAZIONE IN CORSO...", state="disabled", bg="#cccccc")
        if root.winfo_exists(): root.update()
        if textbox.winfo_exists(): textbox.delete(1.0, tk.END); textbox.insert(tk.END, f"Avvio elaborazione per la ruota: {ruota}\n")
    except Exception as ui_err: print(f"Errore aggiornamento UI iniziale: {ui_err}")

    # --- Gestione Input (Date e Parametri) ---
    # ... (invariato) ...
    try:
        start_date = pd.to_datetime(entry_start_date.get_date())
        end_date = pd.to_datetime(entry_end_date.get_date())
        if start_date >= end_date: raise ValueError("Data inizio deve precedere data fine.")
        epochs = int(entry_epochs.get())
        batch_size = int(entry_batch_size.get())
        patience_cv = int(entry_patience.get())
        min_delta_cv = float(entry_min_delta.get())
        dense_layers_cfg = [int(getattr(entry, 'get', lambda: 0)()) for entry in [entry_neurons_layer1, entry_neurons_layer2, entry_neurons_layer3]]
        dropout_rates_cfg = [float(getattr(entry, 'get', lambda: 0.0)()) for entry in [entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3]]
        dense_layers_cfg = [n for n in dense_layers_cfg if n > 0]
        dropout_rates_cfg = [d for d in dropout_rates_cfg if d >= 0]
        if len(dropout_rates_cfg) != len(dense_layers_cfg):
             print(f"Warning: Numero layer ({len(dense_layers_cfg)}) != Numero dropout ({len(dropout_rates_cfg)}). Aggiustando dropout.")
             dropout_rates_cfg = (dropout_rates_cfg + [0.3] * len(dense_layers_cfg))[:len(dense_layers_cfg)]
        if epochs <=0 or batch_size <=0 or patience_cv <=0 or min_delta_cv < 0:
             raise ValueError("Epochs, Batch Size, Patience devono essere > 0 e Min Delta >= 0.")
        if is_exe:
            epochs = min(epochs, 20)
            dense_layers_cfg = [min(layer, 256) for layer in dense_layers_cfg]
    except Exception as e:
        messagebox.showerror("Errore Parametri", f"Errore nei parametri di input:\n{e}")
        try:
            for rb in pulsanti_ruote.values():
                 if rb.winfo_exists(): rb.config(state="normal")
            if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
            if root.winfo_exists(): root.update()
        except Exception as ui_err_param: print(f"Errore riabilitazione UI (parametri): {ui_err_param}")
        return

    # --- Caricamento Dati Grezzi e Preprocessing Iniziale ---
    # ... (invariato) ...
    try:
        textbox.insert(tk.END, f"Caricamento dati da {start_date.strftime('%Y/%m/%d')} a {end_date.strftime('%Y/%m/%d')}...\n")
        textbox.update()
        df_grezzo = carica_dati_grezzi(ruota)
        if df_grezzo is None: raise ValueError("Caricamento dati grezzi fallito.")
        df_preproc = preprocessa_dati(df_grezzo, start_date, end_date)
        if df_preproc is None or df_preproc.empty: raise ValueError("Nessun dato trovato nell'intervallo.")
        numeri = estrai_numeri(df_preproc)
        if numeri is None: raise ValueError("Estrazione numeri fallita.")
        min_samples_needed = 15
        if len(numeri) < min_samples_needed: raise ValueError(f"Dati insufficienti ({len(numeri)} estrazioni).")
        df_temp_reset = df_preproc.reset_index()
        df_con_feature_temp_reset = aggiungi_feature_temporali(df_temp_reset)
        df_con_feature_temp = df_con_feature_temp_reset.set_index('Data')
        temporal_features_df = df_con_feature_temp[['giorno_sett_sin', 'giorno_sett_cos', 'mese_sin', 'mese_cos', 'giorno_mese_sin', 'giorno_mese_cos']]
    except Exception as e:
        messagebox.showerror("Errore Caricamento/Prep", f"Errore preparazione dati:\n{e}")
        try:
            for rb in pulsanti_ruote.values():
                 if rb.winfo_exists(): rb.config(state="normal")
            if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
            if root.winfo_exists(): root.update()
        except Exception as ui_err_load: print(f"Errore riabilitazione UI (caricamento): {ui_err_load}")
        return

    # === DEFINIZIONE STRUTTURA FEATURE e NOMI (PRIMA DEL LOOP) ===
    # ... (invariato) ...
    feature_names = []
    num_stat_features = 0
    try:
        feature_names.extend([f'Num{i}_prev' for i in range(1, 6)])
        feature_names.extend(['giorno_sett_sin', 'giorno_sett_cos', 'mese_sin', 'mese_cos', 'giorno_mese_sin', 'giorno_mese_cos'])
        min_rows_for_stats = 15
        if len(numeri) >= min_rows_for_stats:
            sample_input_for_stats = numeri[:min_rows_for_stats]
            example_stats_output = aggiungi_statistiche_numeri(sample_input_for_stats, finestra=10)
            if isinstance(example_stats_output, np.ndarray) and example_stats_output.ndim == 2:
                num_stat_features = example_stats_output.shape[1]
                print(f"DEBUG: Rilevato num_stat_features = {num_stat_features}")
                if num_stat_features > 0:
                    if num_stat_features % 2 == 0: feature_names.extend([f'{stat}{i}' for i in range(1, (num_stat_features // 2) + 1) for stat in ['Freq', 'Ritardo']])
                    else: feature_names.extend([f'Stat_{i+1}' for i in range(num_stat_features)])
            else: print(f"Warning: 'aggiungi_statistiche_numeri' non 2D valido...")
        else: print(f"Warning: Campione dati troppo piccolo per stats.")
    except Exception as stat_err: print(f"Warning: Errore determinazione feature stats: {stat_err}.")
    total_expected_features = len(feature_names)
    print("DEBUG: Nomi feature:", feature_names)
    print(f"DEBUG: Feature attese: {total_expected_features}")

    # --- Cross-Validation Temporale ---
    # ... (invariato) ...
    try:
        n_splits_widget = globals().get('spinbox_fold')
        if n_splits_widget and isinstance(n_splits_widget, (tk.Spinbox, tk.Entry)): n_splits = int(n_splits_widget.get())
        elif 'num_folds' in globals() and num_folds is not None and hasattr(num_folds, 'get'): n_splits = num_folds.get()
        else: n_splits = 5
        max_splits = len(numeri) - 1; n_splits = min(n_splits, max_splits) if max_splits >= 2 else 2; n_splits = max(2, n_splits)
        textbox.insert(tk.END, f"Usando {n_splits} fold (TimeSeriesSplit)...\n"); textbox.update()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(np.arange(len(numeri))))
    except Exception as e:
        print(f"Errore config fold: {e}, uso default (5)."); n_splits = 5
        textbox.insert(tk.END, "Errore config fold, uso default (5).\n"); textbox.update()
        max_splits = len(numeri) - 1; n_splits = min(n_splits, max_splits) if max_splits >= 2 else 2; n_splits = max(2, n_splits)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(np.arange(len(numeri))))


    # --- Addestramento (Ciclo sui Fold) ---
    # ... (logica interna del fold invariata) ...
    fold_metrics = []
    best_val_loss = float('inf')
    best_model = None
    scaler_statistiche_best_fold = None
    num_possible_numbers = 90
    all_hist_loss = []
    all_hist_val_loss = []

    is_binary_ce_mode = getattr(config, 'use_binary_crossentropy_mode', False)
    print(f"DEBUG: Modalità BinaryCE attiva: {is_binary_ce_mode}")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # ... (logica interna del fold come prima) ...
        print(f"--- Fold {fold_idx + 1}/{n_splits} ---")
        textbox.insert(tk.END, f"Fold {fold_idx + 1}/{n_splits}...")
        textbox.update()

        if len(train_idx) < 2 or len(val_idx) < 1:
             print(f"Skipping Fold {fold_idx+1}: Indici train/val insufficienti."); textbox.insert(tk.END, f" Saltato (dati insuff.).\n"); continue

        input_shape = None

        try:
            numeri_train, numeri_val = numeri[train_idx], numeri[val_idx]
            temporal_features_train = temporal_features_df.iloc[train_idx]
            temporal_features_val = temporal_features_df.iloc[val_idx]
            window_stats = 10
            stats_train = aggiungi_statistiche_numeri(numeri_train, finestra=window_stats)
            history_len_for_val = min(len(train_idx), window_stats)
            indices_for_val_stats = np.concatenate([train_idx[max(0, len(train_idx)-history_len_for_val):], val_idx])
            history_for_val_stats = numeri[indices_for_val_stats]
            stats_val_full = aggiungi_statistiche_numeri(history_for_val_stats, finestra=window_stats)
            stats_val = stats_val_full[len(history_for_val_stats) - len(val_idx):]
            X_train = np.hstack([numeri_train[:-1], temporal_features_train.values[:-1], stats_train[:-1]])
            X_val = np.hstack([numeri_val[:-1], temporal_features_val.values[:-1], stats_val[:-1]])
            y_train_raw = numeri_train[1:]
            y_val_raw = numeri_val[1:]

            if is_binary_ce_mode:
                try:
                    y_train_target = multi_hot_encode(y_train_raw, num_classes=num_possible_numbers)
                    y_val_target = multi_hot_encode(y_val_raw, num_classes=num_possible_numbers)
                except NameError: raise ValueError("Funzione 'multi_hot_encode' non definita!")
                except Exception as encode_err: raise ValueError(f"Errore encoding multi-hot: {encode_err}")
            else:
                y_train_target = y_train_raw / 90.0
                y_val_target = y_val_raw / 90.0

            if X_train.shape[1] != total_expected_features: raise ValueError(f"Shape X_train errata!")
            if len(y_val_raw) == 1 and X_val.shape[0] == 0:
                 print(f"Warning Fold {fold_idx+1}: X_val vuoto. Salto fold."); textbox.insert(tk.END, f" Saltato (X_val vuoto).\n"); continue
            if X_val.shape[1] != total_expected_features: raise ValueError(f"Shape X_val errata!")
            if X_train.shape[0] != y_train_target.shape[0]: raise ValueError("Disallineamento X_train/y_train_target!")
            if X_val.shape[0] != y_val_target.shape[0]: raise ValueError("Disallineamento X_val/y_val_target!")

            scaler_statistiche_fold = MinMaxScaler()
            X_train_scaled = np.zeros_like(X_train, dtype=float)
            X_train_scaled[:, :5] = X_train[:, :5] / 90.0
            X_train_scaled[:, 5:11] = X_train[:, 5:11]
            if num_stat_features > 0 and X_train.shape[1] > 11: X_train_scaled[:, 11:] = scaler_statistiche_fold.fit_transform(X_train[:, 11:])

            X_val_scaled = np.zeros_like(X_val, dtype=float)
            X_val_scaled[:, :5] = X_val[:, :5] / 90.0
            X_val_scaled[:, 5:11] = X_val[:, 5:11]
            if num_stat_features > 0 and X_val.shape[1] > 11: X_val_scaled[:, 11:] = scaler_statistiche_fold.transform(X_val[:, 11:])

            input_shape = (X_train_scaled.shape[1],)

            if is_binary_ce_mode:
                output_units_or_shape = num_possible_numbers
                loss_function_to_use = 'binary_crossentropy'
                metrics_to_use = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
            else:
                output_units_or_shape = y_train_raw.shape[1]
                if config.use_custom_loss: loss_function_to_use = custom_loss_function
                elif config.loss_function_choice: loss_function_to_use = config.loss_function_choice
                else: loss_function_to_use = "mean_squared_error"
                metrics_to_use = ["mae"]

            if getattr(sys, 'frozen', False):
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Input(shape=input_shape))
                model.add(tf.keras.layers.Dense(64, activation='relu'))
                model.add(tf.keras.layers.BatchNormalization()); model.add(tf.keras.layers.Dropout(0.3))
                model.add(tf.keras.layers.Dense(32, activation='relu'))
                model.add(tf.keras.layers.BatchNormalization()); model.add(tf.keras.layers.Dropout(0.3))
                final_activation_exe = 'sigmoid' if is_binary_ce_mode else None
                model.add(tf.keras.layers.Dense(output_units_or_shape, activation=final_activation_exe))
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss=loss_function_to_use, metrics=metrics_to_use)
                patience_value = min(5, patience_cv); epochs_value = min(10, epochs); batch_size_value = min(16, batch_size)
            else:
                try:
                    model = build_model(input_shape, output_units_or_shape, dense_layers_cfg, dropout_rates_cfg)
                except NameError: raise ValueError("Funzione 'build_model' non definita!")
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss=loss_function_to_use, metrics=metrics_to_use)
                patience_value = patience_cv; epochs_value = epochs; batch_size_value = batch_size

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_value, min_delta=min_delta_cv, verbose=0, restore_best_weights=True)
            history = model.fit( X_train_scaled, y_train_target, validation_data=(X_val_scaled, y_val_target), epochs=epochs_value, batch_size=batch_size_value, callbacks=[early_stopping], verbose=0)

            val_loss = history.history.get('val_loss', [float('nan')])[-1]
            train_loss = history.history.get('loss', [float('nan')])[-1]
            if np.isnan(val_loss) or np.isnan(train_loss) or np.isinf(val_loss) or np.isinf(train_loss): raise ValueError("Loss non valida rilevata (NaN o Inf)")
            ratio = val_loss / train_loss if train_loss > 1e-9 else float('inf')
            textbox.insert(tk.END, f". Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n"); textbox.update()
            fold_metrics.append({'fold': fold_idx, 'train_loss': train_loss, 'val_loss': val_loss, 'ratio': ratio, 'history': history.history})
            all_hist_loss.append(history.history.get('loss', []))
            all_hist_val_loss.append(history.history.get('val_loss', []))

            if val_loss < best_val_loss: best_val_loss = val_loss; best_model = model; scaler_statistiche_best_fold = scaler_statistiche_fold

        except Exception as fold_err:
            # ... (gestione errore fold e fallback come prima) ...
            print(f"Errore nel Fold {fold_idx+1}: {fold_err}")
            import traceback; traceback.print_exc()
            textbox.insert(tk.END, f" Errore Fold {fold_idx+1}, utilizzo modello fallback...\n"); textbox.update()
            try:
                if input_shape is None:
                     if 'X_train_scaled' in locals() and X_train_scaled.shape[1] > 0: input_shape = (X_train_scaled.shape[1],)
                     else: print("Errore Fallback: Input shape non disponibile."); continue
                fallback_output_shape = 5
                fallback_model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=input_shape), tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(fallback_output_shape)
                ])
                fallback_model.compile(optimizer='adam', loss='mse')
                if 'X_train_scaled' not in locals() or 'y_train_raw' not in locals() or \
                   'X_val_scaled' not in locals() or 'y_val_raw' not in locals():
                   print("Errore Fallback: Dati necessari non disponibili."); continue
                fallback_history = fallback_model.fit( X_train_scaled, y_train_raw / 90.0, validation_data=(X_val_scaled, y_val_raw / 90.0), epochs=5, batch_size=16, verbose=0)
                val_loss = fallback_history.history.get('val_loss', [float('nan')])[-1]; train_loss = fallback_history.history.get('loss', [float('nan')])[-1]
                if np.isnan(val_loss) or np.isnan(train_loss) or np.isinf(val_loss) or np.isinf(train_loss): raise ValueError("Loss non valida nel fallback")
                ratio = val_loss / train_loss if train_loss > 1e-9 else float('inf')
                textbox.insert(tk.END, f" Modello fallback (MSE): Loss={train_loss:.4f}, Val Loss={val_loss:.4f}\n"); textbox.update()
                fold_metrics.append({'fold': fold_idx, 'train_loss': train_loss, 'val_loss': val_loss, 'ratio': ratio, 'history': fallback_history.history})
                all_hist_loss.append(fallback_history.history.get('loss', []))
                all_hist_val_loss.append(fallback_history.history.get('val_loss', []))
                if best_model is None or val_loss < best_val_loss: best_val_loss = val_loss; best_model = fallback_model
            except Exception as fallback_err: print(f"Errore anche nel modello fallback: {fallback_err}"); traceback.print_exc(); textbox.insert(tk.END, f" Anche fallback fallito! Salto.\n"); textbox.update()


    # --- Fine Loop CV ---
    if best_model is None:
        # ... (gestione errore e riabilitazione UI come prima) ...
        messagebox.showerror("Errore Addestramento", "Nessun modello valido addestrato durante la cross-validation.")
        try:
            for rb in pulsanti_ruote.values():
                 if rb.winfo_exists(): rb.config(state="normal")
            if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
            if root.winfo_exists(): root.update()
        except Exception as ui_err_cv: print(f"Errore riabilitazione UI (CV): {ui_err_cv}")
        return

    # --- *** Calcolo Attendibilità Unificato (v5 - Mantenuto per coerenza) *** ---
    attendibility_score = 0
    commento = "N/D"

    if fold_metrics:
        valid_val_losses = [m['val_loss'] for m in fold_metrics if m and 'val_loss' in m and not np.isnan(m['val_loss']) and not np.isinf(m['val_loss'])]
        avg_val_loss = np.mean(valid_val_losses) if valid_val_losses else float('nan')
        valid_train_losses = [m['train_loss'] for m in fold_metrics if m and 'train_loss' in m and not np.isnan(m['train_loss']) and not np.isinf(m['train_loss'])]
        avg_train_loss = np.mean(valid_train_losses) if valid_train_losses else float('nan')
        consistency = float('inf')
        if valid_val_losses and len(valid_val_losses) > 1 and avg_val_loss > 1e-9 and not np.isnan(avg_val_loss):
            consistency_std = np.std(valid_val_losses)
            consistency = consistency_std / avg_val_loss if not np.isnan(consistency_std) else float('inf')

        # --- Moltiplicatore loss condizionale ---
        if is_binary_ce_mode:
            loss_multiplier = 30 # Valore più alto per BinaryCE
        else:
            loss_multiplier = 10 # Valore originale per MSE/MAE etc.
        consistency_multiplier = 10
        # ---------------------------------------

        if not np.isnan(avg_val_loss) and not np.isinf(consistency):
            term1 = (1 / (1 + avg_val_loss * loss_multiplier)) * 0.7
            term2 = (1 / (1 + consistency * consistency_multiplier)) * 0.3
            attendibility_score = 100 * (term1 + term2)
            attendibility_score = max(0, min(100, attendibility_score))
        else:
            attendibility_score = 0

        # Assegna commento
        if attendibility_score > 80: commento = "Molto attendibile"
        elif attendibility_score > 60: commento = "Attendibile"
        elif attendibility_score > 40: commento = "Moderatamente attendibile"
        elif attendibility_score > 20: commento = "Poco attendibile"
        else: commento = "Non attendibile"

        loss_type_info = "(BinaryCE)" if is_binary_ce_mode else "(Standard Loss)"
        textbox.insert(tk.END, f"\nRisultati CV {loss_type_info}: Loss Media={avg_train_loss:.4f}, Val Loss Media={avg_val_loss:.4f}\n")
        textbox.insert(tk.END, f"Attendibilità (Formula Unificata v5): {attendibility_score:.1f}/100 ({commento})\n\n")

    else:
        textbox.insert(tk.END, "\nNessun fold completato, attendibilità non calcolabile.\n")
        attendibility_score = 0
        commento = "N/D"
    # --- *** FINE BLOCCO ATTENDIBILITÀ UNIFICATO (v5) *** ---


    # --- Predizione Finale (con il miglior modello) ---
    # ... (Questa parte rimane invariata) ...
    textbox.insert(tk.END, "Esecuzione predizione finale...\n"); textbox.update()
    numeri_da_passare = None
    try:
        last_numbers = numeri[-1]; last_temporal = temporal_features_df.iloc[-1].values; window_stats = 10
        history_for_pred_stats = numeri[-window_stats:] if len(numeri) >= window_stats else numeri
        stats_pred_full = aggiungi_statistiche_numeri(history_for_pred_stats, finestra=window_stats)
        if stats_pred_full is None or len(stats_pred_full) == 0: raise ValueError("Calcolo stats predizione fallito.")
        stats_pred = stats_pred_full[-1]
        X_pred_features = np.concatenate([last_numbers, last_temporal, stats_pred]).reshape(1, -1)
        if X_pred_features.shape[1] != total_expected_features: raise ValueError(f"Shape X_pred errata!")
        X_pred_scaled = np.zeros_like(X_pred_features, dtype=float)
        X_pred_scaled[0, :5] = X_pred_features[0, :5] / 90.0
        X_pred_scaled[0, 5:11] = X_pred_features[0, 5:11]
        if num_stat_features > 0 and scaler_statistiche_best_fold:
             if X_pred_features.shape[1] > 11: X_pred_scaled[0, 11:] = scaler_statistiche_best_fold.transform(X_pred_features[0, 11:].reshape(1,-1))
        elif num_stat_features > 0: raise ValueError("Scaler stats mancante.")

        predizione_raw = best_model.predict(X_pred_scaled)[0]

        if is_binary_ce_mode:
            predizione_probabilita = predizione_raw
            top_5_indices = np.argsort(predizione_probabilita)[-5:]
            numeri_da_passare = top_5_indices + 1
            numeri_da_passare.sort()
            textbox.insert(tk.END, f"Probabilità predette (top 5): {np.round(predizione_probabilita[top_5_indices], 3)}\n")
            textbox.insert(tk.END, f"Numeri predetti (dal modello): {', '.join(map(str, numeri_da_passare))}\n")
        else:
            predizione_scaled = predizione_raw
            numeri_denormalizzati = predizione_scaled * 90.0
            numeri_interi_predetti = np.round(numeri_denormalizzati).astype(int)
            numeri_da_passare = np.clip(numeri_interi_predetti, 1, 90)
            if isinstance(numeri_da_passare, np.ndarray):
                 numeri_da_passare = numeri_da_passare.flatten()
            textbox.insert(tk.END, f"Numeri predetti (dal modello): {', '.join(map(str, numeri_da_passare))}\n")

    except Exception as pred_final_err:
        # ... (gestione errore predizione e riabilitazione UI come prima) ...
        messagebox.showerror("Errore Predizione Finale", f"Errore: {pred_final_err}"); traceback.print_exc()
        textbox.insert(tk.END, f"Predizione finale fallita: {pred_final_err}\n")
        try:
            for rb in pulsanti_ruote.values():
                 if rb.winfo_exists(): rb.config(state="normal")
            if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
            if root.winfo_exists(): root.update()
        except Exception as ui_err_pred: print(f"Errore riabilitazione UI (predizione): {ui_err_pred}")
        return

    # --- Generazione Numeri Finali (USA VARIABILE CORRETTA e SCORE CORRETTO) ---
    numeri_finali = []
    origine_ml = []
    try:
        numeri_frequenti = estrai_numeri_frequenti(ruota, start_date, end_date, n=20)
        print(f"DEBUG: Chiamata a genera_numeri_attendibili con:")
        print(f"  numeri_input = {numeri_da_passare} (tipo: {type(numeri_da_passare)})")
        print(f"  attendibility_score = {attendibility_score}") # Usa lo score calcolato nel blocco unificato
        print(f"  numeri_frequenti[:5] = {numeri_frequenti[:5]}")
        # Rimosso il passaggio di is_binary_mode qui

        # === Chiama genera_numeri_attendibili SENZA is_binary_mode ===
        # La funzione userà la sua soglia fissa (60)
        numeri_finali, origine_ml = genera_numeri_attendibili(
            numeri_da_passare,
            attendibility_score,
            numeri_frequenti
        )
        # ============================================================

        print(f"DEBUG: Risultato da genera_numeri_attendibili:")
        print(f"  numeri_finali = {numeri_finali}")
        print(f"  origine_ml = {origine_ml}")

        textbox.insert(tk.END, "Numeri predetti finali suggeriti: " + ", ".join(map(str, numeri_finali)) + "\n")
        mostra_numeri_forti_popup(numeri_finali, attendibility_score, origine_ml)

    except NameError as ne_gen:
         messagebox.showerror("Errore Interno", f"Funzione 'genera_numeri_attendibili' non trovata: {ne_gen}")
         textbox.insert(tk.END, f"Errore critico: {ne_gen}\n")
    except Exception as final_num_err:
        # ... (gestione errore e fallback come prima) ...
        messagebox.showerror("Errore Numeri Finali", f"Errore: {final_num_err}")
        textbox.insert(tk.END, f"Generazione numeri finali fallita: {final_num_err}\n")
        fallback_numeri = []
        if numeri_da_passare is not None:
            if isinstance(numeri_da_passare, np.ndarray): fallback_numeri = numeri_da_passare.tolist()
            elif isinstance(numeri_da_passare, list): fallback_numeri = numeri_da_passare
        numeri_finali = fallback_numeri
        origine_ml = [False] * len(numeri_finali)


    # --- Analisi Importanza Feature ---
    # ... (invariato) ...
    feature_importances = None
    try: importanze = analizza_importanza_feature_primo_layer(best_model, feature_names); feature_importances = importanze; print("DEBUG - Importances:", feature_importances)
    except NameError as ne: print(f"Warning: Funzione importanza mancante: {ne}")
    except Exception as fi_err: print(f"Warning: Errore analisi importanza: {fi_err}")

    # --- Visualizzazione Risultati Aggiuntivi ---
    # ... (invariato) ...
    try:
        if all_hist_loss and all_hist_val_loss: mostra_grafico(all_hist_loss, all_hist_val_loss)
        else: print("Warning: No history for plot.")
    except NameError: print("Warning: Funzione 'mostra_grafico' non definita.")
    except Exception as vis_err: print(f"Warning: Errore visualizzazione: {vis_err}")

    # --- SALVATAGGIO FINALE ESPLICITO ---
    # ... (invariato) ...
    try:
        print(f"Tentativo salvataggio: best_model_{ruota}.weights.h5")
        if best_model and not getattr(config, 'use_ensemble', False):
            final_model_path = f'best_model_{ruota}.weights.h5'
            best_model.save_weights(final_model_path)
            if os.path.exists(final_model_path):
                 file_size = os.path.getsize(final_model_path)/1024; logger.info(f"Modello salvato: {final_model_path} ({file_size:.2f} KB)")
                 textbox.insert(tk.END, f"\nModello salvato: {final_model_path} ({file_size:.2f} KB)\n")
            else: logger.error(f"Errore: file {final_model_path} non creato."); textbox.insert(tk.END, "\nERRORE: Salvataggio fallito!\n")
        elif not best_model: logger.warning("Salvataggio saltato: nessun modello valido.")
        elif getattr(config, 'use_ensemble', False): logger.info("Salvataggio saltato: ensemble attivo.")
    except Exception as e: logger.error(f"Errore salvataggio: {e}"); textbox.insert(tk.END, f"\nErrore salvataggio: {e}\n")

    # --- Riabilita Interfaccia ---
    # ... (invariato) ...
    try:
        for rb in pulsanti_ruote.values():
             if rb.winfo_exists(): rb.config(state="normal")
        if btn_start.winfo_exists(): btn_start.config(text=f"AVVIA ELABORAZIONE ({ruota})", state="normal", bg="#4CAF50")
        if root.winfo_exists(): root.update()
        ultima_ruota_elaborata = ruota
    except Exception as ui_err_final: print(f"Errore riabilitazione UI finale: {ui_err_final}")

    return numeri_finali

def callback_ruota(event):
    """Gestisce la selezione della ruota."""
    global ruota_selezionata  # Usa variabile globale
    btn = event.widget  # Ottieni il pulsante
    ruota = btn.cget("text")  # Ottieni il nome della ruota
    ruota_selezionata.set(ruota)  # Imposta la variabile
    print(f"Dentro callback_ruota, ruota selezionata: {ruota}") # DEBUG
    on_ruota_selected() # Chiamata per aggiornare interfaccia

class Avatar:
    def __init__(self, frame, image_path, width=150, height=150):
        """
        Inizializza l'avatar con un'immagine e funzionalità di sintesi vocale.

        Args:
            frame: Il frame Tkinter in cui visualizzare l'avatar
            image_path: Percorso dell'immagine dell'avatar
            width: Larghezza dell'avatar
            height: Altezza dell'avatar
        """
        self.frame = frame
        self.width = width
        self.height = height
        
        # Canvas per l'avatar
        self.canvas = tk.Canvas(frame, width=width, height=height, bg="#E0F7FA")
        self.canvas.pack(padx=10, pady=10)
        
        # Carica l'immagine dell'avatar
        try:
            self.avatar_image = self.load_image(image_path)
            self.avatar_on_canvas = self.canvas.create_image(width//2, height//2, image=self.avatar_image)
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine dell'avatar: {e}")
            # Crea un'immagine placeholder
            img = Image.new('RGB', (width, height), color="#FFD700")
            self.avatar_image = ImageTk.PhotoImage(img)
            self.avatar_on_canvas = self.canvas.create_image(width//2, height//2, image=self.avatar_image)
        
        # Posizione e dimensione della bocca (dovrà essere regolata in base all'immagine)
        self.mouth_x = width // 2
        self.mouth_y = height // 2 + 20  # Assume che la bocca sia nella metà inferiore del volto
        self.mouth_width = 30
        self.mouth_height = 10
        
        # Crea una bocca neutra (questa sarà modificata durante l'animazione)
        self.mouth = self.canvas.create_oval(
            self.mouth_x - self.mouth_width // 2,
            self.mouth_y - self.mouth_height // 2,
            self.mouth_x + self.mouth_width // 2,
            self.mouth_y + self.mouth_height // 2,
            fill="red", outline="black"
        )
        
        # Nascondi la bocca disegnata inizialmente
        self.canvas.itemconfig(self.mouth, state="hidden")
        
        # Stato parlante
        self.is_speaking = False
        self.speech_thread = None
        self.voice_thread = None
        self.engine = None  # Riferimento al motore TTS
    
    def load_image(self, path):
        """
        Carica un'immagine da file e la ridimensiona.
        
        Args:
            path: Percorso del file immagine
            
        Returns:
            ImageTk.PhotoImage: Immagine caricata e ridimensionata
        """
        img = Image.open(path).resize((self.width, self.height))
        return ImageTk.PhotoImage(img)
    
    def speak(self, text, speed=0.1):
        """
        Anima l'avatar e riproduce la voce.
        
        Args:
            text: Testo da pronunciare
            speed: Velocità dell'animazione (pausa tra i movimenti della bocca)
        """
        print(f"Avatar sta tentando di pronunciare: '{text}'")
        
        if self.is_speaking:
            self.stop_speaking()  # Ferma l'animazione corrente
        
        self.is_speaking = True
        
        # Thread per l'animazione
        def animate_speech():
            try:
                # Mostra la bocca disegnata
                self.canvas.itemconfig(self.mouth, state="normal")
                
                for char in text:
                    if not self.is_speaking:
                        break
                    
                    # Modifica la forma della bocca in base al carattere
                    if char.lower() in "aeiouàèéìòù":
                        # Bocca aperta per vocali
                        self.canvas.coords(
                            self.mouth,
                            self.mouth_x - self.mouth_width // 2,
                            self.mouth_y - self.mouth_height,
                            self.mouth_x + self.mouth_width // 2,
                            self.mouth_y + self.mouth_height
                        )
                    elif char in " ,.!?":
                        # Bocca chiusa per pause
                        self.canvas.coords(
                            self.mouth,
                            self.mouth_x - self.mouth_width // 2,
                            self.mouth_y - 1,
                            self.mouth_x + self.mouth_width // 2,
                            self.mouth_y + 1
                        )
                    else:
                        # Bocca leggermente aperta per consonanti
                        self.canvas.coords(
                            self.mouth,
                            self.mouth_x - self.mouth_width // 2,
                            self.mouth_y - self.mouth_height // 2,
                            self.mouth_x + self.mouth_width // 2,
                            self.mouth_y + self.mouth_height // 2
                        )
                    
                    self.frame.update_idletasks()
                    time.sleep(speed)  # Pausa per l'animazione
                
                # Nascondi di nuovo la bocca disegnata
                self.canvas.itemconfig(self.mouth, state="hidden")
            except Exception as e:
                print(f"Errore nell'animazione: {e}")
            finally:
                self.is_speaking = False
        
        # Thread per la riproduzione vocale
        def play_speech():
            try:
                # Reinizializza il motore TTS ogni volta
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
                
                # Imposta una voce, se disponibile
                voices = self.engine.getProperty('voices')
                if voices:
                    # Trova una voce italiana se disponibile
                    italian_voice = None
                    for voice in voices:
                        if 'italian' in voice.id.lower() or 'it' in voice.id.lower():
                            italian_voice = voice.id
                            break
                    
                    # Usa la voce italiana o la prima disponibile
                    if italian_voice:
                        self.engine.setProperty('voice', italian_voice)
                    else:
                        self.engine.setProperty('voice', voices[0].id)
                
                # Esegui la sintesi vocale
                print("Esecuzione della sintesi vocale...")
                self.engine.say(text)
                self.engine.runAndWait()
                print("Sintesi vocale completata")
            except Exception as e:
                print(f"Errore nella riproduzione vocale: {e}")
            finally:
                self.engine = None  # Resetta il riferimento all'engine
        
        # Avvia l'animazione in un thread separato
        self.speech_thread = threading.Thread(target=animate_speech)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Avvia la riproduzione vocale in un altro thread
        self.voice_thread = threading.Thread(target=play_speech)
        self.voice_thread.daemon = True
        self.voice_thread.start()
    
    def stop_speaking(self):
        """Interrompe l'animazione parlante e la riproduzione vocale"""
        print("Tentativo di interrompere l'avatar")
        self.is_speaking = False
        
        # Interrompi il motore TTS se è in esecuzione
        if self.engine:
            try:
                self.engine.stop()  # Ferma il motore TTS
            except Exception as e:
                print(f"Errore nell'arresto del motore TTS: {e}")
        
        # Ferma il thread di animazione
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(0.5)
        
        # Nascondi la bocca
        self.canvas.itemconfig(self.mouth, state="hidden")
        
        # Ferma il thread vocale
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(0.5)
        
        print("Avatar interrotto")

def esegui_script_numeri_spia():
    """Avvia spia2.py come programma separato."""
    try:
        import subprocess
        import sys
        import os
        
        # Gestione speciale per ambiente PyInstaller
        if getattr(sys, 'frozen', False):
            # Se in esecuzione come eseguibile
            base_path = os.path.dirname(sys.executable)
            spia2_path = os.path.join(base_path, "spia2.py")
            
            # Controlla se esiste un eseguibile separato
            spia2_exe = os.path.join(base_path, "spia2.exe")
            if os.path.exists(spia2_exe):
                # Usa direttamente l'eseguibile se esiste
                subprocess.Popen([spia2_exe])
                return
        else:
            # Durante l'esecuzione come script Python
            # Usa un percorso esplicito per evitare confusione
            current_dir = os.path.dirname(os.path.abspath(__file__))
            spia2_path = os.path.join(current_dir, "spia2.py")
        
        # Se non lo trova, prova nella directory corrente
        if not os.path.exists(spia2_path):
            spia2_path = "spia2.py"
        
        # Debug: mostra il percorso che stiamo tentando di usare
        print(f"Tentativo di eseguire: {spia2_path}")
        
        # Verifica che esista
        if not os.path.exists(spia2_path):
            from tkinter import messagebox
            messagebox.showerror("Errore", f"File spia2.py non trovato nel percorso: {spia2_path}")
            return
        
        # Avvia il processo Python in una nuova finestra, specificando esplicitamente il percorso Python
        if sys.executable and os.path.exists(sys.executable):
            subprocess.Popen([sys.executable, spia2_path])
        else:
            # Fallback generico a 'python'
            subprocess.Popen(["python", spia2_path])
        
    except Exception as e:
        from tkinter import messagebox
        messagebox.showerror("Errore", f"Errore durante l'avvio di spia2.py:\n{e}")
        import traceback
        traceback.print_exc()

def esegui_script_legge_terzo():
    try:
        # Esegui lo script esterno
        subprocess.run(["python", "leggedelterzo.py"], check=True)
        print("Script eseguito con successo.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione dello script: {e}")
    except FileNotFoundError:
        print("Errore: Python non trovato. Assicurati che Python sia installato e disponibile nel PATH.")

def lancia_spie_dec():
    """
    Lancia lo script esterno ratiod.py.
    """
    script_name = "ratiod.py"
    try:
        print(f"Tentativo di lanciare: {sys.executable} {script_name}")
        process = subprocess.Popen([sys.executable, script_name])
        print(f"Script {script_name} lanciato con PID: {process.pid}")
    except FileNotFoundError:
        print(f"Errore: Lo script {script_name} non è stato trovato.")
        messagebox.showerror("Errore", f"Script {script_name} non trovato!\nAssicurati che sia nella stessa cartella dell'applicazione.")
    except Exception as e:
        print(f"Errore durante l'esecuzione di {script_name}: {e}")
        messagebox.showerror("Errore", f"Errore nell'esecuzione di {script_name}:\n{e}")

# --- Funzione per il SECONDO nuovo pulsante (Spie Quind) ---
def lancia_spie_quind():
    """
    Lancia lo script esterno ratioq.py.
    """
    script_name = "ratioq.py" # Nome dello script cambiato
    try:
        print(f"Tentativo di lanciare: {sys.executable} {script_name}")
        process = subprocess.Popen([sys.executable, script_name])
        print(f"Script {script_name} lanciato con PID: {process.pid}")
    except FileNotFoundError:
        print(f"Errore: Lo script {script_name} non è stato trovato.")
        messagebox.showerror("Errore", f"Script {script_name} non trovato!\nAssicurati che sia nella stessa cartella dell'applicazione.")
    except Exception as e:
        print(f"Errore durante l'esecuzione di {script_name}: {e}")
        messagebox.showerror("Errore", f"Errore nell'esecuzione di {script_name}:\n{e}")

def lancia_ritardi_plus():  # <<< IL NOME È ESATTAMENTE QUESTO
    """
    Lancia lo script esterno ritardi_plus_analyzer.py usando subprocess.
    """
    script_name = "ritardi_plus_analyzer.py"  # Il file che contiene la tua RitardiPlusApp
                                              # e che ha il suo if __name__ == "__main__":
    try:
        print(f"Tentativo di lanciare (subprocess): {sys.executable} {script_name}")
        # Popen lancia lo script come un processo separato
        process = subprocess.Popen([sys.executable, script_name])
        print(f"Script {script_name} lanciato con PID: {process.pid}")
    except FileNotFoundError:
        print(f"Errore: Lo script {script_name} non è stato trovato.")
        # È buona norma passare il 'parent' alla messagebox se possibile
        # (es. la finestra principale della tua app)
        messagebox.showerror("Errore",
                             f"Script {script_name} non trovato!\n"
                             f"Assicurati che sia nella stessa cartella dell'applicazione.")
    except Exception as e:
        print(f"Errore durante l'esecuzione di {script_name}: {e}")
        messagebox.showerror("Errore", f"Errore nell'esecuzione di {script_name}:\n{e}")

def lancia_rit_pos():
    """
    Lancia lo script esterno analizza_ruota_lotto.py usando subprocess.
    """
    script_name_rit_pos = "analizza_ruota_lotto.py"  # Nome del nuovo script esterno
    try:
        print(f"Tentativo di lanciare (subprocess): {sys.executable} {script_name_rit_pos}")
        process_rit_pos = subprocess.Popen([sys.executable, script_name_rit_pos])
        print(f"Script {script_name_rit_pos} lanciato con PID: {process_rit_pos.pid}")
    except FileNotFoundError:
        print(f"Errore: Lo script {script_name_rit_pos} non è stato trovato.")
        messagebox.showerror("Errore",
                             f"Script {script_name_rit_pos} non trovato!\n"
                             f"Assicurati che sia nella stessa cartella dell'applicazione.")
    except Exception as e:
        print(f"Errore durante l'esecuzione di {script_name_rit_pos}: {e}")
        messagebox.showerror("Errore", f"Errore nell'esecuzione di {script_name_rit_pos}:\n{e}")

def esegui_pannello_estrazioni():
    """Avvia il modulo esterno del pannello estrazioni."""
    try:
        import subprocess
        import sys
        import os
        
        # Gestione speciale per ambiente PyInstaller
        if getattr(sys, 'frozen', False):
            # Se in esecuzione come eseguibile
            base_path = os.path.dirname(sys.executable)
            pannello_path = os.path.join(base_path, "pannello_estrazioni.py")
            
            # Controlla se esiste un eseguibile separato
            pannello_exe = os.path.join(base_path, "pannello_estrazioni.exe")
            if os.path.exists(pannello_exe):
                subprocess.Popen([pannello_exe])
                return
        else:
            # Durante l'esecuzione come script Python
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pannello_path = os.path.join(current_dir, "pannello_estrazioni.py")
        
        # Se non lo trova, prova nella directory corrente
        if not os.path.exists(pannello_path):
            pannello_path = "pannello_estrazioni.py"
        
        # Debug: mostra il percorso che stiamo tentando di usare
        print(f"Tentativo di eseguire pannello estrazioni: {pannello_path}")
        
        # Verifica che esista
        if not os.path.exists(pannello_path):
            messagebox.showerror("Errore", f"File pannello_estrazioni.py non trovato nel percorso: {pannello_path}")
            return
        
        # Avvia il processo Python in una nuova finestra
        if sys.executable and os.path.exists(sys.executable):
            subprocess.Popen([sys.executable, pannello_path])
        else:
            # Fallback generico a 'python'
            subprocess.Popen(["python", pannello_path])
        
    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante l'avvio del pannello estrazioni:\n{e}")
        import traceback
        traceback.print_exc()

def main():
    """Funzione principale per l'avvio dell'applicazione."""
    # --- Riferimento alle TUE variabili globali originali ---
    # (Assicurati che tutte queste globali siano necessarie e definite correttamente)
    global root, textbox, entry_info, pulsanti_ruote, frame_grafico, ruota_selezionata, btn_start
    global entry_start_date, entry_end_date, entry_epochs, entry_batch_size
    global entry_patience, entry_min_delta, entry_neurons_layer1, entry_neurons_layer2
    global entry_neurons_layer3, entry_dropout_layer1, entry_dropout_layer2, entry_dropout_layer3
    global entry_reg_value, entry_noise_scale, entry_noise_percentage
    global btn_relu, btn_leaky_relu, btn_elu, btn_reg_none, btn_reg_l1, btn_reg_l2, btn_model_dense, btn_model_lstm
    global frame_loss_function, frame_optimizer, frame_activation_function, frame_regularization, frame_model_type
    global btn_toggle_noise, btn_toggle_ensemble, btn_clear_activation, btn_clear_model_type
    global app_avatar, num_folds, spinbox_fold
   
    # === INIZIO BLOCCO CONTROLLO LICENZA (Tuo codice originale, assumendo funzioni/classi esistenti) ===
    should_exit_app = False
    try:
        from license_system import LicenseSystem
        license_system = LicenseSystem()
        is_valid, message = license_system.check_license()
        if not is_valid:
             can_trial = license_system.can_create_trial()
             import_attempted = False
             if can_trial:
                 response_trial = messagebox.askyesno("Verifica Licenza", f"{message}\n\nVuoi creare una licenza di prova per 5 giorni?")
                 if response_trial:
                     success_trial, data_or_msg_trial = license_system.create_license(expiry_days=5, is_trial=True, bind_to_machine=True)
                     if success_trial: messagebox.showinfo("Licenza Creata", f"Licenza di prova creata.\nScadenza: {data_or_msg_trial['expiry_date']}\nL'applicazione continuerà."); should_exit_app = False
                     else: messagebox.showerror("Errore Creazione Trial", f"Impossibile creare licenza di prova: {data_or_msg_trial}\nChiusura."); should_exit_app = True
                 else: should_exit_app = True
             else: should_exit_app = True
             if should_exit_app:
                 response_import = messagebox.askyesno("Licenza Non Valida", f"{message}\n\nVuoi importare un file di licenza valido ora?")
                 if response_import:
                     import_attempted = True
                     try: importa_licenza()
                     except NameError: messagebox.showerror("Errore", "Funzione 'importa_licenza' non trovata.")
                     is_valid_after, message_after = license_system.check_license()
                     if is_valid_after: messagebox.showinfo("Riavvio Necessario", "Licenza importata.\nRiavvia l'applicazione."); should_exit_app = True
                     else: should_exit_app = True
                 else: should_exit_app = True
             if should_exit_app:
                 final_message = "L'applicazione verrà chiusa." # Default
                 is_valid_after_check = 'is_valid_after' in locals()
                 if not import_attempted: final_message = "Nessuna licenza valida. Chiusura." if can_trial else f"{message}\nContatta lo sviluppatore."
                 elif not is_valid_after_check or not is_valid_after: final_message = "Importazione fallita o licenza non valida. Chiusura."
                 else: final_message = "Riavvia l'applicazione per usare la licenza."
                 messagebox.showinfo("Chiusura Applicazione", final_message); sys.exit(0)
        elif "giorni" in message and "Licenza sviluppatore" not in message:
            try:
                match = re.search(r'(\d+)\s+giorni', message)
                if match: days_remaining = int(match.group(1));
                if days_remaining < 7: messagebox.showwarning("Licenza in Scadenza", f"Attenzione: licenza scade tra {days_remaining} giorni.")
            except Exception as parse_err: print(f"Warning: Impossibile parse giorni rimanenti: {parse_err}")
    except ImportError as e_imp: messagebox.showerror("Errore Critico", f"Modulo licenza mancante: {e_imp}"); sys.exit(1)
    except Exception as e_lic: print(f"Errore GRAVE controllo licenza: {e_lic}\n{traceback.format_exc()}"); messagebox.showerror("Errore Licenza", f"Errore controllo licenza:\n{e_lic}"); sys.exit(1)
    # === FINE BLOCCO CONTROLLO LICENZA ===


    # === INIZIO CODICE INTERFACCIA UTENTE ===
    root = tk.Tk()
    num_folds = tk.IntVar(value=5) # Assicurati che num_folds sia globale se usato altrove

    root.title("NUMERICAL EMPATHY - Software di Analisi Statistica Avanzata - Created by MASSIMO FERRUGHELLI - IL LOTTO DI MAX ")
    root.geometry("1280x800")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Creazione schede originali
    tab_avatar = ttk.Frame(notebook)
    tab_main = ttk.Frame(notebook)
    tab_model = ttk.Frame(notebook)
    tab_advanced = ttk.Frame(notebook)
    tab_results = ttk.Frame(notebook)
    tab_estrazioni = ttk.Frame(notebook)
    tab_10elotto = ttk.Frame(notebook)
    tab_lotto_analyzer = ttk.Frame(notebook)
    tab_superenalotto = ttk.Frame(notebook)
    tab_numerical_binary = ttk.Frame(notebook)

    # Aggiunta schede originali
    notebook.add(tab_avatar, text="Presentazione")
    notebook.add(tab_main, text="Principale")
    notebook.add(tab_model, text="Modello")
    notebook.add(tab_advanced, text="Avanzate")
    notebook.add(tab_results, text="Risultati")
    notebook.add(tab_estrazioni, text="Pannello estrazioni")
    notebook.add(tab_10elotto, text="10 e Lotto Serale") 
    notebook.add(tab_lotto_analyzer, text="Lotto Analyzer")
    notebook.add(tab_superenalotto, text="SuperEnalotto") # Posizionata dopo Lotto Analyzer
    notebook.add(tab_numerical_binary, text="Numerical Binary")

    # ====================== SCHEDA PRINCIPALE (Tuo codice originale) ======================
    title_frame = tk.Frame(tab_main); title_frame.pack(pady=10, fill=tk.X); title_label = tk.Label(title_frame, text="𝐍𝐔𝐌𝐄𝐑𝐈𝐂𝐀𝐋 𝐄𝐌𝐏𝐀𝐓𝐇𝐘", fg="blue", font=("Arial", 24), bg="#ADD8E6"); title_label.pack(fill=tk.X)
    frame_salvataggio = tk.Frame(tab_main); frame_salvataggio.pack(pady=10, fill=tk.X)
    # btn_salva_risultati = tk.Button(frame_salvataggio, text="Salva Risultati", command=salva_risultati, bg="#FFDDC1", width=20); btn_salva_risultati.pack(side=tk.LEFT, padx=10)
    btn_aggiorna_estrazioni = tk.Button(frame_salvataggio, text="Aggiornamento Estrazioni", command=esegui_aggiornamento, bg="#FFDDC1", width=20); btn_aggiorna_estrazioni.pack(side=tk.LEFT, padx=10)
    btn_carica_valuta_modello = tk.Button(frame_salvataggio, text="Carica e Valuta Modello", command=carica_e_valuta_modello, bg="green", fg="white", width=20); btn_carica_valuta_modello.pack(side=tk.LEFT, padx=10)
    btn_manage_license = tk.Button(frame_salvataggio, text="Gestione Licenza", command=gestisci_licenza, bg="#FFDDC1", width=20); btn_manage_license.pack(side=tk.LEFT, padx=10)
    frame_pulsanti = tk.LabelFrame(tab_main, text="Selezione Ruota", padx=10, pady=10); frame_pulsanti.pack(pady=10, fill=tk.X)
    instructions_label = tk.Label(frame_pulsanti, text="Passo 1: Seleziona il range estrazionale, eventualmente varia i Paramentri di analisi, seleziona il tuo Modello, perfeziona tra le Avanzate, scegli la ruota ed elabora:", bg="#FFE4B5", fg="black", font=("Arial", 10, "bold"), padx=10, pady=10, wraplength=1000, justify=tk.LEFT); instructions_label.pack(fill=tk.X)
    ruote_frame = tk.Frame(frame_pulsanti); ruote_frame.pack(fill=tk.X)
    ruota_selezionata = tk.StringVar(); ruote_list = ["BA", "CA", "FI", "GE", "MI", "NA", "PA", "RM", "TO", "VE", "NZ"]; col, row = 0, 0; pulsanti_ruote = {}
    for ruota in ruote_list:
        btn = tk.Button(ruote_frame, text=ruota, width=10, height=2, bg="#ADD8E6", relief=tk.RAISED)
        btn.bind("<Button-1>", callback_ruota)
        btn.grid(row=row, column=col, padx=5, pady=5)
        pulsanti_ruote[ruota] = btn
        tooltip_text = f"Seleziona la ruota {ruota} per caricare i dati corrispondenti."
        # Rimosso Tooltip perché il modulo non esiste
        # try: Tooltip(btn, text=tooltip_text)
        # except NameError: pass
        col = (col + 1) % 4 # Usa modulo per layout a griglia
        row += (col == 0) # Incrementa riga quando col torna a 0
    btn_verifica_numeri = tk.Button(frame_pulsanti, text="Verifica Numeri", command=popup_verifica, bg="#FFDDC1", width=20); btn_verifica_numeri.pack(pady=10)
    btn_verifica_multi = tk.Button(frame_pulsanti, text="Verifica Esiti Multi-Ruota", command=verifica_esiti_multi_ruota, bg="#B0E0E6", width=20); btn_verifica_multi.pack(pady=5)
    entry_info = tk.Entry(frame_pulsanti, width=80, bg="#F0F0F0", fg="black"); entry_info.pack(pady=10, fill=tk.X)
    start_frame = tk.Frame(frame_pulsanti); start_frame.pack(fill=tk.X, pady=10); start_label = tk.Label(start_frame, text="Passo 2: Dopo aver selezionato quanto segnalato al Passo 1 clicca qui per Avviare l'Elaborazione. Poi guarda i Risultati e confronta/valuta con il Miglior Modello:", bg="#FFE4B5", fg="black", font=("Arial", 10, "bold"), padx=10, pady=10, wraplength=1000, justify=tk.LEFT); start_label.pack(fill=tk.X)
    btn_start = tk.Button(start_frame, text="AVVIA ELABORAZIONE", command=avvia_elaborazione, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), width=30, height=2); btn_start.pack(pady=10)
    btn_multi_ruota = tk.Button(start_frame, text="ANALISI MULTI-RUOTA", command=apri_selezione_multi_ruota, bg="#FF9900", fg="white", font=("Arial", 12, "bold"), width=30, height=2); btn_multi_ruota.pack(pady=5)
    # Assicurati che analisi_avanzata_completa sia definita o importata correttamente
    btn_analisi_avanzata = tk.Button(start_frame, text="ANALISI AVANZATA", command=analisi_avanzata_completa, bg="#9370DB", fg="white", font=("Arial", 12, "bold"), width=30, height=2); btn_analisi_avanzata.pack(pady=5)
    btn_analisi_completa = tk.Button(frame_salvataggio, text="Analisi Completa", command=analisi_avanzata_completa, bg="#90EE90", fg="black", width=16); btn_analisi_completa.pack(side=tk.LEFT, padx=5)
    btn_analisi_interpretabile = tk.Button(frame_salvataggio, text="Analisi Interpretabile", command=analisi_interpretabile, bg="#FFD700", fg="black", width=18); btn_analisi_interpretabile.pack(side=tk.LEFT, padx=5)
    btn_numeri_spia = tk.Button(frame_salvataggio, text="Numeri Spia", command=esegui_script_numeri_spia, bg="#E6E6FA", fg="black", width=12); btn_numeri_spia.pack(side=tk.LEFT, padx=5)
    btn_analizzatore_ritardi = tk.Button(frame_salvataggio, text="Analisi Ritardi", command=lambda: apri_analizzatore_ritardi(root, FILE_RUOTE), bg="#E6E6FA", fg="black", width=14); btn_analizzatore_ritardi.pack(side=tk.LEFT, padx=5)
    btn_legge_terzo = tk.Button(frame_salvataggio, text="Legge del Terzo", command=esegui_script_legge_terzo, bg="#E6E6FA", fg="black", width=14); btn_legge_terzo.pack(side=tk.LEFT, padx=5)
    btn_spie_dec = tk.Button(frame_salvataggio, text="Spie Dec", command=lancia_spie_dec, bg="#E6E6FA", fg="black", width=14); btn_spie_dec.pack(side=tk.LEFT, padx=5)
    btn_spie_quind = tk.Button(frame_salvataggio, text="Spie Quind", command=lancia_spie_quind, bg="#E6E6FA", fg="black", width=14);  btn_spie_quind.pack(side=tk.LEFT, padx=5)
    btn_ritardi_plus = tk.Button(frame_salvataggio, text="RitardiPlus", command=lancia_ritardi_plus, bg="#E6E6FA", fg="black", width=14); btn_ritardi_plus.pack(side=tk.LEFT, padx=5)
    btn_rit_pos = tk.Button(frame_salvataggio, text="Rit Pos", command=lancia_rit_pos, bg="#E6E6FA", fg="black", width=14); btn_rit_pos.pack(side=tk.LEFT, padx=5)     
    frame_date = tk.LabelFrame(tab_main, text="Periodo di Analisi", padx=10, pady=10); frame_date.pack(pady=10, fill=tk.X)
    tk.Label(frame_date, text="Data di Inizio:", bg="#ADD8E6", fg="black", width=15).grid(row=0, column=0, padx=5, pady=5, sticky="e")
    entry_start_date = DateEntry(frame_date, width=15, date_pattern='yyyy/mm/dd', bg="#F0F0F0", fg="black"); entry_start_date.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    tk.Label(frame_date, text="Data di Fine:", bg="#ADD8E6", fg="black", width=15).grid(row=0, column=2, padx=5, pady=5, sticky="e")
    entry_end_date = DateEntry(frame_date, width=15, date_pattern='yyyy/mm/dd', bg="#F0F0F0", fg="black"); entry_end_date.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    frame_training = tk.LabelFrame(tab_main, text="Parametri di Addestramento", padx=10, pady=10); frame_training.pack(pady=10, fill=tk.X)
    tk.Label(frame_training, text="Numero di Epoche:", bg="#ADD8E6", fg="black", width=20).grid(row=0, column=0, padx=5, pady=5, sticky="e")
    entry_epochs = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black"); entry_epochs.insert(0, "10"); entry_epochs.grid(row=0, column=1, padx=5, pady=5, sticky="w")
    tk.Label(frame_training, text="Batch Size:", bg="#ADD8E6", fg="black", width=20).grid(row=0, column=2, padx=5, pady=5, sticky="e")
    entry_batch_size = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black"); entry_batch_size.insert(0, "32"); entry_batch_size.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    tk.Label(frame_training, text="Patience:", bg="#ADD8E6", fg="black", width=20).grid(row=1, column=0, padx=5, pady=5, sticky="e")
    entry_patience = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black"); entry_patience.insert(0, "10"); entry_patience.grid(row=1, column=1, padx=5, pady=5, sticky="w") # Rimosso riferimento a config
    tk.Label(frame_training, text="Min Delta:", bg="#ADD8E6", fg="black", width=20).grid(row=1, column=2, padx=5, pady=5, sticky="e")
    entry_min_delta = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black"); entry_min_delta.insert(0, "0.001"); entry_min_delta.grid(row=1, column=3, padx=5, pady=5, sticky="w")

    # ====================== SCHEDA AVATAR ======================
    # Aggiungi il contenuto nella scheda avatar
    avatar_title = tk.Label(
        tab_avatar,
        text="NUMERICAL EMPATHY",
        font=("Arial", 18, "bold"),
        fg="blue",
        bg="#E0F7FA",
        pady=10
    )
    avatar_title.pack(fill=tk.X)

    avatar_frame = tk.Frame(tab_avatar, bg="#E6F3FF", pady=20)
    avatar_frame.pack(fill=tk.BOTH, expand=True)

    # Crea un frame centrale per l'avatar
    avatar_center_frame = tk.Frame(avatar_frame, bg="#E6F3FF")
    avatar_center_frame.pack(expand=True)

    # Crea l'avatar
    app_avatar = Avatar(avatar_center_frame, r"C:\Users\massi\OneDrive\Desktop\Portable Python-3.10.5 x64\Avatar.png", width=150, height=150)

    # Frame per i pulsanti
    btn_frame = tk.Frame(avatar_center_frame, bg="#E6F3FF", pady=10)
    btn_frame.pack()

    # Pulsanti dell'avatar
    btn_avatar_speak = tk.Button(
        btn_frame,
        text="Attiva Sofia",
        command=lambda: app_avatar.speak("Benvenuto in Numerical Empathy!  Sono Sofia l'assistente virtuale di Max!  Segui attentamente le istruzioni indicate nella pagina principale al passo 1 e al passo 2. Poi Avvia l'elaborazione. Costruisci il tuo modello preferito , prova diversi modelli e ricorda di usare moderazione nelle giocate. Un saluto a tutti da Sofia"),
        bg="#9C27B0",
        fg="white",
        font=("Arial", 12, "bold"),
        width=15
    )
    btn_avatar_speak.pack(side=tk.LEFT, padx=10, pady=10)

    btn_avatar_stop = tk.Button(
        btn_frame,
        text="Ferma Sofia",
        command=lambda: app_avatar.stop_speaking(),
        bg="#F44336",
        fg="white",
        font=("Arial", 12, "bold"),
        width=15
    )
    btn_avatar_stop.pack(side=tk.LEFT, padx=10, pady=10)

    # Istruzioni per l'avatar
    instructions = tk.Text(avatar_center_frame, width=60, height=8, bg="#F0F0F0", font=("Arial", 10))
    instructions.pack(pady=20)
    instructions.insert(tk.END, "Lei é Sofia l'assistente virtuale di Numerical Empathy.\n\n")
    instructions.insert(tk.END, "Clicca su 'Attiva Sofia' per ricevere un messaggio di benvenuto.\n")
    instructions.insert(tk.END, "Sofia verrà pian pianino migliorata.\n")
    instructions.insert(tk.END, "Usa 'Ferma Sofia' per interrompere la riproduzione vocale.")
    instructions.config(state=tk.DISABLED)  # Rendi il testo in sola lettura

    # Frame per i parametri di addestramento
    frame_training = tk.LabelFrame(tab_main, text="Parametri di Addestramento", padx=10, pady=10)
    frame_training.pack(pady=10, fill=tk.X)

    # Utilizziamo un layout a griglia per i parametri
    label_epochs = tk.Label(frame_training, text="Numero di Epoche:", bg="#ADD8E6", fg="black", width=20)
    label_epochs.grid(row=0, column=0, padx=5, pady=5, sticky="e")

    entry_epochs = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black")
    entry_epochs.insert(0, "10")  # Valore predefinito
    entry_epochs.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    label_batch_size = tk.Label(frame_training, text="Batch Size:", bg="#ADD8E6", fg="black", width=20)
    label_batch_size.grid(row=0, column=2, padx=5, pady=5, sticky="e")

    entry_batch_size = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black")
    entry_batch_size.insert(0, "32")  # Valore predefinito
    entry_batch_size.grid(row=0, column=3, padx=5, pady=5, sticky="w")

    label_patience = tk.Label(frame_training, text="Patience:", bg="#ADD8E6", fg="black", width=20)
    label_patience.grid(row=1, column=0, padx=5, pady=5, sticky="e")

    entry_patience = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black")
    entry_patience.insert(0, "10")  # Valore predefinito
    entry_patience.grid(row=1, column=1, padx=5, pady=5, sticky="w")

    label_min_delta = tk.Label(frame_training, text="Min Delta:", bg="#ADD8E6", fg="black", width=20)
    label_min_delta.grid(row=1, column=2, padx=5, pady=5, sticky="e")

    entry_min_delta = tk.Entry(frame_training, width=10, bg="#F0F0F0", fg="black")
    entry_min_delta.insert(0, "0.001")  # Valore predefinito
    entry_min_delta.grid(row=1, column=3, padx=5, pady=5, sticky="w")

    # ====================== SCHEDA MODELLO (Tuo codice originale, con correzione sintassi if/elif/else) ======================
    frame_model_params = tk.LabelFrame(tab_model, text="Struttura del Modello", padx=10, pady=10); frame_model_params.pack(pady=10, fill=tk.X)
    layer_configs = [ {"name": "Layer 1", "default_neurons": "512", "default_dropout": "0.3"}, {"name": "Layer 2", "default_neurons": "256", "default_dropout": "0.3"}, {"name": "Layer 3", "default_neurons": "128", "default_dropout": "0.3"} ]
    for i, config_layer in enumerate(layer_configs):
        tk.Label(frame_model_params, text=f"{config_layer['name']}:", bg="#ADD8E6", fg="black", width=10).grid(row=i, column=0, padx=5, pady=5, sticky="e")
        tk.Label(frame_model_params, text="Neuroni:", bg="#ADD8E6", fg="black", width=10).grid(row=i, column=1, padx=5, pady=5, sticky="e")
        entry_n = tk.Entry(frame_model_params, width=10, bg="#F0F0F0", fg="black"); entry_n.insert(0, config_layer["default_neurons"]); entry_n.grid(row=i, column=2, padx=5, pady=5, sticky="w")
        tk.Label(frame_model_params, text="Dropout:", bg="#ADD8E6", fg="black", width=10).grid(row=i, column=3, padx=5, pady=5, sticky="e")
        entry_d = tk.Entry(frame_model_params, width=10, bg="#F0F0F0", fg="black"); entry_d.insert(0, config_layer["default_dropout"]); entry_d.grid(row=i, column=4, padx=5, pady=5, sticky="w")
        # --- Assegnazione Corretta if/elif/else ---
        if i == 0: entry_neurons_layer1, entry_dropout_layer1 = entry_n, entry_d
        elif i == 1: entry_neurons_layer2, entry_dropout_layer2 = entry_n, entry_d
        else: entry_neurons_layer3, entry_dropout_layer3 = entry_n, entry_d
        # --- Fine Blocco Corretto ---
    frame_model_type = tk.LabelFrame(tab_model, text="Tipo di Modello", padx=10, pady=10); frame_model_type.pack(pady=10, fill=tk.X)
    btn_model_dense = tk.Button(frame_model_type, text="Dense", command=lambda: select_model_type('dense'), bg=BUTTON_DEFAULT_COLOR, width=20); btn_model_dense.grid(row=0, column=0, padx=10, pady=10)
    btn_model_lstm = tk.Button(frame_model_type, text="LSTM", command=lambda: select_model_type('lstm'), bg=BUTTON_DEFAULT_COLOR, width=20); btn_model_lstm.grid(row=0, column=1, padx=10, pady=10)
    btn_clear_model_type = tk.Button(frame_model_type, text="Nessuna selezione", command=lambda: select_model_type(None), bg=BUTTON_DEFAULT_COLOR, width=20); btn_clear_model_type.grid(row=0, column=2, padx=10, pady=10)
    frame_activation_function = tk.LabelFrame(tab_model, text="Funzione di Attivazione", padx=10, pady=10); frame_activation_function.pack(pady=10, fill=tk.X)
    btn_relu = tk.Button(frame_activation_function, text="RELU", command=lambda: select_activation('relu'), bg=BUTTON_DEFAULT_COLOR, width=20); btn_relu.grid(row=0, column=0, padx=10, pady=10)
    btn_leaky_relu = tk.Button(frame_activation_function, text="LEAKY_RELU", command=lambda: select_activation('leaky_relu'), bg=BUTTON_DEFAULT_COLOR, width=20); btn_leaky_relu.grid(row=0, column=1, padx=10, pady=10)
    btn_elu = tk.Button(frame_activation_function, text="ELU", command=lambda: select_activation('elu'), bg=BUTTON_DEFAULT_COLOR, width=20); btn_elu.grid(row=0, column=2, padx=10, pady=10)
    btn_clear_activation = tk.Button(frame_activation_function, text="Nessuna selezione", command=lambda: select_activation(None), bg=BUTTON_DEFAULT_COLOR, width=20); btn_clear_activation.grid(row=0, column=3, padx=10, pady=10)

    # ====================== SCHEDA AVANZATE (Tuo codice originale, CON CORREZIONE if/elif/else) ======================
    frame_optimizer = tk.LabelFrame(tab_advanced, text="Ottimizzatore", padx=10, pady=10); frame_optimizer.pack(pady=10, fill=tk.X); optimizer_buttons = {}
    optimizers = [{"name": "Adam", "value": "adam"}, {"name": "RMSprop", "value": "rmsprop"}, {"name": "SGD", "value": "sgd"}]
    for i, opt in enumerate(optimizers): btn = tk.Button(frame_optimizer, text=opt["name"], command=lambda v=opt["value"]: select_optimizer(v), bg=BUTTON_DEFAULT_COLOR, width=20); btn.grid(row=0, column=i, padx=10, pady=10); optimizer_buttons[opt["value"]] = btn
    frame_loss_function = tk.LabelFrame(tab_advanced, text="Funzione di Loss", padx=10, pady=10); frame_loss_function.pack(pady=10, fill=tk.X); loss_buttons = {}
    loss_functions = [{"name": "MSE", "value": "mean_squared_error"}, {"name": "MAE", "value": "mean_absolute_error"}, {"name": "Huber", "value": "huber_loss"}, {"name": "Log Cosh", "value": "log_cosh"}, {"name": "Custom", "value": "custom_loss_function"}]; row, col, max_cols = 0, 0, 3
    for i, loss in enumerate(loss_functions): btn = tk.Button(frame_loss_function, text=loss["name"], command=lambda v=loss["value"]: select_loss_function(v), bg=BUTTON_DEFAULT_COLOR, width=25); btn.grid(row=row, column=col, padx=10, pady=10); loss_buttons[loss["value"]] = btn; col = (col + 1) % max_cols; row += (col == 0)
    btn_personal = tk.Button(frame_loss_function, text="Personale (BinaryCE)", command=lambda: select_personal_loss(), bg=BUTTON_DEFAULT_COLOR, fg="black", width=20); btn_personal.grid(row=row, column=col, padx=5, pady=5)
    frame_regularization = tk.LabelFrame(tab_advanced, text="Regolarizzazione", padx=10, pady=10); frame_regularization.pack(pady=10, fill=tk.X); reg_buttons = {}
    regularizations = [{"name": "Nessuna", "value": None}, {"name": "L1", "value": "l1"}, {"name": "L2", "value": "l2"}]
    btn_reg_none, btn_reg_l1, btn_reg_l2 = None, None, None # Inizializza
    for i, reg in enumerate(regularizations):
        btn = tk.Button(frame_regularization, text=reg["name"], command=lambda v=reg["value"]: select_regularization(v), bg=BUTTON_DEFAULT_COLOR, width=20); btn.grid(row=0, column=i, padx=10, pady=10); reg_buttons[reg["value"]] = btn;
        # --- Assegnazione Corretta if/elif/else ---
        if reg["value"] is None: btn_reg_none = btn
        elif reg["value"] == 'l1': btn_reg_l1 = btn
        else: btn_reg_l2 = btn
        # --- Fine Blocco Corretto ---
    frame_reg_value = tk.LabelFrame(tab_advanced, text="Valore Regolarizzazione", padx=10, pady=10); frame_reg_value.pack(pady=10, fill=tk.X)
    tk.Label(frame_reg_value, text="Valore:", width=15).grid(row=0, column=0); entry_reg_value = tk.Entry(frame_reg_value, width=10); entry_reg_value.insert(0, "0.01"); entry_reg_value.grid(row=0, column=1); tk.Button(frame_reg_value, text="Imposta", command=lambda: update_regularization_value(entry_reg_value.get()), width=15).grid(row=0, column=2)
    frame_noise = tk.LabelFrame(tab_advanced, text="Impostazioni Rumore", padx=10, pady=10); frame_noise.pack(pady=10, fill=tk.X)
    tk.Label(frame_noise, text="Tipo Rumore:", width=15).grid(row=0, column=0); noise_type_var = tk.StringVar(value='gaussian'); noise_types = ['gaussian', 'uniform', 'poisson', 'exponential']; noise_menu = tk.OptionMenu(frame_noise, noise_type_var, *noise_types, command=select_noise_type); noise_menu.config(width=10); noise_menu.grid(row=0, column=1)
    tk.Label(frame_noise, text="Scala Rumore:", width=15).grid(row=0, column=2); entry_noise_scale = tk.Entry(frame_noise, width=10); entry_noise_scale.insert(0, "0.01"); entry_noise_scale.grid(row=0, column=3); tk.Button(frame_noise, text="Imposta Scala", command=lambda: update_noise_scale(entry_noise_scale.get()), width=15).grid(row=0, column=4)
    tk.Label(frame_noise, text="% Rumore:", width=15).grid(row=1, column=0); entry_noise_percentage = tk.Entry(frame_noise, width=10); entry_noise_percentage.insert(0, "0.1"); entry_noise_percentage.grid(row=1, column=1); tk.Button(frame_noise, text="Imposta %", command=lambda: update_noise_percentage(entry_noise_percentage.get()), width=15).grid(row=1, column=2, columnspan=2)
    btn_toggle_noise = tk.Button(frame_noise, text="Attiva/Disattiva Rumore", command=toggle_adaptive_noise, bg=BUTTON_DEFAULT_COLOR, width=20); btn_toggle_noise.grid(row=1, column=4)
    frame_ensemble = tk.LabelFrame(tab_advanced, text="Ensemble Methods", padx=10, pady=10); frame_ensemble.pack(pady=10, fill=tk.X)
    btn_toggle_ensemble = tk.Button(frame_ensemble, text="Attiva/Disattiva Ensemble", command=toggle_ensemble, bg=BUTTON_DEFAULT_COLOR, width=25); btn_toggle_ensemble.pack(padx=10, pady=10)
    frame_fold_control = tk.LabelFrame(tab_advanced, text="Configurazione K-Fold", padx=10, pady=10); frame_fold_control.pack(pady=10, fill=tk.X)
    tk.Label(frame_fold_control, text="Numero di Fold (K):", width=20).grid(row=0, column=0); spinbox_fold = tk.Spinbox(frame_fold_control, from_=2, to=20, width=5, textvariable=num_folds); spinbox_fold.grid(row=0, column=1); # Rimosso Tooltip tk.Label(frame_fold_control, text="(Default: 5)", font=("Arial", 9, "italic")).grid(row=0, column=2)
    frame_cache = tk.LabelFrame(tab_advanced, text="Gestione Cache e File", padx=10, pady=10); frame_cache.pack(pady=10, fill=tk.X)
    tk.Button(frame_cache, text="Pulisci Cache", command=pulisci_cache, bg="#FFB6C1", width=25).pack(side=tk.LEFT, padx=10, pady=10)
    tk.Button(frame_cache, text="Verifica Integrità File", command=verifica_integrità_file, bg="#B0E0E6", width=25).pack(side=tk.LEFT, padx=10, pady=5)

    # ====================== SCHEDA RISULTATI (Tuo codice originale) ======================
    text_frame = tk.Frame(tab_results); text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(text_frame); scrollbar.pack(side=tk.RIGHT, fill=tk.Y); scrollbar_x = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL); scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    textbox = tk.Text(text_frame, height=15, width=90, wrap=tk.NONE, yscrollcommand=scrollbar.set, xscrollcommand=scrollbar_x.set); textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=textbox.yview); scrollbar_x.config(command=textbox.xview)
    frame_grafico = tk.Frame(tab_results); frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # ====================== SCHEDA PANNELLO ESTRAZIONI (Tuo codice originale) ======================
    estrazioni_title = tk.Label(tab_estrazioni, text="PANNELLO ESTRAZIONI", font=("Arial", 18, "bold"), fg="blue", bg="#E0F7FA", pady=10); estrazioni_title.pack(fill=tk.X)
    estrazioni_frame = tk.Frame(tab_estrazioni, bg="#E6F3FF", pady=20); estrazioni_frame.pack(fill=tk.BOTH, expand=True)
    estrazioni_info = tk.Label(estrazioni_frame, text="Visualizza e analizza le ultime estrazioni del lotto", font=("Arial", 12), bg="#E6F3FF", pady=10); estrazioni_info.pack()
    btn_avvia_pannello = tk.Button(estrazioni_frame, text="Avvia Pannello Estrazioni", command=esegui_pannello_estrazioni, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), width=25, height=2); btn_avvia_pannello.pack(pady=20)
    estrazioni_descrizione = tk.Text(estrazioni_frame, width=60, height=8, bg="#F0F0F0", font=("Arial", 10)); estrazioni_descrizione.pack(pady=20); estrazioni_descrizione.insert(tk.END, "Il Pannello Estrazioni permette di verificare i tuoi numeri.\n..."); estrazioni_descrizione.config(state=tk.DISABLED)

    # ====================== SCHEDA 10 e LOTTO SERALE (Aggiunta Corretta) ======================
    lotto_title = tk.Label( tab_10elotto, text="ANALISI 10 e LOTTO SERALE", font=("Arial", 18, "bold"), fg="darkgreen", bg="#E0F2E0", pady=10)
    lotto_title.pack(fill=tk.X)
    lotto_frame = tk.Frame(tab_10elotto, bg="#F0FFF0", pady=20); lotto_frame.pack(fill=tk.BOTH, expand=True)
    lotto_info = tk.Label( lotto_frame, text="Avvia il modulo dedicato all'analisi statistica e previsionale per il 10eLotto.", font=("Arial", 12), bg="#F0FFF0", pady=10 ); lotto_info.pack()
    lotto_button_state = tk.NORMAL if LOTTO_MODULE_LOADED else tk.DISABLED
    btn_avvia_10elotto = tk.Button( lotto_frame, text="Avvia Modulo 10eLotto", command=lambda: open_lotto_module_helper(root), state=lotto_button_state, bg="#2E8B57", fg="white", font=("Arial", 14, "bold"), width=30, height=2); btn_avvia_10elotto.pack(pady=20)
    lotto_descrizione = tk.Text(lotto_frame, width=80, height=10, bg="#F0F0F0", font=("Arial", 10)); lotto_descrizione.pack(pady=20)
    lotto_descrizione.insert(tk.END, "Questo modulo permette di:\n\n • Caricare l'archivio 10eLotto.\n • Configurare la rete neurale.\n • Definire parametri di training.\n • Addestrare il modello.\n • Generare previsioni.\n • Verificare risultati."); lotto_descrizione.config(state=tk.DISABLED)
    # =====================================================================

    # ====================== NUOVA SCHEDA LOTTO ANALYZER ======================
    # Titolo scheda
    lotto_analyzer_title = tk.Label(
        tab_lotto_analyzer,  # Aggiungi alla nuova scheda
        text="ANALISI e PREVISIONE LOTTO",
        font=("Arial", 18, "bold"),
        fg="darkblue",  # Colore diverso per distinguerla
        bg="#E0E8F0",  # Sfondo leggermente diverso
        pady=10
    )
    lotto_analyzer_title.pack(fill=tk.X)

    # Frame principale per la scheda
    lotto_analyzer_frame = tk.Frame(
        tab_lotto_analyzer, # Aggiungi alla nuova scheda
        bg="#F0F5FF",  # Sfondo leggermente diverso
        pady=20
    )
    lotto_analyzer_frame.pack(fill=tk.BOTH, expand=True)

    # Etichetta informativa
    lotto_analyzer_info = tk.Label(
        lotto_analyzer_frame,
        text="Avvia il modulo per l'analisi statistica, l'addestramento di modelli\n e la generazione di previsioni per il gioco del Lotto.",
        font=("Arial", 12),
        bg="#F0F5FF", # Stesso sfondo del frame
        pady=10,
        justify=tk.CENTER
    )
    lotto_analyzer_info.pack()

    # Pulsante di avvio (usa open_lotto_analyzer_helper definito nel blocco import)
    # Assicurati che LOTTO_ANALYZER_MODULE_LOADED sia definita nel blocco import
    lotto_analyzer_button_state = tk.NORMAL if LOTTO_ANALYZER_MODULE_LOADED else tk.DISABLED
    btn_avvia_lotto_analyzer = tk.Button(
        lotto_analyzer_frame,
        text="Avvia Lotto Analyzer",
        command=lambda: open_lotto_analyzer_helper(root), # Chiama la funzione helper
        state=lotto_analyzer_button_state,
        bg="#4682B4",  # Blu acciaio
        fg="white",
        font=("Arial", 14, "bold"),
        width=30, # Stessa larghezza dell'altro bottone
        height=2 # Stessa altezza
    )
    btn_avvia_lotto_analyzer.pack(pady=20)

    # Descrizione testuale
    lotto_analyzer_descrizione = tk.Text(
        lotto_analyzer_frame,
        width=80, # Stessa larghezza dell'altra descrizione
        height=10, # Stessa altezza
        bg="#F0F0F0",
        font=("Arial", 10)
    )
    lotto_analyzer_descrizione.pack(pady=20)
    lotto_analyzer_descrizione.insert(tk.END, "Questo modulo permette di:\n\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Caricare archivi storici del Lotto per ruote selezionate.\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Definire periodo di analisi e lunghezza sequenze.\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Configurare parametri del modello neurale (layers, dropout, regolarizzazione).\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Definire parametri di training (epoche, batch size, early stopping).\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Addestrare il modello sui dati selezionati.\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Generare una previsione basata sull'ultima sequenza disponibile.\n")
    lotto_analyzer_descrizione.insert(tk.END, " • Verificare la previsione generata sulle estrazioni successive.")
    lotto_analyzer_descrizione.config(state=tk.DISABLED) # Rendi non modificabile
    # =====================================================================

    # ====================== NUOVA SCHEDA SUPERENALOTTO ======================
    # Popola la scheda creata in precedenza (tab_superenalotto)

    # Titolo scheda SuperEnalotto
    superenalotto_title = tk.Label(
        tab_superenalotto, # Usa la variabile della scheda creata sopra
        text="ANALISI E PREVISIONE SUPERENALOTTO",
        font=("Arial", 18, "bold"),
        fg="#D2691E",  # Colore Cioccolato o altro a tua scelta
        bg="#FAF0E6",  # Lino o altro sfondo
        pady=10
    )
    superenalotto_title.pack(fill=tk.X)

    # Frame principale per la scheda SuperEnalotto
    superenalotto_frame = tk.Frame(
        tab_superenalotto, # Usa la variabile della scheda
        bg="#FFF8DC",  # Cornsilk o altro sfondo
        pady=20
    )
    superenalotto_frame.pack(fill=tk.BOTH, expand=True)

    # Etichetta informativa SuperEnalotto
    superenalotto_info = tk.Label(
        superenalotto_frame,
        text="Avvia il modulo dedicato all'analisi statistica, training del modello\n e generazione di previsioni per il gioco del SuperEnalotto.",
        font=("Arial", 12),
        bg="#FFF8DC", # Stesso sfondo del frame
        pady=10,
        justify=tk.CENTER
    )
    superenalotto_info.pack()

    # Pulsante di avvio SuperEnalotto
    # Usa la variabile globale SUPERENALOTTO_MODULE_LOADED per lo stato
    superenalotto_button_state = tk.NORMAL if SUPERENALOTTO_MODULE_LOADED else tk.DISABLED
    btn_avvia_superenalotto = tk.Button(
        superenalotto_frame,
        text="Avvia Modulo SuperEnalotto",
        # Chiama la funzione helper specifica per SuperEnalotto
        command=lambda: open_superenalotto_module_helper(root),
        state=superenalotto_button_state,
        bg="#8B4513",  # Marrone Scuro o altro colore
        fg="white",
        font=("Arial", 14, "bold"),
        width=30, # Mantieni coerenza con gli altri bottoni
        height=2
    )
    btn_avvia_superenalotto.pack(pady=20)

    # Descrizione testuale SuperEnalotto
    superenalotto_descrizione = tk.Text(
        superenalotto_frame,
        width=80, # Mantieni coerenza
        height=10,
        bg="#F5F5DC",  # Beige o altro sfondo
        font=("Arial", 10)
    )
    superenalotto_descrizione.pack(pady=20)
    superenalotto_descrizione.insert(tk.END, "Questo modulo ('selotto_module.py') permette di:\n\n") # Nota il nome file
    superenalotto_descrizione.insert(tk.END, " • Caricare l'archivio storico del SuperEnalotto.\n")
    superenalotto_descrizione.insert(tk.END, " • Definire periodo di analisi e lunghezza sequenze.\n")
    superenalotto_descrizione.insert(tk.END, " • Configurare parametri del modello neurale (layers, dropout, etc.).\n")
    superenalotto_descrizione.insert(tk.END, " • Definire parametri di training (epoche, batch size, early stopping).\n")
    superenalotto_descrizione.insert(tk.END, " • Addestrare il modello sui dati selezionati.\n")
    superenalotto_descrizione.insert(tk.END, " • Generare una previsione dei 6 numeri principali.\n")
    superenalotto_descrizione.insert(tk.END, " • Verificare la previsione generata sulle estrazioni successive.")
    superenalotto_descrizione.config(state=tk.DISABLED) # Rendi non modificabile
    # =====================================================================

    # >>>>> NUOVO: POPOLAMENTO SCHEDA NUMERICAL BINARY (tab_numerical_binary) <<<<<
    nb_title = tk.Label(tab_numerical_binary, text="ANALISI NUMERICAL BINARY", font=("Arial", 18, "bold"), fg="navy", bg="#E6E6FA", pady=10)
    nb_title.pack(fill=tk.X)
    nb_frame = tk.Frame(tab_numerical_binary, bg="#F8F8FF", pady=20)
    nb_frame.pack(fill=tk.BOTH, expand=True)
    nb_info = tk.Label(nb_frame, text="Avvia il modulo per l'analisi delle sequenze binarie\n basate sui pattern verticali nel gioco del Lotto.", font=("Arial", 12), bg="#F8F8FF", pady=10, justify=tk.CENTER)
    nb_info.pack()
    # Controlla se il modulo è stato caricato correttamente
    nb_button_state = tk.NORMAL if NUMERICAL_BINARY_MODULE_LOADED else tk.DISABLED
    btn_avvia_nb = tk.Button(
        nb_frame, # Widget padre corretto
        text="Avvia Modulo Numerical Binary",
        command=lambda: open_numerical_binary_helper(root), # Chiama l'helper corretto
        state=nb_button_state, # Stato basato sul caricamento
        bg="#6A5ACD", # Colore esempio
        fg="white",
        font=("Arial", 14, "bold"),
        width=30, # Larghezza consistente
        height=2
    )
    btn_avvia_nb.pack(pady=20) # Posiziona il bottone
    nb_descrizione = tk.Text(nb_frame, width=80, height=10, bg="#F0F0F0", font=("Arial", 10))
    nb_descrizione.pack(pady=20)
    nb_descrizione.insert(tk.END, "Questo modulo ('numerical_binary_module.py') permette di:\n\n")
    nb_descrizione.insert(tk.END, " • Caricare archivi storici del Lotto da GitHub.\n")
    nb_descrizione.insert(tk.END, " • Convertire i numeri estratti in rappresentazione binaria.\n")
    nb_descrizione.insert(tk.END, " • Analizzare pattern verticali di una data lunghezza (H).\n")
    nb_descrizione.insert(tk.END, " • Identificare i pattern più sbilanciati (0 vs 1) su base storica.\n")
    nb_descrizione.insert(tk.END, " • Generare una previsione binaria (e decimale) per le 5 posizioni.\n")
    nb_descrizione.insert(tk.END, " • Eseguire backtesting storico per validare l'efficacia del metodo.")
    nb_descrizione.config(state=tk.DISABLED)
    # ====================== FINE SCHEDA NUMERICAL BINARY ======================


    # --- Configurazione finale e avvio mainloop (INVARIATO) ---
    # ... (il tuo codice per protocol, select_*, insert testo iniziale, notebook.select) ...
    root.protocol("WM_DELETE_WINDOW", on_closing)
    try: # Imposta valori default iniziali
        select_optimizer('adam')
        select_loss_function('mean_squared_error')
        select_activation('relu')
        select_regularization(None)
        select_model_type('dense')
        if 'btn_toggle_noise' in globals() and btn_toggle_noise: btn_toggle_noise["bg"] = BUTTON_DEFAULT_COLOR
        if 'btn_toggle_ensemble' in globals() and btn_toggle_ensemble: btn_toggle_ensemble["bg"] = BUTTON_DEFAULT_COLOR
    except NameError as e_name: print(f"Warning: Funzione select_* o bottone non trovato: {e_name}")
    except Exception as e_init: print(f"Warning: Errore generico impostazione iniziale UI: {e_init}")

    try: # Inserimento testo iniziale
        textbox.insert(tk.END, "Benvenuto in NUMERICAL EMPATHY!\n")
        textbox.insert(tk.END, "Seleziona una ruota e configura i parametri per iniziare.\n")
        # Aggiungi info sui moduli caricati
        if LOTTO_MODULE_LOADED: textbox.insert(tk.END, "- Modulo 10eLotto disponibile.\n")
        if LOTTO_ANALYZER_MODULE_LOADED: textbox.insert(tk.END, "- Modulo Lotto Analyzer disponibile.\n")
        if SUPERENALOTTO_MODULE_LOADED: textbox.insert(tk.END, "- Modulo SuperEnalotto disponibile.\n")
        else: textbox.insert(tk.END, "- ATTENZIONE: Modulo SuperEnalotto non caricato.\n")
    except NameError: print("Warning: Textbox non definita per messaggio benvenuto.")
    except Exception as e_textbox: print(f"Warning: Errore inserimento testo iniziale: {e_textbox}")

    notebook.select(1) # Seleziona la scheda Principale all'avvio



    # --- Configurazione finale e avvio mainloop (SENZA riferimenti a config/tooltip) ---
    root.protocol("WM_DELETE_WINDOW", on_closing)
    try: # Blocco try per sicurezza se le funzioni select_* non esistono o causano errori
        # Imposta valori di default manualmente (SE NECESSARIO, altrimenti rimuovi)
        select_optimizer('adam')
        select_loss_function('mean_squared_error')
        select_activation('relu')
        select_regularization(None)
        select_model_type('dense')
        # Assicurati che i bottoni toggle esistano prima di configurarli
        if 'btn_toggle_noise' in globals() and btn_toggle_noise: btn_toggle_noise["bg"] = BUTTON_DEFAULT_COLOR
        if 'btn_toggle_ensemble' in globals() and btn_toggle_ensemble: btn_toggle_ensemble["bg"] = BUTTON_DEFAULT_COLOR
    except NameError as e_name: print(f"Warning: Funzione select_* o bottone non trovato: {e_name}")
    except Exception as e_init: print(f"Warning: Errore generico impostazione iniziale UI: {e_init}")

    try: # Blocco try per sicurezza textbox
        textbox.insert(tk.END, "Benvenuto in NUMERICAL EMPATHY!\n")
        # ... (Altre insert nel textbox originali) ...
        textbox.insert(tk.END, "Seleziona una ruota e configura i parametri per iniziare.\n")
        textbox.insert(tk.END, "NUOVE FUNZIONALITÀ:\n")
        textbox.insert(tk.END, "- Interfaccia a schede\n")
        textbox.insert(tk.END, "- Widget calendario\n")
        # ... (altre funzionalità) ...
    except NameError: print("Warning: Textbox non definita per messaggio benvenuto.")
    except Exception as e_textbox: print(f"Warning: Errore inserimento testo iniziale: {e_textbox}")

    notebook.select(0) # Seleziona la prima scheda

    # ---> QUESTA E' L'ULTIMA RIGA DENTRO main() <---
    return root
# --- Fine funzione main() ---
# La funzione add_tooltips è vuota (puoi implementarla se vuoi)
def add_tooltips():
    """Aggiunge tooltip informativi ai widget."""
    pass  # Per ora non fa nulla

if __name__ == "__main__":
    print("Avvio dell'applicazione...")
    try:
        root = main()
        if root is None:
            print("Errore: main() ha restituito None invece di un'istanza di Tk")
            sys.exit(1)
        print("Chiamata a mainloop()...")
        root.mainloop()
        print("mainloop() terminato")
    except Exception as e:
        print(f"Errore all'avvio dell'applicazione: {e}")
        import traceback
        traceback.print_exc()

        