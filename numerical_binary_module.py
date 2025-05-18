import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from tkcalendar import DateEntry
import queue
from threading import Thread
import requests
import traceback
import io
import pandas as pd
from datetime import datetime, timedelta
# from itertools import product # Non più usato direttamente in questo file
import math
from abc import ABC, abstractmethod # Aggiunto per PredictionStrategy

# --- Configurazione GitHub ---
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"

# Definizione URL Ruote
URL_RUOTE = {
    'BA': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/BARI.txt',
    'CA': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/CAGLIARI.txt',
    'FI': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/FIRENZE.txt',
    'GE': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/GENOVA.txt',
    'MI': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/MILANO.txt',
    'NA': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/NAPOLI.txt',
    'PA': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/PALERMO.txt',
    'RO': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/ROMA.txt',
    'TO': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/TORINO.txt',
    'VE': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/VENEZIA.txt',
    'NZ': f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/NAZIONALE.txt'
}

# --- Costanti Configurabili ---
DEFAULT_LUNGHEZZA_PATTERN_VERTICALE = 4 # Default per la strategia
DEFAULT_COLPI_VERIFICA = 9
NUM_POSIZIONI = 5
BITS_PER_NUMERO = 7

# ==============================================================================
# DEFINIZIONE STRATEGIE DI PREDIZIONE
# ==============================================================================
class PredictionStrategy(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    @abstractmethod
    def get_description(self) -> str:
        pass
    @abstractmethod
    def predict(self, data_to_analyze: pd.DataFrame, bit_offset: int, **kwargs) -> tuple[list[int], list[str]]:
        pass
    def get_config_options(self) -> list[dict]: # Opzioni di configurazione per la UI
        return []

class HorizontalIntraPositionPatternStrategy(PredictionStrategy):
    def __init__(self, lunghezza_pattern_orizzontale=3): # Quanti bit precedenti guardare
        self.lunghezza_pattern = int(lunghezza_pattern_orizzontale)
        if self.lunghezza_pattern < 1:
            self.lunghezza_pattern = 1
        self.name = "Pattern Orizzontali Intra-Posizione"
        self.description = f"Analizza pattern orizzontali di {self.lunghezza_pattern} bit adiacenti precedenti per predire il bit corrente."

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_config_options(self) -> list[dict]:
        return [
            {'name': 'lunghezza_pattern_orizzontale', 'type': 'int', 'default': self.lunghezza_pattern,
             'label': 'Lunghezza Pattern Orizz. (K):', 'min': 1, 'max': BITS_PER_NUMERO -1, # Max è BITS_PER_NUMERO-1
             'var_name': 'lunghezza_pattern_orizz_var'} # Dovrai creare questa var in setup_variables
        ]

    def predict(self, data_to_analyze: pd.DataFrame, bit_offset: int, **kwargs) -> tuple[list[int], list[str]]:
        final_predicted_bits = [0] * BITS_PER_NUMERO
        analysis_details = [""] * BITS_PER_NUMERO

        current_k = kwargs.get('lunghezza_pattern_orizzontale', self.lunghezza_pattern)

        if not isinstance(data_to_analyze, pd.DataFrame) or 'BinarioCompleto' not in data_to_analyze.columns:
            # ... (gestione errore dati) ...
            return final_predicted_bits, analysis_details

        # Estrai i 7 bit per la posizione corrente da tutte le estrazioni storiche
        # Esempio: se bit_offset = 0 (P1), vogliamo le colonne di bit da 0 a 6
        # Se bit_offset = 7 (P2), vogliamo le colonne di bit da 7 a 13
        
        # Prendiamo solo la porzione rilevante di BinarioCompleto per questa posizione
        # e la dividiamo in 7 bit individuali per ogni estrazione
        
        historical_position_bits_list = []
        for bin_completo_str in data_to_analyze['BinarioCompleto']:
            if len(bin_completo_str) >= bit_offset + BITS_PER_NUMERO:
                numero_bin_str = bin_completo_str[bit_offset : bit_offset + BITS_PER_NUMERO]
                historical_position_bits_list.append([int(b) for b in numero_bin_str])
            # else: gestisci estrazioni con stringa binaria troppo corta, se possibile

        if not historical_position_bits_list or len(historical_position_bits_list) < 2: # Almeno 2 per avere pattern e outcome
            for i in range(BITS_PER_NUMERO):
                analysis_details[i] = f"Bit PosRel {i}(Abs {bit_offset+i}): Pred=0 (Dati storici insuff. per pos)"
            return final_predicted_bits, analysis_details

        # Invertiamo: ora la prima riga è l'estrazione più recente
        reversed_historical_position_bits = list(reversed(historical_position_bits_list))

        for bit_idx_target in range(BITS_PER_NUMERO): # Da 0 a 6 (posizione relativa all'interno del numero)
            if bit_idx_target < current_k:
                # Non ci sono abbastanza bit precedenti per formare un pattern per i primi K bit
                final_predicted_bits[bit_idx_target] = 0 # O altra logica di default
                analysis_details[bit_idx_target] = f"Bit PosRel {bit_idx_target}(Abs {bit_offset+bit_idx_target}): Pred=0 (K={current_k} > Indice)"
                continue

            pattern_stats = {} # { pattern_tuple: {'matches': 0, 'zeros': 0, 'ones': 0} }

            # Analizziamo le estrazioni storiche (esclusa la più recente, che è quella da predire)
            # In realtà, qui stiamo predicendo i bit per la *prossima* estrazione non vista,
            # quindi usiamo tutte le estrazioni fornite in reversed_historical_position_bits
            # come base per trovare i pattern e cosa li segue.
            # L'outcome è il bit_idx_target della *stessa riga* (estrazione)
            
            for i in range(len(reversed_historical_position_bits)): # Per ogni estrazione storica
                estrazione_corrente_bits = reversed_historical_position_bits[i]

                # Forma il pattern dai k bit precedenti al bit_idx_target
                start_pattern_idx = bit_idx_target - current_k
                pattern_tuple = tuple(estrazione_corrente_bits[start_pattern_idx : bit_idx_target])
                
                following_bit = estrazione_corrente_bits[bit_idx_target] # Il bit che vogliamo predire

                if pattern_tuple not in pattern_stats:
                    pattern_stats[pattern_tuple] = {'matches': 0, 'zeros': 0, 'ones': 0}
                
                pattern_stats[pattern_tuple]['matches'] += 1
                if following_bit == 1:
                    pattern_stats[pattern_tuple]['ones'] += 1
                else:
                    pattern_stats[pattern_tuple]['zeros'] += 1
            
            # Ora, trova il pattern più recente nei dati per fare la predizione effettiva.
            # Il pattern più recente è quello dell'ultima estrazione disponibile.
            if not reversed_historical_position_bits: # Dovrebbe essere già gestito
                final_predicted_bits[bit_idx_target] = 0
                analysis_details[bit_idx_target] = f"Bit PosRel {bit_idx_target}(Abs {bit_offset+bit_idx_target}): Pred=0 (No dati recenti)"
                continue

            last_extraction_bits = reversed_historical_position_bits[0] # Estrazione più recente
            
            # Pattern da cercare per la predizione attuale
            actual_pattern_to_predict_from = tuple(last_extraction_bits[bit_idx_target - current_k : bit_idx_target])

            if actual_pattern_to_predict_from in pattern_stats:
                stats = pattern_stats[actual_pattern_to_predict_from]
                if stats['ones'] > stats['zeros']:
                    final_predicted_bits[bit_idx_target] = 1
                elif stats['zeros'] > stats['ones']:
                    final_predicted_bits[bit_idx_target] = 0
                else: # Parità, o nessuna occorrenza forte
                    final_predicted_bits[bit_idx_target] = 0 # Default o altra logica (es. bit più frequente in generale)
                
                pattern_str = "".join(map(str, actual_pattern_to_predict_from))
                analysis_details[bit_idx_target] = (
                    f"Bit PosRel {bit_idx_target}(Abs {bit_offset+bit_idx_target}): Pred={final_predicted_bits[bit_idx_target]} "
                    f"(PatOriz='{pattern_str}': Z={stats['zeros']}, U={stats['ones']}, M={stats['matches']})"
                )
            else:
                final_predicted_bits[bit_idx_target] = 0 # Pattern non visto o non significativo
                pattern_str = "".join(map(str, actual_pattern_to_predict_from))
                analysis_details[bit_idx_target] = f"Bit PosRel {bit_idx_target}(Abs {bit_offset+bit_idx_target}): Pred=0 (PatOriz='{pattern_str}' non trovato storicamente)"

        return final_predicted_bits, analysis_details

class SimpleMixedPatternStrategy(PredictionStrategy):
    def __init__(self,
                 lunghezza_pattern_verticale=3,
                 usa_componente_pos_prec=True,  # Nuovo parametro
                 usa_componente_bit_prec=True): # Nuovo parametro

        self.h = int(lunghezza_pattern_verticale)
        if self.h < 1: self.h = 1

        self.usa_pos_prec = usa_componente_pos_prec
        self.usa_bit_prec = usa_componente_bit_prec
        self.name = "Pattern Misti Semplici"

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        opts = []
        if self.usa_pos_prec: opts.append("PosPrec")
        if self.usa_bit_prec: opts.append("BitPrec")
        active_opts = ", ".join(opts) if opts else "Solo VertStandard"
        return f"Misto: H={self.h}. Componenti attive: {active_opts}."

    def get_config_options(self) -> list[dict]:
        return [
            {'name': 'lunghezza_pattern_verticale', 'type': 'int', 'default': self.h,
             'label': 'Lunghezza Pattern (H) Misto:', 'min': 1, 'max': 10, # O altro max sensato
             'var_name': 'lunghezza_pattern_misto_h_var'},
            {'name': 'usa_componente_pos_prec', 'type': 'bool', 'default': self.usa_pos_prec,
             'label': 'Usa Componente da Posizione Prec.?',
             'var_name': 'misto_usa_pos_prec_var'}, # Dovrai creare queste var tk.BooleanVar
            {'name': 'usa_componente_bit_prec', 'type': 'bool', 'default': self.usa_bit_prec,
             'label': 'Usa Componente da Bit Prec.?',
             'var_name': 'misto_usa_bit_prec_var'}
        ]

    def _get_vertical_prediction(self, reversed_binary_strings, target_abs_bit_index, h_val):
        """Sotto-funzione per analizzare un singolo pattern verticale e predire."""
        if len(reversed_binary_strings) <= h_val:
            return 0, "Dati Insuff." # Predizione default, dettaglio

        pattern_stats = {} # { pattern_tuple: {'matches':0, 'zeros':0, 'ones':0}}
        for i in range(len(reversed_binary_strings) - h_val):
            # Pattern dalle estrazioni più vecchie (E-H ... E-1)
            history_window_bits = [row[target_abs_bit_index] for row in reversed_binary_strings[i+1 : i+1+h_val] if len(row) > target_abs_bit_index]

            if len(history_window_bits) != h_val: continue # Pattern incompleto

            pattern_tuple = tuple(history_window_bits)
            
            # Outcome dall'estrazione E (quella subito dopo la history_window)
            if len(reversed_binary_strings[i]) <= target_abs_bit_index: continue
            following_bit = reversed_binary_strings[i][target_abs_bit_index]

            if pattern_tuple not in pattern_stats:
                pattern_stats[pattern_tuple] = {'matches': 0, 'zeros': 0, 'ones': 0}
            pattern_stats[pattern_tuple]['matches'] += 1
            if following_bit == 1:
                pattern_stats[pattern_tuple]['ones'] += 1
            else:
                pattern_stats[pattern_tuple]['zeros'] += 1

        # Pattern più recente per la predizione
        latest_pattern_bits = [row[target_abs_bit_index] for row in reversed_binary_strings[0 : h_val] if len(row) > target_abs_bit_index]

        if len(latest_pattern_bits) != h_val:
            return 0, "Pat Recente Insuff."

        latest_pattern_tuple = tuple(latest_pattern_bits)
        
        pred = 0 # Default
        detail_suffix = f"Pat='{'' .join(map(str,latest_pattern_tuple))}'"

        if latest_pattern_tuple in pattern_stats:
            stats = pattern_stats[latest_pattern_tuple]
            if stats['matches'] > 0: # Considera una soglia minima di matches se vuoi
                if stats['ones'] > stats['zeros']: pred = 1
                elif stats['zeros'] > stats['ones']: pred = 0
                # In caso di parità, pred rimane 0 (o altra logica)
            detail_suffix += f" (Z:{stats['zeros']}-U:{stats['ones']}-M:{stats['matches']})"
        else:
            detail_suffix += " (Non Trovato)"
        
        return pred, detail_suffix


    def predict(self, data_to_analyze: pd.DataFrame, bit_offset: int, **kwargs) -> tuple[list[int], list[str]]:
        final_predicted_bits = [0] * BITS_PER_NUMERO
        analysis_details = [""] * BITS_PER_NUMERO

        # Ottieni i parametri attuali (potrebbero venire da kwargs per backtest o dalla UI)
        current_h = kwargs.get('lunghezza_pattern_verticale', self.h)
        current_usa_pos_prec = kwargs.get('usa_componente_pos_prec', self.usa_pos_prec)
        current_usa_bit_prec = kwargs.get('usa_componente_bit_prec', self.usa_bit_prec)

        if not isinstance(data_to_analyze, pd.DataFrame) or 'BinarioCompleto' not in data_to_analyze.columns:
            for i in range(BITS_PER_NUMERO): analysis_details[i] = "Errore Dati Input (Misto)"
            return final_predicted_bits, analysis_details

        # Prepara i dati: lista di liste di bit (interi) per ogni estrazione, invertita (più recente prima)
        # Ogni sub-lista interna contiene TUTTI i 35 bit dell'estrazione.
        reversed_full_extraction_bits = []
        for bin_str in data_to_analyze['BinarioCompleto'].tolist():
            if len(bin_str) == NUM_POSIZIONI * BITS_PER_NUMERO: # Solo estrazioni complete
                reversed_full_extraction_bits.append([int(b) for b in bin_str])
        reversed_full_extraction_bits.reverse() # Più recente è all'indice 0

        if len(reversed_full_extraction_bits) <= current_h:
            for i in range(BITS_PER_NUMERO): analysis_details[i] = f"Dati Insuff. H={current_h} (Misto)"
            return final_predicted_bits, analysis_details

        for bit_pos_relative in range(BITS_PER_NUMERO): # 0-6, bit target all'interno del numero
            target_abs_bit_index = bit_offset + bit_pos_relative # Indice assoluto del bit (0-34)
            
            votes_for_0 = 0
            votes_for_1 = 0
            current_detail_parts = []

            # 1. Componente Verticale Standard (CVS)
            pred_cvs, detail_cvs = self._get_vertical_prediction(reversed_full_extraction_bits, target_abs_bit_index, current_h)
            if pred_cvs == 0: votes_for_0 += 1
            else: votes_for_1 += 1
            current_detail_parts.append(f"CVS:{pred_cvs} ({detail_cvs})")

            # 2. Componente Verticale Orizzontalmente Spostata (CVOS) - da Posizione Precedente
            if current_usa_pos_prec and bit_offset >= BITS_PER_NUMERO: # Se non è la prima posizione (P1)
                target_abs_bit_index_pos_prec = target_abs_bit_index - BITS_PER_NUMERO
                pred_cvos, detail_cvos = self._get_vertical_prediction(reversed_full_extraction_bits, target_abs_bit_index_pos_prec, current_h)
                if pred_cvos == 0: votes_for_0 += 1
                else: votes_for_1 += 1
                current_detail_parts.append(f"CVOS:{pred_cvos} ({detail_cvos})")
            
            # 3. Componente Verticale Verticalmente Spostata (CVVS) - da Bit Precedente
            if current_usa_bit_prec and bit_pos_relative > 0:
                target_abs_bit_index_bit_prec = target_abs_bit_index - 1
                pred_cvvs, detail_cvvs = self._get_vertical_prediction(reversed_full_extraction_bits, target_abs_bit_index_bit_prec, current_h)
                if pred_cvvs == 0: votes_for_0 += 1
                else: votes_for_1 += 1
                current_detail_parts.append(f"CVVS:{pred_cvvs} ({detail_cvvs})")

            # Decisione finale basata sui voti
            if votes_for_1 > votes_for_0:
                final_predicted_bits[bit_pos_relative] = 1
            elif votes_for_0 > votes_for_1:
                final_predicted_bits[bit_pos_relative] = 0
            else: # Parità di voti
                final_predicted_bits[bit_pos_relative] = 0 # Default (o usa la predizione CVS)

            analysis_details[bit_pos_relative] = f"BitPR{bit_pos_relative}(Abs{target_abs_bit_index}): Pred={final_predicted_bits[bit_pos_relative]}. Voti[0:{votes_for_0},1:{votes_for_1}]. Dett: {' | '.join(current_detail_parts)}"
            
        return final_predicted_bits, analysis_details

class BestVerticalPatternStrategy(PredictionStrategy):
    def __init__(self, lunghezza_pattern_verticale=DEFAULT_LUNGHEZZA_PATTERN_VERTICALE):
        self.lunghezza_pattern = int(lunghezza_pattern_verticale)
        if self.lunghezza_pattern < 1:
            self.lunghezza_pattern = 1 # Minimo
        self.name = "Migliori Pattern Verticali"
        self.description = f"Analizza pattern verticali di lunghezza H per predire i bit."

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_config_options(self) -> list[dict]:
        return [
            {'name': 'lunghezza_pattern_verticale', 'type': 'int', 'default': self.lunghezza_pattern,
             'label': 'Lunghezza Pattern Verticale (H):', 'min': 1, 'max': 10, 'var_name': 'lunghezza_pattern_var'}
        ]

    def predict(self, data_to_analyze: pd.DataFrame, bit_offset: int, **kwargs) -> tuple[list[int], list[str]]:
        final_predicted_bits = [0] * BITS_PER_NUMERO
        analysis_details = [""] * BITS_PER_NUMERO

        # Override lunghezza_pattern se passato via kwargs (utile per backtest con parametro fisso)
        current_lunghezza_pattern = kwargs.get('lunghezza_pattern_verticale', self.lunghezza_pattern)

        if not isinstance(data_to_analyze, pd.DataFrame) or 'BinarioCompleto' not in data_to_analyze.columns:
            for i in range(BITS_PER_NUMERO):
                analysis_details[i] = f"Bit Pos {i}(+{bit_offset}): Errore Dati Input"
            return final_predicted_bits, analysis_details

        binary_strings = data_to_analyze['BinarioCompleto'].tolist()
        reversed_data = list(reversed(binary_strings))
        num_rows = len(reversed_data)

        if num_rows <= current_lunghezza_pattern:
            for i in range(BITS_PER_NUMERO):
                analysis_details[i] = f"Bit Pos {i}(+{bit_offset}): Predetto=0 (Dati insuff. H={current_lunghezza_pattern}, N={num_rows})"
            return final_predicted_bits, analysis_details

        for bit_pos_relative in range(BITS_PER_NUMERO):
            target_bit_index = bit_offset + bit_pos_relative
            pattern_stats_for_col = {}
            for i in range(num_rows - current_lunghezza_pattern):
                history_window_rows = reversed_data[i : i + current_lunghezza_pattern]
                prediction_row = reversed_data[i + current_lunghezza_pattern]
                try:
                    if len(prediction_row) <= target_bit_index or any(len(hr) <= target_bit_index for hr in history_window_rows):
                        continue
                    vertical_pattern_tuple = tuple(int(row[target_bit_index]) for row in history_window_rows)
                    following_bit = int(prediction_row[target_bit_index])

                    if vertical_pattern_tuple not in pattern_stats_for_col:
                        pattern_stats_for_col[vertical_pattern_tuple] = {'matches': 0, 'zeros': 0, 'ones': 0}
                    pattern_stats_for_col[vertical_pattern_tuple]['matches'] += 1
                    if following_bit == 1:
                        pattern_stats_for_col[vertical_pattern_tuple]['ones'] += 1
                    else:
                        pattern_stats_for_col[vertical_pattern_tuple]['zeros'] += 1
                except (ValueError, IndexError):
                    continue

            best_pattern = None
            max_imbalance = -1
            best_pattern_stats = {}

            if not pattern_stats_for_col:
                final_predicted_bits[bit_pos_relative] = 0
                analysis_details[bit_pos_relative] = f"Bit Pos {bit_pos_relative}(+{bit_offset}): Predetto=0 (Nessun pattern H={current_lunghezza_pattern})"
                continue

            for pattern, stats in pattern_stats_for_col.items():
                imbalance = abs(stats['ones'] - stats['zeros'])
                matches = stats['matches']
                if matches > 0:
                    if imbalance > max_imbalance:
                        max_imbalance = imbalance
                        best_pattern = pattern
                        best_pattern_stats = stats
                    elif imbalance == max_imbalance and matches > best_pattern_stats.get('matches', 0):
                        best_pattern = pattern
                        best_pattern_stats = stats

            if best_pattern is not None and best_pattern_stats:
                predicted_bit = 1 if best_pattern_stats['ones'] > best_pattern_stats['zeros'] else 0
                final_predicted_bits[bit_pos_relative] = predicted_bit
                pattern_str = "".join(map(str, best_pattern))
                analysis_details[bit_pos_relative] = (
                    f"Bit Pos {bit_pos_relative}(+{bit_offset}): Predetto={predicted_bit} "
                    f"(BestPat='{pattern_str}': Z={best_pattern_stats['zeros']}, U={best_pattern_stats['ones']}, M={best_pattern_stats['matches']})"
                )
            else:
                final_predicted_bits[bit_pos_relative] = 0
                analysis_details[bit_pos_relative] = f"Bit Pos {bit_pos_relative}(+{bit_offset}): Predetto=0 (Nessun pattern predittivo H={current_lunghezza_pattern})"
        return final_predicted_bits, analysis_details

# Registro delle strategie disponibili
AVAILABLE_STRATEGIES = {
    "vertical_pattern": BestVerticalPatternStrategy,
    "horizontal_intra_pos_pattern": HorizontalIntraPositionPatternStrategy,
    "simple_mixed_pattern": SimpleMixedPatternStrategy,
    # Aggiungi qui altre classi di strategie
}
DEFAULT_STRATEGY_KEY = "vertical_pattern"

# ==============================================================================
# CLASSE PRINCIPALE DELL'APPLICAZIONE
# ==============================================================================
class SequenzaSpiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Binary - Il Lotto di Max - Strategie")

        # --- INSERISCI QUI LA MODIFICA PER LA GEOMETRIA ---
        # Esempio: imposta la finestra a 850 pixel di larghezza e 680 pixel di altezza
        # Sostituisci 850 e 680 con i valori che desideri.
        larghezza_desiderata = 850
        altezza_desiderata = 800 # Modifica questo valore per abbassare/alzare l'altezza
        self.root.geometry(f"{larghezza_desiderata}x{altezza_desiderata}")
        # ----------------------------------------------------

        self.setup_variables()
        self.create_ui()
        self.queue = queue.Queue()
        self.check_queue()
        self._on_strategy_select() # Per inizializzare la UI dei parametri strategia
    def setup_variables(self):
        self.historical_data = None
        self.ruota_var = tk.StringVar(value="BA")
        self.selected_ruota_code = "BA"
        self.loaded_info = "Nessun dato caricato."
        self.all_predictions = {}
        self.window_size_var = tk.StringVar(value="60")
        self.prediction_labels = []

        # Variabili per le strategie
        self.strategy_var = tk.StringVar(value=DEFAULT_STRATEGY_KEY)
        self.current_strategy_instance = None
        self.strategy_config_widgets = {} # Per i widget di configurazione dinamici

        # Variabile specifica per la lunghezza pattern della strategia di default
        self.lunghezza_pattern_var = tk.IntVar(value=DEFAULT_LUNGHEZZA_PATTERN_VERTICALE)
        self.lunghezza_pattern_orizz_var = tk.IntVar(value=3) # O un altro default
        self.lunghezza_pattern_misto_h_var = tk.IntVar(value=3)
        self.misto_usa_pos_prec_var = tk.BooleanVar(value=True)
        self.misto_usa_bit_prec_var = tk.BooleanVar(value=True)


    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True); main_frame.columnconfigure(0, weight=1)
        current_row = 0

        # --- 1. Caricamento Dati ---
        load_data_frame = ttk.LabelFrame(main_frame, text="1. Caricamento Dati Ruota", padding="10"); load_data_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); load_data_frame.columnconfigure(1, weight=1); current_row += 1
        ruota_subframe = ttk.Frame(load_data_frame); ruota_subframe.grid(row=0, column=0, columnspan=2, sticky="ew", pady=2); ttk.Label(ruota_subframe, text="Seleziona Ruota:", anchor="w").pack(side=tk.LEFT, padx=5); ruota_menu = ttk.OptionMenu(ruota_subframe, self.ruota_var, "BA", *URL_RUOTE.keys(), command=self._update_selected_ruota); ruota_menu.pack(side=tk.LEFT, padx=5); self.ruota_label = ttk.Label(ruota_subframe, text="Ruota: Bari", anchor="w"); self.ruota_label.pack(side=tk.LEFT, padx=5); ttk.Button(ruota_subframe, text="Carica Dati Ruota", width=18, command=self.carica_dati_ruota).pack(side=tk.LEFT, padx=10)
        self.loaded_file_label = ttk.Label(load_data_frame, text=self.loaded_info, anchor="w"); self.loaded_file_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(5,2))

        # --- Anteprima Dati ---
        preview_frame = ttk.LabelFrame(main_frame, text="Anteprima Dati Binari Caricati (Ultime 5)", padding="10"); preview_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); preview_frame.columnconfigure(0, weight=1); current_row += 1
        self.binary_preview = scrolledtext.ScrolledText(preview_frame, height=5, width=60, wrap=tk.WORD, font=('Courier New', 9)); self.binary_preview.grid(row=0, column=0, sticky="ew"); self.binary_preview.insert('1.0', "Nessun dato caricato."); self.binary_preview.config(state=tk.DISABLED)

        # --- 2. Selezione e Configurazione Strategia ---
        strategy_frame = ttk.LabelFrame(main_frame, text="2. Selezione e Configurazione Strategia", padding="10")
        strategy_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5)
        current_row += 1

        ttk.Label(strategy_frame, text="Strategia:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        strategy_names = [AVAILABLE_STRATEGIES[key]().get_name() for key in AVAILABLE_STRATEGIES.keys()]
        self.strategy_combobox = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, values=list(AVAILABLE_STRATEGIES.keys()), state="readonly", width=30)
        # Mappiamo i nomi leggibili ai codici chiave per il Combobox, se preferito
        # Per ora usiamo le chiavi, e aggiorniamo un'etichetta con il nome completo se necessario
        self.strategy_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.strategy_combobox.bind("<<ComboboxSelected>>", self._on_strategy_select)
        self.strategy_desc_label = ttk.Label(strategy_frame, text="Descrizione strategia...")
        self.strategy_desc_label.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.strategy_config_frame = ttk.Frame(strategy_frame, padding="5") # Frame per i parametri dinamici
        self.strategy_config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5,0))


        # --- 4. Backtesting su Periodo Storico --- (Rinumerato)
        backtest_frame = ttk.LabelFrame(main_frame, text="4. Verifica Multi-Posizione su Periodo Storico (Backtest)", padding="10"); backtest_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=10); current_row += 1
        controls_frame = ttk.Frame(backtest_frame); controls_frame.pack(pady=5, fill=tk.X)
        ttk.Label(controls_frame, text="Data Inizio:").pack(side=tk.LEFT, padx=(5,2))
        self.start_date_entry = DateEntry(controls_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy/mm/dd'); self.start_date_entry.pack(side=tk.LEFT, padx=(0,10)); one_month_ago = datetime.now() - timedelta(days=30); self.start_date_entry.set_date(one_month_ago)
        ttk.Label(controls_frame, text="Data Fine:").pack(side=tk.LEFT, padx=(10,2))
        self.end_date_entry = DateEntry(controls_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy/mm/dd'); self.end_date_entry.pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(controls_frame, text="Colpi Verifica:").pack(side=tk.LEFT, padx=(10, 2))
        self.colpi_var = tk.IntVar(value=DEFAULT_COLPI_VERIFICA)
        self.colpi_spinbox = ttk.Spinbox(controls_frame, from_=1, to=36, width=5, textvariable=self.colpi_var); self.colpi_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(controls_frame, text="Finestra Storica (0=Tutte):").pack(side=tk.LEFT, padx=(10, 2))
        self.window_size_entry = ttk.Entry(controls_frame, width=6, textvariable=self.window_size_var); self.window_size_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(backtest_frame, text=">> ESEGUI BACKTEST MULTI-POSIZIONE <<", style='Accent.TButton', command=self.start_backtest).pack(pady=5, ipady=4)

        # --- 3. Predizione --- (Rinumerato)
        analysis_frame = ttk.LabelFrame(main_frame, text="3. Predizione Prossima Estrazione (Tutte le Posizioni)", padding="10"); analysis_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=10); current_row += 1
        ttk.Button(analysis_frame, text=">> CALCOLA PREDIZIONE PER TUTTE LE POSIZIONI <<", style='Accent.TButton', command=self.start_prediction_analysis).pack(padx=15, ipady=4)

        # --- 5. Risultato Predizione --- (Rinumerato)
        pred_display_frame = ttk.LabelFrame(main_frame, text="5. Ultima Predizione Calcolata", padding="10")
        pred_display_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); pred_display_frame.columnconfigure(1, weight=1); current_row += 1
        self.prediction_labels = []
        for i in range(NUM_POSIZIONI):
            ttk.Label(pred_display_frame, text=f"P{i+1}:").grid(row=i, column=0, padx=(5,2), pady=1, sticky="w")
            bits_label = ttk.Label(pred_display_frame, text="[ _ _ _ _ _ _ _ ]", font=('Courier New', 11, 'bold'), anchor="w")
            bits_label.grid(row=i, column=1, padx=(0,5), pady=1, sticky="ew")
            self.prediction_labels.append(bits_label)
        button_pred_frame = ttk.Frame(pred_display_frame); button_pred_frame.grid(row=NUM_POSIZIONI, column=0, columnspan=2, pady=(5, 0), sticky='ew')
        ttk.Button(button_pred_frame, text="Svuota Predizioni", width=15, command=self.svuota_predizioni).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_pred_frame, text="Mostra Decimali", width=15, command=self.mostra_decimali_predizioni).pack(side=tk.LEFT, padx=5)

        # --- Progress bar ---
        self.progress = ttk.Progressbar(main_frame, mode='determinate'); self.progress.grid(row=current_row, column=0, sticky="ew", padx=5, pady=(5, 0)); current_row += 1

        # --- Log Analisi / Risultati Backtest ---
        results_frame = ttk.LabelFrame(main_frame, text="Log / Risultati Predizioni e Backtest", padding="10"); results_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5); main_frame.rowconfigure(current_row, weight=1); results_frame.columnconfigure(0, weight=1); results_frame.rowconfigure(1, weight=1)
        button_frame = ttk.Frame(results_frame); button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5)); ttk.Button(button_frame, text="Copia Log", width=12, command=self.copy_results).pack(side=tk.LEFT, padx=5); ttk.Button(button_frame, text="Cancella Log", width=12, command=self.clear_results).pack(side=tk.LEFT, padx=5)
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=70, wrap=tk.WORD, font=('Consolas', 9)); self.results_text.grid(row=1, column=0, sticky="nsew")
        try: style = ttk.Style(); style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), foreground='navy')
        except tk.TclError: pass
        self._update_selected_ruota(self.ruota_var.get())
    def _on_strategy_select(self, event=None):
        """Chiamato quando una strategia viene selezionata dal Combobox."""
        selected_key = self.strategy_var.get()
        StrategyClass = AVAILABLE_STRATEGIES.get(selected_key)

        if not StrategyClass:
            # Gestione errore se la chiave non corrisponde a nessuna strategia
            # (dovrebbe essere raro se il combobox è popolato correttamente)
            if hasattr(self, 'strategy_desc_label') and self.strategy_desc_label.winfo_exists():
                 self.strategy_desc_label.config(text="Strategia non valida o non trovata.")
            # Potresti anche mostrare un messagebox, ma attenzione a non entrare in loop di errori all'avvio
            # messagebox.showerror("Errore Interno", f"Chiave strategia '{selected_key}' non riconosciuta.", parent=self.root)
            return

        # Crea un'istanza temporanea per ottenere info e opzioni di configurazione
        # I parametri effettivi verranno presi dalla UI al momento dell'esecuzione
        try:
            # Il costruttore della strategia potrebbe richiedere parametri se non hanno default
            # Per ora, assumiamo costruttori senza argomenti o con default per questa istanza temporanea
            temp_instance_for_info = StrategyClass()
        except Exception as e:
            error_msg = f"Errore durante l'istanziazione temporanea di {selected_key} per UI: {e}"
            print(error_msg) # Log per lo sviluppatore
            if hasattr(self, 'strategy_desc_label') and self.strategy_desc_label.winfo_exists():
                self.strategy_desc_label.config(text="Errore caricamento config. strategia.")
            # messagebox.showerror("Errore Configurazione Strategia", error_msg, parent=self.root)
            return

        if hasattr(self, 'strategy_desc_label') and self.strategy_desc_label.winfo_exists():
            self.strategy_desc_label.config(text=temp_instance_for_info.get_description())

        # Pulisci i vecchi widget di configurazione
        if hasattr(self, 'strategy_config_frame'): # Assicurati che il frame esista
            for widget in self.strategy_config_frame.winfo_children():
                widget.destroy()
        self.strategy_config_widgets.clear()

        # Crea nuovi widget di configurazione per la strategia selezionata
        config_options = temp_instance_for_info.get_config_options()
        current_row = 0
        if hasattr(self, 'strategy_config_frame'): # Assicurati che il frame esista
            if not config_options:
                 ttk.Label(self.strategy_config_frame, text="Nessuna opzione configurabile per questa strategia.").grid(row=0, column=0, columnspan=2, padx=5, pady=2, sticky="w")
            else:
                for option in config_options:
                    ttk.Label(self.strategy_config_frame, text=option['label']).grid(row=current_row, column=0, padx=5, pady=2, sticky="w")
                    
                    var_name = option.get('var_name')
                    default_value = option.get('default')

                    if option['type'] == 'int':
                        # Prova a ottenere la variabile tk esistente, altrimenti creane una nuova
                        var_to_use = getattr(self, var_name, tk.IntVar())
                        if not hasattr(self, var_name) or not isinstance(var_to_use, tk.IntVar):
                            var_to_use = tk.IntVar(value=default_value)
                            setattr(self, var_name, var_to_use) # Assicurati che sia memorizzata nell'istanza
                        else: # Se esiste già, imposta il suo valore al default specificato DALLA STRATEGIA
                              # Questo è utile se cambi strategia e poi torni indietro
                            var_to_use.set(default_value)


                        spin = ttk.Spinbox(self.strategy_config_frame, from_=option.get('min', 0), to=option.get('max', 100),
                                           textvariable=var_to_use, width=8)
                        spin.grid(row=current_row, column=1, padx=5, pady=2, sticky="ew")
                        self.strategy_config_widgets[option['name']] = var_to_use

                    elif option['type'] == 'bool': # <<<--- NUOVA SEZIONE PER I BOOLEANI
                        # Prova a ottenere la variabile tk esistente, altrimenti creane una nuova
                        var_to_use = getattr(self, var_name, tk.BooleanVar())
                        if not hasattr(self, var_name) or not isinstance(var_to_use, tk.BooleanVar):
                            var_to_use = tk.BooleanVar(value=default_value)
                            setattr(self, var_name, var_to_use) # Assicurati che sia memorizzata
                        else:
                            var_to_use.set(default_value)

                        # Usiamo text="" per il Checkbutton perché l'etichetta è già stata creata
                        check = ttk.Checkbutton(self.strategy_config_frame, text="", variable=var_to_use)
                        check.grid(row=current_row, column=1, padx=5, pady=2, sticky="w") # sticky "w" per allinearlo a sinistra
                        self.strategy_config_widgets[option['name']] = var_to_use
                    
                    # Aggiungere altri tipi di widget (es. 'string', 'float') se necessario
                    # elif option['type'] == 'string':
                    #     # ... logica per tk.StringVar e ttk.Entry ...

                    current_row += 1
        else:
            print("ERRORE: self.strategy_config_frame non trovato durante _on_strategy_select.")

    def _get_current_strategy_instance(self):
        """Crea e restituisce un'istanza della strategia selezionata con i parametri correnti."""
        selected_key = self.strategy_var.get()
        StrategyClass = AVAILABLE_STRATEGIES.get(selected_key)
        if not StrategyClass:
            self.queue.put(("error", f"Strategia '{selected_key}' non trovata."))
            return None

        # Raccogli i parametri dalla UI
        strategy_params = {}
        temp_instance_for_options = StrategyClass() # Per ottenere i nomi delle opzioni
        for option_spec in temp_instance_for_options.get_config_options():
            param_name = option_spec['name']
            if param_name in self.strategy_config_widgets:
                try:
                    strategy_params[param_name] = self.strategy_config_widgets[param_name].get()
                except tk.TclError: # Potrebbe succedere se il widget non è ancora pronto o il valore non è valido
                    strategy_params[param_name] = option_spec['default']
                    print(f"Warning: Impossibile ottenere il valore per {param_name}, usando default {option_spec['default']}")
            else: # Se il widget non esiste per qualche motivo, usa il default
                 strategy_params[param_name] = option_spec['default']


        try:
            return StrategyClass(**strategy_params)
        except Exception as e:
            self.queue.put(("error", f"Errore nell'istanziare la strategia {selected_key}: {e}"))
            return None

    # --- METODI DI CARICAMENTO DATI --- (Invariato)
    def _update_selected_ruota(self, value): self.selected_ruota_code = value; self.ruota_label.config(text=f"Ruota: {self.get_ruota_name(value)}")
    def get_ruota_name(self, code): ruote_nomi = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova','MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma','TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}; return ruote_nomi.get(code, code)
    def carica_dati_ruota(self):
        self.all_predictions = {}; self.svuota_predizioni()
        ruota_code = self.ruota_var.get(); url = URL_RUOTE.get(ruota_code);
        if not url: messagebox.showerror("Errore Configurazione", f"URL non configurato per {ruota_code}."); return
        self.loaded_file_label.config(text=f"Caricamento {self.get_ruota_name(ruota_code)}..."); self.root.update_idletasks(); self.clear_results(); self.historical_data = None
        try:
            response = requests.get(url, timeout=15); response.raise_for_status(); col_names = ['DataStr', 'RuotaCode', 'N1', 'N2', 'N3', 'N4', 'N5']
            df = pd.read_csv(io.StringIO(response.text), sep='\t', header=None, names=col_names, on_bad_lines='skip', low_memory=False)
            df['Data'] = pd.to_datetime(df['DataStr'], format='%Y/%m/%d', errors='coerce'); df.dropna(subset=['Data'], inplace=True)
            num_cols = ['N1', 'N2', 'N3', 'N4', 'N5'];
            for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=num_cols, inplace=True);
            for col in num_cols: df[col] = df[col].astype(int)
            df = df[df[num_cols].apply(lambda x: (x >= 1) & (x <= 90)).all(axis=1)]
            if df.empty: raise ValueError("Nessun dato valido dopo pulizia.")
            df['BinarioCompleto'] = df[num_cols].apply(lambda row: "".join(f"{int(num):0{BITS_PER_NUMERO}b}" for num in row), axis=1)
            for i in range(NUM_POSIZIONI): start_bit = i * BITS_PER_NUMERO; end_bit = start_bit + BITS_PER_NUMERO; df[f'BinarioPos{i+1}'] = df['BinarioCompleto'].str[start_bit:end_bit]
            df.set_index('Data', inplace=True); df.sort_index(inplace=True)
            self.historical_data = df[['BinarioCompleto'] + [f'BinarioPos{i+1}' for i in range(NUM_POSIZIONI)]].copy()
            self.loaded_info = f"Dati Caricati: Ruota {self.get_ruota_name(ruota_code)} ({len(self.historical_data)} estrazioni)"; self.loaded_file_label.config(text=self.loaded_info)
            self._update_preview(); messagebox.showinfo("Caricato", f"{len(self.historical_data)} estrazioni valide caricate per {self.get_ruota_name(ruota_code)}.")
            messagebox.showinfo("Caricato", f"{len(self.historical_data)} estrazioni valide caricate per {self.get_ruota_name(ruota_code)}.", parent=self.root)
        except requests.exceptions.Timeout: messagebox.showerror("Errore Rete", f"Timeout GitHub ({url})."); self.loaded_file_label.config(text="Errore Timeout")
        except requests.exceptions.HTTPError as e: messagebox.showerror("Errore HTTP", f"Errore {e.response.status_code} GitHub ({url})."); self.loaded_file_label.config(text=f"Errore HTTP {e.response.status_code}")
        except requests.exceptions.RequestException as e: messagebox.showerror("Errore Rete", f"Impossibile scaricare: {e}"); self.loaded_file_label.config(text="Errore Rete")
        except ValueError as e: messagebox.showerror("Errore Dati", f"{e}"); self.loaded_file_label.config(text="Errore dati")
        except Exception as e: messagebox.showerror("Errore", f"Errore caricamento: {e}\n{traceback.format_exc()}"); self.loaded_file_label.config(text="Errore")
    def _update_preview(self):
        self.binary_preview.config(state=tk.NORMAL); self.binary_preview.delete('1.0', tk.END)
        if self.historical_data is not None and not self.historical_data.empty:
            preview_df = self.historical_data.tail(5); preview_text = "\n".join(preview_df['BinarioCompleto'].tolist())
            if len(self.historical_data) > 5: preview_text = "...\n" + preview_text
            self.binary_preview.insert('1.0', preview_text)
        else: self.binary_preview.insert('1.0', "Nessun dato.")
        self.binary_preview.config(state=tk.DISABLED)

    # --- LOGICA DI ANALISI MULTI-POSIZIONE (con Strategie) ---
    def start_prediction_analysis(self): # Rinominato da start_best_vertical_pattern_analysis
        if self.historical_data is None or self.historical_data.empty: messagebox.showwarning("Dati Mancanti", "Caricare i dati."); return

        strategy_instance = self._get_current_strategy_instance()
        if not strategy_instance: return

        try:
            window_size_str = self.window_size_var.get(); historical_window_size = int(window_size_str)
            if historical_window_size < 0: raise ValueError("La finestra storica non può essere negativa.")
        except ValueError: messagebox.showerror("Errore Input", f"Valore non valido per Finestra Storica: '{window_size_str}'. Inserire un numero intero >= 0."); return

        self.clear_results(); window_info_title = f"(Finestra: {historical_window_size})" if historical_window_size > 0 else "(Finestra: Tutte)"
        self.results_text.insert(tk.END, f"Avvio analisi con strategia '{strategy_instance.get_name()}' {window_info_title} per tutte le {NUM_POSIZIONI} posizioni...\n");
        self.progress['value'] = 0; self.svuota_predizioni();
        Thread(target=self.run_prediction_analysis_thread, args=(self.historical_data.copy(), historical_window_size, strategy_instance), daemon=True).start()

    def run_prediction_analysis_thread(self, data_df, historical_window_size_param, strategy_instance): # Modificato per strategia
        try:
            predictions = {}; analysis_reports = []
            if historical_window_size_param > 0 and len(data_df) > historical_window_size_param: recent_data_df = data_df.tail(historical_window_size_param).copy(); window_info = f"(Ultime {len(recent_data_df)} estrazioni, Finestra={historical_window_size_param})";
            else: recent_data_df = data_df.copy(); window_info = f"(Tutte le {len(recent_data_df)} estrazioni)";
            total_steps = NUM_POSIZIONI; completed_steps = 0

            # Raccogli parametri strategia una volta, se necessario per il report
            strategy_config_details = []
            for opt in strategy_instance.get_config_options():
                if opt['name'] in self.strategy_config_widgets:
                     strategy_config_details.append(f"{opt['label'].replace(':','')} {self.strategy_config_widgets[opt['name']].get()}")

            strategy_params_report = ", ".join(strategy_config_details)
            if strategy_params_report: strategy_params_report = f" ({strategy_params_report})"


            for i in range(NUM_POSIZIONI):
                bit_offset = i * BITS_PER_NUMERO
                # Passa i parametri della strategia al metodo predict se necessario (tramite kwargs o se sono già nell'istanza)
                predicted_bits, analysis_details = strategy_instance.predict(recent_data_df, bit_offset)
                predictions[bit_offset] = predicted_bits
                binary_string = "".join(map(str, predicted_bits));
                try: decimal_value = int(binary_string, 2)
                except ValueError: decimal_value = "Errore"
                pos_report = f"\n--- Predizione Posizione {i+1} (Offset {bit_offset}) ---\n" + "\n".join(analysis_details) + "\n" + f"Numero Binario Predetto (Pos {i+1}): {binary_string}\n" + f"Valore Decimale (Pos {i+1}): {decimal_value}\n"; analysis_reports.append(pos_report)
                completed_steps += 1; progress_val = (completed_steps / total_steps) * 100; self.root.after(0, lambda p=progress_val: self.progress.config(value=p))

            final_report = f"\n=== PREDIZIONE COMPLETA (Strategia: {strategy_instance.get_name()}{strategy_params_report}) ===\n"
            final_report += f"(Basata su {len(recent_data_df)} estrazioni storiche {window_info})\n" + "".join(analysis_reports) + "--- Riepilogo Predizioni Decimali ---\n"
            summary = [];
            for i in range(NUM_POSIZIONI):
                offset = i * BITS_PER_NUMERO; bits = predictions.get(offset, [None]*BITS_PER_NUMERO); bin_str = "".join(map(str, bits));
                try: dec_val = int(bin_str, 2)
                except: dec_val = "?";
                summary.append(f"Pos {i+1}: {dec_val} ({bin_str})")
            final_report += ", ".join(summary) + "\n" + "=========================================\n"

            def update_ui():
                self.all_predictions = predictions.copy()
                self._update_prediction_display()
                self.results_text.insert(tk.END, final_report); self.results_text.see(tk.END); self.progress['value'] = 100
            self.root.after(0, update_ui)
        except Exception as e: error_msg = f"Errore analisi Multi-Posizione: {e}\n{traceback.format_exc()}"; print(f"ERRORE MULTI-POS: {error_msg}"); self.queue.put(("error", error_msg)); self.root.after(0, lambda: self.progress.config(value=0))

    # --- METODI PER BACKTESTING MULTI-POSIZIONE (con Strategie) ---
    def start_backtest(self):
        if self.historical_data is None or self.historical_data.empty: messagebox.showwarning("Dati Mancanti", "Caricare i dati."); return

        strategy_instance = self._get_current_strategy_instance()
        if not strategy_instance: return

        try:
            start_date = self.start_date_entry.get_date(); end_date = self.end_date_entry.get_date(); start_dt = pd.to_datetime(start_date); end_dt = pd.to_datetime(end_date)
            colpi_verifica = self.colpi_var.get(); window_size_str = self.window_size_var.get(); historical_window_size = int(window_size_str)
            if historical_window_size < 0: raise ValueError("La finestra storica non può essere negativa.")
            if not (1 <= colpi_verifica <= 36): raise ValueError("Numero di colpi di verifica non valido (1-36).")
            if start_dt >= end_dt: raise ValueError("Data inizio deve essere precedente alla data fine.")
            backtest_prediction_dates = self.historical_data[(self.historical_data.index >= start_dt) & (self.historical_data.index <= end_dt)].index
            if backtest_prediction_dates.empty: messagebox.showinfo("Backtest", "Nessuna estrazione nel periodo per fare predizioni."); return
        except ValueError as e: messagebox.showerror("Errore Input", str(e)); return
        except Exception as e: messagebox.showerror("Errore Avvio Backtest", f"Errore imprevisto nella validazione: {e}\n{traceback.format_exc()}"); return

        self.clear_results()
        window_info_title = f"(Finestra: {historical_window_size})" if historical_window_size > 0 else "(Finestra: Tutte)"

        # Raccogli parametri strategia una volta per il report
        strategy_config_details = []
        for opt in strategy_instance.get_config_options():
            if opt['name'] in self.strategy_config_widgets:
                 strategy_config_details.append(f"{opt['label'].replace(':','')} {self.strategy_config_widgets[opt['name']].get()}")
        strategy_params_report = ", ".join(strategy_config_details)
        if strategy_params_report: strategy_params_report = f" ({strategy_params_report})"


        self.results_text.insert(tk.END, f"Avvio backtest Multi-Posizione con strategia '{strategy_instance.get_name()}{strategy_params_report}' {window_info_title} (Verifica a {colpi_verifica} colpi)\n")
        self.results_text.insert(tk.END, f"Periodo predizioni: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}...\n")
        self.results_text.insert(tk.END, "Verifica: OK se ALMENO UNO dei 5 numeri predetti esce in UNA delle 5 posizioni reali entro i colpi.\n\n")
        self.progress['value'] = 0; self.svuota_predizioni();
        Thread(target=self.run_backtest_thread, args=(backtest_prediction_dates, colpi_verifica, historical_window_size, strategy_instance), daemon=True).start()

    def run_backtest_thread(self, prediction_dates, colpi_verifica, historical_window_size_param, strategy_instance): # Modificato per strategia
        results = []; total_dates = len(prediction_dates); dates_processed = 0
        try:
            all_dates = self.historical_data.index; actual_columns = ['BinarioCompleto'] + [f'BinarioPos{i+1}' for i in range(NUM_POSIZIONI)]; all_actuals_df = self.historical_data[actual_columns]

            # Prendi i parametri della strategia una volta per il report se necessario (es. per il campo 'detail' di ogni riga)
            # Potresti voler passare i parametri individuali alla funzione predict della strategia se
            # la strategia li accetta via kwargs e non solo dal costruttore.
            # Per ora, l'istanza della strategia è già configurata.
            strategy_param_val_for_detail = ""
            if strategy_instance.get_name() == "Migliori Pattern Verticali": # Esempio specifico
                for opt in strategy_instance.get_config_options():
                    if opt['name'] == 'lunghezza_pattern_verticale' and opt['name'] in self.strategy_config_widgets:
                        strategy_param_val_for_detail = f"H={self.strategy_config_widgets[opt['name']].get()} "
                        break

            for target_date in prediction_dates:
                data_for_prediction_full = self.historical_data[all_dates < target_date]
                if historical_window_size_param > 0 and len(data_for_prediction_full) > historical_window_size_param: data_for_prediction_limited = data_for_prediction_full.tail(historical_window_size_param).copy()
                else: data_for_prediction_limited = data_for_prediction_full.copy()

                predicted_numbers_bin = ["0"*BITS_PER_NUMERO]*NUM_POSIZIONI; predicted_numbers_dec = ["-"]*NUM_POSIZIONI
                pred_detail_prefix = strategy_param_val_for_detail # Aggiungi parametro strategia al dettaglio
                pred_detail_suffix = ""
                window_len_used = len(data_for_prediction_limited)

                # Determina se la strategia può operare o se i dati sono insufficienti
                # Questo potrebbe essere un metodo della strategia stessa: strategy_instance.can_predict(data_for_prediction_limited)
                # Per ora, usiamo una logica simile a prima per BestVerticalPatternStrategy
                can_predict_flag = True
                if strategy_instance.get_name() == "Migliori Pattern Verticali":
                    # La strategia BestVerticalPatternStrategy ha il suo parametro lunghezza_pattern
                    # che viene usato internamente per determinare dati insuff.
                    # Per ora, la gestione dati insuff. è dentro strategy_instance.predict()
                    # Il dettaglio "(D.Insuff...)" verrà dal report di analysis_details se la predizione fallisce
                    # Potremmo volerlo più esplicito qui.
                    pass # La logica di insufficienza è in predict

                if not can_predict_flag: # Se avessimo un flag esplicito
                     if historical_window_size_param > 0 and len(data_for_prediction_full) > historical_window_size_param: pred_detail_suffix = f"(D.Insuff. Win={window_len_used})"
                     else: pred_detail_suffix = f"(D.Insuff. Tot={window_len_used})"
                else:
                    temp_predictions = {}
                    for i in range(NUM_POSIZIONI):
                        bit_offset = i*BITS_PER_NUMERO
                        predicted_bits, _analysis_details_backtest = strategy_instance.predict(data_for_prediction_limited, bit_offset)
                        # _analysis_details_backtest non viene usato nel report tabellare del backtest, ma potrebbe
                        temp_predictions[bit_offset] = predicted_bits

                    for i in range(NUM_POSIZIONI):
                        offset = i*BITS_PER_NUMERO; bits = temp_predictions.get(offset,[0]*BITS_PER_NUMERO); bin_str = "".join(map(str,bits)); predicted_numbers_bin[i] = bin_str;
                        try: predicted_numbers_dec[i] = str(int(bin_str, 2))
                        except: predicted_numbers_dec[i] = "?"
                    if not pred_detail_suffix: pred_detail_suffix = f"(W={window_len_used})"

                pred_detail = pred_detail_prefix + pred_detail_suffix.strip()

                found = False; hit_colpo = 0; actual_numbers_at_hit = ["-"]*5; hit_details = ""
                try:
                    start_check_loc = all_dates.get_loc(target_date)
                    if start_check_loc + 1 >= len(all_dates): outcome = ('SKIP', 0); hit_details="No estraz. succ."; results.append({'date_pred': target_date.strftime('%Y/%m/%d'), 'predicted_dec': predicted_numbers_dec, 'detail': pred_detail, 'outcome': outcome, 'hit_details': hit_details}); dates_processed += 1; continue
                except KeyError: continue # Data non trovata, salta

                for j in range(colpi_verifica):
                    check_loc = start_check_loc + 1 + j;
                    if check_loc >= len(all_dates): break
                    try: actual_row = all_actuals_df.iloc[check_loc]; actual_numbers_bin_at_j = [actual_row[f'BinarioPos{k+1}'] for k in range(NUM_POSIZIONI)]
                    except IndexError: break
                    match_found_this_colpo = False
                    for p_idx, predicted_bin in enumerate(predicted_numbers_bin):
                        if predicted_numbers_dec[p_idx] == '-': continue
                        if predicted_bin in actual_numbers_bin_at_j: found = True; hit_colpo = j + 1; actual_numbers_at_hit = [str(int(b, 2)) if b.isdigit() else '?' for b in actual_numbers_bin_at_j]; a_idx = actual_numbers_bin_at_j.index(predicted_bin); hit_details = f"P{p_idx+1}({predicted_numbers_dec[p_idx]}) = A{a_idx+1}({actual_numbers_at_hit[a_idx]}) @C{hit_colpo}"; match_found_this_colpo = True; break
                    if match_found_this_colpo: break
                if found: outcome = ('OK', hit_colpo)
                else: outcome = ('NO', colpi_verifica)
                results.append({'date_pred': target_date.strftime('%Y/%m/%d'), 'predicted_dec': predicted_numbers_dec, 'detail': pred_detail, 'outcome': outcome, 'hit_details': hit_details if found else f"Nessun match in {colpi_verifica}c"})
                dates_processed += 1; progress_val = math.ceil((dates_processed / total_dates) * 100)
                if dates_processed % 5 == 0 or dates_processed == total_dates: self.root.after(0, lambda p=progress_val: self.progress.config(value=p))

            if not results: final_report = "\nNessuna predizione valida generata o verificata nel periodo."; self.queue.put(("info", final_report)); self.root.after(0, lambda: self.progress.config(value=100)); return
            valid_results=[r for r in results if r['outcome'][0]!='SKIP']; total_valid_preds=len(valid_results); correct_count=sum(1 for r in valid_results if r['outcome'][0]=='OK'); accuracy=(correct_count/total_valid_preds)*100 if total_valid_preds>0 else 0; avg_colpo=sum(r['outcome'][1] for r in valid_results if r['outcome'][0]=='OK')/correct_count if correct_count > 0 else 0

            # Raccogli parametri strategia per il report finale
            strategy_config_details_final = []
            for opt in strategy_instance.get_config_options():
                if opt['name'] in self.strategy_config_widgets:
                     strategy_config_details_final.append(f"{opt['label'].replace(':','')} {self.strategy_config_widgets[opt['name']].get()}")
            strategy_params_report_final = ", ".join(strategy_config_details_final)
            if strategy_params_report_final: strategy_params_report_final = f" ({strategy_params_report_final})"


            final_report = f"\n=== BACKTEST MULTI-POSIZIONE COMPLETATO (Strategia: {strategy_instance.get_name()}{strategy_params_report_final}, Verifica a {colpi_verifica} colpi) ===\n"
            window_report = f"Finestra Storica: {historical_window_size_param}" if historical_window_size_param > 0 else "Finestra Storica: Tutte"; final_report += f"({window_report})\n"; final_report += f"Periodo Analizzato (date predizione): {results[0]['date_pred']} - {results[-1]['date_pred']}\n"; final_report += f"Predizioni Valide Effettuate: {total_valid_preds}\n"; final_report += f"Predizioni Vincenti (almeno 1 su 5): {correct_count} ({accuracy:.2f}%)\n";
            if correct_count > 0 : final_report += f"Colpo medio di uscita: {avg_colpo:.2f}\n"
            final_report += "---------------------------------------------------------------------------------------\n"; header_preds="  ".join([f"P{i+1}" for i in range(NUM_POSIZIONI)]); final_report += f"Data Pred.  {header_preds}      Dett.      Esito   Dettaglio Uscita (Num=Num @Colpo)\n"; final_report += "---------------------------------------------------------------------------------------\n"
            for r in results:
                 pred_str=" ".join(f"{p:>2}" for p in r['predicted_dec']); detail_str=r['detail'].ljust(11); esito_str=r['outcome'][0]; colpo_str=f"{r['outcome'][1]:>3}" if esito_str=='OK' else f"(>{colpi_verifica})" if esito_str=='NO' else "---"; dettaglio_uscita=r['hit_details'];
                 final_report += f"{r['date_pred']}  {pred_str} {detail_str} {esito_str:<4} {colpo_str}  {dettaglio_uscita}\n"
            final_report += "=======================================================================================\n"

            def update_ui_final(): self.results_text.insert(tk.END, final_report); self.results_text.see(tk.END); self.progress['value'] = 100
            self.root.after(0, update_ui_final)
        except Exception as e: error_msg = f"Errore durante il backtest Multi-Posizione: {e}\n{traceback.format_exc()}"; print(f"ERRORE BACKTEST MULTI: {error_msg}"); self.queue.put(("error", error_msg)); self.root.after(0, lambda: self.progress.config(value=0))


    # --- METODI DI UTILITÀ ---
    def _update_prediction_display(self):
        for i in range(NUM_POSIZIONI):
            offset = i * BITS_PER_NUMERO
            bits = self.all_predictions.get(offset, [None] * BITS_PER_NUMERO)
            formatted_bits = "[" + " ".join(str(b) if b is not None else "_" for b in bits) + "]"
            if i < len(self.prediction_labels): self.prediction_labels[i].config(text=formatted_bits)

    def check_queue(self):
        try:
            while not self.queue.empty():
                msg_type, content = self.queue.get_nowait()
                if msg_type == "error": messagebox.showerror("Errore dal Thread", content)
                elif msg_type == "info":
                    # Potrebbe essere un report del backtest, quindi inseriscilo nel log
                    if "BACKTEST" in content or "PREDIZIONE COMPLETA" in content or "Nessuna predizione valida" in content:
                        self.results_text.insert(tk.END, content + "\n")
                        self.results_text.see(tk.END)
                    else:
                        messagebox.showinfo("Info dal Thread", content)
        except queue.Empty: pass
        except Exception as e: print(f"!!! Errore in check_queue: {e}")
        finally: self.root.after(100, self.check_queue)

    def svuota_predizioni(self): self.all_predictions = {}; self._update_prediction_display()
    def mostra_decimali_predizioni(self):
        if not self.all_predictions: messagebox.showwarning("Predizioni Mancanti", "Nessuna predizione calcolata."); return
        output_lines = []; valid_prediction_found = False
        for i in range(NUM_POSIZIONI):
            offset = i * BITS_PER_NUMERO; bits = self.all_predictions.get(offset)
            if bits and None not in bits:
                try: binary_string = "".join(map(str, bits)); val = int(binary_string, 2); output_lines.append(f"Pos {i+1}: {val} ({binary_string})"); valid_prediction_found = True
                except ValueError: output_lines.append(f"Pos {i+1}: Errore ({''.join(map(str, bits))})")
            else: output_lines.append(f"Pos {i+1}: --- ([ _ _ _ _ _ _ _ ])")
        if not valid_prediction_found: messagebox.showwarning("Predizioni Incomplete", "Nessuna predizione valida completa calcolata.")
        else: messagebox.showinfo("Valori Decimali Predetti", "\n".join(output_lines))

    def copy_results(self):
        try:
            text_to_copy = self.results_text.get("1.0", tk.END)
            if text_to_copy.strip(): self.root.clipboard_clear(); self.root.clipboard_append(text_to_copy); messagebox.showinfo("Copiato", "Log copiato.")
            else: messagebox.showinfo("Info", "Nessun log da copiare.")
        except tk.TclError: messagebox.showwarning("Attenzione", "Impossibile accedere agli appunti.")
        except Exception as e: messagebox.showerror("Errore Copia", f"Errore: {e}")
    def clear_results(self): self.results_text.delete("1.0", tk.END)

# ==============================================================================
# FUNZIONE DI LANCIO
# ==============================================================================
def launch_numerical_binary_window(parent_window):
    try:
        nb_window = tk.Toplevel(parent_window)
        # nb_window.grab_set() # Rimosso per testare più facilmente, puoi riattivarlo
        nb_window.focus_set()
        app_instance = SequenzaSpiaApp(nb_window)
    except Exception as e:
        messagebox.showerror("Errore Avvio Modulo", f"Impossibile avviare il modulo Numerical Binary:\n{e}\n{traceback.format_exc()}", parent=parent_window)
        if 'nb_window' in locals() and nb_window.winfo_exists(): nb_window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    # Per testare standalone, aggiungi un pulsante per lanciare
    def open_app():
        launch_numerical_binary_window(root)
    ttk.Button(root, text="Apri Numerical Binary Strategie", command=open_app).pack(padx=20, pady=20)
    root.mainloop()
