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
from itertools import product
import math # Aggiunto per ceil

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
LUNGHEZZA_PATTERN_VERTICALE = 4
DEFAULT_COLPI_VERIFICA = 9 # Numero di colpi default per il backtest
NUM_POSIZIONI = 5 # Analizziamo 5 numeri estratti
BITS_PER_NUMERO = 7 # Ogni numero è rappresentato da 7 bit

# ==============================================================================
# CLASSE PRINCIPALE DELL'APPLICAZIONE
# ==============================================================================
class SequenzaSpiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Binary - Il Lotto di Max -")
        self.setup_variables()
        self.create_ui()
        self.queue = queue.Queue()
        self.check_queue()

    def setup_variables(self):
        """Inizializza le variabili di istanza."""
        self.historical_data = None
        # self.carrello non è più usato per la visualizzazione principale
        self.ruota_var = tk.StringVar(value="BA")
        self.selected_ruota_code = "BA"
        self.loaded_info = "Nessun dato caricato."
        self.all_predictions = {} # Conserva tutte le predizioni {offset: bits_list}
        self.window_size_var = tk.StringVar(value="300")
        self.prediction_labels = [] # Lista per le etichette delle predizioni

    def create_ui(self):
        """Crea l'interfaccia utente."""
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True); main_frame.columnconfigure(0, weight=1)
        current_row = 0

        # --- 1. Caricamento Dati --- (Invariato)
        load_data_frame = ttk.LabelFrame(main_frame, text="1. Caricamento Dati Ruota", padding="10"); load_data_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); load_data_frame.columnconfigure(1, weight=1); current_row += 1
        ruota_subframe = ttk.Frame(load_data_frame); ruota_subframe.grid(row=0, column=0, columnspan=2, sticky="ew", pady=2); ttk.Label(ruota_subframe, text="Seleziona Ruota:", anchor="w").pack(side=tk.LEFT, padx=5); ruota_menu = ttk.OptionMenu(ruota_subframe, self.ruota_var, "BA", *URL_RUOTE.keys(), command=self._update_selected_ruota); ruota_menu.pack(side=tk.LEFT, padx=5); self.ruota_label = ttk.Label(ruota_subframe, text="Ruota: Bari", anchor="w"); self.ruota_label.pack(side=tk.LEFT, padx=5); ttk.Button(ruota_subframe, text="Carica Dati Ruota", width=18, command=self.carica_dati_ruota).pack(side=tk.LEFT, padx=10)
        self.loaded_file_label = ttk.Label(load_data_frame, text=self.loaded_info, anchor="w"); self.loaded_file_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(5,2))

        # --- Anteprima Dati --- (Invariato)
        preview_frame = ttk.LabelFrame(main_frame, text="Anteprima Dati Binari Caricati (Ultime 5)", padding="10"); preview_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5); preview_frame.columnconfigure(0, weight=1); current_row += 1
        self.binary_preview = scrolledtext.ScrolledText(preview_frame, height=5, width=60, wrap=tk.WORD, font=('Courier New', 9)); self.binary_preview.grid(row=0, column=0, sticky="ew"); self.binary_preview.insert('1.0', "Nessun dato caricato."); self.binary_preview.config(state=tk.DISABLED)

        # --- 3. Backtesting su Periodo Storico --- (Invariato)
        backtest_frame = ttk.LabelFrame(main_frame, text="3. Verifica Multi-Posizione su Periodo Storico (Backtest)", padding="10"); backtest_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=10); current_row += 1
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

        # --- 2. Predizione --- (Invariato)
        analysis_frame = ttk.LabelFrame(main_frame, text="2. Predizione Prossima Estrazione (Tutte le Posizioni)", padding="10"); analysis_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=10); current_row += 1
        ttk.Button(analysis_frame, text=">> CALCOLA PREDIZIONE PER TUTTE LE POSIZIONI <<", style='Accent.TButton', command=self.start_best_vertical_pattern_analysis).pack(padx=15, ipady=4)

        # --- 4. Risultato Predizione --- (Modificato per 5 label)
        pred_display_frame = ttk.LabelFrame(main_frame, text="4. Ultima Predizione Calcolata", padding="10")
        pred_display_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5)
        pred_display_frame.columnconfigure(1, weight=1)
        current_row += 1
        self.prediction_labels = []
        for i in range(NUM_POSIZIONI):
            ttk.Label(pred_display_frame, text=f"P{i+1}:").grid(row=i, column=0, padx=(5,2), pady=1, sticky="w")
            bits_label = ttk.Label(pred_display_frame, text="[ _ _ _ _ _ _ _ ]", font=('Courier New', 11, 'bold'), anchor="w")
            bits_label.grid(row=i, column=1, padx=(0,5), pady=1, sticky="ew")
            self.prediction_labels.append(bits_label)
        button_pred_frame = ttk.Frame(pred_display_frame)
        button_pred_frame.grid(row=NUM_POSIZIONI, column=0, columnspan=2, pady=(5, 0), sticky='ew')
        ttk.Button(button_pred_frame, text="Svuota Predizioni", width=15, command=self.svuota_predizioni).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_pred_frame, text="Mostra Decimali", width=15, command=self.mostra_decimali_predizioni).pack(side=tk.LEFT, padx=5)

        # --- Progress bar --- (Invariato)
        self.progress = ttk.Progressbar(main_frame, mode='determinate'); self.progress.grid(row=current_row, column=0, sticky="ew", padx=5, pady=(5, 0)); current_row += 1

        # --- Log Analisi / Risultati Backtest --- (Invariato)
        results_frame = ttk.LabelFrame(main_frame, text="Log / Risultati Predizioni e Backtest", padding="10"); results_frame.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5); main_frame.rowconfigure(current_row, weight=1); results_frame.columnconfigure(0, weight=1); results_frame.rowconfigure(1, weight=1)
        button_frame = ttk.Frame(results_frame); button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5)); ttk.Button(button_frame, text="Copia Log", width=12, command=self.copy_results).pack(side=tk.LEFT, padx=5); ttk.Button(button_frame, text="Cancella Log", width=12, command=self.clear_results).pack(side=tk.LEFT, padx=5)
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=70, wrap=tk.WORD, font=('Consolas', 9)); self.results_text.grid(row=1, column=0, sticky="nsew")
        try: style = ttk.Style(); style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'), foreground='navy')
        except tk.TclError: pass
        self._update_selected_ruota(self.ruota_var.get())

    # --- METODI DI CARICAMENTO DATI --- (Invariato)
    def _update_selected_ruota(self, value): self.selected_ruota_code = value; self.ruota_label.config(text=f"Ruota: {self.get_ruota_name(value)}")
    def get_ruota_name(self, code): ruote_nomi = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova','MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma','TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}; return ruote_nomi.get(code, code)
    def carica_dati_ruota(self):
        self.all_predictions = {}; self.svuota_predizioni() # Svuota le predizioni quando carichi nuova ruota
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
        except requests.exceptions.Timeout: messagebox.showerror("Errore Rete", f"Timeout GitHub ({url})."); self.loaded_file_label.config(text="Errore Timeout")
        except requests.exceptions.HTTPError as e: messagebox.showerror("Errore HTTP", f"Errore {e.response.status_code} GitHub ({url})."); self.loaded_file_label.config(text=f"Errore HTTP {e.response.status_code}")
        except requests.exceptions.RequestException as e: messagebox.showerror("Errore Rete", f"Impossibile scaricare: {e}"); self.loaded_file_label.config(text="Errore Rete")
        except ValueError as e: messagebox.showerror("Errore Dati", f"{e}"); self.loaded_file_label.config(text="Errore dati")
        except Exception as e: messagebox.showerror("Errore", f"Errore caricamento: {e}\n{traceback.format_exc()}"); self.loaded_file_label.config(text="Errore")
    def _update_preview(self): # Invariato
        self.binary_preview.config(state=tk.NORMAL); self.binary_preview.delete('1.0', tk.END)
        if self.historical_data is not None and not self.historical_data.empty:
            preview_df = self.historical_data.tail(5); preview_text = "\n".join(preview_df['BinarioCompleto'].tolist())
            if len(self.historical_data) > 5:
                preview_text = "...\n" + preview_text
            self.binary_preview.insert('1.0', preview_text)
        else:
            self.binary_preview.insert('1.0', "Nessun dato.")
        self.binary_preview.config(state=tk.DISABLED)

    # --- LOGICA DI ANALISI MULTI-POSIZIONE --- (Invariato)
    def start_best_vertical_pattern_analysis(self):
        if self.historical_data is None or self.historical_data.empty: messagebox.showwarning("Dati Mancanti", "Caricare i dati."); return
        try:
            window_size_str = self.window_size_var.get(); historical_window_size = int(window_size_str)
            if historical_window_size < 0: raise ValueError("La finestra storica non può essere negativa.")
        except ValueError: messagebox.showerror("Errore Input", f"Valore non valido per Finestra Storica: '{window_size_str}'. Inserire un numero intero >= 0."); return
        self.clear_results(); window_info_title = f"(Finestra: {historical_window_size})" if historical_window_size > 0 else "(Finestra: Tutte)"
        self.results_text.insert(tk.END, f"Avvio analisi 'Migliori Pattern Verticali' {window_info_title} (H={LUNGHEZZA_PATTERN_VERTICALE}) per tutte le {NUM_POSIZIONI} posizioni...\n");
        self.progress['value'] = 0; self.svuota_predizioni();
        Thread(target=self.run_best_vertical_pattern_analysis_thread, args=(self.historical_data.copy(), historical_window_size), daemon=True).start()
    def run_best_vertical_pattern_analysis_thread(self, data_df, historical_window_size_param): # Invariato
        try:
            predictions = {}; analysis_reports = []
            if historical_window_size_param > 0 and len(data_df) > historical_window_size_param: recent_data_df = data_df.tail(historical_window_size_param).copy(); window_info = f"(Ultime {len(recent_data_df)} estrazioni, Finestra={historical_window_size_param})";
            else: recent_data_df = data_df.copy(); window_info = f"(Tutte le {len(recent_data_df)} estrazioni)";
            total_steps = NUM_POSIZIONI; completed_steps = 0
            for i in range(NUM_POSIZIONI):
                bit_offset = i * BITS_PER_NUMERO; predicted_bits, analysis_details = self._run_prediction_logic_best_vertical(recent_data_df, bit_offset); predictions[bit_offset] = predicted_bits
                binary_string = "".join(map(str, predicted_bits));
                try: decimal_value = int(binary_string, 2)
                except ValueError: decimal_value = "Errore"
                pos_report = f"\n--- Predizione Posizione {i+1} (Offset {bit_offset}) ---\n" + "\n".join(analysis_details) + "\n" + f"Numero Binario Predetto (Pos {i+1}): {binary_string}\n" + f"Valore Decimale (Pos {i+1}): {decimal_value}\n"; analysis_reports.append(pos_report)
                completed_steps += 1; progress_val = (completed_steps / total_steps) * 100; self.root.after(0, lambda p=progress_val: self.progress.config(value=p))
            final_report = f"\n=== PREDIZIONE COMPLETA (Migliori Pattern Verticali H={LUNGHEZZA_PATTERN_VERTICALE}) ===\n" + f"(Basata su {len(recent_data_df)} estrazioni storiche {window_info})\n" + "".join(analysis_reports) + "--- Riepilogo Predizioni Decimali ---\n"
            summary = [];
            for i in range(NUM_POSIZIONI):
                offset = i * BITS_PER_NUMERO; bits = predictions.get(offset, [None]*BITS_PER_NUMERO); bin_str = "".join(map(str, bits));
                try: dec_val = int(bin_str, 2)
                except: dec_val = "?";
                summary.append(f"Pos {i+1}: {dec_val} ({bin_str})")
            final_report += ", ".join(summary) + "\n" + "=========================================\n"
            def update_ui():
                self.all_predictions = predictions.copy()
                self._update_prediction_display() # Chiama nuovo metodo per aggiornare UI
                self.results_text.insert(tk.END, final_report); self.results_text.see(tk.END); self.progress['value'] = 100
            self.root.after(0, update_ui)
        except Exception as e: error_msg = f"Errore analisi Multi-Posizione: {e}\n{traceback.format_exc()}"; print(f"ERRORE MULTI-POS: {error_msg}"); self.root.after(0, lambda: messagebox.showerror("Errore Analisi", error_msg)); self.root.after(0, lambda: self.progress.config(value=0))
    def _run_prediction_logic_best_vertical(self, data_to_analyze, bit_offset): # Invariato
        final_predicted_bits = [0] * BITS_PER_NUMERO; analysis_details = [""] * BITS_PER_NUMERO
        if isinstance(data_to_analyze, pd.DataFrame) and 'BinarioCompleto' in data_to_analyze.columns: binary_strings = data_to_analyze['BinarioCompleto'].tolist()
        else: print(f"ERRORE: Dati di input non validi per offset {bit_offset}."); [analysis_details.__setitem__(i, f"Bit Pos {i}(+{bit_offset}): Errore Dati") for i in range(BITS_PER_NUMERO)]; return final_predicted_bits, analysis_details
        reversed_data = list(reversed(binary_strings)); num_rows = len(reversed_data)
        if num_rows <= LUNGHEZZA_PATTERN_VERTICALE: print(f"Dati insuff ({num_rows}) per pattern H={LUNGHEZZA_PATTERN_VERTICALE} (Offset {bit_offset})."); [analysis_details.__setitem__(i, f"Bit Pos {i}(+{bit_offset}): Predetto=0 (Dati insuff.)") for i in range(BITS_PER_NUMERO)]; return final_predicted_bits, analysis_details
        for bit_pos_relative in range(BITS_PER_NUMERO):
            target_bit_index = bit_offset + bit_pos_relative; pattern_stats_for_col = {}
            for i in range(num_rows - LUNGHEZZA_PATTERN_VERTICALE):
                history_window_rows = reversed_data[i : i + LUNGHEZZA_PATTERN_VERTICALE]; prediction_row = reversed_data[i + LUNGHEZZA_PATTERN_VERTICALE]
                try:
                    if len(prediction_row) <= target_bit_index or any(len(hr) <= target_bit_index for hr in history_window_rows): continue
                    vertical_pattern_tuple = tuple(int(row[target_bit_index]) for row in history_window_rows); following_bit = int(prediction_row[target_bit_index])
                    if vertical_pattern_tuple not in pattern_stats_for_col: pattern_stats_for_col[vertical_pattern_tuple] = {'matches': 0, 'zeros': 0, 'ones': 0}
                    pattern_stats_for_col[vertical_pattern_tuple]['matches'] += 1
                    if following_bit == 1: pattern_stats_for_col[vertical_pattern_tuple]['ones'] += 1
                    else: pattern_stats_for_col[vertical_pattern_tuple]['zeros'] += 1
                except (ValueError, IndexError): continue
            best_pattern = None; max_imbalance = -1; best_pattern_stats = {}
            if not pattern_stats_for_col: final_predicted_bits[bit_pos_relative] = 0; analysis_details[bit_pos_relative] = f"Bit Pos {bit_pos_relative}(+{bit_offset}): Predetto=0 (Nessun pattern H={LUNGHEZZA_PATTERN_VERTICALE})"; continue
            for pattern, stats in pattern_stats_for_col.items():
                imbalance = abs(stats['ones'] - stats['zeros']); matches = stats['matches']
                if matches > 0:
                    if imbalance > max_imbalance: max_imbalance = imbalance; best_pattern = pattern; best_pattern_stats = stats
                    elif imbalance == max_imbalance and matches > best_pattern_stats.get('matches', 0): best_pattern = pattern; best_pattern_stats = stats
            if best_pattern is not None and best_pattern_stats: predicted_bit = 1 if best_pattern_stats['ones'] > best_pattern_stats['zeros'] else 0; final_predicted_bits[bit_pos_relative] = predicted_bit; pattern_str = "".join(map(str, best_pattern)); analysis_details[bit_pos_relative] = (f"Bit Pos {bit_pos_relative}(+{bit_offset}): Predetto={predicted_bit} (BestPat='{pattern_str}': Z={best_pattern_stats['zeros']}, U={best_pattern_stats['ones']}, M={best_pattern_stats['matches']})")
            else: final_predicted_bits[bit_pos_relative] = 0; analysis_details[bit_pos_relative] = f"Bit Pos {bit_pos_relative}(+{bit_offset}): Predetto=0 (Nessun pattern predittivo)"
        return final_predicted_bits, analysis_details

    # --- METODI PER BACKTESTING MULTI-POSIZIONE --- (Invariato)
    def start_backtest(self):
        if self.historical_data is None or self.historical_data.empty: messagebox.showwarning("Dati Mancanti", "Caricare i dati."); return
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
        self.clear_results(); window_info_title = f"(Finestra: {historical_window_size})" if historical_window_size > 0 else "(Finestra: Tutte)"; self.results_text.insert(tk.END, f"Avvio backtest Multi-Posizione {window_info_title} (H={LUNGHEZZA_PATTERN_VERTICALE}, Verifica a {colpi_verifica} colpi)\n"); self.results_text.insert(tk.END, f"Periodo predizioni: {start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}...\n"); self.results_text.insert(tk.END, "Verifica: OK se ALMENO UNO dei 5 numeri predetti esce in UNA delle 5 posizioni reali entro i colpi.\n\n")
        self.progress['value'] = 0; self.svuota_predizioni();
        Thread(target=self.run_backtest_thread, args=(backtest_prediction_dates, colpi_verifica, historical_window_size), daemon=True).start()
    def run_backtest_thread(self, prediction_dates, colpi_verifica, historical_window_size_param): # Invariato
        results = []; total_dates = len(prediction_dates); dates_processed = 0
        try:
            all_dates = self.historical_data.index; actual_columns = ['BinarioCompleto'] + [f'BinarioPos{i+1}' for i in range(NUM_POSIZIONI)]; all_actuals_df = self.historical_data[actual_columns]
            for target_date in prediction_dates:
                data_for_prediction_full = self.historical_data[all_dates < target_date]
                if historical_window_size_param > 0 and len(data_for_prediction_full) > historical_window_size_param: data_for_prediction_limited = data_for_prediction_full.tail(historical_window_size_param).copy()
                else: data_for_prediction_limited = data_for_prediction_full.copy()
                predicted_numbers_bin = ["0"*BITS_PER_NUMERO]*NUM_POSIZIONI; predicted_numbers_dec = ["-"]*NUM_POSIZIONI; pred_detail = ""; window_len_used = len(data_for_prediction_limited)
                if window_len_used <= LUNGHEZZA_PATTERN_VERTICALE:
                    if historical_window_size_param > 0 and len(data_for_prediction_full) > historical_window_size_param: pred_detail = f" (D.Insuff. Win={window_len_used})"
                    else: pred_detail = f" (D.Insuff. Tot={window_len_used})"
                else:
                    temp_predictions = {}
                    for i in range(NUM_POSIZIONI): bit_offset = i*BITS_PER_NUMERO; predicted_bits, _ = self._run_prediction_logic_best_vertical(data_for_prediction_limited, bit_offset); temp_predictions[bit_offset] = predicted_bits
                    for i in range(NUM_POSIZIONI):
                        offset = i*BITS_PER_NUMERO; bits = temp_predictions.get(offset,[0]*BITS_PER_NUMERO); bin_str = "".join(map(str,bits)); predicted_numbers_bin[i] = bin_str;
                        try: predicted_numbers_dec[i] = str(int(bin_str, 2))
                        except: predicted_numbers_dec[i] = "?"
                    if not pred_detail: pred_detail = f" (W={window_len_used})"
                found = False; hit_colpo = 0; actual_numbers_at_hit = ["-"]*5; hit_details = ""
                try:
                    start_check_loc = all_dates.get_loc(target_date)
                    if start_check_loc + 1 >= len(all_dates): outcome = ('SKIP', 0); hit_details="No estraz. succ."; results.append({'date_pred': target_date.strftime('%Y/%m/%d'), 'predicted_dec': predicted_numbers_dec, 'detail': pred_detail, 'outcome': outcome, 'hit_details': hit_details}); dates_processed += 1; continue
                except KeyError: continue
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
            # Fine ciclo backtest
            if not results: final_report = "\nNessuna predizione valida generata o verificata nel periodo."; self.root.after(0, lambda: self.results_text.insert(tk.END, final_report)); self.root.after(0, lambda: self.progress.config(value=100)); return
            valid_results=[r for r in results if r['outcome'][0]!='SKIP']; total_valid_preds=len(valid_results); correct_count=sum(1 for r in valid_results if r['outcome'][0]=='OK'); accuracy=(correct_count/total_valid_preds)*100 if total_valid_preds>0 else 0; avg_colpo=sum(r['outcome'][1] for r in valid_results if r['outcome'][0]=='OK')/correct_count if correct_count > 0 else 0
            final_report = f"\n=== BACKTEST MULTI-POSIZIONE COMPLETATO (Verifica a {colpi_verifica} colpi) ===\n"; window_report = f"Finestra Storica: {historical_window_size_param}" if historical_window_size_param > 0 else "Finestra Storica: Tutte"; final_report += f"({window_report})\n"; final_report += f"Periodo Analizzato (date predizione): {results[0]['date_pred']} - {results[-1]['date_pred']}\n"; final_report += f"Predizioni Valide Effettuate: {total_valid_preds}\n"; final_report += f"Predizioni Vincenti (almeno 1 su 5): {correct_count} ({accuracy:.2f}%)\n";
            if correct_count > 0 : final_report += f"Colpo medio di uscita: {avg_colpo:.2f}\n"
            final_report += "---------------------------------------------------------------------------------------\n"; header_preds="  ".join([f"P{i+1}" for i in range(NUM_POSIZIONI)]); final_report += f"Data Pred.  {header_preds}      Dett.   Esito   Dettaglio Uscita (Num=Num @Colpo)\n"; final_report += "---------------------------------------------------------------------------------------\n"
            # MODIFICA: Rimosso limite righe log
            # max_lines_in_log=100; lines_shown=0 # Rimosso
            for r in results:
                 pred_str=" ".join(f"{p:>2}" for p in r['predicted_dec']); detail_str=r['detail'].ljust(11); esito_str=r['outcome'][0]; colpo_str=f"{r['outcome'][1]:>3}" if esito_str=='OK' else f"(>{colpi_verifica})" if esito_str=='NO' else "---"; dettaglio_uscita=r['hit_details'];
                 final_report += f"{r['date_pred']}  {pred_str} {detail_str} {esito_str:<4} {colpo_str}  {dettaglio_uscita}\n"
                 # lines_shown+=1 # Rimosso
                 # Blocco if per limitare righe rimosso
            final_report += "=======================================================================================\n"
            def update_ui_final(): self.results_text.insert(tk.END, final_report); self.results_text.see(tk.END); self.progress['value'] = 100
            self.root.after(0, update_ui_final)
        except Exception as e: error_msg = f"Errore durante il backtest Multi-Posizione: {e}\n{traceback.format_exc()}"; print(f"ERRORE BACKTEST MULTI: {error_msg}"); self.root.after(0, lambda: messagebox.showerror("Errore Backtest", error_msg)); self.root.after(0, lambda: self.progress.config(value=0))

    # --- METODI DI UTILITÀ ---
    # Metodo _format_carrello rimosso (non più necessario)

    # NUOVO METODO per aggiornare le 5 etichette
    def _update_prediction_display(self):
        """Aggiorna le etichette dell'interfaccia con le predizioni calcolate."""
        for i in range(NUM_POSIZIONI):
            offset = i * BITS_PER_NUMERO
            bits = self.all_predictions.get(offset, [None] * BITS_PER_NUMERO)
            formatted_bits = "[" + " ".join(str(b) if b is not None else "_" for b in bits) + "]"
            # Assicura che l'indice sia valido prima di configurare
            if i < len(self.prediction_labels):
                self.prediction_labels[i].config(text=formatted_bits)

    def check_queue(self): # Invariato
        try:
            while not self.queue.empty():
                msg_type, content = self.queue.get_nowait()
                if msg_type == "error": messagebox.showerror("Errore dal Thread", content)
                elif msg_type == "info": messagebox.showinfo("Info dal Thread", content)
        except queue.Empty: pass
        except Exception as e: print(f"!!! Errore in check_queue: {e}")
        finally: self.root.after(100, self.check_queue)

    # Metodo svuota_carrello RINOMINATO e MODIFICATO
    def svuota_predizioni(self):
        """Svuota le predizioni calcolate e aggiorna l'interfaccia."""
        self.all_predictions = {} # Svuota il dizionario
        self._update_prediction_display() # Aggiorna le etichette UI

    # Metodo mostra_decimale_carrello RINOMINATO e MODIFICATO
    def mostra_decimali_predizioni(self):
        """Mostra i valori decimali di tutte le predizioni calcolate."""
        if not self.all_predictions:
            messagebox.showwarning("Predizioni Mancanti", "Nessuna predizione calcolata.")
            return
        output_lines = []
        valid_prediction_found = False
        for i in range(NUM_POSIZIONI):
            offset = i * BITS_PER_NUMERO
            bits = self.all_predictions.get(offset)
            if bits and None not in bits:
                try:
                    binary_string = "".join(map(str, bits))
                    val = int(binary_string, 2)
                    output_lines.append(f"Pos {i+1}: {val} ({binary_string})")
                    valid_prediction_found = True
                except ValueError:
                    output_lines.append(f"Pos {i+1}: Errore ({''.join(map(str, bits))})")
            else:
                output_lines.append(f"Pos {i+1}: --- ([ _ _ _ _ _ _ _ ])")
        if not valid_prediction_found:
             messagebox.showwarning("Predizioni Incomplete", "Nessuna predizione valida completa calcolata.")
        else:
             messagebox.showinfo("Valori Decimali Predetti", "\n".join(output_lines))

    def copy_results(self): # Invariato
        try:
            text_to_copy = self.results_text.get("1.0", tk.END)
            if text_to_copy.strip():
                self.root.clipboard_clear(); self.root.clipboard_append(text_to_copy); messagebox.showinfo("Copiato", "Log copiato.")
            else: messagebox.showinfo("Info", "Nessun log da copiare.")
        except tk.TclError: messagebox.showwarning("Attenzione", "Impossibile accedere agli appunti.")
        except Exception as e: messagebox.showerror("Errore Copia", f"Errore: {e}")
    def clear_results(self): # Invariato
        self.results_text.delete("1.0", tk.END)

# ==============================================================================
# FUNZIONE DI LANCIO PER INTEGRAZIONE
# ==============================================================================
def launch_numerical_binary_window(parent_window):
    """
    Crea una nuova finestra Toplevel e vi inserisce l'applicazione SequenzaSpiaApp.
    Questa è la funzione chiamata dal programma principale (Empathx).
    """
    try:
        # Crea una nuova finestra figlia della finestra principale passata
        nb_window = tk.Toplevel(parent_window)
        nb_window.grab_set() # Rende la finestra modale (opzionale, rimuovi se non desiderato)
        nb_window.focus_set()

        # Istanzia la classe dell'applicazione passando la NUOVA finestra Toplevel come root
        app_instance = SequenzaSpiaApp(nb_window)

        # (Opzionale) Potresti voler attendere che questa finestra si chiuda
        # parent_window.wait_window(nb_window)

    except Exception as e:
        messagebox.showerror("Errore Avvio Modulo",
                             f"Impossibile avviare il modulo Numerical Binary:\n{e}\n{traceback.format_exc()}",
                             parent=parent_window)
        print(f"ERRORE durante il lancio di Numerical Binary: {e}")
        if 'nb_window' in locals() and nb_window.winfo_exists():
             nb_window.destroy() # Chiudi la finestra se creata ma l'app fallisce

