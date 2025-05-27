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
from itertools import combinations
from collections import Counter
import calendar

# --- Import per Matplotlib ---
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# --- Fine Import per Matplotlib ---

# --- Configurazione GitHub ---
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"

# Definizione URL Ruote e nomi per la GUI
RUOTE_DISPONIBILI = {
    'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova',
    'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma',
    'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'
}
URL_RUOTE = {
    key: f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{value.upper()}.txt'
    for key, value in RUOTE_DISPONIBILI.items()
}
RUOTE_NOMI = list(RUOTE_DISPONIBILI.keys())
MESI_NOMI = [calendar.month_name[i] for i in range(1, 13)]

class LottoApp:
    def __init__(self, master):
        self.master = master
        master.title("Ricerca Lunghette - Il Lotto di Max - ")
        master.geometry("850x750") 

        self.data_cache = {}
        self.analysis_running = False
        self.q = queue.Queue()

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- WIDGET DI INPUT (come nel tuo codice originale) ---
        period_frame = ttk.LabelFrame(main_frame, text="Periodo Storico per Analisi Frequenze", padding="10")
        period_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(period_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = DateEntry(period_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.set_date(datetime.now() - timedelta(days=365*2))
        ttk.Label(period_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = DateEntry(period_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.set_date(datetime.now() - timedelta(days=1))

        adv_filters_frame = ttk.LabelFrame(main_frame, text="Filtri Temporali Avanzati per Analisi", padding="10")
        adv_filters_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(adv_filters_frame, text="Indice Estrazione del Mese:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        MAX_INDICE_ESTRAZIONE_MESE = 19 # Puoi aggiustare questo valore se necessario
        self.indice_mese_values = ["Tutte"] + [f"{i}ª del mese" for i in range(1, MAX_INDICE_ESTRAZIONE_MESE + 1)]
        self.indice_mese_combo = ttk.Combobox(adv_filters_frame, values=self.indice_mese_values, width=15, state="readonly")
        self.indice_mese_combo.current(0)
        self.indice_mese_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        mesi_filter_frame = ttk.LabelFrame(adv_filters_frame, text="Seleziona Mesi Specifici (se nessuno, tutti)", padding="5")
        mesi_filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        self.mesi_vars = {}
        for i, nome_mese in enumerate(MESI_NOMI):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(mesi_filter_frame, text=nome_mese, variable=var)
            cb.grid(row=i // 4, column=i % 4, sticky="w", padx=3, pady=2)
            self.mesi_vars[i+1] = var
        btn_frame_mesi = ttk.Frame(mesi_filter_frame)
        btn_frame_mesi.grid(row=(len(MESI_NOMI) + 3) // 4, column=0, columnspan=4, pady=5)
        ttk.Button(btn_frame_mesi, text="Seleziona Tutti i Mesi", command=lambda: self.toggle_all_mesi(True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame_mesi, text="Deseleziona Tutti i Mesi", command=lambda: self.toggle_all_mesi(False)).pack(side=tk.LEFT, padx=2)

        ruote_analisi_frame = ttk.LabelFrame(main_frame, text="Ruote per Analisi Frequenze", padding="10")
        ruote_analisi_frame.grid(row=2, column=0, sticky="nsew", pady=5, padx=(0,5))
        self.ruote_analisi_vars = {}
        for i, ruota_key in enumerate(RUOTE_NOMI):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(ruote_analisi_frame, text=RUOTE_DISPONIBILI[ruota_key], variable=var)
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=5)
            self.ruote_analisi_vars[ruota_key] = var
        btn_frame_analisi = ttk.Frame(ruote_analisi_frame)
        btn_frame_analisi.grid(row=(len(RUOTE_NOMI) + 1) // 2, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame_analisi, text="Seleziona Tutte", command=lambda: self.toggle_all_ruote(self.ruote_analisi_vars, True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame_analisi, text="Deseleziona Tutte", command=lambda: self.toggle_all_ruote(self.ruote_analisi_vars, False)).pack(side=tk.LEFT, padx=2)

        ruote_gioco_frame = ttk.LabelFrame(main_frame, text="Ruote per Verifica Giocata", padding="10")
        ruote_gioco_frame.grid(row=2, column=1, sticky="nsew", pady=5, padx=(5,0))
        self.ruote_gioco_vars = {}
        for i, ruota_key in enumerate(RUOTE_NOMI):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(ruote_gioco_frame, text=RUOTE_DISPONIBILI[ruota_key], variable=var)
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=5)
            self.ruote_gioco_vars[ruota_key] = var
        btn_frame_gioco = ttk.Frame(ruote_gioco_frame)
        btn_frame_gioco.grid(row=(len(RUOTE_NOMI) + 1) // 2, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame_gioco, text="Seleziona Tutte", command=lambda: self.toggle_all_ruote(self.ruote_gioco_vars, True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame_gioco, text="Deseleziona Tutte", command=lambda: self.toggle_all_ruote(self.ruote_gioco_vars, False)).pack(side=tk.LEFT, padx=2)

        params_frame = ttk.LabelFrame(main_frame, text="Parametri Ricerca e Gioco", padding="10")
        params_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(params_frame, text="Lunghezza Lunghetta:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lunghetta_len_combo = ttk.Combobox(params_frame, values=list(range(2, 11)), width=5, state="readonly")
        self.lunghetta_len_combo.current(3)
        self.lunghetta_len_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(params_frame, text="Sorte da Giocare:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sorte_combo = ttk.Combobox(params_frame, values=["Ambo", "Terno", "Quaterna"], width=10, state="readonly")
        self.sorte_combo.current(0)
        self.sorte_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(params_frame, text="Colpi di Gioco:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.colpi_spinbox = ttk.Spinbox(params_frame, from_=1, to=20, width=5)
        self.colpi_spinbox.set(9)
        self.colpi_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        # --- FINE WIDGET DI INPUT ---

        action_buttons_frame = ttk.Frame(main_frame)
        action_buttons_frame.grid(row=4, column=0, columnspan=2, pady=10) 
        self.start_button = ttk.Button(action_buttons_frame, text="Avvia Analisi", command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.show_chart_button = ttk.Button(action_buttons_frame, text="Mostra Grafico", command=self.open_chart_window)
        self.show_chart_button.pack(side=tk.LEFT, padx=5)
        self.show_chart_button.config(state=tk.DISABLED)
        
        # --- NUOVO PULSANTE PER STATISTICHE ---
        self.show_stats_button = ttk.Button(action_buttons_frame, text="Mostra Statistiche", command=self.open_stats_window)
        self.show_stats_button.pack(side=tk.LEFT, padx=5)
        self.show_stats_button.config(state=tk.DISABLED)
        # --- FINE NUOVO PULSANTE ---

        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")

        self.results_text = scrolledtext.ScrolledText(main_frame, height=20, width=80, wrap=tk.WORD)
        self.results_text.grid(row=6, column=0, columnspan=2, pady=5, sticky="nsew")
        
        try: plt.style.use('seaborn-v0_8-whitegrid')
        except: pass 
        self.chart_figure = Figure(figsize=(8, 6), dpi=100)
        self.ax_freq_chart = self.chart_figure.add_subplot(111)
        self.chart_window = None 
        self.active_chart_canvas_for_toplevel = None
        self.last_chart_data = None 
        self.current_chart_type = "single_numbers" 

        # Attributi per la finestra delle statistiche
        self.stats_window = None
        self.last_stats_data = None # Dati per popolare le statistiche
        self.stats_labels = {} # Dizionario per le Label nella finestra Toplevel delle statistiche

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.master.after(100, self.process_queue)


    def toggle_all_mesi(self, select_state):
        for var in self.mesi_vars.values(): var.set(select_state)

    def toggle_all_ruote(self, ruote_vars_dict, select_state):
        for var in ruote_vars_dict.values(): var.set(select_state)

    def log_message(self, message, is_result=False):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)

    def _on_chart_window_close(self):
        if self.chart_window: self.chart_window.destroy()
        self.chart_window = None
        self.active_chart_canvas_for_toplevel = None

    def open_chart_window(self):
        if self.chart_window is not None and self.chart_window.winfo_exists():
            self.chart_window.lift()
            if hasattr(self, 'active_chart_canvas_for_toplevel') and self.active_chart_canvas_for_toplevel:
                 self.update_frequency_chart(self.last_chart_data, self.current_chart_type)
            return
        self.chart_window = tk.Toplevel(self.master)
        self.chart_window.title("Grafico Risultati Analisi") # Titolo più generico
        self.chart_window.geometry("800x600") # Più grande
        self.chart_window.protocol("WM_DELETE_WINDOW", self._on_chart_window_close)
        self.active_chart_canvas_for_toplevel = FigureCanvasTkAgg(self.chart_figure, master=self.chart_window)
        chart_canvas_widget = self.active_chart_canvas_for_toplevel.get_tk_widget()
        chart_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.update_frequency_chart(self.last_chart_data, self.current_chart_type)

    def update_frequency_chart(self, chart_data=None, chart_type="single_numbers"):
        if not hasattr(self, 'active_chart_canvas_for_toplevel') or \
           not self.active_chart_canvas_for_toplevel or \
           not self.chart_window or \
           not self.chart_window.winfo_exists():
            return 
        
        canvas_to_draw = self.active_chart_canvas_for_toplevel
        self.ax_freq_chart.clear()
        title = "Grafico Analisi"; xlabel = "Elemento"; ylabel = "Valore"
        rotate_labels_angle = 0
        ha_rotation = 'center'

        if chart_type == "single_numbers":
            title = "Frequenza Numeri Singoli (Top)"
            xlabel = "Numero"; ylabel = "Frequenza"
            rotate_labels_angle = 30; ha_rotation = 'right'
        elif chart_type == "backtest_performance":
            title = f"Successi Backtest (Top {len(chart_data) if chart_data else 'N'})"
            xlabel = "Lunghetta"; ylabel = "Numero Successi Trigger"
            rotate_labels_angle = 45; ha_rotation = 'right'
        
        if chart_data and isinstance(chart_data, list) and len(chart_data) > 0:
            if chart_type == "backtest_performance":
                labels = [item[0] for item in chart_data] # item[0] è già la stringa della lunghetta
            else: # Per numeri singoli o altre combinazioni (se le implementassi)
                labels = [str(item[0]).replace(' ', '') for item in chart_data] 
            
            values = [item[1] for item in chart_data]
            bars = self.ax_freq_chart.bar(labels, values, color='teal' if chart_type == "backtest_performance" else 'skyblue')
            self.ax_freq_chart.set_ylabel(ylabel)
            max_val = max(values) if values else 1
            min_val = min(values) if values else 0
            # Adatta il range dell'asse y per una migliore visualizzazione
            if max_val > 0 :
                self.ax_freq_chart.set_ylim(bottom=0, top=max_val + 0.1 * max_val)


            for bar in bars:
                yval = bar.get_height()
                # Non mostrare 0 sopra le barre se il valore è 0
                if yval > 0:
                    self.ax_freq_chart.text(bar.get_x() + bar.get_width()/2.0, yval, 
                                            round(yval), ha='center', va='bottom', fontsize=8, color='black')
        else:
            self.ax_freq_chart.text(0.5, 0.5, "Dati per il grafico non disponibili.\nAvvia un'analisi per popolarlo.", 
                                    horizontalalignment='center', verticalalignment='center', 
                                    transform=self.ax_freq_chart.transAxes, fontsize=10, color='gray', wrap=True)
        
        self.ax_freq_chart.set_title(title, fontsize=12, fontweight='bold')
        self.ax_freq_chart.set_xlabel(xlabel, fontsize=10)
        
        if chart_data and len(chart_data) > 0 : # Applica rotazione solo se ci sono etichette
            self.ax_freq_chart.tick_params(axis='x', labelsize=8, rotation=rotate_labels_angle, labelrotation=rotate_labels_angle, pad=5)
            if rotate_labels_angle > 0: # Prova a impostare l'allineamento per le etichette ruotate
                 plt.setp(self.ax_freq_chart.get_xticklabels(), ha=ha_rotation)
        else:
            self.ax_freq_chart.tick_params(axis='x', labelsize=8)
            
        self.ax_freq_chart.tick_params(axis='y', labelsize=8)
        self.ax_freq_chart.grid(True, linestyle='--', alpha=0.7) # Aggiungi griglia
        
        try: self.chart_figure.tight_layout(pad=2.5)
        except Exception: pass 
        
        canvas_to_draw.draw()

    # All'interno della classe LottoApp

    def _on_stats_window_close(self): # Invariato
        if self.stats_window:
            self.stats_window.destroy()
        self.stats_window = None
        # Non resettare self.stats_labels qui, potrebbero essere ricreate
        # o potremmo voler mantenere i dati per un ripopolamento veloce.
        # Per ora, lasciamo che open_stats_window gestisca la (ri)creazione.

    def open_stats_window(self):
        if self.stats_window is not None and self.stats_window.winfo_exists():
            self.stats_window.lift()
            # Se la finestra è già aperta, aggiorna con gli ultimi dati
            self.display_advanced_stats(self.last_stats_data)
            return

        self.stats_window = tk.Toplevel(self.master)
        self.stats_window.title("Statistiche Avanzate Lunghette")
        self.stats_window.geometry("750x400") # Più largo per il Treeview
        self.stats_window.resizable(True, True) # Permetti ridimensionamento
        self.stats_window.protocol("WM_DELETE_WINDOW", self._on_stats_window_close)

        container = ttk.Frame(self.stats_window, padding="10")
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Statistiche Avanzate per le Top Lunghette:", 
                  font=('Helvetica', 13, 'bold')).pack(pady=(0,10))
        
        # Creazione del Treeview
        cols = ("Lunghetta", "Sorte", "RA", "RSMax", "Freq", "Successi BT", "% Successo BT")
        self.stats_tree = ttk.Treeview(container, columns=cols, show='headings', height=10)
        
        for col_name in cols:
            self.stats_tree.heading(col_name, text=col_name)
            self.stats_tree.column(col_name, width=100, anchor='center') # Larghezza di default
        
        # Adatta larghezze specifiche
        self.stats_tree.column("Lunghetta", width=150, anchor='w')
        self.stats_tree.column("Sorte", width=60)
        self.stats_tree.column("RA", width=50)
        self.stats_tree.column("RSMax", width=70)
        self.stats_tree.column("Freq", width=50)
        self.stats_tree.column("Successi BT", width=80)
        self.stats_tree.column("% Successo BT", width=100)


        # Scrollbar per il Treeview
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.stats_tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X, before=self.stats_tree) # Prima del treeview per non coprirlo

        # Aggiorna con i dati correnti, se disponibili
        self.display_advanced_stats(self.last_stats_data)

    def display_advanced_stats(self, stats_data_list=None):
        # stats_data_list ora è una LISTA di dizionari
        if not self.stats_window or not self.stats_window.winfo_exists() or not hasattr(self, 'stats_tree'):
            return

        # Pulisci il Treeview precedente
        for i in self.stats_tree.get_children():
            self.stats_tree.delete(i)

        if stats_data_list and isinstance(stats_data_list, list):
            for i, stats_item in enumerate(stats_data_list):
                if isinstance(stats_item, dict):
                    lunghetta_str = str(stats_item.get('lunghetta', 'N/D')).replace(' ', '')
                    sorte = stats_item.get('sorte', 'N/D')
                    ra = stats_item.get('RA', 'N/D')
                    rs_max = stats_item.get('RSMax', 'N/D')
                    freq = stats_item.get('Freq', 'N/D')
                    succ_bt = stats_item.get('success_triggers_backtest', 'N/A')
                    perc_succ_bt_val = stats_item.get('percent_success_backtest', 'N/A')
                    perc_succ_bt = f"{perc_succ_bt_val:.2f}%" if isinstance(perc_succ_bt_val, (int, float)) else 'N/A'


                    self.stats_tree.insert("", tk.END, iid=str(i), values=(
                        lunghetta_str, sorte, ra, rs_max, freq, succ_bt, perc_succ_bt
                    ))
        else:
            # Potresti inserire una riga fittizia o lasciare vuoto
            self.stats_tree.insert("", tk.END, values=("Nessun dato di statistica disponibile.", "", "", "", "", "", ""))

    def _calculate_stats_for_lunghetta(self, lunghetta_da_analizzare, sorte_len, 
                                       processed_historical_dfs, # Non servono più selected_ruote_gioco e all_data_loaded_full per RA in questa logica
                                       current_end_date_hist): # Non strettamente necessario per RA se usiamo solo processed_historical_dfs
        """
        Calcola Ritardo Attuale (RA), Ritardo Storico Massimo (RSMax) e Frequenza Storica (Freq)
        BASANDOSI SUI DATI STORICI FILTRATI (processed_historical_dfs).
        RA qui significa: da quanti eventi filtrati consecutivi (andando a ritroso dall'ultimo evento filtrato)
        la sorte non è uscita.

        Args:
            lunghetta_da_analizzare (list): Lista di numeri [int].
            sorte_len (int): Lunghezza della sorte (es. 2 per ambo).
            processed_historical_dfs (dict): Dizionario {ruota_key: DataFrame} dei dati storici filtrati.
            current_end_date_hist (datetime): Usato per contesto, ma i dati sono già filtrati fino a questa data.
        Returns:
            dict: Un dizionario con le statistiche calcolate: {"RA", "RSMax", "Freq"}.
        """
        stats = {"RA": "N/D", "RSMax": 0, "Freq": 0}
        
        if not lunghetta_da_analizzare or len(lunghetta_da_analizzare) < sorte_len:
            return stats

        target_combinations_sets = [set(combo) for combo in combinations(sorted(lunghetta_da_analizzare), sorte_len)]
        if not target_combinations_sets:
            return stats

        all_processed_historic_extractions = []
        if processed_historical_dfs:
            for df_hist in processed_historical_dfs.values():
                for _, row in df_hist.iterrows():
                    try:
                        numeri_estratti_dict = {int(row[f'N{i}']) for i in range(1, 6) if pd.notna(row[f'N{i}'])}
                        if len(numeri_estratti_dict) == 5:
                             all_processed_historic_extractions.append({
                                "data": row['Data'],
                                "numeri": numeri_estratti_dict
                            })
                    except (ValueError, KeyError):
                        continue 
        
        if not all_processed_historic_extractions:
            stats["RSMax"] = "N/D (no dati storici filtrati)"
            stats["Freq"] = "N/D (no dati storici filtrati)"
            stats["RA"] = "N/D (no dati storici filtrati)"
            return stats

        # Ordina per calcolo RSMax, Freq (dal più vecchio al più recente)
        all_processed_historic_extractions_sorted_asc = sorted(all_processed_historic_extractions, key=lambda x: x['data'])
        
        current_rs = 0
        max_rs = 0
        freq_count = 0

        for extraction in all_processed_historic_extractions_sorted_asc:
            found_in_extraction_hist = False
            for target_combo_set in target_combinations_sets:
                if target_combo_set.issubset(extraction['numeri']):
                    found_in_extraction_hist = True
                    break
            if found_in_extraction_hist:
                freq_count += 1
                if current_rs > max_rs: max_rs = current_rs
                current_rs = 0 
            else:
                current_rs += 1
        if current_rs > max_rs: max_rs = current_rs
        stats["RSMax"] = max_rs
        stats["Freq"] = freq_count

        # --- Calcolo Ritardo Attuale (RA) sui dati storici filtrati ---
        # Ordina dalla più recente alla più vecchia per il RA
        all_processed_historic_extractions_sorted_desc = sorted(all_processed_historic_extractions, key=lambda x: x['data'], reverse=True)
        
        ra_count = 0
        found_for_ra = False
        for extraction in all_processed_historic_extractions_sorted_desc: 
            is_present_in_this_extraction = False
            for target_combo_set in target_combinations_sets:
                if target_combo_set.issubset(extraction['numeri']):
                    is_present_in_this_extraction = True
                    break 
            if is_present_in_this_extraction:
                found_for_ra = True
                break 
            else:
                ra_count += 1
        
        if found_for_ra:
            stats["RA"] = ra_count
        else:
            # Se non è mai uscita in TUTTI i dati storici filtrati, il RA è la loro dimensione totale
            stats["RA"] = len(all_processed_historic_extractions_sorted_desc) 
        
        return stats
    
    def fetch_data_from_github(self, url):
        # ... (il tuo codice fetch_data_from_github) ...
        try:
            response = requests.get(url); response.raise_for_status(); return response.text
        except requests.exceptions.RequestException as e:
            self.q.put(f"Errore download da {url}: {e}"); return None

    def parse_lotto_data(self, content, wheel_name):
        # ... (il tuo codice parse_lotto_data) ...
        if not content: return pd.DataFrame()
        try:
            df = pd.read_csv(io.StringIO(content), sep='\s+', header=None, names=['Data', 'Ruota'] + [f'N{i}' for i in range(1, 6)], engine='python')
            def parse_date_custom(date_str):
                try: return datetime.strptime(date_str, '%Y/%m/%d')
                except ValueError: pass
                for fmt in ('%Y.%m.%d', '%d/%m/%Y', '%Y-%m-%d'):
                    try: return datetime.strptime(date_str, fmt)
                    except ValueError: continue
                self.q.put(f"WARN: Formato data non riconosciuto '{date_str}' per {wheel_name}. Riga ignorata."); return pd.NaT
            df['Data'] = df['Data'].apply(parse_date_custom)
            df.dropna(subset=['Data'], inplace=True)
            for col in [f'N{i}' for i in range(1, 6)]: df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[f'N{i}' for i in range(1, 6)], inplace=True)
            for col in [f'N{i}' for i in range(1, 6)]: df[col] = df[col].astype(int)
            df['Ruota'] = df['Ruota'].astype(str).str.upper()
            return df.sort_values(by='Data')
        except Exception as e:
            self.q.put(f"Errore parsing dati per {wheel_name}: {e}"); return pd.DataFrame()


    def start_analysis(self): # Modificato per resettare le statistiche
        if self.analysis_running: messagebox.showwarning("Attenzione", "Un'analisi è già in corso."); return
        self.results_text.delete(1.0, tk.END); self.progress_bar["value"] = 0
        
        self.show_chart_button.config(state=tk.DISABLED) 
        self.last_chart_data = None 
        if self.chart_window and self.chart_window.winfo_exists(): self.update_frequency_chart(None, self.current_chart_type)
        
        self.show_stats_button.config(state=tk.DISABLED) # Disabilita pulsante statistiche
        self.last_stats_data = None # Pulisci dati statistiche precedenti
        if self.stats_window and self.stats_window.winfo_exists(): self.display_advanced_stats(None) # Pulisci finestra statistiche

        try:
            start_date_hist = datetime.combine(self.start_date_entry.get_date(), datetime.min.time())
            end_date_hist = datetime.combine(self.end_date_entry.get_date(), datetime.max.time())
            if start_date_hist >= end_date_hist: messagebox.showerror("Errore Date", "Data inizio deve precedere data fine."); return
            selected_ruote_analisi = [r for r, v in self.ruote_analisi_vars.items() if v.get()]
            if not selected_ruote_analisi: messagebox.showerror("Errore", "Selezionare ruote per analisi frequenze."); return
            indice_mese_str = self.indice_mese_combo.get()
            extraction_index_in_month = int(indice_mese_str.split("ª")[0]) if indice_mese_str != "Tutte" else None
            selected_months = [m_num for m_num, var in self.mesi_vars.items() if var.get()]
            lunghetta_len = int(self.lunghetta_len_combo.get())
            sorte_text = self.sorte_combo.get()
            sorte_len = {"Ambo": 2, "Terno": 3, "Quaterna": 4}[sorte_text]
            if lunghetta_len < sorte_len: messagebox.showerror("Errore", f"Lunghetta ({lunghetta_len}) < sorte ({sorte_text}={sorte_len})."); return
            colpi_gioco = int(self.colpi_spinbox.get())
            selected_ruote_gioco = [r for r, v in self.ruote_gioco_vars.items() if v.get()]
            self.analysis_running = True; self.start_button.config(state=tk.DISABLED)
            self.log_message("Avvio analisi...")
            analysis_params = {"start_date_hist": start_date_hist, "end_date_hist": end_date_hist, "selected_ruote_analisi": selected_ruote_analisi, "extraction_index_in_month": extraction_index_in_month, "selected_months": selected_months, "lunghetta_len": lunghetta_len, "sorte_len": sorte_len, "sorte_text": sorte_text, "colpi_gioco": colpi_gioco, "selected_ruote_gioco": selected_ruote_gioco}
            Thread(target=self._perform_analysis_thread, args=(analysis_params,), daemon=True).start()
        except ValueError as e: messagebox.showerror("Errore Input", f"Errore parametri: {e}"); self.analysis_running = False; self.start_button.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Errore", f"Errore non gestito: {e}\n{traceback.format_exc()}"); self.analysis_running = False; self.start_button.config(state=tk.NORMAL)


    def _perform_analysis_thread(self, params):
        try:
            self.q.put(("clear_chart_data", None, "initial_clear"))
            self.q.put(("clear_advanced_stats", None)) 

            # --- FASE 1: Caricamento Dati Complessivo, Filtro Periodo Base, Filtri Avanzati ---
            self.q.put("--- FASE 1: Caricamento Dati Complessivo ---")
            all_data_loaded = {} 
            all_wheels_to_load = list(set(params["selected_ruote_analisi"] + params["selected_ruote_gioco"]))
            total_wheels_to_load = len(all_wheels_to_load)
            processed_wheels = 0
            if not all_wheels_to_load: self.q.put("Nessuna ruota selezionata. Analisi interrotta."); self.q.put("analysis_complete"); return
            for wheel_key in all_wheels_to_load:
                self.q.put(f"Caricamento dati per {RUOTE_DISPONIBILI[wheel_key]}...")
                df_full = self.data_cache.get(wheel_key)
                if df_full is None or df_full.empty:
                    content = self.fetch_data_from_github(URL_RUOTE[wheel_key])
                    if content: df_full = self.parse_lotto_data(content, wheel_key)
                    if df_full is not None and not df_full.empty: self.data_cache[wheel_key] = df_full.copy(); self.q.put(f"Dati per {RUOTE_DISPONIBILI[wheel_key]} scaricati.")
                    else: self.q.put(f"Nessun dato valido per {RUOTE_DISPONIBILI[wheel_key]}.")
                else: self.q.put(f"Dati per {RUOTE_DISPONIBILI[wheel_key]} caricati dalla cache.")
                if df_full is not None and not df_full.empty: all_data_loaded[wheel_key] = df_full
                else: self.q.put(f"Attenzione: {RUOTE_DISPONIBILI[wheel_key]} esclusa per mancanza dati.")
                processed_wheels += 1
                self.q.put(("progress", processed_wheels * (30 / total_wheels_to_load if total_wheels_to_load > 0 else 0)))
            if not all_data_loaded: self.q.put("Nessun dato caricato. Analisi interrotta."); self.q.put("analysis_complete"); return
            
            self.q.put("\n--- APPLICAZIONE FILTRO PERIODO STORICO BASE ---")
            historical_data_dfs_initial_period = {}
            ruote_analisi_effettive_f1 = [] # Usiamo un nome diverso per chiarezza
            for k_f1 in params["selected_ruote_analisi"]:
                if k_f1 in all_data_loaded:
                    df_temp_f1 = all_data_loaded[k_f1][(all_data_loaded[k_f1]['Data'] >= params["start_date_hist"]) & (all_data_loaded[k_f1]['Data'] <= params["end_date_hist"])]
                    if not df_temp_f1.empty:
                        historical_data_dfs_initial_period[k_f1] = df_temp_f1
                        ruote_analisi_effettive_f1.append(k_f1)
            
            if not historical_data_dfs_initial_period: self.q.put("Nessun dato storico nel periodo base. Analisi interrotta."); self.q.put("analysis_complete"); return
            self.q.put(("progress", 35))

            self.q.put("\n--- APPLICAZIONE FILTRI TEMPORALI AVANZATI ---")
            processed_historical_data_dfs = {}
            for wheel_key_f1_adv, df_original_f1_adv in historical_data_dfs_initial_period.items(): # Iteriamo solo sulle ruote che hanno passato il filtro periodo
                df_to_filter_f1_adv = df_original_f1_adv.copy()
                if params["selected_months"]:
                    df_to_filter_f1_adv = df_to_filter_f1_adv[df_to_filter_f1_adv['Data'].dt.month.isin(params["selected_months"])]
                if df_to_filter_f1_adv.empty: continue
                
                if params["extraction_index_in_month"]:
                    df_to_filter_f1_adv = df_to_filter_f1_adv.sort_values(by='Data') # Assicura ordinamento per cumcount
                    # Usare .loc per evitare SettingWithCopyWarning, anche se qui il copy() precedente dovrebbe aver aiutato
                    df_to_filter_f1_adv.loc[:, 'AnnoMeseGroup'] = df_to_filter_f1_adv['Data'].dt.to_period('M')
                    df_to_filter_f1_adv.loc[:, 'IndiceNelMese'] = df_to_filter_f1_adv.groupby('AnnoMeseGroup')['Data'].cumcount() + 1
                    df_to_filter_f1_adv = df_to_filter_f1_adv[df_to_filter_f1_adv['IndiceNelMese'] == params["extraction_index_in_month"]]
                    df_to_filter_f1_adv = df_to_filter_f1_adv.drop(columns=['AnnoMeseGroup', 'IndiceNelMese'], errors='ignore')
                
                if not df_to_filter_f1_adv.empty:
                    processed_historical_data_dfs[wheel_key_f1_adv] = df_to_filter_f1_adv
                    self.q.put(f"Dati filtrati per {RUOTE_DISPONIBILI[wheel_key_f1_adv]}: {len(processed_historical_data_dfs[wheel_key_f1_adv])} estrazioni.")
            
            if not processed_historical_data_dfs: self.q.put("Nessun dato storico dopo filtri avanzati. Analisi interrotta."); self.q.put("analysis_complete"); return
            self.q.put(("progress", 40))

            # --- FASE 2 con invio dati al grafico e dettagli sfaldamenti ---
            self.q.put(f"\n--- FASE 2: Identificazione Lunghette Candidate Storiche e Loro Backtesting ---")
            lunghezza_serie_da_cercare = params["lunghetta_len"]
            top_base_lunghettas_to_test_tuples = []
            if lunghezza_serie_da_cercare <= 5:
                self.q.put(f"Ricerca combinazioni di {lunghezza_serie_da_cercare} numeri.")
                possible_base_lunghettas_counter = Counter()
                num_estrazioni_per_conteggio_base = 0
                if processed_historical_data_dfs:
                    for df_hist_data_base in processed_historical_data_dfs.values():
                        num_estrazioni_per_conteggio_base += len(df_hist_data_base)
                        for _, row_base in df_hist_data_base.iterrows():
                            try:
                                extracted_numbers_base = sorted([int(row_base[f'N{i}']) for i in range(1, 6)])
                                if len(extracted_numbers_base) == 5: # Assicura che ci siano 5 numeri validi
                                    for combo_base in combinations(extracted_numbers_base, lunghezza_serie_da_cercare):
                                        possible_base_lunghettas_counter[combo_base] += 1
                            except (ValueError, KeyError): continue 
                top_base_lunghettas_to_test_tuples = possible_base_lunghettas_counter.most_common(20) # Prendi le top 20 per il backtest
                self.q.put(f"Conteggio lunghette base ({lunghezza_serie_da_cercare} numeri) completato su {num_estrazioni_per_conteggio_base} estrazioni filtrate.")
                self.q.put(f"Trovate {len(possible_base_lunghettas_counter)} lunghette base uniche. Selezionate le top {len(top_base_lunghettas_to_test_tuples)} per il backtesting.")
            else: # > 5
                self.q.put(f"Modalità lunghetta > 5 ({lunghezza_serie_da_cercare} numeri): si identificano i numeri singoli più frequenti.")
                single_number_counter = Counter()
                if processed_historical_data_dfs:
                    for df_hist_data_base in processed_historical_data_dfs.values():
                        for _, row_base in df_hist_data_base.iterrows():
                            try:
                                for i in range(1, 6): single_number_counter[int(row_base[f'N{i}'])] += 1
                            except (ValueError, KeyError): continue
                if not single_number_counter:
                    self.q.put("Nessun numero trovato per calcolare la frequenza singola. Analisi interrotta.")
                    self.q.put(("clear_chart_data", None, "single_numbers"))
                    self.q.put("analysis_complete"); return
                top_single_numbers_with_freq = single_number_counter.most_common(lunghezza_serie_da_cercare)
                if top_single_numbers_with_freq:
                    data_per_grafico_singoli = [(num, freq) for num, freq in top_single_numbers_with_freq]
                    self.q.put(("update_chart_data", data_per_grafico_singoli, "single_numbers"))
                else: self.q.put(("clear_chart_data", None, "single_numbers"))
                derived_lunghetta_list = sorted([num for num, freq in top_single_numbers_with_freq])
                if len(derived_lunghetta_list) < lunghezza_serie_da_cercare: self.q.put(f"Attenzione: Trovati solo {len(derived_lunghetta_list)} numeri unici, meno dei {lunghezza_serie_da_cercare} richiesti.")
                if not derived_lunghetta_list or len(derived_lunghetta_list) < params['sorte_len']: self.q.put(f"Numeri ({len(derived_lunghetta_list)}) insuff. per sorte ({params['sorte_len']}). Analisi interrotta."); self.q.put("analysis_complete"); return
                aggregated_freq = sum(freq for num, freq in top_single_numbers_with_freq if num in derived_lunghetta_list)
                top_base_lunghettas_to_test_tuples = [(tuple(derived_lunghetta_list), aggregated_freq)]
                self.q.put(f"Lunghetta derivata: {derived_lunghetta_list} (Freq. aggregata: {aggregated_freq})")
            
            if not top_base_lunghettas_to_test_tuples: self.q.put("Nessuna lunghetta da testare. Analisi interrotta."); self.q.put("analysis_complete"); return
            self.q.put(("progress", 45))

            all_backtest_results = []
            # Iterare su processed_historical_data_dfs.items() per avere wheel_key_trigger
            for i_base, (base_lunghetta_tuple, freq_base) in enumerate(top_base_lunghettas_to_test_tuples):
                base_lunghetta = list(base_lunghetta_tuple)
                if len(base_lunghetta) < params['sorte_len']: continue
                candidate_combinations_for_this_base = list(combinations(sorted(base_lunghetta), params['sorte_len']))
                if not candidate_combinations_for_this_base: continue
                
                bt_triggers_con_successo_per_lb, bt_giocate_effettuate_per_lb = 0, 0
                bt_date_successi_per_lb = [] # Lista per raccogliere le date dei successi per questa lunghetta

                # Ciclo corretto per avere wheel_key_trigger_analisi
                for wheel_key_trigger_analisi, df_triggers_spec_ruota_analisi in processed_historical_data_dfs.items():
                    for _, trigger_row_bt_event in df_triggers_spec_ruota_analisi.sort_values(by='Data').iterrows():
                        data_trigger_bt_event = trigger_row_bt_event['Data']
                        if data_trigger_bt_event + timedelta(days=params['colpi_gioco']*4) > params['end_date_hist']: continue 
                        
                        almeno_una_giocata_possibile_per_trigger, successo_per_trigger_su_qualsiasi_ruota_gioco = False, False
                        for wheel_key_gioco_bt_verify in params["selected_ruote_gioco"]:
                            df_completo_ruota_gioco_bt_verify = all_data_loaded.get(wheel_key_gioco_bt_verify)
                            if df_completo_ruota_gioco_bt_verify is None: continue
                            
                            start_verifica_bt_verify = data_trigger_bt_event + timedelta(days=1)
                            df_finestra_bt_verify = df_completo_ruota_gioco_bt_verify[
                                (df_completo_ruota_gioco_bt_verify['Data'] >= start_verifica_bt_verify) & 
                                (df_completo_ruota_gioco_bt_verify['Data'] <= params['end_date_hist'])
                            ].head(params['colpi_gioco'])
                            
                            if df_finestra_bt_verify.empty: continue
                            almeno_una_giocata_possibile_per_trigger = True

                            for combo_bt_verify in candidate_combinations_for_this_base:
                                set_combo_bt_verify = set(combo_bt_verify)
                                for colpo_idx_bt_v_loop, (_idx_iter_bt, row_bt_colpo_verify_series) in enumerate(df_finestra_bt_verify.iterrows()):
                                    try:
                                        numeri_estrazione_bt_verify = {int(row_bt_colpo_verify_series[f'N{k}']) for k in range(1, 6)}
                                        if set_combo_bt_verify.issubset(numeri_estrazione_bt_verify):
                                            msg_dettaglio_bt = (
                                                f"  DETTAGLIO BACKTEST (FASE 2): Lunghetta {sorted(base_lunghetta)} -> Sorte {params['sorte_text']} {sorted(list(combo_bt_verify))} "
                                                f"uscita su {RUOTE_DISPONIBILI[wheel_key_gioco_bt_verify]} il {row_bt_colpo_verify_series['Data'].strftime('%Y-%m-%d')} "
                                                f"(Colpo {colpo_idx_bt_v_loop + 1} dopo trigger del {data_trigger_bt_event.strftime('%Y-%m-%d')} "
                                                f"su {RUOTE_DISPONIBILI[wheel_key_trigger_analisi]})" 
                                            )
                                            self.q.put(msg_dettaglio_bt)
                                            bt_date_successi_per_lb.append(row_bt_colpo_verify_series['Data'])
                                            successo_per_trigger_su_qualsiasi_ruota_gioco = True; break
                                    except (ValueError, KeyError): continue
                                if successo_per_trigger_su_qualsiasi_ruota_gioco: break
                            if successo_per_trigger_su_qualsiasi_ruota_gioco: break 
                        
                        if almeno_una_giocata_possibile_per_trigger:
                            bt_giocate_effettuate_per_lb += 1
                            if successo_per_trigger_su_qualsiasi_ruota_gioco: 
                                bt_triggers_con_successo_per_lb += 1
                
                percent_successo_bt_lb = (bt_triggers_con_successo_per_lb / bt_giocate_effettuate_per_lb) * 100 if bt_giocate_effettuate_per_lb > 0 else 0
                all_backtest_results.append({
                    "lunghetta": sorted(base_lunghetta), 
                    "success_triggers": bt_triggers_con_successo_per_lb, 
                    "plays_made": bt_giocate_effettuate_per_lb, 
                    "percent_success": percent_successo_bt_lb, 
                    "original_freq": freq_base,
                    "date_successi_bt": bt_date_successi_per_lb # Aggiunta lista date successi
                })
                self.q.put(("progress", 45 + (i_base + 1) * (40 / len(top_base_lunghettas_to_test_tuples) if top_base_lunghettas_to_test_tuples else 0) ))
            
            # --- INVIO DATI PER GRAFICO BASATO SUI RISULTATI DEL BACKTEST ---
            if lunghezza_serie_da_cercare <= 5: # Mostra grafico performance backtest solo per <=5
                if all_backtest_results:
                    chart_data_backtest = sorted(all_backtest_results, key=lambda x: x.get("success_triggers", 0), reverse=True)
                    # Prendi fino a 15 risultati per il grafico
                    data_to_send_to_chart = [(str(tuple(res_bt['lunghetta'])).replace(' ', ''), res_bt['success_triggers']) for res_bt in chart_data_backtest[:15]]
                    if data_to_send_to_chart: self.q.put(("update_chart_data", data_to_send_to_chart, "backtest_performance"))
                    else: self.q.put(("clear_chart_data", None, "backtest_performance")) # Pulisci se non ci sono dati
                else: self.q.put(("clear_chart_data", None, "backtest_performance")) # Pulisci se no risultati backtest
            # Se > 5, i dati dei numeri singoli sono già stati inviati al grafico in precedenza.

            self.q.put("\n--- FASE 3: Riepilogo Lunghette con Migliore Performance Storica (Backtest) ---")
            sorted_results = sorted(all_backtest_results, key=lambda x: (x.get("success_triggers", 0), x.get("percent_success", 0), x.get("original_freq", 0)), reverse=True)
            final_lunghetta_scelta_per_gioco = None
            dati_prima_lunghetta_backtest = {} 
            
            if sorted_results:
                freq_label = "Freq. grezza" if lunghezza_serie_da_cercare <= 5 else "Freq. aggregata"
                if lunghezza_serie_da_cercare <= 5:
                    self.q.put("Top lunghette candidate ordinate per performance nel backtest:")
                    for i_res, res in enumerate(sorted_results[:10]): # Mostra top 10
                        anni_successi_str = ""
                        if res.get("date_successi_bt"):
                            anni_unici = sorted(list(set(data_succ.year for data_succ in res["date_successi_bt"])))
                            if anni_unici: anni_successi_str = f" (Anni BT: {', '.join(map(str, anni_unici))})"
                        self.q.put(f"  Pos. {i_res+1}: LUNGHETTA {res['lunghetta']} -> {res['success_triggers']} successi su {res['plays_made']} giocate ({res['percent_success']:.2f}%)"
                                   f"{anni_successi_str} ({freq_label}: {res['original_freq']})")
                else: # Caso lunghetta > 5 (solo una derivata)
                    res = sorted_results[0]
                    anni_successi_str = ""
                    if res.get("date_successi_bt"):
                        anni_unici = sorted(list(set(data_succ.year for data_succ in res["date_successi_bt"])))
                        if anni_unici: anni_successi_str = f" (Anni BT: {', '.join(map(str, anni_unici))})"
                    self.q.put(f"Performance lunghetta derivata ({res['lunghetta']}): {res['success_triggers']} successi su {res['plays_made']} giocate ({res['percent_success']:.2f}%)"
                               f"{anni_successi_str} ({freq_label}: {res['original_freq']})")
                
                final_lunghetta_scelta_per_gioco = sorted_results[0]['lunghetta']
                dati_prima_lunghetta_backtest = sorted_results[0] 
                self.q.put(f"\nLunghetta Selezionata per Gioco Futuro (FASE 4): {final_lunghetta_scelta_per_gioco} ({dati_prima_lunghetta_backtest.get('success_triggers','N/A')} successi su {dati_prima_lunghetta_backtest.get('plays_made','N/A')} giocate nel BT)")
            else: 
                self.q.put("Nessun risultato di backtest. Analisi interrotta."); self.q.put("analysis_complete"); return
            
            if not final_lunghetta_scelta_per_gioco: self.q.put("Impossibile selezionare una lunghetta. Analisi interrotta."); self.q.put("analysis_complete"); return
            if len(final_lunghetta_scelta_per_gioco) < params['sorte_len']: self.q.put(f"Errore: lunghetta finale ({len(final_lunghetta_scelta_per_gioco)}) troppo corta per sorte ({params['sorte_len']}). Analisi interrotta."); self.q.put("analysis_complete"); return
            
            candidate_combinations_final_for_future = list(combinations(final_lunghetta_scelta_per_gioco, params['sorte_len']))
            if not candidate_combinations_final_for_future: self.q.put(f"Nessuna combinazione generabile dalla lunghetta finale. Analisi interrotta."); self.q.put("analysis_complete"); return
            self.q.put(f"Generate {len(candidate_combinations_final_for_future)} {params['sorte_text']} da {final_lunghetta_scelta_per_gioco} per verifica futura.")
            
            # --- CALCOLO E INVIO STATISTICHE AVANZATE PER LE TOP LUNGHETTE ---
            self.q.put("\n--- Calcolo Statistiche Avanzate per le Top Lunghette ---")
            num_top_lunghettas_for_stats = 5 
            all_stats_payload = [] 
            items_to_calculate_stats_for = []
            if lunghezza_serie_da_cercare <= 5:
                items_to_calculate_stats_for = sorted_results[:num_top_lunghettas_for_stats]
            elif final_lunghetta_scelta_per_gioco and dati_prima_lunghetta_backtest: 
                items_to_calculate_stats_for = [dati_prima_lunghetta_backtest] 
            
            if not items_to_calculate_stats_for:
                self.q.put("Nessuna lunghetta valida per calcolare statistiche avanzate.")
                self.q.put(("clear_advanced_stats", None))
            else:
                for i_stat, backtest_result_item_stat in enumerate(items_to_calculate_stats_for):
                    current_lunghetta_stat = backtest_result_item_stat.get('lunghetta')
                    if not current_lunghetta_stat: continue
                    self.q.put(f"Calcolo stats per lunghetta {i_stat+1}: {current_lunghetta_stat}...")
                    calculated_stats_adv = self._calculate_stats_for_lunghetta(
                        lunghetta_da_analizzare=current_lunghetta_stat,
                        sorte_len=params['sorte_len'],
                        processed_historical_dfs=processed_historical_data_dfs, # Usa i dati filtrati
                        current_end_date_hist=params['end_date_hist'] 
                    )
                    stats_entry_adv = {
                        "lunghetta": current_lunghetta_stat,
                        "sorte": params['sorte_text'],
                        "RA": calculated_stats_adv.get('RA', "Errore"),
                        "RSMax": calculated_stats_adv.get('RSMax', "Errore"),
                        "Freq": calculated_stats_adv.get('Freq', "Errore"),
                        "success_triggers_backtest": backtest_result_item_stat.get('success_triggers', 'N/A'),
                        "plays_made_backtest": backtest_result_item_stat.get('plays_made', 'N/A'),
                        "percent_success_backtest": backtest_result_item_stat.get('percent_success', 'N/A')
                    }
                    all_stats_payload.append(stats_entry_adv)
                if all_stats_payload:
                    self.q.put(("update_advanced_stats", all_stats_payload))
                    self.q.put("Statistiche avanzate calcolate per le top lunghette.")
                else: 
                    self.q.put("Nessun dato di statistica avanzata da mostrare.")
                    self.q.put(("clear_advanced_stats", None))
            self.q.put(("progress", 85))

            # --- FASE 4: Verifica Colpi (Futuro) ---
            self.q.put(f"\n--- FASE 4: Verifica Giocate per {params['colpi_gioco']} Colpi (Futuro, trigger filtrati) ---")
            start_date_for_future_logic_f4 = params['end_date_hist'].date() + timedelta(days=1)
            future_data_start_date_f4 = datetime.combine(start_date_for_future_logic_f4, datetime.min.time())
            found_predictions_f4_details = [] 
            
            if not all_data_loaded: self.q.put("ERRORE CRITICO FASE 4: all_data_loaded non disponibile."); self.q.put("analysis_complete"); return
            if not params["selected_ruote_gioco"]: self.q.put("ATTENZIONE: Nessuna ruota di gioco selezionata per la FASE 4!"); # Non interrompere, ma logga

            future_trigger_events_f4 = []
            self.q.put(f"Identificazione eventi trigger futuri (post {future_data_start_date_f4.strftime('%Y-%m-%d')})...")
            for wheel_key_trg_f4 in params["selected_ruote_analisi"]: # Usa ruote di analisi per i trigger
                if wheel_key_trg_f4 not in all_data_loaded: continue
                df_trg_orig_f4 = all_data_loaded[wheel_key_trg_f4]
                df_trg_f4 = df_trg_orig_f4[df_trg_orig_f4['Data'] >= future_data_start_date_f4].copy() # Filtra per data futura
                if df_trg_f4.empty: continue
                
                # Applica filtri temporali avanzati ai dati trigger futuri
                if params["selected_months"]: 
                    df_trg_f4 = df_trg_f4[df_trg_f4['Data'].dt.month.isin(params["selected_months"])]
                if df_trg_f4.empty: continue
                
                if params["extraction_index_in_month"]:
                    df_trg_f4 = df_trg_f4.sort_values(by='Data')
                    df_trg_f4.loc[:, 'AnnoMeseGroup_f4'] = df_trg_f4['Data'].dt.to_period('M')
                    df_trg_f4.loc[:, 'IndiceNelMese_f4'] = df_trg_f4.groupby('AnnoMeseGroup_f4')['Data'].cumcount() + 1
                    df_trg_f4 = df_trg_f4[df_trg_f4['IndiceNelMese_f4'] == params["extraction_index_in_month"]]
                    df_trg_f4 = df_trg_f4.drop(columns=['AnnoMeseGroup_f4', 'IndiceNelMese_f4'], errors='ignore')
                
                for _idx_trg_iter_f4, r_trg_series_f4 in df_trg_f4.iterrows():
                    future_trigger_events_f4.append({"data_trigger": r_trg_series_f4['Data'], "ruota_trigger_nome": RUOTE_DISPONIBILI[wheel_key_trg_f4]})
            
            if not future_trigger_events_f4: 
                self.q.put("Nessun evento trigger futuro trovato dopo l'applicazione dei filtri.")
            else:
                future_trigger_events_f4.sort(key=lambda x: x['data_trigger'])
                self.q.put(f"Trovati {len(future_trigger_events_f4)} eventi trigger futuri filtrati.")
                num_triggers_processed_f4, max_triggers_to_process_f4 = 0, len(future_trigger_events_f4)
                
                for trigger_event_f4_item in future_trigger_events_f4:
                    play_start_dt_f4 = trigger_event_f4_item['data_trigger'] + timedelta(days=1)
                    self.q.put(f"\n  Verifica per trigger {trigger_event_f4_item['data_trigger'].strftime('%Y-%m-%d')} su {trigger_event_f4_item['ruota_trigger_nome']}:")
                    
                    if not params["selected_ruote_gioco"]: 
                        self.q.put("    Nessuna ruota di gioco selezionata per la verifica."); continue # Salta questo trigger se non ci sono ruote di gioco
                    
                    for combo_play_f4_item in candidate_combinations_final_for_future:
                        success_this_combo_f4_flag = False
                        for wheel_play_f4_key_item in params["selected_ruote_gioco"]:
                            if wheel_play_f4_key_item not in all_data_loaded: continue
                            df_play_f4_full_item = all_data_loaded[wheel_play_f4_key_item]
                            # Prendi i dati per la finestra di gioco, non superare la data più recente disponibile nel dataset
                            df_play_window_f4_item = df_play_f4_full_item[df_play_f4_full_item['Data'] >= play_start_dt_f4].head(params["colpi_gioco"])
                            if df_play_window_f4_item.empty: continue
                            
                            for colpo_idx_f4_loop, (_idx_iter_f4_item, r_play_f4_series_item) in enumerate(df_play_window_f4_item.iterrows()):
                                try:
                                    numeri_estrazione_f4 = {int(r_play_f4_series_item[f'N{k_f4}']) for k_f4 in range(1,6)}
                                    if set(combo_play_f4_item).issubset(numeri_estrazione_f4):
                                        msg_f4_success = (f"    SUCCESSO FASE4! {params['sorte_text']} {sorted(list(combo_play_f4_item))} su {RUOTE_DISPONIBILI[wheel_play_f4_key_item]} "
                                                          f"il {r_play_f4_series_item['Data'].strftime('%Y-%m-%d')} (Colpo {colpo_idx_f4_loop+1} dopo trigger "
                                                          f"{trigger_event_f4_item['data_trigger'].strftime('%Y-%m-%d')} su {trigger_event_f4_item['ruota_trigger_nome']})")
                                        found_predictions_f4_details.append({"data_oggetto": r_play_f4_series_item['Data'], "messaggio_stringa": msg_f4_success})
                                        success_this_combo_f4_flag = True; break
                                except (ValueError, KeyError): continue
                            if success_this_combo_f4_flag: break 
                    num_triggers_processed_f4 += 1
                    progress_val_f4_calc = 85 + ( (num_triggers_processed_f4 / max_triggers_to_process_f4) * 14 if max_triggers_to_process_f4 > 0 else 0)
                    self.q.put(("progress", min(progress_val_f4_calc, 99) )) # Limita a 99 prima del finally
            
            # Riepilogo FASE 4
            if not params["selected_ruote_gioco"] and future_trigger_events_f4: 
                self.q.put(f"\nNessuna ruota di gioco è stata selezionata, quindi impossibile verificare successi per i trigger futuri.")
            elif not found_predictions_f4_details:
                if future_trigger_events_f4: # Solo se c'erano trigger da processare
                    self.q.put(f"\nNessuna delle combinazioni selezionate è uscita dopo gli eventi trigger futuri filtrati (FASE 4).")
                # Se non c'erano future_trigger_events_f4, il messaggio è già stato dato
            else:
                found_predictions_f4_sorted = sorted(found_predictions_f4_details, key=lambda x: x['data_oggetto'])
                self.q.put(f"\n--- Riepilogo Successi (FASE 4) ({len(found_predictions_f4_sorted)} trovati, ordinati per data) ---")
                for p_f4_item in found_predictions_f4_sorted: self.q.put((p_f4_item['messaggio_stringa'], True))

        except Exception as e:
            self.q.put(f"Errore critico durante l'analisi: {e}\n{traceback.format_exc()}")
            self.q.put(("clear_chart_data", None, "error_occurred")) 
            self.q.put(("clear_advanced_stats", None)) 
        finally:
            self.q.put(("progress", 100))
            self.q.put("analysis_complete")

    def process_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    if item[0] == "progress": self.progress_bar["value"] = item[1]
                    elif item[0] == "update_chart_data":
                        _, chart_data, chart_type = item
                        self.last_chart_data = chart_data
                        self.current_chart_type = chart_type
                        self.show_chart_button.config(state=tk.NORMAL)
                        if self.chart_window and self.chart_window.winfo_exists():
                            self.update_frequency_chart(self.last_chart_data, self.current_chart_type)
                    elif item[0] == "clear_chart_data":
                        self.last_chart_data = None
                        self.current_chart_type = item[2] if len(item) > 2 else "single_numbers"
                        if self.chart_window and self.chart_window.winfo_exists():
                            self.update_frequency_chart(None, self.current_chart_type)
                    # --- NUOVA GESTIONE PER STATISTICHE ---
                    elif item[0] == "update_advanced_stats":
                        _, stats_data = item
                        self.last_stats_data = stats_data
                        self.show_stats_button.config(state=tk.NORMAL) # Abilita pulsante
                        if self.stats_window and self.stats_window.winfo_exists():
                            self.display_advanced_stats(self.last_stats_data)
                    elif item[0] == "clear_advanced_stats":
                        self.last_stats_data = None
                        # self.show_stats_button.config(state=tk.DISABLED) # Opzionale: disabilita
                        if self.stats_window and self.stats_window.winfo_exists():
                            self.display_advanced_stats(None) # Pulisce la finestra stats
                    # --- FINE GESTIONE STATISTICHE ---
                    else: self.log_message(item[0], is_result=item[1] if len(item) > 1 else False)
                elif item == "analysis_complete":
                    self.log_message("\nAnalisi completata.")
                    self.progress_bar["value"] = 100
                    self.analysis_running = False
                    self.start_button.config(state=tk.NORMAL)
                    if self.last_chart_data: self.show_chart_button.config(state=tk.NORMAL)
                    if self.last_stats_data: self.show_stats_button.config(state=tk.NORMAL) # Abilita se ci sono dati stats
                else: self.log_message(str(item))
        except queue.Empty: pass
        finally: self.master.after(100, self.process_queue)

# --- Blocco Main ---
if __name__ == '__main__':
    root = tk.Tk()
    app = LottoApp(root)
    root.mainloop()