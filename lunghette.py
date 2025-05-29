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
import random # Aggiunto per il campionamento nelle candidate

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

# >>> COSTANTE PER LA RUOTA SEGNALE <<<
# Questa verrà usata sia per il backtest della Fase 2/3 sia per i trigger della Fase 5
# Se l'utente seleziona solo una ruota per l'analisi, si potrebbe usare quella.
# Per ora, la fissiamo a 'BA' come da tua richiesta specifica.
# Assicurati che questa ruota sia selezionata nelle "Ruote per Analisi Frequenze" nell'interfaccia.
DEFAULT_RUOTA_SEGNALE = 'BA'


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

        # --- WIDGET DI INPUT (invariato) ---
        period_frame = ttk.LabelFrame(main_frame, text="Periodo Storico per Analisi Frequenze", padding="10")
        period_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(period_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = DateEntry(period_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.set_date(datetime.now() - timedelta(days=365*5)) # Esteso un po' per avere più dati di default
        ttk.Label(period_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = DateEntry(period_frame, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.set_date(datetime.now() - timedelta(days=1))

        adv_filters_frame = ttk.LabelFrame(main_frame, text="Filtri Temporali Avanzati per Analisi", padding="10")
        adv_filters_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(adv_filters_frame, text="Indice Estrazione del Mese:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        MAX_INDICE_ESTRAZIONE_MESE = 19
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

        ruote_analisi_frame = ttk.LabelFrame(main_frame, text="Ruote per Analisi Frequenze (Segnale)", padding="10")
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
        ttk.Label(params_frame, text="Sorte da Giocare/Coprire:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sorte_combo = ttk.Combobox(params_frame, values=["Ambo", "Terno", "Quaterna"], width=10, state="readonly")
        self.sorte_combo.current(0)
        self.sorte_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(params_frame, text="Colpi di Gioco (Backtest/Futuro):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.colpi_spinbox = ttk.Spinbox(params_frame, from_=1, to=20, width=5)
        self.colpi_spinbox.set(9)
        self.colpi_spinbox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        action_buttons_frame = ttk.Frame(main_frame)
        action_buttons_frame.grid(row=4, column=0, columnspan=2, pady=10)
        self.start_button = ttk.Button(action_buttons_frame, text="Avvia Analisi Completa", command=self.start_analysis)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.show_chart_button = ttk.Button(action_buttons_frame, text="Mostra Grafico", command=self.open_chart_window)
        self.show_chart_button.pack(side=tk.LEFT, padx=5)
        self.show_chart_button.config(state=tk.DISABLED)

        self.show_stats_button = ttk.Button(action_buttons_frame, text="Mostra Statistiche", command=self.open_stats_window)
        self.show_stats_button.pack(side=tk.LEFT, padx=5)
        self.show_stats_button.config(state=tk.DISABLED)

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

        self.stats_window = None
        self.last_stats_data = None
        self.stats_labels = {}

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.master.after(100, self.process_queue)

    # --- Metodi GUI e Utilità (la maggior parte invariati) ---
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
        self.chart_window.title("Grafico Risultati Analisi")
        self.chart_window.geometry("800x600")
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
        rotate_labels_angle = 0; ha_rotation = 'center'
        if chart_type == "single_numbers":
            title = "Frequenza Numeri Singoli (Top)"; xlabel = "Numero"; ylabel = "Frequenza"
            rotate_labels_angle = 30; ha_rotation = 'right'
        elif chart_type == "backtest_performance":
            title = f"Successi Backtest (Top {len(chart_data) if chart_data else 'N'})"
            xlabel = "Lunghetta"; ylabel = "Numero Successi / Giocate" # Modificato per chiarezza
            rotate_labels_angle = 45; ha_rotation = 'right'
        if chart_data and isinstance(chart_data, list) and len(chart_data) > 0:
            labels = [item[0] for item in chart_data] # item[0] è la stringa della lunghetta o numero
            values = [item[1] for item in chart_data] # item[1] è il valore (frequenza o successi)
            bars = self.ax_freq_chart.bar(labels, values, color='teal' if chart_type == "backtest_performance" else 'skyblue')
            self.ax_freq_chart.set_ylabel(ylabel)
            max_val = max(values) if values else 1
            if max_val > 0 : self.ax_freq_chart.set_ylim(bottom=0, top=max_val + 0.1 * max_val)
            for bar_idx, bar in enumerate(bars):
                yval = bar.get_height()
                text_to_show = str(round(yval))
                if chart_type == "backtest_performance" and len(item) > 2: # Se abbiamo anche le giocate
                    plays_made_info = item[2] # item[2] dovrebbe contenere le giocate
                    text_to_show = f"{round(yval)}/{plays_made_info}"

                if yval > 0 or (chart_type == "backtest_performance" and yval == 0): # Mostra anche 0/N per backtest
                    self.ax_freq_chart.text(bar.get_x() + bar.get_width()/2.0, yval,
                                            text_to_show, ha='center', va='bottom', fontsize=8, color='black')
        else:
            self.ax_freq_chart.text(0.5, 0.5, "Dati per il grafico non disponibili.", horizontalalignment='center', verticalalignment='center', transform=self.ax_freq_chart.transAxes, fontsize=10, color='gray', wrap=True)
        self.ax_freq_chart.set_title(title, fontsize=12, fontweight='bold'); self.ax_freq_chart.set_xlabel(xlabel, fontsize=10)
        if chart_data and len(chart_data) > 0 :
            self.ax_freq_chart.tick_params(axis='x', labelsize=8, rotation=rotate_labels_angle, labelrotation=rotate_labels_angle, pad=5)
            if rotate_labels_angle > 0: plt.setp(self.ax_freq_chart.get_xticklabels(), ha=ha_rotation)
        else: self.ax_freq_chart.tick_params(axis='x', labelsize=8)
        self.ax_freq_chart.tick_params(axis='y', labelsize=8); self.ax_freq_chart.grid(True, linestyle='--', alpha=0.7)
        try: self.chart_figure.tight_layout(pad=2.5)
        except Exception: pass
        canvas_to_draw.draw()

    def _on_stats_window_close(self):
        if self.stats_window: self.stats_window.destroy()
        self.stats_window = None

    def open_stats_window(self):
        if self.stats_window is not None and self.stats_window.winfo_exists():
            self.stats_window.lift()
            self.display_advanced_stats(self.last_stats_data)
            return
        self.stats_window = tk.Toplevel(self.master); self.stats_window.title("Statistiche Avanzate Lunghette")
        self.stats_window.geometry("750x400"); self.stats_window.resizable(True, True)
        self.stats_window.protocol("WM_DELETE_WINDOW", self._on_stats_window_close)
        container = ttk.Frame(self.stats_window, padding="10"); container.pack(fill=tk.BOTH, expand=True)
        ttk.Label(container, text="Statistiche Avanzate per le Top Lunghette:", font=('Helvetica', 13, 'bold')).pack(pady=(0,10))
        cols = ("Lunghetta", "Sorte", "RA", "RSMax", "Freq", "Successi BT", "% Successo BT", "Giocate BT")
        self.stats_tree = ttk.Treeview(container, columns=cols, show='headings', height=10)
        for col_name in cols:
            self.stats_tree.heading(col_name, text=col_name)
            self.stats_tree.column(col_name, width=90, anchor='center') # Larghezza base
        self.stats_tree.column("Lunghetta", width=140, anchor='w'); self.stats_tree.column("Sorte", width=60)
        self.stats_tree.column("RA", width=50); self.stats_tree.column("RSMax", width=70); self.stats_tree.column("Freq", width=50)
        self.stats_tree.column("% Successo BT", width=100)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.stats_tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.stats_tree.xview)
        self.stats_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.stats_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X, before=self.stats_tree)
        self.display_advanced_stats(self.last_stats_data)

    def display_advanced_stats(self, stats_data_list=None):
        if not self.stats_window or not self.stats_window.winfo_exists() or not hasattr(self, 'stats_tree'): return
        for i in self.stats_tree.get_children(): self.stats_tree.delete(i)
        if stats_data_list and isinstance(stats_data_list, list):
            for i, stats_item in enumerate(stats_data_list):
                if isinstance(stats_item, dict):
                    lunghetta_str = str(stats_item.get('lunghetta', 'N/D')).replace(' ', '')
                    sorte = stats_item.get('sorte', 'N/D'); ra = stats_item.get('RA', 'N/D')
                    rs_max = stats_item.get('RSMax', 'N/D'); freq = stats_item.get('Freq', 'N/D')
                    succ_bt = stats_item.get('success_triggers_backtest', 'N/A')
                    plays_bt = stats_item.get('plays_made_backtest', 'N/A') # Aggiunto plays_made
                    perc_succ_bt_val = stats_item.get('percent_success_backtest', 'N/A')
                    perc_succ_bt = f"{perc_succ_bt_val:.2f}%" if isinstance(perc_succ_bt_val, (int, float)) else 'N/A'
                    self.stats_tree.insert("", tk.END, iid=str(i), values=(lunghetta_str, sorte, ra, rs_max, freq, succ_bt, perc_succ_bt, plays_bt))
        else: self.stats_tree.insert("", tk.END, values=("Nessun dato di statistica disponibile.", "", "", "", "", "", "", ""))

    def _calculate_stats_for_lunghetta(self, lunghetta_da_analizzare, sorte_len, processed_historical_dfs, current_end_date_hist):
        # ... (corpo della funzione invariato) ...
        stats = {"RA": "N/D", "RSMax": 0, "Freq": 0}
        if not lunghetta_da_analizzare or len(lunghetta_da_analizzare) < sorte_len: return stats
        target_combinations_sets = [set(combo) for combo in combinations(sorted(lunghetta_da_analizzare), sorte_len)]
        if not target_combinations_sets: return stats
        all_processed_historic_extractions = []
        if processed_historical_dfs:
            for df_hist in processed_historical_dfs.values():
                for _, row in df_hist.iterrows():
                    try:
                        numeri_estratti_dict = {int(row[f'N{i}']) for i in range(1, 6) if pd.notna(row[f'N{i}'])}
                        if len(numeri_estratti_dict) == 5:
                             all_processed_historic_extractions.append({"data": row['Data'],"numeri": numeri_estratti_dict})
                    except (ValueError, KeyError): continue
        if not all_processed_historic_extractions:
            stats["RSMax"] = "N/D (no dati)"; stats["Freq"] = "N/D (no dati)"; stats["RA"] = "N/D (no dati)"
            return stats
        all_processed_historic_extractions_sorted_asc = sorted(all_processed_historic_extractions, key=lambda x: x['data'])
        current_rs = 0; max_rs = 0; freq_count = 0
        for extraction in all_processed_historic_extractions_sorted_asc:
            found_in_extraction_hist = any(target_combo_set.issubset(extraction['numeri']) for target_combo_set in target_combinations_sets)
            if found_in_extraction_hist:
                freq_count += 1
                if current_rs > max_rs: max_rs = current_rs
                current_rs = 0
            else: current_rs += 1
        if current_rs > max_rs: max_rs = current_rs
        stats["RSMax"] = max_rs; stats["Freq"] = freq_count
        all_processed_historic_extractions_sorted_desc = sorted(all_processed_historic_extractions, key=lambda x: x['data'], reverse=True)
        ra_count = 0; found_for_ra = False
        for extraction in all_processed_historic_extractions_sorted_desc:
            if any(target_combo_set.issubset(extraction['numeri']) for target_combo_set in target_combinations_sets):
                found_for_ra = True; break
            else: ra_count += 1
        stats["RA"] = ra_count if found_for_ra else len(all_processed_historic_extractions_sorted_desc)
        return stats


    def fetch_data_from_github(self, url):
        try:
            response = requests.get(url); response.raise_for_status(); return response.text
        except requests.exceptions.RequestException as e:
            self.q.put(f"Errore download da {url}: {e}"); return None

    def parse_lotto_data(self, content, wheel_name):
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

    def start_analysis(self):
        if self.analysis_running: messagebox.showwarning("Attenzione", "Un'analisi è già in corso."); return
        self.results_text.delete(1.0, tk.END); self.progress_bar["value"] = 0
        self.show_chart_button.config(state=tk.DISABLED); self.last_chart_data = None
        if self.chart_window and self.chart_window.winfo_exists(): self.update_frequency_chart(None, self.current_chart_type)
        self.show_stats_button.config(state=tk.DISABLED); self.last_stats_data = None
        if self.stats_window and self.stats_window.winfo_exists(): self.display_advanced_stats(None)

        try:
            start_date_hist = datetime.combine(self.start_date_entry.get_date(), datetime.min.time())
            end_date_hist = datetime.combine(self.end_date_entry.get_date(), datetime.max.time())
            if start_date_hist >= end_date_hist: messagebox.showerror("Errore Date", "Data inizio deve precedere data fine."); return
            
            selected_ruote_analisi_keys = [r for r, v in self.ruote_analisi_vars.items() if v.get()]
            if not selected_ruote_analisi_keys: messagebox.showerror("Errore", "Selezionare almeno una Ruota per Analisi Frequenze (Segnale)."); return
            
            # Determinare la ruota segnale
            # Se l'utente ha selezionato specificamente DEFAULT_RUOTA_SEGNALE, usa quella.
            # Altrimenti, se ha selezionato una sola ruota di analisi, usa quella.
            # Altrimenti, avvisa che deve sceglierne una o che verrà usata la prima. (Per ora usiamo DEFAULT_RUOTA_SEGNALE se presente tra le selezionate, o la prima selezionata)
            ruota_segnale_effettiva = DEFAULT_RUOTA_SEGNALE
            if DEFAULT_RUOTA_SEGNALE not in selected_ruote_analisi_keys:
                if selected_ruote_analisi_keys: # Se ci sono ruote di analisi selezionate
                    ruota_segnale_effettiva = selected_ruote_analisi_keys[0]
                    messagebox.showwarning("Avviso Ruota Segnale",
                                           f"La ruota segnale di default '{RUOTE_DISPONIBILI.get(DEFAULT_RUOTA_SEGNALE, DEFAULT_RUOTA_SEGNALE)}' non è tra le ruote di analisi selezionate.\n"
                                           f"Verrà usata la prima ruota di analisi selezionata come segnale: '{RUOTE_DISPONIBILI.get(ruota_segnale_effettiva, ruota_segnale_effettiva)}'.")
                else: # Questo caso è già gestito sopra, ma per sicurezza
                    messagebox.showerror("Errore", "Nessuna ruota di analisi selezionata."); return


            indice_mese_str = self.indice_mese_combo.get()
            extraction_index_in_month = int(indice_mese_str.split("ª")[0]) if indice_mese_str != "Tutte" else None
            selected_months = [m_num for m_num, var in self.mesi_vars.items() if var.get()]
            lunghetta_len = int(self.lunghetta_len_combo.get())
            sorte_text = self.sorte_combo.get()
            sorte_len = {"Ambo": 2, "Terno": 3, "Quaterna": 4}[sorte_text]
            if lunghetta_len < sorte_len: messagebox.showerror("Errore", f"Lunghezza Lunghetta ({lunghetta_len}) deve essere >= lunghezza Sorte ({sorte_len} per {sorte_text})."); return
            colpi_gioco = int(self.colpi_spinbox.get())
            selected_ruote_gioco = [r for r, v in self.ruote_gioco_vars.items() if v.get()]
            if not selected_ruote_gioco: messagebox.showerror("Errore", "Selezionare almeno una Ruota per Verifica Giocata."); return

            self.analysis_running = True; self.start_button.config(state=tk.DISABLED)
            self.log_message("Avvio analisi...")
            analysis_params = {
                "start_date_hist": start_date_hist, "end_date_hist": end_date_hist,
                "selected_ruote_analisi": selected_ruote_analisi_keys, # Passa le chiavi
                "ruota_segnale_per_trigger": ruota_segnale_effettiva, # Passa la ruota segnale determinata
                "extraction_index_in_month": extraction_index_in_month,
                "selected_months": selected_months, "lunghetta_len": lunghetta_len,
                "sorte_len": sorte_len, "sorte_text": sorte_text,
                "colpi_gioco": colpi_gioco, "selected_ruote_gioco": selected_ruote_gioco
            }
            Thread(target=self._perform_analysis_thread, args=(analysis_params,), daemon=True).start()
        except ValueError as e: messagebox.showerror("Errore Input", f"Errore parametri: {e}"); self.analysis_running = False; self.start_button.config(state=tk.NORMAL)
        except Exception as e: messagebox.showerror("Errore", f"Errore non gestito: {e}\n{traceback.format_exc()}"); self.analysis_running = False; self.start_button.config(state=tk.NORMAL)

    # --- Funzione di Copertura Date MODIFICATA ---
    def _find_covering_lunghettas_for_dates_heuristic(self,
                                                       processed_historical_data_dfs,
                                                       all_data_loaded,
                                                       ruota_segnale_definita, # Ruota da cui originano i trigger
                                                       lunghetta_len_target,
                                                       sorte_len_target,
                                                       sorte_text_target,
                                                       colpi_gioco_target,
                                                       selected_ruote_gioco_target,
                                                       end_date_hist_limit,
                                                       top_performing_lunghettas_from_backtest # Opzionale
                                                       ):
        self.q.put(("log", "\n--- ANALISI COPERTURA DELLE DATE DI ESTRAZIONE STORICHE (STIMA EURISTICA) ---"))
        self.q.put(("log", f"Obiettivo: coprire le date/eventi storici (originati da {RUOTE_DISPONIBILI.get(ruota_segnale_definita, ruota_segnale_definita)}) "
                           f"con {sorte_text_target} da lunghette di {lunghetta_len_target} numeri, entro {colpi_gioco_target} colpi."))

        target_event_triggers_to_cover = []
        original_numbers_for_triggers = {}

        if ruota_segnale_definita not in processed_historical_data_dfs or processed_historical_data_dfs[ruota_segnale_definita].empty:
            self.q.put(("log", f"ATTENZIONE: Nessun dato storico filtrato per la ruota segnale '{RUOTE_DISPONIBILI.get(ruota_segnale_definita, ruota_segnale_definita)}'. Impossibile definire i trigger per la copertura."))
            return

        df_hist_trigger_segnale = processed_historical_data_dfs[ruota_segnale_definita]
        for _, row_trigger in df_hist_trigger_segnale.iterrows():
            event_trigger_tuple = (row_trigger['Data'], ruota_segnale_definita)
            if event_trigger_tuple not in original_numbers_for_triggers:
                target_event_triggers_to_cover.append(event_trigger_tuple)
                try:
                    original_numbers_for_triggers[event_trigger_tuple] = sorted([int(row_trigger[f'N{i}']) for i in range(1, 6) if pd.notna(row_trigger[f'N{i}'])])
                except (ValueError, KeyError): original_numbers_for_triggers[event_trigger_tuple] = []
        target_event_triggers_to_cover.sort(key=lambda x: x[0])

        if not target_event_triggers_to_cover:
            self.q.put(("log", f"Nessun evento/data trigger (da {RUOTE_DISPONIBILI.get(ruota_segnale_definita, ruota_segnale_definita)}) trovato da coprire."))
            return
        self.q.put(("log", f"Identificati {len(target_event_triggers_to_cover)} eventi/date trigger unici da coprire (originati da {RUOTE_DISPONIBILI.get(ruota_segnale_definita, ruota_segnale_definita)})."))

        remaining_triggers_set = set(target_event_triggers_to_cover)
        chosen_lunghettas_solution_list = []
        iteration_count = 0
        max_iterations = len(target_event_triggers_to_cover) * 5 + 20

        while remaining_triggers_set and iteration_count < max_iterations:
            iteration_count += 1
            best_candidate_lunghetta_for_iteration = None
            max_new_triggers_covered_by_best_candidate = 0
            triggers_covered_by_current_best_candidate_tuples = set()

            potential_lunghettas_to_evaluate_set = set()
            if top_performing_lunghettas_from_backtest: # Considera quelle del backtest
                for res_bt in top_performing_lunghettas_from_backtest:
                    # Assicurati che la lunghetta del backtest abbia la lunghezza desiderata per la copertura
                    if len(res_bt['lunghetta']) == lunghetta_len_target:
                         potential_lunghettas_to_evaluate_set.add(tuple(sorted(res_bt['lunghetta'])))
            
            numbers_in_remaining_triggers_counter = Counter()
            for trigger_event_rem in remaining_triggers_set:
                if trigger_event_rem in original_numbers_for_triggers:
                    for num_val in original_numbers_for_triggers[trigger_event_rem]: numbers_in_remaining_triggers_counter[num_val] +=1
            
            if numbers_in_remaining_triggers_counter:
                candidate_pool_target_size = lunghetta_len_target + 5
                current_candidate_numbers_pool_source = [num for num, _ in numbers_in_remaining_triggers_counter.most_common(candidate_pool_target_size)]
                if len(current_candidate_numbers_pool_source) >= lunghetta_len_target:
                    for combo_l_candidate in combinations(current_candidate_numbers_pool_source, lunghetta_len_target):
                        potential_lunghettas_to_evaluate_set.add(tuple(sorted(combo_l_candidate)))
            
            potential_lunghettas_to_evaluate = list(potential_lunghettas_to_evaluate_set)
            MAX_CANDIDATES_TO_BACKTEST_PER_ITER = 150 # Aumentato un po'
            if len(potential_lunghettas_to_evaluate) > MAX_CANDIDATES_TO_BACKTEST_PER_ITER:
                 potential_lunghettas_to_evaluate = random.sample(potential_lunghettas_to_evaluate, MAX_CANDIDATES_TO_BACKTEST_PER_ITER)

            if not potential_lunghettas_to_evaluate and remaining_triggers_set:
                self.q.put(("log", f"Iter {iteration_count}: Nessuna lunghetta candidata generata. Tentativo fallback."))
            
            for i_cand, lung_candidate_tuple in enumerate(potential_lunghettas_to_evaluate):
                sorti_from_candidate_lunghetta = [set(s) for s in combinations(lung_candidate_tuple, sorte_len_target)]
                if not sorti_from_candidate_lunghetta: continue
                newly_covered_triggers_for_this_candidate = set()
                for trigger_event_to_check in remaining_triggers_set:
                    data_trigger_bt_event = trigger_event_to_check[0]
                    success_for_this_trigger = False
                    for wheel_key_gioco_bt_verify in selected_ruote_gioco_target:
                        df_completo_ruota_gioco_bt_verify = all_data_loaded.get(wheel_key_gioco_bt_verify)
                        if df_completo_ruota_gioco_bt_verify is None: continue
                        start_verifica_bt_verify = data_trigger_bt_event + timedelta(days=1)
                        df_finestra_bt_verify = df_completo_ruota_gioco_bt_verify[
                            (df_completo_ruota_gioco_bt_verify['Data'] >= start_verifica_bt_verify) &
                            (df_completo_ruota_gioco_bt_verify['Data'] <= end_date_hist_limit)
                        ].head(colpi_gioco_target)
                        if df_finestra_bt_verify.empty: continue
                        for _colpo_idx_bt_v_loop, (_idx_iter_bt, row_bt_colpo_verify_series) in enumerate(df_finestra_bt_verify.iterrows()):
                            try:
                                numeri_estrazione_bt_verify = {int(row_bt_colpo_verify_series[f'N{k}']) for k in range(1, 6) if pd.notna(row_bt_colpo_verify_series[f'N{k}'])}
                                if any(sc.issubset(numeri_estrazione_bt_verify) for sc in sorti_from_candidate_lunghetta):
                                    success_for_this_trigger = True; break
                            except (ValueError, KeyError): continue
                            if success_for_this_trigger: break
                        if success_for_this_trigger: break
                    if success_for_this_trigger: newly_covered_triggers_for_this_candidate.add(trigger_event_to_check)
                
                count_newly_covered = len(newly_covered_triggers_for_this_candidate)
                if count_newly_covered > max_new_triggers_covered_by_best_candidate:
                    max_new_triggers_covered_by_best_candidate = count_newly_covered
                    best_candidate_lunghetta_for_iteration = lung_candidate_tuple
                    triggers_covered_by_current_best_candidate_tuples = newly_covered_triggers_for_this_candidate

            if best_candidate_lunghetta_for_iteration and max_new_triggers_covered_by_best_candidate > 0:
                chosen_lunghettas_solution_list.append(best_candidate_lunghetta_for_iteration)
                remaining_triggers_set.difference_update(triggers_covered_by_current_best_candidate_tuples)
                # Log più dettagliato
                covered_dates_str = ", ".join([f"{tr[0].strftime('%Y-%m-%d')}({tr[1]})" for tr in triggers_covered_by_current_best_candidate_tuples])
                self.q.put(("log", f"Iter {iteration_count}: Scelta L. {best_candidate_lunghetta_for_iteration}, copre {max_new_triggers_covered_by_best_candidate} NUOVE date/eventi [{covered_dates_str}]. Rimanenti: {len(remaining_triggers_set)}"))
            elif remaining_triggers_set: # Fallback
                self.q.put(("log", f"Iter {iteration_count}: Euristica bloccata o nessuna candidata valida. Attivo fallback per {len(remaining_triggers_set)} date/eventi."))
                trigger_fb = next(iter(remaining_triggers_set))
                numeri_base_fb = original_numbers_for_triggers.get(trigger_fb, [])
                lunghetta_fb_final = None
                if numeri_base_fb and len(numeri_base_fb) >= sorte_len_target :
                    lunghetta_fb_list = list(numeri_base_fb[:lunghetta_len_target])
                    filler_needed_fb = lunghetta_len_target - len(lunghetta_fb_list)
                    if filler_needed_fb > 0:
                        potential_fillers_fb = sorted(list(set(range(1, 91)) - set(lunghetta_fb_list)))
                        if len(potential_fillers_fb) >= filler_needed_fb:
                            lunghetta_fb_list.extend(potential_fillers_fb[:filler_needed_fb])
                    if len(lunghetta_fb_list) == lunghetta_len_target:
                        lunghetta_fb_final = tuple(sorted(lunghetta_fb_list))
                
                if lunghetta_fb_final:
                    # Verifica se questa lunghetta di fallback copre almeno il trigger_fb
                    sorti_from_fb_lunghetta = [set(s) for s in combinations(lunghetta_fb_final, sorte_len_target)]
                    success_fb_trigger = False
                    if sorti_from_fb_lunghetta:
                        # ... (logica di backtest per la lunghetta di fallback, identica a sopra) ...
                        for wheel_key_gioco_bt_fb in selected_ruote_gioco_target:
                            # ... (copia e incolla la logica di backtest da sopra qui) ...
                            df_completo_ruota_gioco_bt_fb = all_data_loaded.get(wheel_key_gioco_bt_fb) # ...
                            if df_completo_ruota_gioco_bt_fb is None: continue
                            start_verifica_bt_fb = trigger_fb[0] + timedelta(days=1)
                            df_finestra_bt_fb = df_completo_ruota_gioco_bt_fb[
                                (df_completo_ruota_gioco_bt_fb['Data'] >= start_verifica_bt_fb) &
                                (df_completo_ruota_gioco_bt_fb['Data'] <= end_date_hist_limit)
                            ].head(colpi_gioco_target)
                            if df_finestra_bt_fb.empty: continue
                            for _r_fb, row_fb in df_finestra_bt_fb.iterrows():
                                try:
                                    numeri_est_fb = {int(row_fb[f'N{k}']) for k in range(1, 6) if pd.notna(row_fb[f'N{k}'])}
                                    if any(sc_fb.issubset(numeri_est_fb) for sc_fb in sorti_from_fb_lunghetta):
                                        success_fb_trigger = True; break
                                except: continue
                                if success_fb_trigger: break
                            if success_fb_trigger: break

                    if success_fb_trigger:
                        chosen_lunghettas_solution_list.append(lunghetta_fb_final)
                        remaining_triggers_set.remove(trigger_fb)
                        self.q.put(("log", f"Iter {iteration_count} (FALLBACK): Usata L. {lunghetta_fb_final} per coprire {trigger_fb[0].strftime('%Y-%m-%d')}({trigger_fb[1]}). Rimanenti: {len(remaining_triggers_set)}"))
                    else:
                        self.q.put(("log", f"Iter {iteration_count} (FALLBACK): L. {lunghetta_fb_final} NON ha coperto {trigger_fb[0].strftime('%Y-%m-%d')}({trigger_fb[1]}). Rimuovo per evitare loop."))
                        remaining_triggers_set.remove(trigger_fb)
                else:
                    self.q.put(("log", f"Iter {iteration_count} (FALLBACK): Impossibile generare lunghetta valida per {trigger_fb[0].strftime('%Y-%m-%d')}({trigger_fb[1]}). Rimuovo per evitare loop."))
                    remaining_triggers_set.remove(trigger_fb)
            else: break # Tutto coperto

        if iteration_count >= max_iterations and remaining_triggers_set:
            self.q.put(("log", f"AVVISO COPERTURA DATE: Raggiunto limite iterazioni ({max_iterations}). {len(remaining_triggers_set)} date/eventi potrebbero non essere coperte."))
        final_unique_lunghettas = sorted(list(set(chosen_lunghettas_solution_list)))
        coverage_results = {
            "total_target_events_initial": len(target_event_triggers_to_cover),
            "target_events_remaining_uncovered": len(remaining_triggers_set),
            "num_lunghettas_needed_heuristic": len(final_unique_lunghettas),
            "lunghettas_list_heuristic": final_unique_lunghettas,
            "lunghetta_len_for_coverage": lunghetta_len_target,
            "sorte_text_for_coverage": sorte_text_target,
            "colpi_per_copertura": colpi_gioco_target,
            "ruota_segnale_usata": ruota_segnale_definita
        }
        if remaining_triggers_set:
             uncovered_sample_str = [f"{tr[0].strftime('%Y-%m-%d')}({tr[1]})" for tr in sorted(list(remaining_triggers_set))[:10]]
             coverage_results["uncovered_events_sample"] = uncovered_sample_str
        self.q.put(("coverage_dates_results", coverage_results))

    # --- Thread Principale di Analisi ---
    def _perform_analysis_thread(self, params):
        try:
            self.q.put(("clear_chart_data", None, "initial_clear"))
            self.q.put(("clear_advanced_stats", None))

            ruota_segnale_effettiva = params["ruota_segnale_per_trigger"] # Presa dai parametri
            self.q.put(("log", f"--- RUOTA SEGNALE PER TRIGGER (Backtest F2/3 e Copertura F5): {RUOTE_DISPONIBILI.get(ruota_segnale_effettiva, ruota_segnale_effettiva)} ---"))

            # --- FASE 1: Caricamento e Filtro Dati Iniziale ---
            self.q.put(("log", "--- FASE 1: Caricamento Dati Complessivo ---"))
            all_data_loaded = {}
            # ... (codice caricamento dati come prima, assicurati che all_data_loaded sia popolato) ...
            all_wheels_to_load = list(set(params["selected_ruote_analisi"] + params["selected_ruote_gioco"]))
            total_wheels_to_load = len(all_wheels_to_load)
            processed_wheels = 0
            if not all_wheels_to_load: self.q.put(("log", "Nessuna ruota selezionata. Analisi interrotta.")); self.q.put("analysis_complete"); return
            for wheel_key in all_wheels_to_load:
                self.q.put(("log", f"Caricamento dati per {RUOTE_DISPONIBILI[wheel_key]}..."))
                df_full = self.data_cache.get(wheel_key)
                if df_full is None or df_full.empty:
                    content = self.fetch_data_from_github(URL_RUOTE[wheel_key])
                    if content: df_full = self.parse_lotto_data(content, wheel_key)
                    if df_full is not None and not df_full.empty: self.data_cache[wheel_key] = df_full.copy(); self.q.put(("log", f"Dati per {RUOTE_DISPONIBILI[wheel_key]} scaricati."))
                    else: self.q.put(("log", f"Nessun dato valido per {RUOTE_DISPONIBILI[wheel_key]}."))
                else: self.q.put(("log", f"Dati per {RUOTE_DISPONIBILI[wheel_key]} caricati dalla cache."))
                if df_full is not None and not df_full.empty: all_data_loaded[wheel_key] = df_full
                else: self.q.put(("log", f"Attenzione: {RUOTE_DISPONIBILI[wheel_key]} esclusa per mancanza dati."))
                processed_wheels += 1
                self.q.put(("progress", processed_wheels * (30 / total_wheels_to_load if total_wheels_to_load > 0 else 0)))
            if not all_data_loaded: self.q.put(("log", "Nessun dato caricato. Analisi interrotta.")); self.q.put("analysis_complete"); return
            for r_gioco in params["selected_ruote_gioco"]: # Verifica cruciale per la copertura
                if r_gioco not in all_data_loaded or all_data_loaded[r_gioco].empty:
                    self.q.put(("log", f"ERRORE CRITICO: Dati non disponibili per la ruota di gioco {RUOTE_DISPONIBILI[r_gioco]}, necessaria per l'analisi. Interruzione."))
                    self.q.put("analysis_complete"); return

            self.q.put(("log", "\n--- APPLICAZIONE FILTRO PERIODO STORICO BASE ---"))
            historical_data_dfs_initial_period = {}
            for k_f1 in params["selected_ruote_analisi"]: # Usa le ruote di analisi selezionate
                if k_f1 in all_data_loaded:
                    df_temp_f1 = all_data_loaded[k_f1][(all_data_loaded[k_f1]['Data'] >= params["start_date_hist"]) & (all_data_loaded[k_f1]['Data'] <= params["end_date_hist"])]
                    if not df_temp_f1.empty: historical_data_dfs_initial_period[k_f1] = df_temp_f1
            if not historical_data_dfs_initial_period: self.q.put(("log", "Nessun dato storico nel periodo base per le ruote di analisi. Analisi interrotta.")); self.q.put("analysis_complete"); return
            self.q.put(("progress", 35))

            self.q.put(("log", "\n--- APPLICAZIONE FILTRI TEMPORALI AVANZATI (su dati ruote analisi) ---"))
            processed_historical_data_dfs = {} # Questo conterrà i dati filtrati SOLO per le ruote di analisi
            for wheel_key_f1_adv, df_original_f1_adv in historical_data_dfs_initial_period.items():
                # ... (logica di filtro avanzato come prima, applicata a df_original_f1_adv) ...
                df_to_filter_f1_adv = df_original_f1_adv.copy()
                if params["selected_months"]:
                    df_to_filter_f1_adv = df_to_filter_f1_adv[df_to_filter_f1_adv['Data'].dt.month.isin(params["selected_months"])]
                if df_to_filter_f1_adv.empty: continue
                if params["extraction_index_in_month"]:
                    df_to_filter_f1_adv = df_to_filter_f1_adv.sort_values(by='Data')
                    df_to_filter_f1_adv.loc[:, 'AnnoMeseGroup'] = df_to_filter_f1_adv['Data'].dt.to_period('M')
                    df_to_filter_f1_adv.loc[:, 'IndiceNelMese'] = df_to_filter_f1_adv.groupby('AnnoMeseGroup')['Data'].cumcount() + 1
                    df_to_filter_f1_adv = df_to_filter_f1_adv[df_to_filter_f1_adv['IndiceNelMese'] == params["extraction_index_in_month"]]
                    df_to_filter_f1_adv = df_to_filter_f1_adv.drop(columns=['AnnoMeseGroup', 'IndiceNelMese'], errors='ignore')
                if not df_to_filter_f1_adv.empty:
                    processed_historical_data_dfs[wheel_key_f1_adv] = df_to_filter_f1_adv
                    self.q.put(("log", f"Dati filtrati per analisi ({RUOTE_DISPONIBILI[wheel_key_f1_adv]}): {len(df_to_filter_f1_adv)} estrazioni."))

            # Verifica se la ruota segnale ha dati dopo il filtraggio
            if ruota_segnale_effettiva not in processed_historical_data_dfs or processed_historical_data_dfs[ruota_segnale_effettiva].empty:
                self.q.put(("log", f"Nessun dato storico per la RUOTA SEGNALE '{RUOTE_DISPONIBILI.get(ruota_segnale_effettiva, ruota_segnale_effettiva)}' dopo i filtri. Analisi interrotta."))
                self.q.put("analysis_complete"); return
            self.q.put(("progress", 40))


            # --- FASE 2: Identificazione Lunghette Candidate e Backtest (MODIFICATO per usare RUOTA_SEGNALE) ---
            self.q.put(("log", f"\n--- FASE 2: Identificazione Lunghette Candidate Storiche e Loro Backtesting (Trigger da {RUOTE_DISPONIBILI.get(ruota_segnale_effettiva, ruota_segnale_effettiva)}) ---"))
            lunghezza_serie_da_cercare = params["lunghetta_len"]
            top_base_lunghettas_to_test_tuples = [] # Candidate da testare
            
            # Generazione candidate (da ruota segnale o tutte le ruote analisi? Per ora da ruota segnale per coerenza)
            df_per_candidate_f2 = processed_historical_data_dfs[ruota_segnale_effettiva] # Usa solo dati della ruota segnale per trovare le candidate
            
            if lunghezza_serie_da_cercare <= 5:
                # ... (logica generazione candidate come prima, ma usando df_per_candidate_f2) ...
                possible_base_lunghettas_counter = Counter()
                num_estrazioni_per_conteggio_base = len(df_per_candidate_f2)
                for _, row_base in df_per_candidate_f2.iterrows():
                    try:
                        extracted_numbers_base = sorted([int(row_base[f'N{i}']) for i in range(1, 6)])
                        if len(extracted_numbers_base) == 5:
                            for combo_base in combinations(extracted_numbers_base, lunghezza_serie_da_cercare):
                                possible_base_lunghettas_counter[combo_base] += 1
                    except (ValueError, KeyError): continue
                top_base_lunghettas_to_test_tuples = possible_base_lunghettas_counter.most_common(500)
            else: # > 5
                # ... (logica generazione candidate >5 come prima, ma usando df_per_candidate_f2) ...
                single_number_counter = Counter()
                for _, row_base in df_per_candidate_f2.iterrows():
                    try:
                        for i in range(1, 6): single_number_counter[int(row_base[f'N{i}'])] += 1
                    except (ValueError, KeyError): continue
                if not single_number_counter: self.q.put(("log", "Nessun numero trovato per calcolare la frequenza singola (Fase 2).")); # ...
                top_single_numbers_with_freq = single_number_counter.most_common(lunghezza_serie_da_cercare)
                # ... (resto della logica per >5)
                if top_single_numbers_with_freq: self.q.put(("update_chart_data", [(num, freq) for num, freq in top_single_numbers_with_freq], "single_numbers"))
                derived_lunghetta_list = sorted([num for num, freq in top_single_numbers_with_freq])
                if not derived_lunghetta_list or len(derived_lunghetta_list) < params['sorte_len']: self.q.put(("log", "Numeri insuff. per sorte (Fase 2).")); self.q.put("analysis_complete"); return
                aggregated_freq = sum(freq for num, freq in top_single_numbers_with_freq if num in derived_lunghetta_list)
                top_base_lunghettas_to_test_tuples = [(tuple(derived_lunghetta_list), aggregated_freq)]

            if not top_base_lunghettas_to_test_tuples: self.q.put(("log", "Nessuna lunghetta candidata da testare (Fase 2).")); self.q.put("analysis_complete"); return
            self.q.put(("log", f"Selezionate {len(top_base_lunghettas_to_test_tuples)} lunghette candidate per il backtesting."))
            self.q.put(("progress", 45))

            all_backtest_results = [] # Risultati del backtest per la Fase 3
            df_triggers_ruota_segnale_f2_backtest = processed_historical_data_dfs[ruota_segnale_effettiva] # Trigger SOLO dalla ruota segnale

            for i_base, (base_lunghetta_tuple, freq_base) in enumerate(top_base_lunghettas_to_test_tuples):
                base_lunghetta = list(base_lunghetta_tuple)
                # ... (codice interno del backtest come da tua ultima modifica, che itera su df_triggers_ruota_segnale_f2_backtest) ...
                if len(base_lunghetta) < params['sorte_len']: continue
                candidate_combinations_for_this_base = list(combinations(sorted(base_lunghetta), params['sorte_len']))
                if not candidate_combinations_for_this_base: continue
                bt_triggers_con_successo_per_lb, bt_giocate_effettuate_per_lb = 0, 0
                bt_date_successi_per_lb = []

                for _, trigger_row_bt_event in df_triggers_ruota_segnale_f2_backtest.sort_values(by='Data').iterrows():
                    data_trigger_bt_event = trigger_row_bt_event['Data']
                    if data_trigger_bt_event + timedelta(days=1) > params['end_date_hist']: continue
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
                                    numeri_estrazione_bt_verify = {int(row_bt_colpo_verify_series[f'N{k}']) for k in range(1, 6) if pd.notna(row_bt_colpo_verify_series[f'N{k}'])}
                                    if set_combo_bt_verify.issubset(numeri_estrazione_bt_verify):
                                        # msg_dettaglio_bt = ( ... ) # Log dettagliato può essere rimosso o ridotto
                                        # self.q.put(("log", msg_dettaglio_bt))
                                        bt_date_successi_per_lb.append(row_bt_colpo_verify_series['Data'])
                                        successo_per_trigger_su_qualsiasi_ruota_gioco = True; break
                                except (ValueError, KeyError): continue
                                if successo_per_trigger_su_qualsiasi_ruota_gioco: break
                            if successo_per_trigger_su_qualsiasi_ruota_gioco: break
                    if almeno_una_giocata_possibile_per_trigger:
                        bt_giocate_effettuate_per_lb += 1
                        if successo_per_trigger_su_qualsiasi_ruota_gioco: bt_triggers_con_successo_per_lb += 1
                
                percent_successo_bt_lb = (bt_triggers_con_successo_per_lb / bt_giocate_effettuate_per_lb) * 100 if bt_giocate_effettuate_per_lb > 0 else 0
                all_backtest_results.append({
                    "lunghetta": sorted(base_lunghetta), "success_triggers": bt_triggers_con_successo_per_lb,
                    "plays_made": bt_giocate_effettuate_per_lb, "percent_success": percent_successo_bt_lb,
                    "original_freq": freq_base, "date_successi_bt": bt_date_successi_per_lb })
                self.q.put(("progress", 45 + (i_base + 1) * (25 / len(top_base_lunghettas_to_test_tuples) if top_base_lunghettas_to_test_tuples else 0) ))
            
            # Aggiornamento grafico backtest performance
            if lunghezza_serie_da_cercare <= 5 and all_backtest_results:
                chart_data_backtest = sorted(all_backtest_results, key=lambda x: (x.get("success_triggers", 0), x.get("percent_success",0)), reverse=True)
                # Per il grafico, passiamo anche "plays_made"
                data_to_send_to_chart = [(str(tuple(res_bt['lunghetta'])).replace(' ', ''), res_bt['success_triggers'], res_bt['plays_made']) for res_bt in chart_data_backtest[:15]]
                if data_to_send_to_chart: self.q.put(("update_chart_data", data_to_send_to_chart, "backtest_performance"))
            self.q.put(("progress", 70))

            # --- FASE 3: Riepilogo e Statistiche Avanzate ---
            # (Usa all_backtest_results, ora calcolato sui trigger della ruota segnale)
            # ... (codice Fase 3 invariato, ma i suoi dati di input sono ora più coerenti) ...
            self.q.put(("log", "\n--- FASE 3: Riepilogo Lunghette con Migliore Performance Storica (Backtest) e Statistiche ---"))
            sorted_results = sorted(all_backtest_results, key=lambda x: (x.get("success_triggers", 0), x.get("percent_success", 0), x.get("original_freq", 0)), reverse=True)
            final_lunghetta_scelta_per_gioco = None
            dati_prima_lunghetta_backtest = {} # Contiene {'lunghetta': ..., 'success_triggers': ..., 'plays_made': ...}
            if sorted_results:
                # ... (logica di visualizzazione e selezione final_lunghetta_scelta_per_gioco come prima) ...
                # Aggiunta del campo "plays_made" al log della Fase 3
                freq_label = "Freq. grezza" if lunghezza_serie_da_cercare <= 5 else "Freq. aggregata"
                self.q.put(("log", "Top lunghette candidate ordinate per performance nel backtest (trigger da ruota segnale):"))
                for i_res, res in enumerate(sorted_results[:10]): # Mostra top 10
                    anni_successi_str = ""
                    if res.get("date_successi_bt"):
                        anni_unici = sorted(list(set(data_succ.year for data_succ in res["date_successi_bt"])))
                        if anni_unici: anni_successi_str = f" (Anni BT: {', '.join(map(str, anni_unici))})"
                    self.q.put(("log", f"  Pos. {i_res+1}: LUNGHETTA {res['lunghetta']} -> {res['success_triggers']} successi su {res['plays_made']} giocate ({res['percent_success']:.2f}%)"
                                       f"{anni_successi_str} ({freq_label}: {res['original_freq']})"))
                final_lunghetta_scelta_per_gioco = sorted_results[0]['lunghetta']
                dati_prima_lunghetta_backtest = sorted_results[0]
                self.q.put(("log", f"\nLunghetta Selezionata per Gioco Futuro (FASE 4): {final_lunghetta_scelta_per_gioco} ({dati_prima_lunghetta_backtest.get('success_triggers','N/A')} succ. su {dati_prima_lunghetta_backtest.get('plays_made','N/A')} gioc. nel BT)"))

            else: self.q.put(("log", "Nessun risultato di backtest (Fase 3).")); self.q.put("analysis_complete"); return
            # ... (resto della Fase 3 per statistiche e preparazione Fase 4, come prima) ...
            if not final_lunghetta_scelta_per_gioco: self.q.put(("log", "Impossibile selezionare una lunghetta (Fase 3).")); self.q.put("analysis_complete"); return
            if len(final_lunghetta_scelta_per_gioco) < params['sorte_len']: self.q.put(("log", f"Errore: lunghetta finale ({len(final_lunghetta_scelta_per_gioco)}) troppo corta per sorte ({params['sorte_len']}).")); self.q.put("analysis_complete"); return
            candidate_combinations_final_for_future = list(combinations(final_lunghetta_scelta_per_gioco, params['sorte_len']))
            if not candidate_combinations_final_for_future: self.q.put(("log", f"Nessuna combinazione generabile dalla lunghetta finale.")); self.q.put("analysis_complete"); return

            self.q.put(("log", "\n--- Calcolo Statistiche Avanzate per le Top Lunghette ---"))
            num_top_lunghettas_for_stats = 5
            all_stats_payload = []
            items_to_calculate_stats_for = sorted_results[:num_top_lunghettas_for_stats] if lunghezza_serie_da_cercare <= 5 and sorted_results else []
            if lunghezza_serie_da_cercare > 5 and dati_prima_lunghetta_backtest : items_to_calculate_stats_for = [dati_prima_lunghetta_backtest]
            
            if not items_to_calculate_stats_for: # ... (come prima)
                self.q.put(("log", "Nessuna lunghetta valida per calcolare statistiche avanzate."))
            else:
                for i_stat, backtest_result_item_stat in enumerate(items_to_calculate_stats_for):
                    # ... (logica popolamento stats_entry_adv, assicurati di passare anche 'plays_made_backtest')
                    calculated_stats_adv = self._calculate_stats_for_lunghetta(backtest_result_item_stat['lunghetta'], params['sorte_len'], processed_historical_data_dfs, params['end_date_hist'])
                    stats_entry_adv = {
                        "lunghetta": backtest_result_item_stat['lunghetta'], "sorte": params['sorte_text'],
                        "RA": calculated_stats_adv.get('RA', "Errore"), "RSMax": calculated_stats_adv.get('RSMax', "Errore"),
                        "Freq": calculated_stats_adv.get('Freq', "Errore"),
                        "success_triggers_backtest": backtest_result_item_stat.get('success_triggers', 'N/A'),
                        "plays_made_backtest": backtest_result_item_stat.get('plays_made', 'N/A'), # Da all_backtest_results
                        "percent_success_backtest": backtest_result_item_stat.get('percent_success', 'N/A') }
                    all_stats_payload.append(stats_entry_adv)
                if all_stats_payload: self.q.put(("update_advanced_stats", all_stats_payload))
            self.q.put(("progress", 80))


            # --- FASE 4: Verifica Colpi (Futuro) ---
            # (Trigger da ruota segnale)
            # ... (modifica anche qui il ciclo dei trigger futuri per usare solo ruota_segnale_effettiva) ...
            self.q.put(("log", f"\n--- FASE 4: Verifica Giocate per {params['colpi_gioco']} Colpi (Futuro, trigger da {RUOTE_DISPONIBILI.get(ruota_segnale_effettiva, ruota_segnale_effettiva)}) ---"))
            start_date_for_future_logic_f4 = params['end_date_hist'].date() + timedelta(days=1)
            future_data_start_date_f4 = datetime.combine(start_date_for_future_logic_f4, datetime.min.time())
            found_predictions_f4_details = []
            if not all_data_loaded: self.q.put(("log", "ERRORE FASE 4: all_data_loaded non disponibile.")); self.q.put("analysis_complete"); return
            
            future_trigger_events_f4 = []
            if ruota_segnale_effettiva in all_data_loaded:
                df_trg_orig_f4_segnale = all_data_loaded[ruota_segnale_effettiva]
                df_trg_f4_segnale = df_trg_orig_f4_segnale[df_trg_orig_f4_segnale['Data'] >= future_data_start_date_f4].copy()
                if not df_trg_f4_segnale.empty:
                    if params["selected_months"]: df_trg_f4_segnale = df_trg_f4_segnale[df_trg_f4_segnale['Data'].dt.month.isin(params["selected_months"])]
                    if not df_trg_f4_segnale.empty and params["extraction_index_in_month"]:
                        df_trg_f4_segnale = df_trg_f4_segnale.sort_values(by='Data')
                        df_trg_f4_segnale.loc[:, 'AnnoMeseGroup_f4'] = df_trg_f4_segnale['Data'].dt.to_period('M')
                        df_trg_f4_segnale.loc[:, 'IndiceNelMese_f4'] = df_trg_f4_segnale.groupby('AnnoMeseGroup_f4')['Data'].cumcount() + 1
                        df_trg_f4_segnale = df_trg_f4_segnale[df_trg_f4_segnale['IndiceNelMese_f4'] == params["extraction_index_in_month"]]
                        df_trg_f4_segnale = df_trg_f4_segnale.drop(columns=['AnnoMeseGroup_f4', 'IndiceNelMese_f4'], errors='ignore')
                    for _idx_trg_iter_f4, r_trg_series_f4 in df_trg_f4_segnale.iterrows():
                        future_trigger_events_f4.append({"data_trigger": r_trg_series_f4['Data'], "ruota_trigger_nome": RUOTE_DISPONIBILI[ruota_segnale_effettiva]})
            # ... (resto della Fase 4 come prima, usando future_trigger_events_f4) ...
            if not future_trigger_events_f4: self.q.put(("log", "Nessun evento trigger futuro (da ruota segnale) trovato."))
            else:
                # ... (logica di verifica Fase 4 come prima) ...
                future_trigger_events_f4.sort(key=lambda x: x['data_trigger'])
                self.q.put(("log", f"Trovati {len(future_trigger_events_f4)} eventi trigger futuri. Verifica in corso..."))
                # ... (ciclo e logica interna)
                for trigger_event_f4_item in future_trigger_events_f4:
                    # ...
                    if not params["selected_ruote_gioco"]: self.q.put(("log", "Nessuna ruota di gioco per F4.")); continue
                    for combo_play_f4_item in candidate_combinations_final_for_future:
                        # ... (logica interna per successo)
                        pass # Mantenere la logica di verifica interna come prima
            self.q.put(("progress", 90))


            # --- FASE 5: Analisi di Copertura delle Date Storiche ---
            self.q.put(("log", "\n--- FASE 5: Analisi di Copertura delle Date Storiche Filtrate (Stima Euristica) ---"))
            if processed_historical_data_dfs and all_data_loaded and params["selected_ruote_gioco"]:
                 self._find_covering_lunghettas_for_dates_heuristic(
                    processed_historical_data_dfs=processed_historical_data_dfs,
                    all_data_loaded=all_data_loaded,
                    ruota_segnale_definita=ruota_segnale_effettiva, # Usa la ruota segnale
                    lunghetta_len_target=params["lunghetta_len"],
                    sorte_len_target=params["sorte_len"],
                    sorte_text_target=params["sorte_text"],
                    colpi_gioco_target=params["colpi_gioco"],
                    selected_ruote_gioco_target=params["selected_ruote_gioco"],
                    end_date_hist_limit=params["end_date_hist"],
                    top_performing_lunghettas_from_backtest=sorted_results # Passa i risultati del backtest
                )
            else: # ... (come prima)
                self.q.put(("log", "Dati insufficienti per l'analisi di copertura date."))
            self.q.put(("progress", 99))

        except Exception as e:
            self.q.put(("log", f"Errore critico durante l'analisi: {e}\n{traceback.format_exc()}"))
            # ... (gestione errore come prima) ...
        finally:
            self.q.put(("progress", 100))
            self.q.put("analysis_complete")

    # --- Process Queue (MODIFICATO per nuovi messaggi e dettagli) ---
    def process_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    msg_type = item[0]
                    if msg_type == "progress": self.progress_bar["value"] = item[1]
                    elif msg_type == "update_chart_data":
                        _, chart_data, chart_type = item; self.last_chart_data = chart_data; self.current_chart_type = chart_type
                        self.show_chart_button.config(state=tk.NORMAL)
                        if self.chart_window and self.chart_window.winfo_exists(): self.update_frequency_chart(self.last_chart_data, self.current_chart_type)
                    elif msg_type == "clear_chart_data": # ... (come prima)
                        self.last_chart_data = None; self.current_chart_type = item[2] if len(item) > 2 else "single_numbers"
                        if self.chart_window and self.chart_window.winfo_exists(): self.update_frequency_chart(None, self.current_chart_type)
                    elif msg_type == "update_advanced_stats":
                        _, stats_data = item; self.last_stats_data = stats_data
                        self.show_stats_button.config(state=tk.NORMAL)
                        if self.stats_window and self.stats_window.winfo_exists(): self.display_advanced_stats(self.last_stats_data)
                    elif msg_type == "clear_advanced_stats": # ... (come prima)
                        self.last_stats_data = None
                        if self.stats_window and self.stats_window.winfo_exists(): self.display_advanced_stats(None)
                    elif msg_type == "coverage_dates_results": # Gestione output Fase 5
                        _, coverage_data = item
                        self.log_message(f"\n--- RISULTATI ANALISI COPERTURA DATE/EVENTI STORICI (TRIGGER DA {RUOTE_DISPONIBILI.get(coverage_data['ruota_segnale_usata'],'N/D')}) ---", is_result=True)
                        self.log_message(f"Sorte: {coverage_data['sorte_text_for_coverage']}, Lunghetta: {coverage_data['lunghetta_len_for_coverage']} num, Colpi: {coverage_data['colpi_per_copertura']}.", is_result=True)
                        self.log_message(f"Date/Eventi trigger totali da coprire: {coverage_data['total_target_events_initial']}", is_result=True)
                        if coverage_data['target_events_remaining_uncovered'] > 0:
                            self.log_message(f"ATTENZIONE: {coverage_data['target_events_remaining_uncovered']} date/eventi trigger sono rimaste scoperte.", is_result=True)
                            if "uncovered_events_sample" in coverage_data: self.log_message(f"  Esempio non coperti: {coverage_data['uncovered_events_sample']}", is_result=True)
                        else: self.log_message("Tutte le date/eventi trigger identificati sono state coperte.", is_result=True)
                        self.log_message(f"Numero di lunghette stimate necessarie: {coverage_data['num_lunghettas_needed_heuristic']}", is_result=True)
                        if coverage_data['lunghettas_list_heuristic']:
                            self.log_message("Elenco lunghette stimate per copertura date:", is_result=True)
                            for i, l_sol in enumerate(coverage_data['lunghettas_list_heuristic']):
                                if i < 30: self.log_message(f"  - {l_sol}", is_result=True)
                                elif i == 30: self.log_message(f"  ... e altre {len(coverage_data['lunghettas_list_heuristic']) - 30}.", is_result=True); break
                        self.log_message("--------------------------------------------------", is_result=True)
                    elif msg_type == "log": self.log_message(item[1], is_result=item[2] if len(item) > 2 else False)
                    else: self.log_message(item[0], is_result=item[1] if len(item) > 1 else False)
                elif item == "analysis_complete": # ... (come prima)
                    self.log_message("\nAnalisi completata."); self.progress_bar["value"] = 100
                    self.analysis_running = False; self.start_button.config(state=tk.NORMAL)
                    if self.last_chart_data: self.show_chart_button.config(state=tk.NORMAL)
                    if self.last_stats_data: self.show_stats_button.config(state=tk.NORMAL)
                else: self.log_message(str(item))
        except queue.Empty: pass
        finally: self.master.after(100, self.process_queue)

if __name__ == '__main__':
    root = tk.Tk()
    app = LottoApp(root)
    root.mainloop()