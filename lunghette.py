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
import math # math non è usato al momento, ma lo lasciamo se serve in futuro
from collections import Counter
import calendar # Per i nomi dei mesi

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
MESI_NOMI = [calendar.month_name[i] for i in range(1, 13)] # Gennaio, Febbraio, ...

class LottoApp:
    def __init__(self, master):
        self.master = master
        master.title("Analizzatore Lotto Avanzato")
        master.geometry("850x750")

        self.data_cache = {}
        self.analysis_running = False
        self.q = queue.Queue()

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sezione Periodo Storico ---
        period_frame = ttk.LabelFrame(main_frame, text="Periodo Storico per Analisi Frequenze", padding="10")
        period_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(period_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = DateEntry(period_frame, width=12, background='darkblue',
                                          foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        default_start_date = datetime.now() - timedelta(days=365*2)
        self.start_date_entry.set_date(default_start_date)

        ttk.Label(period_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = DateEntry(period_frame, width=12, background='darkblue',
                                        foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.set_date(datetime.now() - timedelta(days=1))

        # --- Sezione Filtri Temporali Avanzati ---
        adv_filters_frame = ttk.LabelFrame(main_frame, text="Filtri Temporali Avanzati per Analisi", padding="10")
        adv_filters_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(adv_filters_frame, text="Indice Estrazione del Mese:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.indice_mese_values = ["Tutte"] + [f"{i}ª del mese" for i in range(1, 17)]
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

        # --- Sezione Ruote per Analisi ---
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

        # --- Sezione Ruote di Gioco ---
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

        # --- Sezione Parametri Ricerca ---
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

        self.start_button = ttk.Button(main_frame, text="Avvia Analisi", command=self.start_analysis)
        self.start_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")

        self.results_text = scrolledtext.ScrolledText(main_frame, height=15, width=80, wrap=tk.WORD)
        self.results_text.grid(row=6, column=0, columnspan=2, pady=5, sticky="nsew")

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.master.after(100, self.process_queue)

    def toggle_all_mesi(self, select_state):
        for var in self.mesi_vars.values():
            var.set(select_state)

    def toggle_all_ruote(self, ruote_vars_dict, select_state):
        for var in ruote_vars_dict.values():
            var.set(select_state)

    def log_message(self, message, is_result=False):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)
        if is_result:
            print(f"RISULTATO: {message}")

    def fetch_data_from_github(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.q.put(f"Errore download da {url}: {e}")
            return None

    def parse_lotto_data(self, content, wheel_name):
        if not content: return pd.DataFrame()
        try:
            data = io.StringIO(content)
            df = pd.read_csv(data, sep='\s+', header=None,
                             names=['Data', 'Ruota'] + [f'N{i}' for i in range(1, 6)],
                             engine='python')
            def parse_date_flexible(date_str):
                for fmt in ('%Y.%m.%d', '%d/%m/%Y', '%Y-%m-%d'):
                    try: return datetime.strptime(date_str, fmt)
                    except ValueError: continue
                try:
                    date_str_norm = date_str.replace('.', '-').replace('/', '-')
                    return datetime.strptime(date_str_norm, '%Y-%m-%d')
                except ValueError:
                     self.q.put(f"WARN: Formato data non riconosciuto '{date_str}' per {wheel_name}. Riga ignorata.")
                     return pd.NaT
            df['Data'] = df['Data'].apply(parse_date_flexible)
            df.dropna(subset=['Data'], inplace=True)
            for col in [f'N{i}' for i in range(1, 6)]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[f'N{i}' for i in range(1, 6)], inplace=True)
            for col in [f'N{i}' for i in range(1, 6)]: df[col] = df[col].astype(int)
            df['Ruota'] = df['Ruota'].astype(str).str.upper()
            return df.sort_values(by='Data')
        except Exception as e:
            self.q.put(f"Errore parsing dati per {wheel_name}: {e}\n{traceback.format_exc()}")
            return pd.DataFrame()

    def start_analysis(self):
        if self.analysis_running:
            messagebox.showwarning("Attenzione", "Un'analisi è già in corso.")
            return
        self.results_text.delete(1.0, tk.END)
        self.progress_bar["value"] = 0
        try:
            start_date_hist = datetime.combine(self.start_date_entry.get_date(), datetime.min.time())
            end_date_hist = datetime.combine(self.end_date_entry.get_date(), datetime.max.time())
            if start_date_hist >= end_date_hist:
                messagebox.showerror("Errore Date", "Data inizio deve precedere data fine.")
                return
            selected_ruote_analisi = [r for r, v in self.ruote_analisi_vars.items() if v.get()]
            if not selected_ruote_analisi:
                messagebox.showerror("Errore", "Selezionare ruote per analisi frequenze.")
                return
            indice_mese_str = self.indice_mese_combo.get()
            extraction_index_in_month = None
            if indice_mese_str != "Tutte":
                extraction_index_in_month = int(indice_mese_str.split("ª")[0])
            selected_months = [month_num for month_num, var in self.mesi_vars.items() if var.get()]
            lunghetta_len = int(self.lunghetta_len_combo.get())
            sorte_text = self.sorte_combo.get()
            sorte_map = {"Ambo": 2, "Terno": 3, "Quaterna": 4}
            sorte_len = sorte_map[sorte_text]
            if lunghetta_len < sorte_len:
                 messagebox.showerror("Errore", f"Lunghetta ({lunghetta_len}) < sorte ({sorte_text}={sorte_len}).")
                 return
            colpi_gioco = int(self.colpi_spinbox.get())
            selected_ruote_gioco = [r for r, v in self.ruote_gioco_vars.items() if v.get()]
            if not selected_ruote_gioco:
                messagebox.showerror("Errore", "Selezionare ruote per verifica giocata.")
                return
            
            self.analysis_running = True
            self.start_button.config(state=tk.DISABLED)
            self.log_message("Avvio analisi...")
            analysis_params = {
                "start_date_hist": start_date_hist, "end_date_hist": end_date_hist,
                "selected_ruote_analisi": selected_ruote_analisi,
                "extraction_index_in_month": extraction_index_in_month,
                "selected_months": selected_months,
                "lunghetta_len": lunghetta_len, "sorte_len": sorte_len, "sorte_text": sorte_text,
                "colpi_gioco": colpi_gioco, "selected_ruote_gioco": selected_ruote_gioco
            }
            Thread(target=self._perform_analysis_thread, args=(analysis_params,), daemon=True).start()
        except ValueError as e:
            messagebox.showerror("Errore Input", f"Errore parametri: {e}")
            self.analysis_running = False; self.start_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore: {e}\n{traceback.format_exc()}")
            self.analysis_running = False; self.start_button.config(state=tk.NORMAL)

    def _perform_analysis_thread(self, params):
        try:
            # --- FASE 1: Caricamento Dati Complessivo, Filtro Periodo Base, Filtri Avanzati ---
            self.q.put("--- FASE 1: Caricamento Dati Complessivo ---")
            all_data_loaded = {} 
            all_wheels_to_load = list(set(params["selected_ruote_analisi"] + params["selected_ruote_gioco"]))
            total_wheels_to_load = len(all_wheels_to_load)
            processed_wheels = 0

            if not all_wheels_to_load:
                self.q.put("Nessuna ruota selezionata per analisi o gioco. Analisi interrotta.")
                self.q.put("analysis_complete"); return

            for wheel_key in all_wheels_to_load:
                self.q.put(f"Caricamento dati completi per {RUOTE_DISPONIBILI[wheel_key]}...")
                df_full = None 
                if wheel_key in self.data_cache:
                    df_full = self.data_cache[wheel_key].copy()
                    self.q.put(f"Dati per {RUOTE_DISPONIBILI[wheel_key]} caricati dalla cache.")
                else:
                    content = self.fetch_data_from_github(URL_RUOTE[wheel_key])
                    if content:
                        df_full = self.parse_lotto_data(content, wheel_key)
                        if df_full is not None and not df_full.empty:
                            self.data_cache[wheel_key] = df_full.copy()
                            self.q.put(f"Dati per {RUOTE_DISPONIBILI[wheel_key]} scaricati e parsati.")
                        else:
                            self.q.put(f"Nessun dato valido per {RUOTE_DISPONIBILI[wheel_key]}.")
                    else:
                        self.q.put(f"Download fallito per {RUOTE_DISPONIBILI[wheel_key]}.")
                
                if df_full is not None and not df_full.empty:
                    all_data_loaded[wheel_key] = df_full
                else:
                     self.q.put(f"Attenzione: {RUOTE_DISPONIBILI[wheel_key]} non verrà inclusa per mancanza dati.")
                
                processed_wheels += 1
                self.q.put(("progress", processed_wheels * (30 / total_wheels_to_load if total_wheels_to_load > 0 else 0)))
            
            if not all_data_loaded:
                self.q.put("Nessun dato caricato per alcuna ruota. Controllare connessione o file sorgente. Analisi interrotta.")
                self.q.put("analysis_complete"); return

            self.q.put("\n--- APPLICAZIONE FILTRO PERIODO STORICO BASE ---")
            historical_data_dfs_initial_period = {}
            ruote_analisi_effettive = [] 
            for wheel_key_hist_filter in params["selected_ruote_analisi"]:
                if wheel_key_hist_filter in all_data_loaded: 
                    df_wheel_hist = all_data_loaded[wheel_key_hist_filter]
                    df_hist_filtered = df_wheel_hist[
                        (df_wheel_hist['Data'] >= params["start_date_hist"]) & 
                        (df_wheel_hist['Data'] <= params["end_date_hist"])
                    ]
                    if not df_hist_filtered.empty:
                        historical_data_dfs_initial_period[wheel_key_hist_filter] = df_hist_filtered
                        ruote_analisi_effettive.append(wheel_key_hist_filter)
                        # self.q.put(f"Dati storici base trovati per {RUOTE_DISPONIBILI[wheel_key_hist_filter]}: {len(df_hist_filtered)} estrazioni.") # Meno verboso
                    # else:
                        # self.q.put(f"Nessun dato storico per {RUOTE_DISPONIBILI[wheel_key_hist_filter]} nel periodo base [{params['start_date_hist'].date()}-{params['end_date_hist'].date()}].")
                # else:
                    # self.q.put(f"Dati per la ruota di analisi {RUOTE_DISPONIBILI[wheel_key_hist_filter]} non sono stati caricati precedentemente.")
            
            if not historical_data_dfs_initial_period:
                self.q.put("Nessun dato storico nel periodo base per le ruote di analisi selezionate e caricate. Analisi interrotta.")
                self.q.put("analysis_complete"); return
            
            self.q.put(("progress", 35))

            self.q.put("\n--- APPLICAZIONE FILTRI TEMPORALI AVANZATI (se specificati) ---")
            processed_historical_data_dfs = {} 
            for wheel_key_adv_filter in ruote_analisi_effettive:
                df_hist_original_adv = historical_data_dfs_initial_period[wheel_key_adv_filter]
                df_to_filter_adv = df_hist_original_adv.copy()
                if params["selected_months"]: 
                    # self.q.put(f"Applicazione filtro mesi per {RUOTE_DISPONIBILI[wheel_key_adv_filter]}") # Meno verboso
                    df_to_filter_adv = df_to_filter_adv[df_to_filter_adv['Data'].dt.month.isin(params["selected_months"])]
                    if df_to_filter_adv.empty:
                        # self.q.put(f"Nessun dato per {RUOTE_DISPONIBILI[wheel_key_adv_filter]} dopo filtro mesi.") # Meno verboso
                        continue 
                extraction_index = params["extraction_index_in_month"]
                if not df_to_filter_adv.empty and extraction_index: 
                    # self.q.put(f"Applicazione filtro indice estrazione n.{extraction_index} per {RUOTE_DISPONIBILI[wheel_key_adv_filter]}") # Meno verboso
                    df_to_filter_adv = df_to_filter_adv.sort_values(by='Data') 
                    df_to_filter_adv['AnnoMeseGroup'] = df_to_filter_adv['Data'].dt.to_period('M') 
                    df_to_filter_adv['IndiceNelMese'] = df_to_filter_adv.groupby('AnnoMeseGroup').cumcount() + 1
                    df_to_filter_adv = df_to_filter_adv[df_to_filter_adv['IndiceNelMese'] == extraction_index]
                    df_to_filter_adv = df_to_filter_adv.drop(columns=['AnnoMeseGroup', 'IndiceNelMese'], errors='ignore')
                    if df_to_filter_adv.empty:
                        # self.q.put(f"Nessun dato per {RUOTE_DISPONIBILI[wheel_key_adv_filter]} dopo filtro indice estrazione.") # Meno verboso
                        continue 
                if not df_to_filter_adv.empty:
                    processed_historical_data_dfs[wheel_key_adv_filter] = df_to_filter_adv
                    self.q.put(f"Dati filtrati avanzati per {RUOTE_DISPONIBILI[wheel_key_adv_filter]} (usati per derivazione/trigger): {len(df_to_filter_adv)} estrazioni.")
            
            if not processed_historical_data_dfs: 
                self.q.put("Nessun dato storico è rimasto dopo l'applicazione di tutti i filtri avanzati. Analisi interrotta.")
                self.q.put("analysis_complete"); return
            self.q.put(("progress", 40)) 

            self.q.put(f"\n--- FASE 2: Identificazione Lunghette Candidate Storiche e Loro Backtesting ---")
            lunghezza_serie_da_cercare = params["lunghetta_len"]
            possible_base_lunghettas_counter = Counter()
            num_estrazioni_per_conteggio_base = 0
            if processed_historical_data_dfs:
                for wheel_key_base, df_hist_data_base in processed_historical_data_dfs.items():
                    num_estrazioni_per_conteggio_base += len(df_hist_data_base)
                    for _, row_base in df_hist_data_base.iterrows():
                        try:
                            extracted_numbers_base = sorted([int(row_base[f'N{i}']) for i in range(1, 6)])
                            if len(extracted_numbers_base) >= lunghezza_serie_da_cercare:
                                for combo_base in combinations(extracted_numbers_base, lunghezza_serie_da_cercare):
                                    possible_base_lunghettas_counter[combo_base] += 1
                        except ValueError: continue 
            
            num_base_lunghettas_to_test = 20 
            top_base_lunghettas_to_test_tuples = possible_base_lunghettas_counter.most_common(num_base_lunghettas_to_test)
            self.q.put(f"Conteggio lunghette base ({lunghezza_serie_da_cercare} numeri) completato su {num_estrazioni_per_conteggio_base} estrazioni filtrate.")
            self.q.put(f"Trovate {len(possible_base_lunghettas_counter)} lunghette base uniche. Selezionate le top {len(top_base_lunghettas_to_test_tuples)} per il backtesting.")

            if not top_base_lunghettas_to_test_tuples:
                self.q.put("Nessuna lunghetta base storica trovata da testare. Analisi interrotta.")
                self.q.put("analysis_complete"); return
            self.q.put(("progress", 45))

            all_backtest_results = []
            for i_base, (base_lunghetta_tuple, freq_base) in enumerate(top_base_lunghettas_to_test_tuples):
                base_lunghetta = list(base_lunghetta_tuple)
                # self.q.put(f"\n--- Backtesting Lunghetta Candidata {i_base+1}/{len(top_base_lunghettas_to_test_tuples)}: {sorted(base_lunghetta)} (Freq. grezza: {freq_base}) ---") # Meno verboso

                if len(base_lunghetta) < params['sorte_len']: continue
                candidate_combinations_for_this_base = list(combinations(sorted(base_lunghetta), params['sorte_len']))
                if not candidate_combinations_for_this_base: continue
                
                bt_total_triggers_per_lb = 0
                bt_triggers_con_successo_per_lb = 0
                bt_giocate_effettuate_per_lb = 0
                if not all_data_loaded: self.q.put("  ERRORE: all_data_loaded non disponibile."); continue

                # Itera sugli eventi trigger basati sulle ruote di analisi
                for wheel_key_trigger_analisi in processed_historical_data_dfs.keys():
                    df_triggers_spec_ruota_analisi = processed_historical_data_dfs[wheel_key_trigger_analisi]
                    for _, trigger_row_bt_event in df_triggers_spec_ruota_analisi.sort_values(by='Data').iterrows():
                        data_trigger_bt_event = trigger_row_bt_event['Data']
                        if data_trigger_bt_event + timedelta(days=params['colpi_gioco']*3) > params['end_date_hist']: continue
                        
                        bt_total_triggers_per_lb += 1
                        almeno_una_giocata_possibile_per_trigger = False
                        successo_per_trigger_su_qualsiasi_ruota_gioco = False

                        # Per questo trigger, verifichiamo su TUTTE le selected_ruote_gioco
                        for wheel_key_gioco_bt_verify in params["selected_ruote_gioco"]: # Usa ruote di gioco
                            df_completo_ruota_gioco_bt_verify = all_data_loaded.get(wheel_key_gioco_bt_verify)
                            if df_completo_ruota_gioco_bt_verify is None: continue

                            start_verifica_bt_verify = data_trigger_bt_event + timedelta(days=1)
                            df_finestra_bt_verify = df_completo_ruota_gioco_bt_verify[
                                (df_completo_ruota_gioco_bt_verify['Data'] >= start_verifica_bt_verify) &
                                (df_completo_ruota_gioco_bt_verify['Data'] <= params['end_date_hist'])
                            ].head(params['colpi_gioco'])

                            if df_finestra_bt_verify.empty or len(df_finestra_bt_verify) < params['colpi_gioco']: continue
                            almeno_una_giocata_possibile_per_trigger = True

                            for combo_bt_verify in candidate_combinations_for_this_base:
                                set_combo_bt_verify = set(combo_bt_verify)
                                for colpo_idx_bt_verify, (_, row_bt_colpo_verify) in enumerate(df_finestra_bt_verify.iterrows()):
                                    try:
                                        numeri_estrazione_bt_verify = {int(row_bt_colpo_verify[f'N{k}']) for k in range(1, 6)}
                                        if set_combo_bt_verify.issubset(numeri_estrazione_bt_verify):
                                            self.q.put(f"  BT Lngh.{i_base+1} {sorted(base_lunghetta)}: {params['sorte_text']} {sorted(list(combo_bt_verify))} "
                                                       f"OK su {RUOTE_DISPONIBILI[wheel_key_gioco_bt_verify]} il {row_bt_colpo_verify['Data'].strftime('%Y-%m-%d')} "
                                                       f"(Colpo {colpo_idx_bt_verify + 1} post-trigger {data_trigger_bt_event.strftime('%Y-%m-%d')} da {RUOTE_DISPONIBILI[wheel_key_trigger_analisi]})")
                                            successo_per_trigger_su_qualsiasi_ruota_gioco = True; break
                                    except ValueError: continue
                                if successo_per_trigger_su_qualsiasi_ruota_gioco: break
                            if successo_per_trigger_su_qualsiasi_ruota_gioco: break
                        
                        if almeno_una_giocata_possibile_per_trigger:
                            bt_giocate_effettuate_per_lb += 1
                            if successo_per_trigger_su_qualsiasi_ruota_gioco:
                                bt_triggers_con_successo_per_lb += 1
                
                percent_successo_bt_lb = (bt_triggers_con_successo_per_lb / bt_giocate_effettuate_per_lb) * 100 if bt_giocate_effettuate_per_lb > 0 else 0
                # self.q.put(f"  Risultato Backtest per {sorted(base_lunghetta)}: {bt_triggers_con_successo_per_lb} successi su {bt_giocate_effettuate_per_lb} giocate ({percent_successo_bt_lb:.2f}%) "
                #           f"da {bt_total_triggers_per_lb} eventi trigger disponibili.") # Meno verboso nel loop principale
                all_backtest_results.append({
                    "lunghetta": sorted(base_lunghetta), "success_triggers": bt_triggers_con_successo_per_lb,
                    "plays_made": bt_giocate_effettuate_per_lb, "percent_success": percent_successo_bt_lb,
                    "total_triggers_available": bt_total_triggers_per_lb, "original_freq": freq_base
                })
                self.q.put(("progress", 45 + (i_base + 1) * (40 / len(top_base_lunghettas_to_test_tuples))))

            self.q.put("\n--- FASE 3: Riepilogo Lunghette con Migliore Performance Storica (Backtest) ---")
            sorted_results = sorted(all_backtest_results, key=lambda x: (x["success_triggers"], x["percent_success"], x["original_freq"]), reverse=True)
            final_lunghetta_scelta_per_gioco = None
            if sorted_results:
                self.q.put("Top lunghette candidate ordinate per performance nel backtest:")
                for i_res, res in enumerate(sorted_results):
                    if i_res < 10: 
                        self.q.put(f"  Pos. {i_res+1}: LUNGHETTA {res['lunghetta']} -> "
                                   f"{res['success_triggers']} trigger con successo su {res['plays_made']} giocate ({res['percent_success']:.2f}%) "
                                   f"(Freq. grezza: {res['original_freq']})")
                final_lunghetta_scelta_per_gioco = sorted_results[0]['lunghetta'] # Scegli la migliore
                self.q.put(f"\nLunghetta Selezionata per Gioco Futuro (FASE 4): {final_lunghetta_scelta_per_gioco} "
                           f"(ha avuto {sorted_results[0]['success_triggers']} successi nel backtest su {sorted_results[0]['plays_made']} giocate)")
            else:
                 self.q.put("Nessun risultato di backtest da analizzare. Analisi interrotta.")
                 self.q.put("analysis_complete"); return

            if not final_lunghetta_scelta_per_gioco:
                self.q.put("Impossibile selezionare una lunghetta per il gioco futuro. Analisi interrotta.")
                self.q.put("analysis_complete"); return

            candidate_combinations_final_for_future = list(combinations(final_lunghetta_scelta_per_gioco, params['sorte_len']))
            self.q.put(f"Generate {len(candidate_combinations_final_for_future)} {params['sorte_text']} da {final_lunghetta_scelta_per_gioco} per la verifica futura.")
            self.q.put(("progress", 85))

            # --- (Info Addizionale) Presenza Storica Lorda --- (Opzionale, puoi rimuoverla)
            # ... (blocco FASE 3b come prima, se vuoi tenerlo) ...

# --- FASE 4: Verifica Colpi (nel periodo futuro) ---
            self.q.put(f"\n--- FASE 4: Verifica Giocate per {params['colpi_gioco']} Colpi (nel periodo futuro) ---")
            future_data_start_date = params['end_date_hist'] + timedelta(days=1)
            found_predictions_f4 = [] 
            if not all_data_loaded: 
                self.q.put("ERRORE CRITICO: all_data_loaded non è definito all'inizio della FASE 4.")
                self.q.put("analysis_complete"); return
            
            future_data_dfs_f4 = {} 
            if not params["selected_ruote_gioco"]: self.q.put("ATTENZIONE: Lista ruote di gioco vuota (FASE 4)!")

            for wheel_key_f4 in params["selected_ruote_gioco"]:
                if wheel_key_f4 in all_data_loaded:
                    df_wheel_f4 = all_data_loaded[wheel_key_f4]
                    df_future_f4_data = df_wheel_f4[df_wheel_f4['Data'] >= future_data_start_date].copy()
                    if not df_future_f4_data.empty: future_data_dfs_f4[wheel_key_f4] = df_future_f4_data.sort_values(by='Data')
            
            if not future_data_dfs_f4:
                self.q.put("Nessun dato futuro disponibile per verifica (FASE 4).")
                self.q.put("analysis_complete"); return 
            
            current_progress_f4 = 85
            for i_f4, combo_to_check_f4 in enumerate(candidate_combinations_final_for_future): # Iteriamo su ogni combinazione da giocare
                combo_set_f4 = set(combo_to_check_f4) 
                successo_per_questa_combo_f4 = False # Flag per questa combinazione

                for wheel_key_gioco_verify_f4 in params["selected_ruote_gioco"]: # Iteriamo su ogni ruota di gioco
                    if wheel_key_gioco_verify_f4 not in future_data_dfs_f4: continue
                    
                    df_future_wheel_f4_data = future_data_dfs_f4[wheel_key_gioco_verify_f4]
                    msg_f4_ruota = None # Inizializza msg_f4 per questa ruota e combinazione

                    for colpo_idx_f4, (idx_estrazione_f4, row_f4) in enumerate(df_future_wheel_f4_data.head(params["colpi_gioco"]).iterrows()):
                        try:
                            extracted_numbers_in_draw_f4 = {int(row_f4[f'N{k}']) for k in range(1, 6)} 
                            if combo_set_f4.issubset(extracted_numbers_in_draw_f4):
                                msg_f4_ruota = (f"SUCCESSO FASE4! {params['sorte_text']} {sorted(list(combo_to_check_f4))} "
                                       f"trovato su {RUOTE_DISPONIBILI[wheel_key_gioco_verify_f4]} il {row_f4['Data'].strftime('%Y-%m-%d')} (Colpo {colpo_idx_f4 + 1})")
                                self.q.put((msg_f4_ruota, True))
                                found_predictions_f4.append(msg_f4_ruota)
                                successo_per_questa_combo_f4 = True 
                                break # Esci dal loop dei colpi per questa ruota, successo trovato
                        except ValueError as ve_f4:
                            self.q.put(f"WARN: Errore conversione numero in verifica colpi (FASE 4). Data: {row_f4.get('Data','N/A')}, Ruota: {wheel_key_gioco_verify_f4}. Errore: {ve_f4}")
                            continue 
                    
                    if successo_per_questa_combo_f4: # Se trovato su una ruota, non serve controllare altre ruote per QUESTA combinazione
                        break 
                
                # Aggiorna la progress bar dopo aver processato una combinazione su tutte le sue ruote di gioco
                current_progress_f4 += ( (100-85) / len(candidate_combinations_final_for_future) if candidate_combinations_final_for_future else 0)
                self.q.put(("progress", min(current_progress_f4, 99)))

            # Riepilogo finale della FASE 4
            if not found_predictions_f4:
                self.q.put(f"\nNessuna delle combinazioni di {params['sorte_text']} selezionate è uscita entro {params['colpi_gioco']} colpi sulle ruote di gioco (FASE 4).")
            else:
                self.q.put(f"\n--- Riepilogo Successi (FASE 4) ({len(found_predictions_f4)} trovati) ---")
                for fp_msg_f4_item in found_predictions_f4: # Nome variabile diverso per evitare confusione
                    self.q.put((fp_msg_f4_item, True))

        except Exception as e:
            self.q.put(f"Errore critico durante l'analisi: {e}\n{traceback.format_exc()}")
        finally:
            self.q.put("analysis_complete")

    def process_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    if item[0] == "progress": self.progress_bar["value"] = item[1]
                    else: self.log_message(item[0], is_result=item[1])
                elif item == "analysis_complete":
                    self.log_message("\nAnalisi completata.")
                    self.progress_bar["value"] = 100
                    self.analysis_running = False
                    self.start_button.config(state=tk.NORMAL)
                else: self.log_message(str(item))
        except queue.Empty: pass
        finally: self.master.after(100, self.process_queue)

# Il blocco if __name__ == '__main__': rimane invariato
if __name__ == '__main__':
    root = tk.Tk()
    app = LottoApp(root)
    root.mainloop()