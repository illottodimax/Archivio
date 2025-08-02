# ==============================================================================
# SEZIONE IMPORT: Tutte le librerie necessarie
# ==============================================================================
import io
import os
import itertools
import threading
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import requests
from tkcalendar import DateEntry

# ==============================================================================
# SEZIONE 1: IL MOTORE DI CALCOLO
# ==============================================================================
class LottoAnalyzer:
    def __init__(self, data_source='github', local_path=None, status_callback=None):
        self.estrazioni = {}
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
        self.GITHUB_USER = "illottodimax"; self.GITHUB_REPO = "Archivio"; self.GITHUB_BRANCH = "main"
        self.URL_RUOTE = {key: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{value.upper()}.txt' for key, value in self.RUOTE_DISPONIBILI.items()}
        if data_source == 'local' and not local_path: raise ValueError("Il percorso locale 'local_path' è necessario.")
        self.data_source = data_source; self.local_path = local_path
        self.status_callback = status_callback

    def _update_status(self, message):
        if self.status_callback: self.status_callback(message)

    def carica_dati(self, ruote=None):
        if ruote is None: ruote_da_caricare = self.RUOTE_DISPONIBILI.keys()
        else: ruote_da_caricare = [r.upper() for r in ruote if r.upper() in self.RUOTE_DISPONIBILI]
        for ruota_key in ruote_da_caricare:
            if ruota_key not in self.estrazioni:
                nome_file_log = ""
                try:
                    nome_file = f"{self.RUOTE_DISPONIBILI[ruota_key].upper()}.txt"; nome_file_log = nome_file
                    self._update_status(f"Caricamento dati per {self.RUOTE_DISPONIBILI[ruota_key]}...")
                    data_content = None
                    if self.data_source == 'github':
                        response = requests.get(self.URL_RUOTE[ruota_key]); response.raise_for_status()
                        data_content = response.text
                    else:
                        file_path = os.path.join(self.local_path, nome_file)
                        with open(file_path, 'r', encoding='utf-8') as f: data_content = f.read()
                    nomi_colonne = ['Data', 'Ruota_File', 'N1', 'N2', 'N3', 'N4', 'N5']
                    formato_data = '%Y/%m/%d'
                    df = pd.read_csv(io.StringIO(data_content), sep=r'\s+|\t', header=None, names=nomi_colonne, engine='python')
                    df['Data'] = pd.to_datetime(df['Data'], format=formato_data)
                    self.estrazioni[ruota_key] = df.sort_values(by='Data').reset_index(drop=True)
                    self._update_status(f"Dati per {self.RUOTE_DISPONIBILI[ruota_key]} caricati.")
                except Exception as e:
                    err_msg = str(e)
                    if "match format" in err_msg: err_msg += f"\n\nIl programma si aspettava il formato data '{formato_data}' e colonne '{nomi_colonne}'."
                    raise Exception(f"ERRORE per {nome_file_log}: {err_msg}")
        return True
    
    def backtest_decina(self, previsione_virtuale, colpi_di_gioco):
        self._update_status("Esecuzione backtest duale...")
        ruota = previsione_virtuale['ruota']; df = self.estrazioni[ruota]
        data_inizio_gioco = previsione_virtuale['data_previsione']; df_successive = df[df['Data'] > data_inizio_gioco].copy()
        df_gioco = df_successive.head(colpi_di_gioco); esito_estratto = {'trovato': False, 'stato': 'NEGATIVO'}; esito_ambo = {'trovato': False, 'stato': 'NEGATIVO'}
        ambate_da_cercare = set(previsione_virtuale.get('ambate_convergenti', [])); ambi_da_cercare = previsione_virtuale.get('ambi_frequenti', [])
        for i, riga in df_gioco.iterrows():
            numeri_estratti = {riga['N1'], riga['N2'], riga['N3'], riga['N4'], riga['N5']}; colpo_corrente = df_successive.index.get_loc(i) + 1
            if ambate_da_cercare and not esito_estratto['trovato']:
                intersezione = numeri_estratti.intersection(ambate_da_cercare)
                if intersezione: esito_estratto = {'trovato': True, 'stato': 'VINTA', 'numero_uscito': list(intersezione)[0], 'data_uscita': riga['Data'], 'colpo': colpo_corrente}
            if ambi_da_cercare and not esito_ambo['trovato']:
                for ambo in ambi_da_cercare:
                    if set(ambo).issubset(numeri_estratti): esito_ambo = {'trovato': True, 'stato': 'VINTA', 'ambo_uscito': ambo, 'data_uscita': riga['Data'], 'colpo': colpo_corrente}; break
        colpi_passati = len(df_gioco)
        if not esito_estratto['trovato'] and colpi_passati < colpi_di_gioco: esito_estratto['stato'] = 'IN GIOCO'; esito_estratto['colpi_passati'] = colpi_passati
        if ambi_da_cercare:
            if not esito_ambo['trovato'] and colpi_passati < colpi_di_gioco: esito_ambo['stato'] = 'IN GIOCO'; esito_ambo['colpi_passati'] = colpi_passati
        else: esito_ambo = {'trovato': False, 'stato': 'NON_APPLICABILE'}
        return {'esito_estratto': esito_estratto, 'esito_ambo': esito_ambo}

    def verifica_esito_previsioni(self, lista_previsioni, colpi_di_gioco=12):
        self._update_status("Verifica esiti (Ambate e Laterali)...")
        for previsione in lista_previsioni:
            ruota = previsione['ruota']; df = self.estrazioni[ruota]; data_inizio_gioco = previsione['data_previsione']; df_successive = df[df['Data'] > data_inizio_gioco].copy()
            df_gioco = df_successive.head(colpi_di_gioco); ambate_da_cercare = set(previsione['ambate_previste']); laterali_da_cercare = set()
            for ambata in ambate_da_cercare: laterali_da_cercare.add(ambata - 1 if ambata > 1 else 90); laterali_da_cercare.add(ambata + 1 if ambata < 90 else 1)
            laterali_da_cercare = laterali_da_cercare - ambate_da_cercare; previsione['esito'] = {'trovato': False, 'stato': 'NEGATIVO'}
            for i, riga in df_gioco.iterrows():
                numeri_estratti = {riga['N1'], riga['N2'], riga['N3'], riga['N4'], riga['N5']}; intersezione = numeri_estratti.intersection(ambate_da_cercare)
                if intersezione: previsione['esito'] = {'trovato': True, 'stato': 'VINTA', 'numero_uscito': list(intersezione)[0], 'data_uscita': riga['Data'], 'colpo': df_successive.index.get_loc(i) + 1}; break
            if not previsione['esito']['trovato'] and laterali_da_cercare:
                for i, riga in df_gioco.iterrows():
                    numeri_estratti = {riga['N1'], riga['N2'], riga['N3'], riga['N4'], riga['N5']}; intersezione_laterale = numeri_estratti.intersection(laterali_da_cercare)
                    if intersezione_laterale: previsione['esito'] = {'trovato': True, 'stato': 'LATERALE', 'numero_uscito': list(intersezione_laterale)[0], 'data_uscita': riga['Data'], 'colpo': df_successive.index.get_loc(i) + 1}; break
            if not previsione['esito']['trovato'] and len(df_gioco) < colpi_di_gioco: previsione['esito']['stato'] = 'IN GIOCO'; previsione['esito']['colpi_passati'] = len(df_gioco)
    
    @staticmethod
    def _get_decina(numero):
        if 1 <= numero <= 89: return (numero - 1) // 10
        elif numero == 90: return 8
        return -1
        
    @staticmethod
    def _trova_ambi_in_decina(estrazione):
        ambi_trovati = [];
        for n1, n2 in itertools.combinations(sorted(estrazione), 2):
            if LottoAnalyzer._get_decina(n1) == LottoAnalyzer._get_decina(n2): ambi_trovati.append(tuple(sorted((n1,n2))))
        return ambi_trovati
        
    def analizza_previsioni_da_segmenti(self, ruota, data_inizio, data_fine, num_segmenti_da_confrontare=2, colpi_di_controllo_ritroso=5):
        ruota = ruota.upper();
        if ruota not in self.estrazioni: self.carica_dati(ruote=[ruota])
        df = self.estrazioni[ruota]; df_analisi = df[(df['Data'] >= pd.to_datetime(data_inizio)) & (df['Data'] <= pd.to_datetime(data_fine))].copy()
        segmenti_attivi_per_decina = defaultdict(lambda: deque(maxlen=num_segmenti_da_confrontare)); previsioni_trovate = []
        for i, (index, riga) in enumerate(df_analisi.iterrows()):
            if i % 100 == 0: self._update_status(f"Analisi Segmenti: {riga['Data'].strftime('%d-%m-%Y')}...")
            data_corrente = riga['Data']; numeri_estratti_correnti = {riga['N1'], riga['N2'], riga['N3'], riga['N4'], riga['N5']}
            for decina, lista_segmenti in segmenti_attivi_per_decina.items():
                for segmento in lista_segmenti: segmento['numeri_rimanenti'].difference_update(numeri_estratti_correnti)
            ambi_correnti = self._trova_ambi_in_decina(list(numeri_estratti_correnti))
            if ambi_correnti:
                for ambo in ambi_correnti:
                    n1, n2 = ambo; decina = self._get_decina(n1); segmento_originale = set(range(n1, n2 + 1))
                    indice_fine_ritroso = index; indice_inizio_ritroso = max(0, index - colpi_di_controllo_ritroso)
                    df_ritroso = df.iloc[indice_inizio_ritroso:index]
                    numeri_usciti_nel_ritroso = set(df_ritroso[['N1', 'N2', 'N3', 'N4', 'N5']].values.ravel())
                    numeri_da_escludere = numeri_usciti_nel_ritroso.union(set(ambo)); numeri_rimanenti = segmento_originale - numeri_da_escludere
                    nuovo_segmento = {'data_creazione': data_corrente, 'ambo_generatore': ambo, 'segmento_originale': sorted(list(segmento_originale)), 'numeri_rimanenti': numeri_rimanenti}
                    segmenti_attivi_per_decina[decina].append(nuovo_segmento)
                    segmenti_da_controllare = segmenti_attivi_per_decina[decina]
                    if len(segmenti_da_controllare) == num_segmenti_da_confrontare:
                        lista_set_rimanenti = [s['numeri_rimanenti'] for s in segmenti_da_controllare if s['numeri_rimanenti']]
                        if len(lista_set_rimanenti) == num_segmenti_da_confrontare:
                            previsione = set.intersection(*lista_set_rimanenti)
                            if previsione:
                                previsione_info = {'data_previsione': data_corrente, 'ruota': ruota, 'decina': decina, 'ambate_previste': sorted(list(previsione)), 'segmenti_convergenti': list(segmenti_da_controllare), 'colpi_ritroso_usati': colpi_di_controllo_ritroso}
                                previsioni_trovate.append(previsione_info)
        return previsioni_trovate

    @staticmethod
    def _get_laterali(n):
        prec = n - 1 if n > 1 else 90; succ = n + 1 if n < 90 else 1
        return {prec, n, succ}

    def analizza_decina_avanzata(self, ruota, decina_id, data_riferimento, lookback_colpi=18):
        ruota = ruota.upper();
        if ruota not in self.estrazioni: self.carica_dati(ruote=[ruota])
        df = self.estrazioni[ruota]; data_fine = pd.to_datetime(data_riferimento)
        df_periodo = df[df['Data'] <= data_fine].tail(lookback_colpi)
        numeri_della_decina = set(range(decina_id * 10 + 1, decina_id * 10 + 11))
        estratti_nel_periodo = df_periodo[['N1', 'N2', 'N3', 'N4', 'N5']].values.ravel()
        freq_decina = Counter(n for n in estratti_nel_periodo if n in numeri_della_decina)
        numeri_usciti = set(freq_decina.keys()); direttrici_generali = set()
        for n in numeri_usciti: direttrici_generali.update(self._get_laterali(n))
        direttrici_generali = direttrici_generali.intersection(numeri_della_decina)
        ambi_nella_decina = []
        for _, riga in df_periodo.iterrows():
            estratti_riga = {riga['N1'], riga['N2'], riga['N3'], riga['N4'], riga['N5']}; ambi_riga = self._trova_ambi_in_decina(estratti_riga)
            for ambo in ambi_riga:
                if ambo[0] in numeri_della_decina: ambi_nella_decina.append(ambo)
        freq_ambi = Counter(n for ambo in ambi_nella_decina for n in ambo); direttrici_da_ambi = set()
        for ambo in ambi_nella_decina: direttrici_da_ambi.update(self._get_laterali(ambo[0])); direttrici_da_ambi.update(self._get_laterali(ambo[1]))
        direttrici_da_ambi = direttrici_da_ambi.intersection(numeri_della_decina)
        punteggio = Counter()
        for n, freq in freq_decina.items(): punteggio[n] += freq
        for n in direttrici_generali: punteggio[n] += 2
        for n, freq in freq_ambi.items(): punteggio[n] += freq * 2
        for n in direttrici_da_ambi: punteggio[n] += 3
        return {'frequenze': freq_decina, 'direttrici_generali': sorted(list(direttrici_generali)), 'ambi_trovati': ambi_nella_decina, 'frequenze_ambi': freq_ambi, 'direttrici_da_ambi': sorted(list(direttrici_da_ambi)), 'punteggio_finale': punteggio}

# ==============================================================================
# SEZIONE 2: L'INTERFACCIA GRAFICA
# ==============================================================================
class SegmentiApp:
    def __init__(self, root):
        self.root = root; self.root.title("I Segmenti di Max 3.0"); self.root.geometry("900x750")
        self.RUOTE_MAP = {'Bari':'BA', 'Cagliari':'CA', 'Firenze':'FI', 'Genova':'GE', 'Milano':'MI', 'Napoli':'NA', 'Palermo':'PA', 'Roma':'RO', 'Torino':'TO', 'Venezia':'VE', 'Nazionale':'NZ'}
        self.DECINE_MAP = {f"{i*10+1}-{i*10+10}": i for i in range(9)}
        self.style = ttk.Style(self.root); self.style.theme_use('clam')
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        path_frame = ttk.LabelFrame(main_frame, text="Impostazioni Dati Locali", padding="10"); path_frame.pack(fill=tk.X, expand=False, pady=(0, 10)); path_frame.columnconfigure(1, weight=1)
        ttk.Label(path_frame, text="Cartella Dati:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W); self.local_path_var = tk.StringVar(); path_entry = ttk.Entry(path_frame, textvariable=self.local_path_var, state='readonly'); path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        browse_button = ttk.Button(path_frame, text="Sfoglia...", command=self.browse_for_folder); browse_button.grid(row=0, column=2, padx=5, pady=5)
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop'); self.local_path_var.set(desktop_path)
        self.notebook = ttk.Notebook(main_frame); self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_segmenti = ttk.Frame(self.notebook, padding="10"); self.tab_direttrici = ttk.Frame(self.notebook, padding="10"); self.tab_radar = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_segmenti, text="Analisi Segmenti"); self.notebook.add(self.tab_direttrici, text="Analisi Decine"); self.notebook.add(self.tab_radar, text="🔥 Radar di Max")
        self.create_segmenti_widgets(self.tab_segmenti); self.create_direttrici_widgets(self.tab_direttrici); self.create_radar_widgets(self.tab_radar)
        self.status_var = tk.StringVar(value="Pronto."); status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5"); status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.analyzer = None

    def browse_for_folder(self):
        initial_dir = self.local_path_var.get();
        if not os.path.isdir(initial_dir): initial_dir = os.path.expanduser('~')
        folder_selected = filedialog.askdirectory(initialdir=initial_dir, title="Seleziona la cartella con i file .txt")
        if folder_selected: self.local_path_var.set(folder_selected); self.update_status(f"Nuovo percorso locale impostato: {folder_selected}")
    
    def create_segmenti_widgets(self, parent_frame):
        input_frame = ttk.LabelFrame(parent_frame, text="Impostazioni Analisi Segmenti", padding="10"); input_frame.pack(fill=tk.X, expand=False); input_frame.columnconfigure(1, weight=1); input_frame.columnconfigure(3, weight=1)
        self.source_var_seg = tk.StringVar(value="github"); ttk.Label(input_frame, text="Fonte Dati:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W); ttk.Radiobutton(input_frame, text="Locali", variable=self.source_var_seg, value="local").grid(row=0, column=1, sticky=tk.W); ttk.Radiobutton(input_frame, text="GitHub", variable=self.source_var_seg, value="github").grid(row=0, column=1, sticky=tk.E)
        ttk.Label(input_frame, text="Ruota:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W); self.ruota_var_seg = tk.StringVar(); ttk.Combobox(input_frame, textvariable=self.ruota_var_seg, values=list(self.RUOTE_MAP.keys()), state="readonly").grid(row=1, column=1, sticky=tk.EW); self.ruota_var_seg.set("Bari")
        ttk.Label(input_frame, text="Data Inizio:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W); self.data_inizio_seg = DateEntry(input_frame, date_pattern='y-mm-dd', width=12); self.data_inizio_seg.set_date(datetime(datetime.now().year, 1, 1)); self.data_inizio_seg.grid(row=2, column=1, sticky=tk.EW)
        ttk.Label(input_frame, text="Data Fine:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W); self.data_fine_seg = DateEntry(input_frame, date_pattern='y-mm-dd', width=12); self.data_fine_seg.set_date(datetime.now()); self.data_fine_seg.grid(row=3, column=1, sticky=tk.EW)
        ttk.Label(input_frame, text="Controllo Indietro:").grid(row=0, column=2, padx=(20, 5), pady=5, sticky=tk.W); self.ritroso_var_seg = tk.IntVar(value=5); ttk.Spinbox(input_frame, from_=1, to=50, textvariable=self.ritroso_var_seg, width=5).grid(row=0, column=3, sticky=tk.W)
        ttk.Label(input_frame, text="Verifica Esito:").grid(row=1, column=2, padx=(20, 5), pady=5, sticky=tk.W); self.gioco_var_seg = tk.IntVar(value=12); ttk.Spinbox(input_frame, from_=1, to=50, textvariable=self.gioco_var_seg, width=5).grid(row=1, column=3, sticky=tk.W)
        self.run_button_seg = ttk.Button(input_frame, text="AVVIA ANALISI SEGMENTI", command=lambda: self.start_analysis_thread('segmenti')); self.run_button_seg.grid(row=3, column=2, columnspan=2, pady=10)
        output_frame = ttk.LabelFrame(parent_frame, text="Risultati Segmenti", padding="10"); output_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.results_text_seg = tk.Text(output_frame, wrap=tk.WORD, state='disabled', height=10, font=("Courier New", 9)); scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.results_text_seg.yview); self.results_text_seg.config(yscrollcommand=scrollbar.set); self.results_text_seg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_direttrici_widgets(self, parent_frame):
        input_frame = ttk.LabelFrame(parent_frame, text="Impostazioni Analisi Decine", padding="10"); input_frame.pack(fill=tk.X, expand=False); input_frame.columnconfigure(1, weight=1)
        self.source_var_dir = tk.StringVar(value="github"); ttk.Label(input_frame, text="Fonte Dati:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W); ttk.Radiobutton(input_frame, text="Locali", variable=self.source_var_dir, value="local").grid(row=0, column=1, sticky=tk.W); ttk.Radiobutton(input_frame, text="GitHub", variable=self.source_var_dir, value="github").grid(row=0, column=1, sticky=tk.E)
        ttk.Label(input_frame, text="Ruota:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W); self.ruota_var_dir = tk.StringVar(); self.ruota_var_dir.set("Bari"); ttk.Combobox(input_frame, textvariable=self.ruota_var_dir, values=list(self.RUOTE_MAP.keys()), state="readonly").grid(row=1, column=1, columnspan=3, sticky=tk.EW)
        ttk.Label(input_frame, text="Decina:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W); self.decina_var_dir = tk.StringVar(); self.decina_var_dir.set("51-60"); ttk.Combobox(input_frame, textvariable=self.decina_var_dir, values=list(self.DECINE_MAP.keys()), state="readonly").grid(row=2, column=1, columnspan=3, sticky=tk.EW)
        ttk.Label(input_frame, text="Data di Riferimento:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W); self.data_riferimento_dir = DateEntry(input_frame, date_pattern='y-mm-dd', width=12); self.data_riferimento_dir.set_date(datetime.now()); self.data_riferimento_dir.grid(row=3, column=1, columnspan=3, sticky=tk.EW)
        ttk.Label(input_frame, text="Colpi di Analisi:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W); self.lookback_var_dir = tk.IntVar(value=18); ttk.Spinbox(input_frame, from_=5, to=50, textvariable=self.lookback_var_dir, width=5).grid(row=4, column=1, sticky=tk.W)
        ttk.Label(input_frame, text="Numeri per Ambo:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W); self.numeri_ambo_var = tk.IntVar(value=4); ttk.Spinbox(input_frame, from_=3, to=5, textvariable=self.numeri_ambo_var, width=5).grid(row=5, column=1, sticky=tk.W)
        ttk.Label(input_frame, text="Verifica Esito (colpi):").grid(row=4, column=2, padx=(20, 5), pady=5, sticky=tk.W); self.gioco_var_dir = tk.IntVar(value=12); ttk.Spinbox(input_frame, from_=1, to=50, textvariable=self.gioco_var_dir, width=5).grid(row=4, column=3, sticky=tk.W)
        self.run_button_dir = ttk.Button(input_frame, text="AVVIA ANALISI E BACKTEST", command=lambda: self.start_analysis_thread('direttrici')); self.run_button_dir.grid(row=5, column=2, columnspan=2, pady=10)
        output_frame = ttk.LabelFrame(parent_frame, text="Risultati Analisi Decina e Backtest", padding="10"); output_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.results_text_dir = tk.Text(output_frame, wrap=tk.WORD, state='disabled', height=10, font=("Courier New", 10)); scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.results_text_dir.yview); self.results_text_dir.config(yscrollcommand=scrollbar.set); self.results_text_dir.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_radar_widgets(self, parent_frame):
        input_frame = ttk.LabelFrame(parent_frame, text="Impostazioni Radar", padding="10"); input_frame.pack(fill=tk.X, expand=False); input_frame.columnconfigure(1, weight=1)
        self.source_var_radar = tk.StringVar(value="github"); ttk.Label(input_frame, text="Fonte Dati:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W); ttk.Radiobutton(input_frame, text="Locali", variable=self.source_var_radar, value="local").grid(row=0, column=1, sticky=tk.W); ttk.Radiobutton(input_frame, text="GitHub", variable=self.source_var_radar, value="github").grid(row=0, column=1, sticky=tk.E)
        ttk.Label(input_frame, text="Data di Riferimento:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W); self.data_riferimento_radar = DateEntry(input_frame, date_pattern='y-mm-dd', width=12); self.data_riferimento_radar.set_date(datetime.now()); self.data_riferimento_radar.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(input_frame, text="Colpi di Analisi:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W); self.lookback_var_radar = tk.IntVar(value=18); ttk.Spinbox(input_frame, from_=5, to=50, textvariable=self.lookback_var_radar, width=5).grid(row=2, column=1, sticky=tk.W)
        self.run_button_radar = ttk.Button(input_frame, text="ATTIVA RADAR", style="Accent.TButton", command=lambda: self.start_analysis_thread('radar')); self.run_button_radar.grid(row=3, column=0, columnspan=2, pady=10)
        output_frame = ttk.LabelFrame(parent_frame, text="Classifica Radar", padding="10"); output_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.results_text_radar = tk.Text(output_frame, wrap=tk.WORD, state='disabled', height=10, font=("Courier New", 10)); scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.results_text_radar.yview); self.results_text_radar.config(yscrollcommand=scrollbar.set); self.results_text_radar.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.style.configure("Accent.TButton", foreground="white", background="navy")
    
    def start_analysis_thread(self, analysis_type):
        self.run_button_seg.config(state="disabled"); self.run_button_dir.config(state="disabled"); self.run_button_radar.config(state="disabled")
        if analysis_type == 'segmenti': target_func = self.run_analysis_segmenti; text_widget = self.results_text_seg
        elif analysis_type == 'direttrici': target_func = self.run_analysis_direttrici; text_widget = self.results_text_dir
        else: target_func = self.run_analysis_radar; text_widget = self.results_text_radar
        text_widget.config(state='normal'); text_widget.delete('1.0', tk.END); text_widget.config(state='disabled')
        analysis_thread = threading.Thread(target=target_func); analysis_thread.daemon = True; analysis_thread.start()

    def run_analysis_segmenti(self):
        try:
            source = self.source_var_seg.get(); percorso_locale = self.local_path_var.get()
            if source == 'local' and not os.path.isdir(percorso_locale): messagebox.showerror("Errore Percorso", "Se si utilizza la fonte 'Locali', specificare una cartella valida."); self.update_status("Errore: percorso locale non valido."); self.root.after(0, self.enable_buttons); return
            ruota_nome = self.ruota_var_seg.get(); ruota_sigla = self.RUOTE_MAP[ruota_nome]; data_inizio = self.data_inizio_seg.get_date().strftime('%Y-%m-%d'); data_fine = self.data_fine_seg.get_date().strftime('%Y-%m-%d')
            colpi_ritroso = self.ritroso_var_seg.get(); colpi_gioco = self.gioco_var_seg.get()
            self.update_status(f"Inizializzazione per fonte '{source}'...")
            self.analyzer = LottoAnalyzer(data_source=source, local_path=percorso_locale, status_callback=self.update_status)
            previsioni = self.analyzer.analizza_previsioni_da_segmenti(ruota=ruota_sigla, data_inizio=data_inizio, data_fine=data_fine, colpi_di_controllo_ritroso=colpi_ritroso)
            if previsioni: self.analyzer.verifica_esito_previsioni(previsioni, colpi_gioco)
            report = self.genera_report_segmenti(previsioni, ruota_nome, colpi_gioco)
            self.update_status("Analisi Segmenti completata.")
        except Exception as e: report = f"ERRORE: {e}"; self.update_status("Errore!"); messagebox.showerror("Errore", f"Si è verificato un errore:\n{e}")
        self.root.after(0, self.display_results, report, 'segmenti')

    def run_analysis_direttrici(self):
        try:
            source = self.source_var_dir.get(); percorso_locale = self.local_path_var.get()
            if source == 'local' and not os.path.isdir(percorso_locale): messagebox.showerror("Errore Percorso", "Se si utilizza la fonte 'Locali', specificare una cartella valida."); self.update_status("Errore: percorso locale non valido."); self.root.after(0, self.enable_buttons); return
            ruota_nome = self.ruota_var_dir.get(); ruota_sigla = self.RUOTE_MAP[ruota_nome]; decina_str = self.decina_var_dir.get(); decina_id = self.DECINE_MAP[decina_str]
            data_riferimento = pd.to_datetime(self.data_riferimento_dir.get_date()); lookback = self.lookback_var_dir.get(); colpi_gioco = self.gioco_var_dir.get(); numeri_per_ambo = self.numeri_ambo_var.get()
            self.update_status(f"Inizializzazione per fonte '{source}'...")
            self.analyzer = LottoAnalyzer(data_source=source, local_path=percorso_locale, status_callback=self.update_status)
            risultati = self.analyzer.analizza_decina_avanzata(ruota=ruota_sigla, decina_id=decina_id, data_riferimento=data_riferimento, lookback_colpi=lookback)
            esiti_backtest = None
            if risultati['punteggio_finale']:
                ambate_convergenti = [num for num, score in risultati['punteggio_finale'].most_common(3)]; numeri_frequenti = [num for num, score in risultati['frequenze'].most_common(numeri_per_ambo)]; ambi_frequenti = list(itertools.combinations(numeri_frequenti, 2)) if len(numeri_frequenti) >= 2 else []
                previsione_virtuale = {'ruota': ruota_sigla, 'data_previsione': data_riferimento, 'ambate_convergenti': ambate_convergenti, 'ambi_frequenti': ambi_frequenti}
                esiti_backtest = self.analyzer.backtest_decina(previsione_virtuale, colpi_gioco)
            report = self.genera_report_direttrici(risultati, ruota_nome, decina_str, lookback, esiti_backtest, colpi_gioco, numeri_per_ambo)
            self.update_status("Analisi Decina completata.")
        except Exception as e: report = f"ERRORE: {e}"; self.update_status("Errore!"); messagebox.showerror("Errore", f"Si è verificato un errore:\n{e}")
        self.root.after(0, self.display_results, report, 'direttrici')

    def run_analysis_radar(self):
        try:
            source = self.source_var_radar.get(); percorso_locale = self.local_path_var.get()
            if source == 'local' and not os.path.isdir(percorso_locale): messagebox.showerror("Errore Percorso", "Se si utilizza la fonte 'Locali', specificare una cartella valida."); self.update_status("Errore: percorso locale non valido."); self.root.after(0, self.enable_buttons); return
            data_riferimento = self.data_riferimento_radar.get_date(); lookback = self.lookback_var_radar.get()
            self.update_status("Inizializzazione Radar..."); self.analyzer = LottoAnalyzer(data_source=source, local_path=percorso_locale, status_callback=self.update_status)
            all_results = []
            for ruota_nome, ruota_sigla in self.RUOTE_MAP.items():
                self.analyzer.carica_dati(ruote=[ruota_sigla])
                for decina_str, decina_id in self.DECINE_MAP.items():
                    self.update_status(f"Radar: Analisi {ruota_nome} - Decina {decina_str}...")
                    risultati_analisi = self.analyzer.analizza_decina_avanzata(ruota_sigla, decina_id, data_riferimento, lookback)
                    total_score = sum(risultati_analisi['punteggio_finale'].values())
                    if total_score > 0: all_results.append({'ruota': ruota_nome, 'decina': decina_str, 'punteggio': total_score})
            sorted_results = sorted(all_results, key=lambda x: x['punteggio'], reverse=True)
            report = self.genera_report_radar(sorted_results, data_riferimento.strftime('%d-%m-%Y'))
            self.update_status("Scansione Radar completata.")
        except Exception as e: report = f"ERRORE: {e}"; self.update_status("Errore!"); messagebox.showerror("Errore", f"Si è verificato un errore:\n{e}")
        self.root.after(0, self.display_results, report, 'radar')
    
    def display_results(self, report, analysis_type):
        if analysis_type == 'segmenti': text_widget = self.results_text_seg
        elif analysis_type == 'direttrici': text_widget = self.results_text_dir
        else: text_widget = self.results_text_radar
        text_widget.config(state='normal'); text_widget.delete('1.0', tk.END); text_widget.insert(tk.END, report); text_widget.config(state='disabled')
        self.enable_buttons()

    def enable_buttons(self):
        self.run_button_seg.config(state="normal"); self.run_button_dir.config(state="normal"); self.run_button_radar.config(state="normal")
        
    def update_status(self, message):
        self.root.after(0, self.status_var.set, message)

    def genera_report_segmenti(self, lista_previsioni, ruota_nome, colpi_gioco_totali):
        if not lista_previsioni: return f"\nNessuna previsione trovata per {ruota_nome}."
        report_lines = [f"--- TROVATE {len(lista_previsioni)} PREVISIONI ---"]; vinte = 0; vinte_laterali = 0
        for i, prev in enumerate(lista_previsioni):
            ambate_str = ", ".join(map(str, prev['ambate_previste'])); laterali = set()
            for ambata in prev['ambate_previste']: laterali.add(ambata - 1 if ambata > 1 else 90); laterali.add(ambata + 1 if ambata < 90 else 1)
            laterali_str = ", ".join(map(str, sorted(list(laterali - set(prev['ambate_previste'])))))
            report_lines.append(f"\n================ PREVISIONE #{i+1} ================"); report_lines.append(f"Data Elaborazione: {prev['data_previsione'].strftime('%d/%m/%Y')} | Ruota: {prev['ruota']}")
            esito = prev.get('esito', {});
            if esito.get('stato') == 'VINTA': vinte += 1; report_lines.append(f"✅ ESITO: VINTA! Uscito il {esito['numero_uscito']} il {esito['data_uscita'].strftime('%d/%m/%Y')} al {esito['colpo']}° colpo.")
            elif esito.get('stato') == 'LATERALE': vinte_laterali += 1; report_lines.append(f"🎯 ESITO LATERALE: Uscito il {esito['numero_uscito']} il {esito['data_uscita'].strftime('%d/%m/%Y')} al {esito['colpo']}° colpo.")
            elif esito.get('stato') == 'IN GIOCO': report_lines.append(f"🟡 ESITO: IN GIOCO! (verificati {esito.get('colpi_passati', 0)} colpi su {colpi_gioco_totali})")
            else: report_lines.append(f"❌ ESITO: NEGATIVO (non uscito nei colpi verificati).")
            report_lines.append(f"🔥 AMBATE PREVISTE: {ambate_str}"); report_lines.append(f"🎯 Laterali Suggeriti: {laterali_str}"); report_lines.append("-" * 25); report_lines.append("Generata dalla convergenza dei seguenti segmenti:")
            for j, seg in enumerate(prev['segmenti_convergenti']):
                ambo_str = f"{seg['ambo_generatore'][0]}-{seg['ambo_generatore'][1]}"; rimanenti_str = ", ".join(map(str, sorted(list(seg['numeri_rimanenti'])))) if seg['numeri_rimanenti'] else "Nessuno"
                report_lines.append(f"  Segmento {j+1}: Creato il {seg['data_creazione'].strftime('%d/%m/%Y')} dall'ambo {ambo_str}"); report_lines.append(f"    -> Rimanenti al {prev['data_previsione'].strftime('%d/%m/%Y')}: {{{rimanenti_str}}}")
            report_lines.append("==============================================")
        if lista_previsioni: report_lines.append(f"\nRIEPILOGO FINALE: {vinte} vinte dirette + {vinte_laterali} per laterale su {len(lista_previsioni)} totali.")
        return "\n".join(report_lines)

    def genera_report_direttrici(self, res, ruota, decina, lookback, esiti, colpi_gioco, num_per_ambo):
        report = []; report.append(f"--- ANALISI AVANZATA DECINA {decina} SU {ruota} ---"); report.append(f"Periodo: Ultime {lookback} estrazioni"); report.append("="*55)
        report.append("\n1. ANALISI FREQUENZA SINGOLI NUMERI");
        if not res['frequenze']: report.append("  Nessun numero della decina uscito nel periodo.")
        else:
            for num, freq in res['frequenze'].most_common(): report.append(f"  - Numero {num}: uscito {freq} volta/e")
        report.append("\n2. ANALISI DIRETTRICI GENERALI");
        if not res['direttrici_generali']: report.append("  Nessuna direttrice generata.")
        else: report.append(f"  Numeri indicati: {', '.join(map(str, res['direttrici_generali']))}")
        report.append("\n3. ANALISI AMBI IN DECINA");
        if not res['ambi_trovati']: report.append("  Nessun ambo in decina sortito nel periodo.")
        else:
            report.append(f"  Ambi trovati: {'; '.join([f'{a[0]}-{a[1]}' for a in res['ambi_trovati']])}"); report.append("  Forza dei numeri (partecipazione ad ambi):")
            for num, freq in res['frequenze_ambi'].most_common(): report.append(f"    - Numero {num}: presente in {freq} ambo/i")
            report.append("  Direttrici generate dagli ambi:"); report.append(f"    Numeri indicati: {', '.join(map(str, res['direttrici_da_ambi']))}")
        report.append("\n" + "="*55); report.append("4. CONVERGENZA E PUNTEGGIO FINALE")
        if not res['punteggio_finale']: report.append("  Nessuna convergenza trovata.")
        else:
            top_3_conv = res['punteggio_finale'].most_common(3); top_freq = res['frequenze'].most_common(num_per_ambo)
            report.append("  * Top 3 per Convergenza (Estratto): " + ", ".join([f"{n} (p.{s})" for n,s in top_3_conv]))
            report.append(f"  * Top {num_per_ambo} per Frequenza (Ambo): " + ", ".join([f"{n} (f.{s})" for n,s in top_freq]))
        report.append("\n" + "="*55); report.append("5. RISULTATO DEL BACKTEST DUALE")
        if esiti:
            report.append("\n--- Ipotesi 1: ESTRATTO dai 3 più convergenti ---")
            estratto_esito = esiti['esito_estratto']
            if estratto_esito.get('stato') == 'VINTA': report.append(f"  ✅ VINTA! Uscito il {estratto_esito['numero_uscito']} il {estratto_esito['data_uscita'].strftime('%d/%m/%Y')} al {estratto_esito['colpo']}° colpo.")
            elif estratto_esito.get('stato') == 'IN GIOCO': report.append(f"  🟡 IN GIOCO! (verificati {estratto_esito.get('colpi_passati', 0)} colpi su {colpi_gioco})")
            else: report.append(f"  ❌ NEGATIVO (non uscito nei {colpi_gioco} colpi).")
            report.append(f"\n--- Ipotesi 2: AMBO dai {num_per_ambo} più frequenti ---")
            ambo_esito = esiti['esito_ambo']
            if ambo_esito.get('stato') == 'NON_APPLICABILE': report.append("  Non applicabile (meno di 2 numeri frequenti trovati).")
            elif ambo_esito.get('stato') == 'VINTA': report.append(f"  ✅ VINTA! Uscito l'ambo {ambo_esito['ambo_uscito'][0]}-{ambo_esito['ambo_uscito'][1]} il {ambo_esito['data_uscita'].strftime('%d/%m/%Y')} al {ambo_esito['colpo']}° colpo.")
            elif ambo_esito.get('stato') == 'IN GIOCO': report.append(f"  🟡 IN GIOCO! (verificati {ambo_esito.get('colpi_passati', 0)} colpi su {colpi_gioco})")
            else: report.append(f"  ❌ NEGATIVO (non uscito nei {colpi_gioco} colpi).")
        else: report.append("  Impossibile eseguire il backtest (nessun numero convergente).")
        report.append("="*55)
        return "\n".join(report)
        
    def genera_report_radar(self, sorted_results, data_riferimento):
        report = []
        report.append(f"--- RADAR DI MAX: CONDIZIONI MIGLIORI AL {data_riferimento} ---")
        report.append("=" * 60)
        
        if not sorted_results:
            report.append("\nNessuna condizione significativa trovata con i criteri attuali.")
            return "\n".join(report)

        report.append(f"{'Pos.':<5}{'Ruota':<12}{'Decina':<10}{'Punteggio'}")
        report.append(f"{'----':<5}{'----------':<12}{'--------':<10}{'---------'}")
        
        for i, res in enumerate(sorted_results[:10]): # Mostra la Top 10
            pos = f"{i+1}."
            ruota = res['ruota']
            decina = res['decina']
            punteggio = res['punteggio']
            report.append(f"{pos:<5}{ruota:<12}{decina:<10}{punteggio}")
            
        report.append("\n" + "=" * 60)
        report.append("* Usare la scheda 'Analisi Decine' per investigare queste condizioni.")
        return "\n".join(report)
# ==============================================================================
# SEZIONE 3: IL BLOCCO DI AVVIO DELL'APPLICAZIONE
# ==============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = SegmentiApp(root)
    root.mainloop()