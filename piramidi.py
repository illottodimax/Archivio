import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import requests
from datetime import datetime, date, timedelta
from collections import Counter
import threading
import queue
import traceback
import locale

# Imposta la lingua italiana per tkcalendar
try:
    locale.setlocale(locale.LC_TIME, 'it_IT.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Italian_Italy.1252')
    except locale.Error:
        print("Attenzione: Locale 'it_IT' non trovato. Il calendario potrebbe apparire in inglese.")

try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Libreria Mancante", "La libreria 'tkcalendar' non è installata.\n\nPer favore, installala eseguendo questo comando nel tuo terminale:\n\npip install tkcalendar")
    exit()

# --- COSTANTI E MAPPE RUOTE ---
RUOTE_NOMI_COMPLETI = [ "Bari", "Cagliari", "Firenze", "Genova", "Milano", "Napoli", "Palermo", "Roma", "Torino", "Venezia", "Nazionale" ]
RUOTE_PRINCIPALI = [r for r in RUOTE_NOMI_COMPLETI if r != "Nazionale"]
RUOTE_DIAMETRALI = { "Bari": "Napoli", "Cagliari": "Palermo", "Firenze": "Roma", "Genova": "Torino", "Milano": "Venezia", "Napoli": "Bari", "Palermo": "Cagliari", "Roma": "Firenze", "Torino": "Genova", "Venezia": "Milano", "Nazionale": "Nazionale" }
RUOTE_GEMELLE = { "Bari": "Venezia", "Cagliari": "Genova", "Firenze": "Milano", "Napoli": "Palermo", "Roma": "Torino", "Genova": "Cagliari", "Milano": "Firenze", "Palermo": "Napoli", "Torino": "Roma", "Venezia": "Bari", "Nazionale": "Nazionale" }

# --- FUNZIONI LOGICHE ---
def regola_fuori_90(numero):
    if numero is None: return None
    if numero == 0: return 90
    while numero <= 0: numero += 90
    while numero > 90: numero -= 90
    return numero
def calcola_vertibile(numero):
    """
    Calcola il numero vertibile secondo le regole complete della lottologia.
    Questa versione corregge la gestione dei "numeretti" (1-9) e mantiene
    la logica per gemelli e altri casi.
    """
    if not (1 <= numero <= 90):
        return None

    # --- CORREZIONE: Gestione prioritaria dei "numeretti" da 1 a 9 ---
    # Questa è la regola che risolve il problema segnalato (es. 9 -> 90).
    if 1 <= numero <= 9:
        return numero * 10

    # Da qui in poi, la logica gestisce i numeri a due cifre (10-90)
    s_num = str(numero)
    decina, unita = s_num[0], s_num[1]

    # Caso 1: Numeri Gemelli (es. 11 -> 19, 22 -> 29)
    if decina == unita:
        return 99 if numero == 99 else int(decina + "9")

    # Caso 2: Vertibili dei Gemelli (es. 19 -> 11, 29 -> 22)
    if unita == '9' and decina != '9':
        return int(decina + decina)
    
    # Caso 3: Tutti gli altri numeri (inversione delle cifre)
    # es. 27 -> 72, e anche 90 -> 9 (perché int('09') fa 9)
    return int(unita + decina)
def piramida_estrazione(numeri_partenza):
    if len(numeri_partenza) != 5: return None
    livello_corrente = list(numeri_partenza)
    while len(livello_corrente) > 1:
        livello_successivo = [regola_fuori_90(livello_corrente[i] + livello_corrente[i+1]) for i in range(len(livello_corrente) - 1)]
        livello_corrente = livello_successivo
    return livello_corrente[0] if livello_corrente else None
def trova_abbinamenti_frequenti(ambata_target, ruote_da_controllare, storico_dati, data_limite, num_abbinamenti):
    if num_abbinamenti == 0 or ambata_target is None: return []
    conteggio = Counter()
    for data_storico, estrazione in storico_dati.items():
        if data_storico < data_limite:
            numeri_vincenti_estrazione = set()
            ambata_trovata_in_estrazione = False
            for ruota in ruote_da_controllare:
                numeri_ruota = estrazione.get(ruota)
                if numeri_ruota and ambata_target in numeri_ruota:
                    ambata_trovata_in_estrazione = True
                    for n in numeri_ruota:
                        if n != ambata_target: numeri_vincenti_estrazione.add(n)
            if ambata_trovata_in_estrazione:
                for numero_compagno in numeri_vincenti_estrazione: conteggio[numero_compagno] += 1
    return [num for num, count in conteggio.most_common(num_abbinamenti)]
def calcola_ambetto_vicini(numero):
    precedente = 90 if numero == 1 else numero - 1
    successivo = 1 if numero == 90 else numero + 1
    return {precedente, successivo}

class ArchivioLotto:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.estrazioni_per_ruota = {}
        self.dati_per_analisi = {}
        self.date_ordinate = []
        self.date_to_index = {}
        self.GITHUB_USER = "illottodimax"
        self.GITHUB_REPO = "Archivio"
        self.GITHUB_BRANCH = "main"
        self.RUOTE_NOMI_COMPLETI = RUOTE_NOMI_COMPLETI
        self.URL_RUOTE = {nome: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{nome.upper()}.txt' for nome in self.RUOTE_NOMI_COMPLETI}
        self.data_source = 'GitHub'
        self.local_path = None
        self.is_initialized = False

    def _log(self, message): self.output_queue.put(message)
    def inizializza(self, force_reload=False):
        if self.is_initialized and not force_reload: self._log("Archivio già inizializzato."); return True
        self._log("Inizio inizializzazione archivio...")
        if self.data_source == 'Locale' and (not self.local_path or not os.path.isdir(self.local_path)): raise FileNotFoundError("Percorso locale non valido o non impostato.")
        self.estrazioni_per_ruota.clear()
        for i, ruota_nome in enumerate(self.RUOTE_NOMI_COMPLETI):
            self._log(f"Caricando {ruota_nome} ({i+1}/{len(self.RUOTE_NOMI_COMPLETI)})...")
            try:
                if self.data_source == 'GitHub':
                    response = requests.get(self.URL_RUOTE[ruota_nome], timeout=15); response.raise_for_status(); linee = response.text.strip().split('\n')
                else:
                    with open(os.path.join(self.local_path, f"{ruota_nome.upper()}.txt"), 'r', encoding='utf-8') as f: linee = f.readlines()
                self.estrazioni_per_ruota[ruota_nome] = self._parse_estrazioni(linee)
            except Exception as e: raise RuntimeError(f"Impossibile caricare i dati per {ruota_nome}: {e}")
        self._prepara_dati_per_analisi()
        self.is_initialized = True
        self.output_queue.put("ARCHIVIO_PRONTO")
        return True
    def _parse_estrazioni(self, linee):
        parsed_data = []
        for l in linee:
            parts = l.strip().split('\t')
            if len(parts) >= 7:
                try:
                    data_obj = datetime.strptime(parts[0], '%Y/%m/%d').date()
                    numeri = [int(n) for n in parts[2:7] if n.isdigit() and 1 <= int(n) <= 90]
                    if len(numeri) == 5:
                        parsed_data.append({'data': data_obj, 'numeri': numeri})
                except (ValueError, IndexError):
                    pass
        return parsed_data
    def _prepara_dati_per_analisi(self):
        self._log("Preparo e allineo i dati per l'analisi...")
        tutte_le_date = {e['data'] for estrazioni in self.estrazioni_per_ruota.values() for e in estrazioni}
        self.date_ordinate = sorted(list(tutte_le_date))
        self.date_to_index = {data: i for i, data in enumerate(self.date_ordinate)}
        self.dati_per_analisi = {data: {ruota: None for ruota in self.RUOTE_NOMI_COMPLETI} for data in self.date_ordinate}
        for ruota, estrazioni in self.estrazioni_per_ruota.items():
            for e in estrazioni:
                if e['data'] in self.dati_per_analisi: self.dati_per_analisi[e['data']][ruota] = e['numeri']
        self._log("Dati allineati.")

# --- CLASSE DELL'APPLICAZIONE GRAFICA ---
class PiramidatoreAvanzatoApp:
    def __init__(self, master):
        self.master = master
        master.title("Piramidatore Strategico Definitivo - by Max Lotto -")
        master.geometry("950x950")
        style = ttk.Style(master)
        style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
        self.output_queue = queue.Queue()
        self.archivio = ArchivioLotto(self.output_queue)
        self.source_var = tk.StringVar(value="GitHub")
        self.local_path_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop"))
        self.ruota_riferimento_var = tk.StringVar(value=RUOTE_NOMI_COMPLETI[0])
        self.tipo_ruota_gioco_var = tk.StringVar(value="stessa")
        self.ruota_scelta_utente_var = tk.StringVar(value=RUOTE_NOMI_COMPLETI[1])
        self.colpi_gioco_var = tk.IntVar(value=9)
        self.num_ambate_var = tk.IntVar(value=1)
        self.num_abbinamenti_var = tk.IntVar(value=3)
        self.gioca_ambetto_var = tk.BooleanVar(value=True)
        self.filtro_attivo_var = tk.StringVar(value="nessuno")
        self.numero_spia_var = tk.IntVar(value=90)
        self.num_ritardatari_var = tk.IntVar(value=5)
        self.somma_min_var = tk.IntVar(value=15)
        self.somma_max_var = tk.IntVar(value=250)
        self.estratto_min_var = tk.IntVar(value=1)
        self.estratto_max_var = tk.IntVar(value=10)
        self.previsioni_in_corso = []
        self._crea_widgets()
        self._process_queue()

    def _update_ruota_scelta_state(self, *args):
        if hasattr(self, 'ruota_scelta_utente_combo'): self.ruota_scelta_utente_combo.config(state="readonly" if self.tipo_ruota_gioco_var.get() == "scelta_utente" else "disabled")

    def _update_filtro_state(self, *args):
        filtro = self.filtro_attivo_var.get()
        if hasattr(self, 'numero_spia_entry'): self.numero_spia_entry.config(state="normal" if filtro == "spia" else "disabled")
        if hasattr(self, 'num_ritardatari_spin'): self.num_ritardatari_spin.config(state="readonly" if filtro == "ritardatari" else "disabled")
        if hasattr(self, 'somma_min_entry'): self.somma_min_entry.config(state="normal" if filtro == "somma_range" else "disabled")
        if hasattr(self, 'somma_max_entry'): self.somma_max_entry.config(state="normal" if filtro == "somma_range" else "disabled")
        if hasattr(self, 'estratto_min_entry'): self.estratto_min_entry.config(state="normal" if filtro == "estratto_range" else "disabled")
        if hasattr(self, 'estratto_max_entry'): self.estratto_max_entry.config(state="normal" if filtro == "estratto_range" else "disabled")

    def _crea_widgets(self):
        main_frame = ttk.Frame(self.master, padding="10"); main_frame.pack(expand=True, fill="both")
        main_frame.columnconfigure(0, weight=1); main_frame.rowconfigure(4, weight=1)
        source_frame = ttk.LabelFrame(main_frame, text="1. Scegli Fonte Dati", padding="10"); source_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5)); source_frame.columnconfigure(1, weight=1)
        ttk.Radiobutton(source_frame, text="GitHub (Online)", variable=self.source_var, value="GitHub", command=self._update_ui_state).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(source_frame, text="Cartella Locale", variable=self.source_var, value="Locale", command=self._update_ui_state).grid(row=1, column=0, sticky="w")
        self.local_path_entry = ttk.Entry(source_frame, textvariable=self.local_path_var, width=50); self.local_path_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.browse_button = ttk.Button(source_frame, text="Sfoglia...", command=self._select_local_path); self.browse_button.grid(row=1, column=2, sticky="w")
        self.load_button = ttk.Button(source_frame, text="Carica/Aggiorna Archivio", command=self._start_archivio_thread); self.load_button.grid(row=0, column=1, columnspan=2, padx=5, sticky='w')
        date_frame = ttk.LabelFrame(main_frame, text="2. Scegli Periodo di Ricerca", padding="10"); date_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Label(date_frame, text="Da:").pack(side="left", padx=(0, 5)); calendar_style = {'selectbackground': '#0078d4', 'selectforeground': 'white', 'normalbackground': 'white', 'weekendbackground': '#f0f0f0', 'headersbackground': '#e0e0e0'}
        self.start_date_entry = DateEntry(date_frame, width=12, date_pattern='dd/MM/yyyy', locale='it_IT', **calendar_style); self.start_date_entry.pack(side="left"); self.start_date_entry.set_date(date.today() - timedelta(days=365))
        ttk.Label(date_frame, text="A:").pack(side="left", padx=(15, 5))
        self.end_date_entry = DateEntry(date_frame, width=12, date_pattern='dd/MM/yyyy', locale='it_IT', **calendar_style); self.end_date_entry.pack(side="left"); self.end_date_entry.set_date(date.today())
        param_frame = ttk.LabelFrame(main_frame, text="3. Imposta Parametri Metodo", padding="10"); param_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ruota_gioco_frame = ttk.Frame(param_frame); ruota_gioco_frame.grid(row=1, column=1, columnspan=5, sticky='w')
        ttk.Radiobutton(ruota_gioco_frame, text="Stessa Ruota", variable=self.tipo_ruota_gioco_var, value="stessa", command=self._update_ruota_scelta_state).pack(side="left", padx=2)
        ttk.Radiobutton(ruota_gioco_frame, text="Consecutiva", variable=self.tipo_ruota_gioco_var, value="consecutiva", command=self._update_ruota_scelta_state).pack(side="left", padx=2)
        ttk.Radiobutton(ruota_gioco_frame, text="Diametrale", variable=self.tipo_ruota_gioco_var, value="diametrale", command=self._update_ruota_scelta_state).pack(side="left", padx=2)
        ttk.Radiobutton(ruota_gioco_frame, text="Gemella", variable=self.tipo_ruota_gioco_var, value="gemella", command=self._update_ruota_scelta_state).pack(side="left", padx=2)
        ttk.Radiobutton(ruota_gioco_frame, text="Tutte", variable=self.tipo_ruota_gioco_var, value="tutte", command=self._update_ruota_scelta_state).pack(side="left", padx=2)
        ttk.Radiobutton(ruota_gioco_frame, text="A Scelta", variable=self.tipo_ruota_gioco_var, value="scelta_utente", command=self._update_ruota_scelta_state).pack(side="left", padx=2)
        ttk.Label(param_frame, text="Ruota Riferimento:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Combobox(param_frame, textvariable=self.ruota_riferimento_var, values=RUOTE_NOMI_COMPLETI, state="readonly").grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="Gioca su Ruota:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="Colpi Gioco:").grid(row=0, column=3, padx=(20, 5), pady=5, sticky="w"); ttk.Spinbox(param_frame, from_=1, to=99, textvariable=self.colpi_gioco_var, width=5).grid(row=0, column=4, pady=5, sticky="w")
        ttk.Label(param_frame, text="...più ruota scelta:").grid(row=2, column=0, padx=5, pady=(0,5), sticky="e")
        self.ruota_scelta_utente_combo = ttk.Combobox(param_frame, textvariable=self.ruota_scelta_utente_var, values=RUOTE_NOMI_COMPLETI, state="disabled"); self.ruota_scelta_utente_combo.grid(row=2, column=1, columnspan=2, padx=5, pady=(0,5), sticky="w")
        ttk.Label(param_frame, text="N. Ambate (1/2):").grid(row=3, column=0, padx=5, pady=5, sticky="w"); ttk.Spinbox(param_frame, from_=1, to=2, textvariable=self.num_ambate_var, width=5).grid(row=3, column=1, pady=5, sticky="w")
        ttk.Label(param_frame, text="N. Abbinamenti (0-5):").grid(row=3, column=2, padx=(10, 5), pady=5, sticky="w"); ttk.Spinbox(param_frame, from_=0, to=5, textvariable=self.num_abbinamenti_var, width=5).grid(row=3, column=3, pady=5, sticky="w")
        ttk.Checkbutton(param_frame, text="Gioca per Ambetto", variable=self.gioca_ambetto_var).grid(row=3, column=4, padx=(20,5), sticky="w")
        filtro_frame = ttk.LabelFrame(main_frame, text="4. Condizione di Gioco (Opzionale)", padding="10"); filtro_frame.grid(row=3, column=0, sticky="ew", pady=5)
        f1 = ttk.Frame(filtro_frame); f1.pack(fill='x', pady=2); ttk.Radiobutton(f1, text="Nessun Filtro", variable=self.filtro_attivo_var, value="nessuno", command=self._update_filtro_state).pack(side="left", padx=5)
        f2 = ttk.Frame(filtro_frame); f2.pack(fill='x', pady=2); ttk.Radiobutton(f2, text="Presenza Numero Spia:", variable=self.filtro_attivo_var, value="spia", command=self._update_filtro_state).pack(side="left", padx=5); self.numero_spia_entry = ttk.Entry(f2, textvariable=self.numero_spia_var, width=5); self.numero_spia_entry.pack(side="left", padx=2)
        f3 = ttk.Frame(filtro_frame); f3.pack(fill='x', pady=2); ttk.Radiobutton(f3, text="Presenza Ritardatario (Top", variable=self.filtro_attivo_var, value="ritardatari", command=self._update_filtro_state).pack(side="left", padx=5); self.num_ritardatari_spin = ttk.Spinbox(f3, from_=1, to=10, textvariable=self.num_ritardatari_var, width=4); self.num_ritardatari_spin.pack(side="left"); ttk.Label(f3, text=")").pack(side="left", padx=(0,5))
        f4 = ttk.Frame(filtro_frame); f4.pack(fill='x', pady=2); ttk.Radiobutton(f4, text="Somma Estratti tra:", variable=self.filtro_attivo_var, value="somma_range", command=self._update_filtro_state).pack(side="left", padx=5); self.somma_min_entry = ttk.Entry(f4, textvariable=self.somma_min_var, width=5); self.somma_min_entry.pack(side="left", padx=2); ttk.Label(f4, text="e").pack(side="left", padx=2); self.somma_max_entry = ttk.Entry(f4, textvariable=self.somma_max_var, width=5); self.somma_max_entry.pack(side="left", padx=2)
        f5 = ttk.Frame(filtro_frame); f5.pack(fill='x', pady=2); ttk.Radiobutton(f5, text="Presenza Estratto tra:", variable=self.filtro_attivo_var, value="estratto_range", command=self._update_filtro_state).pack(side="left", padx=5); self.estratto_min_entry = ttk.Entry(f5, textvariable=self.estratto_min_var, width=5); self.estratto_min_entry.pack(side="left", padx=2); ttk.Label(f5, text="e").pack(side="left", padx=2); self.estratto_max_entry = ttk.Entry(f5, textvariable=self.estratto_max_var, width=5); self.estratto_max_entry.pack(side="left", padx=2)
        result_frame = ttk.LabelFrame(main_frame, text="Log e Risultati", padding="10"); result_frame.grid(row=4, column=0, sticky="nsew", pady=5); result_frame.rowconfigure(0, weight=1); result_frame.columnconfigure(0, weight=1)
        self.output_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=("Courier New", 9)); self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.tag_config('header', font=("Courier New", 9, "bold")); self.output_text.tag_config('vincita', foreground='#009688', font=("Courier New", 9, "bold")); self.output_text.tag_config('incorso', foreground='#1E88E5'); self.output_text.tag_config('negativo', foreground='#E53935'); self.output_text.tag_config('info', foreground='#00695C'); self.output_text.tag_config('summary', font=("Courier New", 10, "bold"), foreground='#005a9e'); self.output_text.tag_config('playable', font=("Courier New", 10, "bold"), foreground='#D81B60', background='#FFF9C4')
        self._insert_colored_text("Benvenuto! Carica l'archivio, imposta i parametri e avvia l'analisi.", 'header')
        action_frame = ttk.Frame(main_frame); action_frame.grid(row=5, column=0, sticky="ew", pady=(5,0)); action_frame.columnconfigure(0, weight=1); action_frame.columnconfigure(1, weight=1)
        self.analyze_button = ttk.Button(action_frame, text="AVVIA ANALISI STRATEGICA", command=self._avvia_analisi, state="disabled"); self.analyze_button.grid(row=0, column=0, sticky="ew", padx=(0,5), ipady=5)
        self.export_button = ttk.Button(action_frame, text="Esporta Risultati in .txt", command=self._esporta_risultati, state="disabled"); self.export_button.grid(row=0, column=1, sticky="ew", padx=(5,0), ipady=5)
        self._update_ui_state(); self._update_ruota_scelta_state(); self._update_filtro_state()

    def _insert_text_at_start(self, text, tag): self.output_text.config(state="normal"); self.output_text.insert("1.0", text, tag); self.output_text.config(state="disabled")
    def _on_date_selected(self, event): self.master.focus_set()
    def _insert_colored_text(self, text, tag): self.output_text.config(state="normal"); self.output_text.insert(tk.END, text + "\n", tag); self.output_text.config(state="disabled"); self.output_text.see(tk.END)
    def _update_ui_state(self, *args): is_local = self.source_var.get() == "Locale"; self.local_path_entry.config(state="normal" if is_local else "disabled"); self.browse_button.config(state="normal" if is_local else "disabled")
    def _select_local_path(self):
        path = filedialog.askdirectory(title="Seleziona la cartella con gli archivi .txt")
        if path: self.local_path_var.set(path)
    def _process_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg == "TASK_DONE": self.load_button.config(state="normal")
                elif msg == "ARCHIVIO_PRONTO": self.load_button.config(state="normal"); self.analyze_button.config(state="normal"); self.export_button.config(state="normal"); self._insert_colored_text("\nArchivio inizializzato e pronto.", "header")
                else: self._insert_colored_text(msg, None)
        except queue.Empty: pass
        self.master.after(100, self._process_queue)
    def _start_archivio_thread(self): self.load_button.config(state="disabled"); self.analyze_button.config(state="disabled"); self.export_button.config(state="disabled"); self.output_text.config(state="normal"); self.output_text.delete("1.0", tk.END); self.output_text.config(state="disabled"); self.archivio.data_source = self.source_var.get(); self.archivio.local_path = self.local_path_var.get(); thread = threading.Thread(target=self._worker_load_archivio, daemon=True); thread.start()
    def _worker_load_archivio(self):
        try: self.archivio.inizializza(force_reload=True)
        except Exception as e: tb_str = traceback.format_exc(); self._insert_colored_text(f"\nERRORE CRITICO DURANTE IL CARICAMENTO:\n{e}\n{tb_str}", 'negativo'); self.output_queue.put("TASK_DONE")
    def _esporta_risultati(self):
        contenuto = self.output_text.get("1.0", tk.END)
        if not contenuto.strip(): messagebox.showwarning("Esportazione Vuota", "Non ci sono risultati da esportare."); return
        nome_file = f"Analisi_Piramidatore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        percorso_file = filedialog.asksaveasfilename(initialfile=nome_file, defaultextension=".txt", filetypes=[("File di Testo", "*.txt"), ("Tutti i file", "*.*")])
        if percorso_file:
            try:
                with open(percorso_file, 'w', encoding='utf-8') as f: f.write(contenuto)
                messagebox.showinfo("Esportazione Riuscita", f"Risultati salvati con successo in:\n{percorso_file}")
            except Exception as e: messagebox.showerror("Errore di Salvataggio", f"Impossibile salvare il file:\n{e}")
    def _aggiorna_statistiche_cumulative(self, stats, esito_str, colpo):
        stats['vincite_totali'] += 1; stats['colpi_totali_vincita'] += colpo; ambata_gia_contata = False
        if any(s in esito_str for s in ["AMBO", "TERNO", "QUATERNA", "CINQUINA"]): stats['ambata'] += 1; ambata_gia_contata = True
        if "CINQUINA" in esito_str: stats['cinquina'] += 1
        if "QUATERNA" in esito_str: stats['quaterna'] += 1
        if "TERNO" in esito_str: stats['terno'] += 1
        if "AMBO" in esito_str: stats['ambo'] += 1
        if "AMBETTO" in esito_str: stats['ambetto'] += 1
        if not ambata_gia_contata and "Ambata" in esito_str:
            if "vert." in esito_str: stats['ambata_v'] += 1
            else: stats['ambata'] += 1
    def _visualizza_riepilogo_giocata(self, previsione_data):
        if not previsione_data: return
        lines = ["\n", "="*80, "              ⚡ PREVISIONE PIU' URGENTE IN GIOCO ⚡", "="*80]
        ruote_gioco_str = " e ".join(r.upper() for r in previsione_data['ruote_gioco'])
        lines.append(f"  - Data di Calcolo:    {previsione_data['data_calcolo'].strftime('%d/%m/%Y')} su {previsione_data['ruota_riferimento'].upper()}")
        lines.append(f"  - Ruote di Gioco:     {ruote_gioco_str}")
        lines.append(f"  - Stato:              {previsione_data['colpi_rimanenti']} Colpi Rimanenti (su {previsione_data['colpi_totali']})")
        lines.append(  "-"*80)
        ambate = [previsione_data['ambata1']]
        if self.num_ambate_var.get() == 2 and previsione_data['ambata2']: ambate.append(previsione_data['ambata2'])
        lines.append(f"  > AMBATA/E:           { ' - '.join(map(str, sorted(ambate))) }")
        abbinamenti1 = sorted(previsione_data['abbinamenti1']); abbinamenti2 = sorted(previsione_data['abbinamenti2'])
        if abbinamenti1: lines.append(f"  > Giocata 1 (Ambo+):  {'-'.join(map(str, sorted([previsione_data['ambata1']] + abbinamenti1)))}")
        if self.num_ambate_var.get() == 2 and previsione_data['ambata2'] and abbinamenti2: lines.append(f"  > Giocata 2 (Ambo+):  {'-'.join(map(str, sorted([previsione_data['ambata2']] + abbinamenti2)))}")
        if self.gioca_ambetto_var.get(): lines.append(f"  > GIOCO AMBETTO:      Sì")
        lines.append("="*80 + "\n\n"); self._insert_text_at_start("\n".join(lines), 'playable')
    def _visualizza_riepilogo(self, stats):
        totale = stats["previsioni_totali"];
        if totale == 0: self._insert_colored_text("\nNessuna previsione generata con i filtri attuali.", 'info'); return
        vincite, negativi, in_corso = stats['vincite_totali'], stats['negativi'], stats['in_corso']
        perc_vincite = (vincite / totale) * 100 if totale > 0 else 0
        attesa_media = (stats['colpi_totali_vincita'] / vincite) if vincite > 0 else 0
        summary_lines = ["\n\n" + "="*80,"--- RIEPILOGO STATISTICO DEL PERIODO ---".center(80),"="*80,
            f" Previsioni Generate (con filtri): {totale}", "-"*80,
            f" Esiti Positivi: {vincite:<4} ({perc_vincite:6.2f}%)", f" Esiti Negativi: {negativi:<4} ({(negativi/totale)*100 if totale>0 else 0:.2f}%)",
            f" Previsioni in Corso: {in_corso:<4} ({(in_corso/totale)*100 if totale>0 else 0:.2f}%)", f" Attesa Media Vincita: {attesa_media:.2f} colpi", "-"*80, " Dettaglio Cumulativo delle Sorti Vincenti:",]
        dettaglio_sorti = {"Ambata/e": stats['ambata'] + stats['ambata_v'],"Ambetto": stats['ambetto'],"Ambo": stats['ambo'], "Terno": stats['terno'],"Quaterna": stats['quaterna'],"Cinquina": stats['cinquina']}
        for sorte, conteggio in dettaglio_sorti.items():
            if conteggio > 0: summary_lines.append(f"   - {sorte:<15} {conteggio:<4} ({(conteggio / vincite) * 100 if vincite > 0 else 0:6.2f}% delle vincite)")
        summary_lines.append("="*80); self._insert_colored_text("\n".join(summary_lines), 'summary')

    def _calcola_ritardo_su_ruota(self, numero, ruota, indice_partenza):
        if numero is None: return 0
        ritardo = 0
        for i in range(indice_partenza - 1, -1, -1):
            ritardo += 1
            numeri_ruota = self.archivio.dati_per_analisi[self.archivio.date_ordinate[i]].get(ruota)
            if numeri_ruota and numero in numeri_ruota: return ritardo
        return ritardo 
    def _calcola_statistiche_numero(self, numero, ruota, indice_fine):
        if numero is None: return 0, 0
        frequenza, ritardo_max, ritardo_corrente = 0, 0, 0
        for i in range(indice_fine - 1, -1, -1):
            ritardo_corrente += 1
            numeri_ruota = self.archivio.dati_per_analisi[self.archivio.date_ordinate[i]].get(ruota)
            if numeri_ruota and numero in numeri_ruota:
                frequenza += 1
                if ritardo_corrente > ritardo_max: ritardo_max = ritardo_corrente
                ritardo_corrente = 0
        if ritardo_corrente > ritardo_max: ritardo_max = ritardo_corrente
        return frequenza, ritardo_max

    def _trova_ritardatari_su_ruota(self, ruota, indice_fine, quanti):
        ritardi = {n: self._calcola_ritardo_su_ruota(n, ruota, indice_fine) for n in range(1, 91)}
        return sorted(ritardi, key=ritardi.get, reverse=True)[:quanti]

    def _avvia_analisi(self):
        if not self.archivio.is_initialized: messagebox.showerror("Errore", "L'archivio non è stato ancora caricato."); return
        try: start_date, end_date = self.start_date_entry.get_date(), self.end_date_entry.get_date()
        except ValueError: messagebox.showerror("Errore Date", "Date non valide."); return
        if start_date > end_date: messagebox.showerror("Errore Date", "Data inizio > data fine."); return
        
        ruota_riferimento = self.ruota_riferimento_var.get(); tipo_ruota_scelta = self.tipo_ruota_gioco_var.get()
        colpi_di_gioco = self.colpi_gioco_var.get(); num_ambate = self.num_ambate_var.get(); num_abbinamenti = self.num_abbinamenti_var.get(); gioca_ambetto = self.gioca_ambetto_var.get()
        filtro_attivo = self.filtro_attivo_var.get(); num_ritardatari_filtro = self.num_ritardatari_var.get()
        try:
            numero_spia_filtro = self.numero_spia_var.get() if filtro_attivo == "spia" else 0
            somma_min_filtro = self.somma_min_var.get() if filtro_attivo == "somma_range" else 0
            somma_max_filtro = self.somma_max_var.get() if filtro_attivo == "somma_range" else 0
            estratto_min_filtro = self.estratto_min_var.get() if filtro_attivo == "estratto_range" else 0
            estratto_max_filtro = self.estratto_max_var.get() if filtro_attivo == "estratto_range" else 0
        except tk.TclError: messagebox.showerror("Errore Filtro", "I valori per i filtri devono essere numeri interi."); return
        
        self.output_text.config(state="normal"); self.output_text.delete("1.0", tk.END); self.output_text.config(state="disabled")
        statistiche = {"previsioni_totali": 0, "vincite_totali": 0, "negativi": 0, "in_corso": 0,"colpi_totali_vincita": 0, "ambata": 0, "ambata_v": 0, "ambetto": 0, "ambo": 0, "terno": 0, "quaterna": 0, "cinquina": 0}
        self.previsioni_in_corso = []
        
        ruote_da_controllare = [];
        if tipo_ruota_scelta == "stessa": ruote_da_controllare = [ruota_riferimento]
        elif tipo_ruota_scelta == "tutte": ruote_da_controllare = RUOTE_PRINCIPALI
        elif tipo_ruota_scelta == "consecutiva": idx = RUOTE_NOMI_COMPLETI.index(ruota_riferimento); ruota_target = RUOTE_NOMI_COMPLETI[(idx + 1) % len(RUOTE_NOMI_COMPLETI)]; ruote_da_controllare = sorted(list(set([ruota_riferimento, ruota_target])))
        elif tipo_ruota_scelta == "diametrale": ruota_target = RUOTE_DIAMETRALI[ruota_riferimento]; ruote_da_controllare = sorted(list(set([ruota_riferimento, ruota_target])))
        elif tipo_ruota_scelta == "gemella": ruota_target = RUOTE_GEMELLE[ruota_riferimento]; ruote_da_controllare = sorted(list(set([ruota_riferimento, ruota_target])))
        elif tipo_ruota_scelta == "scelta_utente": ruota_target = self.ruota_scelta_utente_var.get(); ruote_da_controllare = sorted(list(set([ruota_riferimento, ruota_target])))
        
        ruote_gioco_str = "TUTTE (solo per Ambo e sorti superiori)" if tipo_ruota_scelta == "tutte" else " e ".join(r.upper() for r in ruote_da_controllare)
        self._insert_colored_text(f"--- Analisi dal {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')} ---", 'header')
        self._insert_colored_text(f"Metodo: Piramide su {ruota_riferimento} | Gioco su {ruote_gioco_str} per {colpi_di_gioco} colpi", 'header')
        if filtro_attivo != 'nessuno': self._insert_colored_text(f"CONDIZIONE DI GIOCO ATTIVA: {filtro_attivo.upper()}", 'header')
        self._insert_colored_text("=" * 80, 'header')

        for i, data_corrente in enumerate(self.archivio.date_ordinate):
            if start_date <= data_corrente <= end_date:
                estrazione = self.archivio.dati_per_analisi.get(data_corrente)
                numeri_input = estrazione.get(ruota_riferimento) if estrazione else None
                if numeri_input:
                    filtro_passato = False
                    if filtro_attivo == "nessuno": filtro_passato = True
                    elif filtro_attivo == "spia" and numero_spia_filtro in numeri_input: filtro_passato = True
                    elif filtro_attivo == "ritardatari":
                        # Calcolo ritardatari ONNESSTO, usando solo dati fino al momento del calcolo (i)
                        ritardatari_top = self._trova_ritardatari_su_ruota(ruota_riferimento, i, num_ritardatari_filtro)
                        if set(numeri_input).intersection(set(ritardatari_top)): filtro_passato = True
                    elif filtro_attivo == "somma_range":
                        somma_attuale = sum(numeri_input)
                        if somma_min_filtro <= somma_attuale <= somma_max_filtro:
                            filtro_passato = True
                    elif filtro_attivo == "estratto_range":
                        if any(estratto_min_filtro <= n <= estratto_max_filtro for n in numeri_input):
                            filtro_passato = True

                    if filtro_passato:
                        statistiche["previsioni_totali"] += 1
                        ambata1 = piramida_estrazione(numeri_input)
                        if ambata1 is None: continue 
                        ambata2 = calcola_vertibile(ambata1)
                        
                        # --- INIZIO PARTE CORRETTA ---
                        # Calcoliamo TUTTE le informazioni ORA e le salviamo, usando solo dati passati.
                        # L'indice per le statistiche è SEMPRE "i", che rappresenta il "presente" del ciclo.
                        indice_per_statistiche = i
                        
                        abbinamenti1 = trova_abbinamenti_frequenti(ambata1, ruote_da_controllare, self.archivio.dati_per_analisi, data_corrente, num_abbinamenti)
                        abbinamenti2 = []
                        if num_ambate == 2 and ambata2 is not None: abbinamenti2 = trova_abbinamenti_frequenti(ambata2, ruote_da_controllare, self.archivio.dati_per_analisi, data_corrente, num_abbinamenti)
                        # --- FINE PARTE CORRETTA ---
                        
                        self._insert_colored_text(f"\nEstrazione del {data_corrente.strftime('%d/%m/%Y')} su {ruota_riferimento} {numeri_input}", 'header')
                        
                        info_lines = []
                        info_lines.append(f"  Analisi Avanzata Ambata {ambata1}:")
                        for r in ruote_da_controllare:
                            rit_att = self._calcola_ritardo_su_ruota(ambata1, r, indice_per_statistiche)
                            freq, rit_max = self._calcola_statistiche_numero(ambata1, r, indice_per_statistiche)
                            info_lines.append(f"    - {r.upper():<10} | Rit.Att: {rit_att:<3} | Freq: {freq:<4} | Rit.Max: {rit_max:<3}")
                        if num_ambate == 2 and ambata2 is not None:
                            info_lines.append(f"  Analisi Avanzata Ambata {ambata2} (Vertibile):")
                            for r in ruote_da_controllare:
                                rit_att = self._calcola_ritardo_su_ruota(ambata2, r, indice_per_statistiche)
                                freq, rit_max = self._calcola_statistiche_numero(ambata2, r, indice_per_statistiche)
                                info_lines.append(f"    - {r.upper():<10} | Rit.Att: {rit_att:<3} | Freq: {freq:<4} | Rit.Max: {rit_max:<3}")
                        
                        self._insert_colored_text("\n".join(info_lines), 'info')
                        
                        if abbinamenti1: self._insert_colored_text(f"  Abbinamenti per {ambata1}: {sorted(abbinamenti1)}", 'info')
                        if abbinamenti2: self._insert_colored_text(f"  Abbinamenti per {ambata2}: {sorted(abbinamenti2)}", 'info')
                        
                        lunghetta1 = {ambata1, *abbinamenti1}; lunghetta2 = set()
                        if num_ambate == 2 and ambata2 is not None: lunghetta2 = {ambata2, *abbinamenti2}
                        
                        esito_trovato, esito_str, colpo_vincita = False, "", 0
                        for colpo in range(1, colpi_di_gioco + 1):
                            indice_futuro = i + colpo
                            if indice_futuro >= len(self.archivio.date_ordinate): break
                            data_vincita = self.archivio.date_ordinate[indice_futuro]; estrazione_futura = self.archivio.dati_per_analisi[data_vincita]
                            for ruota_check in ruote_da_controllare:
                                numeri_estratti = estrazione_futura.get(ruota_check)
                                if numeri_estratti:
                                    numeri_estratti_set = set(numeri_estratti)
                                    if num_abbinamenti > 0:
                                        intersezione_vincente = None
                                        if len(lunghetta1.intersection(numeri_estratti_set)) >= 2: intersezione_vincente = lunghetta1.intersection(numeri_estratti_set)
                                        elif lunghetta2 and len(lunghetta2.intersection(numeri_estratti_set)) >= 2: intersezione_vincente = lunghetta2.intersection(numeri_estratti_set)
                                        if intersezione_vincente:
                                            esito_trovato = True; colpo_vincita = colpo; sorte_map = {2: "AMBO", 3: "TERNO", 4: "QUATERNA", 5: "CINQUINA"}; sorte_vinta = sorte_map.get(len(intersezione_vincente), f"{len(intersezione_vincente)} NUMERI"); numeri_vinti_str = "-".join(map(str, sorted(list(intersezione_vincente)))); esito_str = f"{sorte_vinta} {numeri_vinti_str} al {colpo}° colpo ({data_vincita.strftime('%d/%m')}) su {ruota_check.upper()}"; break
                                    if tipo_ruota_scelta != 'tutte':
                                        if not esito_trovato and gioca_ambetto:
                                            vicini_ambata1 = calcola_ambetto_vicini(ambata1); vicini_ambata2 = calcola_ambetto_vicini(ambata2) if num_ambate == 2 and ambata2 else set()
                                            if ambata1 in numeri_estratti_set and not vicini_ambata2.isdisjoint(numeri_estratti_set): vicino = list(vicini_ambata2.intersection(numeri_estratti_set))[0]; esito_trovato = True; colpo_vincita = colpo; esito_str = f"AMBETTO {ambata1}-{vicino} al {colpo}° colpo ({data_vincita.strftime('%d/%m')}) su {ruota_check.upper()}"; break
                                            if num_ambate == 2 and ambata2 is not None and ambata2 in numeri_estratti_set and not vicini_ambata1.isdisjoint(numeri_estratti_set): vicino = list(vicini_ambata1.intersection(numeri_estratti_set))[0]; esito_trovato = True; colpo_vincita = colpo; esito_str = f"AMBETTO {ambata2}-{vicino} al {colpo}° colpo ({data_vincita.strftime('%d/%m')}) su {ruota_check.upper()}"; break
                                        if not esito_trovato and ambata1 in numeri_estratti_set: esito_trovato = True; colpo_vincita = colpo; esito_str = f"Ambata {ambata1} al {colpo}° colpo ({data_vincita.strftime('%d/%m')}) su {ruota_check.upper()}"; break
                                        if not esito_trovato and num_ambate == 2 and ambata2 is not None and ambata2 in numeri_estratti_set: esito_trovato = True; colpo_vincita = colpo; esito_str = f"Ambata (vert.) {ambata2} al {colpo}° colpo ({data_vincita.strftime('%d/%m')}) su {ruota_check.upper()}"; break
                            if esito_trovato: break
                        
                        is_in_corso = not esito_trovato and (i + colpi_di_gioco >= len(self.archivio.date_ordinate))
                        if esito_trovato: self._insert_colored_text(f"  -> ESITO: {esito_str}", 'vincita'); self._aggiorna_statistiche_cumulative(statistiche, esito_str, colpo_vincita)
                        elif is_in_corso:
                            colpi_verificati = len(self.archivio.date_ordinate) - 1 - i
                            colpi_rimanenti = colpi_di_gioco - colpi_verificati
                            messaggio_corso = f"IN CORSO (ancora per {colpi_rimanenti} colp{'o' if colpi_rimanenti==1 else 'i'})"
                            self._insert_colored_text(f"  -> ESITO: {messaggio_corso}", 'incorso')
                            statistiche["in_corso"] += 1
                            # QUI SALVIAMO LA PREVISIONE "CONGELATA" CON TUTTE LE SUE INFO ORIGINALI
                            self.previsioni_in_corso.append({ 
                                "data_calcolo": data_corrente, 
                                "ruota_riferimento": ruota_riferimento, 
                                "ruote_gioco": ruote_da_controllare, 
                                "ambata1": ambata1, 
                                "ambata2": ambata2, 
                                "abbinamenti1": abbinamenti1, # Abbinamenti originali
                                "abbinamenti2": abbinamenti2, # Abbinamenti originali
                                "colpi_rimanenti": colpi_rimanenti, 
                                "colpi_totali": colpi_di_gioco 
                            })
                        else: self._insert_colored_text(f"  -> ESITO: NEGATIVO in {colpi_di_gioco} colpi", 'negativo'); statistiche["negativi"] += 1
        
        if self.previsioni_in_corso:
            previsione_piu_urgente = min(self.previsioni_in_corso, key=lambda p: p['colpi_rimanenti'])
            # La funzione di visualizzazione ora userà i dati salvati e non ricalcolerà nulla.
            self._visualizza_riepilogo_giocata(previsione_piu_urgente)
        self._visualizza_riepilogo(statistiche)
        self.output_text.see("1.0")

if __name__ == "__main__":
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="clam") 
    except ImportError:
        root = tk.Tk()
    
    app = PiramidatoreAvanzatoApp(root)
    root.mainloop()