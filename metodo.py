import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import requests
from datetime import datetime, date
from collections import Counter
from itertools import combinations
import threading
import queue
from dateutil.relativedelta import relativedelta

try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Libreria Mancante", "Le librerie 'tkcalendar' e 'python-dateutil' non sono installate.\n\nPer favore, installale eseguendo questi comandi nel tuo terminale:\n\npip install tkcalendar\npip install python-dateutil")
    exit()

# --- CLASSE GESTIONE ARCHIVIO ---
class ArchivioLotto:
    def __init__(self, output_queue):
        self.output_queue = output_queue; self.estrazioni_per_ruota = {}; self.dati_per_analisi = {}; self.date_ordinate = []; self.date_to_index = {}
        self.GITHUB_USER = "illottodimax"; self.GITHUB_REPO = "Archivio"; self.GITHUB_BRANCH = "main"
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
        self.URL_RUOTE = {k: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{v.upper()}.txt' for k, v in self.RUOTE_DISPONIBILI.items()}
    def _log(self, message): self.output_queue.put(message)
    def _reset_archivio(self): self.estrazioni_per_ruota.clear(); self.dati_per_analisi.clear(); self.date_ordinate = []; self.date_to_index = {}
    def inizializza_da_github(self):
        self._reset_archivio(); self._log("Inizio caricamento archivio da GitHub...")
        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            self._log(f"Caricamento {ruota_nome} ({i+1}/{len(self.RUOTE_DISPONIBILI)})...")
            try: response = requests.get(self.URL_RUOTE[ruota_key], timeout=15); response.raise_for_status(); self.estrazioni_per_ruota[ruota_key] = self._parse_estrazioni(response.text.strip().split('\n'))
            except Exception as e: self._log(f"ERRORE: Impossibile caricare {ruota_nome}: {e}"); self.estrazioni_per_ruota[ruota_key] = []
        self._prepara_dati_per_analisi()
    def inizializza_da_locale(self, percorso_cartella):
        self._reset_archivio(); self._log(f"Inizio caricamento archivio da: {percorso_cartella}")
        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            nome_file = f"{ruota_nome.upper()}.txt"; percorso_file = os.path.join(percorso_cartella, nome_file)
            if os.path.exists(percorso_file):
                try:
                    with open(percorso_file, 'r', encoding='utf-8') as f: self.estrazioni_per_ruota[ruota_key] = self._parse_estrazioni(f.readlines())
                except Exception as e: self._log(f" -> ERRORE: Impossibile leggere il file {nome_file}: {e}")
        self._prepara_dati_per_analisi()
    def _parse_estrazioni(self, linee):
        parsed_data = []
        for l in linee:
            parts = l.strip().split()
            if len(parts) == 7:
                try: data = datetime.strptime(parts[0], '%Y/%m/%d').date(); numeri = [int(n) for n in parts[2:]]
                except (ValueError, IndexError): continue
                if len(numeri) == 5: parsed_data.append({'data': data, 'numeri': numeri})
        return parsed_data
    def _prepara_dati_per_analisi(self):
        self._log("Allineamento e indicizzazione dati...")
        tutte_le_date_set = {e['data'] for estrazioni in self.estrazioni_per_ruota.values() if estrazioni for e in estrazioni}
        if not tutte_le_date_set: self._log("ERRORE CRITICO: Nessuna data di estrazione valida trovata."); return
        self.date_ordinate = sorted(list(tutte_le_date_set)); self.date_to_index = {data: i for i, data in enumerate(self.date_ordinate)}
        self.dati_per_analisi = {data: {ruota: None for ruota in self.RUOTE_DISPONIBILI} for data in self.date_ordinate}
        for ruota_key, estrazioni in self.estrazioni_per_ruota.items():
            if estrazioni:
                for estrazione in estrazioni:
                    if estrazione['data'] in self.dati_per_analisi: self.dati_per_analisi[estrazione['data']][ruota_key] = estrazione['numeri']
        self._log("-" * 30); self._log(f"Archivio caricato. Trovate {len(self.date_ordinate)} estrazioni."); self._log(f"Prima estrazione: {self.date_ordinate[0].strftime('%d/%m/%Y')}"); self._log(f"Ultima estrazione: {self.date_ordinate[-1].strftime('%d/%m/%Y')}"); self._log("Operazione di caricamento completata.")
    def get_date_from_monthly_index(self, year, month, index_name):
        if not self.date_ordinate: return None
        estrazioni_del_mese = [d for d in self.date_ordinate if d.year == year and d.month == month]
        if not estrazioni_del_mese: return None
        if index_name == "Ultima del mese": return estrazioni_del_mese[-1]
        elif index_name == "Penultima del mese": return estrazioni_del_mese[-2] if len(estrazioni_del_mese) > 1 else None
        else:
            try: target_index_one_based = int(index_name.split('ª')[0]); target_index_zero_based = target_index_one_based - 1
            except (ValueError, IndexError): return None
            if 0 <= target_index_zero_based < len(estrazioni_del_mese): return estrazioni_del_mese[target_index_zero_based]
            return None

# --- MOTORE DEL METODO FIGURA ---
class MetodoFigura:
    def __init__(self, archivio_lotto: ArchivioLotto):
        if not archivio_lotto.date_ordinate: raise ValueError("L'archivio non è stato caricato o è vuoto.")
        self.archivio = archivio_lotto
    @staticmethod
    def calcola_figura(numero: int) -> int:
        if not isinstance(numero, int) or numero <= 0 or numero > 90: return 0
        if numero == 90: return 9
        n = numero
        while n > 9: n = sum(int(digit) for digit in str(n))
        return n
    def get_nome_figura(self, numero_figura: int) -> str:
        return {1: "Uno", 2: "Due", 3: "Tre", 4: "Quattro", 5: "Cinque", 6: "Sei", 7: "Sette", 8: "Otto", 9: "Nove"}.get(numero_figura, "Sconosciuta")
    def analizza(self, ruota_calcolo: str, estratto_index: int, data_riferimento: datetime.date, ruote_gioco: list, colpi_analisi: int, num_ambate: int, num_abbinamenti: int, start_date_history, end_date_history):
        try: estrazione_riferimento = self.archivio.dati_per_analisi[data_riferimento][ruota_calcolo]; numero_guida = estrazione_riferimento[estratto_index]; figura_num = self.calcola_figura(numero_guida); figura_nome = self.get_nome_figura(figura_num)
        except (KeyError, TypeError, IndexError): return {'success': False, 'message': f"Dati non disponibili."}
        casi_storici = []
        for i, data_storica in enumerate(self.archivio.date_ordinate):
            if data_storica >= data_riferimento: break
            if not (start_date_history <= data_storica <= end_date_history): continue
            try:
                numero_storico = self.archivio.dati_per_analisi[data_storica][ruota_calcolo][estratto_index]
                if self.calcola_figura(numero_storico) == figura_num: casi_storici.append(i)
            except (KeyError, TypeError, IndexError): continue
        if not casi_storici: return {'success': False, 'message': f"Nessun evento storico trovato nel periodo selezionato."}
        frequenze_globali = Counter()
        for i_storico in casi_storici:
            for colpo in range(1, colpi_analisi + 1):
                i_futuro = i_storico + colpo
                if i_futuro < len(self.archivio.date_ordinate):
                    data_futura = self.archivio.date_ordinate[i_futuro]
                    for ruota_g in ruote_gioco:
                        numeri_estratti = self.archivio.dati_per_analisi[data_futura].get(ruota_g)
                        if numeri_estratti: frequenze_globali.update(numeri_estratti)
        ambate_principali = [num for num, freq in frequenze_globali.most_common(num_ambate)]
        if not ambate_principali: return {'success': False, 'message': "Nessuna ambata trovata dall'analisi."}
        previsioni_dettagliate = []
        for ambata in ambate_principali:
            frequenze_abbinamenti = Counter()
            for i_storico in casi_storici:
                for colpo in range(1, colpi_analisi + 1):
                    i_futuro = i_storico + colpo
                    if i_futuro < len(self.archivio.date_ordinate):
                        data_futura = self.archivio.date_ordinate[i_futuro]
                        for ruota_g in ruote_gioco:
                            numeri_estratti = self.archivio.dati_per_analisi[data_futura].get(ruota_g)
                            if numeri_estratti and ambata in numeri_estratti:
                                abbinamenti_trovati = [n for n in numeri_estratti if n != ambata]; frequenze_abbinamenti.update(abbinamenti_trovati)
            migliori_abbinamenti = frequenze_abbinamenti.most_common(num_abbinamenti)
            previsioni_dettagliate.append({'ambata': ambata, 'abbinamenti': migliori_abbinamenti})
        return {'success': True, 'eventi_trovati': len(casi_storici), 'previsioni_dettagliate': previsioni_dettagliate, 'data_riferimento': data_riferimento, 'figura_guida_nome': figura_nome, 'figura_guida_num': figura_num, 'numero_guida': numero_guida, 'ruote_gioco': ruote_gioco, 'colpi_analisi': colpi_analisi}

# --- APPLICAZIONE TKINTER ---
class LottoApp:
    def __init__(self, root):
        self.root = root; self.root.title("Damiano Show - Le Figure Posizionali -"); self.root.geometry("850x800")
        self.running = True; self.msg_queue = queue.Queue(); self.archivio = ArchivioLotto(self.msg_queue)
        self.ultima_previsione = None; self.setup_ui(); self.root.protocol("WM_DELETE_WINDOW", self.on_closing); self.process_queue()
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        controls_frame = ttk.LabelFrame(main_frame, text="Controlli", padding="10"); controls_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="ns")
        load_frame = ttk.LabelFrame(controls_frame, text="1. Carica Archivio"); load_frame.pack(fill='x', padx=5, pady=5)
        load_github_btn = ttk.Button(load_frame, text="Carica da GitHub", command=self.inizializza_archivio_github_threaded); load_github_btn.pack(side=tk.LEFT, expand=True, fill='x', padx=(0,2))
        load_local_btn = ttk.Button(load_frame, text="Carica da Cartella", command=self.inizializza_archivio_locale_threaded); load_local_btn.pack(side=tk.LEFT, expand=True, fill='x', padx=(2,0))
        history_frame = ttk.LabelFrame(controls_frame, text="2. Periodo Storico per Analisi"); history_frame.pack(fill='x', padx=5, pady=10)
        ttk.Label(history_frame, text="Da:").pack(side=tk.LEFT, padx=(5,0)); self.history_start_date = DateEntry(history_frame, width=10, date_pattern='dd/mm/yyyy', locale='it_IT'); self.history_start_date.pack(side=tk.LEFT, padx=(0,10))
        self.history_start_date.set_date(date.today() - relativedelta(years=3))
        ttk.Label(history_frame, text="A:").pack(side=tk.LEFT); self.history_end_date = DateEntry(history_frame, width=10, date_pattern='dd/mm/yyyy', locale='it_IT'); self.history_end_date.pack(side=tk.LEFT, padx=(0,5))
        calc_frame = ttk.LabelFrame(controls_frame, text="3. Calcolo Singolo / Backtest"); calc_frame.pack(fill='x', padx=5, pady=5)
        periodo_frame = ttk.Frame(calc_frame); periodo_frame.pack(pady=2, fill='x'); current_year = datetime.now().year; self.year_var = tk.StringVar(value=str(current_year))
        ttk.Label(periodo_frame, text="Anno:").pack(side=tk.LEFT, padx=5); ttk.Spinbox(periodo_frame, from_=2000, to=current_year + 1, textvariable=self.year_var, width=6).pack(side=tk.LEFT)
        self.mesi = {"Gennaio":1, "Febbraio":2, "Marzo":3, "Aprile":4, "Maggio":5, "Giugno":6, "Luglio":7, "Agosto":8, "Settembre":9, "Ottobre":10, "Novembre":11, "Dicembre":12}; self.mese_var = tk.StringVar(value=list(self.mesi.keys())[datetime.now().month - 1])
        ttk.Label(periodo_frame, text="Mese:").pack(side=tk.LEFT, padx=(10,0)); ttk.Combobox(periodo_frame, textvariable=self.mese_var, values=list(self.mesi.keys()), state='readonly', width=10).pack(side=tk.LEFT, expand=True, fill='x')
        ttk.Label(calc_frame, text="Indice Estrazione Mensile:").pack(padx=5, anchor='w', pady=(5,0)); self.index_var = tk.StringVar(value="Ultima del mese"); indici_numerici = [f"{i}ª del mese" for i in range(1, 19)]; indici_speciali = ["Penultima del mese", "Ultima del mese"]; indici_mensili = indici_numerici + indici_speciali
        ttk.Combobox(calc_frame, textvariable=self.index_var, values=indici_mensili, state='readonly').pack(padx=5, pady=2, fill='x')
        ttk.Label(calc_frame, text="Ruota di Calcolo:").pack(padx=5, anchor='w', pady=(5,0)); self.ruota_calcolo_var = tk.StringVar(value='RO'); ttk.Combobox(calc_frame, textvariable=self.ruota_calcolo_var, values=list(self.archivio.RUOTE_DISPONIBILI.keys()), state='readonly').pack(padx=5, pady=2, fill='x')
        ttk.Label(calc_frame, text="Posizione Estratto:").pack(padx=5, anchor='w', pady=(5,0)); self.estratto_var = tk.StringVar(value='1° Estratto'); ttk.Combobox(calc_frame, textvariable=self.estratto_var, values=[f"{i}° Estratto" for i in range(1, 6)], state='readonly').pack(padx=5, pady=2, fill='x')
        game_frame = ttk.LabelFrame(controls_frame, text="4. Impostazioni di Gioco"); game_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(game_frame, text="Numero Ambate:").pack(padx=5, anchor='w'); self.ambate_var = tk.StringVar(value="2"); ttk.Spinbox(game_frame, from_=1, to=5, textvariable=self.ambate_var).pack(padx=5, pady=2, fill='x')
        ttk.Label(game_frame, text="Numero Abbinamenti:").pack(padx=5, anchor='w', pady=(5,0)); self.abbinamenti_var = tk.StringVar(value="5"); ttk.Spinbox(game_frame, from_=1, to=5, textvariable=self.abbinamenti_var).pack(padx=5, pady=2, fill='x')
        ttk.Label(game_frame, text="Ruote di Gioco:").pack(padx=5, anchor='w', pady=(5,0)); self.ruote_gioco_vars = {}; ruote_frame = ttk.Frame(game_frame); ruote_frame.pack(fill='x', padx=5)
        for i, (code, name) in enumerate(self.archivio.RUOTE_DISPONIBILI.items()):
            var = tk.BooleanVar(value=(code in ['RO', 'NZ'])); cb = ttk.Checkbutton(ruote_frame, text=code, variable=var); cb.grid(row=i//4, column=i%4, sticky='w'); self.ruote_gioco_vars[code] = var
        ttk.Label(game_frame, text="Colpi di Gioco:").pack(padx=5, anchor='w', pady=(5,0)); self.colpi_var = tk.StringVar(value="9"); ttk.Spinbox(game_frame, from_=1, to=20, textvariable=self.colpi_var).pack(padx=5, pady=2, fill='x')
        action_frame = ttk.Frame(controls_frame); action_frame.pack(fill='x', padx=5, pady=15)
        analyze_button = ttk.Button(action_frame, text="Esegui Calcolo Singolo", command=self.esegui_analisi_metodo); analyze_button.pack(fill='x', pady=2)
        self.verify_button = ttk.Button(action_frame, text="Verifica Ultimo Calcolo", command=self.verifica_esiti_threaded, state=tk.DISABLED); self.verify_button.pack(fill='x', pady=2)
        ttk.Separator(action_frame, orient='horizontal').pack(fill='x', pady=10)
        backtest_button = ttk.Button(action_frame, text="🚀 Avvia Backtest Automatico", command=self.avvia_backtest); backtest_button.pack(fill='x', pady=2)
        log_frame = ttk.LabelFrame(main_frame, text="Log e Risultati", padding="10"); log_frame.grid(row=0, column=1, sticky="nsew")
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=70, height=25); self.log_text.pack(fill=tk.BOTH, expand=True); self.log_text.configure(state='disabled')
        main_frame.grid_columnconfigure(1, weight=1); main_frame.grid_rowconfigure(0, weight=1)
    def _get_common_params(self):
        if not self.archivio.date_ordinate: messagebox.showerror("Errore", "Archivio non caricato."); return None
        params = {}
        params['history_start'] = self.history_start_date.get_date(); params['history_end'] = self.history_end_date.get_date()
        params['ruota_calcolo'] = self.ruota_calcolo_var.get(); params['estratto_index'] = int(self.estratto_var.get().split('°')[0]) - 1
        params['ruote_gioco'] = [code for code, var in self.ruote_gioco_vars.items() if var.get()]
        if not params['ruote_gioco']: messagebox.showwarning("Attenzione", "Selezionare almeno una ruota di gioco."); return None
        try: params['colpi'] = int(self.colpi_var.get()); params['num_ambate'] = int(self.ambate_var.get()); params['num_abbinamenti'] = int(self.abbinamenti_var.get())
        except ValueError: messagebox.showwarning("Attenzione", "Parametri di gioco non validi."); return None
        return params
    def log(self, message): self.log_text.configure(state='normal'); self.log_text.insert(tk.END, message + "\n"); self.log_text.configure(state='disabled'); self.log_text.see(tk.END)
    def pulisci_log(self): self.log_text.configure(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.configure(state='disabled')
    def process_queue(self):
        try:
            while True: self.log(self.msg_queue.get_nowait())
        except queue.Empty:
            if self.running: self.root.after(100, self.process_queue)
    def on_closing(self): self.running = False; self.root.destroy()
    def _threaded_task(self, task, *args, **kwargs): thread = threading.Thread(target=task, args=args, kwargs=kwargs, daemon=True); thread.start()
    def inizializza_archivio_github_threaded(self): self.pulisci_log(); self._threaded_task(self.archivio.inizializza_da_github)
    def inizializza_archivio_locale_threaded(self):
        percorso = filedialog.askdirectory(title="Seleziona la cartella con i file archivio")
        if percorso: self.pulisci_log(); self._threaded_task(self.archivio.inizializza_da_locale, percorso)
    def esegui_analisi_metodo(self):
        self.pulisci_log(); params = self._get_common_params();
        if not params: return
        try:
            year = int(self.year_var.get()); mese_nome = self.mese_var.get(); index_nome = self.index_var.get()
            mese_num = self.mesi[mese_nome]; data_riferimento = self.archivio.get_date_from_monthly_index(year, mese_num, index_nome)
            if data_riferimento is None: messagebox.showerror("Errore", f"Estrazione '{index_nome}' non trovata per {mese_nome} {year}."); return
        except Exception as e: messagebox.showerror("Errore", f"Impossibile determinare la data di calcolo: {e}"); return
        self.verify_button.config(state=tk.DISABLED); self.ultima_previsione = None
        metodo = MetodoFigura(self.archivio)
        self._threaded_task(self._run_and_display_analysis, metodo=metodo, data_riferimento=data_riferimento, params=params)
    def _run_and_display_analysis(self, metodo, data_riferimento, params, silent=False):
        risultato = metodo.analizza(params['ruota_calcolo'], params['estratto_index'], data_riferimento, params['ruote_gioco'], params['colpi'], params['num_ambate'], params['num_abbinamenti'], params['history_start'], params['history_end'])
        if silent: return risultato
        if risultato['success']:
            self.msg_queue.put(f"Analisi per {risultato['data_riferimento'].strftime('%d/%m/%Y')} completata su {risultato['eventi_trovati']} eventi storici.")
            self.msg_queue.put(f"Condizione: Figura {risultato['figura_guida_nome']} ({risultato['figura_guida_num']}) (da {risultato['numero_guida']})")
            self.msg_queue.put("-" * 30); self.msg_queue.put(f"PREVISIONI PRONTE:")
            for i, previsione in enumerate(risultato['previsioni_dettagliate']):
                ambata = previsione['ambata']; abbinamenti_str = [f"{num}({freq})" for num, freq in previsione['abbinamenti']]
                self.msg_queue.put(f"Previsione {i+1}: Ambata {ambata} con Abbinamenti: {', '.join(abbinamenti_str) if abbinamenti_str else 'Nessuno'}")
            self.msg_queue.put("-" * 30); nomi_ruote = [self.archivio.RUOTE_DISPONIBILI.get(r,r) for r in risultato['ruote_gioco']]; self.msg_queue.put(f"Giocare su: {', '.join(nomi_ruote)}")
            self.msg_queue.put(f"Per un massimo di {risultato['colpi_analisi']} colpi.")
            self.ultima_previsione = risultato; self.verify_button.config(state=tk.NORMAL); self.msg_queue.put("-> Pulsante 'Verifica Ultimo Calcolo' abilitato.")
        else: self.msg_queue.put(f"ERRORE ANALISI per {data_riferimento.strftime('%d/%m/%Y')}: {risultato['message']}")
    def verifica_esiti_threaded(self):
        if not self.ultima_previsione: messagebox.showinfo("Info", "Nessuna previsione da verificare."); return
        self.verify_button.config(state=tk.DISABLED); self._threaded_task(self._run_verification, previsione=self.ultima_previsione, silent=False)
    def _run_verification(self, previsione, silent=False):
        # <<< CORREZIONE ERRORE CHIAVE: Uso 'data_riferimento' invece di 'data_calcolo'
        ruote_da_giocare = previsione['ruote_gioco']; colpi_max = previsione['colpi_analisi']; data_inizio = previsione['data_riferimento']
        if not silent: self.msg_queue.put(f"\n--- VERIFICA ESITI PREVISIONE ---"); self.msg_queue.put(f"Previsione del: {data_inizio.strftime('%d/%m/%Y')}")
        indice_inizio = self.archivio.date_to_index.get(data_inizio, -1);
        if indice_inizio == -1: return {'success': False, 'message': "Data inizio non trovata"}
        vincita_generale_trovata = False
        for i, previsione_dett in enumerate(previsione['previsioni_dettagliate']):
            ambata = previsione_dett['ambata']; abbinamenti = [num for num, freq in previsione_dett['abbinamenti']]; numeri_in_gioco = set([ambata] + abbinamenti)
            for colpo in range(1, colpi_max + 1):
                indice_corrente = indice_inizio + colpo
                if indice_corrente >= len(self.archivio.date_ordinate): break
                data_corrente = self.archivio.date_ordinate[indice_corrente]
                for ruota in ruote_da_giocare:
                    numeri_estratti = self.archivio.dati_per_analisi[data_corrente].get(ruota)
                    if numeri_estratti:
                        numeri_vincenti = numeri_in_gioco.intersection(set(numeri_estratti))
                        if ambata in numeri_vincenti and len(numeri_vincenti) >= 2:
                            vincita_generale_trovata = True
                            if silent: return {'success': True, 'colpo': colpo}
                            esito = "AMBO" if len(numeri_vincenti) == 2 else "TERNO" if len(numeri_vincenti) == 3 else "QUATERNA" if len(numeri_vincenti) == 4 else "CINQUINA"
                            self.msg_queue.put("-" * 30); self.msg_queue.put(f"🎉 VINCITA (Previsione {i+1})! 🎉"); self.msg_queue.put(f"Esito: {esito} con {sorted(list(numeri_vincenti))} su {self.archivio.RUOTE_DISPONIBILI[ruota]} al colpo n°{colpo} ({data_corrente.strftime('%d/%m/%Y')})")
                            self.verify_button.config(state=tk.NORMAL); return
        if silent: return {'success': False}
        if not vincita_generale_trovata: self.msg_queue.put(f"--- NESSUNA VINCITA TROVATA entro {colpi_max} colpi. ---")
        self.verify_button.config(state=tk.NORMAL)
    def avvia_backtest(self):
        self.pulisci_log(); params = self._get_common_params()
        if not params: return
        popup = tk.Toplevel(self.root); popup.title("Imposta Periodo Backtest"); ttk.Label(popup, text="Seleziona il periodo in cui cercare gli eventi di calcolo.").pack(padx=10, pady=10)
        start_frame = ttk.Frame(popup); start_frame.pack(padx=10, pady=5); ttk.Label(start_frame, text="Da:").pack(side=tk.LEFT); bt_start_date = DateEntry(start_frame, width=12, date_pattern='dd/mm/yyyy', locale='it_IT'); bt_start_date.pack(side=tk.LEFT)
        bt_start_date.set_date(date.today() - relativedelta(years=1))
        end_frame = ttk.Frame(popup); end_frame.pack(padx=10, pady=5); ttk.Label(end_frame, text="A:").pack(side=tk.LEFT); bt_end_date = DateEntry(end_frame, width=12, date_pattern='dd/mm/yyyy', locale='it_IT'); bt_end_date.pack(side=tk.LEFT)
        def on_ok():
            params['backtest_start'] = bt_start_date.get_date(); params['backtest_end'] = bt_end_date.get_date(); params['index_nome'] = self.index_var.get()
            popup.destroy(); self._threaded_task(self._run_backtest, params=params)
        ttk.Button(popup, text="Avvia", command=on_ok).pack(pady=10)
    def _run_backtest(self, params):
        self.msg_queue.put("--- INIZIO BACKTEST AUTOMATICO ---")
        metodo = MetodoFigura(self.archivio); current_date = params['backtest_start']; stats = {'previsioni': 0, 'vincite': 0, 'colpi_vincita': []}
        while current_date <= params['backtest_end']:
            data_riferimento = self.archivio.get_date_from_monthly_index(current_date.year, current_date.month, params['index_nome'])
            if data_riferimento and params['backtest_start'] <= data_riferimento <= params['backtest_end']:
                self.msg_queue.put(f"Test per {params['index_nome']} di {current_date.strftime('%B %Y')} -> Estrazione del {data_riferimento.strftime('%d/%m/%Y')}")
                risultato_analisi = self._run_and_display_analysis(metodo, data_riferimento, params, silent=True)
                if risultato_analisi and risultato_analisi['success']:
                    stats['previsioni'] += 1; risultato_verifica = self._run_verification(risultato_analisi, silent=True)
                    if risultato_verifica['success']: stats['vincite'] += 1; stats['colpi_vincita'].append(risultato_verifica['colpo'])
            current_date += relativedelta(months=1)
        self.msg_queue.put("\n--- REPORT FINALE BACKTEST ---"); self.msg_queue.put(f"Periodo Testato: da {params['backtest_start'].strftime('%d/%m/%Y')} a {params['backtest_end'].strftime('%d/%m/%Y')}")
        self.msg_queue.put(f"Previsioni Generate: {stats['previsioni']}"); self.msg_queue.put(f"Previsioni Vincenti: {stats['vincite']}")
        if stats['previsioni'] > 0: percentuale_vincita = (stats['vincite'] / stats['previsioni']) * 100; self.msg_queue.put(f"Percentuale di Successo: {percentuale_vincita:.2f}%")
        if stats['vincite'] > 0: colpo_medio = sum(stats['colpi_vincita']) / len(stats['colpi_vincita']); self.msg_queue.put(f"Colpo Medio di Vincita: {colpo_medio:.2f}")
        self.msg_queue.put("--------------------------------")

if __name__ == '__main__':
    root = tk.Tk(); app = LottoApp(root); root.mainloop()