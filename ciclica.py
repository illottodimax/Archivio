import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
from collections import Counter
from datetime import datetime, timedelta
import requests
import os
from pathlib import Path
import threading
import queue

# ==============================================================================
# CLASSE "MOTORE" DELL'ANALISI (Backend)
# ==============================================================================
class LottoAnalyzer:
    def __init__(self):
        self.estrazioni = {}
        self.GITHUB_USER = "illottodimax"
        self.GITHUB_REPO = "Archivio"
        self.GITHUB_BRANCH = "main"
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
        self.URL_RUOTE = {key: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{value.upper()}.txt' for key, value in self.RUOTE_DISPONIBILI.items()}
        self.data_source = 'github'
        self.local_path = None

    def _parse_estrazioni(self, linee):
        estrazioni_temp = []
        for linea in linee:
            parti = linea.strip().split('\t')
            if len(parti) >= 7:
                try:
                    data = datetime.strptime(parti[0], '%Y/%m/%d').date()
                    numeri = [int(n) for n in parti[2:7]]
                    estrazioni_temp.append({'data': data, 'numeri': numeri})
                except (ValueError, IndexError): continue
        return sorted(estrazioni_temp, key=lambda x: x['data'])

    def carica_dati_per_ruote(self, lista_ruote, status_callback, force_reload=False):
        for i, ruota in enumerate(lista_ruote):
            if ruota in self.estrazioni and not force_reload: continue
            status_callback(f"Caricando dati per {self.RUOTE_DISPONIBILI[ruota]} ({i+1}/{len(lista_ruote)})...")
            try:
                if self.data_source == 'github':
                    response = requests.get(self.URL_RUOTE[ruota], timeout=15); response.raise_for_status(); linee = response.text.strip().split('\n')
                elif self.data_source == 'local':
                    percorso_file = os.path.join(self.local_path, f"{self.RUOTE_DISPONIBILI[ruota].upper()}.txt")
                    with open(percorso_file, 'r', encoding='utf-8') as f: linee = f.readlines()
                self.estrazioni[ruota] = self._parse_estrazioni(linee)
            except Exception as e:
                self.estrazioni[ruota] = []; raise e

    def _filtra_dati_per_giorno(self, dati_sorgente, giorno_settimana=None):
        if not giorno_settimana or giorno_settimana == "Tutti": return dati_sorgente
        giorni_map = {'Martedì': 1, 'Giovedì': 3, 'Venerdì': 4, 'Sabato': 5}
        weekday_target = giorni_map.get(giorno_settimana)
        return [e for e in dati_sorgente if e['data'].weekday() == weekday_target]

    def _calcola_ritardo_ruota(self, dati_ruota, numero):
        for i, estrazione in enumerate(reversed(dati_ruota)):
            if numero in estrazione['numeri']: return i
        return len(dati_ruota)

    def esegui_analisi_completa(self, **params):
        self.carica_dati_per_ruote(params['lista_ruote'], params['status_callback'])
        params['status_callback']("Analisi di convergenza in corso...")
        
        punteggio_convergenza = Counter()
        dettagli_presenza_per_numero = {i: {} for i in range(1, 91)}
        blocchi_analisi_per_ruota = {}
        ruote_valide_per_analisi = []
        data_ultima_estrazione_usata = None

        for ruota in params['lista_ruote']:
            archivio_filtrato_giorno = self._filtra_dati_per_giorno(self.estrazioni.get(ruota, []), params['giorno_settimana'])
            universo_dati = [e for e in archivio_filtrato_giorno if e['data'] <= params['data_riferimento']]
            
            estrazioni_per_analisi = params['lunghezza_ciclo'] * params['num_cicli_analisi']
            if len(universo_dati) < estrazioni_per_analisi:
                params['status_callback'](f"Dati insuff. per {ruota}."); continue
            
            ruote_valide_per_analisi.append(ruota)
            blocco_analisi = universo_dati[-estrazioni_per_analisi:]
            blocchi_analisi_per_ruota[ruota] = blocco_analisi
            if not data_ultima_estrazione_usata or blocco_analisi[-1]['data'] > data_ultima_estrazione_usata:
                data_ultima_estrazione_usata = blocco_analisi[-1]['data']

            presenza_ciclica_ruota = Counter()
            for i in range(0, len(blocco_analisi), params['lunghezza_ciclo']):
                ciclo = blocco_analisi[i:i+params['lunghezza_ciclo']]
                for numero in set(num for estrazione in ciclo for num in estrazione['numeri']):
                    presenza_ciclica_ruota[numero] += 1
            
            for numero, conteggio in presenza_ciclica_ruota.items():
                punteggio_convergenza[numero] += conteggio
                dettagli_presenza_per_numero[numero][ruota] = conteggio

        if not ruote_valide_per_analisi: return "Nessuna ruota ha dati sufficienti per l'analisi."
        
        migliori_ambate_raw = punteggio_convergenza.most_common(params['num_ambate'])
        protagonisti = [{'numero': num, 'punteggio_convergenza': score} for num, score in migliori_ambate_raw]
        set_protagonisti = {p['numero'] for p in protagonisti}
        
        verifica = self.esegui_verifica(params, set_protagonisti, ruote_valide_per_analisi, blocchi_analisi_per_ruota, data_ultima_estrazione_usata)
        
        for p in protagonisti:
            p['dettagli_ruote'] = dettagli_presenza_per_numero.get(p['numero'], {}); p['ritardi_ruote'] = {r: self._calcola_ritardo_ruota(blocchi_analisi_per_ruota[r], p['numero']) for r in ruote_valide_per_analisi}
        
        abbinamenti = Counter({num: score for ruota in ruote_valide_per_analisi for num, score in Counter(num for e in blocchi_analisi_per_ruota[ruota] for num in e['numeri'] if num not in set_protagonisti).items()}).most_common(params['num_abbinamenti'])
        previsione = {'ambate': protagonisti, 'abbinamenti': abbinamenti}
            
        return {'previsione': previsione, 'verifica': verifica, 'ruote_usate': ruote_valide_per_analisi, 'data_riferimento_usata': params['data_riferimento']}

    def esegui_verifica(self, params, set_protagonisti, ruote_valide, blocchi_analisi, data_ultima_analisi):
        is_ricerca = params['data_riferimento'] < datetime.now().date()
        
        if is_ricerca:
            # Questa parte per la verifica predittiva rimane invariata
            estrazioni_future = {ruota: [e for e in self._filtra_dati_per_giorno(self.estrazioni.get(ruota, []), params['giorno_settimana']) if e['data'] > data_ultima_analisi] for ruota in ruote_valide}
            
            vincita = False
            colpi_effettivamente_verificati = max([len(v) for v in estrazioni_future.values()] or [0])

            for i in range(min(params['colpi_di_gioco'], colpi_effettivamente_verificati)):
                if any(set_protagonisti.intersection(set(estrazioni_future[ruota][i]['numeri'])) for ruota in ruote_valide if i < len(estrazioni_future[ruota])):
                    vincita = True; break
            
            # Aggiungiamo comunque i dati statistici anche per la ricerca, sono utili
            vincite_per_ciclo = 0
            cicli_scoperti = []
            for i in range(params['num_cicli_analisi']):
                vincita_nel_ciclo = False
                for ruota in ruote_valide:
                    # Controlla se il segmento di ciclo per la verifica esiste
                    start_index = i * params['lunghezza_ciclo']
                    end_index = start_index + params['colpi_di_gioco']
                    if end_index <= len(blocchi_analisi[ruota]):
                        segmento_gioco = blocchi_analisi[ruota][start_index : end_index]
                        if any(set_protagonisti.intersection(set(e['numeri'])) for e in segmento_gioco):
                            vincita_nel_ciclo = True
                            break # Trovata vincita per questa ruota, passa al ciclo successivo
                if vincita_nel_ciclo:
                    vincite_per_ciclo += 1
                else:
                    cicli_scoperti.append(i + 1) # Aggiunge il numero del ciclo (partendo da 1)

            return {'tipo': 'predittiva', 'esito_vincita': vincita, 'colpi_verificati': min(params['colpi_di_gioco'], colpi_effettivamente_verificati), 'colpi_richiesti': params['colpi_di_gioco'], 'copertura_globale': vincite_per_ciclo, 'cicli_totali': params['num_cicli_analisi'], 'cicli_scoperti': cicli_scoperti}
        else:
            # Logica per la verifica statistica (data futura)
            vincite_per_ciclo = 0
            cicli_scoperti = []
            for i in range(params['num_cicli_analisi']):
                vincita_nel_ciclo = False
                for ruota in ruote_valide:
                    start_index = i * params['lunghezza_ciclo']
                    end_index = start_index + params['colpi_di_gioco']
                    if end_index <= len(blocchi_analisi[ruota]):
                        segmento_gioco = blocchi_analisi[ruota][start_index : end_index]
                        if any(set_protagonisti.intersection(set(e['numeri'])) for e in segmento_gioco):
                            vincita_nel_ciclo = True
                            break
                if vincita_nel_ciclo:
                    vincite_per_ciclo += 1
                else:
                    cicli_scoperti.append(i + 1)
            
            return {'tipo': 'statistica', 'copertura_globale': vincite_per_ciclo, 'cicli_totali': params['num_cicli_analisi'], 'cicli_scoperti': cicli_scoperti}

    def verifica_esiti_previsione(self, previsione_salvata, status_callback):
        ruote = previsione_salvata['ruote']
        # Recuperiamo entrambi i set di numeri per le due giocate distinte
        ambate_giocate = set(previsione_salvata['ambate'])
        lunghetta_giocata = set(previsione_salvata['lunghetta'])
        colpi_max = previsione_salvata['colpi']
        data_previsione = previsione_salvata['data_previsione']
        giorno_settimana = previsione_salvata['giorno_settimana']

        self.carica_dati_per_ruote(ruote, status_callback, force_reload=True)

        estrazioni_per_data = {}
        for r in ruote:
            archivio_filtrato_giorno = self._filtra_dati_per_giorno(self.estrazioni.get(r, []), giorno_settimana)
            nuove_estrazioni = [e for e in archivio_filtrato_giorno if e['data'] > data_previsione]
            for estrazione in nuove_estrazioni:
                data = estrazione['data']
                if data not in estrazioni_per_data:
                    estrazioni_per_data[data] = []
                
                estrazione_con_ruota = estrazione.copy()
                estrazione_con_ruota['ruota'] = r
                estrazioni_per_data[data].append(estrazione_con_ruota)

        if not estrazioni_per_data:
            return "Nessuna nuova estrazione da verificare dalla data della previsione."
        
        date_ordinate = sorted(estrazioni_per_data.keys())
        colpi_verificati = len(date_ordinate)

        # miglior_esito terrà traccia della vincita di valore più alto
        miglior_esito = {'punti': 0, 'colpo': 0, 'ruota': '', 'numeri': []}
        
        for i, data_colpo in enumerate(date_ordinate[:colpi_max]):
            colpo_attuale = i + 1
            estrazioni_del_giorno = estrazioni_per_data[data_colpo]
            
            for estrazione in estrazioni_del_giorno:
                numeri_estratti = set(estrazione['numeri'])
                
                # --- LOGICA DI VERIFICA CON PRIORITÀ ---
                punti_vincita_corrente = 0
                numeri_vincenti_correnti = set()

                # 1. Controlla la giocata di valore maggiore: Ambo e superiori sulla lunghetta
                intersezione_lunghetta = lunghetta_giocata.intersection(numeri_estratti)
                if len(intersezione_lunghetta) >= 2:
                    punti_vincita_corrente = len(intersezione_lunghetta)
                    numeri_vincenti_correnti = intersezione_lunghetta
                
                # 2. Se non ha vinto la lunghetta, controlla la giocata di valore minore: Ambata
                else:
                    intersezione_ambata = ambate_giocate.intersection(numeri_estratti)
                    if len(intersezione_ambata) > 0:
                        punti_vincita_corrente = 1
                        numeri_vincenti_correnti = intersezione_ambata
                
                # Aggiorna il miglior esito generale SOLO se la vincita di questa estrazione è la migliore finora
                if punti_vincita_corrente > miglior_esito['punti']:
                    miglior_esito['punti'] = punti_vincita_corrente
                    miglior_esito['colpo'] = colpo_attuale
                    miglior_esito['ruota'] = estrazione['ruota']
                    miglior_esito['numeri'] = sorted(list(numeri_vincenti_correnti))
        
        # Preparazione del messaggio di output
        if miglior_esito['punti'] > 0:
            sorti = {1: "AMBATA", 2: "AMBO", 3: "TERNO", 4: "QUATERNA", 5: "CINQUINA"}
            sorte_vinta = sorti.get(miglior_esito['punti'], f"{miglior_esito['punti']} punti")
            giorno_str = f" di {giorno_settimana}" if giorno_settimana != "Tutti" else ""
            
            return (f"VINCITA!\n\n"
                    f"Sorte Massima Rilevata: {sorte_vinta.upper()}\n"
                    f"Numeri: {miglior_esito['numeri']}\n"
                    f"Ruota: {self.RUOTE_DISPONIBILI[miglior_esito['ruota']]}\n"
                    f"Colpo: {miglior_esito['colpo']}° utile{giorno_str}.")
        
        if colpi_verificati < colpi_max:
            return f"PREVISIONE IN CORSO. Nessun esito nei primi {colpi_verificati} su {colpi_max} colpi utili."
        else:
            return f"ESITO NEGATIVO. Nessuna vincita di Ambata o superiore rilevata nei {colpi_max} colpi utili."

    def trova_lunghetta_garantita(self, status_callback, **params):
        from itertools import combinations
        
        self.carica_dati_per_ruote(params['lista_ruote'], status_callback)
        status_callback("Preparazione dei cicli di analisi...")

        dati_per_analisi_completi = []
        for ruota in params['lista_ruote']:
            archivio_filtrato = self._filtra_dati_per_giorno(self.estrazioni.get(ruota, []), params['giorno_settimana'])
            universo_dati = [e for e in archivio_filtrato if e['data'] <= params['data_riferimento']]
            estrazioni_totali_necessarie = params['lunghezza_ciclo'] * params['num_cicli_analisi']
            if len(universo_dati) < estrazioni_totali_necessarie:
                status_callback(f"Dati insufficienti per {ruota}. La ruota sarà esclusa."); continue
            dati_per_analisi_completi.extend(universo_dati[-estrazioni_totali_necessarie:])

        if not dati_per_analisi_completi: return "Dati insufficienti su tutte le ruote selezionate per eseguire l'analisi."
        dati_per_analisi_completi.sort(key=lambda x: x['data'])
        
        lista_cicli_numeri = []
        for i in range(0, len(dati_per_analisi_completi), params['lunghezza_ciclo']):
            blocco_estrazioni = dati_per_analisi_completi[i : i + params['lunghezza_ciclo']]
            numeri_del_ciclo = set(num for estrazione in blocco_estrazioni for num in estrazione['numeri'])
            lista_cicli_numeri.append(numeri_del_ciclo)
        
        status_callback("Selezione dei numeri candidati più frequenti...")
        conteggio_globale = Counter(num for ciclo in lista_cicli_numeri for num in ciclo)
        candidati = [num for num, freq in conteggio_globale.most_common(25)]
        
        if len(candidati) < params['dimensione_lunghetta']:
            return f"Non ci sono abbastanza numeri candidati ({len(candidati)}) per formare una lunghetta di {params['dimensione_lunghetta']}."

        status_callback(f"Test delle combinazioni per lunghette di {params['dimensione_lunghetta']}...")
        
        try: from math import comb; tot_combinazioni = comb(len(candidati), params['dimensione_lunghetta'])
        except ImportError: tot_combinazioni = 'molte'

        lunghette_con_dati = []
        num_cicli_analisi = len(lista_cicli_numeri)
        
        for i, lunghetta_tuple in enumerate(combinations(candidati, params['dimensione_lunghetta'])):
            if i % 500 == 0: status_callback(f"Test combinazione {i}/{tot_combinazioni}...")
            lunghetta = set(lunghetta_tuple)
            is_garantita = True; punteggio_totale = 0; ultimo_ciclo_con_vincita = -1
            for idx_ciclo, ciclo in enumerate(lista_cicli_numeri):
                punti_nel_ciclo = len(lunghetta.intersection(ciclo))
                if punti_nel_ciclo < 2: is_garantita = False; break 
                punteggio_totale += punti_nel_ciclo; ultimo_ciclo_con_vincita = idx_ciclo
            if is_garantita:
                ritardo_ciclico = (num_cicli_analisi - 1) - ultimo_ciclo_con_vincita
                lunghette_con_dati.append({'lunghetta': sorted(list(lunghetta)), 'punteggio': punteggio_totale, 'ritardo': ritardo_ciclico})
        
        if not lunghette_con_dati: return None

        lunghette_ordinate = sorted(lunghette_con_dati, key=lambda x: (x['punteggio'], x['ritardo']), reverse=True)
        miglior_lunghetta_dati = lunghette_ordinate[0]

        # --- NUOVA LOGICA DI SIMULAZIONE GIOCO ---
        status_callback("Simulazione di gioco sulla migliore lunghetta...")
        set_migliore_lunghetta = set(miglior_lunghetta_dati['lunghetta'])
        vincite_simulate = 0
        for i in range(0, len(dati_per_analisi_completi), params['lunghezza_ciclo']):
            ciclo_di_estrazioni = dati_per_analisi_completi[i : i + params['lunghezza_ciclo']]
            # Considera solo i 'colpi di gioco' iniziali di ogni ciclo
            gioco_simulato = ciclo_di_estrazioni[:params['colpi_di_gioco']]
            for estrazione in gioco_simulato:
                if len(set_migliore_lunghetta.intersection(set(estrazione['numeri']))) >= 2:
                    vincite_simulate += 1
                    break # Trovata vincita nel ciclo, passa al successivo
        
        miglior_lunghetta_dati['vincite_simulate'] = vincite_simulate
        # --- FINE NUOVA LOGICA ---

        status_callback("Analisi completata.")
        return miglior_lunghetta_dati

# ==============================================================================
# CLASSE "INTERFACCIA GRAFICA" (Frontend)
# ==============================================================================
class LottoApp(tk.Tk):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.title("CICLICA - Created by Max Lotto -"); self.geometry("850x700")
        self.ultima_previsione = None 
        style = ttk.Style(self); style.theme_use('clam'); style.configure('TButton', padding=6, relief="flat", background="#e1e1e1"); style.configure('TLabel', padding=5)

        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        controls_frame = ttk.LabelFrame(main_frame, text="Impostazioni Analisi", padding="10"); controls_frame.pack(fill=tk.X, expand=False, pady=5)
        
        ttk.Label(controls_frame, text="Fonte Dati:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        self.source_var = tk.StringVar(value="github")
        ttk.Radiobutton(controls_frame, text="Online", variable=self.source_var, value="github", command=self.toggle_local_path).grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(controls_frame, text="Locale", variable=self.source_var, value="local", command=self.toggle_local_path).grid(row=0, column=2, sticky='w')
        self.local_path_button = ttk.Button(controls_frame, text="Sfoglia...", command=self.browse_folder, state=tk.DISABLED); self.local_path_button.grid(row=0, column=3, padx=5)
        
        ttk.Label(controls_frame, text="Ruote:", font=('Helvetica', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=5)
        ruote_frame = ttk.Frame(controls_frame); ruote_frame.grid(row=1, column=1, columnspan=5, sticky='w')
        self.ruote_vars = {}
        for i, (code, name) in enumerate(analyzer.RUOTE_DISPONIBILI.items()):
            var = tk.BooleanVar(); cb = ttk.Checkbutton(ruote_frame, text=code, variable=var); cb.grid(row=0, column=i, padx=2); self.ruote_vars[code] = var

        params_frame = ttk.Frame(controls_frame); params_frame.grid(row=2, column=0, columnspan=6, pady=10, sticky='w')
        self.data_label = self.add_param_entry(params_frame, "Data Riferimento:", 0, 0, is_date=True)
        self.add_param_entry(params_frame, "Numero Cicli:", 1, 0)
        self.add_param_entry(params_frame, "Estrazioni per Ciclo:", 2, 0)
        self.add_param_entry(params_frame, "Ambate da trovare:", 0, 2)
        self.add_param_entry(params_frame, "Abbinamenti da trovare:", 1, 2)
        self.add_param_entry(params_frame, "Colpi di gioco:", 2, 2)
        self.add_param_entry(params_frame, "Dimensione Lunghetta Garantita:", 3, 0)

        ttk.Label(params_frame, text="Giorno Settimana:").grid(row=0, column=4, sticky='e', padx=(10, 5))
        self.giorno_var = ttk.Combobox(params_frame, values=["Tutti", "Martedì", "Giovedì", "Venerdì", "Sabato"], state="readonly", width=12); self.giorno_var.set("Tutti")
        self.giorno_var.grid(row=0, column=5, sticky='w')

        action_frame = ttk.Frame(main_frame, padding="5 0"); action_frame.pack(fill=tk.X, expand=True)
        self.run_button = ttk.Button(action_frame, text="ANALISI CONVERGENZA", command=self.start_analysis_thread); self.run_button.pack(side=tk.LEFT, padx=(0,5), expand=True)
        self.lunghetta_button = ttk.Button(action_frame, text="TROVA LUNGHETTA GARANTITA", command=self.start_lunghetta_garantita_thread); self.lunghetta_button.pack(side=tk.LEFT, padx=5, expand=True)
        self.verify_button = ttk.Button(action_frame, text="Verifica Ultima Previsione", command=self.start_verifica_esiti_thread, state=tk.DISABLED); self.verify_button.pack(side=tk.LEFT, padx=(5,0), expand=True)

        status_frame = ttk.Frame(main_frame); status_frame.pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(status_frame, text="Pronto."); self.status_label.pack()

        output_frame = ttk.LabelFrame(main_frame, text="Risultati", padding="10"); output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=15, font=('Courier New', 10)); self.output_text.pack(fill=tk.BOTH, expand=True)
        
        self.queue = queue.Queue(); self.process_queue()
        self.configure_tags()

    def add_param_entry(self, parent, label_text, r, c, is_date=False):
        label = ttk.Label(parent, text=label_text); label.grid(row=r, column=c, sticky='w', padx=(0, 5), pady=2)
        if is_date:
            entry = DateEntry(parent, width=12, date_pattern='dd/mm/yyyy'); entry.set_date(datetime.now())
        else: entry = ttk.Entry(parent, width=10, justify='center')
        entry.grid(row=r, column=c+1, sticky='w', padx=(0, 20))
        setattr(self, label_text.lower().replace(" ", "_").replace(":", ""), entry)
        return label

    def toggle_local_path(self): self.local_path_button.config(state=tk.NORMAL if self.source_var.get() == "local" else tk.DISABLED)

    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path: self.analyzer.local_path = folder_path; self.status_label.config(text=f"Cartella: {folder_path}")
        
    def start_analysis_thread(self):
        try:
            # --- AGGIUNGI QUESTA RIGA QUI ---
            # Comunica al motore di analisi quale fonte dati usare (Online o Locale)
            self.analyzer.data_source = self.source_var.get()
            # --- FINE DELLA MODIFICA ---

            params = {
                "lista_ruote": [code for code, var in self.ruote_vars.items() if var.get()],
                "data_riferimento": self.data_riferimento.get_date(),
                "num_cicli_analisi": int(self.numero_cicli.get()), "lunghezza_ciclo": int(self.estrazioni_per_ciclo.get()),
                "num_ambate": int(self.ambate_da_trovare.get()), "num_abbinamenti": int(self.abbinamenti_da_trovare.get()),
                "colpi_di_gioco": int(self.colpi_di_gioco.get()), "giorno_settimana": self.giorno_var.get()
            }
            if not params["lista_ruote"]: messagebox.showerror("Errore", "Seleziona almeno una ruota."); return
            # Ora questo controllo funziona in coppia con l'impostazione che abbiamo appena aggiunto
            if self.source_var.get() == 'local' and not self.analyzer.local_path: messagebox.showerror("Errore", "Seleziona una cartella."); return

            self.run_button.config(state=tk.DISABLED); self.verify_button.config(state=tk.DISABLED)
            self.output_text.delete(1.0, tk.END)
            self.ultima_previsione = None 
            threading.Thread(target=self.run_analysis_logic, args=(params,), daemon=True).start()
        except ValueError: messagebox.showerror("Errore di Input", "Controlla i campi numerici."); self.on_analysis_end()
        except Exception as e: messagebox.showerror("Errore Imprevisto", str(e)); self.on_analysis_end()

    def start_lunghetta_garantita_thread(self):
        try:
            self.analyzer.data_source = self.source_var.get()
            params = {
                "lista_ruote": [code for code, var in self.ruote_vars.items() if var.get()],
                "data_riferimento": self.data_riferimento.get_date(),
                "num_cicli_analisi": int(self.numero_cicli.get()),
                "lunghezza_ciclo": int(self.estrazioni_per_ciclo.get()),
                "dimensione_lunghetta": int(self.dimensione_lunghetta_garantita.get()),
                "giorno_settimana": self.giorno_var.get(),
                # --- AGGIUNGI QUESTO PARAMETRO ESSENZIALE ---
                "colpi_di_gioco": int(self.colpi_di_gioco.get())
            }
            if not params["lista_ruote"]: messagebox.showerror("Errore", "Seleziona almeno una ruota."); return
            if params["dimensione_lunghetta"] < 2 or params["dimensione_lunghetta"] > 10: messagebox.showerror("Errore", "La dimensione della lunghetta deve essere tra 2 e 10."); return
            if self.source_var.get() == 'local' and not self.analyzer.local_path: messagebox.showerror("Errore", "Seleziona una cartella."); return

            self.run_button.config(state=tk.DISABLED); self.lunghetta_button.config(state=tk.DISABLED); self.verify_button.config(state=tk.DISABLED)
            self.output_text.delete(1.0, tk.END)
            self.ultima_previsione = None 
            threading.Thread(target=self.run_lunghetta_garantita_logic, args=(params,), daemon=True).start()
        except ValueError: messagebox.showerror("Errore di Input", "Controlla i campi numerici."); self.on_analysis_end()
        except Exception as e: messagebox.showerror("Errore Imprevisto", str(e)); self.on_analysis_end()

    def run_lunghetta_garantita_logic(self, params):
        try:
            # --- MODIFICA QUI ---
            # Ora passiamo la funzione di status come argomento separato, come dovrebbe essere
            result = self.analyzer.trova_lunghetta_garantita(
                status_callback=lambda msg: self.queue.put(('status', msg)),
                **params
            )
            self.queue.put(('lunghetta_result', result))
        except Exception as e: 
            self.queue.put(('error', f"ERRORE DURANTE LA RICERCA LUNGHETTA: {e}"))

    def start_verifica_esiti_thread(self):
        if not self.ultima_previsione: messagebox.showwarning("Attenzione", "Nessuna previsione da verificare."); return
        self.run_button.config(state=tk.DISABLED); self.verify_button.config(state=tk.DISABLED)
        threading.Thread(target=self.run_verifica_logic, daemon=True).start()

    def run_analysis_logic(self, params):
        try:
            result = self.analyzer.esegui_analisi_completa(status_callback=lambda msg: self.queue.put(('status', msg)), **params)
            self.queue.put(('result', result))
        except Exception as e: self.queue.put(('error', f"ERRORE DURANTE L'ANALISI: {e}"))

    def run_verifica_logic(self):
        try:
            result = self.analyzer.verifica_esiti_previsione(self.ultima_previsione, status_callback=lambda msg: self.queue.put(('status', msg)))
            self.queue.put(('esito', result))
        except Exception as e: self.queue.put(('error', f"ERRORE DURANTE LA VERIFICA: {e}"))

    def display_lunghetta_results(self, result):
        self.output_text.delete(1.0, tk.END)
        if isinstance(result, str):
            self.output_text.insert(tk.END, result, 'negative'); return

        self.output_text.insert(tk.END, "RICERCA DELLA MIGLIORE LUNGHETTA GARANTITA\n", 'title')
        dim_richiesta = self.dimensione_lunghetta_garantita.get()
        self.output_text.insert(tk.END, f"Parametri usati: {self.numero_cicli.get()} cicli da {self.estrazioni_per_ciclo.get()} estrazioni.\n", 'subheader')
        self.output_text.insert(tk.END, f"Dimensione richiesta: {dim_richiesta} numeri.\n\n", 'subheader')
        
        if not result:
            self.output_text.insert(tk.END, "NESSUNA LUNGHETTA TROVATA\n", 'negative')
            self.output_text.insert(tk.END, "Nessuna combinazione ha garantito l'ambo in tutti i cicli analizzati.\n\n", 'subheader')
            self.output_text.insert(tk.END, "Suggerimenti:\n- Prova ad aumentare la dimensione della lunghetta.\n- Prova a diminuire il numero di cicli di analisi.", 'subheader')
        else:
            lunghetta = result['lunghetta']
            punteggio = result['punteggio']
            ritardo = result['ritardo']
            vincite_simulate = result.get('vincite_simulate', 0)
            cicli_totali = int(self.numero_cicli.get())
            colpi_gioco = int(self.colpi_di_gioco.get())
            percentuale = (vincite_simulate / cicli_totali) * 100 if cicli_totali > 0 else 0
            
            num_str = " - ".join(map(str, lunghetta))
            
            self.output_text.insert(tk.END, "La Migliore Lunghetta Trovata:\n\n", 'positive')
            self.output_text.insert(tk.END, f"  Numeri: ", 'header')
            self.output_text.insert(tk.END, f"{num_str}\n\n", 'ambata_color')
            
            self.output_text.insert(tk.END, "Statistiche di Performance:\n", 'header')
            self.output_text.insert(tk.END, f"- Punteggio Totale: {punteggio} (somma dei punti: Ambo=2, Terno=3, etc.)\n", 'subheader')
            self.output_text.insert(tk.END, f"- Ritardo Ciclico: {ritardo} cicli (da quanti cicli interi non esce l'ambo)\n\n", 'subheader')

            # --- MODIFICA INIZIA QUI ---

            # Logica per scegliere il nome corretto della formazione
            dimensione_trovata = len(lunghetta)
            nomi_formazioni = {
                2: "coppia", 3: "terzina", 4: "quartina", 5: "cinquina",
                6: "sestina", 7: "settina", 8: "ottina", 9: "novina", 10: "decina"
            }
            nome_formazione = nomi_formazioni.get(dimensione_trovata, "formazione")

            # VISUALIZZAZIONE DELLA SIMULAZIONE
            self.output_text.insert(tk.END, "Simulazione di Gioco (Back-test):\n", 'header')
            # Usa la variabile 'nome_formazione' invece della parola fissa "sestina"
            self.output_text.insert(tk.END, f"Giocando questa {nome_formazione} per i primi {colpi_gioco} colpi di ogni ciclo:\n", 'subheader')
            
            # --- MODIFICA FINISCE QUI ---
            
            self.output_text.insert(tk.END, f"- Esito: ", 'subheader')
            # Ho lasciato la tua logica originale per il colore del testo
            self.output_text.insert(tk.END, f"Vincente in {vincite_simulate} su {cicli_totali} cicli ({percentuale:.1f}%)\n", 'positive' if percentuale > 50 else 'negative')

    def process_queue(self):
        try:
            msg_type, data = self.queue.get_nowait()
            if msg_type == 'status': self.status_label.config(text=data)
            elif msg_type == 'error': messagebox.showerror("Errore", data); self.on_analysis_end()
            elif msg_type == 'result': self.display_results(data); self.on_analysis_end()
            elif msg_type == 'esito': messagebox.showinfo("Esito Verifica", data); self.on_analysis_end()
            # --- AGGIUNGI QUESTO ELIF ---
            elif msg_type == 'lunghetta_result': self.display_lunghetta_results(data); self.on_analysis_end()
        except queue.Empty: pass
        finally: self.after(100, self.process_queue)
        
    def on_analysis_end(self): 
        self.run_button.config(state=tk.NORMAL)
        self.lunghetta_button.config(state=tk.NORMAL) # <-- Aggiungi questa riga
        if self.ultima_previsione:
            self.verify_button.config(state=tk.NORMAL)
        self.status_label.config(text="Pronto.")
    
    def configure_tags(self):
        self.output_text.tag_configure('title', font=('Courier New', 12, 'bold'), justify='center', spacing3=10)
        self.output_text.tag_configure('header', font=('Courier New', 10, 'bold'), foreground='#003366', spacing1=8)
        self.output_text.tag_configure('subheader', font=('Courier New', 10, 'italic'), foreground='#555555')
        self.output_text.tag_configure('positive', font=('Courier New', 10, 'bold'), foreground='green')
        self.output_text.tag_configure('negative', font=('Courier New', 10, 'bold'), foreground='red')
        self.output_text.tag_configure('ambata_color', foreground='#FF0000', font=('Courier New', 10, 'bold')) 
        self.output_text.tag_configure('abbinamento_color', foreground='#00AA00', font=('Courier New', 10, 'bold'))

    def display_results(self, result):
        self.output_text.delete(1.0, tk.END)
        if isinstance(result, str):
            self.output_text.insert(tk.END, result, 'negative')
            return
        if not result or not result.get('previsione'):
            self.output_text.insert(tk.END, "Analisi non ha prodotto risultati.", 'negative')
            return

        previsione, verifica, ruote_usate = result['previsione'], result['verifica'], result['ruote_usate']
        self.output_text.insert(tk.END, f"ANALISI DI CONVERGENZA SU: {', '.join(ruote_usate)}\n", 'title')
        
        ambate_set = {a['numero'] for a in previsione['ambate']}
        abbinamenti_set = {int(n[0]) for n in previsione['abbinamenti']}
        giocata_completa = sorted(list(ambate_set.union(abbinamenti_set)))
        
        self.ultima_previsione = {
            'ruote': ruote_usate, 'ambate': ambate_set, 'lunghetta': giocata_completa,
            'colpi': int(self.colpi_di_gioco.get()), 'data_previsione': result['data_riferimento_usata'],
            'giorno_settimana': self.giorno_var.get()
        }
        
        # --- SEZIONE PREVISIONE ---
        self.output_text.insert(tk.END, "\n--- PREVISIONE GENERATA ---\n", 'header')
        num_cicli_mostrati = verifica.get('cicli_totali', int(self.numero_cicli.get()))
        self.output_text.insert(tk.END, f"Analisi basata su {num_cicli_mostrati} cicli terminati il {result['data_riferimento_usata'].strftime('%d/%m/%Y')}.\n", 'subheader')
        
        self.output_text.insert(tk.END, "\n>> MIGLIORI ESTRATTI PER CONVERGENZA CICLICA\n", 'subheader')
        for ambata in previsione['ambate']:
            max_punteggio = num_cicli_mostrati * len(ruote_usate)
            ritardi_str = ", ".join([f"{r}:{d}" for r, d in ambata['ritardi_ruote'].items()])
            self.output_text.insert(tk.END, f"- Estratto "); self.output_text.insert(tk.END, f"{ambata['numero']:<2}", 'ambata_color')
            self.output_text.insert(tk.END, f": Punteggio {ambata['punteggio_convergenza']}/{max_punteggio} | Ritardi Attuali: {ritardi_str}\n")
            dettagli_str = "  (Dettaglio Presenza: " + ", ".join([f"{r}: {c}/{num_cicli_mostrati}" for r, c in ambata['dettagli_ruote'].items()]) + ")\n"
            self.output_text.insert(tk.END, dettagli_str, 'subheader')

        self.output_text.insert(tk.END, "\n>> MIGLIORI ABBINAMENTI PER AMBO\n", 'subheader')
        self.output_text.insert(tk.END, f"Numeri suggeriti: ")
        abbinamenti_list = [str(n[0]) for n in previsione['abbinamenti']]
        for i, num_str in enumerate(abbinamenti_list):
            self.output_text.insert(tk.END, num_str, 'abbinamento_color')
            if i < len(abbinamenti_list) - 1: self.output_text.insert(tk.END, ", ")
        self.output_text.insert(tk.END, "\n")
        
        self.output_text.insert(tk.END, f"\nProposta per LUNGHETTA: ")
        for i, num in enumerate(giocata_completa):
            tag = 'ambata_color' if num in ambate_set else 'abbinamento_color'
            self.output_text.insert(tk.END, str(num), tag)
            if i < len(giocata_completa) - 1: self.output_text.insert(tk.END, " - ")
        self.output_text.insert(tk.END, "\n")
        
        # --- SEZIONE VERIFICA STATISTICA (CON DETTAGLI) ---
        self.output_text.insert(tk.END, "\n--- VERIFICA STATISTICA DELLA STRATEGIA ---\n", 'header')
        self.output_text.insert(tk.END, "Questa sezione mostra l'affidabilità storica della strategia, simulando\nla giocata sui cicli passati per darti un'idea della sua performance.\n", 'subheader')
        self.output_text.insert(tk.END, f"Simulazione: Giocare gli estratti proposti per i primi {self.colpi_di_gioco.get()} colpi di ogni ciclo.\n\n", 'subheader')
        
        self.output_text.insert(tk.END, "RISULTATO DELLA SIMULAZIONE:\n", 'header')
        
        cicli_totali = verifica['cicli_totali']
        copertura_globale = verifica.get('copertura_globale', 0)
        cop_perc = (copertura_globale / cicli_totali) * 100 if cicli_totali > 0 else 0
        tag = 'positive' if cop_perc >= 75 else 'negative'
        
        self.output_text.insert(tk.END, f"- Copertura: ", 'subheader')
        self.output_text.insert(tk.END, f"Vincente in {copertura_globale} su {cicli_totali} cicli ", tag)
        self.output_text.insert(tk.END, f"({cop_perc:.1f}%)\n", tag)
        
        cicli_scoperti = verifica.get('cicli_scoperti', [])
        if cicli_scoperti:
            # Mostra al massimo 10 cicli scoperti per non intasare l'output
            cicli_da_mostrare = ", ".join(map(str, cicli_scoperti[:10]))
            if len(cicli_scoperti) > 10:
                cicli_da_mostrare += "..."
            self.output_text.insert(tk.END, f"- Cicli Scoperti: ", 'subheader')
            self.output_text.insert(tk.END, f"N.{cicli_da_mostrare}\n", 'negative')
        else:
            self.output_text.insert(tk.END, f"- Cicli Scoperti: ", 'subheader')
            self.output_text.insert(tk.END, "Nessuno (copertura del 100%)\n", 'positive')

        self.output_text.insert(tk.END, "\n(Nota: una performance passata non garantisce risultati futuri).\n\n", 'subheader')

        # --- SEZIONE ESITO PREVISIONE ATTUALE ---
        self.output_text.insert(tk.END, "--- ESITO DELLA PREVISIONE ATTUALE ---\n", 'header')
        is_ricerca = self.data_riferimento.get_date() < datetime.now().date()
        data_riferimento = self.data_riferimento.get_date()
        
        if is_ricerca:
            self.output_text.insert(tk.END, f"Controllo esito nei {self.colpi_di_gioco.get()} colpi successivi al {data_riferimento.strftime('%d/%m/%Y')}.\n", 'subheader')
            if verifica.get('esito_vincita'):
                self.output_text.insert(tk.END, f"\nESITO BACK-TEST: POSITIVO\n", 'positive')
            else:
                self.output_text.insert(tk.END, f"\nESITO BACK-TEST: NEGATIVO\n", 'negative')
        else:
            self.output_text.insert(tk.END, f"Controllo esito nei {self.colpi_di_gioco.get()} colpi successivi al {data_riferimento.strftime('%d/%m/%Y')}.\n", 'subheader')
            colpi_verificati = verifica.get('colpi_verificati', 0)
            colpi_richiesti = verifica.get('colpi_richiesti', self.colpi_di_gioco.get())
            if colpi_verificati > 0 and colpi_verificati < colpi_richiesti:
                self.output_text.insert(tk.END, f"\nESITO: PREVISIONE IN CORSO\n", 'subheader')
            else:
                self.output_text.insert(tk.END, f"\nESITO: PREVISIONE IN GIOCO\n", 'positive')
            self.output_text.insert(tk.END, f"(verificati {colpi_verificati}/{colpi_richiesti} colpi)\n", 'subheader')

if __name__ == "__main__":
    analyzer = LottoAnalyzer()
    app = LottoApp(analyzer)
    app.mainloop()