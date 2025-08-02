# ==============================================================================
# SCRIPT CAPOLAVORO FINALE - VERSIONE CON SUGGERIMENTO SEMPRE VISIBILE
# ==============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import itertools
from collections import defaultdict
import requests
import queue
import threading
from datetime import datetime
import numpy as np

try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Libreria Mancante", "La libreria 'tkcalendar' non è installata.\n\nEsegui: pip install tkcalendar")
    exit()

# --- CLASSE 1: ArchivioLotto ---
class ArchivioLotto:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.estrazioni_per_ruota = {} 
        self.GITHUB_USER = "illottodimax"
        self.GITHUB_REPO = "Archivio"
        self.GITHUB_BRANCH = "main"
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
    
    def _log(self, message): 
        self.output_queue.put(message)

    def _parse_estrazioni(self, linee):
        parsed_data = []
        for linea in linee:
            parti = linea.strip().split('\t')
            if len(parti) >= 7:
                try:
                    data_str = datetime.strptime(parti[0], '%Y/%m/%d').strftime('%Y-%m-%d')
                    numeri = sorted([int(n) for n in parti[2:7]])
                    if len(numeri) == 5 and all(1 <= num <= 90 for num in numeri):
                        parsed_data.append({'data': data_str, 'numeri': numeri})
                except (ValueError, IndexError): 
                    continue
        return parsed_data[::-1]
    
    def carica_dati(self):
        self._log("Inizio caricamento archivio da GitHub...")
        for ruota_key, ruota_nome in self.RUOTE_DISPONIBILI.items():
            self._log(f"-> Caricamento {ruota_nome}...")
            url = f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{ruota_nome.upper()}.txt'
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    estrazioni = self._parse_estrazioni(response.text.strip().split('\n'))
                    if estrazioni:
                        self.estrazioni_per_ruota[ruota_nome.upper()] = estrazioni
                        self._log(f"   OK: Caricate {len(estrazioni)} estrazioni.")
                else: 
                    self._log(f"   ERRORE HTTP {response.status_code} per {ruota_nome}.")
            except requests.RequestException as e: 
                self._log(f"   ERRORE di rete per {ruota_nome}: {e}")
        self._log("\nArchivio caricato.")
        return self.estrazioni_per_ruota

# --- CLASSE 2: LottoAnalyzer ---
class LottoAnalyzer:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.mappa_codice_nome = {k: v.upper() for k, v in ArchivioLotto(None).RUOTE_DISPONIBILI.items()}
    
    def _log(self, message): 
        self.output_queue.put(message)

    def _prepara_dati_per_data(self, estrazioni_caricate, ruote_analisi):
        dati_per_data = defaultdict(list)
        for codice_ruota in ruote_analisi:
            nome_ruota = self.mappa_codice_nome.get(codice_ruota.upper())
            if nome_ruota in estrazioni_caricate:
                for estrazione in estrazioni_caricate[nome_ruota]: 
                    estrazione['ruota'] = nome_ruota
                    dati_per_data[estrazione['data']].append(estrazione)
        date_ordinate = sorted(dati_per_data.keys())
        return dati_per_data, date_ordinate

    def analizza_equilibrio_ciclico(self, estrazioni_caricate, ruote_analisi, data_fine_analisi_dt, num_cicli, ampiezza_ciclo_date):
        self._log("\n📊 Inizio analisi di equilibrio ciclico...")
        dati_per_data, date_ordinate = self._prepara_dati_per_data(estrazioni_caricate, ruote_analisi)
        data_fine_str = data_fine_analisi_dt.strftime('%Y-%m-%d')
        try: 
            indice_fine_analisi = next(i for i, data in reversed(list(enumerate(date_ordinate))) if data <= data_fine_str)
        except StopIteration: 
            self._log(f"ERRORE: Nessuna estrazione trovata prima del {data_fine_str}.")
            return None
        totale_date_necessarie = num_cicli * ampiezza_ciclo_date
        indice_inizio_analisi = indice_fine_analisi - totale_date_necessarie
        if indice_inizio_analisi < 0: 
            self._log(f"ERRORE: Dati insuff. Servono {totale_date_necessarie} date, trovate solo {indice_fine_analisi}.")
            return None
        date_da_analizzare = date_ordinate[indice_inizio_analisi:indice_fine_analisi]
        cicli_di_date = [date_da_analizzare[i:i + ampiezza_ciclo_date] for i in range(0, len(date_da_analizzare), ampiezza_ciclo_date)]
        self._log("Calcolo della 'storia' per ogni ambo...")
        storia_ambi = defaultdict(list)
        tutti_ambi_possibili = list(itertools.combinations(range(1, 91), 2))
        for ambo in tutti_ambi_possibili:
            for ciclo_date in cicli_di_date:
                conteggio = sum(1 for data in ciclo_date for estrazione in dati_per_data[data] if ambo[0] in estrazione['numeri'] and ambo[1] in estrazione['numeri'])
                storia_ambi[ambo].append(conteggio)
        self._log("Ricerca degli equilibri...")
        equilibri_trovati = defaultdict(list)
        for ambo, storia in storia_ambi.items(): 
            equilibri_trovati[tuple(storia)].append(ambo)
        coppie_in_equilibrio = {storia: ambi for storia, ambi in equilibri_trovati.items() if len(ambi) > 1}
        self._log("Analisi completata.")
        return {'coppie_in_equilibrio': coppie_in_equilibrio, 'dati_per_data': dati_per_data, 'date_ordinate': date_ordinate, 'indice_fine_analisi': indice_fine_analisi, 'indice_inizio_analisi': indice_inizio_analisi}

    def analizza_equilibrio_ambate(self, estrazioni_caricate, ruote_analisi, data_fine_analisi_dt, num_cicli, ampiezza_ciclo_date):
        self._log("\n📊 Inizio analisi di equilibrio ciclico per AMBATE...")
        dati_per_data, date_ordinate = self._prepara_dati_per_data(estrazioni_caricate, ruote_analisi)
        data_fine_str = data_fine_analisi_dt.strftime('%Y-%m-%d')
        try: 
            indice_fine_analisi = next(i for i, data in reversed(list(enumerate(date_ordinate))) if data <= data_fine_str)
        except StopIteration: 
            self._log(f"ERRORE: Nessuna estrazione trovata prima del {data_fine_str}.")
            return None
        totale_date_necessarie = num_cicli * ampiezza_ciclo_date
        indice_inizio_analisi = indice_fine_analisi - totale_date_necessarie
        if indice_inizio_analisi < 0: 
            self._log(f"ERRORE: Dati insuff. Servono {totale_date_necessarie} date, trovate solo {indice_fine_analisi}.")
            return None
        date_da_analizzare = date_ordinate[indice_inizio_analisi:indice_fine_analisi]
        cicli_di_date = [date_da_analizzare[i:i + ampiezza_ciclo_date] for i in range(0, len(date_da_analizzare), ampiezza_ciclo_date)]
        self._log("Calcolo della 'storia' per ogni numero...")
        storia_numeri = defaultdict(list)
        tutti_i_numeri_possibili = range(1, 91)
        for numero in tutti_i_numeri_possibili:
            for ciclo_date in cicli_di_date:
                conteggio = sum(1 for data in ciclo_date for estrazione in dati_per_data[data] if numero in estrazione['numeri'])
                storia_numeri[numero].append(conteggio)
        self._log("Ricerca degli equilibri...")
        equilibri_trovati = defaultdict(list)
        for numero, storia in storia_numeri.items(): 
            equilibri_trovati[tuple(storia)].append(numero)
        coppie_in_equilibrio = {storia: numeri for storia, numeri in equilibri_trovati.items() if len(numeri) > 1}
        self._log("Analisi completata.")
        return {'coppie_in_equilibrio': coppie_in_equilibrio, 'dati_per_data': dati_per_data, 'date_ordinate': date_ordinate, 'indice_fine_analisi': indice_fine_analisi, 'indice_inizio_analisi': indice_inizio_analisi}

    def analizza_equilibrio_numero_su_ruote(self, estrazioni_caricate, ruote_analisi, data_fine_analisi_dt, num_cicli, ampiezza_ciclo_date):
        self._log("\n📊 Inizio analisi di equilibrio di un numero su più ruote...")
        if len(ruote_analisi) < 2:
            self._log("ERRORE: Seleziona almeno due ruote di analisi per questa modalità.")
            return None

        tutte_le_date_ordinate = sorted(list(set(data for nome_ruota_mappa in estrazioni_caricate for estrazione in estrazioni_caricate[nome_ruota_mappa] for data in [estrazione['data']])))
        if not tutte_le_date_ordinate:
            self._log("ERRORE: Nessuna data trovata nell'archivio.")
            return None

        data_fine_str_originale = data_fine_analisi_dt.strftime('%Y-%m-%d')
        try: 
            indice_fine_analisi_globale = next(i for i, data in reversed(list(enumerate(tutte_le_date_ordinate))) if data <= data_fine_str_originale)
            data_fine_str_valida = tutte_le_date_ordinate[indice_fine_analisi_globale]
            if data_fine_str_valida != data_fine_str_originale:
                self._log(f"AVVISO: La data {data_fine_str_originale} non è un giorno di estrazione. L'analisi verrà eseguita fino alla data valida precedente: {data_fine_str_valida}.")
            data_valida_usata_dt = datetime.strptime(data_fine_str_valida, '%Y-%m-%d')
        except StopIteration: 
            self._log(f"ERRORE: Nessuna estrazione trovata prima o in data {data_fine_str_originale}.")
            return None

        storie_per_ruota = defaultdict(dict)
        data_fine_str = data_valida_usata_dt.strftime('%Y-%m-%d')

        for codice_ruota in ruote_analisi:
            nome_ruota = self.mappa_codice_nome.get(codice_ruota.upper())
            if nome_ruota not in estrazioni_caricate: continue

            dati_per_data_ruota, date_ordinate_ruota = self._prepara_dati_per_data(estrazioni_caricate, [codice_ruota])

            try:
                indice_fine_analisi = date_ordinate_ruota.index(data_fine_str)
            except ValueError:
                continue 
            
            totale_date_necessarie = num_cicli * ampiezza_ciclo_date
            if indice_fine_analisi + 1 < totale_date_necessarie:
                continue
            
            indice_inizio_analisi = (indice_fine_analisi + 1) - totale_date_necessarie
            date_da_analizzare = date_ordinate_ruota[indice_inizio_analisi : indice_fine_analisi + 1]
            
            cicli_di_date = [date_da_analizzare[i:i + ampiezza_ciclo_date] for i in range(0, len(date_da_analizzare), ampiezza_ciclo_date)]
            if len(cicli_di_date) < num_cicli: continue

            for numero in range(1, 91):
                storia_numero = [sum(1 for data in ciclo for estrazione in dati_per_data_ruota[data] if numero in estrazione['numeri']) for ciclo in cicli_di_date]
                storie_per_ruota[nome_ruota][numero] = tuple(storia_numero)
        
        self._log("Confronto delle storie tra le ruote...")
        equilibri_trovati = defaultdict(lambda: defaultdict(list))
        for nome_ruota, storie_numeri in storie_per_ruota.items():
            for numero, storia in storie_numeri.items():
                equilibri_trovati[numero][storia].append(nome_ruota)
        
        numeri_in_equilibrio = {}
        for numero, storie in equilibri_trovati.items():
            for storia, ruote in storie.items():
                if len(ruote) > 1:
                    if numero not in numeri_in_equilibrio:
                        numeri_in_equilibrio[numero] = []
                    numeri_in_equilibrio[numero].append({'storia': storia, 'ruote': ruote})
        
        self._log(f"Analisi completata. Trovati {len(numeri_in_equilibrio)} numeri in equilibrio.")
        
        # Ecco la modifica chiave: restituiamo un dizionario che contiene anche la data valida
        return {
            'numeri_in_equilibrio': numeri_in_equilibrio, 
            'estrazioni_caricate': estrazioni_caricate,
            'data_valida_usata': data_valida_usata_dt # <-- Passaggio del testimone corretto!
        }

    def trova_ruote_squilibrate(self, ambi_list, dati_per_data, date_periodo_analisi, n_ruote):
        squilibrio_per_ruota = {}
        nomi_ruote_analisi = {e['ruota'] for data in date_periodo_analisi for e in dati_per_data[data]}
        for nome_ruota in nomi_ruote_analisi:
            frequenze_locali = [sum(1 for data in date_periodo_analisi for e in dati_per_data[data] if e['ruota'] == nome_ruota and ambo[0] in e['numeri'] and ambo[1] in e['numeri']) for ambo in ambi_list]
            squilibrio_per_ruota[nome_ruota] = np.std(frequenze_locali)
        ruote_ordinate = sorted(squilibrio_per_ruota.items(), key=lambda item: item[1], reverse=True)
        return [ruota[0] for ruota in ruote_ordinate[:n_ruote]]

    def trova_ruote_squilibrate_ambate(self, numeri_list, dati_per_data, date_periodo_analisi, n_ruote):
        squilibrio_per_ruota = {}
        nomi_ruote_analisi = {e['ruota'] for data in date_periodo_analisi for e in dati_per_data[data]}
        for nome_ruota in nomi_ruote_analisi:
            frequenze_locali = [sum(1 for data in date_periodo_analisi for e in dati_per_data[data] if e['ruota'] == nome_ruota and numero in e['numeri']) for numero in numeri_list]
            squilibrio_per_ruota[nome_ruota] = np.std(frequenze_locali)
        ruote_ordinate = sorted(squilibrio_per_ruota.items(), key=lambda item: item[1], reverse=True)
        return [ruota[0] for ruota in ruote_ordinate[:n_ruote]]

    def trova_max_storico_equilibrio(self, ambi_in_equilibrio, dati_per_data, date_ordinate, ampiezza_ciclo_date, indice_inizio_analisi, num_cicli_attuali):
        max_equilibrio = num_cicli_attuali
        indice_ciclo_precedente = indice_inizio_analisi
        while indice_ciclo_precedente >= ampiezza_ciclo_date:
            inizio = indice_ciclo_precedente - ampiezza_ciclo_date
            fine = indice_ciclo_precedente
            ciclo_date_precedente = date_ordinate[inizio:fine]
            freq_primo_ambo = sum(1 for data in ciclo_date_precedente for estrazione in dati_per_data[data] if ambi_in_equilibrio[0][0] in estrazione['numeri'] and ambi_in_equilibrio[0][1] in estrazione['numeri'])
            equilibrio_mantenuto = True
            for i in range(1, len(ambi_in_equilibrio)):
                freq_corrente = sum(1 for data in ciclo_date_precedente for estrazione in dati_per_data[data] if ambi_in_equilibrio[i][0] in estrazione['numeri'] and ambi_in_equilibrio[i][1] in estrazione['numeri'])
                if freq_corrente != freq_primo_ambo:
                    equilibrio_mantenuto = False
                    break
            if equilibrio_mantenuto:
                max_equilibrio += 1
                indice_ciclo_precedente -= ampiezza_ciclo_date
            else:
                break
        return max_equilibrio

    def trova_max_storico_equilibrio_ambate(self, numeri_in_equilibrio, dati_per_data, date_ordinate, ampiezza_ciclo_date, indice_inizio_analisi, num_cicli_attuali):
        max_equilibrio = num_cicli_attuali
        indice_ciclo_precedente = indice_inizio_analisi
        while indice_ciclo_precedente >= ampiezza_ciclo_date:
            inizio = indice_ciclo_precedente - ampiezza_ciclo_date
            fine = indice_ciclo_precedente
            ciclo_date_precedente = date_ordinate[inizio:fine]
            freq_primo_numero = sum(1 for data in ciclo_date_precedente for estrazione in dati_per_data[data] if numeri_in_equilibrio[0] in estrazione['numeri'])
            equilibrio_mantenuto = True
            for i in range(1, len(numeri_in_equilibrio)):
                freq_corrente = sum(1 for data in ciclo_date_precedente for estrazione in dati_per_data[data] if numeri_in_equilibrio[i] in estrazione['numeri'])
                if freq_corrente != freq_primo_numero:
                    equilibrio_mantenuto = False
                    break
            if equilibrio_mantenuto:
                max_equilibrio += 1
                indice_ciclo_precedente -= ampiezza_ciclo_date
            else:
                break
        return max_equilibrio

    def verifica_sfaldamento(self, analisi_risultati, ruote_gioco, colpi_gioco):
        if not analisi_risultati: return 0
        date_future = analisi_risultati['date_ordinate'][analisi_risultati['indice_fine_analisi'] + 1:]
        nomi_ruote_gioco = [self.mappa_codice_nome[codice] for codice in ruote_gioco]
        date_effettivamente_controllate = len(date_future)
        for storia, dati_gruppo in analisi_risultati['coppie_in_equilibrio'].items():
            esiti_gruppo = {}
            ambi_list = dati_gruppo['ambi']
            for ambo in ambi_list:
                esito_ambo = {'esito': 'Negativo', 'dettagli': '', 'colpo_uscita': colpi_gioco + 1 if colpi_gioco is not None else float('inf')}
                date_controllate_per_ambo = 0
                for data in date_future:
                    if colpi_gioco is not None and date_controllate_per_ambo >= colpi_gioco: break
                    date_controllate_per_ambo += 1
                    for estrazione in analisi_risultati['dati_per_data'][data]:
                        if estrazione['ruota'] in nomi_ruote_gioco:
                            if ambo[0] in estrazione['numeri'] and ambo[1] in estrazione['numeri']:
                                esito_ambo['esito'] = 'Uscito'
                                esito_ambo['dettagli'] = f"in data {estrazione['data']} su {estrazione['ruota']} dopo {date_controllate_per_ambo} estrazioni (date)."
                                esito_ambo['colpo_uscita'] = date_controllate_per_ambo
                                break
                    if esito_ambo['esito'] == 'Uscito': break
                esiti_gruppo[ambo] = esito_ambo
            dati_gruppo['esiti'] = esiti_gruppo
        return date_effettivamente_controllate

    def verifica_sfaldamento_ambate(self, analisi_risultati, ruote_gioco, colpi_gioco):
        if not analisi_risultati: return 0
        date_future = analisi_risultati['date_ordinate'][analisi_risultati['indice_fine_analisi'] + 1:]
        nomi_ruote_gioco = [self.mappa_codice_nome[codice] for codice in ruote_gioco]
        date_effettivamente_controllate = len(date_future)
        for storia, dati_gruppo in analisi_risultati['coppie_in_equilibrio'].items():
            esiti_gruppo = {}
            numeri_list = dati_gruppo['numeri']
            for numero in numeri_list:
                esito_numero = {'esito': 'Negativo', 'dettagli': '', 'colpo_uscita': colpi_gioco + 1 if colpi_gioco is not None else float('inf')}
                date_controllate_per_numero = 0
                for data in date_future:
                    if colpi_gioco is not None and date_controllate_per_numero >= colpi_gioco: break
                    date_controllate_per_numero += 1
                    for estrazione in analisi_risultati['dati_per_data'][data]:
                        if estrazione['ruota'] in nomi_ruote_gioco:
                            if numero in estrazione['numeri']:
                                esito_numero['esito'] = 'Uscito'
                                esito_numero['dettagli'] = f"in data {estrazione['data']} su {estrazione['ruota']} dopo {date_controllate_per_numero} estrazioni (date)."
                                esito_numero['colpo_uscita'] = date_controllate_per_numero
                                break
                    if esito_numero['esito'] == 'Uscito': break
                esiti_gruppo[numero] = esito_numero
            dati_gruppo['esiti'] = esiti_gruppo
        return date_effettivamente_controllate

# --- CLASSE 3: App ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EQUILIBRIUM - Analizzatore Equilibrio Ciclico - Created by Max Lotto")
        self.geometry("1100x750")
        self.estrazioni_caricate = None
        self.thread_attivo = None
        self.log_queue = queue.Queue()
        self.archivio = ArchivioLotto(self.log_queue)
        self.analyzer = LottoAnalyzer(self.log_queue)
        self.crea_widgets()
        self.processa_log_queue()
        self.carica_dati_iniziali()

    def crea_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        analysis_choice_frame = ttk.LabelFrame(main_frame, text="Tipo di Analisi", padding="10")
        analysis_choice_frame.pack(fill=tk.X, pady=5)
        self.analysis_type_var = tk.StringVar(value="Ambi")
        
        radio_ambi = ttk.Radiobutton(analysis_choice_frame, text="Analisi Equilibrio Ambi", variable=self.analysis_type_var, value="Ambi")
        radio_ambi.pack(side=tk.LEFT, padx=5)
        
        radio_ambate = ttk.Radiobutton(analysis_choice_frame, text="Analisi Equilibrio Ambate", variable=self.analysis_type_var, value="Ambate")
        radio_ambate.pack(side=tk.LEFT, padx=5)
        
        radio_numero_su_ruote = ttk.Radiobutton(analysis_choice_frame, text="Equilibrio di 1 Numero su Più Ruote", variable=self.analysis_type_var, value="NumeroSuRuote")
        radio_numero_su_ruote.pack(side=tk.LEFT, padx=5)

        config_frame = ttk.LabelFrame(main_frame, text="Parametri di Analisi", padding="10")
        config_frame.pack(fill=tk.X, pady=5)

        label_data = ttk.Label(config_frame, text="Analisi fino al:")
        label_data.grid(row=0, column=0, padx=5, pady=8, sticky="w")
        self.date_entry = DateEntry(config_frame, width=12, background='darkblue', foreground='white', borderwidth=2, locale='it_IT', date_pattern='dd/mm/yyyy')
        self.date_entry.grid(row=0, column=1, padx=5, pady=8, sticky="w")

        label_num_cicli = ttk.Label(config_frame, text="Numero Cicli:")
        label_num_cicli.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_cicli_var = tk.StringVar(value="12")
        entry_num_cicli = ttk.Entry(config_frame, textvariable=self.num_cicli_var, width=10)
        entry_num_cicli.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        label_ampiezza_ciclo = ttk.Label(config_frame, text="Ampiezza Ciclo (Date):")
        label_ampiezza_ciclo.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.ampiezza_ciclo_var = tk.StringVar(value="30")
        entry_ampiezza_ciclo = ttk.Entry(config_frame, textvariable=self.ampiezza_ciclo_var, width=10)
        entry_ampiezza_ciclo.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        label_colpi_gioco = ttk.Label(config_frame, text="Colpi di Gioco (Date):")
        label_colpi_gioco.grid(row=1, column=4, padx=5, pady=5, sticky="w")
        self.colpi_gioco_var = tk.StringVar(value="18")
        entry_colpi_gioco = ttk.Entry(config_frame, textvariable=self.colpi_gioco_var, width=10)
        entry_colpi_gioco.grid(row=1, column=5, padx=5, pady=5, sticky="w")
        
        label_max_gruppi = ttk.Label(config_frame, text="Max Gruppi da Mostrare:")
        label_max_gruppi.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.max_gruppi_var = tk.StringVar(value="5")
        entry_max_gruppi = ttk.Entry(config_frame, textvariable=self.max_gruppi_var, width=10)
        entry_max_gruppi.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        label_n_ruote = ttk.Label(config_frame, text="N. Ruote Suggerite:")
        label_n_ruote.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.n_ruote_suggerite_var = tk.StringVar(value="3")
        entry_n_ruote = ttk.Entry(config_frame, textvariable=self.n_ruote_suggerite_var, width=10)
        entry_n_ruote.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        ruote_frame_container = ttk.Frame(main_frame)
        ruote_frame_container.pack(fill=tk.X, pady=5)

        ruote_analisi_frame = ttk.LabelFrame(ruote_frame_container, text="Ruote di Analisi", padding="10")
        ruote_analisi_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.ruote_analisi_vars = {}
        for codice, nome in self.archivio.RUOTE_DISPONIBILI.items():
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(ruote_analisi_frame, text=nome, variable=var)
            chk.pack(anchor="w")
            self.ruote_analisi_vars[codice] = var

        ruote_gioco_frame = ttk.LabelFrame(ruote_frame_container, text="Ruote di Gioco (per Verifica)", padding="10")
        ruote_gioco_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.ruote_gioco_vars = {}
        for codice, nome in self.archivio.RUOTE_DISPONIBILI.items():
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(ruote_gioco_frame, text=nome, variable=var)
            chk.pack(anchor="w")
            self.ruote_gioco_vars[codice] = var
        
        self.start_button = ttk.Button(main_frame, text="Analizza e Verifica", command=self.avvia_analisi, state="disabled")
        self.start_button.pack(pady=10, fill=tk.X)

        monospace_font = ("Courier New", 10)
        self.log_text = scrolledtext.ScrolledText(main_frame, wrap=tk.NONE, height=15, font=monospace_font)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

    def avvia_analisi(self):
        if self.thread_attivo and self.thread_attivo.is_alive():
            messagebox.showwarning("Attenzione", "Un'analisi è già in corso.")
            return
        try:
            data_riferimento = self.date_entry.get_date()
            params = {
                'num_cicli': int(self.num_cicli_var.get()), 
                'ampiezza_ciclo': int(self.ampiezza_ciclo_var.get()), 
                'colpi_gioco': int(self.colpi_gioco_var.get()), 
                'max_gruppi': int(self.max_gruppi_var.get()), 
                'n_ruote_suggerite': int(self.n_ruote_suggerite_var.get())
            }
            if any(x <= 0 for x in params.values()): raise ValueError
        except ValueError:
            messagebox.showerror("Errore Input", "I parametri devono essere numeri interi positivi.")
            return
        ruote_analisi = [c for c, v in self.ruote_analisi_vars.items() if v.get()]
        ruote_gioco = [c for c, v in self.ruote_gioco_vars.items() if v.get()]
        if not ruote_analisi or not ruote_gioco:
            messagebox.showerror("Errore Input", "Seleziona almeno una ruota di analisi e una di gioco.")
            return
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')
        self.start_button.config(state="disabled")
        self.thread_attivo = threading.Thread(target=self._worker_analisi, args=(data_riferimento, ruote_analisi, ruote_gioco, params), daemon=True)
        self.thread_attivo.start()

    def _worker_analisi(self, data_riferimento, ruote_analisi, ruote_gioco, params):
        tipo_analisi = self.analysis_type_var.get()
        analisi_risultati = None

        if tipo_analisi == "NumeroSuRuote":
            # --- INIZIO MODIFICA ---
            # La funzione di analisi ora restituisce un dizionario che include la data valida utilizzata
            analisi_risultati = self.analyzer.analizza_equilibrio_numero_su_ruote(self.estrazioni_caricate, ruote_analisi, data_riferimento, params['num_cicli'], params['ampiezza_ciclo'])
            
            # Controlliamo se l'analisi ha prodotto risultati validi
            if analisi_risultati and analisi_risultati.get('numeri_in_equilibrio'):
                # Estraiamo la data corretta che l'analizzatore ha usato (potrebbe essere diversa da quella originale)
                data_valida_usata_dt = analisi_risultati['data_valida_usata']
                # Passiamo la data CORRETTA alla funzione di stampa
                self.stampa_risultati_numero_su_ruote(analisi_risultati, params, data_valida_usata_dt)
            else:
                 # Questo 'else' gestisce sia il caso di analisi fallita, sia il caso in cui non trova equilibri
                 if analisi_risultati is not None: # Se l'analisi è andata a buon fine ma non ha trovato nulla
                    self.log_queue.put("\nNessun numero trovato in equilibrio con i parametri scelti.")
            # --- FINE MODIFICA ---

        elif tipo_analisi == "Ambi":
            # Questa parte rimane invariata, ma per coerenza futura andrebbe aggiornata con la stessa logica
            analisi_risultati = self.analyzer.analizza_equilibrio_ciclico(self.estrazioni_caricate, ruote_analisi, data_riferimento, params['num_cicli'], params['ampiezza_ciclo'])
            if analisi_risultati and analisi_risultati.get('coppie_in_equilibrio'):
                self.log_queue.put("Calcolo del massimo storico e delle ruote suggerite per AMBI...")
                date_periodo_analisi = analisi_risultati['date_ordinate'][analisi_risultati['indice_inizio_analisi']:analisi_risultati['indice_fine_analisi']]
                storie_da_processare = list(analisi_risultati['coppie_in_equilibrio'].keys())
                for storia in storie_da_processare:
                    ambi_list = analisi_risultati['coppie_in_equilibrio'][storia]
                    max_eq = self.analyzer.trova_max_storico_equilibrio(ambi_list, analisi_risultati['dati_per_data'], analisi_risultati['date_ordinate'], params['ampiezza_ciclo'], analisi_risultati['indice_inizio_analisi'], params['num_cicli'])
                    ruote_suggerite = self.analyzer.trova_ruote_squilibrate(ambi_list, analisi_risultati['dati_per_data'], date_periodo_analisi, params['n_ruote_suggerite'])
                    analisi_risultati['coppie_in_equilibrio'][storia] = {'ambi': ambi_list, 'max_storico': max_eq, 'ruote_suggerite': ruote_suggerite}
                date_controllate = self.analyzer.verifica_sfaldamento(analisi_risultati, ruote_gioco, params['colpi_gioco'])
                self.stampa_risultati_su_gui(analisi_risultati, params, date_controllate, tipo_analisi)
        
        elif tipo_analisi == "Ambate":
            # Questa parte rimane invariata, ma per coerenza futura andrebbe aggiornata con la stessa logica
            analisi_risultati = self.analyzer.analizza_equilibrio_ambate(self.estrazioni_caricate, ruote_analisi, data_riferimento, params['num_cicli'], params['ampiezza_ciclo'])
            if analisi_risultati and analisi_risultati.get('coppie_in_equilibrio'):
                self.log_queue.put("Calcolo del massimo storico e delle ruote suggerite per AMBATE...")
                date_periodo_analisi = analisi_risultati['date_ordinate'][analisi_risultati['indice_inizio_analisi']:analisi_risultati['indice_fine_analisi']]
                storie_da_processare = list(analisi_risultati['coppie_in_equilibrio'].keys())
                for storia in storie_da_processare:
                    numeri_list = analisi_risultati['coppie_in_equilibrio'][storia]
                    max_eq = self.analyzer.trova_max_storico_equilibrio_ambate(numeri_list, analisi_risultati['dati_per_data'], analisi_risultati['date_ordinate'], params['ampiezza_ciclo'], analisi_risultati['indice_inizio_analisi'], params['num_cicli'])
                    ruote_suggerite = self.analyzer.trova_ruote_squilibrate_ambate(numeri_list, analisi_risultati['dati_per_data'], date_periodo_analisi, params['n_ruote_suggerite'])
                    analisi_risultati['coppie_in_equilibrio'][storia] = {'numeri': numeri_list, 'max_storico': max_eq, 'ruote_suggerite': ruote_suggerite}
                date_controllate = self.analyzer.verifica_sfaldamento_ambate(analisi_risultati, ruote_gioco, params['colpi_gioco'])
                self.stampa_risultati_su_gui(analisi_risultati, params, date_controllate, tipo_analisi)

        # Questo controllo generale rimane valido
        if not analisi_risultati:
            self.log_queue.put("\nAnalisi non riuscita o nessun gruppo in equilibrio trovato.")
        
        self.start_button.config(state="normal")
        self.thread_attivo = None
        
    def stampa_risultati_su_gui(self, analisi, params, date_controllate, tipo_analisi):
        self.log_queue.put("\n" + "="*90)
        self.log_queue.put("RISULTATI FINALI")
        self.log_queue.put("="*90)

        key_elementi = 'numeri' if tipo_analisi == "Ambate" else 'ambi'
        equilibri_validi = {k: v for k, v in analisi['coppie_in_equilibrio'].items() if isinstance(v, dict) and key_elementi in v and 'esiti' in v}
        
        if not equilibri_validi:
            self.log_queue.put("\nNessun gruppo valido da mostrare.")
            return

        sorted_equilibri = sorted(equilibri_validi.items(), key=lambda item: (len(item[1][key_elementi]), sum(item[0])))
        gruppi_al_massimo_storico = [eq for eq in sorted_equilibri if len(eq[1][key_elementi]) >= 2 and eq[1]['max_storico'] is not None and len(eq[0]) == eq[1]['max_storico']]
        gruppi_da_mostrare = gruppi_al_massimo_storico[:params['max_gruppi']]
        
        if not gruppi_da_mostrare: 
            self.log_queue.put("\nNessun gruppo ha raggiunto il suo massimo storico con i parametri scelti.")
            return
        
        self.log_queue.put(f"Mostrando i {len(gruppi_da_mostrare)} gruppi migliori che hanno raggiunto il loro massimo storico:")
        for i, (storia, dati_gruppo) in enumerate(gruppi_da_mostrare, 1):
            elementi_list = dati_gruppo[key_elementi]
            max_storico = dati_gruppo['max_storico']
            ruote_suggerite = dati_gruppo['ruote_suggerite']
            esiti_gruppo = dati_gruppo['esiti']
            
            self.log_queue.put(f"\n--- Gruppo n.{i} | Uscite Tot: {sum(storia)} | Equilibrio Raggiunto: {max_storico} cicli (Max Storico) ---")
            
            COL_WIDTH = 8
            CYCLE_COL_WIDTH = 6
            header_label = 'Numero' if tipo_analisi == "Ambate" else 'Ambo'
            header = f"{header_label:<{COL_WIDTH}}" + "".join([f"|{f'C{c+1}':^{CYCLE_COL_WIDTH-1}}" for c in range(len(storia))])
            self.log_queue.put(header)
            self.log_queue.put("-" * len(header))

            for elemento in elementi_list:
                elem_str = str(elemento) if tipo_analisi == "Ambate" else f"{elemento[0]}-{elemento[1]}"
                riga = f"{elem_str:<{COL_WIDTH}}" + "".join([f"|{str(f):^{CYCLE_COL_WIDTH-1}}" for f in storia])
                self.log_queue.put(riga)
            
            self.log_queue.put(f"\n  -> Suggerimento Strategico (basato sull'analisi):")
            self.log_queue.put(f"     Ruote più squilibrate: {', '.join(ruote_suggerite)}")
            
            self.log_queue.put(f"\n  -> ESITO (Verifica su {params['colpi_gioco']} colpi, ruote di gioco selezionate):")
            
            almeno_uno_uscito = any(esito['esito'] == 'Uscito' for esito in esiti_gruppo.values())
            
            if almeno_uno_uscito:
                self.log_queue.put("\n     STATO: EQUILIBRIO ROTTO. Previsione conclusa con esito positivo.")
                for elemento in elementi_list:
                    esito_elem = esiti_gruppo[elemento]
                    elem_str = str(elemento) if tipo_analisi == "Ambate" else f"{elemento[0]}-{elemento[1]}"
                    self.log_queue.put(f"     - {header_label} {elem_str}: {esito_elem['esito']}. {esito_elem['dettagli']}")
            else:
                if date_controllate >= params['colpi_gioco']:
                    self.log_queue.put(f"\n     STATO: ESITO NEGATIVO. I {params['colpi_gioco']} colpi sono trascorsi senza sfaldamento.")
                else:
                    colpi_rimanenti = params['colpi_gioco'] - date_controllate
                    self.log_queue.put(f"\n     STATO: PREVISIONE ATTIVA. L'equilibrio non si è ancora rotto.")
                    self.log_queue.put(f"     Sono trascorse {date_controllate} estrazioni su {params['colpi_gioco']}. Colpi rimanenti: {colpi_rimanenti}.")
            self.log_queue.put("\n" + "-"*40)

    def stampa_risultati_numero_su_ruote(self, analisi, params, data_riferimento):
        self.log_queue.put("\n" + "="*90)
        self.log_queue.put("RISULTATI FINALI - Equilibrio di un Numero su più Ruote")
        self.log_queue.put("="*90)
        
        numeri_da_mostrare = list(analisi['numeri_in_equilibrio'].items())[:params['max_gruppi']]
        if not numeri_da_mostrare:
            self.log_queue.put("\nNessun risultato da mostrare con i parametri scelti.")
            return

        estrazioni_caricate = analisi.get('estrazioni_caricate', {})
        data_riferimento_str = data_riferimento.strftime('%Y-%m-%d')
        colpi_gioco = params['colpi_gioco']

        dati_per_data = defaultdict(lambda: defaultdict(list))
        for ruota, estrazioni_ruota in estrazioni_caricate.items():
            for estrazione in estrazioni_ruota:
                dati_per_data[estrazione['data']][ruota] = estrazione['numeri']
        date_ordinate = sorted(dati_per_data.keys())
        
        try:
            # L'indice di partenza per la verifica è la data di analisi stessa.
            indice_partenza_verifica = date_ordinate.index(data_riferimento_str)
        except ValueError:
            self.log_queue.put(f"ERRORE: La data di analisi {data_riferimento_str} non corrisponde a una data di estrazione valida nell'archivio.")
            return

        for i, (numero, equilibri) in enumerate(numeri_da_mostrare, 1):
            for eq in equilibri:
                storia = eq['storia']
                ruote = eq['ruote']
                
                self.log_queue.put(f"\n--- Gruppo n.{i}: Numero {numero} in equilibrio su {len(ruote)} ruote ---")
                self.log_queue.put(f"Ruote Coinvolte: {', '.join(ruote)}")
                self.log_queue.put(f"Equilibrio per {len(storia)} cicli. Uscite Totali per ruota: {sum(storia)}")
                
                header = "".join([f"|{f'C{c+1}':^5}" for c in range(len(storia))])
                riga = "".join([f"|{str(f):^5}" for f in storia])
                self.log_queue.put(f"Storia  {header}|")
                self.log_queue.put(f"Uscite  {riga}|")

                # --- LOGICA DI VERIFICA DEFINITIVA ---
                self.log_queue.put(f"\n  -> Analisi Strategica (Verifica su {colpi_gioco} colpi):")
                
                esito_trovato = False
                # LA CORREZIONE CHIAVE: si parte da 'indice_partenza_verifica', non da +1.
                date_da_controllare = date_ordinate[indice_partenza_verifica : indice_partenza_verifica + colpi_gioco]
                
                for colpo, data_corrente in enumerate(date_da_controllare, 1):
                    estrazioni_giorno = dati_per_data[data_corrente]
                    for nome_ruota in ruote:
                        if numero in estrazioni_giorno.get(nome_ruota, []):
                            self.log_queue.put(f"     ESITO: Vinto al {colpo}° colpo!")
                            self.log_queue.put(f"     Il numero {numero} è sortito su {nome_ruota} in data {data_corrente}.")
                            self.log_queue.put(f"     PREVISIONE CONCLUSA.")
                            esito_trovato = True
                            break
                    if esito_trovato:
                        break
                
                if not esito_trovato:
                    colpi_verificati = len(date_da_controllare)
                    if colpi_verificati >= colpi_gioco:
                        self.log_queue.put(f"     ESITO: Negativo.")
                        self.log_queue.put(f"     Il numero non è sortito nei {colpi_gioco} colpi verificati.")
                    else:
                        self.log_queue.put(f"     SUGGERIMENTO: Giocare il numero {numero} come ambata sulle ruote: {', '.join(ruote)}.")
                        self.log_queue.put(f"     STATO: PREVISIONE ATTIVA. L'equilibrio non si è ancora rotto.")
                        self.log_queue.put(f"     (Verificati {colpi_verificati} colpi su {colpi_gioco} previsti, archivio terminato).")
            self.log_queue.put("\n" + "-"*40)

    def log(self, m):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, m + "\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)
    
    def processa_log_queue(self):
        try:
            while True: 
                self.log(self.log_queue.get_nowait())
        except queue.Empty: 
            pass
        self.after(100, self.processa_log_queue)
    
    def carica_dati_iniziali(self):
        self.log("Avvio caricamento dati...")
        self.thread_attivo = threading.Thread(target=self._worker_carica_dati, daemon=True)
        self.thread_attivo.start()
    
    def _worker_carica_dati(self):
        self.estrazioni_caricate = self.archivio.carica_dati()
        if self.estrazioni_caricate: 
            self.log_queue.put("✅ Archivio pronto.")
            self.start_button.config(state="normal")
        else: 
            self.log_queue.put("❌ ERRORE: Impossibile caricare i dati.")
        self.thread_attivo = None

if __name__ == "__main__":
    app = App()
    app.mainloop()