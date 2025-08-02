import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import requests
from datetime import datetime, timedelta
from collections import Counter
from itertools import combinations
import threading
import queue
import traceback

try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Libreria Mancante", "La libreria 'tkcalendar' non è installata.\n\nPer favore, installala eseguendo questo comando nel tuo terminale:\n\npip install tkcalendar")
    exit()

# --- CLASSE GESTIONE ARCHIVIO ---
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
        
        self.RUOTE_DISPONIBILI = {
            'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova',
            'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma',
            'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'
        }
        self.URL_RUOTE = {k: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{v.upper()}.txt' for k, v in self.RUOTE_DISPONIBILI.items()}
        
        self.data_source = 'GitHub'
        self.local_path = None

    def _log(self, message):
        self.output_queue.put(message)

    def inizializza(self, force_reload=False):
        self._log("Inizio inizializzazione archivio...")
        if self.data_source == 'Locale' and (not self.local_path or not os.path.isdir(self.local_path)):
            raise FileNotFoundError("Percorso locale non valido o non impostato.")

        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            if ruota_key in self.estrazioni_per_ruota and not force_reload:
                continue
            self._log(f"Caricando {ruota_nome} ({i+1}/{len(self.RUOTE_DISPONIBILI)})...")
            try:
                if self.data_source == 'GitHub':
                    linee = requests.get(self.URL_RUOTE[ruota_key], timeout=15).text.strip().split('\n')
                else:
                    with open(os.path.join(self.local_path, f"{ruota_nome.upper()}.txt"), 'r', encoding='utf-8') as f:
                        linee = f.readlines()
                self.estrazioni_per_ruota[ruota_key] = self._parse_estrazioni(linee)
            except Exception as e:
                raise RuntimeError(f"Impossibile caricare i dati per {ruota_nome}: {e}")
        
        self._prepara_dati_per_analisi()
        self._log("\nArchivio inizializzato e pronto.")

    def _parse_estrazioni(self, linee):
        parsed_data = []
        for l in linee:
            parts = l.strip().split('\t')
            if len(parts) >= 7:
                try:
                    data_str = datetime.strptime(parts[0], '%Y/%m/%d').strftime('%Y-%m-%d')
                    numeri = [int(n) for n in parts[2:7] if n.isdigit() and 1 <= int(n) <= 90]
                    if len(numeri) == 5:
                        parsed_data.append({'data': data_str, 'numeri': numeri})
                except (ValueError, IndexError):
                    pass
        return parsed_data

    def _prepara_dati_per_analisi(self):
        self._log("Preparo e allineo i dati per l'analisi...")
        tutte_le_date = {e['data'] for estrazioni in self.estrazioni_per_ruota.values() for e in estrazioni}
        self.date_ordinate = sorted(list(tutte_le_date))
        self.date_to_index = {data: i for i, data in enumerate(self.date_ordinate)}
        
        self.dati_per_analisi = {data: {ruota: None for ruota in self.RUOTE_DISPONIBILI} for data in self.date_ordinate}
        for ruota, estrazioni in self.estrazioni_per_ruota.items():
            for e in estrazioni:
                if e['data'] in self.dati_per_analisi:
                    self.dati_per_analisi[e['data']][ruota] = e['numeri']
        
        self.dati_per_analisi = {d: v for d, v in self.dati_per_analisi.items() if any(v.values())}
        self._log("Dati allineati.")

# --- CLASSE ANALIZZATORE ---
class AnalizzatoreMarker:
    def __init__(self, archivio, output_queue):
        self.archivio = archivio
        self.output_queue = output_queue

    def _log(self, message, end='\n'):
        self.output_queue.put(message + end)

    def trova_marker_per_bersaglio(self, params):
        bersaglio = params['bersaglio']
        ruote_bersaglio_keys = params['ruote_bersaglio_keys']
        start_date_str = params['start_date_str']
        end_date_str = params['end_date_str']
        finestra_antecedente = params['finestra_antecedente']
        ruote_marker_keys = params['ruote_marker_keys']
        
        bersaglio_set = set(bersaglio)
        self._log("\n--- Avvio Analisi Marker ---")
        self._log(f"Bersaglio: {'-'.join(map(str, bersaglio))} su {', '.join(ruote_bersaglio_keys)}")
        self._log(f"Periodo: dal {start_date_str} al {end_date_str}")
        
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            date_nel_range = [d for d in self.archivio.date_ordinate if start_dt <= datetime.strptime(d, '%Y-%m-%d') <= end_dt]
        except ValueError:
            self._log("Formato data non valido.")
            params['risultato_analisi'] = None
            return

        eventi_bersaglio_indici = []
        for data_str in date_nel_range:
            estrazione = self.archivio.dati_per_analisi.get(data_str, {})
            for ruota_b_key in ruote_bersaglio_keys:
                numeri_ruota = estrazione.get(ruota_b_key)
                if numeri_ruota and bersaglio_set.issubset(set(numeri_ruota)):
                    eventi_bersaglio_indici.append(self.archivio.date_to_index[data_str])
                    break 

        eventi_bersaglio_unici = sorted(list(set(eventi_bersaglio_indici)))
        num_eventi_trovati = len(eventi_bersaglio_unici)
        
        if num_eventi_trovati == 0:
            self._log("Nessuna sortita del bersaglio trovata nel periodo.")
            params['risultato_analisi'] = None
            return
        
        self._log(f"Trovate {num_eventi_trovati} sortite. Analizzo gli antecedenti...")
        
        # MODIFICATO: Inizializziamo due contatori, uno per i numeri e uno per gli ambi
        marker_numeri_counter = Counter()
        marker_ambi_counter = Counter() # NUOVO

        for indice_evento in eventi_bersaglio_unici:
            numeri_nella_finestra = set()
            for i in range(1, finestra_antecedente + 1):
                indice_antecedente = indice_evento - i
                if indice_antecedente < 0: break
                estrazione_antecedente = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[indice_antecedente], {})
                for ruota_m_key in ruote_marker_keys:
                    numeri = estrazione_antecedente.get(ruota_m_key)
                    if numeri: numeri_nella_finestra.update(numeri)
            
            # Conta i numeri singoli (logica esistente)
            for numero in numeri_nella_finestra:
                marker_numeri_counter[numero] += 1
            
            # NUOVO: Conta gli ambi se ci sono almeno 2 numeri nella finestra
            if len(numeri_nella_finestra) >= 2:
                # Usiamo combinations per generare tutte le coppie uniche
                for ambo in combinations(sorted(list(numeri_nella_finestra)), 2):
                    marker_ambi_counter[ambo] += 1
        
        self.output_queue.put("\n" + "="*80 + "\n")
        self.output_queue.put("--- ANALISI MARKER COMPLETATA ---\n")
        self.output_queue.put(f"Su un totale di {num_eventi_trovati} eventi, ecco le frequenze dei marker:\n")
        self.output_queue.put("="*80 + "\n")
        self.output_queue.put("\n>>> CLASSIFICA MARKER (NUMERI SINGOLI) <<<\n")
        for numero, freq in marker_numeri_counter.most_common(15):
            self.output_queue.put(f"  - Numero {numero:<2}: {freq:>4} volte ({(freq / num_eventi_trovati * 100):6.2f}%)\n")
        
        # NUOVO: Aggiungiamo la stampa della classifica degli ambi
        self.output_queue.put("\n>>> CLASSIFICA MARKER (AMBI) <<<\n")
        if not marker_ambi_counter:
            self.output_queue.put("  (Nessun ambo trovato con i parametri attuali)\n")
        else:
            for ambo, freq in marker_ambi_counter.most_common(15):
                ambo_str = f"{ambo[0]}-{ambo[1]}"
                self.output_queue.put(f"  - Ambo {ambo_str:<5}: {freq:>4} volte ({(freq / num_eventi_trovati * 100):6.2f}%)\n")

        # Memorizziamo solo i risultati dei numeri singoli per la logica successiva
        params['risultato_analisi'] = marker_numeri_counter

    def esegui_backtest(self, params):
        bersaglio = params['bersaglio']
        ruote_bersaglio_keys = params['ruote_bersaglio_keys']
        start_date_str = params['start_date_str']
        end_date_str = params['end_date_str']
        finestra_antecedente = params['finestra_antecedente']
        ruote_marker_keys = params['ruote_marker_keys']
        soglia_attivazione = params['soglia_attivazione']
        colpi_di_gioco = params['colpi_di_gioco']
        bersaglio_set = set(bersaglio)
        bersaglio_tipo = "Ambata" if len(bersaglio_set) == 1 else "Ambo"
        bersaglio_str = "-".join(map(str, bersaglio))
        self._log("\n" + "="*80)
        self._log("### INIZIO SIMULAZIONE BACKTEST ###")
        self._log(f"Periodo: da {start_date_str} a {end_date_str}")
        self._log("="*80)
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            date_nel_range_indices = [i for i, d in enumerate(self.archivio.date_ordinate) if start_dt <= datetime.strptime(d, '%Y-%m-%d') <= end_dt]
        except ValueError: self._log("Formato data non valido."); return
        previsioni_giocate = 0; vincite = 0; spesa_totale = 0.0; incasso_totale = 0.0
        dettaglio_vincite = Counter()
        i = 0
        while i < len(date_nel_range_indices):
            indice_data_corrente = date_nel_range_indices[i]
            data_corrente_str = self.archivio.date_ordinate[indice_data_corrente]
            indice_fine_studio = indice_data_corrente - 1
            if indice_fine_studio < 0: i += 1; continue
            eventi_passati = []
            for j in range(indice_fine_studio, -1, -1):
                estrazione_passata = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[j], {})
                for ruota_b_key in ruote_bersaglio_keys:
                    numeri_ruota = estrazione_passata.get(ruota_b_key)
                    if numeri_ruota and bersaglio_set.issubset(set(numeri_ruota)): eventi_passati.append(j); break
            if len(eventi_passati) < 10: i += 1; continue
            marker_counter_storico = Counter()
            for indice_evento_passato in eventi_passati:
                numeri_finestra = set()
                for k in range(1, finestra_antecedente + 1):
                    idx_ante = indice_evento_passato - k
                    if idx_ante < 0: break
                    estrazione_ante = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[idx_ante], {})
                    for ruota_m_key in ruote_marker_keys:
                        if estrazione_ante.get(ruota_m_key): numeri_finestra.update(estrazione_ante[ruota_m_key])
                for num in numeri_finestra: marker_counter_storico[num] += 1
            top_marker_storici = {num for num, freq in marker_counter_storico.most_common(soglia_attivazione)}
            if len(top_marker_storici) < soglia_attivazione: i += 1; continue
            numeri_usciti_finestra_corrente = set()
            for k in range(1, finestra_antecedente + 1):
                idx_ante_corrente = indice_data_corrente - k
                if idx_ante_corrente < 0: break
                estrazione_ante = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[idx_ante_corrente], {})
                for ruota_m_key in ruote_marker_keys:
                    if estrazione_ante.get(ruota_m_key): numeri_usciti_finestra_corrente.update(estrazione_ante[ruota_m_key])
            if len(top_marker_storici.intersection(numeri_usciti_finestra_corrente)) >= soglia_attivazione:
                previsioni_giocate += 1; self._log(f"\n[{data_corrente_str}] CONDIZIONE ATTIVATA. Gioco: {bersaglio_str}")
                vincita_trovata = False; colpi_effettuati = 0
                for colpo in range(1, colpi_di_gioco + 1):
                    colpi_effettuati += 1; spesa_totale += 1.0
                    indice_futuro = indice_data_corrente + colpo
                    if indice_futuro >= len(self.archivio.date_ordinate): self._log(f"  Colpo {colpo}: Fine archivio."); break
                    data_futura_str = self.archivio.date_ordinate[indice_futuro]
                    estrazione_futura = self.archivio.dati_per_analisi.get(data_futura_str, {})
                    for ruota_b_key in ruote_bersaglio_keys:
                        if estrazione_futura.get(ruota_b_key) and bersaglio_set.issubset(set(estrazione_futura[ruota_b_key])):
                            self._log(f"  Colpo {colpo} ({data_futura_str}): VINCITA su {ruota_b_key}!"); vincite += 1; dettaglio_vincite[colpo] += 1; vincita_trovata = True; incasso_totale += 10.34; break
                    if vincita_trovata: break
                if not vincita_trovata: self._log(f"  -> ESITO NEGATIVO dopo {colpi_effettuati} colpi.")
                i += colpi_effettuati
            else: i += 1
        self._log("\n" + "="*80); self._log("### REPORT FINALE BACKTEST ###"); self._log(f"Previsioni: {previsioni_giocate}, Vincite: {vincite}, Spesa: {spesa_totale:.2f}€, Incasso: {incasso_totale:.2f}€, Utile: {incasso_totale - spesa_totale:.2f}€")

    def controlla_segnali_attivi(self, params):
        # Estrai tutti i parametri necessari
        bersaglio = params['bersaglio']
        ruote_bersaglio_keys = params['ruote_bersaglio_keys']
        start_date_str = params['start_date_str'] # Data di inizio
        end_date_str = params['end_date_str'] # La data di controllo
        finestra_antecedente = params['finestra_antecedente']
        ruote_marker_keys = params['ruote_marker_keys']
        soglia_attivazione = params['soglia_attivazione']
        bersaglio_set = set(bersaglio)

        self._log(f"\nControllo segnali attivi alla data del {end_date_str}...")

        try:
            date_ordinate = self.archivio.date_ordinate
            date_to_index = self.archivio.date_to_index
            
            valid_end_dates = [d for d in date_ordinate if d <= end_date_str]
            if not valid_end_dates:
                raise ValueError(f"Nessuna data trovata <= {end_date_str}.")
            data_riferimento_effettiva = valid_end_dates[-1]
            indice_riferimento = date_to_index[data_riferimento_effettiva]
            if data_riferimento_effettiva != end_date_str:
                self._log(f"(Data effettiva usata: {data_riferimento_effettiva})")

            valid_start_dates = [d for d in date_ordinate if d >= start_date_str]
            if not valid_start_dates:
                self._log("La data di inizio è successiva all'ultima data in archivio.")
                return
            indice_inizio_ricerca = date_to_index[valid_start_dates[0]]

        except (ValueError, KeyError):
            self._log(f"Date non valide o non trovate nell'archivio.")
            return

        # --- LOGICA STORICA ---
        eventi_passati = []
        for j in range(indice_riferimento - 1, indice_inizio_ricerca - 1, -1):
            estrazione_passata = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[j], {})
            for ruota_b_key in ruote_bersaglio_keys:
                numeri_ruota = estrazione_passata.get(ruota_b_key)
                if numeri_ruota and bersaglio_set.issubset(set(numeri_ruota)):
                    eventi_passati.append(j)
                    break
        
        if not eventi_passati:
            self._log("Nessuna sortita storica del bersaglio trovata nel periodo specificato.")
            return
            
        marker_counter_storico = Counter()
        for indice_evento_passato in eventi_passati:
            numeri_finestra = set()
            for k in range(1, finestra_antecedente + 1):
                idx_ante = indice_evento_passato - k
                
                # ### ECCO LA CORREZIONE FINALE E DECISIVA ###
                # La condizione deve controllare l'indice di inizio, non solo < 0
                if idx_ante < indice_inizio_ricerca: break
                
                estrazione_ante = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[idx_ante], {})
                for ruota_m_key in ruote_marker_keys:
                    if estrazione_ante.get(ruota_m_key):
                        numeri_finestra.update(estrazione_ante[ruota_m_key])
            for num in numeri_finestra:
                marker_counter_storico[num] += 1

        top_marker_storici = {num for num, freq in marker_counter_storico.most_common(soglia_attivazione)}
        self._log(f"I {soglia_attivazione} marker storici più frequenti nel periodo {start_date_str} -> {data_riferimento_effettiva} sono: {sorted(list(top_marker_storici))}")

        if len(top_marker_storici) < soglia_attivazione:
            self._log(f"Attenzione: l'analisi storica ha prodotto solo {len(top_marker_storici)}/{soglia_attivazione} marker unici.")
        
        # --- CONTROLLO FINESTRA ANTECEDENTE ---
        numeri_usciti_recentemente = set()
        for i in range(1, finestra_antecedente + 1):
            indice_controllo = indice_riferimento - i
            if indice_controllo < indice_inizio_ricerca: break
            estrazione_recente = self.archivio.dati_per_analisi.get(self.archivio.date_ordinate[indice_controllo], {})
            for ruota_m_key in ruote_marker_keys:
                if estrazione_recente.get(ruota_m_key):
                    numeri_usciti_recentemente.update(estrazione_recente[ruota_m_key])
        
        marker_attivati = top_marker_storici.intersection(numeri_usciti_recentemente)
        
        self.output_queue.put("\n" + "="*80 + "\n")
        self.output_queue.put(f"### CONTROLLO SEGNALI ATTIVI ALLA DATA DEL {data_riferimento_effettiva} ###\n")
        self.output_queue.put(f"Bersaglio: {'-'.join(map(str, params['bersaglio']))}\n")
        if len(marker_attivati) >= soglia_attivazione:
            self.output_queue.put(">>> SEGNALE ATTIVO! <<<\n")
            self.output_queue.put(f"Trovati i marker necessari: {sorted(list(marker_attivati))}\n")
        else:
            self.output_queue.put(">>> NESSUN SEGNALE ATTIVO <<<\n")
            self.output_queue.put(f"(Trovati solo {len(marker_attivati)}/{soglia_attivazione} marker necessari)\n")
        self.output_queue.put("="*80 + "\n")

    def trova_previsioni_in_attesa(self, params):
        start_date_str = params['start_date_str']; end_date_str = params['end_date_str']
        ruote_bersaglio_keys = params['ruote_bersaglio_keys']; finestra_antecedente = params['finestra_antecedente']
        ruote_marker_keys = params['ruote_marker_keys']; soglia_attivazione = params['soglia_attivazione']
        colpi_di_gioco = params['colpi_di_gioco']
        self._log("\n" + "="*80); self._log(f"### ANALISI ULTIMA ATTIVAZIONE (dal {start_date_str} al {end_date_str}) ###"); self._log("="*80)
        try:
            date_ordinate = self.archivio.date_ordinate; date_to_index = self.archivio.date_to_index
            valid_start_dates = [d for d in date_ordinate if d >= start_date_str]
            if not valid_start_dates: self._log("La data di inizio è successiva all'ultima data in archivio."); return
            indice_inizio_ricerca = date_to_index[valid_start_dates[0]]
            valid_end_dates = [d for d in date_ordinate if d <= end_date_str]
            if not valid_end_dates: self._log("La data di fine è precedente alla prima data in archivio."); return
            indice_riferimento = date_to_index[valid_end_dates[-1]]
        except (KeyError, IndexError): self._log("Date di inizio o fine non valide."); return
        
        previsioni_in_attesa = []
        previsioni_sfaldate = []
        previsioni_negative = []

        for numero_da_testare in range(1, 91):
            self._log(f"Analizzando il numero {numero_da_testare}/90...", end='\r')
            bersaglio_set = {numero_da_testare}
            data_attivazione_trovata = None; indice_attivazione = -1
            
            # Trova l'ultima attivazione
            for i in range(1, len(date_ordinate)):
                indice_controllo = indice_riferimento - i
                if indice_controllo < indice_inizio_ricerca: break
                eventi_passati = []
                for j in range(indice_controllo - 1, indice_inizio_ricerca -1, -1):
                    estrazione_passata = self.archivio.dati_per_analisi.get(date_ordinate[j], {})
                    for ruota_b_key in ruote_bersaglio_keys:
                        if estrazione_passata.get(ruota_b_key) and bersaglio_set.issubset(set(estrazione_passata[ruota_b_key])):
                            eventi_passati.append(j); break
                if len(eventi_passati) < 10: continue
                marker_counter_storico = Counter()
                for indice_evento in eventi_passati:
                    numeri_finestra = set()
                    for k in range(1, finestra_antecedente + 1):
                        idx_ante = indice_evento - k
                        if idx_ante < indice_inizio_ricerca: break
                        estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante], {})
                        for ruota_m_key in ruote_marker_keys:
                            if estrazione_ante.get(ruota_m_key): numeri_finestra.update(estrazione_ante[ruota_m_key])
                    for num in numeri_finestra: marker_counter_storico[num] += 1
                top_marker_storici = {num for num, freq in marker_counter_storico.most_common(soglia_attivazione)}
                if len(top_marker_storici) < soglia_attivazione: continue
                numeri_usciti_finestra_corrente = set()
                for k in range(1, finestra_antecedente + 1):
                    idx_ante_corrente = indice_controllo - k
                    if idx_ante_corrente < indice_inizio_ricerca: break
                    estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante_corrente], {})
                    for ruota_m_key in ruote_marker_keys:
                        if estrazione_ante.get(ruota_m_key): numeri_usciti_finestra_corrente.update(estrazione_ante[ruota_m_key])
                if len(top_marker_storici.intersection(numeri_usciti_finestra_corrente)) >= soglia_attivazione:
                    indice_attivazione = indice_controllo; data_attivazione_trovata = date_ordinate[indice_attivazione]; break
            
            # Se un'attivazione è stata trovata, classificane l'esito
            if data_attivazione_trovata:
                colpi_trascorsi = indice_riferimento - indice_attivazione
                esito_trovato = False
                dettagli_sfaldamento = {}

                for colpo in range(1, colpi_di_gioco + 1):
                    if colpo > colpi_trascorsi: break
                    indice_verifica = indice_attivazione + colpo
                    estrazione_verifica = self.archivio.dati_per_analisi.get(date_ordinate[indice_verifica], {})
                    for ruota_b_key in ruote_bersaglio_keys:
                        if estrazione_verifica.get(ruota_b_key) and bersaglio_set.issubset(set(estrazione_verifica[ruota_b_key])):
                            esito_trovato = True
                            dettagli_sfaldamento = {'colpo': colpo, 'data': date_ordinate[indice_verifica], 'ruota': ruota_b_key}
                            break
                    if esito_trovato: break
                
                # Classifica il risultato
                if esito_trovato:
                    previsioni_sfaldate.append({'numero': numero_da_testare, 'data_attivazione': data_attivazione_trovata, 'dettagli_sfaldamento': dettagli_sfaldamento})
                elif colpi_trascorsi < colpi_di_gioco:
                    previsioni_in_attesa.append({'numero': numero_da_testare, 'data_attivazione': data_attivazione_trovata, 'colpi_trascorsi': colpi_trascorsi, 'colpi_rimanenti': colpi_di_gioco - colpi_trascorsi, 'colpi_totali': colpi_di_gioco})
                else: # Giocata conclusa con esito negativo
                    previsioni_negative.append({'numero': numero_da_testare, 'data_attivazione': data_attivazione_trovata, 'colpi_gioco': colpi_di_gioco})

        self._log("Scansione completata.                                   ")

        # Ordina i risultati
        previsioni_in_attesa.sort(key=lambda x: x['colpi_rimanenti'])
        previsioni_sfaldate.sort(key=lambda x: x['dettagli_sfaldamento']['data'], reverse=True)

        # Stampa i risultati
        self.output_queue.put("\n" + "="*80 + "\n")
        
        if not previsioni_in_attesa and not previsioni_sfaldate and not previsioni_negative:
             self.output_queue.put("### NESSUNA ATTIVAZIONE TROVATA NEL PERIODO ANALIZZATO ###\n")
             return

        # 1. Previsioni in Attesa (le più importanti)
        if previsioni_in_attesa:
            self.output_queue.put(f"--- TROVATE {len(previsioni_in_attesa)} PREVISIONI ANCORA IN GIOCO ---\n")
            for i, prev in enumerate(previsioni_in_attesa):
                self.output_queue.put(f"{i+1}. Numero: {prev['numero']} (IN ATTESA)\n   - Attivata il: {prev['data_attivazione']}\n   - Colpi Trascorsi: {prev['colpi_trascorsi']}\n   - Colpi Rimanenti: {prev['colpi_rimanenti']} (su {prev['colpi_totali']})\n" + "-"*40 + "\n")
        else:
            self.output_queue.put("--- NESSUNA PREVISIONE ANCORA IN GIOCO ---\n\n")

        # 2. Previsioni Sfaldate (esiti positivi)
        if previsioni_sfaldate:
            self.output_queue.put(f"--- TROVATE {len(previsioni_sfaldate)} PREVISIONI SFALDATE (Esiti Positivi) ---\n")
            for i, prev in enumerate(previsioni_sfaldate):
                dettagli = prev['dettagli_sfaldamento']
                self.output_queue.put(f"{i+1}. Numero: {prev['numero']} (SFALDATO)\n   - Attivata il: {prev['data_attivazione']}\n   - Uscito il: {dettagli['data']} su {dettagli['ruota']} al colpo {dettagli['colpo']}\n" + "-"*40 + "\n")
        
        # 3. Previsioni Negative
        if previsioni_negative:
            self.output_queue.put(f"--- TROVATE {len(previsioni_negative)} PREVISIONI CONCLUSE (Esiti Negativi) ---\n")
            for i, prev in enumerate(previsioni_negative):
                self.output_queue.put(f"{i+1}. Numero: {prev['numero']} (NEGATIVO)\n   - Attivata il: {prev['data_attivazione']}\n   - Esito: Non uscito nei {prev['colpi_gioco']} colpi\n" + "-"*40 + "\n")

        self.output_queue.put("\nRicerca Completata.\n")

# --- CLASSE INTERFACCIA GRAFICA ---
class MarkerLottoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Marker Lotto - Ricerca Antecedenti - Created by Max Lotto -")
        self.geometry("900x850")
        self.output_queue = queue.Queue()
        self.archivio = ArchivioLotto(self.output_queue)
        self.analizzatore = AnalizzatoreMarker(self.archivio, self.output_queue)
        self.risultato_ultima_analisi = None
        self._create_widgets()
        self.after(100, self._process_queue)

    def _create_widgets(self):
        # 1. Contenitore principale per Canvas e Scrollbar
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True)

        # 2. Canvas per l'area scorrevole e Scrollbar
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        # 3. Metti la scrollbar a destra e il canvas a riempire lo spazio rimanente
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # 4. Crea il frame che conterrà TUTTI i widget e che verrà scrollato
        scrollable_frame = ttk.Frame(canvas)

        # 5. Aggiungi il frame al canvas
        scrollable_frame_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # 6. Logica di binding (collegamento) per far funzionare lo scroll
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(scrollable_frame_window, width=event.width)
        
        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        # 7. Aggiunta dello scroll con la rotellina del mouse
        def on_mouse_wheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                if event.num == 5: canvas.yview_scroll(1, "units")
                elif event.num == 4: canvas.yview_scroll(-1, "units")
        
        self.bind_all("<MouseWheel>", on_mouse_wheel)
        self.bind_all("<Button-4>", on_mouse_wheel)
        self.bind_all("<Button-5>", on_mouse_wheel)

        # --- WIDGETS DENTRO "scrollable_frame" ---
        
        archivio_frame = ttk.LabelFrame(scrollable_frame, text="Controllo Archivio Dati", padding=10)
        archivio_frame.pack(fill="x", expand=True, pady=5, padx=5)
        source_frame = ttk.Frame(archivio_frame)
        source_frame.pack(fill="x", pady=2)
        ttk.Label(source_frame, text="Fonte Dati:").pack(side="left", padx=5)
        self.data_source_var = tk.StringVar(value='GitHub')
        ttk.Radiobutton(source_frame, text="GitHub", variable=self.data_source_var, value='GitHub', command=self._toggle_local_path).pack(side="left")
        ttk.Radiobutton(source_frame, text="Locale", variable=self.data_source_var, value='Locale', command=self._toggle_local_path).pack(side="left", padx=5)
        self.local_path_label = ttk.Label(source_frame, text="N/A")
        self.local_path_label.pack(side="left", padx=5)
        self.browse_button = ttk.Button(source_frame, text="Sfoglia...", command=self._select_local_path)
        self.browse_button.pack(side="left")
        self.init_button = ttk.Button(archivio_frame, text="1. Inizializza Archivio", command=self._run_initialize_archive)
        self.init_button.pack(fill="x", pady=(5,0))

        analysis_frame = ttk.LabelFrame(scrollable_frame, text="Impostazioni Analisi e Strategia", padding=10)
        analysis_frame.pack(fill="x", expand=True, pady=5, padx=5)
        
        target_frame = ttk.LabelFrame(analysis_frame, text="Bersaglio e Ruote", padding=10)
        target_frame.pack(fill="x", expand=True, pady=5)
        
        ttk.Label(target_frame, text="Numero/Ambo Bersaglio (es: 90 o 21-54):").grid(row=0, column=0, sticky='w')
        self.bersaglio_entry = ttk.Entry(target_frame, width=15)
        self.bersaglio_entry.grid(row=1, column=0)
        # self.bersaglio_entry.insert(0, "90")
        ttk.Label(target_frame, text="su Ruota/e Bersaglio:").grid(row=0, column=1, sticky='w', padx=(10,0))
        self.ruote_bersaglio_frame = ttk.Frame(target_frame)
        self.ruote_bersaglio_frame.grid(row=1, column=1)
        
        self.ruote_bersaglio_vars = {}
        self.tutte_bersaglio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ruote_bersaglio_frame, text="Tutte", variable=self.tutte_bersaglio_var, command=self._toggle_all_bersaglio_ruote).pack(side="left")
        # FIX: Rimosso fill='y' dal separatore per non farlo diventare troppo alto
        ttk.Separator(self.ruote_bersaglio_frame, orient='vertical').pack(side='left', padx=5)
        for ruota_key in self.archivio.RUOTE_DISPONIBILI.keys():
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.ruote_bersaglio_frame, text=ruota_key, variable=var).pack(side="left")
            self.ruote_bersaglio_vars[ruota_key] = var
            
        params_frame = ttk.Frame(analysis_frame)
        params_frame.pack(fill="x", expand=True, pady=5)
        ttk.Label(params_frame, text="Finestra Antecedente (colpi):", font="-weight bold").pack(side="left")
        self.finestra_spinbox = ttk.Spinbox(params_frame, from_=1, to_=99, textvariable=tk.IntVar(value=9), width=5)
        self.finestra_spinbox.pack(side="left", padx=5)
        
        period_frame = ttk.Frame(analysis_frame)
        period_frame.pack(fill="x", expand=True, pady=5)
        ttk.Label(period_frame, text="Periodo Dal:").pack(side="left")
        self.start_date_entry = DateEntry(period_frame, width=10, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.start_date_entry.set_date(datetime(2024, 1, 1))
        self.start_date_entry.pack(side="left", padx=5)
        ttk.Label(period_frame, text="Al:").pack(side="left", padx=(10,0))
        self.end_date_entry = DateEntry(period_frame, width=10, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.end_date_entry.pack(side="left", padx=5)
        self.use_last_date_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(period_frame, text="Usa ultima data archivio", variable=self.use_last_date_var, command=self._toggle_date_entry).pack(side="left", padx=10)
        
        ruote_marker_frame = ttk.LabelFrame(analysis_frame, text="Cerca i Marker su queste Ruote", padding=10)
        ruote_marker_frame.pack(fill="x", expand=True, pady=10)
        
        self.ruote_marker_vars = {}
        cols = 12
        for i, ruota_key in enumerate(self.archivio.RUOTE_DISPONIBILI.keys()):
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(ruote_marker_frame, text=ruota_key, variable=var).grid(row=i // cols, column=i % cols, sticky='w')
            self.ruote_marker_vars[ruota_key] = var
        
        # FIX: Posizionamento corretto della casella "Tutte" per i marker
        self.tutte_marker_var = tk.BooleanVar(value=False)
        tutte_marker_cb = ttk.Checkbutton(ruote_marker_frame, text="Tutte", variable=self.tutte_marker_var, command=self._toggle_all_marker_ruote)
        num_ruote = len(self.archivio.RUOTE_DISPONIBILI)
        last_row_used = (num_ruote - 1) // cols
        tutte_marker_cb.grid(row=last_row_used + 1, column=0, columnspan=cols, sticky='w', pady=(10,0))

        # Pannello Azioni
        action_frame = ttk.LabelFrame(scrollable_frame, text="Azioni", padding=10)
        action_frame.pack(fill="x", expand=True, pady=5, padx=5)
        self.run_button = ttk.Button(action_frame, text="2. Avvia Analisi Marker", command=self._run_analysis)
        self.run_button.pack(fill="x", pady=2)
        backtest_params_frame = ttk.Frame(action_frame)
        backtest_params_frame.pack(fill="x", pady=5)
        ttk.Label(backtest_params_frame, text="Attiva con almeno:").grid(row=0, column=0, sticky='w')
        self.marker_attivazione_spinbox = ttk.Spinbox(backtest_params_frame, from_=1, to_=10, textvariable=tk.IntVar(value=2), width=5)
        self.marker_attivazione_spinbox.grid(row=0, column=1, sticky='w')
        ttk.Label(backtest_params_frame, text="Marker e gioca per:").grid(row=0, column=2, sticky='w', padx=(10,0))
        self.colpi_gioco_spinbox = ttk.Spinbox(backtest_params_frame, from_=1, to_=99, textvariable=tk.IntVar(value=18), width=5)
        self.colpi_gioco_spinbox.grid(row=0, column=3, sticky='w')
        ttk.Label(backtest_params_frame, text="colpi.").grid(row=0, column=4, sticky='w')
        self.run_backtest_button = ttk.Button(action_frame, text="3. Avvia Backtest della Strategia", command=self._run_backtest)
        self.run_backtest_button.pack(fill="x", pady=2)
        self.run_live_check_button = ttk.Button(action_frame, text="4. Controlla Segnali Attivi", command=self._run_live_check)
        self.run_live_check_button.pack(fill="x", pady=2)
        self.run_find_pending_button = ttk.Button(action_frame, text="5. Ricerca Previsioni in Attesa (solo Ambate)", command=self._run_find_pending)
        self.run_find_pending_button.pack(fill="x", pady=2)
        ttk.Button(action_frame, text="Pulisci Output", command=self._clear_output).pack(fill="x", pady=(10, 2))
        
        # Pannello Output
        self.output_text = scrolledtext.ScrolledText(scrollable_frame, wrap=tk.WORD, state='disabled', height=25, font=('Consolas', 10))
        self.output_text.pack(fill="x", pady=5, padx=5)

        # Chiamate iniziali
        self._toggle_local_path()
        self._toggle_date_entry()

    # --- NUOVE FUNZIONI DI SUPPORTO PER LE CASELLE "TUTTE" ---
    def _toggle_all_bersaglio_ruote(self):
        is_checked = self.tutte_bersaglio_var.get()
        for var in self.ruote_bersaglio_vars.values():
            var.set(is_checked)
    
    def _toggle_all_marker_ruote(self):
        is_checked = self.tutte_marker_var.get()
        for var in self.ruote_marker_vars.values():
            var.set(is_checked)

    def _toggle_date_entry(self):
        self.end_date_entry.config(state='disabled' if self.use_last_date_var.get() else 'normal')

    def _get_common_params(self, require_bersaglio=True):
        params = {}
        if require_bersaglio:
            try:
                numeri_bersaglio = [int(n) for n in self.bersaglio_entry.get().replace(',', '-').split('-') if n.strip().isdigit()]
                if not (1 <= len(numeri_bersaglio) <= 2 and all(1 <= num <= 90 for num in numeri_bersaglio)): raise ValueError
                params['bersaglio'] = tuple(sorted(numeri_bersaglio))
            except ValueError:
                messagebox.showwarning("Input Errato", "Inserire 1 o 2 numeri bersaglio validi (1-90)."); return None
        
        ruote_bersaglio_keys = [k for k, v in self.ruote_bersaglio_vars.items() if v.get()]
        if not ruote_bersaglio_keys:
            messagebox.showwarning("Input Errato", "Seleziona almeno una ruota bersaglio."); return None
            
        ruote_marker_keys = [k for k, v in self.ruote_marker_vars.items() if v.get()]
        if not ruote_marker_keys:
            messagebox.showwarning("Input Errato", "Seleziona almeno una ruota per i marker."); return None
        
        if not self.archivio.date_ordinate:
            messagebox.showerror("Errore", "Archivio non inizializzato."); return None
        
        end_date = self.archivio.date_ordinate[-1] if self.use_last_date_var.get() else self.end_date_entry.get_date().strftime('%Y-%m-%d')
        
        params.update({
            'ruote_bersaglio_keys': ruote_bersaglio_keys,
            'start_date_str': self.start_date_entry.get_date().strftime('%Y-%m-%d'),
            'end_date_str': end_date,
            'finestra_antecedente': int(self.finestra_spinbox.get()),
            'ruote_marker_keys': ruote_marker_keys,
        })
        return params

    def _run_task(self, task_function, params):
        if params is None: return
        self._clear_output()
        threading.Thread(target=task_function, args=(params,), daemon=True).start()

    def _run_initialize_archive(self):
        self._clear_output()
        threading.Thread(target=self._initialize_archive_task, daemon=True).start()

    def _run_analysis(self):
        if not self.archivio.dati_per_analisi: messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        params = self._get_common_params()
        if params:
            def task_wrapper(p):
                self.analizzatore.trova_marker_per_bersaglio(p)
                self.risultato_ultima_analisi = p.get('risultato_analisi')
            self._run_task(task_wrapper, params)

    def _run_backtest(self):
        if not self.archivio.dati_per_analisi: messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        params = self._get_common_params()
        if params:
            params['soglia_attivazione'] = int(self.marker_attivazione_spinbox.get())
            params['colpi_di_gioco'] = int(self.colpi_gioco_spinbox.get())
            self._run_task(self.analizzatore.esegui_backtest, params)

    def _run_find_pending(self):
        if not self.archivio.dati_per_analisi: messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        params = self._get_common_params(require_bersaglio=False)
        if params:
            params['soglia_attivazione'] = int(self.marker_attivazione_spinbox.get())
            params['colpi_di_gioco'] = int(self.colpi_gioco_spinbox.get())
            self._run_task(self.analizzatore.trova_previsioni_in_attesa, params)

    def _run_live_check(self):
        if not self.archivio.dati_per_analisi:
            messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio.")
            return
        
        # RIMOSSO IL CONTROLLO 'if self.risultato_ultima_analisi is None:'
        # perché la funzione ora è autonoma.

        params = self._get_common_params()
        if params:
            # NON passiamo più 'risultato_analisi', ma passiamo la soglia direttamente
            params['soglia_attivazione'] = int(self.marker_attivazione_spinbox.get())
            self._run_task(self.analizzatore.controlla_segnali_attivi, params)

    def _select_local_path(self):
        folder = filedialog.askdirectory()
        if folder: 
            self.archivio.local_path = folder
            self.local_path_label.config(text=os.path.basename(folder))

    def _toggle_local_path(self):
        state = 'normal' if self.data_source_var.get() == 'Locale' else 'disabled'
        self.local_path_label.config(state=state)
        self.browse_button.config(state=state)
        self.archivio.data_source = self.data_source_var.get()

    def _initialize_archive_task(self):
        try:
            self.archivio.inizializza()
            last_date_obj = datetime.strptime(self.archivio.date_ordinate[-1], '%Y-%m-%d')
            self.end_date_entry.set_date(last_date_obj)
        except Exception as e:
            self.output_queue.put(f"ERRORE INIZIALIZZAZIONE: {e}\n")
            traceback.print_exc()

    def _clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')

    def _process_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait()
                self.output_text.config(state='normal')
                if '\r' in msg:
                    current_content = self.output_text.get("1.0", tk.END)
                    last_line_start_index = current_content.rfind('\n', 0, -2)
                    if last_line_start_index == -1:
                        last_line_start = "1.0"
                    else:
                        last_line_start = f"1.0 + {last_line_start_index+1}c"
                    
                    self.output_text.delete(last_line_start, tk.END)
                    self.output_text.insert(tk.END, msg.strip('\r'))
                else:
                    self.output_text.insert(tk.END, msg)
                self.output_text.see(tk.END)
                self.output_text.config(state='disabled')
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_queue)

if __name__ == "__main__":
    app = MarkerLottoApp()
    app.mainloop()