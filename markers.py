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

    def _verifica_attivazione(self, marker_data, numeri_recenti, use_score, soglia_standard, soglia_punteggio):
        # MODALITÀ STANDARD (CONTEGGIO)
        if not use_score:
            # marker_data è il counter, ne estraiamo i numeri ordinati per frequenza
            top_marker_list = [num for num, freq in marker_data.most_common(soglia_standard)]
            marker_attivati = set(top_marker_list).intersection(numeri_recenti)
            
            is_active = len(marker_attivati) >= soglia_standard
            dettaglio = f"Trovati marker: {sorted(list(marker_attivati))}" if is_active else f"Trovati solo {len(marker_attivati)}/{soglia_standard} marker necessari."
            return is_active, dettaglio
            
        # MODALITÀ PUNTEGGIO BASATA SULLA FREQUENZA
        marker_counter = marker_data # Rinominiamo per chiarezza
        
        if not marker_counter:
            return False, "Nessun marker storico trovato per l'analisi a punteggio."

        # Trova i 3 livelli di frequenza più alti
        frequenze_uniche = sorted(list(set(marker_counter.values())), reverse=True)
        
        punti_per_frequenza = {}
        if len(frequenze_uniche) >= 1: punti_per_frequenza[frequenze_uniche[0]] = 3
        if len(frequenze_uniche) >= 2: punti_per_frequenza[frequenze_uniche[1]] = 2
        if len(frequenze_uniche) >= 3: punti_per_frequenza[frequenze_uniche[2]] = 1

        # Calcola il punteggio
        punteggio_attuale = 0
        marker_a_punti_list = []
        
        marker_attivati = set(marker_counter.keys()).intersection(numeri_recenti)
        
        for m in sorted(list(marker_attivati)):
            freq = marker_counter.get(m, 0)
            punti = punti_per_frequenza.get(freq, 0)
            if punti > 0:
                punteggio_attuale += punti
                marker_a_punti_list.append(f"{m}({punti}pt)")

        is_active = punteggio_attuale >= soglia_punteggio
        dettaglio = f"Punteggio {punteggio_attuale}/{soglia_punteggio}. Marker attivati: {', '.join(marker_a_punti_list)}"
        
        return is_active, dettaglio

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
        for numero, freq in marker_numeri_counter.most_common(10):
            self.output_queue.put(f"  - Numero {numero:<2}: {freq:>4} volte ({(freq / num_eventi_trovati * 100):6.2f}%)\n")
        
        # NUOVO: Aggiungiamo la stampa della classifica degli ambi
        self.output_queue.put("\n>>> CLASSIFICA MARKER (AMBI) <<<\n")
        if not marker_ambi_counter:
            self.output_queue.put("  (Nessun ambo trovato con i parametri attuali)\n")
        else:
            for ambo, freq in marker_ambi_counter.most_common(10):
                ambo_str = f"{ambo[0]}-{ambo[1]}"
                self.output_queue.put(f"  - Ambo {ambo_str:<5}: {freq:>4} volte ({(freq / num_eventi_trovati * 100):6.2f}%)\n")

        # Memorizziamo solo i risultati dei numeri singoli per la logica successiva
        params['risultato_analisi'] = marker_numeri_counter

    def esegui_backtest(self, params):
        bersaglio, ruote_bersaglio_keys, start_date_str, end_date_str, finestra_antecedente, ruote_marker_keys, colpi_di_gioco = \
            params['bersaglio'], params['ruote_bersaglio_keys'], params['start_date_str'], params['end_date_str'], params['finestra_antecedente'], params['ruote_marker_keys'], params['colpi_di_gioco']
        soglia_attivazione = params.get('soglia_attivazione', 2)
        use_score_system, target_score = params.get('use_score_system', False), params.get('target_score', 4)
        
        bersaglio_set = set(bersaglio)
        self._log(f"\n{'='*80}\n### BACKTEST STRATEGIA REALISTICA (CON ROI) ###\nPeriodo: da {start_date_str} a {end_date_str}\n{'='*80}")
        self._log(f"Strategia: Quando la condizione è attiva, giocare {'-'.join(map(str, bersaglio))} per max {colpi_di_gioco} colpi. La giocata si ferma alla vincita.")

        try:
            date_ordinate, date_to_index = self.archivio.date_ordinate, self.archivio.date_to_index
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            date_nel_range_indices = [i for i, d in enumerate(date_ordinate) if start_dt <= datetime.strptime(d, '%Y-%m-%d') <= end_dt]
            if not date_nel_range_indices:
                self._log("Range di date non valido.")
                return
            indice_inizio_ricerca = date_nel_range_indices[0]
        except (ValueError, IndexError): 
            self._log("Formato data non valido o range vuoto.")
            return
        
        previsioni_giocate, vincite_totali = 0, 0
        spesa_totale, incasso_totale = 0.0, 0.0
        indice_prossima_ricerca = 0

        for i in date_nel_range_indices:
            if i < indice_prossima_ricerca:
                continue

            # --- Logica di attivazione ---
            indice_logica = i - 1
            if indice_logica < indice_inizio_ricerca: continue

            eventi_passati = [j for j in range(indice_logica, indice_inizio_ricerca - 1, -1) 
                              if any(bersaglio_set.issubset(set(self.archivio.dati_per_analisi.get(date_ordinate[j], {}).get(k, []))) for k in ruote_bersaglio_keys)]
            if len(eventi_passati) < 10: continue 

            marker_counter_storico = Counter()
            for indice_evento_passato in eventi_passati:
                numeri_finestra = set()
                for k in range(1, finestra_antecedente + 1):
                    if (idx_ante := indice_evento_passato - k) < indice_inizio_ricerca: break
                    estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante], {})
                    for ruota_m_key in ruote_marker_keys:
                        if (numeri := estrazione_ante.get(ruota_m_key)): numeri_finestra.update(numeri)
                for num in numeri_finestra: marker_counter_storico[num] += 1
            
            numeri_usciti_recentemente = set()
            # ### CORREZIONE QUI ###
            # La finestra dei marker recenti per un'attivazione il giorno 'i'
            # deve andare da 'i-1' indietro.
            for k in range(1, finestra_antecedente + 1):
                if (idx_ante_corrente := i - k) < 0: break # Usa 'i - k' invece di 'i - 1 - k'
                estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante_corrente], {})
                for ruota_m_key in ruote_marker_keys:
                    if (numeri := estrazione_ante.get(ruota_m_key)): numeri_usciti_recentemente.update(numeri)
            
            is_active, dettaglio = self._verifica_attivazione(marker_counter_storico, numeri_usciti_recentemente, use_score_system, soglia_attivazione, target_score)
            # --- Fine logica attivazione ---

            if is_active:
                previsioni_giocate += 1
                self._log(f"\n[{date_ordinate[i]}] ATTIVAZIONE n.{previsioni_giocate}. {dettaglio}")
                
                vincita_trovata = False
                costo_previsione_corrente = 0.0
                archivio_terminato = False

                for colpo in range(1, colpi_di_gioco + 1):
                    spesa_totale += 1.0
                    costo_previsione_corrente += 1.0
                    
                    indice_esito = i + colpo
                    if indice_esito >= len(date_ordinate): 
                        self._log(f"  Colpo {colpo}: Fine archivio.");
                        archivio_terminato = True
                        indice_prossima_ricerca = indice_esito
                        break
                    
                    data_esito = date_ordinate[indice_esito]
                    estrazione_vincita = self.archivio.dati_per_analisi.get(data_esito, {})
                    for ruota_b in ruote_bersaglio_keys:
                        if bersaglio_set.issubset(set(estrazione_vincita.get(ruota_b, []))):
                            self._log(f"  VINCITA al colpo {colpo} ({data_esito}) su {ruota_b}!"); 
                            vincite_totali += 1
                            vincita_trovata = True
                            incasso_corrente = 10.33 
                            incasso_totale += incasso_corrente
                            self._log(f"    Spesa per questa previsione: {costo_previsione_corrente:.2f}€, Incasso: {incasso_corrente:.2f}€")
                            indice_prossima_ricerca = indice_esito + 1 
                            break
                    if vincita_trovata: break
                
                if not vincita_trovata:
                    if archivio_terminato:
                        self._log(f"  ESITO IN CORSO (Archivio terminato).")
                        self._log(f"    Spesa finora per questa previsione: {costo_previsione_corrente:.2f}€")
                    else:
                        self._log(f"  ESITO NEGATIVO dopo {colpi_di_gioco} colpi.")
                        self._log(f"    Spesa per questa previsione: {costo_previsione_corrente:.2f}€")
                        indice_prossima_ricerca = i + colpi_di_gioco + 1

        percentuale_successo = (vincite_totali / previsioni_giocate * 100) if previsioni_giocate > 0 else 0
        utile_netto = incasso_totale - spesa_totale
        self._log(f"\n{'='*80}\n### REPORT FINALE BACKTEST REALISTICO ###\n")
        self._log(f"Previsioni Giocate: {previsioni_giocate}")
        self._log(f"Vincite: {vincite_totali}")
        self._log(f"Percentuale di Successo: {percentuale_successo:.2f}%")
        self._log("-" * 40)
        self._log(f"Spesa Totale: {spesa_totale:.2f}€")
        self._log(f"Incasso Totale: {incasso_totale:.2f}€")
        self._log(f"Utile/Perdita (ROI): {utile_netto:.2f}€")
        self._log(f"\n{'='*80}")

    def controlla_segnali_attivi(self, params):
        bersaglio, ruote_bersaglio_keys, start_date_str, end_date_str, finestra_antecedente, ruote_marker_keys = \
            params['bersaglio'], params['ruote_bersaglio_keys'], params['start_date_str'], params['end_date_str'], params['finestra_antecedente'], params['ruote_marker_keys']
        soglia_attivazione = params.get('soglia_attivazione', 2)
        use_score_system, target_score = params.get('use_score_system', False), params.get('target_score', 4)
        bersaglio_set = set(bersaglio)

        try:
            date_ordinate, date_to_index = self.archivio.date_ordinate, self.archivio.date_to_index
            valid_end_dates = [d for d in date_ordinate if d <= end_date_str]; data_riferimento_effettiva = valid_end_dates[-1]
            indice_riferimento = date_to_index[data_riferimento_effettiva]
            valid_start_dates = [d for d in date_ordinate if d >= start_date_str]; data_inizio_effettiva = valid_start_dates[0]
            indice_inizio_ricerca = date_to_index[data_inizio_effettiva]
        except (IndexError, KeyError):
            self._log("Date non valide o range vuoto."); return

        self._log(f"\n{'='*80}\n### VERDETTO PER BERSAGLIO {'-'.join(map(str, bersaglio))} ALLA DATA {data_riferimento_effettiva} ###\n{'='*80}")
        
        # 'indice_riferimento' è il giorno in cui si decide se giocare.
        # La logica si basa sui dati fino al giorno prima.
        indice_logica = indice_riferimento - 1 

        if indice_logica < indice_inizio_ricerca:
            self._log("Data di riferimento troppo vicina all'inizio del periodo per l'analisi.")
            return

        eventi_passati = [j for j in range(indice_logica, indice_inizio_ricerca - 1, -1) if any(bersaglio_set.issubset(set(self.archivio.dati_per_analisi.get(date_ordinate[j], {}).get(k, []))) for k in ruote_bersaglio_keys)]
        
        if len(eventi_passati) < 10:
            self._log("Storico insufficiente (<10 casi) per definire i marker."); return

        marker_counter_storico = Counter()
        for indice_evento in eventi_passati:
            numeri_finestra = set()
            for k in range(1, finestra_antecedente + 1):
                if (idx_ante := indice_evento - k) < indice_inizio_ricerca: break
                estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante], {})
                for ruota_m_key in ruote_marker_keys:
                    if (numeri := estrazione_ante.get(ruota_m_key)): numeri_finestra.update(numeri)
            for num in numeri_finestra: marker_counter_storico[num] += 1

        numeri_usciti_recentemente = set()
        # ### CORREZIONE QUI ###
        # La finestra dei marker recenti per la decisione del giorno 'indice_riferimento'
        # deve andare da 'indice_riferimento - 1' indietro.
        for k in range(1, finestra_antecedente + 1):
            if (idx_ante_corrente := indice_riferimento - k) < 0: break # Usa 'indice_riferimento - k'
            estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante_corrente], {})
            for ruota_m_key in ruote_marker_keys:
                if (numeri := estrazione_ante.get(ruota_m_key)): numeri_usciti_recentemente.update(numeri)
        
        is_active, dettaglio = self._verifica_attivazione(marker_counter_storico, numeri_usciti_recentemente, use_score_system, soglia_attivazione, target_score)
        
        if is_active:
            self._log(f">>> CONDIZIONE ATTIVA <<<")
            self._log(f"Motivo: {dettaglio}")
            self._log(f"Previsione Consigliata: Giocare {'-'.join(map(str, bersaglio))} su {', '.join(ruote_bersaglio_keys)}")
        else:
            self._log(f">>> CONDIZIONE NON ATTIVA <<<")
            self._log(f"Motivo: {dettaglio}")
        self._log(f"{'='*80}")


    def trova_previsioni_in_attesa(self, params):
        start_date_str, end_date_str, ruote_bersaglio_keys, finestra_antecedente, ruote_marker_keys, colpi_di_gioco = \
            params['start_date_str'], params['end_date_str'], params['ruote_bersaglio_keys'], params['finestra_antecedente'], params['ruote_marker_keys'], params['colpi_di_gioco']
        soglia_attivazione = params.get('soglia_attivazione', 2)
        use_score_system, target_score = params.get('use_score_system', False), params.get('target_score', 4)

        bersaglio_da_ui = params.get('bersaglio')

        self._log(f"\n{'='*80}\n### RICERCA PREVISIONI (dal {start_date_str} al {end_date_str}) ###\n{'='*80}")
        try:
            date_ordinate = self.archivio.date_ordinate
            valid_start_dates = [d for d in date_ordinate if d >= start_date_str]; indice_inizio_ricerca = self.archivio.date_to_index[valid_start_dates[0]]
            valid_end_dates = [d for d in date_ordinate if d <= end_date_str]; indice_riferimento_ricerca = self.archivio.date_to_index[valid_end_dates[-1]]
            indice_ultimo_dato_archivio = len(date_ordinate) - 1
        except (ValueError, IndexError): self._log("Date di inizio o fine non valide."); return

        lista_bersagli_da_analizzare = []
        if bersaglio_da_ui:
            lista_bersagli_da_analizzare.append(bersaglio_da_ui)
            self._log(f"Analizzando il bersaglio specifico: {'-'.join(map(str, bersaglio_da_ui))}...")
        else:
            lista_bersagli_da_analizzare = [(n,) for n in range(1, 91)]
            self._log(f"Nessun bersaglio specifico, avvio scansione su tutti i 90 numeri...")

        ultime_attivazioni_valide = {}
        indice_prossima_ricerca_per_bersaglio = {}

        for i in range(indice_inizio_ricerca, indice_riferimento_ricerca + 1):
            if not bersaglio_da_ui and i % 10 == 0:
                self._log(f"Scansione in corso... Data: {date_ordinate[i]}", end='\r')

            for bersaglio in lista_bersagli_da_analizzare:
                if i < indice_prossima_ricerca_per_bersaglio.get(bersaglio, 0):
                    continue
                
                bersaglio_set = set(bersaglio)
                indice_logica = i - 1
                if indice_logica < indice_inizio_ricerca: continue

                eventi_passati = [j for j in range(indice_logica, indice_inizio_ricerca - 1, -1) if any(bersaglio_set.issubset(set(self.archivio.dati_per_analisi.get(date_ordinate[j], {}).get(k, []))) for k in ruote_bersaglio_keys)]
                if len(eventi_passati) < 10: continue

                marker_counter_storico = Counter()
                for indice_evento in eventi_passati:
                    numeri_finestra = set()
                    for k in range(1, finestra_antecedente + 1):
                        if (idx_ante := indice_evento - k) < indice_inizio_ricerca: break
                        estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante], {})
                        for ruota_m_key in ruote_marker_keys:
                            if (numeri := estrazione_ante.get(ruota_m_key)): numeri_finestra.update(numeri)
                    for num_m in numeri_finestra: marker_counter_storico[num_m] += 1
                
                numeri_usciti_recentemente = set()
                for k in range(1, finestra_antecedente + 1):
                    if (idx_ante_corrente := i - k) < 0: break
                    estrazione_ante = self.archivio.dati_per_analisi.get(date_ordinate[idx_ante_corrente], {})
                    for ruota_m_key in ruote_marker_keys:
                        if (numeri_m := estrazione_ante.get(ruota_m_key)): numeri_usciti_recentemente.update(numeri_m)
                
                is_active, _ = self._verifica_attivazione(marker_counter_storico, numeri_usciti_recentemente, use_score_system, soglia_attivazione, target_score)

                if is_active:
                    ultime_attivazioni_valide[bersaglio] = {'indice_attivazione': i, 'data_attivazione': date_ordinate[i]}
                    # ### LA CORREZIONE DECISIVA È QUI ###
                    # Il cooldown deve essere identico a quello del backtest.
                    # Dopo una giocata di N colpi, si riparte dal N+1-esimo giorno.
                    indice_prossima_ricerca_per_bersaglio[bersaglio] = i + colpi_di_gioco + 1

        self._log("Scansione completata. Classifico i risultati...                      ", end='\r')
        self._log("")
        previsioni_in_attesa, previsioni_sfaldate, previsioni_negative = [], [], []

        for bersaglio, dati_attivazione in ultime_attivazioni_valide.items():
            bersaglio_set = set(bersaglio)
            indice_attivazione = dati_attivazione['indice_attivazione']
            data_attivazione = dati_attivazione['data_attivazione']
            colpi_trascorsi = indice_ultimo_dato_archivio - indice_attivazione
            
            esito_trovato, dettagli_sfaldamento = False, {}
            for colpo in range(1, colpi_di_gioco + 1):
                if colpo > colpi_trascorsi: break
                indice_esito = indice_attivazione + colpo
                if indice_esito >= len(date_ordinate): break
                
                estrazione_vincita = self.archivio.dati_per_analisi.get(date_ordinate[indice_esito], {})
                for ruota_b_key in ruote_bersaglio_keys:
                    if bersaglio_set.issubset(set(estrazione_vincita.get(ruota_b_key, []))):
                        esito_trovato, dettagli_sfaldamento = True, {'colpo': colpo, 'data': date_ordinate[indice_esito], 'ruota': ruota_b_key}; break
                if esito_trovato: break
            
            dati_previsione = {'bersaglio': bersaglio, 'data_attivazione': data_attivazione}
            if esito_trovato:
                dati_previsione['dettagli_sfaldamento'] = dettagli_sfaldamento
                previsioni_sfaldate.append(dati_previsione)
            elif colpi_trascorsi < colpi_di_gioco:
                dati_previsione.update({'colpi_trascorsi': colpi_trascorsi, 'colpi_rimanenti': colpi_di_gioco - colpi_trascorsi, 'colpi_totali': colpi_di_gioco})
                previsioni_in_attesa.append(dati_previsione)
            else:
                dati_previsione['colpi_gioco'] = colpi_di_gioco
                previsioni_negative.append(dati_previsione)

        previsioni_in_attesa.sort(key=lambda x: x['colpi_rimanenti'])
        previsioni_sfaldate.sort(key=lambda x: x['dettagli_sfaldamento']['data'], reverse=True)
        previsioni_negative.sort(key=lambda x: x['data_attivazione'], reverse=True)
        self.output_queue.put("\n" + "="*80 + "\n")
        
        if not previsioni_in_attesa and not previsioni_sfaldate and not previsioni_negative: 
            self.output_queue.put("### NESSUNA ATTIVAZIONE VALIDA TROVATA NEL PERIODO ANALIZZATO ###\n")
            return
            
        if previsioni_in_attesa:
            self.output_queue.put(f"--- TROVATE {len(previsioni_in_attesa)} PREVISIONI ANCORA IN GIOCO ---\n")
            for i, p in enumerate(previsioni_in_attesa): 
                bersaglio_str = '-'.join(map(str, p['bersaglio']))
                label = "Bersaglio" if len(p['bersaglio']) > 1 else "Numero"
                self.output_queue.put(f"{i+1}. {label}: {bersaglio_str} (IN ATTESA)\n   - Attivata il: {p['data_attivazione']}\n   - Colpi Trascorsi: {p['colpi_trascorsi']}\n   - Colpi Rimanenti: {p['colpi_rimanenti']} (su {p['colpi_totali']})\n" + "-"*40 + "\n")
        
        if previsioni_sfaldate:
            self.output_queue.put(f"--- TROVATE {len(previsioni_sfaldate)} PREVISIONI SFALDATE (Esiti Positivi) ---\n")
            for i, p in enumerate(previsioni_sfaldate):
                bersaglio_str = '-'.join(map(str, p['bersaglio']))
                label = "Bersaglio" if len(p['bersaglio']) > 1 else "Numero"
                self.output_queue.put(f"{i+1}. {label}: {bersaglio_str} (SFALDATO)\n   - Attivata il: {p['data_attivazione']}\n   - Uscito il: {p['dettagli_sfaldamento']['data']} su {p['dettagli_sfaldamento']['ruota']} al colpo {p['dettagli_sfaldamento']['colpo']}\n" + "-"*40 + "\n")
        
        if previsioni_negative:
            self.output_queue.put(f"--- TROVATE {len(previsioni_negative)} PREVISIONI CONCLUSE (Esiti Negativi) ---\n")
            for i, p in enumerate(previsioni_negative): 
                bersaglio_str = '-'.join(map(str, p['bersaglio']))
                label = "Bersaglio" if len(p['bersaglio']) > 1 else "Numero"
                self.output_queue.put(f"{i+1}. {label}: {bersaglio_str} (NEGATIVO)\n   - Attivata il: {p['data_attivazione']}\n   - Esito: Non uscito nei {p['colpi_gioco']} colpi\n" + "-"*40 + "\n")

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
        # 1. Contenitore principale e logica di scrolling (INVARIATO)
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True)
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(scrollable_frame_window, width=event.width)
        
        scrollable_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        
        def on_mouse_wheel(event):
            if event.delta: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                if event.num == 5: canvas.yview_scroll(1, "units")
                elif event.num == 4: canvas.yview_scroll(-1, "units")
        
        self.bind_all("<MouseWheel>", on_mouse_wheel)
        self.bind_all("<Button-4>", on_mouse_wheel)
        self.bind_all("<Button-5>", on_mouse_wheel)

        # --- NUOVA STRUTTURA A 3 SEZIONI ---
        # Tutti i widget verranno inseriti in scrollable_frame, ma organizzati in 3 macro-aree.

        # --- SEZIONE 1: IMPOSTAZIONI (Tutto raggruppato qui) ---
        settings_container = ttk.LabelFrame(scrollable_frame, text="1. Impostazioni", padding=10)
        settings_container.pack(fill="x", expand=True, pady=5, padx=5)

        # 1.1 - Controllo Archivio Dati
        archivio_frame = ttk.LabelFrame(settings_container, text="Fonte Dati e Inizializzazione", padding=10)
        archivio_frame.pack(fill="x", expand=True, pady=5)
        
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
        
        # Il pulsante di inizializzazione è stato spostato qui, vicino alla sua funzione logica
        self.init_button = ttk.Button(archivio_frame, text="Inizializza Archivio", command=self._run_initialize_archive)
        self.init_button.pack(fill="x", pady=(10,0))

        # 1.2 - Parametri di Analisi
        analysis_params_frame = ttk.LabelFrame(settings_container, text="Parametri di Analisi", padding=10)
        analysis_params_frame.pack(fill="x", expand=True, pady=5)

        target_frame = ttk.Frame(analysis_params_frame)
        target_frame.pack(fill="x", expand=True, pady=5)
        ttk.Label(target_frame, text="Numero/Ambo Bersaglio (es: 90 o 21-54):").grid(row=0, column=0, sticky='w')
        self.bersaglio_entry = ttk.Entry(target_frame, width=15)
        self.bersaglio_entry.grid(row=1, column=0, sticky='w')
        ttk.Label(target_frame, text="su Ruota/e Bersaglio:").grid(row=0, column=1, sticky='w', padx=(10,0))
        self.ruote_bersaglio_frame = ttk.Frame(target_frame)
        self.ruote_bersaglio_frame.grid(row=1, column=1, sticky='w')
        self.ruote_bersaglio_vars = {}
        self.tutte_bersaglio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ruote_bersaglio_frame, text="Tutte", variable=self.tutte_bersaglio_var, command=self._toggle_all_bersaglio_ruote).pack(side="left")
        ttk.Separator(self.ruote_bersaglio_frame, orient='vertical').pack(side='left', padx=5, fill='y')
        for ruota_key in self.archivio.RUOTE_DISPONIBILI.keys():
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.ruote_bersaglio_frame, text=ruota_key, variable=var).pack(side="left")
            self.ruote_bersaglio_vars[ruota_key] = var
            
        period_frame = ttk.Frame(analysis_params_frame)
        period_frame.pack(fill="x", expand=True, pady=5)
        ttk.Label(period_frame, text="Periodo Dal:").pack(side="left")
        self.start_date_entry = DateEntry(period_frame, width=10, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.start_date_entry.set_date(datetime(2018, 1, 1))
        self.start_date_entry.pack(side="left", padx=5)
        ttk.Label(period_frame, text="Al:").pack(side="left", padx=(10,0))
        self.end_date_entry = DateEntry(period_frame, width=10, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.end_date_entry.pack(side="left", padx=5)
        self.use_last_date_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(period_frame, text="Usa ultima data archivio", variable=self.use_last_date_var, command=self._toggle_date_entry).pack(side="left", padx=10)

        ruote_marker_frame = ttk.LabelFrame(analysis_params_frame, text="Cerca i Marker su queste Ruote", padding=10)
        ruote_marker_frame.pack(fill="x", expand=True, pady=10)
        self.ruote_marker_vars = {}
        cols = 12
        for i, ruota_key in enumerate(self.archivio.RUOTE_DISPONIBILI.keys()):
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(ruote_marker_frame, text=ruota_key, variable=var).grid(row=i // cols, column=i % cols, sticky='w')
            self.ruote_marker_vars[ruota_key] = var
        self.tutte_marker_var = tk.BooleanVar(value=False)
        tutte_marker_cb = ttk.Checkbutton(ruote_marker_frame, text="Tutte", variable=self.tutte_marker_var, command=self._toggle_all_marker_ruote)
        num_ruote = len(self.archivio.RUOTE_DISPONIBILI)
        last_row_used = (num_ruote - 1) // cols
        tutte_marker_cb.grid(row=last_row_used + 1, column=0, columnspan=cols, sticky='w', pady=(10,0))

        # 1.3 - Impostazioni Strategia (SPOSTATE QUI)
        strategy_params_frame = ttk.LabelFrame(settings_container, text="Impostazioni Strategia", padding=10)
        strategy_params_frame.pack(fill="x", pady=5)
        
        # Parametri comuni finestra e colpi
        common_strategy_frame = ttk.Frame(strategy_params_frame)
        common_strategy_frame.pack(fill='x', expand=True, pady=2)
        ttk.Label(common_strategy_frame, text="Finestra Antecedente (colpi):").pack(side="left")
        self.finestra_spinbox = ttk.Spinbox(common_strategy_frame, from_=1, to_=99, textvariable=tk.IntVar(value=9), width=5)
        self.finestra_spinbox.pack(side="left", padx=5)
        ttk.Label(common_strategy_frame, text="Gioca la previsione per:").pack(side="left", padx=(15,0))
        self.colpi_gioco_spinbox = ttk.Spinbox(common_strategy_frame, from_=1, to_=99, textvariable=tk.IntVar(value=18), width=5)
        self.colpi_gioco_spinbox.pack(side="left", padx=5)
        ttk.Label(common_strategy_frame, text="colpi.").pack(side="left")

        ttk.Separator(strategy_params_frame, orient='horizontal').pack(fill='x', pady=8)
        
        # Logica di attivazione
        activation_frame = ttk.Frame(strategy_params_frame)
        activation_frame.pack(fill='x', expand=True, pady=2)
        ttk.Label(activation_frame, text="Attiva con almeno:").grid(row=0, column=0, sticky='w', pady=2)
        self.marker_attivazione_spinbox = ttk.Spinbox(activation_frame, from_=1, to_=15, textvariable=tk.IntVar(value=2), width=5)
        self.marker_attivazione_spinbox.grid(row=0, column=1, sticky='w', pady=2, padx=5)
        ttk.Label(activation_frame, text="marker presenti.").grid(row=0, column=2, sticky='w', pady=2)
        
        self.use_score_system_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(activation_frame, text="Usa Sistema a Punteggio >=", variable=self.use_score_system_var).grid(row=1, column=0, sticky='w', pady=2)
        self.target_score_spinbox = ttk.Spinbox(activation_frame, from_=1, to_=20, textvariable=tk.IntVar(value=4), width=5)
        self.target_score_spinbox.grid(row=1, column=1, sticky='w', pady=2, padx=5)
        ttk.Label(activation_frame, text="(T1=3, T2=2, T3-5=1 pt)").grid(row=1, column=2, sticky='w', pady=2, padx=5)
        
        
        # --- SEZIONE 2: PANNELLO DI CONTROLLO (Pulsanti Raggruppati) ---
        control_panel = ttk.LabelFrame(scrollable_frame, text="2. Pannello di Controllo", padding=10)
        control_panel.pack(fill="x", expand=True, pady=5, padx=5)
        
        control_panel.grid_columnconfigure((0, 1), weight=1)

        style = ttk.Style()
        style.configure('Big.TButton', font=('', 9), padding=6)
        
        self.run_button = ttk.Button(control_panel, text="Avvia Analisi Marker", command=self._run_analysis, style='Big.TButton')
        self.run_button.grid(row=0, column=0, sticky='ew', padx=5, pady=3)
        
        self.run_backtest_button = ttk.Button(control_panel, text="Avvia Backtest della Strategia", command=self._run_backtest, style='Big.TButton')
        self.run_backtest_button.grid(row=0, column=1, sticky='ew', padx=5, pady=3)
        
        self.run_live_check_button = ttk.Button(control_panel, text="Controlla Segnali Attivi", command=self._run_live_check, style='Big.TButton')
        self.run_live_check_button.grid(row=1, column=0, sticky='ew', padx=5, pady=3)

        # --- NUOVA STRUTTURA PER IL PULSANTE DI RICERCA CON CHECKBOX ---
        find_pending_frame = ttk.Frame(control_panel)
        find_pending_frame.grid(row=1, column=1, sticky='ew', padx=5, pady=3)
        find_pending_frame.grid_columnconfigure(0, weight=1)

        self.run_find_pending_button = ttk.Button(find_pending_frame, text="Ricerca Previsioni in Attesa", command=self._run_find_pending, style='Big.TButton')
        self.run_find_pending_button.grid(row=0, column=0, sticky='ew')
        
        self.scan_all_var = tk.BooleanVar(value=False)
        self.scan_all_checkbox = ttk.Checkbutton(find_pending_frame, text="Scansione Completa", variable=self.scan_all_var)
        self.scan_all_checkbox.grid(row=0, column=1, sticky='w', padx=5)
        

        # --- SEZIONE 3: OUTPUT ---
        output_frame = ttk.LabelFrame(scrollable_frame, text="3. Output", padding=10)
        output_frame.pack(fill="x", expand=True, pady=5, padx=5)
        
        # Spostato qui il pulsante per pulire l'output
        ttk.Button(output_frame, text="Pulisci Output", command=self._clear_output).pack(anchor='ne', pady=(0, 5))
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, state='disabled', height=20, font=('Consolas', 10))
        self.output_text.pack(fill="both", expand=True)

        # Chiamate iniziali (INVARIATE)
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
        bersaglio_str = self.bersaglio_entry.get().strip()
        
        if require_bersaglio:
            if not bersaglio_str: 
                messagebox.showwarning("Input Errato", "Il campo bersaglio non può essere vuoto per questa funzione.")
                return None
        
        # Prova sempre a leggere il campo bersaglio se contiene del testo
        if bersaglio_str:
            try:
                numeri_bersaglio = [int(n) for n in bersaglio_str.replace(',', '-').split('-') if n.strip().isdigit()]
                if not (1 <= len(numeri_bersaglio) <= 2 and all(1 <= num <= 90 for num in numeri_bersaglio)): 
                    raise ValueError("Numero/i bersaglio non validi (1 o 2 numeri tra 1 e 90).")
                params['bersaglio'] = tuple(sorted(numeri_bersaglio))
            except ValueError as e: 
                messagebox.showwarning("Input Errato", f"Errore nel numero bersaglio: {e}")
                return None
        
        ruote_bersaglio_keys = [k for k, v in self.ruote_bersaglio_vars.items() if v.get()]
        if not ruote_bersaglio_keys: messagebox.showwarning("Input Errato", "Seleziona almeno una ruota bersaglio."); return None
        ruote_marker_keys = [k for k, v in self.ruote_marker_vars.items() if v.get()]
        if not ruote_marker_keys: messagebox.showwarning("Input Errato", "Seleziona almeno una ruota per i marker."); return None
        if not self.archivio.date_ordinate: messagebox.showerror("Errore", "Archivio non inizializzato."); return None
        end_date = self.archivio.date_ordinate[-1] if self.use_last_date_var.get() else self.end_date_entry.get_date().strftime('%Y-%m-%d')
        
        params.update({
            'ruote_bersaglio_keys': ruote_bersaglio_keys, 
            'start_date_str': self.start_date_entry.get_date().strftime('%Y-%m-%d'), 
            'end_date_str': end_date, 
            'ruote_marker_keys': ruote_marker_keys
        })
        return params

    def _get_strategy_params(self, params):
        """Raccoglie i parametri della strategia in modo condizionale."""
        try:
            params['finestra_antecedente'] = int(self.finestra_spinbox.get())
            params['colpi_di_gioco'] = int(self.colpi_gioco_spinbox.get())
            
            use_score = self.use_score_system_var.get()
            params['use_score_system'] = use_score
            
            if use_score:
                # Se usiamo il sistema a punteggio, leggi solo il campo del punteggio
                params['target_score'] = int(self.target_score_spinbox.get())
                # Per sicurezza, impostiamo l'altro a un valore di default
                params['soglia_attivazione'] = 0 
            else:
                # Altrimenti, leggi solo il campo della soglia standard
                params['soglia_attivazione'] = int(self.marker_attivazione_spinbox.get())
                # Per sicurezza, impostiamo l'altro a un valore di default
                params['target_score'] = 0

            return True # Indica che i parametri sono stati raccolti con successo
        except ValueError:
            messagebox.showwarning("Input Errato", "I campi numerici della strategia richiesti non possono essere vuoti.")
            return False # Indica fallimento

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
            try:
                # LA RIGA MANCANTE È QUESTA!
                # Aggiungiamo il valore della finestra antecedente ai parametri
                params['finestra_antecedente'] = int(self.finestra_spinbox.get())
            except ValueError:
                messagebox.showwarning("Input Errato", "Il valore per 'Finestra Antecedente' non è valido."); return

            def task_wrapper(p):
                self.analizzatore.trova_marker_per_bersaglio(p)
                self.risultato_ultima_analisi = p.get('risultato_analisi')
            self._run_task(task_wrapper, params)

    def _run_backtest(self):
        if not self.archivio.dati_per_analisi: messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        params = self._get_common_params()
        # SOSTITUISCI IL VECCHIO BLOCCO CON QUESTA RIGA
        if params and self._get_strategy_params(params):
            self._run_task(self.analizzatore.esegui_backtest, params)
 
    def _run_find_pending(self):
        """
        Esegue la ricerca delle previsioni in corso.
        Il suo comportamento dipende dalla checkbox 'Scansione Completa'.
        """
        if not self.archivio.dati_per_analisi: messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        
        # Raccoglie i parametri comuni.
        params = self._get_common_params(require_bersaglio=False)
        
        if params and self._get_strategy_params(params):
            
            # --- QUESTA È LA LOGICA CHIAVE ---
            # Se la casella "Scansione Completa" è spuntata,
            # rimuoviamo il bersaglio dai parametri per forzare la scansione generale.
            if self.scan_all_var.get():
                if 'bersaglio' in params:
                    del params['bersaglio']
            
            self._run_task(self.analizzatore.trova_previsioni_in_attesa, params)

    def _run_live_check(self):
        if not self.archivio.dati_per_analisi: messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        params = self._get_common_params()
        if params and self._get_strategy_params(params):
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