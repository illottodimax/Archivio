import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import requests
from datetime import datetime
from collections import Counter
from itertools import combinations
import threading
import queue
import math
import time
try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Libreria Mancante", "La libreria 'tkcalendar' non è installata.\n\nPer favore, installala eseguendo questo comando nel tuo terminale:\n\npip install tkcalendar")
    exit()

# ==============================================================================
# CLASSE "MOTORE" DELL'ANALISI (Backend)
# ==============================================================================
class AnalizzatoreIsotopiAvanzato:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.estrazioni_per_ruota, self.dati_per_analisi = {}, {}
        self.GITHUB_USER, self.GITHUB_REPO, self.GITHUB_BRANCH = "illottodimax", "Archivio", "main"
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
        self.URL_RUOTE = {k: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{v.upper()}.txt' for k, v in self.RUOTE_DISPONIBILI.items()}
        self.data_source = 'GitHub'
        self.local_path = None
        self.posizioni_str = ['1° Estratto', '2° Estratto', '3° Estratto', '4° Estratto', '5° Estratto']

    def _log(self, message): self.output_queue.put(message)

    def inizializza_archivio(self, force_reload=False):
        # ... questa funzione rimane INVARIATA ...
        self._log("Inizio inizializzazione archivio...")
        if self.data_source == 'Locale':
            self._log(f"Fonte dati impostata su Locale. Percorso: {self.local_path}")
            if not self.local_path or not os.path.isdir(self.local_path):
                raise FileNotFoundError(f"Percorso locale non valido o non impostato.\nSeleziona una cartella valida usando il pulsante 'Sfoglia...'.")
        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            if ruota_key in self.estrazioni_per_ruota and not force_reload: continue
            self._log(f"Caricando {ruota_nome} ({i+1}/{len(self.RUOTE_DISPONIBILI)})...")
            try:
                if self.data_source == 'GitHub':
                    r = requests.get(self.URL_RUOTE[ruota_key], timeout=15); r.raise_for_status(); linee = r.text.strip().split('\n')
                else: 
                    percorso = os.path.join(self.local_path, f"{ruota_nome.upper()}.txt")
                    with open(percorso, 'r', encoding='utf-8') as f: linee = f.readlines()
                self.estrazioni_per_ruota[ruota_key] = self._parse_estrazioni(linee)
            except Exception as e:
                self._log(f" -> FALLITO. Errore: {e}"); raise RuntimeError(f"Impossibile caricare i dati per {ruota_nome}.") from e
        self._prepara_dati_per_analisi()
        self._log("\nArchivio inizializzato. Pronto per l'analisi.")

    def _parse_estrazioni(self, linee):
        # ... questa funzione rimane INVARIATA ...
        return [{'data': datetime.strptime(p[0], '%Y/%m/%d').strftime('%Y-%m-%d'), 'numeri': [int(n) for n in p[2:7]]} for l in linee if len(p:=l.strip().split('\t'))>=7]

    def _prepara_dati_per_analisi(self):
        self._log("Preparo e allineo i dati per l'analisi incrociata...")
        tutte_le_date = set(e['data'] for estrazioni in self.estrazioni_per_ruota.values() for e in estrazioni)
        self.date_ordinate = sorted(list(tutte_le_date))
        self.date_to_index = {data: i for i, data in enumerate(self.date_ordinate)}
        
        # Sostituiamo la dictionary comprehension con un ciclo for per migliorare la responsività della UI
        dati_per_analisi_temp = {}
        for i, data in enumerate(self.date_ordinate):
            estrazioni_giorno = {}
            for ruota, estrazioni in self.estrazioni_per_ruota.items():
                # Trova l'estrazione per la data corrente
                numeri_trovati = next((e['numeri'] for e in estrazioni if e['data'] == data), None)
                if numeri_trovati:
                    estrazioni_giorno[ruota] = numeri_trovati
            
            dati_per_analisi_temp[data] = estrazioni_giorno
            
            # OGNI 100 DATE PROCESSATE, FACCIAMO UNA MICRO-PAUSA PER LA UI
            if i % 100 == 0:
                time.sleep(0.001) # Pausa di 1 millisecondo

        self.dati_per_analisi = dati_per_analisi_temp
        self._log("Dati allineati.")

    ## --- FUNZIONE MODIFICATA PER GESTIRE LE MODALITÀ ---
    def analizza_e_verifica(self, colpi_di_gioco, limite_casi_coppia, data_ref_str, num_ambate, num_abbinamenti, mode, limite_eventi_globale):
        if not self.dati_per_analisi: self._log("\nERRORE: Archivio non inizializzato."); return
        data_da_analizzare = next((d for d in reversed(self.date_ordinate) if d <= data_ref_str), None)
        if not data_da_analizzare:
            self._log(f"\nERRORE: Nessuna estrazione trovata in data {data_ref_str} o precedente."); return
        self._log("\n" + "#"*80 + f"\n##  SCANSIONE E VERIFICA PER L'ESTRAZIONE DEL: {data_da_analizzare}  ##\n" + "#"*80)
        
        estrazione_giorno = self.dati_per_analisi[data_da_analizzare]
        casi_trovati_nel_giorno = [{'numero': estrazione_giorno[r1][p], 'data': data_da_analizzare, 'ruota1': r1, 'ruota2': r2, 'posizione_idx': p}
                                   for r1, r2 in combinations(sorted(estrazione_giorno.keys()), 2)
                                   for p in range(5) if estrazione_giorno.get(r1) and estrazione_giorno.get(r2) and estrazione_giorno[r1][p] == estrazione_giorno[r2][p]]

        if not casi_trovati_nel_giorno:
            self._log(f"\n>>> NESSUN CASO DI ISOTOPIA TROVATO NELL'ESTRAZIONE DEL {data_da_analizzare}."); return
            
        self._log(f"\n>>> Trovati {len(casi_trovati_nel_giorno)} casi di isotopia! Avvio analisi per ciascuno in modalità '{'GLOBALE' if mode == 'globale' else 'CLASSICA'}':")

        for caso in casi_trovati_nel_giorno:
            if mode == 'coppia':
                self._esegui_analisi_e_verifica_per_coppia(caso, colpi_di_gioco, limite_casi_coppia, num_ambate, num_abbinamenti)
            elif mode == 'globale':
                self._esegui_analisi_globale_per_numero(caso, colpi_di_gioco, limite_eventi_globale, num_ambate, num_abbinamenti)

    ## --- QUESTA FUNZIONE RIMANE INVARIATA (MODALITÀ CLASSICA) ---
    def _esegui_analisi_e_verifica_per_coppia(self, caso_isotopo, colpi_gioco, limite_casi, num_ambate, num_abbinamenti):
        r1, r2, pos_idx, num_isotopo = caso_isotopo['ruota1'], caso_isotopo['ruota2'], caso_isotopo['posizione_idx'], caso_isotopo['numero']
        self._log("\n" + "="*80 + f"\n||  ANALISI (Coppia): {r1}-{r2} | ISOTOPO {num_isotopo} | POS: {self.posizioni_str[pos_idx].upper()}  ||\n" + "="*80)
        tutti_casi = self._trova_tutti_casi_isotopi(r1, r2, pos_idx)
        casi_per_stat = self._filtra_casi_per_statistica(tutti_casi, limite_casi, caso_isotopo['data'])
        if not casi_per_stat: self._log(">>> Non ci sono abbastanza casi storici per generare una previsione."); return
        ambate_candidate = self._trova_ambate_candidate(casi_per_stat, [r1, r2])
        if not ambate_candidate: self._log(">>> Nessun numero trovato nelle estrazioni successive ai casi storici."); return

        for i in range(num_ambate):
            if i >= len(ambate_candidate): break
            tipo_ambata, ambata_attuale = ("PRIMARIA", ambate_candidate[0]['numero']) if i == 0 else ("SECONDARIA", ambate_candidate[1]['numero'])
            self._log(f"\n--- PREVISIONE {i+1} (AMBATA {tipo_ambata}) ---")
            abbinamenti = [ab['numero'] for ab in self._trova_abbinamenti_per_consequenzialita(ambata_attuale, casi_per_stat, [r1, r2], num_abbinamenti)]
            lunghetta = sorted([ambata_attuale] + abbinamenti)
            self._log(f"🎯 AMBATA CAPOGIOCO: {ambata_attuale}\n🔗 LUNGHETTA ({len(lunghetta)} numeri): {' - '.join(map(str, lunghetta))}\n📍 RUOTE DI GIOCO: {self.RUOTE_DISPONIBILI[r1]}, {self.RUOTE_DISPONIBILI[r2]}\n⏱️ VALIDITÀ GIOCO: {colpi_gioco} colpi")
            self._log(f"\n--- VERIFICA AFFIDABILITÀ (Backtest per Ambata {ambata_attuale}) ---")
            self._verifica_efficacia(tutti_casi, r1, r2, colpi_gioco, limite_casi, ambata_attuale, set(lunghetta))
            self._controlla_esito_previsione(caso_isotopo['data'], colpi_gioco, r1, r2, ambata_attuale, set(lunghetta))
            
    ## --- NUOVA FUNZIONE PER LA MODALITÀ GLOBALE ---
    def _esegui_analisi_globale_per_numero(self, caso_isotopo, colpi_gioco, limite_eventi, num_ambate, num_abbinamenti):
        r1_gioco, r2_gioco = caso_isotopo['ruota1'], caso_isotopo['ruota2'] # Ruote su cui si giocherà
        pos_idx, num_isotopo = caso_isotopo['posizione_idx'], caso_isotopo['numero']
        self._log("\n" + "="*80 + f"\n||  ANALISI (Globale): NUMERO ISOTOPO {num_isotopo} | POS: {self.posizioni_str[pos_idx].upper()}  ||\n" + "="*80)
        
        tutti_casi_globali = self._trova_tutti_casi_isotopi_globali(num_isotopo, pos_idx)
        casi_per_stat = self._filtra_casi_per_statistica(tutti_casi_globali, limite_eventi, caso_isotopo['data'])
        if not casi_per_stat: self._log(f">>> Non ci sono abbastanza casi storici globali ({len(tutti_casi_globali)} totali) per generare una previsione."); return
        self._log(f"Trovati {len(tutti_casi_globali)} casi storici globali. Ne verranno usati {len(casi_per_stat)} per l'analisi.")

        ambate_candidate = self._trova_ambate_candidate_globale(casi_per_stat)
        if not ambate_candidate: self._log(">>> Nessun numero trovato nelle estrazioni successive ai casi storici globali."); return

        for i in range(num_ambate):
            if i >= len(ambate_candidate): break
            tipo_ambata = "PRIMARIA" if i == 0 else "SECONDARIA"
            ambata_attuale = ambate_candidate[i]['numero']
            self._log(f"\n--- PREVISIONE {i+1} (AMBATA {tipo_ambata}) ---")
            abbinamenti = [ab['numero'] for ab in self._trova_abbinamenti_globale(ambata_attuale, casi_per_stat, num_abbinamenti)]
            lunghetta = sorted([ambata_attuale] + abbinamenti)
            self._log(f"🎯 AMBATA CAPOGIOCO: {ambata_attuale}\n🔗 LUNGHETTA ({len(lunghetta)} numeri): {' - '.join(map(str, lunghetta))}\n📍 RUOTE DI GIOCO: {self.RUOTE_DISPONIBILI[r1_gioco]}, {self.RUOTE_DISPONIBILI[r2_gioco]}\n⏱️ VALIDITÀ GIOCO: {colpi_gioco} colpi")
            self._log(f"\n--- VERIFICA AFFIDABILITÀ (Backtest Globale per Ambata {ambata_attuale}) ---")
            self._verifica_efficacia_globale(tutti_casi_globali, colpi_gioco, limite_eventi, ambata_attuale, set(lunghetta))
            self._controlla_esito_previsione(caso_isotopo['data'], colpi_gioco, r1_gioco, r2_gioco, ambata_attuale, set(lunghetta))
            
    ## --- NUOVA FUNZIONE DI VERIFICA GLOBALE ---
    def _verifica_efficacia_globale(self, tutti_casi, colpi_gioco, limite_eventi, ambata_test, lunghetta_test):
        casi_testabili = [e for i, e in enumerate(tutti_casi) if i < len(tutti_casi) - 1]
        if limite_eventi > 0 and len(casi_testabili) > limite_eventi:
            casi_testabili = casi_testabili[-limite_eventi:]
        if not casi_testabili: self._log("Non ci sono abbastanza casi storici per una verifica affidabile."); return

        stats_ambata_secca, punti_realizzati = 0, []
        for evento in casi_testabili:
            idx, r1_storico, r2_storico = self.date_to_index.get(evento['data']), evento['ruota1'], evento['ruota2']
            for colpo in range(1, colpi_gioco + 1):
                idx_v = idx + colpo
                if idx_v < len(self.date_ordinate):
                    estrazione_v = self.dati_per_analisi[self.date_ordinate[idx_v]]
                    if ambata_test in estrazione_v.get(r1_storico, []) or ambata_test in estrazione_v.get(r2_storico, []):
                        stats_ambata_secca += 1
                        break
        
        for evento in casi_testabili:
            idx, r1_storico, r2_storico, punti_caso = self.date_to_index.get(evento['data']), evento['ruota1'], evento['ruota2'], 0
            for colpo in range(1, colpi_gioco + 1):
                idx_v = idx + colpo
                if idx_v >= len(self.date_ordinate): break
                estrazione_v = self.dati_per_analisi[self.date_ordinate[idx_v]]
                punti = max(len(lunghetta_test.intersection(set(estrazione_v.get(r1_storico, [])))), len(lunghetta_test.intersection(set(estrazione_v.get(r2_storico, [])))))
                if punti > 0: punti_caso = punti; break
            punti_realizzati.append(punti_caso)

        casi_q, casi_t, casi_a = sum(1 for p in punti_realizzati if p >= 4), sum(1 for p in punti_realizzati if p == 3), sum(1 for p in punti_realizzati if p == 2)
        tot_a, tot_t, tot_q = (casi_a * 1) + (casi_t * 3) + (casi_q * 6), (casi_t * 1) + (casi_q * 4), casi_q
        num_testati = len(casi_testabili)
        self._log(f"Verifica globale effettuata sugli ultimi {num_testati} casi storici:")
        self._log(f"  -> Affidabilità Ambata Secca: {stats_ambata_secca} volte ({(stats_ambata_secca/num_testati)*100:.2f}%)")
        self._log("  --- Affidabilità Lunghetta (Casi Vincenti per Sorte Massima) ---")
        for val, nome in [(casi_a, 'Ambo'), (casi_t, 'Terno'), (casi_q, 'Quaterna')]: self._log(f"  -> {nome:<12} (come esito max): {val} volte ({(val/num_testati)*100:.2f}%)")
        if tot_a > 0 or tot_t > 0 or tot_q > 0:
            self._log("\n  --- Riepilogo Vincite Complessive ---"); self._log(f"  -> Totale Ambi: {tot_a} | Totale Terni: {tot_t} | Totale Quaterne: {tot_q}")

    ## --- NUOVI HELPERS GLOBALI ---
    def _trova_tutti_casi_isotopi_globali(self, numero_isotopo, pos_idx):
        casi = []
        for data, estrazioni_del_giorno in self.dati_per_analisi.items():
            ruote_con_numero_in_pos = [r for r, numeri in estrazioni_del_giorno.items() if len(numeri) > pos_idx and numeri[pos_idx] == numero_isotopo]
            if len(ruote_con_numero_in_pos) >= 2:
                for r1, r2 in combinations(ruote_con_numero_in_pos, 2):
                    casi.append({'numero': numero_isotopo, 'data': data, 'ruota1': r1, 'ruota2': r2})
        return sorted(casi, key=lambda x: x['data'])

    def _trova_ambate_candidate_globale(self, casi):
        counter = Counter()
        for caso in casi:
            estrazione_succ = self.dati_per_analisi.get(caso['data_successiva'], {})
            for r in [caso['ruota1'], caso['ruota2']]:
                counter.update(estrazione_succ.get(r, []))
        return [{'numero': num, 'presenze': freq} for num, freq in counter.most_common(2)] if counter else None
        
    def _trova_abbinamenti_globale(self, ambata, casi, num_abbinamenti):
        counter = Counter()
        for caso in casi:
            estrazione_succ = self.dati_per_analisi.get(caso['data_successiva'], {})
            for r in [caso['ruota1'], caso['ruota2']]:
                numeri_ruota = estrazione_succ.get(r, [])
                if ambata in numeri_ruota:
                    counter.update(n for n in numeri_ruota if n != ambata)
        return [{'numero': num, 'presenze': freq} for num, freq in counter.most_common(num_abbinamenti)]
        
    def _controlla_esito_previsione(self, data_riferimento, colpi_gioco, r1, r2, ambata_test, lunghetta_test):
        self._log("\n--- ESITO DELLA PREVISIONE CORRENTE ---")
        if data_riferimento == self.date_ordinate[-1]:
            # NUOVO MARCATORE: INFO::
            self._log("INFO::STATO: IN CORSO. Questa è l'ultima estrazione, nessuna verifica possibile.")
            return

        idx_riferimento = self.date_to_index.get(data_riferimento)
        for colpo in range(1, colpi_gioco + 1):
            idx_verifica = idx_riferimento + colpo
            if idx_verifica >= len(self.date_ordinate):
                # NUOVO MARCATORE: INFO::
                self._log(f"INFO::STATO: IN CORSO. Controllate {colpo - 1}/{colpi_gioco} estrazioni successive.")
                return

            data_verifica, estrazione = self.date_ordinate[idx_verifica], self.dati_per_analisi[self.date_ordinate[idx_verifica]]
            punti = max(len(lunghetta_test.intersection(set(estrazione.get(r1, [])))), len(lunghetta_test.intersection(set(estrazione.get(r2, [])))))
            
            if punti >= 2:
                numeri_vincenti = lunghetta_test.intersection(set(estrazione.get(r1 if punti == len(lunghetta_test.intersection(set(estrazione.get(r1, [])))) else r2, [])))
                messaggi = ["Quaterna" if punti >= 4 else "Terno" if punti == 3 else "Ambo"]
                if ambata_test in numeri_vincenti: messaggi.append("Ambata Secca")
                # NUOVO MARCATORE: WIN::
                self._log(f"WIN::ESITO: VINCITA! Sorte: {' + '.join(sorted(messaggi))}, Colpo: {colpo}, Data: {data_verifica}.")
                return
            elif ambata_test in estrazione.get(r1, []) or ambata_test in estrazione.get(r2, []):
                # NUOVO MARCATORE: WIN::
                self._log(f"WIN::ESITO: VINCITA! Sorte: Ambata Secca, Colpo: {colpo}, Data: {data_verifica}.")
                return
        
        # NUOVO MARCATORE: FAIL::
        self._log(f"FAIL::ESITO: NEGATIVO. Nessuna vincita nei {colpi_gioco} colpi successivi.")

    def _verifica_efficacia(self, tutti_casi, r1, r2, colpi_gioco, limite_casi, ambata_test, lunghetta_test):
        # ... questa funzione rimane INVARIATA ...
        casi_testabili = [e for i, e in enumerate(tutti_casi) if i < len(tutti_casi) - 1]
        if limite_casi > 0 and len(casi_testabili) > limite_casi: casi_testabili = casi_testabili[-limite_casi:]
        if not casi_testabili: self._log("Non ci sono abbastanza casi storici per una verifica affidabile."); return

        stats_ambata_secca, punti_realizzati = 0, []
        for evento in casi_testabili:
            idx = self.date_to_index.get(evento['data'])
            for colpo in range(1, colpi_gioco + 1):
                if idx + colpo < len(self.date_ordinate) and (ambata_test in self.dati_per_analisi[self.date_ordinate[idx+colpo]].get(r1, []) or ambata_test in self.dati_per_analisi[self.date_ordinate[idx+colpo]].get(r2, [])):
                    stats_ambata_secca += 1; break
        
        for evento in casi_testabili:
            idx, punti_caso = self.date_to_index.get(evento['data']), 0
            for colpo in range(1, colpi_gioco + 1):
                idx_v = idx + colpo; 
                if idx_v >= len(self.date_ordinate): break
                punti = max(len(lunghetta_test.intersection(set(self.dati_per_analisi[self.date_ordinate[idx_v]].get(r1, [])))), len(lunghetta_test.intersection(set(self.dati_per_analisi[self.date_ordinate[idx_v]].get(r2, [])))))
                if punti > 0: punti_caso = punti; break
            punti_realizzati.append(punti_caso)

        casi_q, casi_t, casi_a = sum(1 for p in punti_realizzati if p >= 4), sum(1 for p in punti_realizzati if p == 3), sum(1 for p in punti_realizzati if p == 2)
        tot_a, tot_t, tot_q = (casi_a * 1) + (casi_t * 3) + (casi_q * 6), (casi_t * 1) + (casi_q * 4), casi_q
        
        num_testati = len(casi_testabili)
        self._log(f"Verifica effettuata sugli ultimi {num_testati} casi storici:")
        self._log(f"  -> Affidabilità Ambata Secca: {stats_ambata_secca} volte ({(stats_ambata_secca/num_testati)*100:.2f}%)")
        self._log("  --- Affidabilità Lunghetta (Casi Vincenti per Sorte Massima) ---")
        for val, nome in [(casi_a, 'Ambo'), (casi_t, 'Terno'), (casi_q, 'Quaterna')]: self._log(f"  -> {nome:<12} (come esito max): {val} volte ({(val/num_testati)*100:.2f}%)")
        
        if tot_a > 0 or tot_t > 0 or tot_q > 0:
            self._log("\n  --- Riepilogo Vincite Complessive ---")
            self._log(f"  -> Totale Ambi vinti: {tot_a} | Totale Terni vinti: {tot_t} | Totale Quaterne vinte: {tot_q}")

    def _filtra_casi_per_statistica(self, casi, limite, data_esclusa=None):
        # ... questa funzione rimane INVARIATA ...
        validi = [c for c in casi if (not data_esclusa or c['data'] < data_esclusa) and self.date_to_index[c['data']] + 1 < len(self.date_ordinate)]
        if limite > 0 and len(validi) > limite: validi = validi[-limite:]
        return [{'numero': c['numero'], 'data_evento': c['data'], 'data_successiva': self.date_ordinate[self.date_to_index[c['data']]+1], 'ruota1': c['ruota1'], 'ruota2': c['ruota2']} for c in validi]
    def _trova_tutti_casi_isotopi(self, r1, r2, pos): 
        # ... questa funzione rimane INVARIATA ...
        return [{'numero': d[r1][pos], 'data': data, 'ruota1': r1, 'ruota2': r2} for data, d in self.dati_per_analisi.items() if r1 in d and r2 in d and len(d[r1])>pos and len(d[r2])>pos and d[r1][pos] == d[r2][pos]]
    def _trova_ambate_candidate(self, casi, ruote): 
        # ... questa funzione rimane INVARIATA ...
        c = Counter(n for caso in casi for r in ruote for n in self.dati_per_analisi[caso['data_successiva']].get(r,[])); return [{'numero': num, 'presenze': freq} for num, freq in c.most_common(2)] if c else None
    def _trova_abbinamenti_per_consequenzialita(self, ambata, casi, ruote, num_abbinamenti): 
        # ... questa funzione rimane INVARIATA ...
        c = Counter(n for caso in casi for r in ruote if ambata in self.dati_per_analisi[caso['data_successiva']].get(r,[]) for n in self.dati_per_analisi[caso['data_successiva']][r] if n != ambata); return [{'numero': num, 'presenze': freq} for num, freq in c.most_common(num_abbinamenti)]

class LottoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizzatore Isotopi Avanzato - Created by Max Lotto")
        self.output_queue = queue.Queue()
        self.analizzatore = AnalizzatoreIsotopiAvanzato(self.output_queue)
        
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.X)
        
        control_frame = ttk.LabelFrame(main_frame, text="Pannello di Controllo", padding=(10, 5))
        control_frame.pack(fill=tk.X)
        
        # Colonna 0: Fonte Dati
        ttk.Label(control_frame, text="Fonte Dati:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.source_var = tk.StringVar(value="GitHub")
        source_combo = ttk.Combobox(control_frame, textvariable=self.source_var, values=["GitHub", "Locale"], width=10, state="readonly")
        source_combo.grid(row=1, column=0, padx=5, pady=2)
        source_combo.bind("<<ComboboxSelected>>", self.on_source_change)

        # Colonna 1: Modalità di Analisi
        mode_frame = ttk.LabelFrame(control_frame, text="Modalità Analisi")
        mode_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=2, sticky="ns")
        self.mode_var = tk.StringVar(value="coppia")
        ttk.Radiobutton(mode_frame, text="Per Coppia Ruote (Classica)", variable=self.mode_var, value="coppia").pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(mode_frame, text="Per Numero Isotopo (Globale)", variable=self.mode_var, value="globale").pack(anchor=tk.W, padx=5)

        # Colonna 2: Parametri di gioco
        params_frame = ttk.Frame(control_frame)
        params_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=2)
        ttk.Label(params_frame, text="N. Ambate:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.num_ambate_var = tk.IntVar(value=1)
        ttk.Combobox(params_frame, textvariable=self.num_ambate_var, values=[1, 2], width=5, state="readonly").grid(row=1, column=0, padx=5, pady=2)
        
        ttk.Label(params_frame, text="N. Abbinamenti:").grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.num_abbinamenti_var = tk.IntVar(value=5)
        ttk.Combobox(params_frame, textvariable=self.num_abbinamenti_var, values=list(range(1, 11)), width=5, state="readonly").grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Colpi Gioco:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.colpi_var = tk.StringVar(value="18")
        ttk.Entry(params_frame, textvariable=self.colpi_var, width=5).grid(row=1, column=2, padx=5, pady=2)

        # Colonna 3: Parametri Storici
        history_frame = ttk.Frame(control_frame)
        history_frame.grid(row=0, column=3, rowspan=2, padx=5, pady=2)
        ttk.Label(history_frame, text="Casi Storici (Coppia):").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.casi_var = tk.StringVar(value="10")
        ttk.Entry(history_frame, textvariable=self.casi_var, width=10).grid(row=1, column=0, padx=5, pady=2)

        ttk.Label(history_frame, text="Eventi da Usare (Globale):").grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        self.eventi_var = tk.StringVar(value="10")
        ttk.Entry(history_frame, textvariable=self.eventi_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        self.local_path_label = ttk.Label(control_frame, text="Percorso Cartella:")
        self.local_path_var = tk.StringVar(value="Nessun percorso selezionato")
        self.local_path_entry = ttk.Entry(control_frame, textvariable=self.local_path_var, state="readonly", width=40)
        self.browse_button = ttk.Button(control_frame, text="Sfoglia...", command=self.browse_local_path)
        
        date_frame = ttk.Frame(main_frame)
        date_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(date_frame, text="Data di Riferimento Analisi:").pack(side=tk.LEFT, padx=(0,10))
        self.date_entry = DateEntry(date_frame, width=12, date_pattern='dd/mm/yyyy', locale='it_IT')
        self.date_entry.pack(side=tk.LEFT)
        self.start_button = ttk.Button(date_frame, text="AVVIA ANALISI", command=self.start_analysis_thread, style="Accent.TButton")
        self.start_button.pack(side=tk.RIGHT, padx=5)
        
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'), padding=6)
        
        results_frame = ttk.LabelFrame(root, text="Risultati Analisi", padding=(10, 5))
        results_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, state='disabled', font=("Courier New", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # --- NUOVO: Definiamo i tag per i colori ---
        self.results_text.tag_config('vincita', foreground='#e60000', font=("Courier New", 10, 'bold'))
        self.results_text.tag_config('negativo', foreground='#E59400')
        self.results_text.tag_config('info', foreground='blue')
        
        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
        self.on_source_change()
        self.process_queue()

    def on_source_change(self, event=None):
        if self.source_var.get() == "Locale":
            self.local_path_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
            self.local_path_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
            self.browse_button.grid(row=2, column=3, padx=5, pady=5)
        else:
            self.local_path_label.grid_remove()
            self.local_path_entry.grid_remove()
            self.browse_button.grid_remove()

    def browse_local_path(self):
        path = filedialog.askdirectory(title="Seleziona la cartella archivio")
        self.local_path_var.set(path if path else "Nessun percorso selezionato")
        self.analizzatore.local_path = path

    def start_analysis_thread(self):
        try:
            colpi = int(self.colpi_var.get())
            limite_casi_coppia = int(self.casi_var.get())
            limite_eventi_globale = int(self.eventi_var.get())
            num_ambate = self.num_ambate_var.get()
            num_abbinamenti = self.num_abbinamenti_var.get()
            analysis_mode = self.mode_var.get()
        except ValueError:
            messagebox.showerror("Errore", "Controlla che i campi numerici siano corretti."); return

        if self.source_var.get() == "Locale" and not os.path.isdir(self.analizzatore.local_path):
            messagebox.showerror("Errore", "Seleziona una cartella valida per la fonte 'Locale'."); return
            
        self.start_button.config(state='disabled')
        self.results_text.config(state='normal'); self.results_text.delete('1.0', tk.END); self.results_text.config(state='disabled')
        
        thread_args = (
            self.source_var.get(),
            self.date_entry.get_date().strftime('%Y-%m-%d'),
            colpi,
            limite_casi_coppia,
            num_ambate,
            num_abbinamenti,
            analysis_mode,
            limite_eventi_globale
        )
        threading.Thread(target=self.run_full_analysis, args=thread_args, daemon=True).start()

    def run_full_analysis(self, source, data_ref, colpi, limite_casi, num_ambate, num_abbinamenti, mode, limite_eventi):
        try:
            self.analizzatore.data_source = source
            self.analizzatore.inizializza_archivio()
            self.analizzatore.analizza_e_verifica(colpi, limite_casi, data_ref, num_ambate, num_abbinamenti, mode, limite_eventi)
        except Exception as e:
            self._log(f"\nERRORE CRITICO: {e}")
        finally:
            self.output_queue.put("ANALYSIS_COMPLETE")
            
    def _log(self, message): self.output_queue.put(message)

    def process_queue(self):
        try:
            while True:
                message = self.output_queue.get_nowait()
                if message == "ANALYSIS_COMPLETE":
                    self.start_button.config(state='normal')
                    self.status_var.set("Analisi completata. Pronto.")
                else:
                    self.results_text.config(state='normal')
                    
                    # --- NUOVO: Logica per interpretare i tag e colorare il testo ---
                    tag_da_usare = None
                    testo_da_visualizzare = message

                    if message.startswith("WIN::"):
                        tag_da_usare = 'vincita'
                        testo_da_visualizzare = message.split('::', 1)[1]
                    elif message.startswith("FAIL::"):
                        tag_da_usare = 'negativo'
                        testo_da_visualizzare = message.split('::', 1)[1]
                    elif message.startswith("INFO::"):
                        tag_da_usare = 'info'
                        testo_da_visualizzare = message.split('::', 1)[1]
                    
                    # Inseriamo il testo con il tag corretto (o senza tag se non specificato)
                    self.results_text.insert(tk.END, testo_da_visualizzare + "\n", tag_da_usare)
                    # --- FINE NUOVA LOGICA ---
                    
                    self.results_text.yview(tk.END)
                    self.results_text.config(state='disabled')
                    # Pulisce il marcatore dalla status bar per non vederlo in basso
                    self.status_var.set(message.split('\n')[-1].split('::')[-1])
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = LottoApp(root)
    root.mainloop()