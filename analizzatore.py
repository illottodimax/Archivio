import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import requests
from datetime import datetime
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

class AnalizzatoreLottoAvanzato:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.estrazioni_per_ruota, self.dati_per_analisi = {}, {}
        self.GITHUB_USER, self.GITHUB_REPO, self.GITHUB_BRANCH = "illottodimax", "Archivio", "main"
        self.RUOTE_DISPONIBILI = {
            'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova',
            'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma',
            'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'
        }
        self.URL_RUOTE = {k: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{v.upper()}.txt' for k, v in self.RUOTE_DISPONIBILI.items()}
        self.data_source = 'GitHub'
        self.local_path = None
        self.TERZINE_SIMMETRICHE_LIST = self._generate_terzine_simmetriche()

    def _generate_terzine_simmetriche(self):
        terzine = []
        for i in range(1, 31):
            terzine.append(tuple(sorted((i, i + 30, i + 60))))
        return sorted(list(set(terzine)))

    def _log(self, message):
        self.output_queue.put(message)

    def inizializza_archivio(self, force_reload=False):
        self._log("Inizio inizializzazione archivio...")
        if self.data_source == 'Locale' and (not self.local_path or not os.path.isdir(self.local_path)):
            raise FileNotFoundError("Percorso locale non valido o non impostato.")
        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            if ruota_key in self.estrazioni_per_ruota and not force_reload: continue
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
        self._log("\nArchivio inizializzato. Pronto per l'analisi.")

    def _parse_estrazioni(self, linee):
        parsed_data = []
        for l in linee:
            parts = l.strip().split('\t')
            if len(parts) >= 7:
                try:
                    data_str = datetime.strptime(parts[0], '%Y/%m/%d').strftime('%Y-%m-%d')
                    numeri = [int(n) for n in parts[2:7] if n.isdigit() and 1 <= int(n) <= 90]
                    if len(numeri) == 5: parsed_data.append({'data': data_str, 'numeri': numeri})
                except (ValueError, IndexError): pass
        return parsed_data

    def _prepara_dati_per_analisi(self):
        self._log("Preparo e allineo i dati per l'analisi...")
        tutte_le_date = {e['data'] for estrazioni in self.estrazioni_per_ruota.values() for e in estrazioni}
        self.date_ordinate = sorted(list(tutte_le_date))
        self.date_to_index = {data: i for i, data in enumerate(self.date_ordinate)}
        self.dati_per_analisi = {data: {ruota: None for ruota in self.RUOTE_DISPONIBILI} for data in self.date_ordinate}
        for ruota, estrazioni in self.estrazioni_per_ruota.items():
            for e in estrazioni:
                if e['data'] in self.dati_per_analisi: self.dati_per_analisi[e['data']][ruota] = e['numeri']
        self.dati_per_analisi = {d: v for d, v in self.dati_per_analisi.items() if any(v.values())}
        self._log("Dati allineati.")

    def _get_events_in_range(self, start_date_str, end_date_str):
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
        return [d for d in reversed(self.date_ordinate) if start_dt <= datetime.strptime(d, '%Y-%m-%d') <= end_dt]

    # --- FUNZIONE CORRETTA ---
    def _process_events(self, eventi, num_estrazioni_da_analizzare, direction="future"):
        presenze_per_numero = Counter()
        
        for data_evento_str, _, ruota1_key, ruota2_key in eventi:
            current_date_index = self.date_to_index.get(data_evento_str, -1)
            if current_date_index == -1: 
                continue

            range_di_analisi = range(1, num_estrazioni_da_analizzare + 1) if direction == "future" else range(-1, -num_estrazioni_da_analizzare - 1, -1)

            for k in range_di_analisi:
                target_date_index = current_date_index + k
                if 0 <= target_date_index < len(self.date_ordinate):
                    data_da_analizzare = self.date_ordinate[target_date_index]
                    for ruota_monitorata in [ruota1_key, ruota2_key]:
                        numeri = self.dati_per_analisi.get(data_da_analizzare, {}).get(ruota_monitorata)
                        if numeri:
                            # CORREZIONE: Aggiorna il contatore per ogni numero estratto
                            for num in numeri:
                                presenze_per_numero[num] += 1
                                
        final_counter = Counter()
        if eventi:
            ruota_da_usare = eventi[0][2]
            for numero, freq in presenze_per_numero.items():
                final_counter[(numero, ruota_da_usare)] = freq
                
        return final_counter

    def analizza_multipli_eventi_ambo_diviso(self, num1, num2, ruota1_key, ruota2_key, start_date_str, end_date_str, max_events_to_analyze, num_estrazioni_future):
        ambo, eventi_trovati_raw = tuple(sorted((num1, num2))), []
        for data_corrente_str in self._get_events_in_range(start_date_str, end_date_str):
            if len(eventi_trovati_raw) >= max_events_to_analyze: break
            n_r1 = self.dati_per_analisi.get(data_corrente_str, {}).get(ruota1_key)
            n_r2 = self.dati_per_analisi.get(data_corrente_str, {}).get(ruota2_key)
            if n_r1 and n_r2 and ((ambo[0] in n_r1 and ambo[1] in n_r2) or (ambo[1] in n_r1 and ambo[0] in n_r2)):
                eventi_trovati_raw.append((data_corrente_str, ambo, ruota1_key, ruota2_key))
        
        if not eventi_trovati_raw: return Counter(), [], 0
        
        # --- CORREZIONE APPLICATA QUI ---
        # Ora la statistica per trovare l'ambata si basa sui colpi *precedenti* all'evento.
        # Questo impedisce di "sbirciare" i risultati futuri per creare la previsione.
        # La fase di verifica (fatta dopo) userà invece i colpi futuri per testare questa previsione "onesta".
        lookback_period = 100  # Numero fisso di estrazioni passate da analizzare
        ambate_counter = self._process_events(eventi_trovati_raw, lookback_period, direction="past")
        
        return ambate_counter, eventi_trovati_raw, 0

    def analizza_terzina_simmetrica_divisa_specifica(self, terzina_str, ruota1_key, ruota2_key, start_date_str, end_date_str, max_events_to_analyze, num_estrazioni_future):
        try: terzina = tuple(sorted(int(n) for n in terzina_str.split('-')))
        except: return Counter(), [], 0
        eventi_trovati_raw = []
        for data_corrente_str in self._get_events_in_range(start_date_str, end_date_str):
            if len(eventi_trovati_raw) >= max_events_to_analyze: break
            n_r1 = set(self.dati_per_analisi.get(data_corrente_str, {}).get(ruota1_key) or [])
            n_r2 = set(self.dati_per_analisi.get(data_corrente_str, {}).get(ruota2_key) or [])
            if not n_r1 or not n_r2: continue
            for c in combinations(terzina, 2):
                ambo, singolo = set(c), (set(terzina) - set(c)).pop()
                if (ambo.issubset(n_r1) and singolo in n_r2) or (ambo.issubset(n_r2) and singolo in n_r1):
                    eventi_trovati_raw.append((data_corrente_str, terzina, ruota1_key, ruota2_key)); break
        
        if not eventi_trovati_raw: return Counter(), [], 0
        
        # --- CORREZIONE APPLICATA QUI ---
        # Anche qui, la previsione si basa sul passato per non "sbirciare".
        lookback_period = 100
        ambate_counter = self._process_events(eventi_trovati_raw, lookback_period, direction="past")
        
        return ambate_counter, eventi_trovati_raw, 0

    def analizza_terzina_simmetrica_divisa_automatica(self, ruota1_key, ruota2_key, start_date_str, end_date_str, max_events_to_analyze, num_estrazioni_future):
        eventi_trovati_raw = []
        for data_corrente_str in self._get_events_in_range(start_date_str, end_date_str):
            if len(eventi_trovati_raw) >= max_events_to_analyze: break
            estrazione_corrente = self.dati_per_analisi.get(data_corrente_str, {})
            n_r1 = set(estrazione_corrente.get(ruota1_key) or [])
            n_r2 = set(estrazione_corrente.get(ruota2_key) or [])
            if not n_r1 or not n_r2: continue
            evento_trovato_in_questa_data = False
            for terzina in self.TERZINE_SIMMETRICHE_LIST:
                for c in combinations(terzina, 2):
                    ambo, singolo = set(c), (set(terzina) - set(c)).pop()
                    if (ambo.issubset(n_r1) and singolo in n_r2) or (ambo.issubset(n_r2) and singolo in n_r1):
                        eventi_trovati_raw.append((data_corrente_str, terzina, ruota1_key, ruota2_key)); evento_trovato_in_questa_data = True; break
                if evento_trovato_in_questa_data: break
        
        if not eventi_trovati_raw: return Counter(), [], 0
        
        # --- CORREZIONE APPLICATA QUI ---
        # Coerenza con le altre funzioni: la previsione si basa sul passato.
        lookback_period = 100
        ambate_counter = self._process_events(eventi_trovati_raw, lookback_period, direction="past")
        
        return ambate_counter, eventi_trovati_raw, 0

    def _get_estrazioni_successive(self, evento_recente, num_estrazioni_future):
        data_evento, _, ruota1, ruota2 = evento_recente
        current_date_index = self.date_to_index.get(data_evento, -1)
        if current_date_index == -1: return []
        estrazioni_post_evento = []
        for k in range(1, num_estrazioni_future + 1):
            future_date_index = current_date_index + k
            if future_date_index >= len(self.date_ordinate): break
            data_futura = self.date_ordinate[future_date_index]
            numeri_r1 = self.dati_per_analisi.get(data_futura, {}).get(ruota1)
            numeri_r2 = self.dati_per_analisi.get(data_futura, {}).get(ruota2)
            estrazioni_post_evento.append({"colpo": k, "data": data_futura, "ruota1": {"sigla": ruota1, "numeri": numeri_r1 or []}, "ruota2": {"sigla": ruota2, "numeri": numeri_r2 or []}})
        return estrazioni_post_evento

    def _get_abbinamenti_passati_per_ambata(self, ambata_num, evento_recente, num_estrazioni_da_analizzare):
        abbinamenti_counter = Counter()
        data_evento_str, _, ruota1_key, ruota2_key = evento_recente
        current_date_index = self.date_to_index.get(data_evento_str, -1)
        if current_date_index == -1: return abbinamenti_counter
        for k in range(-1, -num_estrazioni_da_analizzare - 1, -1):
            target_date_index = current_date_index + k
            if target_date_index < 0: break
            data_passata = self.date_ordinate[target_date_index]
            for ruota_monitorata in [ruota1_key, ruota2_key]:
                numeri_passati = self.dati_per_analisi.get(data_passata, {}).get(ruota_monitorata)
                if numeri_passati and ambata_num in numeri_passati:
                    for num in numeri_passati:
                        if num != ambata_num: abbinamenti_counter[num] += 1
        return abbinamenti_counter

    def _find_latest_terzina_event(self, r1_key, r2_key, start_str, end_str):
        for date_str in self._get_events_in_range(start_str, end_str):
            n_r1 = set(self.dati_per_analisi.get(date_str, {}).get(r1_key, []))
            n_r2 = set(self.dati_per_analisi.get(date_str, {}).get(r2_key, []))
            if not n_r1 or not n_r2: continue
            for terzina in self.TERZINE_SIMMETRICHE_LIST:
                for c in combinations(terzina, 2):
                    ambo, singolo = set(c), (set(terzina) - set(c)).pop()
                    if (ambo.issubset(n_r1) and singolo in n_r2) or (ambo.issubset(n_r2) and singolo in n_r1):
                        return (date_str, terzina, r1_key, r2_key)
        return None

    def ricerca_globale_terzine(self, start_date_str, end_date_str, num_estrazioni_future, top_n_ambate, top_n_abbinamenti):
        tutti_i_risultati = []
        coppie_ruote = list(combinations(self.RUOTE_DISPONIBILI.keys(), 2))
        data_ultima_estrazione = self.date_ordinate[-1]
        self._log(f"Avvio ricerca globale su {len(coppie_ruote)} coppie di ruote. Ultima estrazione in archivio: {data_ultima_estrazione}")
        for i, (ruota1_key, ruota2_key) in enumerate(coppie_ruote):
            evento_recente = self._find_latest_terzina_event(ruota1_key, ruota2_key, start_date_str, end_date_str)
            if not evento_recente: continue
            data_evento, terzina_evento, _, _ = evento_recente
            if data_evento == data_ultima_estrazione:
                self._log(f"Trovato evento per PREVISIONE su {ruota1_key}-{ruota2_key} del {data_evento}")
                ambate_counter = self._process_events([evento_recente], 100, direction="past")
                if not ambate_counter: continue
                top_ambate_trovate = ambate_counter.most_common(top_n_ambate)
                caso_da_mostrare = {"tipo": "PREVISIONE", "ruote": f"{ruota1_key}-{ruota2_key}", "data_evento": data_evento, "terzina": "-".join(map(str, terzina_evento)), "previsioni": []}
                for (ambata_num, _), _ in top_ambate_trovate:
                    abbinamenti_counter = self._get_abbinamenti_passati_per_ambata(ambata_num, evento_recente, 100)
                    top_abbinamenti = abbinamenti_counter.most_common(top_n_abbinamenti)
                    caso_da_mostrare["previsioni"].append({"ambata": ambata_num, "abbinamenti": [abb[0] for abb in top_abbinamenti]})
                tutti_i_risultati.append(caso_da_mostrare)
            else:
                estrazioni_successive = self._get_estrazioni_successive(evento_recente, num_estrazioni_future)
                if not estrazioni_successive: continue
                ambate_counter_storico = self._process_events([evento_recente], 100, direction="past")
                if not ambate_counter_storico: continue
                top_ambate_storiche = ambate_counter_storico.most_common(top_n_ambate)
                caso_da_mostrare = {"tipo": "STORICO", "ruote": f"{ruota1_key}-{ruota2_key}", "data_evento": data_evento, "terzina": "-".join(map(str, terzina_evento)), "previsioni_what_if": [], "estrazioni_successive": estrazioni_successive}
                for (ambata_num, _), _ in top_ambate_storiche:
                     abbinamenti_counter_storici = self._get_abbinamenti_passati_per_ambata(ambata_num, evento_recente, 100)
                     top_abbinamenti = abbinamenti_counter_storici.most_common(top_n_abbinamenti)
                     caso_da_mostrare["previsioni_what_if"].append({"ambata": ambata_num, "abbinamenti": [abb[0] for abb in top_abbinamenti]})
                tutti_i_risultati.append(caso_da_mostrare)
        self._log(f"\nRicerca globale completata.")
        return tutti_i_risultati

    def get_top_n_risultati(self, counter, num_events, result_type, ruote_evento_str):
        if not counter or num_events == 0: return []
        presenze_per_numero = Counter()
        for (numero, ruota_key), freq in counter.items():
            presenze_per_numero[numero] += freq
        top_items = presenze_per_numero.most_common(50)
        results_list = []
        label = "Ambata" if result_type == "ambate" else "Abbinamento"
        for numero, freq in top_items:
            media = freq / num_events
            results_list.append({'numero': numero, 'freq': freq, 'media': media, 'label': label, 'ruote_str': ruote_evento_str})
        return results_list
    
    def calculate_ambata_coverage(self, ambate_da_cercare, num_total_events, events_data, num_estrazioni_future):
        if not ambate_da_cercare or num_total_events == 0: return 0.0, 0
        target_items = set(ambate_da_cercare)
        eventi_coperti = set()
        for i, (data_evento_str, _, ruota1_key, ruota2_key) in enumerate(events_data):
            current_date_index = self.date_to_index.get(data_evento_str, -1)
            if current_date_index == -1: continue
            vincita_trovata_per_questo_evento = False
            for k in range(1, num_estrazioni_future + 1):
                future_date_index = current_date_index + k
                if future_date_index >= len(self.date_ordinate): break
                for ruota_monitorata in [ruota1_key, ruota2_key]:
                    numeri_futuri = self.dati_per_analisi.get(self.date_ordinate[future_date_index], {}).get(ruota_monitorata)
                    if numeri_futuri and not set(numeri_futuri).isdisjoint(target_items):
                        eventi_coperti.add(i); vincita_trovata_per_questo_evento = True; break
                if vincita_trovata_per_questo_evento: break
        num_eventi_coperti = len(eventi_coperti)
        percentuale = (num_eventi_coperti / num_total_events * 100) if num_total_events > 0 else 0
        return percentuale, num_eventi_coperti

    def _calculate_abbinamenti_for_ambata(self, ambata_num, events_data, num_estrazioni_future):
        abbinamenti_counter = Counter()
        for data_evento_str, _, ruota1_key, ruota2_key in events_data:
            current_date_index = self.date_to_index.get(data_evento_str, -1)
            if current_date_index == -1: continue
            vincita_trovata_per_questo_evento = False
            for k in range(1, num_estrazioni_future + 1):
                future_date_index = current_date_index + k
                if future_date_index >= len(self.date_ordinate): break
                for ruota_monitorata in [ruota1_key, ruota2_key]:
                    numeri_futuri = self.dati_per_analisi.get(self.date_ordinate[future_date_index], {}).get(ruota_monitorata)
                    if numeri_futuri and ambata_num in numeri_futuri:
                        for num_abb in numeri_futuri:
                            if num_abb != ambata_num: abbinamenti_counter[num_abb] += 1
                        vincita_trovata_per_questo_evento = True; break
                if vincita_trovata_per_questo_evento: break
        final_counter = Counter()
        ruota_da_usare = events_data[0][2]
        for numero, freq in abbinamenti_counter.items():
            final_counter[(numero, ruota_da_usare)] = freq
        return final_counter, 0

    def calculate_ambo_coverage(self, ambata_num, top_abbinamenti_nums, num_total_events, events_data, num_estrazioni_future):
        if not ambata_num or not top_abbinamenti_nums or num_total_events == 0: return 0.0, 0
        ambata_set = {ambata_num}; top_abbinamenti_set = set(top_abbinamenti_nums)
        eventi_coperti = set()
        for i, (data_evento_str, _, ruota1_key, ruota2_key) in enumerate(events_data):
            current_date_index = self.date_to_index.get(data_evento_str, -1)
            if current_date_index == -1: continue
            vincita_trovata_per_questo_evento = False
            for k in range(1, num_estrazioni_future + 1):
                future_date_index = current_date_index + k
                if future_date_index >= len(self.date_ordinate): break
                for ruota_monitorata in [ruota1_key, ruota2_key]:
                    numeri_futuri = self.dati_per_analisi.get(self.date_ordinate[future_date_index], {}).get(ruota_monitorata)
                    if numeri_futuri:
                        numeri_futuri_set = set(numeri_futuri)
                        if not numeri_futuri_set.isdisjoint(ambata_set) and not numeri_futuri_set.isdisjoint(top_abbinamenti_set):
                            eventi_coperti.add(i); vincita_trovata_per_questo_evento = True; break
                if vincita_trovata_per_questo_evento: break
        num_eventi_coperti = len(eventi_coperti)
        percentuale = (num_eventi_coperti / num_total_events * 100) if num_total_events > 0 else 0
        return percentuale, num_eventi_coperti

    def mostra_dettagli_verifica(self, events, num_colpi, ambate_principali, abbinamenti_unici):
        self._log("\n" + "#"*80 + "\n### INIZIO MODALITÀ DI VERIFICA DETTAGLIATA ###")
        self._log(f"Verifica per Ambate Principali: {sorted(list(ambate_principali))}")
        if abbinamenti_unici:
             self._log(f"con Abbinamenti per Ambo/Terno: {sorted(list(abbinamenti_unici))}")
        self._log("#"*80)
        eventi_coperti_dalla_verifica = set()
        
        for i, (data_evento, _, ruota1, ruota2) in enumerate(events):
            self._log(f"\n--- VERIFICA EVENTO {i+1}/{len(events)} (Data: {data_evento}) ---")
            current_date_index = self.date_to_index.get(data_evento, -1)
            if current_date_index == -1: 
                self._log("  ERRORE: Data evento non trovata."); continue

            vincita_trovata = False
            archivio_terminato_presto = False
            colpi_verificati = 0

            for k in range(1, num_colpi + 1):
                future_date_index = current_date_index + k
                if future_date_index >= len(self.date_ordinate):
                    self._log(f"  Colpo {k}: Fine archivio."); archivio_terminato_presto = True; break
                
                colpi_verificati = k
                data_futura = self.date_ordinate[future_date_index]
                n1 = self.dati_per_analisi.get(data_futura, {}).get(ruota1, [])
                n2 = self.dati_per_analisi.get(data_futura, {}).get(ruota2, [])
                numeri_usciti_set = set(n1) | set(n2)
                
                # ### NUOVA LOGICA DI VERIFICA CORRETTA ###
                marker = ""
                # 1. Controlla se è uscita almeno una delle ambate principali
                intersezione_ambate = numeri_usciti_set.intersection(ambate_principali)

                # 2. La vincita scatta SOLO se la condizione sopra è vera
                if intersezione_ambate:
                    vincita_trovata = True
                    eventi_coperti_dalla_verifica.add(i)

                    # 3. Ora calcoliamo la sorte completa, includendo gli abbinamenti
                    intersezione_abbinamenti = numeri_usciti_set.intersection(abbinamenti_unici)
                    numeri_vincenti_totali = intersezione_ambate.union(intersezione_abbinamenti)
                    num_vincenti = len(numeri_vincenti_totali)

                    sorti_map = {1: "AMBATA", 2: "AMBO", 3: "TERNO", 4: "QUATERNA", 5: "CINQUINA"}
                    nome_sorte = sorti_map.get(num_vincenti, f"VINCITA x{num_vincenti}")
                    
                    numeri_vinti_str = sorted(list(numeri_vincenti_totali))
                    marker = f" ---> TROVATO {nome_sorte} {numeri_vinti_str}!"
                
                self._log(f"  Colpo {k} ({data_futura}): {ruota1} {n1} | {ruota2} {n2}{marker}")
                if vincita_trovata: break

            if not vincita_trovata:
                if archivio_terminato_presto:
                    self.output_queue.put(f"  Esito in corso... (Verificati {colpi_verificati}/{num_colpi} colpi)")
                else:
                    self.output_queue.put(f"  Nessuna vincita trovata per questo evento nei {num_colpi} colpi.")

        self._log("\n" + "#"*80)
        self._log(f"RISULTATO VERIFICA: {len(eventi_coperti_dalla_verifica)}/{len(events)} eventi hanno avuto almeno un'uscita.")
        self._log("### FINE MODALITÀ DI VERIFICA ###\n" + "#"*80)

class LottoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NUMERICAL FORMATIONS DIVISION - Created by Max Lotto")
        self.geometry("950x800")
        self.output_queue = queue.Queue()
        self.analizzatore = AnalizzatoreLottoAvanzato(self.output_queue)
        self._create_widgets()
        self.after(100, self._process_queue)

    def _create_widgets(self):
        control_frame = ttk.LabelFrame(self, text="Controlli Analisi", padding="15")
        control_frame.pack(padx=10, pady=10, fill="x")
        source_frame = ttk.Frame(control_frame)
        source_frame.pack(fill="x", pady=2)
        ttk.Label(source_frame, text="Fonte Dati:").pack(side="left", padx=5)
        self.data_source_var = tk.StringVar(value='GitHub')
        ttk.Radiobutton(source_frame, text="GitHub", variable=self.data_source_var, value='GitHub', command=self._toggle_local_path).pack(side="left")
        ttk.Radiobutton(source_frame, text="Locale", variable=self.data_source_var, value='Locale', command=self._toggle_local_path).pack(side="left", padx=5)
        self.local_path_label = ttk.Label(source_frame, text="N/A")
        self.local_path_label.pack(side="left", padx=5)
        self.browse_button = ttk.Button(source_frame, text="Sfoglia...", command=self._select_local_path)
        self.browse_button.pack(side="left")
        self._toggle_local_path()
        analysis_type_frame = ttk.LabelFrame(control_frame, text="Tipo di Analisi", padding=10)
        analysis_type_frame.pack(fill="x", pady=5)
        self.analysis_type_var = tk.StringVar(value='Ambo Diviso')
        ttk.Radiobutton(analysis_type_frame, text="Ambo Diviso", variable=self.analysis_type_var, value='Ambo Diviso', command=self._toggle_analysis_inputs).pack(side="left", padx=5)
        ttk.Radiobutton(analysis_type_frame, text="Terzina (Specifica)", variable=self.analysis_type_var, value='Terzina Specifica', command=self._toggle_analysis_inputs).pack(side="left", padx=5)
        ttk.Radiobutton(analysis_type_frame, text="Terzina (Automatica)", variable=self.analysis_type_var, value='Terzina Automatica', command=self._toggle_analysis_inputs).pack(side="left", padx=5)
        self.global_search_radio = ttk.Radiobutton(analysis_type_frame, text="Ricerca Globale Terzine (Tutte le Ruote)", variable=self.analysis_type_var, value='Ricerca Globale', command=self._toggle_analysis_inputs)
        self.global_search_radio.pack(side="left", padx=5)
        self.input_container = ttk.Frame(control_frame)
        self.input_container.pack(fill="x")
        self.ambo_input_frame = ttk.LabelFrame(self.input_container, text="Dettagli Evento Ambo", padding=10)
        ttk.Label(self.ambo_input_frame, text="Num 1:").pack(side="left")
        self.num1_entry = ttk.Entry(self.ambo_input_frame, width=5, validate="key", validatecommand=(self.register(lambda P: P == "" or (P.isdigit() and 1 <= int(P) <= 90)), '%P'))
        self.num1_entry.pack(side="left", padx=5)
        ttk.Label(self.ambo_input_frame, text="Num 2:").pack(side="left", padx=5)
        self.num2_entry = ttk.Entry(self.ambo_input_frame, width=5, validate="key", validatecommand=(self.register(lambda P: P == "" or (P.isdigit() and 1 <= int(P) <= 90)), '%P'))
        self.num2_entry.pack(side="left", padx=5)
        self.terzina_specifica_input_frame = ttk.LabelFrame(self.input_container, text="Dettagli Evento Terzina", padding=10)
        ttk.Label(self.terzina_specifica_input_frame, text="Terzina:").pack(side="left")
        self.terzina_cb = ttk.Combobox(self.terzina_specifica_input_frame, values=[f"{t[0]}-{t[1]}-{t[2]}" for t in self.analizzatore.TERZINE_SIMMETRICHE_LIST], state="readonly", width=12)
        if self.terzina_cb['values']: self.terzina_cb.set(self.terzina_cb['values'][0])
        self.terzina_cb.pack(side="left", padx=5)
        common_params_frame = ttk.Frame(self.input_container)
        self.ruote_frame = ttk.LabelFrame(common_params_frame, text="Ruote Evento", padding=10)
        self.ruote_frame.pack(side="left", fill="y", padx=(0, 5))
        ttk.Label(self.ruote_frame, text="R1:").pack()
        self.ruota1_cb = ttk.Combobox(self.ruote_frame, values=list(self.analizzatore.RUOTE_DISPONIBILI.keys()), state="readonly", width=4)
        self.ruota1_cb.set('BA'); self.ruota1_cb.pack()
        ttk.Label(self.ruote_frame, text="R2:").pack()
        self.ruota2_cb = ttk.Combobox(self.ruote_frame, values=list(self.analizzatore.RUOTE_DISPONIBILI.keys()), state="readonly", width=4)
        self.ruota2_cb.set('CA'); self.ruota2_cb.pack()
        date_frame = ttk.LabelFrame(common_params_frame, text="Periodo e Opzioni", padding=10)
        date_frame.pack(side="left", fill="both", expand=True)
        ttk.Label(date_frame, text="Dal:").grid(row=0, column=0, padx=2, pady=2)
        self.start_date_entry = DateEntry(date_frame, width=10, date_pattern='yyyy-mm-dd', year=2000, locale='it_IT')
        self.start_date_entry.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(date_frame, text="Al:").grid(row=1, column=0, padx=2, pady=2)
        self.end_date_entry = DateEntry(date_frame, width=10, date_pattern='yyyy-mm-dd', locale='it_IT')
        self.end_date_entry.grid(row=1, column=1, padx=2, pady=2)
        self.max_events_label = ttk.Label(date_frame, text="Max Eventi:")
        self.max_events_label.grid(row=0, column=2, padx=(10,2), pady=2)
        self.max_events_spinbox = ttk.Spinbox(date_frame, from_=1, to_=1000, textvariable=tk.IntVar(value=10), width=5)
        self.max_events_spinbox.grid(row=0, column=3, padx=2, pady=2)
        self.show_verification_var = tk.BooleanVar(value=False)
        verification_check = ttk.Checkbutton(date_frame, text="Mostra Dettagli Verifica", variable=self.show_verification_var)
        verification_check.grid(row=1, column=2, columnspan=2, padx=(10, 2), pady=2, sticky='w')
        prognosi_frame = ttk.LabelFrame(common_params_frame, text="Prognosi", padding=10)
        prognosi_frame.pack(side="left", fill="y", padx=(5, 0))
        ttk.Label(prognosi_frame, text="Colpi:").pack()
        self.num_future_est = tk.IntVar(value=9)
        ttk.Spinbox(prognosi_frame, from_=1, to_=100, textvariable=self.num_future_est, width=5).pack()
        self.top_n_ambate_label = ttk.Label(prognosi_frame, text="Top Ambate:")
        self.top_n_ambate_label.pack()
        self.top_n_ambate = tk.IntVar(value=1)
        self.top_n_ambate_spinbox = ttk.Spinbox(prognosi_frame, from_=1, to_=10, textvariable=self.top_n_ambate, width=5)
        self.top_n_ambate_spinbox.pack()
        self.top_n_abbinamenti_label = ttk.Label(prognosi_frame, text="Top Abb.:")
        self.top_n_abbinamenti_label.pack()
        self.top_n_abbinamenti = tk.IntVar(value=3)
        self.top_n_abbinamenti_spinbox = ttk.Spinbox(prognosi_frame, from_=1, to_=10, textvariable=self.top_n_abbinamenti, width=5)
        self.top_n_abbinamenti_spinbox.pack()
        common_params_frame.pack(fill="x", pady=5)
        self._toggle_analysis_inputs()
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=5)
        ttk.Button(button_frame, text="1. Inizializza Archivio", command=self._run_initialize_archive).pack(side="left", padx=5, expand=True, fill="x")
        self.run_button = ttk.Button(button_frame, text="2. Avvia Analisi", command=self._run_analysis_based_on_type)
        self.run_button.pack(side="left", padx=5, expand=True, fill="x")
        ttk.Button(button_frame, text="Pulisci Output", command=self._clear_output).pack(side="left", padx=5, expand=True, fill="x")
        self.output_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, state='disabled', height=25, font=('Consolas', 10))
        self.output_text.pack(padx=10, pady=10, fill="both", expand=True)

    def _toggle_analysis_inputs(self):
        self.ambo_input_frame.pack_forget(); self.terzina_specifica_input_frame.pack_forget()
        analysis_type = self.analysis_type_var.get(); is_global = analysis_type == 'Ricerca Globale'
        new_ruote_state = 'disabled' if is_global else 'readonly'; self.ruota1_cb.config(state=new_ruote_state); self.ruota2_cb.config(state=new_ruote_state)
        new_max_events_state = 'disabled' if is_global else 'normal'; self.max_events_spinbox.config(state=new_max_events_state)
        self.top_n_ambate_spinbox.config(state='normal'); self.top_n_abbinamenti_spinbox.config(state='normal')
        if is_global: self.max_events_spinbox.set(1)
        if analysis_type == 'Ambo Diviso': self.ambo_input_frame.pack(fill="x", pady=5)
        elif analysis_type == 'Terzina Specifica': self.terzina_specifica_input_frame.pack(fill="x", pady=5)

    def _select_local_path(self):
        folder = filedialog.askdirectory()
        if folder: self.analizzatore.local_path = folder; self.local_path_label.config(text=os.path.basename(folder))

    def _toggle_local_path(self):
        is_local = self.data_source_var.get() == 'Locale'; self.local_path_label.config(state='normal' if is_local else 'disabled')
        self.browse_button.config(state='normal' if is_local else 'disabled'); self.analizzatore.data_source = self.data_source_var.get()

    def _update_output_text(self, message):
        self.output_text.config(state='normal'); self.output_text.insert(tk.END, message + "\n"); self.output_text.see(tk.END); self.output_text.config(state='disabled')

    def _process_queue(self):
        try:
            while True: self._update_output_text(self.output_queue.get_nowait())
        except queue.Empty: pass
        finally: self.after(100, self._process_queue)

    def _run_initialize_archive(self):
        self._clear_output(); threading.Thread(target=self._initialize_archive_task, daemon=True).start()

    def _initialize_archive_task(self):
        try: self.analizzatore.inizializza_archivio()
        except Exception as e: self.output_queue.put(f"ERRORE: {e}"); traceback.print_exc()

    def _run_analysis_based_on_type(self):
        if not self.analizzatore.estrazioni_per_ruota:
            messagebox.showwarning("Dati Mancanti", "Inizializza l'archivio."); return
        try:
            params = {'start_date_str': self.start_date_entry.get_date().strftime('%Y-%m-%d'), 'end_date_str': self.end_date_entry.get_date().strftime('%Y-%m-%d'), 'num_future': self.num_future_est.get(), 'show_verification': self.show_verification_var.get()}
            analysis_type = self.analysis_type_var.get(); self._clear_output()
            if analysis_type == 'Ricerca Globale':
                params.update({'top_n_ambate': self.top_n_ambate.get(), 'top_n_abbinamenti': self.top_n_abbinamenti.get()})
                threading.Thread(target=self._analyze_global_task, args=(params,), daemon=True).start()
            else:
                params.update({'ruota1_key': self.ruota1_cb.get(), 'ruota2_key': self.ruota2_cb.get(), 'max_events': int(self.max_events_spinbox.get()), 'top_n_ambate': self.top_n_ambate.get(), 'top_n_abbinamenti': self.top_n_abbinamenti.get()})
                if params['ruota1_key'] == params['ruota2_key']: messagebox.showwarning("Errore", "Le ruote devono essere diverse."); return
                if analysis_type == 'Ambo Diviso': params['num1'], params['num2'] = int(self.num1_entry.get()), int(self.num2_entry.get())
                elif analysis_type == 'Terzina Specifica': params['terzina_str'] = self.terzina_cb.get()
                threading.Thread(target=self._analyze_event_task, args=(analysis_type, params), daemon=True).start()
        except (ValueError, AttributeError) as e: messagebox.showwarning("Errore Input", f"Controlla i valori inseriti: {e}")

    def _analyze_global_task(self, params):
        try:
            risultati = self.analizzatore.ricerca_globale_terzine(params['start_date_str'], params['end_date_str'], params['num_future'], params['top_n_ambate'], params['top_n_abbinamenti'])
            previsioni_nuove = [r for r in risultati if r['tipo'] == 'PREVISIONE']; storico = [r for r in risultati if r['tipo'] == 'STORICO']
            if previsioni_nuove:
                self.output_queue.put("\n" + "=" * 80 + "\nNUOVE PREVISIONI DA GIOCARE (EVENTI DALL'ULTIMA ESTRAZIONE)\n" + "=" * 80)
                for res in sorted(previsioni_nuove, key=lambda x: x['data_evento'], reverse=True):
                    self.output_queue.put(f"\nEVENTO SU {res['ruote']} del {res['data_evento']} (Terzina: {res['terzina']})")
                    self.output_queue.put("Previsione calcolata su storico, da giocare per i prossimi colpi:")
                    for previsione in res['previsioni']:
                        self.output_queue.put(f"  -> Ambata: {previsione['ambata']}"); self.output_queue.put(f"     Abbinamenti per Ambo: {previsione['abbinamenti']}")
            if storico:
                self.output_queue.put("\n" + "=" * 80 + "\nANALISI STORICA (BACKTESTING ON-THE-FLY)\n" + "=" * 80)
                for res in sorted(storico, key=lambda x: x['data_evento'], reverse=True):
                    self.output_queue.put(f"\nEVENTO SU {res['ruote']} del {res['data_evento']} (Terzina: {res['terzina']})")
                    self.output_queue.put("\nPREVISIONI CHE SAREBBERO STATE GENERATE:")
                    for previsione in res['previsioni_what_if']:
                        self.output_queue.put(f"  -> Ambata: {previsione['ambata']}"); self.output_queue.put(f"     Abbinamenti per Ambo: {previsione['abbinamenti']}")
                    self.output_queue.put("\nESITI NEI COLPI SUCCESSIVI:")
                    if not res['estrazioni_successive']: self.output_queue.put("  Nessuna estrazione successiva trovata nell'archivio.")
                    for estrazione in res['estrazioni_successive']:
                        r1, r2 = estrazione['ruota1'], estrazione['ruota2']; hits = []
                        for previsione in res['previsioni_what_if']:
                            ambata, abbinamenti = previsione['ambata'], set(previsione['abbinamenti'])
                            for ruota_info in [r1, r2]:
                                numeri_set = set(ruota_info['numeri'])
                                if ambata in numeri_set:
                                    abbinamenti_vincenti = numeri_set.intersection(abbinamenti)
                                    if abbinamenti_vincenti: hits.append(f"AMBO {ambata}-{list(abbinamenti_vincenti)[0]} su {ruota_info['sigla']}!")
                                    else: hits.append(f"AMBATA {ambata} su {ruota_info['sigla']}")
                        hit_str = " | VINCITE: " + ", ".join(hits) if hits else ""
                        self.output_queue.put(f"  Colpo {estrazione['colpo']:<2} ({estrazione['data']}): {r1['sigla']} {r1['numeri']} | {r2['sigla']} {r2['numeri']}{hit_str}")
            if not previsioni_nuove and not storico: self.output_queue.put("\nNessun evento di terzina divisa trovato nel periodo specificato.")
        except Exception as e: self.output_queue.put(f"ERRORE: {e}"); traceback.print_exc()

    def _analyze_event_task(self, analysis_type, params):
        try:
            event_display, ambate_counter, events, total_opp = "", Counter(), [], 0
            base_args = { 'ruota1_key': params['ruota1_key'], 'ruota2_key': params['ruota2_key'], 'start_date_str': params['start_date_str'], 'end_date_str': params['end_date_str'], 'max_events_to_analyze': params['max_events'], 'num_estrazioni_future': params['num_future'] }
            
            if analysis_type == 'Ambo Diviso':
                event_display = f"Ambo ({params['num1']}-{params['num2']})"; specific_args = {'num1': params['num1'], 'num2': params['num2']}
                ambate_counter, events, total_opp = self.analizzatore.analizza_multipli_eventi_ambo_diviso(**base_args, **specific_args)
            elif analysis_type == 'Terzina Specifica':
                event_display = f"Terzina ({params['terzina_str']})"; specific_args = {'terzina_str': params['terzina_str']}
                ambate_counter, events, total_opp = self.analizzatore.analizza_terzina_simmetrica_divisa_specifica(**base_args, **specific_args)
            else: # Terzina Automatica
                ambate_counter, events, total_opp = self.analizzatore.analizza_terzina_simmetrica_divisa_automatica(**base_args)
                
                # ### NUOVA LOGICA DI VISUALIZZAZIONE MIGLIORATA ###
                event_display = "Terzina Automatica"
                if events:
                    # Conta quante volte ogni terzina è apparsa negli eventi trovati
                    terzina_counts = Counter(evento[1] for evento in events)
                    # Crea una stringa di dettaglio
                    dettaglio_parts = []
                    for terzina_tuple, count in terzina_counts.items():
                        terzina_str = "-".join(map(str, terzina_tuple))
                        dettaglio_parts.append(f"{terzina_str} ({count} {'volta' if count == 1 else 'volte'})")
                    dettaglio_str = ", ".join(dettaglio_parts)
                    event_display = f"Terzina Automatica (Dettaglio: {dettaglio_str})"
            
            num_events = len(events)
            if num_events == 0: self.output_queue.put(f"\nNessun evento per {event_display} trovato nel periodo specificato."); return
            
            self.output_queue.put(f"\nAnalizzate {num_events} occorrenze per: {event_display} su {params['ruota1_key']}-{params['ruota2_key']}")
            ruote_evento_str = f"{params['ruota1_key']}-{params['ruota2_key']}"
            top_ambate_data = self.analizzatore.get_top_n_risultati(ambate_counter, num_events, "ambate", ruote_evento_str)
            if not top_ambate_data: self.output_queue.put("\nNessuna ambata significativa trovata."); return
            
            self.output_queue.put("\n--- RISULTATI PROGNOSI ---")
            top_n_ambate_to_show = params['top_n_ambate']; top_ambate_shown = top_ambate_data[:top_n_ambate_to_show]
            self.output_queue.put("Riepilogo Migliori Ambate:\n" + "\n".join([f"{d['label']}: {d['numero']:<2} su {d['ruote_str']} (Presenze: {d['freq']}, Media: {d['media']:.2f}/evento)" for d in top_ambate_shown]))

            top_ambate_nums_generali = [d['numero'] for d in top_ambate_shown]
            pct_amb, cov_amb = self.analizzatore.calculate_ambata_coverage(top_ambate_nums_generali, num_events, events, params['num_future'])
            self.output_queue.put(f"COPERTURA AMBATA (Top {len(top_ambate_shown)}): {cov_amb}/{num_events} eventi coperti ({pct_amb:.2f}%)")

            self.output_queue.put("\n--- DETTAGLIO ABBINAMENTI PER AMBO ---")
            previsioni_complete = []
            for ambata_data in top_ambate_shown:
                ambata_num = ambata_data['numero']
                abbinamenti_counter, _ = self.analizzatore._calculate_abbinamenti_for_ambata(ambata_num, events, params['num_future'])
                top_abbinamenti_data = self.analizzatore.get_top_n_risultati(abbinamenti_counter, num_events, "abbinamento", ruote_evento_str)
                top_abbinamenti_to_show = top_abbinamenti_data[:params['top_n_abbinamenti']]
                self.output_queue.put(f"\n-> Per Ambata {ambata_num}:")
                if not top_abbinamenti_to_show: self.output_queue.put("   Nessun abbinamento significativo trovato."); continue
                self.output_queue.put("   Migliori Abbinamenti:\n" + "\n".join([f"   {d['label']}: {d['numero']:<2} su {d['ruote_str']} (Presenze: {d['freq']}, Media: {d['media']:.2f}/evento)" for d in top_abbinamenti_to_show]))
                top_abbinamenti_nums = [d['numero'] for d in top_abbinamenti_to_show]
                pct_ambo, cov_ambo = self.analizzatore.calculate_ambo_coverage(ambata_num, top_abbinamenti_nums, num_events, events, params['num_future'])
                self.output_queue.put(f"   COPERTURA AMBO (singola): {cov_ambo}/{num_events} eventi coperti ({pct_ambo:.2f}%)")
                previsioni_complete.append({'ambata': ambata_num, 'abbinamenti': set(top_abbinamenti_nums)})

            lunghetta = set()
            if previsioni_complete:
                for previsione in previsioni_complete:
                    lunghetta.add(previsione['ambata'])
                    lunghetta.update(previsione['abbinamenti'])
            
            if params.get('show_verification', False) and lunghetta:
                ambate_principali = {d['numero'] for d in top_ambate_shown}
                abbinamenti_unici = lunghetta - ambate_principali
                self.analizzatore.mostra_dettagli_verifica(events, params['num_future'], ambate_principali, abbinamenti_unici)
            
            if previsioni_complete:
                eventi_coperti_in_totale = set()
                for i in range(num_events):
                    data_evento_str, _, ruota1_key, ruota2_key = events[i]
                    current_date_index = self.analizzatore.date_to_index.get(data_evento_str, -1)
                    if current_date_index == -1: continue
                    vincita_trovata_per_questo_evento = False
                    for k in range(1, params['num_future'] + 1):
                        future_date_index = current_date_index + k
                        if future_date_index >= len(self.analizzatore.date_ordinate): break
                        for ruota_monitorata in [ruota1_key, ruota2_key]:
                            numeri_futuri = self.analizzatore.dati_per_analisi.get(self.analizzatore.date_ordinate[future_date_index], {}).get(ruota_monitorata)
                            if numeri_futuri:
                                numeri_futuri_set = set(numeri_futuri)
                                for previsione in previsioni_complete:
                                    if previsione['ambata'] in numeri_futuri_set and not numeri_futuri_set.isdisjoint(previsione['abbinamenti']):
                                        eventi_coperti_in_totale.add(i); vincita_trovata_per_questo_evento = True; break
                            if vincita_trovata_per_questo_evento: break
                        if vincita_trovata_per_questo_evento: break
                num_eventi_coperti_totale = len(eventi_coperti_in_totale)
                pct_totale = (num_eventi_coperti_totale / num_events * 100) if num_events > 0 else 0
                self.output_queue.put("\n" + "="*50 + "\nCOPERTURA AMBO DELLA STRATEGIA COMPLESSIVA")
                self.output_queue.put(f"Giocando le previsioni separate (Ambo secco), si sarebbero coperti:")
                self.output_queue.put(f"{num_eventi_coperti_totale}/{num_events} eventi ({pct_totale:.2f}%)\n" + "="*50)
            
            if lunghetta:
                self.output_queue.put("\n" + "="*50 + "\nANALISI DELLA LUNGHETTA UNICA")
                self.output_queue.put(f"Giocando tutti i {len(lunghetta)} numeri insieme: {sorted(list(lunghetta))}")
                vincite_lunghetta = Counter(); eventi_con_vincita_lunghetta = set()
                for i in range(num_events):
                    data_evento_str, _, ruota1_key, ruota2_key = events[i]
                    current_date_index = self.analizzatore.date_to_index.get(data_evento_str, -1)
                    if current_date_index == -1: continue
                    vincita_trovata_per_questo_evento = False
                    for k in range(1, params['num_future'] + 1):
                        future_date_index = current_date_index + k
                        if future_date_index >= len(self.analizzatore.date_ordinate): break
                        for ruota_monitorata in [ruota1_key, ruota2_key]:
                            numeri_futuri = self.analizzatore.dati_per_analisi.get(self.analizzatore.date_ordinate[future_date_index], {}).get(ruota_monitorata)
                            if numeri_futuri:
                                numeri_vincenti = set(numeri_futuri).intersection(lunghetta)
                                num_vincenti = len(numeri_vincenti)
                                if num_vincenti >= 2: vincite_lunghetta['Ambi'] += 1
                                if num_vincenti >= 3: vincite_lunghetta['Terni'] += 1
                                if num_vincenti >= 4: vincite_lunghetta['Quaterne'] += 1
                                if num_vincenti >= 5: vincite_lunghetta['Cinquine'] += 1
                                if num_vincenti >= 2: eventi_con_vincita_lunghetta.add(i); vincita_trovata_per_questo_evento = True; break
                        if vincita_trovata_per_questo_evento: break
                num_eventi_coperti_lunghetta = len(eventi_con_vincita_lunghetta)
                pct_lunghetta = (num_eventi_coperti_lunghetta / num_events * 100) if num_events > 0 else 0
                self.output_queue.put(f"Copertura Eventi (con almeno Ambo): {num_eventi_coperti_lunghetta}/{num_events} eventi ({pct_lunghetta:.2f}%)")
                self.output_queue.put("Riepilogo Esiti Ottenuti (su tutti i colpi):")
                if vincite_lunghetta:
                    esiti_ordinati = sorted(vincite_lunghetta.items(), key=lambda item: ['Ambi', 'Terni', 'Quaterne', 'Cinquine'].index(item[0]))
                    for esito, quantita in esiti_ordinati: self.output_queue.put(f" - {esito}: {quantita}")
                else: self.output_queue.put(" - Nessun esito positivo.")
                self.output_queue.put("="*50)
            self.output_queue.put(f"\nAnalisi completata.")
        except Exception as e: self.output_queue.put(f"ERRORE: {e}"); traceback.print_exc()

    def _clear_output(self):
        self.output_text.config(state='normal'); self.output_text.delete(1.0, tk.END); self.output_text.config(state='disabled')

if __name__ == "__main__":
    app = LottoApp()
    app.mainloop()