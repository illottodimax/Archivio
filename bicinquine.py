import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import requests
from datetime import datetime, date, timedelta
from collections import Counter
from itertools import combinations
import threading
import queue
import traceback

try:
    from tkcalendar import DateEntry
except ImportError:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Libreria Mancante", "La libreria 'tkcalendar' non è installata.\n\nPer favore, installala eseguendo questo comando nel tuo terminale:\n\npip install tkcalendar")
    exit()

class ArchivioLotto:
    def __init__(self, status_queue):
        self.status_queue = status_queue; self.estrazioni_per_ruota = {}; self.dati_per_analisi = {}; self.date_ordinate = []
        self.GITHUB_USER = "illottodimax"; self.GITHUB_REPO = "Archivio"; self.GITHUB_BRANCH = "main"
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
        self.URL_RUOTE = {k: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{v.upper()}.txt' for k, v in self.RUOTE_DISPONIBILI.items()}
    def _log_status(self, message): self.status_queue.put(message)
    def inizializza(self, data_source, local_path):
        self._log_status("Inizio caricamento archivio..."); total_ruote = len(self.RUOTE_DISPONIBILI)
        for i, (ruota_key, ruota_nome) in enumerate(self.RUOTE_DISPONIBILI.items()):
            self._log_status(f"Caricamento {ruota_nome} ({i+1}/{total_ruote})..."); self.estrazioni_per_ruota[ruota_key] = self._carica_singola_ruota(data_source, local_path, ruota_key, ruota_nome)
        self._prepara_dati_per_analisi(); self._log_status("Archivio pronto.")
    def _carica_singola_ruota(self, data_source, local_path, ruota_key, ruota_nome):
        try:
            if data_source == 'GitHub':
                response = requests.get(self.URL_RUOTE[ruota_key], timeout=15); response.raise_for_status(); linee = response.text.strip().split('\n')
            else:
                if not local_path or not os.path.isdir(local_path): raise FileNotFoundError("Percorso locale non valido o non impostato.")
                with open(os.path.join(local_path, f"{ruota_nome.upper()}.txt"), 'r', encoding='utf-8') as f: linee = f.readlines()
            return self._parse_estrazioni(linee)
        except Exception as e: raise RuntimeError(f"Impossibile caricare {ruota_nome}: {e}")
    def _parse_estrazioni(self, linee):
        parsed_data = []
        for l in linee:
            parts = l.strip().split()
            if len(parts) >= 7:
                try: parsed_data.append({'data': datetime.strptime(parts[0], '%Y/%m/%d').date(), 'numeri': [int(n) for n in parts[2:7] if n.isdigit()]})
                except (ValueError, IndexError): pass
        return parsed_data
    def _prepara_dati_per_analisi(self):
        self._log_status("Allineamento dati...")
        tutte_le_date = sorted({e['data'] for estrazioni in self.estrazioni_per_ruota.values() for e in estrazioni})
        self.date_ordinate = tutte_le_date; self.dati_per_analisi = {data: {ruota: None for ruota in self.RUOTE_DISPONIBILI} for data in self.date_ordinate}
        for ruota, estrazioni in self.estrazioni_per_ruota.items():
            for e in estrazioni:
                if e['data'] in self.dati_per_analisi: self.dati_per_analisi[e['data']][ruota] = e['numeri']

class BicinquineAnalyzer:
    def __init__(self, archivio, status_queue):
        self.archivio = archivio; self.status_queue = status_queue

    def _log_status(self, message): self.status_queue.put(message)

    def _genera_cinquine_simmetriche(self) -> list:
        cinquine = [];
        for i in range(22):
            cinquina = [i + 1, 2 * (i + 1), 45 - i, 46 + i, 90 - i]
            if all(1 <= num <= 90 for num in cinquina) and len(set(cinquina)) == 5:
                cinquine.append(sorted(cinquina))
        return cinquine

    def _genera_cinquine_pentagonali(self) -> list:
        cinquine = [];
        for i in range(1, 19):
            cinquina = [(i + 18 * k -1) % 90 + 1 for k in range(5)]
            if len(set(cinquina)) == 5:
                cinquine.append(sorted(cinquina))
        return cinquine

    def _genera_cinquine_di_cadenza(self) -> list:
        cinquine = [];
        for i in range(1, 11):
            cadenza = i % 10
            base_cinquina = [n for n in range(1, 91) if n % 10 == cadenza]
            if len(base_cinquina) >= 5:
                cinquine.extend([sorted(list(c)) for c in combinations(base_cinquina, 5)][:5])
        return cinquine

    def _genera_cinquine_progressione(self) -> list:
        cinquine = [];
        for ragione in range(3, 13):
            for start in range(1, 91):
                cinquina = [start + i * ragione for i in range(5)]
                if all(1 <= n <= 90 for n in cinquina) and len(set(cinquina)) == 5:
                    cinquine.append(sorted(cinquina))
        return [list(c) for c in set(tuple(c) for c in cinquine)]

    def _genera_cinquine_gemelli(self) -> list:
        gemelli = [11, 22, 33, 44, 55, 66, 77, 88]
        return [sorted(list(c)) for c in combinations(gemelli, 5)]
        
    def _genera_cinquine_vertibili(self) -> list:
        coppie_vertibili_set = set()
        for n in range(10, 90):
            if n % 10 == 0 or n % 11 == 0: continue
            v = 0
            if n % 10 == 9: v = (n // 10) * 11
            else:
                v_temp = int(str(n)[::-1])
                if v_temp <= 90: v = v_temp
            if v > 0: coppie_vertibili_set.add(tuple(sorted((n, v))))
        vertibili = sorted(list(coppie_vertibili_set))
        cinquine = []
        for coppia1, coppia2 in combinations(vertibili, 2):
            for jolly in [11, 33, 55, 77, 90]:
                cinquina = sorted(list(coppia1) + list(coppia2) + [jolly])
                if len(set(cinquina)) == 5: cinquine.append(cinquina)
        return [list(c) for c in set(tuple(c) for c in cinquine)][:100]

    def _genera_cinquine_da_figure(self) -> list:
        cinquine = [];
        for i in range(1, 10):
            figura = [(i + 9 * k - 1) % 90 + 1 for k in range(10)]
            if len(figura) >= 5: cinquine.extend([sorted(list(c)) for c in combinations(figura, 5)][:10])
        return [list(c) for c in set(tuple(c) for c in cinquine)]

    def _genera_cinquine_selezionate(self, tipo_cinquina: str) -> list:
        if tipo_cinquina.startswith("Simmetriche"): return self._genera_cinquine_simmetriche()
        elif tipo_cinquina.startswith("Pentagonali"): return self._genera_cinquine_pentagonali()
        elif tipo_cinquina.startswith("Di Cadenza"): return self._genera_cinquine_di_cadenza()
        elif tipo_cinquina.startswith("Famiglie Matematiche"):
            self._log_status("Generazione Famiglie Matematiche (Progressioni, Gemelli...).")
            tutte_le_cinquine = self._genera_cinquine_progressione() + self._genera_cinquine_gemelli() + self._genera_cinquine_vertibili() + self._genera_cinquine_da_figure()
            return [list(c) for c in set(tuple(c) for c in tutte_le_cinquine)]
        else: return self._genera_cinquine_simmetriche()

    def _crea_coppie_bicinquine(self, cinquine: list) -> list:
        coppie = []
        for i in range(0, len(cinquine) - 1, 2):
            if i + 1 < len(cinquine):
                coppie.append((cinquine[i], cinquine[i + 1]))
        return coppie

    def analizza(self, modalita, tipo_ricerca, ruote_selezionate, data_inizio, data_fine, max_colpi, tipo_cinquina, tipi_ricerca_etichetta):
        self._log_status(f"Fase 1: Generazione cinquine tipo '{tipo_cinquina}'...")
        cinquine_generate = self._genera_cinquine_selezionate(tipo_cinquina)
        coppie_da_testare = self._crea_coppie_bicinquine(cinquine_generate)
        
        if not coppie_da_testare: raise ValueError(f"Nessuna coppia di cinquine generata per il tipo '{tipo_cinquina}'.")

        self._log_status(f"Generate {len(cinquine_generate)} cinquine / {len(coppie_da_testare)} coppie. Inizio analisi...")
        risultati = []
        if modalita == 'Previsione':
            for i, coppia in enumerate(coppie_da_testare):
                self.status_queue.put(f"Fase 2: Calcolo ritardo per '{tipo_ricerca}' coppia {i+1}/{len(coppie_da_testare)}...")
                ritardo = self._calcola_ritardo_reale_coppia(coppia, tipo_ricerca, ruote_selezionate, data_fine)
                risultati.append({"coppia": coppia, "ritardo": ritardo})
            risultati_ordinati = sorted(risultati, key=lambda x: x['ritardo'], reverse=True)
            return self._crea_report_previsione(risultati_ordinati, tipi_ricerca_etichetta, data_fine, tipo_cinquina), risultati_ordinati
        else:
            for i, coppia in enumerate(coppie_da_testare):
                self.status_queue.put(f"Fase 2: Analisi storica per '{tipo_ricerca}' coppia {i+1}/{len(coppie_da_testare)}...")
                res_storico = self._esegui_calcolo_storico(coppia, tipo_ricerca, ruote_selezionate, [(d, self.archivio.dati_per_analisi[d]) for d in self.archivio.date_ordinate if data_inizio <= d <= data_fine], max_colpi)
                tot_cicli = res_storico['vinti'] + res_storico['persi']
                perc_successo = (res_storico['vinti'] / tot_cicli * 100) if tot_cicli > 0 else 0
                risultati.append({"coppia": coppia, **res_storico, "perc_successo": perc_successo})
            risultati_ordinati = sorted(risultati, key=lambda x: x['perc_successo'], reverse=True)
            return self._crea_report_classifica(risultati_ordinati, tipo_ricerca, data_inizio, data_fine, max_colpi, tipo_cinquina)

    def _crea_report_classifica(self, report_ordinato, tipo_ricerca, data_inizio, data_fine, max_colpi, tipo_cinquina):
        periodo = f"{data_inizio.strftime('%d/%m/%Y')} al {data_fine.strftime('%d/%m/%Y')}"
        report = f"CLASSIFICA STORICA - TIPO: {tipo_cinquina}\n"
        report += f"Periodo: {periodo} | Colpi max: {max_colpi} | Ricerca per: {tipo_ricerca}\n"; report += "="*75 + "\n\n"
        for i, res in enumerate(report_ordinato):
            c1_str, c2_str = ','.join(map(str, res['coppia'][0])), ','.join(map(str, res['coppia'][1])); tot_cicli = res['vinti'] + res['persi']
            report += f"#{i+1:02d}) Coppia: [{c1_str}] - [{c2_str}]\n"; report += f"    Percentuale di Successo: {res['perc_successo']:.2f}%\n"
            report += f"    Esiti: {res['vinti']} Vinti / {res['persi']} Persi (su {tot_cicli} cicli)\n"; report += "-"*55 + "\n"
        return report

    def _crea_report_previsione(self, report_ordinato, tipi_ricerca_etichetta, data_fine, tipo_cinquina):
        report = f"PREVISIONE GENERATA ALLA DATA: {data_fine.strftime('%d/%m/%Y')}\n"
        report += f"BASATA SU CINQUINE DI TIPO: {tipo_cinquina}\n"
        testo_ricerca = tipi_ricerca_etichetta if isinstance(tipi_ricerca_etichetta, str) else ', '.join(tipi_ricerca_etichetta)
        report += f"RICERCA PER: {testo_ricerca}\n\n"
        report += "Le coppie in cima sono quelle che non danno un esito da più tempo (ritardo reale).\n\n"
        for i, res in enumerate(report_ordinato[:50]):
            c1_str, c2_str = ','.join(map(str, res['coppia'][0])), ','.join(map(str, res['coppia'][1]))
            report += f"#{i+1:02d}) Coppia: [{c1_str}] - [{c2_str}]\n"; report += f"    Ritardo Reale: {res['ritardo']} estrazioni\n"; report += "-"*55 + "\n"
        return report

    def _get_ambetto_set(self, numbers_set: set) -> set:
        ambetto_set = set()
        for num in numbers_set:
            predecessore = 90 if num == 1 else num - 1; successore = 1 if num == 90 else num + 1
            ambetto_set.update([num, predecessore, successore])
        return ambetto_set
    
    def _calcola_ritardo_reale_coppia(self, coppia, tipo_ricerca, ruote_selezionate, data_fine):
        numeri_in_gioco = set(coppia[0] + coppia[1])
        
        ### NUOVO: Soglia strategica per lo "sfaldamento" della previsione ###
        # Se escono 4 o più ambate dalla bicinquina, consideriamo il colpo a segno
        # anche se non si è formato un ambo. Questo evita di tenere in vita
        # previsioni i cui numeri stanno già uscendo singolarmente.
        # Puoi modificare questo valore (es. 3 o 5) per cambiare la sensibilità.
        SOGLIA_AMBATE_PER_RESET = 4

        if tipo_ricerca == "Ambo":
            elementi_da_cercare = {tuple(sorted(c)) for c in combinations(numeri_in_gioco, 2)}
        else: # Ambata o Ambetto
            elementi_da_cercare = self._get_ambetto_set(numeri_in_gioco) if tipo_ricerca == "Ambetto" else numeri_in_gioco
            
        colpi_validi = [d for d in self.archivio.date_ordinate if d <= data_fine and any(self.archivio.dati_per_analisi.get(d, {}).get(r) for r in ruote_selezionate)]
        
        for i, data_colpo in enumerate(reversed(colpi_validi)):
            numeri_estratti_nel_colpo = set()
            for ruota_key in ruote_selezionate:
                if numeri_ruota := self.archivio.dati_per_analisi.get(data_colpo, {}).get(ruota_key):
                    numeri_estratti_nel_colpo.update(numeri_ruota)
            
            hit_trovato = False
            
            ### MODIFICATO: Logica di ricerca per Ambo potenziata ###
            if tipo_ricerca == "Ambo":
                # Condizione 1 (Primaria): Si è verificato un ambo?
                ambi_estratti = {tuple(sorted(c)) for c in combinations(numeri_estratti_nel_colpo, 2)}
                if not elementi_da_cercare.isdisjoint(ambi_estratti):
                    hit_trovato = True
                
                # Condizione 2 (Secondaria): Si è verificato uno "sfaldamento"?
                if not hit_trovato:
                    ambate_vincenti = numeri_in_gioco.intersection(numeri_estratti_nel_colpo)
                    if len(ambate_vincenti) >= SOGLIA_AMBATE_PER_RESET:
                        hit_trovato = True
            else: # Logica per Ambata e Ambetto (invariata)
                if not elementi_da_cercare.isdisjoint(numeri_estratti_nel_colpo):
                    hit_trovato = True

            if hit_trovato:
                return i # Ritardo trovato, esci dal ciclo
                
        # Se il ciclo finisce senza trovare nessun hit
        return len(colpi_validi)

    def _esegui_calcolo_storico(self, coppia, tipo_ricerca, ruote_selezionate, estrazioni_periodo, max_colpi):
        numeri_in_gioco = set(coppia[0] + coppia[1]); vinti, persi, in_gioco, colpi = 0, 0, False, 0
        if tipo_ricerca == "Ambo": ambi_in_gioco = {tuple(sorted(c)) for c in combinations(numeri_in_gioco, 2)}
        else: numeri_da_cercare_ambata_o_ambetto = self._get_ambetto_set(numeri_in_gioco) if tipo_ricerca == "Ambetto" else numeri_in_gioco
        for data_corrente, estrazioni_giorno in estrazioni_periodo:
            if not in_gioco: in_gioco = True; colpi = 0
            if in_gioco:
                colpi += 1; hit_found = False
                for ruota_key in ruote_selezionate:
                    if estrazioni_giorno[ruota_key]:
                        estrazione = set(estrazioni_giorno[ruota_key])
                        if tipo_ricerca == "Ambo":
                            if not ambi_in_gioco.isdisjoint({tuple(sorted(c)) for c in combinations(estrazione, 2)}): hit_found = True
                        elif not numeri_da_cercare_ambata_o_ambetto.isdisjoint(estrazione): hit_found = True
                        if hit_found: break
                if hit_found: vinti += 1; in_gioco = False
                elif colpi >= max_colpi: persi += 1; in_gioco = False
        return {"vinti": vinti, "persi": persi}

    def calcola_ritardo_numeri(self, numeri_da_controllare, ruote_selezionate, data_fine):
        numeri_da_trovare = set(numeri_da_controllare); ritardi = {num: -1 for num in numeri_da_trovare}
        colpi_validi = [d for d in self.archivio.date_ordinate if d <= data_fine and any(self.archivio.dati_per_analisi.get(d, {}).get(r) for r in ruote_selezionate)]
        for i, data in enumerate(reversed(colpi_validi)):
            if not numeri_da_trovare: break
            numeri_estratti_giorno = set()
            for ruota_key in ruote_selezionate:
                if estrazione := self.archivio.dati_per_analisi[data].get(ruota_key):
                    numeri_estratti_giorno.update(estrazione)
            for num in list(numeri_da_trovare):
                if num in numeri_estratti_giorno: ritardi[num] = i; numeri_da_trovare.remove(num)
        for num in numeri_da_trovare: ritardi[num] = len(colpi_validi)
        return ritardi

    def calcola_ritardo_ambi(self, ambi_da_controllare, ruote_selezionate, data_fine):
        ambi_da_trovare = set(ambi_da_controllare); ritardi = {ambo: -1 for ambo in ambi_da_trovare}
        colpi_validi = [d for d in self.archivio.date_ordinate if d <= data_fine and any(self.archivio.dati_per_analisi.get(d, {}).get(r) for r in ruote_selezionate)]
        for i, data in enumerate(reversed(colpi_validi)):
            if not ambi_da_trovare: break
            ambi_estratti_giorno = set()
            for ruota_key in ruote_selezionate:
                if estrazione := self.archivio.dati_per_analisi[data].get(ruota_key):
                    ambi_estratti_giorno.update({tuple(sorted(c)) for c in combinations(estrazione, 2)})
            for ambo in list(ambi_da_trovare):
                if ambo in ambi_estratti_giorno: ritardi[ambo] = i; ambi_da_trovare.remove(ambo)
        for ambo in ambi_da_trovare: ritardi[ambo] = len(colpi_validi)
        return ritardi

    def calcola_frequenza_numeri(self, numeri, ruote_selezionate, data_fine, num_concorsi=180):
        frequenze = Counter(); numeri_set = set(numeri)
        colpi_validi = [d for d in self.archivio.date_ordinate if d <= data_fine and any(self.archivio.dati_per_analisi.get(d, {}).get(r) for r in ruote_selezionate)]
        estrazioni_periodo = colpi_validi[-num_concorsi:]
        for data in estrazioni_periodo:
            for ruota in ruote_selezionate:
                if estrazione := self.archivio.dati_per_analisi[data].get(ruota):
                    frequenze.update(numeri_set.intersection(estrazione))
        return {num: frequenze.get(num, 0) for num in numeri}

    def calcola_stato_attuale_previsione(self, top_ambate, top_ambi, tipo_ricerca, ruote_selezionate, data_inizio_analisi, data_fine_analisi, max_colpi):
        numeri_da_giocare = set(top_ambate)
        if not numeri_da_giocare and top_ambi:
            numeri_da_giocare.update(num for ambo in top_ambi for num in ambo)

        if not numeri_da_giocare:
            return "Nessuna previsione valida da analizzare."
        
        # Le sorti da cercare sono quelle della previsione attuale
        numeri_da_cercare = self._get_ambetto_set(numeri_da_giocare) if tipo_ricerca == "Ambetto" else numeri_da_giocare
        ambi_da_giocare = {tuple(sorted(a)) for a in top_ambi}

        # Cerca l'ultimo esito nell'intero archivio FINO ALLA DATA IN CUI HAI GENERATO LA PREVISIONE
        colpi_validi_storico = [d for d in self.archivio.date_ordinate if d <= data_fine_analisi and any(self.archivio.dati_per_analisi.get(d, {}).get(r) for r in ruote_selezionate)]
        
        last_hit_date = None
        for data in reversed(colpi_validi_storico):
            hit_found_this_day = False
            numeri_estratti_colpo = set()
            for ruota_key in ruote_selezionate:
                if estrazione_giorno := self.archivio.dati_per_analisi[data].get(ruota_key):
                    numeri_estratti_colpo.update(estrazione_giorno)

            if tipo_ricerca == "Ambo":
                if ambi_da_giocare and not ambi_da_giocare.isdisjoint({tuple(sorted(c)) for c in combinations(numeri_estratti_colpo, 2)}):
                    hit_found_this_day = True
            elif not numeri_da_cercare.isdisjoint(numeri_estratti_colpo):
                hit_found_this_day = True
            
            if hit_found_this_day:
                last_hit_date = data
                break

        # Ora calcoliamo i colpi giocati DALLA DATA DI GENERAZIONE DELLA PREVISIONE
        data_inizio_giocata = data_fine_analisi
        
        # Se c'è stato un esito lo stesso giorno della generazione, il ciclo inizia dall'estrazione successiva
        if last_hit_date == data_fine_analisi:
             try:
                idx_last_hit = self.archivio.date_ordinate.index(last_hit_date)
                if idx_last_hit + 1 < len(self.archivio.date_ordinate):
                    data_inizio_giocata = self.archivio.date_ordinate[idx_last_hit + 1]
             except (ValueError, IndexError):
                pass
        
        # L'archivio futuro parte dalla prima estrazione DOPO la data di analisi
        archivio_futuro = [d for d in self.archivio.date_ordinate if d > data_fine_analisi]
        
        colpi_giocati_dal_ciclo = 0
        for data_estrazione in archivio_futuro:
             # Conta un colpo solo se c'è stata un'estrazione valida sulle ruote scelte
             if any(self.archivio.dati_per_analisi.get(data_estrazione, {}).get(r) for r in ruote_selezionate):
                 colpi_giocati_dal_ciclo += 1
        
        in_gioco = colpi_giocati_dal_ciclo < max_colpi
        colpi_rimanenti = max_colpi - colpi_giocati_dal_ciclo if in_gioco else 0

        # --- REPORT RIDESCRITTO PER CHIAREZZA ---
        report = "--- STATO ATTUALE DELLA PREVISIONE ---\n\n"
        report += f"Previsione generata con dati fino al: {data_fine_analisi.strftime('%d/%m/%Y')}\n"
        report += f"Numeri in gioco: {sorted(list(numeri_da_giocare))}\n"
        if ambi_da_giocare:
             report += f"Ambi in gioco: {sorted(list(ambi_da_giocare))}\n"
        report += "-"*50 + "\n"

        if last_hit_date:
            report += f"Ultimo esito per questa previsione trovato il: {last_hit_date.strftime('%d/%m/%Y')}\n"
        else:
            report += "Nessun esito precedente trovato nell'archivio per questi numeri.\n"

        if colpi_giocati_dal_ciclo == 0:
             report += f"--> La previsione è NUOVA e il ciclo di gioco non è ancora iniziato.\n"
             report += f"Colpi disponibili: {max_colpi}\n"
        elif in_gioco:
            report += f"--> La previsione è ATTIVA.\n"
            report += f"Colpi giocati da dopo il {data_fine_analisi.strftime('%d/%m/%Y')}: {colpi_giocati_dal_ciclo}\n"
            report += f"COLPI DI GIOCO RIMANENTI: {colpi_rimanenti}\n"
        else: # Ciclo concluso
            report += f"--> Il ciclo di gioco è TERMINATO.\n"
            report += f"Sono stati giocati tutti i {max_colpi} colpi senza esito.\n"

        return report

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("La Magia della Coesione - Piattaforma Analisi - Created by Max Lotto -")
        self.root.geometry("850x800")        
        self.status_queue, self.archivio, self.last_results = queue.Queue(), None, None
        
        self.single_analysis_ambate = None
        self.single_analysis_ambi = None
        self.global_analysis_ambate = None
        self.global_analysis_ambi = None
        
        style = ttk.Style(self.root)
        style.theme_use('clam')
        
        controls_frame = ttk.Frame(self.root, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X)
        
        mode_frame = ttk.LabelFrame(controls_frame, text="1. Scegli la Modalità di Analisi", padding="10")
        mode_frame.pack(fill=tk.X, pady=5)
        self.modalita_var = tk.StringVar(value="Previsione")
        ttk.Radiobutton(mode_frame, text="Previsione Prossima Giocata (Cosa giocare?)", variable=self.modalita_var, value="Previsione", command=self._toggle_date_start).pack(anchor="w", padx=20)
        ttk.Radiobutton(mode_frame, text="Classifica Storica (Cosa ha funzionato meglio?)", variable=self.modalita_var, value="Classifica Storica", command=self._toggle_date_start).pack(anchor="w", padx=20)
        
        input_frame = ttk.LabelFrame(controls_frame, text="2. Imposta Parametri di Ricerca", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ruote_frame = ttk.Frame(input_frame); ruote_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ruote_frame, text="Ruote da analizzare:").pack(anchor="w"); self.ruote_vars = {}; ruote_grid = ttk.Frame(ruote_frame); ruote_grid.pack(anchor="w", pady=(0, 10))
        for i, (key, nome) in enumerate(ArchivioLotto(None).RUOTE_DISPONIBILI.items()):
            var = tk.BooleanVar(value=False); cb = ttk.Checkbutton(ruote_grid, text=nome, variable=var); cb.grid(row=i // 6, column=i % 6, sticky="w", padx=5); self.ruote_vars[key] = var
        
        cinquina_frame = ttk.Frame(input_frame); cinquina_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cinquina_frame, text="Tipo di Cinquine di Coesione:").grid(row=0, column=0, padx=5, sticky="w")
        self.tipo_cinquina_var = tk.StringVar()
        self.cinquina_choices = ["Simmetriche (22)", "Pentagonali (18)", "Di Cadenza (50+)", "Famiglie Matematiche (Varie)"]
        self.cinquina_combo = ttk.Combobox(cinquina_frame, textvariable=self.tipo_cinquina_var, values=self.cinquina_choices, state="readonly", width=30)
        self.cinquina_combo.grid(row=0, column=1, sticky="w"); self.cinquina_combo.set(self.cinquina_choices[0])

        config_frame = ttk.Frame(input_frame); config_frame.pack(fill=tk.X, pady=10)
        self.date_start_label = ttk.Label(config_frame, text="Data Inizio:"); self.date_start_label.grid(row=0, column=0, padx=5, sticky="w")
        self.date_start = DateEntry(config_frame, locale='it_IT', date_pattern='dd/mm/yyyy'); self.date_start.grid(row=0, column=1, padx=5, pady=5)
        self.date_start.set_date(date(2009, 9, 15))
        ttk.Label(config_frame, text="Data Fine (Snapshot):").grid(row=0, column=2, padx=15, sticky="w"); self.date_end = DateEntry(config_frame, locale='it_IT', date_pattern='dd/mm/yyyy'); self.date_end.grid(row=0, column=3, padx=5, pady=5)
        
        ricerca_frame = ttk.Frame(input_frame); ricerca_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ricerca_frame, text="Tipo di Ricerca / Prova:").grid(row=0, column=0, padx=5, sticky="w")
        self.ambata_var = tk.BooleanVar(value=True); self.ambetto_var = tk.BooleanVar(value=False); self.ambo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ricerca_frame, text="Ambata", variable=self.ambata_var).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(ricerca_frame, text="Ambetto", variable=self.ambetto_var).grid(row=0, column=2, sticky="w", padx=10)
        ttk.Checkbutton(ricerca_frame, text="Ambo", variable=self.ambo_var).grid(row=0, column=3, sticky="w", padx=10)
        self.tipo_ricerca_var = tk.StringVar(value="Ambata")
        
        # --- NUOVA SEZIONE PER LA PROVA SU TUTTE LE RUOTE ---
        prova_frame = ttk.Frame(input_frame)
        prova_frame.pack(fill=tk.X, pady=5)
        ttk.Label(prova_frame, text="Opzioni per 'Prova Previsione':").grid(row=0, column=0, padx=5, sticky="w")
        self.prova_tutte_ruote_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(prova_frame, text="Esegui prova su TUTTE le ruote", variable=self.prova_tutte_ruote_var).grid(row=0, column=1, padx=10, sticky="w")
        # --- FINE NUOVA SEZIONE ---

        colpi_frame = ttk.Frame(input_frame); colpi_frame.pack(fill=tk.X, pady=5)
        ttk.Label(colpi_frame, text="Numero massimo di colpi:").grid(row=0, column=0, padx=5, sticky="w")
        self.max_colpi_var = tk.IntVar(value=18); ttk.Spinbox(colpi_frame, from_=1, to=99, textvariable=self.max_colpi_var, width=5).grid(row=0, column=1, sticky="w")
        
        source_frame = ttk.LabelFrame(controls_frame, text="3. Scegli la Fonte Dati", padding="10"); source_frame.pack(fill=tk.X, pady=5)
        self.data_source_var = tk.StringVar(value="GitHub")
        ttk.Radiobutton(source_frame, text="GitHub (Online, raccomandato)", variable=self.data_source_var, value="GitHub", command=self._toggle_local_path).pack(anchor="w")
        ttk.Radiobutton(source_frame, text="Cartella Locale", variable=self.data_source_var, value="Locale", command=self._toggle_local_path).pack(anchor="w")
        self.local_path_frame = ttk.Frame(source_frame); self.local_path_frame.pack(fill=tk.X, pady=5); self.local_path_var = tk.StringVar()
        self.local_path_entry = ttk.Entry(self.local_path_frame, textvariable=self.local_path_var, width=50); self.local_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 5))
        self.browse_button = ttk.Button(self.local_path_frame, text="Scegli...", command=self._browse_folder); self.browse_button.pack(side=tk.LEFT); self._toggle_local_path()
        
        action_frame = ttk.LabelFrame(controls_frame, text="Azioni", padding="10"); action_frame.pack(fill=tk.X, pady=10)
        self.start_button = ttk.Button(action_frame, text="AVVIA ANALISI SINGOLA", command=self._start_analysis, style="Accent.TButton"); self.start_button.pack(side=tk.LEFT, padx=5, ipady=5)
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))
        
        self.convergence_button = ttk.Button(action_frame, text="Trova Convergenze", command=self._start_convergence_worker, state="disabled"); self.convergence_button.pack(side=tk.LEFT, padx=5)
        self.global_button = ttk.Button(action_frame, text="ANALISI GLOBALE", command=self._start_global_analysis); self.global_button.pack(side=tk.LEFT, padx=10, ipady=5)
        
        self.synthesis_button = ttk.Button(action_frame, text="SINTESI FINALE", command=self._start_synthesis, state="disabled", style="Synthesis.TButton")
        self.synthesis_button.pack(side=tk.LEFT, padx=10, ipady=5)
        style.configure("Synthesis.TButton", font=("Helvetica", 10, "bold"), foreground="white", background="navy")
        
        ttk.Button(action_frame, text="Pulisci Risultati", command=self._clear_results).pack(side=tk.LEFT, padx=5)
        
        self.results_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=15, font=("Courier New", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)
        self.results_text.config(state='disabled')
        
        self.status_var = tk.StringVar(value="Pronto."); status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5); status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.after(100, self._process_queue); self._toggle_date_start()

    def _set_buttons_state(self, state):
        self.start_button.config(state=state); self.global_button.config(state=state)
        if state == "disabled": self.convergence_button.config(state="disabled")

    def _start_analysis(self):
        if not (ruote := self._validate_inputs()): return
        # MODIFICATO: Pulisce solo l'area di testo, non la memoria dei risultati globali
        self._clear_main_text_area() 
        self._set_buttons_state('disabled')
        
        tipi_ricerca_selezionati = []
        if self.ambata_var.get(): tipi_ricerca_selezionati.append("Ambata")
        if self.ambetto_var.get(): tipi_ricerca_selezionati.append("Ambetto")
        if self.ambo_var.get(): tipi_ricerca_selezionati.append("Ambo")
        if not tipi_ricerca_selezionati:
            messagebox.showerror("Selezione Mancante", "Devi selezionare almeno un tipo di ricerca."); self._set_buttons_state('normal'); return

        tipo_ricerca_principale = ""
        if "Ambo" in tipi_ricerca_selezionati: tipo_ricerca_principale = "Ambo"
        elif "Ambetto" in tipi_ricerca_selezionati: tipo_ricerca_principale = "Ambetto"
        else: tipo_ricerca_principale = "Ambata"
        self.tipo_ricerca_var.set(tipo_ricerca_principale)
        
        params = {"modalita": self.modalita_var.get(), "tipo_ricerca": self.tipo_ricerca_var.get(), "tipi_ricerca_etichetta": tipi_ricerca_selezionati, "ruote_selezionate": ruote, "data_inizio": self.date_start.get_date(), "data_fine": self.date_end.get_date(), "max_colpi": self.max_colpi_var.get(), "data_source": self.data_source_var.get(), "local_path": self.local_path_var.get(), "tipo_cinquina": self.tipo_cinquina_var.get()}
        threading.Thread(target=self._analysis_worker, args=(params,), daemon=True).start()

    def _analysis_worker(self, params):
        try:
            if not self.archivio: self.archivio = ArchivioLotto(self.status_queue); self.archivio.inizializza(params['data_source'], params['local_path'])
            analyzer = BicinquineAnalyzer(self.archivio, self.status_queue)
            risultati = analyzer.analizza(
                modalita=params['modalita'], tipo_ricerca=params['tipo_ricerca'], ruote_selezionate=params['ruote_selezionate'], 
                data_inizio=params['data_inizio'], data_fine=params['data_fine'], max_colpi=params['max_colpi'], 
                tipo_cinquina=params['tipo_cinquina'], tipi_ricerca_etichetta=params['tipi_ricerca_etichetta']
            )
            self.status_queue.put(("risultato_finale", risultati))
        except Exception as e: self.status_queue.put(f"ERRORE: {e}"); traceback.print_exc()
        finally: self.status_queue.put(("enable_buttons",))

    def _start_global_analysis(self):
        if self.modalita_var.get() != "Previsione":
            messagebox.showwarning("Modalità non supportata", "L'Analisi Globale funziona solo in modalità 'Previsione'.\nVerrà eseguita con questa impostazione.")
            self.modalita_var.set("Previsione")
        if not (ruote := self._validate_inputs()): return
        # MODIFICATO: Pulisce solo l'area di testo, non la memoria dei risultati singoli
        self._clear_main_text_area() 
        self._set_buttons_state('disabled')
        tipi_ricerca_selezionati = []
        if self.ambata_var.get(): tipi_ricerca_selezionati.append("Ambata")
        if self.ambetto_var.get(): tipi_ricerca_selezionati.append("Ambetto")
        if self.ambo_var.get(): tipi_ricerca_selezionati.append("Ambo")
        tipo_ricerca_principale = "Ambata"
        if "Ambo" in tipi_ricerca_selezionati: tipo_ricerca_principale = "Ambo"
        elif "Ambetto" in tipi_ricerca_selezionati: tipo_ricerca_principale = "Ambetto"
        params = {"ruote_selezionate": ruote, "data_fine": self.date_end.get_date(), "data_source": self.data_source_var.get(), "local_path": self.local_path_var.get(), "tipo_ricerca": tipo_ricerca_principale}
        threading.Thread(target=self._global_analysis_worker, args=(params,), daemon=True).start()

    def _global_analysis_worker(self, params):
        try:
            if not self.archivio: self.archivio = ArchivioLotto(self.status_queue); self.archivio.inizializza(params['data_source'], params['local_path'])
            all_results_ambate, all_results_ambi = [], []
            
            tipo_ricerca_globale = params.get('tipo_ricerca', 'Ambata')

            for i, tipo_cinquina in enumerate(self.cinquina_choices):
                self.status_queue.put(f"Analisi Globale ({i+1}/{len(self.cinquina_choices)}): {tipo_cinquina} per {tipo_ricerca_globale}...")
                analyzer = BicinquineAnalyzer(self.archivio, self.status_queue)
                
                _, last_results = analyzer.analizza(
                    modalita='Previsione', 
                    tipo_ricerca=tipo_ricerca_globale,
                    ruote_selezionate=params['ruote_selezionate'], 
                    data_inizio=None, 
                    data_fine=params['data_fine'], 
                    max_colpi=0, 
                    tipo_cinquina=tipo_cinquina, 
                    tipi_ricerca_etichetta=[tipo_ricerca_globale]
                )
                
                punteggio_ambate, punteggio_ambi = Counter(), Counter()

                if tipo_ricerca_globale == 'Ambata': num_coppie_da_analizzare = 5
                elif tipo_ricerca_globale == 'Ambetto': num_coppie_da_analizzare = 7
                else: num_coppie_da_analizzare = 9
                
                self.status_queue.put(f"Uso filtro (Top {num_coppie_da_analizzare}) per {tipo_ricerca_globale}...")

                for j, res in enumerate(last_results[:num_coppie_da_analizzare]):
                    punti = num_coppie_da_analizzare - j
                    numeri_coppia = set(res['coppia'][0] + res['coppia'][1])
                    for num in numeri_coppia: punteggio_ambate[num] += punti
                    for ambo in combinations(numeri_coppia, 2): punteggio_ambi[tuple(sorted(ambo))] += punti
                
                top_ambate_con_punteggio = punteggio_ambate.most_common(50)
                numeri_da_controllare_ambata = [num for num, score in top_ambate_con_punteggio]
                ritardi_numeri = analyzer.calcola_ritardo_numeri(numeri_da_controllare_ambata, params['ruote_selezionate'], params['data_fine'])
                frequenze_numeri_ambata = analyzer.calcola_frequenza_numeri(numeri_da_controllare_ambata, params['ruote_selezionate'], params['data_fine'])
                
                for num, p_conv in top_ambate_con_punteggio:
                    p_finale = p_conv + ritardi_numeri.get(num, 0) + frequenze_numeri_ambata.get(num, 0)
                    # MODIFICATO: Aggiungiamo anche il p_conv alla lista dei risultati
                    all_results_ambate.append({"numero": num, "p_finale": p_finale, "p_conv": p_conv})
                
                top_ambi_con_punteggio = punteggio_ambi.most_common(100)
                ambi_da_controllare = [ambo for ambo, score in top_ambi_con_punteggio]
                ritardi_ambi = analyzer.calcola_ritardo_ambi(ambi_da_controllare, params['ruote_selezionate'], params['data_fine'])
                numeri_unici_ambi = set(num for ambo in ambi_da_controllare for num in ambo)
                frequenze_numeri_ambi = analyzer.calcola_frequenza_numeri(list(numeri_unici_ambi), params['ruote_selezionate'], params['data_fine'])
                for ambo, p_conv in top_ambi_con_punteggio:
                    freq_sommata = frequenze_numeri_ambi.get(ambo[0], 0) + frequenze_numeri_ambi.get(ambo[1], 0)
                    p_finale = p_conv + ritardi_ambi.get(ambo, 0) + freq_sommata
                    all_results_ambi.append({"ambo": ambo, "p_finale": p_finale})

            self.status_queue.put("Aggregazione e calcolo Meta-Punteggio finale...")
            meta_ambate, meta_ambi = {}, {}
            for res in all_results_ambate:
                num = res['numero']
                # MODIFICATO: Inizializziamo anche p_conv_totale
                if num not in meta_ambate: meta_ambate[num] = {'p_totale': 0, 'presenze': 0, 'p_conv_totale': 0}
                meta_ambate[num]['p_totale'] += res['p_finale']
                meta_ambate[num]['presenze'] += 1
                meta_ambate[num]['p_conv_totale'] += res['p_conv'] # Sommiamo i p_conv
            
            for res in all_results_ambi:
                ambo = res['ambo']
                if ambo not in meta_ambi: meta_ambi[ambo] = {'p_totale': 0, 'presenze': 0}
                meta_ambi[ambo]['p_totale'] += res['p_finale']
                meta_ambi[ambo]['presenze'] += 1
            
            for data in meta_ambate.values(): data['meta_punteggio'] = (data['presenze'] ** 2) * data['p_totale']
            for data in meta_ambi.values(): data['meta_punteggio'] = (data['presenze'] ** 2) * data['p_totale']
            
            sorted_meta_ambate = sorted(meta_ambate.items(), key=lambda item: item[1]['meta_punteggio'], reverse=True)
            sorted_meta_ambi = sorted(meta_ambi.items(), key=lambda item: item[1]['meta_punteggio'], reverse=True)
            
            top_5_ambate_data = sorted_meta_ambate[:5]
            numeri_top_ambate = [num for num, data in top_5_ambate_data]
            ritardi_meta_ambate = analyzer.calcola_ritardo_numeri(numeri_top_ambate, params['ruote_selezionate'], params['data_fine'])
            
            report = "--- META-CONVERGENZA GLOBALE ---\n"
            report += f"Analisi per: {tipo_ricerca_globale.upper()} | Ordinati per Meta-Punteggio (Presenze^2 * P. Totale).\n\n"
            
            # MODIFICATO: Aggiunta la nuova colonna nell'intestazione
            report += "--- META-AMBATA (Top 5) ---\n"
            report += "Num. | Meta-Punteggio | P. Conv. Medio | Ritardo | Presenze | P. Totale\n" + "-"*79 + "\n"
            
            for num, data in top_5_ambate_data:
                ritardo = ritardi_meta_ambate.get(num, 'N/A')
                # NUOVO: Calcoliamo il P. Conv. Medio
                p_conv_medio = data['p_conv_totale'] / data['presenze'] if data['presenze'] > 0 else 0
                # MODIFICATO: Aggiungiamo il valore al report
                report += f" {num:<4} | {data['meta_punteggio']:<14} | {p_conv_medio:<14.2f} | {ritardo:<7} | {data['presenze']:<8} | {data['p_totale']}\n"
            
            report += "\n--- META-AMBO (Top 10) ---\n"
            report += "Ambo      | Meta-Punteggio | Presenze | P. Totale\n" + "-"*53 + "\n"
            top_10_ambi_data = sorted_meta_ambi[:10]
            for ambo, data in top_10_ambi_data: report += f" {str(ambo):<9} | {data['meta_punteggio']:<14} | {data['presenze']:<8} | {data['p_totale']}\n"
            
            top_meta_ambate = [num for num, data in top_5_ambate_data]
            top_meta_ambi = [ambo for ambo, data in top_10_ambi_data]
            self.status_queue.put(("global_analysis_result", report, top_meta_ambate, top_meta_ambi))
        except Exception as e:
            self.status_queue.put(f"ERRORE GLOBALE: {e}")
            traceback.print_exc()
        finally:
            self.status_queue.put(("enable_buttons",))

    def _show_global_analysis_window(self, report, top_meta_ambate, top_meta_ambi):
        ### MODIFICATO: Memorizza sia le ambate che gli ambi ###
        self.global_analysis_ambate = top_meta_ambate
        self.global_analysis_ambi = top_meta_ambi

        self.status_var.set("Meta-Convergenza calcolata.")
        
        # Abilita il pulsante Sintesi se anche l'analisi singola è stata fatta
        ### MODIFICATO: Controllo robusto basato su None ###
        if self.single_analysis_ambate is not None:
            self.synthesis_button.config(state="normal")

        win = tk.Toplevel(self.root); win.title("Risultato Analisi Globale"); win.geometry("750x600"); win.transient(self.root); win.grab_set()
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Courier New", 10)); text.pack(expand=True, fill="both", padx=10, pady=10)
        text.insert("1.0", report); text.config(state="disabled")
        btn_frame = ttk.Frame(win); btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Prova Previsione Globale", command=lambda: self._start_verification_worker(top_meta_ambate, top_meta_ambi, win)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Chiudi", command=win.destroy).pack(side=tk.LEFT, padx=5)
    
    def _process_queue(self):
        try:
            while True:
                msg = self.status_queue.get_nowait()
                if isinstance(msg, tuple):
                    command, *value = msg
                    if command == "risultato_finale":
                        if self.modalita_var.get() == "Previsione": report_testuale, self.last_results = value[0]; self.convergence_button.config(state="normal")
                        else: report_testuale = value[0]
                        self.results_text.config(state='normal'); self.results_text.insert(tk.END, report_testuale); self.results_text.config(state='disabled'); self.status_var.set("Analisi completata.")
                    elif command == "enable_buttons": self._set_buttons_state('normal')
                    elif command == "convergence_result": self._show_convergence_window(value[0], value[1], value[2])
                    elif command == "global_analysis_result": self._show_global_analysis_window(*value) 
                    elif command == "verification_result": self._show_verification_window(value[0])
                    elif command == "status_result": self._show_status_window(value[0])
                else: self.status_var.set(str(msg))
        except queue.Empty: pass
        finally: self.root.after(100, self._process_queue)

    def _toggle_date_start(self):
        state = 'normal' if self.modalita_var.get() == "Classifica Storica" else 'disabled'; self.date_start.config(state=state); self.date_start_label.config(state=state)
    def _toggle_local_path(self):
        state = 'normal' if self.data_source_var.get() == "Locale" else 'disabled'; self.local_path_entry.config(state=state); self.browse_button.config(state=state)
    def _browse_folder(self):
        if folder_path := filedialog.askdirectory(): self.local_path_var.set(folder_path)
    def _clear_results(self):
        self.results_text.config(state='normal'); self.results_text.delete('1.0', tk.END); self.results_text.config(state='disabled'); self.status_var.set("Pronto.")
        self.convergence_button.config(state="disabled"); self.last_results = None
        
        ### MODIFICATO: Resetta le liste a None quando si pulisce ###
        self.single_analysis_ambate = None
        self.single_analysis_ambi = None
        self.global_analysis_ambate = None
        self.global_analysis_ambi = None
        self.synthesis_button.config(state="disabled")
    def _clear_main_text_area(self):
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', tk.END)
        self.results_text.config(state='disabled')
        self.convergence_button.config(state="disabled")
        self.last_results = None

    def _validate_inputs(self):
        try:
            ruote = [key for key, var in self.ruote_vars.items() if var.get()];
            if not ruote: raise ValueError("Selezionare almeno una ruota.")
            if not self.tipo_cinquina_var.get() and self.modalita_var.get() != 'Global': raise ValueError("Selezionare un tipo di cinquina per l'analisi singola.")
            if self.modalita_var.get() == "Classifica Storica" and self.date_start.get_date() > self.date_end.get_date():
                raise ValueError("La data di inizio non può essere successiva alla data di fine.")
            return ruote
        except Exception as e: messagebox.showerror("Input non valido", str(e)); return None
    def _start_convergence_worker(self):
        if not self.last_results: messagebox.showinfo("Informazione", "Esegui prima un'analisi in modalità 'Previsione'."); return
        self.status_var.set("Calcolo convergenze e ritardi..."); self._set_buttons_state('disabled')
        threading.Thread(target=self._convergence_worker, daemon=True).start()

    def _convergence_worker(self):
        try:
            avviso_speciale = ""
            risultati_per_analisi = [res for res in self.last_results if res.get('ritardo', 0) > 1]

            if not risultati_per_analisi:
                avviso_speciale = "AVVISO: Tutte le coppie hanno avuto un esito recente.\nL'analisi riparte considerando l'intera classifica dei ritardi.\n\n"
                risultati_per_analisi = self.last_results
            
            if not risultati_per_analisi:
                messaggio_vuoto = "Nessun risultato valido trovato dall'analisi iniziale."
                self.status_queue.put(("convergence_result", messaggio_vuoto, [], []))
                return

            punteggio_ambate, punteggio_ambi = Counter(), Counter()
            num_coppie = len(risultati_per_analisi)
            for i, res in enumerate(risultati_per_analisi):
                punti = num_coppie - i
                numeri_coppia = set(res['coppia'][0] + res['coppia'][1])
                for num in numeri_coppia: punteggio_ambate[num] += punti
                for ambo in combinations(numeri_coppia, 2): punteggio_ambi[tuple(sorted(ambo))] += punti
            
            analyzer = BicinquineAnalyzer(self.archivio, self.status_queue)
            ruote_selezionate = [k for k, v in self.ruote_vars.items() if v.get()]
            data_fine = self.date_end.get_date()
            
            # --- Calcolo Super Ambate ---
            top_ambate_con_punteggio = punteggio_ambate.most_common(20)
            numeri_da_controllare_ambata = [num for num, score in top_ambate_con_punteggio]
            ritardi_numeri_ambata = analyzer.calcola_ritardo_numeri(numeri_da_controllare_ambata, ruote_selezionate, data_fine)
            frequenze_numeri_ambata = analyzer.calcola_frequenza_numeri(numeri_da_controllare_ambata, ruote_selezionate, data_fine)
            
            super_ambate_provvisorie = []
            for num, p_conv in top_ambate_con_punteggio:
                ritardo = ritardi_numeri_ambata.get(num, 0)
                freq = frequenze_numeri_ambata.get(num, 0)
                p_finale = p_conv + ritardo + freq
                super_ambate_provvisorie.append({"numero": num, "p_finale": p_finale, "p_conv": p_conv, "ritardo": ritardo, "freq": freq})
            
            super_ambate_ordinate = sorted(super_ambate_provvisorie, key=lambda x: x['p_finale'], reverse=True)
            super_ambate_finali = [sa for sa in super_ambate_ordinate if sa['ritardo'] > 1][:5]

            # --- Calcolo Super Ambi ---
            top_ambi_con_punteggio = punteggio_ambi.most_common(50)
            ambi_da_controllare = [ambo for ambo, score in top_ambi_con_punteggio]
            ritardi_ambi_coppia = analyzer.calcola_ritardo_ambi(ambi_da_controllare, ruote_selezionate, data_fine)
            numeri_unici_in_top_ambi = set(num for ambo in ambi_da_controllare for num in ambo)
            frequenze_numeri_ambi = analyzer.calcola_frequenza_numeri(list(numeri_unici_in_top_ambi), ruote_selezionate, data_fine)

            super_ambi_provvisori = []
            for ambo, p_conv in top_ambi_con_punteggio:
                ritardo_ambo = ritardi_ambi_coppia.get(ambo, 0)
                freq_sommata = frequenze_numeri_ambi.get(ambo[0], 0) + frequenze_numeri_ambi.get(ambo[1], 0)
                p_finale = p_conv + ritardo_ambo + freq_sommata
                super_ambi_provvisori.append({"ambo": ambo, "p_finale": p_finale, "p_conv": p_conv, "ritardo": ritardo_ambo, "freq_sommata": freq_sommata})
            
            super_ambi_ordinati = sorted(super_ambi_provvisori, key=lambda x: x['p_finale'], reverse=True)
            super_ambi_finali = super_ambi_ordinati[:10] # Prendiamo semplicemente i top 10

            # --- Costruzione Report ---
            report = avviso_speciale
            report += "--- SUPER-AMBATA (Ordinata per Punteggio Finale) ---\n"
            report += "Num. | P. Finale | P. Conv. | Ritardo | Freq. (180c)\n" + "-"*54 + "\n"
            for sa in super_ambate_finali: report += f" {sa['numero']:<4} | {sa['p_finale']:<9} | {sa['p_conv']:<8} | {sa['ritardo']:<7} | {sa['freq']}\n"
            
            report += f"\n--- SUPER-AMBO (Ordinato per Punteggio Finale) ---\n"
            report += "Ambo      | P. Finale | P. Conv. | Ritardo | Freq. Somma\n" + "-"*60 + "\n"
            for sa in super_ambi_finali: report += f" {str(sa['ambo']):<9} | {sa['p_finale']:<9} | {sa['p_conv']:<8} | {sa['ritardo']:<7} | {sa['freq_sommata']}\n"
            
            top_ambate = [sa['numero'] for sa in super_ambate_finali]
            top_ambi = [sa['ambo'] for sa in super_ambi_finali]
            self.status_queue.put(("convergence_result", report, top_ambate, top_ambi))
        except Exception as e:
            self.status_queue.put(f"ERRORE in convergenza: {e}")
            traceback.print_exc()
        finally:
            self.status_queue.put(("enable_buttons",))

    def _show_convergence_window(self, report, top_ambate, top_ambi):
        ### MODIFICATO: Memorizza sia le ambate che gli ambi ###
        self.single_analysis_ambate = top_ambate
        self.single_analysis_ambi = top_ambi
        
        self.status_var.set("Convergenze trovate."); self.convergence_button.config(state="normal")
        
        # Abilita il pulsante Sintesi se anche l'analisi globale è stata fatta
        ### MODIFICATO: Controllo robusto basato su None ###
        if self.global_analysis_ambate is not None:
            self.synthesis_button.config(state="normal")

        conv_window = tk.Toplevel(self.root); conv_window.title("Estrattore di Convergenze e Super-Previsione"); conv_window.geometry("700x550"); conv_window.transient(self.root); conv_window.grab_set()
        ttk.Label(conv_window, text=f"Previsione basata su TUTTE le coppie ritardatarie:", font=("Helvetica", 12, "bold")).pack(pady=10)
        result_text = scrolledtext.ScrolledText(conv_window, wrap=tk.WORD, font=("Courier New", 10)); result_text.pack(expand=True, fill="both", padx=10, pady=5)
        result_text.insert("1.0", report); result_text.config(state="disabled")
        btn_frame = ttk.Frame(conv_window); btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Calcola Stato Attuale", command=lambda: self._start_status_worker(top_ambate, top_ambi, conv_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Prova la Previsione", command=lambda: self._start_verification_worker(top_ambate, top_ambi, conv_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Chiudi", command=conv_window.destroy).pack(side=tk.LEFT, padx=5)

    def _start_verification_worker(self, top_ambate, top_ambi, parent_window):
        tipi_ricerca_selezionati = []
        if self.ambo_var.get(): tipi_ricerca_selezionati.append("Ambo")
        if self.ambetto_var.get(): tipi_ricerca_selezionati.append("Ambetto")
        if self.ambata_var.get(): tipi_ricerca_selezionati.append("Ambata")
        
        if not tipi_ricerca_selezionati:
            messagebox.showerror("Nessuna Selezione", "Seleziona almeno un tipo di ricerca (Ambata, Ambetto, Ambo) per provare la previsione.")
            return

        if parent_window:
            parent_window.destroy()
            
        self.status_var.set("Avvio prova della previsione...")
        
        ruote_per_prova = []
        if self.prova_tutte_ruote_var.get():
            ruote_per_prova = list(self.archivio.RUOTE_DISPONIBILI.keys())
        else:
            ruote_per_prova = [k for k, v in self.ruote_vars.items() if v.get()]
            if not ruote_per_prova:
                messagebox.showerror("Ruote non selezionate", "Per la prova sulle ruote selezionate, devi prima sceglierne almeno una.")
                self.status_var.set("Pronto.")
                return
        
        # Le liste top_ambate e top_ambi sono quelle passate direttamente dal pulsante
        # Questo garantisce che la giocata sia sempre quella visualizzata
        params = {
            "top_ambate": top_ambate,
            "top_ambi": top_ambi,
            "ruote": ruote_per_prova,
            "snapshot_date": self.date_end.get_date(),
            "max_colpi": self.max_colpi_var.get(),
            "tipi_ricerca": tipi_ricerca_selezionati
        }
        threading.Thread(target=self._verification_worker, args=(params,), daemon=True).start()

    def _verification_worker(self, params):
        try:
            self.status_var.set("Esecuzione prova su estrazioni successive...");
            analyzer = BicinquineAnalyzer(self.archivio, self.status_queue)
            report = self._esegui_prova_previsione(analyzer, **params)
            self.status_queue.put(("verification_result", report))
        except Exception as e: self.status_queue.put(f"ERRORE in prova: {e}"); traceback.print_exc()

    def _esegui_prova_previsione(self, analyzer, top_ambate, top_ambi, ruote, snapshot_date, max_colpi, tipi_ricerca):
        
        giocata_ambate = set(top_ambate)
        giocata_ambi_precisi = {tuple(sorted(a)) for a in top_ambi}

        report = f"--- PROVA DELLA PREVISIONE (FORWARD-TEST) ---\n"
        ruote_testate_str = "TUTTE" if len(ruote) == len(self.archivio.RUOTE_DISPONIBILI) else ', '.join(ruote)
        report += f"Ruote Testate: {ruote_testate_str}\n"
        report += f"Tipi di Vincita Controllati: {', '.join(tipi_ricerca)}\n"
        report += f"Previsione generata con dati fino al: {snapshot_date.strftime('%d/%m/%Y')}\n"
        
        estrazioni_da_provare = [(d, self.archivio.dati_per_analisi[d]) for d in self.archivio.date_ordinate if d > snapshot_date][:max_colpi]
        if not estrazioni_da_provare: return "Nessuna estrazione futura trovata nell'archivio per provare la previsione."
        
        report += f"Periodo di Prova: {estrazioni_da_provare[0][0].strftime('%d/%m/%Y')} - {estrazioni_da_provare[-1][0].strftime('%d/%m/%Y')} ({len(estrazioni_da_provare)} estrazioni totali)\n"
        
        if "Ambata" in tipi_ricerca and giocata_ambate:
            report += f"Giocata per AMBATA: {sorted(list(giocata_ambate))}\n"
        if "Ambetto" in tipi_ricerca and giocata_ambate:
            report += f"Giocata per AMBETTO (basata su ambate): {sorted(list(giocata_ambate))}\n"
        if "Ambo" in tipi_ricerca and giocata_ambi_precisi:
            report += f"Giocata per AMBO: {sorted(list(giocata_ambi_precisi))}\n"
        
        log_eventi = []
        colpi_delle_vincite = []

        for colpo_continuo, (data_corrente, estrazioni_giorno) in enumerate(estrazioni_da_provare, 1):
            for ruota_key in ruote:
                if not estrazioni_giorno[ruota_key]: continue
                
                estrazione_set = set(estrazioni_giorno[ruota_key])
                nome_ruota = self.archivio.RUOTE_DISPONIBILI.get(ruota_key, ruota_key)
                
                # Controllo AMBO (da Super-Ambi)
                if "Ambo" in tipi_ricerca and giocata_ambi_precisi:
                    ambi_estratti = {tuple(sorted(c)) for c in combinations(estrazione_set, 2)}
                    ambi_vincenti = giocata_ambi_precisi.intersection(ambi_estratti)
                    if ambi_vincenti:
                        log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. AMBO VINCENTE: {list(ambi_vincenti)}")
                        colpi_delle_vincite.append(colpo_continuo)
                
                # --- LOGICA DI REPORTING PER AMBATA MODIFICATA ---
                if "Ambata" in tipi_ricerca and giocata_ambate:
                    estratti_vincenti = giocata_ambate.intersection(estrazione_set)
                    
                    if len(estratti_vincenti) == 1:
                        # Se è uscito un solo numero, è un ESTRATTO
                        log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. ESTRATTO VINCENTE: {list(estratti_vincenti)}")
                        colpi_delle_vincite.append(colpo_continuo)
                    elif len(estratti_vincenti) >= 2:
                        # Se sono usciti 2 o più numeri, li etichettiamo come AMBI, TERNI, etc.
                        ambi_da_estratti = {tuple(sorted(c)) for c in combinations(estratti_vincenti, 2)}
                        terni_da_estratti = {tuple(sorted(c)) for c in combinations(estratti_vincenti, 3)}
                        quaterne_da_estratti = {tuple(sorted(c)) for c in combinations(estratti_vincenti, 4)}
                        cinquine_da_estratti = {tuple(sorted(c)) for c in combinations(estratti_vincenti, 5)}
                        
                        vincita_superiore = False
                        if len(estratti_vincenti) >= 5 and cinquine_da_estratti:
                             log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. CINQUINA tra Estratti: {list(cinquine_da_estratti)}")
                             vincita_superiore = True
                        if not vincita_superiore and len(estratti_vincenti) >= 4 and quaterne_da_estratti:
                             log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. QUATERNA tra Estratti: {list(quaterne_da_estratti)}")
                             vincita_superiore = True
                        if not vincita_superiore and len(estratti_vincenti) >= 3 and terni_da_estratti:
                             log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. TERNO tra Estratti: {list(terni_da_estratti)}")
                             vincita_superiore = True
                        if not vincita_superiore and len(estratti_vincenti) >= 2 and ambi_da_estratti:
                             log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. AMBO tra Estratti: {list(ambi_da_estratti)}")

                        colpi_delle_vincite.append(colpo_continuo)


                # Controllo AMBETTO
                if "Ambetto" in tipi_ricerca and giocata_ambate:
                    intersezione = giocata_ambate.intersection(estrazione_set)
                    if intersezione:
                        for num_uscito in intersezione:
                            predecessore = 90 if num_uscito == 1 else num_uscito - 1
                            successore = 1 if num_uscito == 90 else num_uscito + 1
                            if predecessore in estrazione_set or successore in estrazione_set:
                                log_eventi.append(f"VINCITA al colpo {colpo_continuo} il {data_corrente.strftime('%d/%m/%Y')} su {nome_ruota}. AMBETTO VINCENTE!")
                                colpi_delle_vincite.append(colpo_continuo)
                                break

        vincite_totali = len(log_eventi)
        prima_vincita_str = "Nessuna"
        media_colpi_str = "N/A"
        if colpi_delle_vincite:
            colpi_unici_di_vincita = sorted(list(set(colpi_delle_vincite)))
            prima_vincita_str = f"al colpo {colpi_unici_di_vincita[0]}"
            if len(colpi_unici_di_vincita) > 0:
                 media_colpi_str = f"{(sum(colpi_unici_di_vincita) / len(colpi_unici_di_vincita)):.2f} colpi"

        report += "-"*70 + "\n"
        report += "RIEPILOGO VINCITE TROVATE:\n"
        report += f"  - Vincite Totali nel periodo: {vincite_totali}\n"
        report += f"  - Prima Vincita: {prima_vincita_str}\n"
        report += f"  - Attesa Media per Vincita: {media_colpi_str}\n"
        report += f"\n--- LOG EVENTI DI PROVA ---\n"

        if log_eventi:
            log_ordinato = sorted(log_eventi, key=lambda x: (int(x.split("colpo ")[1].split(" ")[0]), x))
            report += "\n".join(log_ordinato)
        else:
            report += f"Nessuna vincita trovata per le sorti selezionate nei {len(estrazioni_da_provare)} colpi analizzati."
            
        return report

    def _show_verification_window(self, report):
        ver_window = tk.Toplevel(self.root); ver_window.title("Risultati Prova Previsione"); ver_window.geometry("850x600")
        text = scrolledtext.ScrolledText(ver_window, wrap=tk.WORD, font=("Courier New", 10)); text.pack(expand=True, fill="both", padx=10, pady=10)
        text.insert("1.0", report); text.config(state="disabled")
        ttk.Button(ver_window, text="Chiudi", command=ver_window.destroy).pack(pady=10)

    def _start_status_worker(self, top_ambate, top_ambi, parent_window):
        tipo_ricerca_stato = "Ambo"
        if self.ambetto_var.get(): tipo_ricerca_stato = "Ambetto"
        if self.ambata_var.get(): tipo_ricerca_stato = "Ambata"
        
        if parent_window:
            parent_window.destroy()
        
        self.status_var.set("Calcolo lo stato attuale della previsione...")
        
        # MODIFICATO: Passiamo la data di fine analisi (snapshot)
        # e non la data di inizio del periodo storico.
        params = {
            "top_ambate": top_ambate, 
            "top_ambi": top_ambi, 
            "ruote_selezionate": [k for k, v in self.ruote_vars.items() if v.get()], 
            "data_inizio_analisi": self.date_start.get_date(), # Lo passiamo per completezza
            "data_fine_analisi": self.date_end.get_date(),    # Questo è il parametro cruciale
            "max_colpi": self.max_colpi_var.get(), 
            "tipo_ricerca": tipo_ricerca_stato
        }
        threading.Thread(target=self._status_worker, args=(params,), daemon=True).start()

    def _status_worker(self, params):
        try:
            analyzer = BicinquineAnalyzer(self.archivio, self.status_queue); report = analyzer.calcola_stato_attuale_previsione(**params)
            self.status_queue.put(("status_result", report))
        except Exception as e: self.status_queue.put(f"ERRORE in stato attuale: {e}"); traceback.print_exc()

    def _show_status_window(self, report):
        stat_window = tk.Toplevel(self.root); stat_window.title("Stato Attuale Previsione"); stat_window.geometry("600x300"); stat_window.transient(self.root); stat_window.grab_set()
        text = scrolledtext.ScrolledText(stat_window, wrap=tk.WORD, font=("Courier New", 10)); text.pack(expand=True, fill="both", padx=10, pady=10)
        text.insert("1.0", report); text.config(state="disabled"); ttk.Button(stat_window, text="Chiudi", command=stat_window.destroy).pack(pady=10)

    def _start_synthesis(self):
        self.status_var.set("Avvio Sintesi Finale...")
        
        # Gestione sicura nel caso una delle analisi non sia stata eseguita
        ambate_singola_set = set(self.single_analysis_ambate) if self.single_analysis_ambate is not None else set()
        ambate_globale_set = set(self.global_analysis_ambate) if self.global_analysis_ambate is not None else set()
        ambi_singola_set = {tuple(sorted(a)) for a in self.single_analysis_ambi} if self.single_analysis_ambi is not None else set()
        ambi_globale_set = {tuple(sorted(a)) for a in self.global_analysis_ambi} if self.global_analysis_ambi is not None else set()

        # 1. Trova le AMBATE convergenti
        convergenti_assoluti_ambate = sorted(list(ambate_singola_set.intersection(ambate_globale_set)))

        # 2. Trova gli AMBI convergenti
        convergenti_assoluti_ambi = sorted(list(ambi_singola_set.intersection(ambi_globale_set)))

        # 3. Prepara il report testuale
        report = "--- SINTESI FINALE DELLE PREVISIONI ---\n\n"
        report += f"Fonte Dati - Analisi Singola:\n"
        report += f" - Ambate: {sorted(self.single_analysis_ambate) if self.single_analysis_ambate is not None else 'N/A'}\n"
        report += f" - Ambi: {sorted(list(self.single_analysis_ambi)) if self.single_analysis_ambi is not None else 'N/A'}\n\n"
        report += f"Fonte Dati - Analisi Globale:\n"
        report += f" - Ambate: {sorted(self.global_analysis_ambate) if self.global_analysis_ambate is not None else 'N/A'}\n"
        report += f" - Ambi: {sorted(list(self.global_analysis_ambi)) if self.global_analysis_ambi is not None else 'N/A'}\n"
        report += "-"*60 + "\n\n"

        report += ">>> SUPER-AMBATE ASSOLUTE (Convergenza Totale) <<<\n"
        if not convergenti_assoluti_ambate:
            report += "Nessun numero comune trovato tra le due analisi.\n"
        else:
            report += "Numeri segnalati da ENTRAMBI i metodi:\n\n"
            for numero in convergenti_assoluti_ambate:
                report += f"  --==  {numero}  ==--\n"
        
        report += "\n" + "-"*60 + "\n\n"

        report += ">>> SUPER-AMBI ASSOLUTI (Convergenza Totale) <<<\n"
        if not convergenti_assoluti_ambi:
            report += "Nessun ambo comune trovato tra le due analisi.\n"
        else:
            report += "Ambi segnalati da ENTRAMBI i metodi:\n\n"
            for ambo in convergenti_assoluti_ambi:
                report += f"  --==  {str(ambo)}  ==--\n"
        
        # 4. Mostra la finestra con il report e passa le liste CORRETTE e SEPARATE
        self._show_synthesis_window(report, convergenti_assoluti_ambate, convergenti_assoluti_ambi)

    def _show_synthesis_window(self, report, top_ambate_sintesi, top_ambi_sintesi):
        self.status_var.set("Sintesi Finale completata.")
        
        win = tk.Toplevel(self.root)
        win.title("Sintesi Finale - Convergenza Totale")
        win.geometry("700x600")
        win.transient(self.root)
        win.grab_set()
        
        text = scrolledtext.ScrolledText(win, wrap=tk.WORD, font=("Courier New", 11))
        text.pack(expand=True, fill="both", padx=10, pady=10)
        text.insert("1.0", report)
        text.config(state="disabled")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)
        
        # MODIFICATO: Il pulsante ora passa entrambe le liste, separate, al worker
        btn = ttk.Button(btn_frame, text="Prova Giocata Unificata", 
                   command=lambda: self._start_verification_worker(top_ambate_sintesi, top_ambi_sintesi, win))
        btn.pack(side=tk.LEFT, padx=5)
        
        # Disabilita il pulsante se non c'è nulla da provare
        if not top_ambate_sintesi and not top_ambi_sintesi:
            btn.config(state="disabled")

        ttk.Button(btn_frame, text="Chiudi", command=win.destroy).pack(side=tk.LEFT, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()