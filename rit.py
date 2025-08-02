import pandas as pd
import numpy as np
from datetime import datetime, date
from collections import defaultdict, Counter
import statistics
import requests
import os

class LottoAnalyzer:
    # --- MODIFICA CHIAVE: L'init ora accetta la fonte dati ---
    def __init__(self, data_source='github', local_path=None):
        self.estrazioni = {}
        self.RUOTE_DISPONIBILI = {'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova', 'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma', 'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'}
        
        self.GITHUB_USER = "illottodimax"
        self.GITHUB_REPO = "Archivio"
        self.GITHUB_BRANCH = "main"
        self.URL_RUOTE = {key: f'https://raw.githubusercontent.com/{self.GITHUB_USER}/{self.GITHUB_REPO}/{self.GITHUB_BRANCH}/{value.upper()}.txt' for key, value in self.RUOTE_DISPONIBILI.items()}
        
        self.data_source = data_source
        self.local_path = local_path
    
    def _parse_estrazioni(self, linee):
        # ... (Questa funzione non cambia)
        estrazioni_temp = []
        for linea in linee:
            linea_pulita = linea.strip()
            if not linea_pulita:
                continue
            parti = linea_pulita.split('\t')
            if len(parti) >= 7:
                try:
                    data_estrazione = datetime.strptime(parti[0], '%Y/%m/%d').date()
                    numeri = [int(n) for n in parti[2:7]]
                    estrazioni_temp.append({'data': data_estrazione, 'numeri': numeri})
                except (ValueError, IndexError): 
                    continue
        return sorted(estrazioni_temp, key=lambda x: x['data'])
    
    def carica_dati_per_ruote(self, lista_ruote, status_callback, force_reload=False):
        # --- MODIFICA CHIAVE: La logica ora usa la fonte dati scelta ---
        for i, ruota in enumerate(lista_ruote):
            if ruota in self.estrazioni and not force_reload:
                continue

            status_callback(f"Caricando dati per {self.RUOTE_DISPONIBILI[ruota]} ({i+1}/{len(lista_ruote)})...")
            try:
                if self.data_source == 'local':
                    if not self.local_path:
                        raise ValueError("Percorso cartella locale non specificato.")
                    nome_file = f"{self.RUOTE_DISPONIBILI[ruota].upper()}.txt"
                    percorso_file = os.path.join(self.local_path, nome_file)
                    status_callback(f"Leggendo file locale: {percorso_file}")
                    with open(percorso_file, 'r', encoding='utf-8') as f:
                        linee = f.readlines()
                    self.estrazioni[ruota] = self._parse_estrazioni(linee)
                
                else: # Default è 'github'
                    status_callback(f"Scaricando dati da Internet (GitHub)...")
                    response = requests.get(self.URL_RUOTE[ruota], timeout=15)
                    response.raise_for_status()
                    linee = response.text.strip().split('\n')
                    self.estrazioni[ruota] = self._parse_estrazioni(linee)
            
            except FileNotFoundError:
                status_callback(f"!!! ATTENZIONE: File {nome_file} non trovato. Ruota ignorata.")
                self.estrazioni[ruota] = []
            except Exception as e:
                status_callback(f"!!! ERRORE nel caricare {ruota}: {e}")
                self.estrazioni[ruota] = []

# CLASSE 2: RITARDI ANALYZER
class RitardiAnalyzer:
    # ... (Il codice di RitardiAnalyzer è perfetto, non lo riporto per brevità)
    def __init__(self, lotto_analyzer):
        self.analyzer = lotto_analyzer
        self.dati_precalcolati = {}
        self.RIGHI = {
            1: list(range(1, 11)), 2: list(range(11, 21)), 3: list(range(21, 31)),
            4: list(range(31, 41)), 5: list(range(41, 51)), 6: list(range(51, 61)),
            7: list(range(61, 71)), 8: list(range(71, 81)), 9: list(range(81, 91))
        }

    def _precalcola_dati_ruota(self, ruota):
        if ruota in self.dati_precalcolati: return
        estrazioni = self.analyzer.estrazioni.get(ruota)
        if not estrazioni:
            self.dati_precalcolati[ruota] = {}; return
        ritardi_semplici = {n: self._calcola_ritardo_semplice_core(n, estrazioni) for n in range(1, 91)}
        ritardi_validi = [r for r in ritardi_semplici.values() if r is not None]
        media = statistics.mean(ritardi_validi) if ritardi_validi else 0
        massimo = max(ritardi_validi) if ritardi_validi else 1
        ritardi_rigo = {rigo: self._calcola_ritardo_di_rigo_core(rigo, estrazioni) for rigo in self.RIGHI}
        self.dati_precalcolati[ruota] = {
            'ritardi_semplici': ritardi_semplici,
            'media_ritardi_semplici': media,
            'max_ritardo_semplice': massimo,
            'ritardi_rigo': ritardi_rigo
        }

    def _calcola_ritardo_semplice_core(self, numero, estrazioni):
        for i in range(len(estrazioni) - 1, -1, -1):
            if numero in estrazioni[i]['numeri']: return len(estrazioni) - 1 - i
        return len(estrazioni)

    def _calcola_ritardo_di_rigo_core(self, rigo, estrazioni):
        numeri_rigo = set(self.RIGHI[rigo])
        for i in range(len(estrazioni) - 1, -1, -1):
            if not numeri_rigo.isdisjoint(estrazioni[i]['numeri']): return len(estrazioni) - 1 - i
        return len(estrazioni)

    def calcola_ritardo_dei_ritardi(self, numero, ruota, soglia_ritardo=50):
        estrazioni = self.analyzer.estrazioni.get(ruota)
        if not estrazioni: return None
        contatore = 0
        for i in range(len(estrazioni) - 1, -1, -1):
            ritardo_storico = self._calcola_ritardo_semplice_core(numero, estrazioni[:i+1])
            if ritardo_storico >= soglia_ritardo: contatore += 1
            else: break
        return contatore
    
    def calcola_punteggio_convergenza(self, analisi, dati_precalcolati_ruota):
        punteggio = 0
        pesi = {'semplice': 0.35, 'relativo': 0.25, 'rigo': 0.20, 'ritardi': 0.20}
        max_ritardo_s = dati_precalcolati_ruota.get('max_ritardo_semplice', 1)
        if analisi.get('ritardo_semplice'):
            punteggio += (analisi['ritardo_semplice'] / max_ritardo_s) * pesi['semplice']
        if analisi.get('ritardo_relativo', 0) > 0:
            punteggio += (analisi['ritardo_relativo'] / max_ritardo_s) * pesi['relativo']
        if analisi.get('ritardo_di_rigo'):
            max_rigo = 40
            punteggio += (min(analisi['ritardo_di_rigo'], max_rigo) / max_rigo) * pesi['rigo']
        if analisi.get('ritardo_dei_ritardi'):
            max_rdr = 30
            punteggio += (min(analisi['ritardo_dei_ritardi'], max_rdr) / max_rdr) * pesi['ritardi']
        return round(punteggio * 100, 2)

    def trova_convergenze(self, ruota, top_n=10):
        self._precalcola_dati_ruota(ruota)
        dati_precalcolati = self.dati_precalcolati.get(ruota)
        if not dati_precalcolati: return []
        risultati_completi = []
        for numero in range(1, 91):
            rigo_numero = next((r for r, numeri in self.RIGHI.items() if numero in numeri), None)
            analisi = {
                'numero': numero, 'ruota': ruota,
                'ritardo_semplice': dati_precalcolati['ritardi_semplici'].get(numero),
                'ritardo_relativo': dati_precalcolati['ritardi_semplici'].get(numero, 0) - dati_precalcolati['media_ritardi_semplici'],
                'ritardo_di_rigo': dati_precalcolati['ritardi_rigo'].get(rigo_numero) if rigo_numero else None,
                'rigo': rigo_numero,
                'ritardo_dei_ritardi': self.calcola_ritardo_dei_ritardi(numero, ruota, soglia_ritardo=dati_precalcolati['media_ritardi_semplici'] * 2.5)
            }
            if analisi['ritardo_semplice'] is not None:
                analisi['punteggio_convergenza'] = self.calcola_punteggio_convergenza(analisi, dati_precalcolati)
                risultati_completi.append(analisi)
        risultati_completi.sort(key=lambda x: x['punteggio_convergenza'], reverse=True)
        return risultati_completi[:top_n]
    
    def genera_report_completo(self, ruota, top_n=10):
        convergenze = self.trova_convergenze(ruota, top_n=top_n)
        report = f"\n--- RUOTA {self.analyzer.RUOTE_DISPONIBILI.get(ruota, ruota).upper()} ---\n"
        report += f"{'Pos':<4} {'Num':<4} {'R.Semplice':<11} {'R.Relativo':<11} {'Rigo':<5} {'R.Rigo':<7} {'R.Ritardi':<10} {'Punteggio':<10}\n"
        report += f"{'-'*80}\n"
        for i, analisi in enumerate(convergenze, 1):
            report += f"{i:<4} {analisi['numero']:<4} "
            report += f"{analisi.get('ritardo_semplice', 0):<11} "
            report += f"{analisi.get('ritardo_relativo', 0):<11.1f} "
            report += f"{analisi.get('rigo', 'N/A'):<5} "
            report += f"{analisi.get('ritardo_di_rigo', 0):<7} "
            report += f"{analisi.get('ritardo_dei_ritardi', 0):<10} "
            report += f"{analisi.get('punteggio_convergenza', 0):<10.2f}\n"
        return report

# ==============================================================================
# CLASSE 3: BACKTESTER (MODIFICATA COME RICHIESTO)
# Sostituisci la tua vecchia classe con questa.
# ==============================================================================
class Backtester:
    def __init__(self, lotto_analyzer):
        self.full_analyzer = lotto_analyzer

    def run_backtest(self, data_inizio_test, ruote_da_testare, colpi_di_gioco, top_n_numeri, status_callback):
        report_finale = f"--- BACKTEST | DATA: {data_inizio_test.strftime('%d/%m/%Y')} | COLPI: {colpi_di_gioco} ---\n\n"
        status_callback("Caricamento dati per il backtest...")
        self.full_analyzer.carica_dati_per_ruote(ruote_da_testare, status_callback, force_reload=True)
        
        for ruota in ruote_da_testare:
            report_finale += f"--- RUOTA: {self.full_analyzer.RUOTE_DISPONIBILI[ruota]} ---\n"
            estrazioni_complete = self.full_analyzer.estrazioni.get(ruota)
            
            if not estrazioni_complete:
                report_finale += "Nessun dato trovato per questa ruota.\n\n"; continue
            
            estrazioni_storiche = [e for e in estrazioni_complete if e['data'] <= data_inizio_test]
            if not estrazioni_storiche:
                report_finale += "Nessun dato storico per questa data.\n\n"; continue
            
            analyzer_storico = LottoAnalyzer(); analyzer_storico.estrazioni[ruota] = estrazioni_storiche
            ritardi_analyzer_storico = RitardiAnalyzer(analyzer_storico)
            previsione = ritardi_analyzer_storico.trova_convergenze(ruota, top_n=top_n_numeri)
            
            if not previsione:
                report_finale += "Nessun numero suggerito.\n\n"; continue

            # Estrai solo i numeri dalla previsione
            numeri_suggeriti = [s['numero'] for s in previsione]
            report_finale += f"Numeri suggeriti: " + ", ".join(map(str, numeri_suggeriti)) + "\n"
            
            # Prepara i dati per il controllo
            estrazioni_future_reali = [e for e in estrazioni_complete if e['data'] > data_inizio_test]
            estrazioni_da_controllare = estrazioni_future_reali[:colpi_di_gioco]
            
            # --- INIZIO NUOVA LOGICA ---
            
            vincita_trovata = False
            numeri_suggeriti_set = set(numeri_suggeriti)

            # Controlla estrazione per estrazione
            for i, estrazione in enumerate(estrazioni_da_controllare):
                colpo_attuale = i + 1
                numeri_estratti_set = set(estrazione['numeri'])
                
                # Cerca un'intersezione tra i numeri suggeriti e quelli estratti
                vincitori = numeri_suggeriti_set.intersection(numeri_estratti_set)
                
                if vincitori:
                    # Abbiamo una vincita!
                    numero_vincente = list(vincitori)[0] # Prendiamo il primo che troviamo
                    data_vincita = estrazione['data'].strftime('%d/%m/%Y')
                    report_finale += f"  -> Esito: VINTO con il N° {numero_vincente} al {colpo_attuale}° colpo (Data: {data_vincita})\n"
                    vincita_trovata = True
                    break # Interrompi il controllo per questa ruota, la previsione è chiusa.

            # Se, dopo aver controllato tutte le estrazioni disponibili, non abbiamo vinto...
            if not vincita_trovata:
                colpi_verificati = len(estrazioni_da_controllare)
                if colpi_verificati < colpi_di_gioco:
                    colpi_rimanenti = colpi_di_gioco - colpi_verificati
                    report_finale += f"  -> Esito: ANCORA IN GIOCO (Nessun esito dopo {colpi_verificati} colpi, {colpi_rimanenti} rimanenti)\n"
                else:
                    report_finale += f"  -> Esito: PERSO (Nessun esito dopo {colpi_di_gioco} colpi)\n"

            # --- FINE NUOVA LOGICA ---
            
            report_finale += "\n"
            
        return report_finale