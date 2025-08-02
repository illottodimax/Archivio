# archivio_lotto.py
import os
import requests
from datetime import datetime
import queue
import traceback

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
                else: # Non implementata nell'interfaccia ma lasciata per completezza
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