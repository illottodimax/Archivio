import requests
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog

# ==============================================================================
# 1. LOGICA DI BACKEND (con il nuovo motore statistico)
# ==============================================================================
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"

RUOTE_MAP = {
    'BARI': 'BA', 'CAGLIARI': 'CA', 'FIRENZE': 'FI', 'GENOVA': 'GE',
    'MILANO': 'MI', 'NAPOLI': 'NA', 'PALERMO': 'PA', 'ROMA': 'RO',
    'TORINO': 'TO', 'VENEZIA': 'VE', 'NAZIONALE': 'NZ'
}

def carica_estrazioni_da_github(nome_ruota, output_widget):
    nome_file = f"{nome_ruota.upper()}.txt"
    url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{nome_file}"
    if output_widget:
        output_widget.insert(tk.END, f"[INFO] Tentativo di caricamento dati per {nome_ruota.upper()}...\n")
        output_widget.see(tk.END)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if output_widget: output_widget.insert(tk.END, f"[ERRORE] Impossibile caricare il file: {e}\n")
        return []
    
    estrazioni_trovate = []
    for ln, riga in enumerate(response.text.strip().split('\n'), 1):
        parti = riga.strip().split()
        if len(parti) == 7:
            try:
                numeri = [int(n) for n in parti[2:7]]
                data = parti[0]
                estrazioni_trovate.append({'numeri': numeri, 'data': data})
            except ValueError:
                if output_widget: output_widget.insert(tk.END, f"[AVVISO] Riga {ln} ignorata.\n")
    
    return estrazioni_trovate

def calcola_statistiche_complete(archivio):
    """Nuovo motore che calcola tutte le statistiche in una sola passata."""
    if not archivio: return {}
    
    stats = {n: {'freq': 0, 'rit_max': 0, 'last_seen': -1} for n in range(1, 91)}
    tot_estr = len(archivio)

    for i, estrazione_data in enumerate(archivio):
        for numero in estrazione_data['numeri']:
            stats[numero]['freq'] += 1
            # Calcola il ritardo precedente e aggiorna il massimo
            ritardo_precedente = i - stats[numero]['last_seen'] - 1
            if ritardo_precedente > stats[numero]['rit_max']:
                stats[numero]['rit_max'] = ritardo_precedente
            stats[numero]['last_seen'] = i
            
    # Calcola il ritardo attuale e l'Indice di Convenienza finale
    for n in range(1, 91):
        if stats[n]['last_seen'] == -1: # Numero mai uscito
            stats[n]['rit_att'] = tot_estr
            stats[n]['rit_max'] = tot_estr
            stats[n]['ic'] = 0
        else:
            rit_att = tot_estr - 1 - stats[n]['last_seen']
            stats[n]['rit_att'] = rit_att
            if rit_att > stats[n]['rit_max']:
                stats[n]['rit_max'] = rit_att
            
            # Calcolo Indice di Convenienza (IC)
            if stats[n]['freq'] > 0:
                stats[n]['ic'] = (rit_att * stats[n]['freq']) / tot_estr
            else:
                stats[n]['ic'] = 0
    return stats

def trova_numeri_conseguenti(archivio, numero_spia):
    numeri_conseguenti = []
    for i in range(len(archivio) - 1):
        if numero_spia in archivio[i]['numeri']:
            numeri_conseguenti.extend(archivio[i+1]['numeri'])
    return numeri_conseguenti

def genera_previsione_spia(archivio, numero_spia, num_previsione):
    numeri_conseguenti = trova_numeri_conseguenti(archivio, numero_spia)
    if not numeri_conseguenti: return None
    frequenze = defaultdict(int)
    for n in numeri_conseguenti: frequenze[n] += 1
    previsione_ordinata = sorted(frequenze.items(), key=lambda x: x[1], reverse=True)
    return [num for num, freq in previsione_ordinata[:num_previsione]]

def esegui_backtest_comparativo(archivio, nome_ruota, colpi, finestra, num_previsione, num_test=None):
    output_string = ""
    output_string += "\n" + "#"*80 + "\n"
    output_string += f"  AVVIO BACKTEST COMPARATIVO - RUOTA DI {nome_ruota.upper()}\n"
    output_string += f"  Colpi: {colpi} | Finestra: {finestra} | Numeri Giocati: {num_previsione}\n"
    if num_test:
        output_string += f"  Test limitato alle ultime {num_test} occasioni.\n"
    output_string += "#"*80 + "\n"
    
    # Calcola da dove iniziare il ciclo di test
    start_index = finestra
    if num_test and len(archivio) > num_test + finestra + colpi:
        start_index = len(archivio) - num_test - colpi

    risultati_per_posizione = {pos: defaultdict(int) for pos in range(5)}

    for pos in range(5):
        # La stampa dei progressi può essere rimossa se rallenta troppo
        # print(f"--- Test su Posizione Seme: {pos + 1} ---") 
        for i in range(start_index, len(archivio) - colpi):
            archivio_storico = archivio[i - finestra : i]
            numero_spia = archivio[i]['numeri'][pos]
            previsione = genera_previsione_spia(archivio_storico, numero_spia, num_previsione)
            if not previsione: continue
            
            risultati_per_posizione[pos]['previsioni_totali'] += 1
            
            for colpo in range(colpi):
                estrazione_futura = archivio[i + 1 + colpo]['numeri']
                numeri_vincenti = set(previsione).intersection(set(estrazione_futura))
                if len(numeri_vincenti) >= 1: risultati_per_posizione[pos]['ambate'] += 1
                if len(numeri_vincenti) >= 2: risultati_per_posizione[pos]['ambi'] += 1
                if len(numeri_vincenti) >= 3: risultati_per_posizione[pos]['terni'] += 1
                if len(numeri_vincenti) >= 1: break

    # (Il resto della funzione di stampa del riepilogo rimane invariato)
    output_string += "\n\n" + "="*80 + "\n"
    output_string += "  RIEPILOGO FINALE - PERFORMANCE PER POSIZIONE\n"
    output_string += "="*80 + "\n"
    header = f"{'Pos.':<5} | {'P. Testate':<10} | {'Ambate':<10} {'(% succ.)':<10} | {'Ambi':<8} | {'Terni':<8}\n"
    output_string += header
    output_string += "-" * len(header) + "\n"
    tabella_dati = []
    for pos, dati in risultati_per_posizione.items():
        previsioni = dati['previsioni_totali']
        if previsioni > 0:
            perc_successo_ambata = (dati['ambate'] / previsioni) * 100
            tabella_dati.append({'posizione': pos + 1, 'previsioni': previsioni, 'ambate': dati['ambate'], 'percentuale': perc_successo_ambata, 'ambi': dati['ambi'], 'terni': dati['terni']})
    tabella_ordinata = sorted(tabella_dati, key=lambda x: x['percentuale'], reverse=True)
    for res in tabella_ordinata:
        percentuale_str = f"({res['percentuale']:.2f}%)"
        output_string += f"{res['posizione']:<5} | {res['previsioni']:<10} | {res['ambate']:<10} {percentuale_str:<10} | {res['ambi']:<8} | {res['terni']:<8}\n"
    output_string += "-" * len(header) + "\n"
    if tabella_ordinata:
        output_string += f"\n==> POSIZIONE MIGLIORE CONSIGLIATA (per Ambata): {tabella_ordinata[0]['posizione']}° ESTRATTO\n"
    else:
        output_string += "\nNessuna previsione valida generata.\n"
    return output_string

def trova_previsioni_in_gioco(archivio, colpi, finestra, num_previsione):
    output_string = ""
    output_string += "\n" + "#"*70 + "\n"
    output_string += f"  RICERCA PREVISIONI ANCORA IN GIOCO\n"
    output_string += f"  Colpi Max: {colpi} | Finestra: {finestra} | Numeri Giocati: {num_previsione}\n"
    output_string += "#"*70 + "\n"

    statistiche = calcola_statistiche_complete(archivio)
    
    previsioni_attive_per_posizione = {pos: [] for pos in range(5)}
    tutti_i_numeri_attivi = [] # Lista per raccogliere le convergenze

    for pos in range(5):
        for i in range(len(archivio) - colpi, len(archivio)):
            if i < finestra: continue
            
            archivio_storico = archivio[i - finestra : i]
            estrazione_spia = archivio[i]
            numero_spia = estrazione_spia['numeri'][pos]
            data_generazione = estrazione_spia['data']
            
            previsione = genera_previsione_spia(archivio_storico, numero_spia, num_previsione)
            if not previsione: continue
                
            esito_trovato = False
            colpi_trascorsi = len(archivio) - 1 - i
            
            for colpo in range(colpi_trascorsi):
                if set(previsione).intersection(set(archivio[i + 1 + colpo]['numeri'])):
                    esito_trovato = True; break
            
            if not esito_trovato:
                previsione_info = {
                    'data': data_generazione, 'previsione': previsione, 'spia': numero_spia, 'pos_seme': pos + 1,
                    'colpi_trascorsi': colpi_trascorsi + 1, 'colpi_rimanenti': colpi - (colpi_trascorsi + 1)
                }
                previsioni_attive_per_posizione[pos].append(previsione_info)
                tutti_i_numeri_attivi.extend(previsione)

    has_active_predictions = False
    for pos, previsioni in previsioni_attive_per_posizione.items():
        if previsioni:
            has_active_predictions = True
            output_string += f"\n--- Previsioni Attive per POSIZIONE SEME {pos + 1} ---\n"
            for p in previsioni:
                output_string += "-" * 60 + "\n"
                output_string += f"> Generata il: {p['data']} (da Spia: {p['spia']}) - Colpi Rimasti: {p['colpi_rimanenti']}\n"
                header = f"{'Numero':<8} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
                output_string += header
                output_string += "-"*60 + "\n"
                for num in p['previsione']:
                    stats_num = statistiche.get(num, {})
                    output_string += f"{num:<8} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}\n"

    if not has_active_predictions:
        output_string += "\nNessuna previsione risulta ancora in gioco con i parametri specificati.\n"
    
    # --- SEZIONE CONVERGENZE REINSERITA E CORRETTA ---
    output_string += "\n\n" + "="*70 + "\n"
    output_string += "  ANALISI CONVERGENZE SUI NUMERI IN GIOCO\n"
    output_string += "="*70 + "\n"
    
    if not tutti_i_numeri_attivi:
        output_string += "Nessun numero in gioco su cui calcolare le convergenze.\n"
    else:
        frequenze = defaultdict(int)
        for num in tutti_i_numeri_attivi:
            frequenze[num] += 1
        
        convergenze = {num: freq for num, freq in frequenze.items() if freq > 1}
        
        if not convergenze:
            output_string += "Nessuna convergenza trovata (nessun numero appare in più di una previsione).\n"
        else:
            output_string += "I seguenti numeri sono apparsi in più previsioni attive (ordinati per forza):\n\n"
            header = f"{'Numero':<8} | {'Presente in':<12} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
            output_string += header
            output_string += "-"*70 + "\n"
            
            convergenze_ordinate = sorted(convergenze.items(), key=lambda x: x[1], reverse=True)
            for numero, freq in convergenze_ordinate:
                stats_num = statistiche.get(numero, {})
                output_string += f"{numero:<8} | {str(freq)+' previsioni':<12} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}\n"
            
            numero_top = convergenze_ordinate[0][0]
            output_string += f"\n==> NUMERO PIÙ FORTE PER CONVERGENZA: {numero_top}\n"

    output_string += "="*70 + "\n"
        
    return output_string

# ==============================================================================
# 3. CLASSE PER L'INTERFACCIA GRAFICA (GUI)
# ==============================================================================
class LottoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizzatore Spaziometrico Lotto")
        self.geometry("900x700")

        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._crea_widgets_input()
        self._crea_widgets_output()

    def _crea_widgets_input(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="Parametri di Analisi", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Ruota:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ruota_var = tk.StringVar()
        self.ruota_combobox = ttk.Combobox(input_frame, textvariable=self.ruota_var, values=list(RUOTE_MAP.keys()))
        self.ruota_combobox.current(0)
        self.ruota_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(input_frame, text="Colpi di Gioco:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.colpi_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.colpi_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(input_frame, text="Finestra Analisi:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.finestra_var = tk.StringVar(value="500")
        ttk.Entry(input_frame, textvariable=self.finestra_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(input_frame, text="Numeri in Previsione:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.num_numeri_var = tk.StringVar(value="3")
        ttk.Entry(input_frame, textvariable=self.num_numeri_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)

        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=10)
        
        ttk.Button(button_frame, text="Genera Previsione", command=self.run_prediction).pack(side=tk.LEFT, padx=5)
        # Il pulsante ora chiama direttamente il nuovo metodo 'run_backtest'
        ttk.Button(button_frame, text="Backtest Comparativo", command=self.run_backtest).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Mostra Previsioni in Gioco", command=self.run_in_gioco).pack(side=tk.LEFT, padx=5)
    
    def _crea_widgets_output(self):
        output_frame = ttk.LabelFrame(self.main_frame, text="Risultati", padding="10")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.main_frame.rowconfigure(2, weight=1)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=100, height=30, font=("Courier New", 9))
        self.output_text.pack(expand=True, fill="both")

    def _get_params(self):
        try:
            ruota = self.ruota_var.get()
            colpi = int(self.colpi_var.get())
            finestra = int(self.finestra_var.get())
            num_numeri = int(self.num_numeri_var.get())
            if not ruota or colpi <= 0 or finestra <= 0 or not (1 <= num_numeri <= 5):
                raise ValueError("Parametri non validi.")
            return ruota, colpi, finestra, num_numeri
        except (ValueError, tk.TclError):
            self.output_text.insert(tk.END, "[ERRORE] Controlla i parametri.\n")
            return None, None, None, None

    # NUOVA VERSIONE DELLA FUNZIONE BACKTEST, ora come metodo della classe
    def run_backtest(self):
        self.output_text.delete('1.0', tk.END)
        ruota, colpi, finestra, num_numeri = self._get_params()
        if not ruota: return

        # Chiediamo la durata del test
        try:
            user_input = simpledialog.askstring("Input", "Su quante delle ultime estrazioni vuoi eseguire il test?\n(es. 100, lascia vuoto per tutto l'archivio)", parent=self)
            if user_input is None: return # L'utente ha premuto Annulla
            num_test = int(user_input) if user_input else None
        except (ValueError, TypeError):
            self.output_text.insert(tk.END, "[ERRORE] Numero di test non valido.\n"); return

        archivio = carica_estrazioni_da_github(ruota, self.output_text)
        if archivio and len(archivio) > finestra + colpi:
            risultati_stringa = esegui_backtest_comparativo(archivio, ruota, colpi, finestra, num_numeri, num_test)
            self.output_text.insert(tk.END, risultati_stringa)
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti.\n")

    def run_prediction(self):
        self.output_text.delete('1.0', tk.END)
        ruota, _, finestra, num_numeri = self._get_params()
        if not ruota: return
        try:
            user_input = simpledialog.askstring("Input", "Quale posizione estratto usare come seme? (1-5)", parent=self)
            if user_input is None: return
            pos_scelta = int(user_input) - 1
            if not (0 <= pos_scelta <= 4): raise ValueError
        except (TypeError, ValueError):
            self.output_text.insert(tk.END, "[ERRORE] Posizione non valida.\n"); return
            
        archivio = carica_estrazioni_da_github(ruota, self.output_text)
        if archivio and len(archivio) > finestra:
            numero_spia = archivio[-1]['numeri'][pos_scelta]
            previsione = genera_previsione_spia(archivio[:-1], numero_spia, num_numeri)
            output_string = ""
            if previsione:
                statistiche = calcola_statistiche_complete(archivio)
                output_string += f"\nPREVISIONE GENERATA\n{'*'*60}\n"
                output_string += f"Ruota: {ruota.upper()}, Spia: {numero_spia} (da Pos. {pos_scelta+1})\n"
                output_string += "-"*60 + "\n"
                output_string += f"{'Numero':<8} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
                output_string += "-"*60 + "\n"
                for num in previsione:
                    stats_num = statistiche.get(num, {})
                    output_string += f"{num:<8} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}\n"
            else: 
                output_string += f"\nImpossibile generare una previsione per lo spia {numero_spia}.\n"
            self.output_text.insert(tk.END, output_string)
        else: 
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti.\n")

    def run_in_gioco(self):
        self.output_text.delete('1.0', tk.END)
        ruota, colpi, finestra, num_numeri = self._get_params()
        if not ruota: return
        archivio = carica_estrazioni_da_github(ruota, self.output_text)
        if archivio and len(archivio) > finestra + colpi:
            risultati_stringa = trova_previsioni_in_gioco(archivio, colpi, finestra, num_numeri)
            self.output_text.insert(tk.END, risultati_stringa)
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti.\n")

# ==============================================================================
# 4. BLOCCO DI ESECUZIONE PRINCIPALE
# ==============================================================================
if __name__ == "__main__":
    app = LottoApp()
    app.mainloop()