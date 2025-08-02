import os 
import requests
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog, messagebox

# ==============================================================================
# 1. LOGICA DI BACKEND
# ==============================================================================
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"

RUOTE_MAP = {
    'BARI': 'BA', 'CAGLIARI': 'CA', 'FIRENZE': 'FI', 'GENOVA': 'GE',
    'MILANO': 'MI', 'NAPOLI': 'NA', 'PALERMO': 'PA', 'ROMA': 'RO',
    'TORINO': 'TO', 'VENEZIA': 'VE', 'NAZIONALE': 'NZ'
}

def carica_singola_estrazione(sorgente_dati, nome_ruota, output_widget):
    """
    Carica i dati di una singola ruota da una fonte generica (URL o percorso locale).
    
    Args:
        sorgente_dati (str): La base del percorso. Se inizia con 'http', è un URL.
                             Altrimenti, è una cartella locale.
        nome_ruota (str): Il nome della ruota (es. 'BARI').
        output_widget: Il widget di testo per i log.

    Returns:
        list: Una lista di dizionari delle estrazioni.
    """
    nome_file = f"{nome_ruota.upper()}.txt"
    lines = []

    if output_widget:
        output_widget.insert(tk.END, f"[INFO] Caricamento dati per {nome_ruota.upper()}...\n")
        output_widget.see(tk.END)

    try:
        if sorgente_dati.startswith("http"):
            # Logica per caricamento da URL
            url = f"{sorgente_dati}/{nome_file}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
        else:
            # Logica per caricamento da file locale
            import os
            file_path = os.path.join(sorgente_dati, nome_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File non trovato: {file_path}")
            
            # Prova diversi encoding per robustezza
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
            read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        lines = f.read().strip().split('\n')
                    read_success = True
                    break
                except UnicodeDecodeError:
                    continue
            if not read_success:
                raise IOError(f"Impossibile leggere il file {file_path} con gli encoding noti.")
        
        if output_widget:
            output_widget.insert(tk.END, f"[OK] Dati per {nome_ruota.upper()} caricati.\n")

    except Exception as e:
        if output_widget:
            output_widget.insert(tk.END, f"[ERRORE] Impossibile caricare per {nome_ruota}: {e}\n")
        return []

    # La logica di parsing è identica per entrambi i casi
    estrazioni_trovate = []
    for ln, riga in enumerate(lines, 1):
        parti = riga.strip().split()
        if len(parti) == 7:
            try:
                numeri = [int(n) for n in parti[2:7]]
                data = parti[0]
                estrazioni_trovate.append({'numeri': numeri, 'data': data})
            except ValueError:
                if output_widget:
                    output_widget.insert(tk.END, f"[AVVISO] Riga {ln} per {nome_ruota} ignorata.\n")
    return estrazioni_trovate

def calcola_statistiche_complete(archivio):
    if not archivio: return {}
    stats = {n: {'freq': 0, 'rit_max': 0, 'last_seen': -1} for n in range(1, 91)}; tot_estr = len(archivio)
    for i, estrazione_data in enumerate(archivio):
        for numero in estrazione_data['numeri']:
            stats[numero]['freq'] += 1; ritardo_precedente = i - stats[numero]['last_seen'] - 1
            if ritardo_precedente > stats[numero]['rit_max']: stats[numero]['rit_max'] = ritardo_precedente
            stats[numero]['last_seen'] = i
    for n in range(1, 91):
        if stats[n]['last_seen'] == -1: stats[n]['rit_att'] = tot_estr; stats[n]['rit_max'] = tot_estr; stats[n]['ic'] = 0
        else:
            rit_att = tot_estr - 1 - stats[n]['last_seen']; stats[n]['rit_att'] = rit_att
            if rit_att > stats[n]['rit_max']: stats[n]['rit_max'] = rit_att
            if stats[n]['freq'] > 0: stats[n]['ic'] = (rit_att * stats[n]['freq']) / tot_estr
            else: stats[n]['ic'] = 0
    return stats

# --- Funzioni di generazione previsioni ---
def genera_previsione_spia(archivio, numero_spia, num_previsione):
    numeri_conseguenti = []
    for i in range(len(archivio) - 1):
        if numero_spia in archivio[i]['numeri']: numeri_conseguenti.extend(archivio[i+1]['numeri'])
    if not numeri_conseguenti: return None
    frequenze = defaultdict(int)
    for n in numeri_conseguenti: frequenze[n] += 1
    previsione_ordinata = sorted(frequenze.items(), key=lambda x: x[1], reverse=True)
    return [num for num, freq in previsione_ordinata[:num_previsione]]

def genera_previsione_spia_incrociata(archivio_fonte, archivio_dest, numero_spia, num_previsione):
    numeri_conseguenti = []; len_min = min(len(archivio_fonte), len(archivio_dest))
    for i in range(len_min - 1):
        if numero_spia in archivio_fonte[i]['numeri']: numeri_conseguenti.extend(archivio_dest[i+1]['numeri'])
    if not numeri_conseguenti: return None
    frequenze = defaultdict(int)
    for n in numeri_conseguenti: frequenze[n] += 1
    previsione_ordinata = sorted(frequenze.items(), key=lambda x: x[1], reverse=True)
    return [num for num, freq in previsione_ordinata[:num_previsione]]
    
def genera_previsione_armonica(archivio_fonte_finestra, archivio_dest_finestra, num_previsione_per_spia, numeri_spia_sorgente):
    """
    Versione CORRETTA e RESA PIU' FLESSIBILE. Ora elabora direttamente la finestra di archivio
    che le viene passata, senza fare ulteriori selezioni.
    """
    is_incrociato = (id(archivio_fonte_finestra) != id(archivio_dest_finestra))
    convergenze_totali = defaultdict(int)

    # 1. & 2. Genera una previsione per ognuno dei 5 numeri spia
    for spia in numeri_spia_sorgente:
        # Usa le funzioni spia come motore interno, passando direttamente la finestra ricevuta
        prev = genera_previsione_spia_incrociata(archivio_fonte_finestra, archivio_dest_finestra, spia, num_previsione_per_spia) if is_incrociato else genera_previsione_spia(archivio_fonte_finestra, spia, num_previsione_per_spia)
        
        # 3. Mette tutti i numeri nel "calderone" e li conta
        if prev:
            for numero in prev:
                convergenze_totali[numero] += 1
    
    # 4. Ordina TUTTI i numeri trovati in base alla loro frequenza
    previsione_completa_ordinata = sorted(convergenze_totali.items(), key=lambda x: x[1], reverse=True)
    
    # 5. Estrae i primi 5 numeri più frequenti
    previsione_finale = previsione_completa_ordinata[:5]
    
    return previsione_finale

def esegui_backtest_comparativo(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione, num_test=None):
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1]); output_string = "\n" + "#"*80 + "\n"
    if is_incrociato: output_string += f"  AVVIO BACKTEST POSIZIONALE (GIOCO SU DOPPIA RUOTA)\n"; output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else: output_string += f"  AVVIO BACKTEST COMPARATIVO - RUOTA SINGOLA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Colpi: {colpi} | Finestra: {finestra} | Numeri Giocati: {num_previsione}\n"
    if num_test: output_string += f"  Test limitato alle ultime {num_test} occasioni.\n"
    output_string += "#"*80 + "\n"; len_min = min(len(archivio_fonte), len(archivio_dest)); start_index = finestra
    if num_test and len_min > num_test + finestra + colpi: start_index = len_min - num_test - colpi
    risultati_per_posizione = {pos: defaultdict(int) for pos in range(5)}
    for pos in range(5):
        for i in range(start_index, len_min - colpi):
            numero_spia = archivio_fonte[i]['numeri'][pos]
            previsione = genera_previsione_spia_incrociata(archivio_fonte[i-finestra:i], archivio_dest[i-finestra:i], numero_spia, num_previsione) if is_incrociato else genera_previsione_spia(archivio_fonte[i-finestra:i], numero_spia, num_previsione)
            if not previsione: continue
            risultati_per_posizione[pos]['previsioni_totali'] += 1
            for colpo in range(colpi):
                if i + 1 + colpo >= len_min: break
                estrazione_futura_dest = set(archivio_dest[i + 1 + colpo]['numeri']); estrazione_futura_fonte = set(archivio_fonte[i + 1 + colpo]['numeri']) if is_incrociato else estrazione_futura_dest
                vincita_dest = set(previsione).intersection(estrazione_futura_dest); vincita_fonte = set(previsione).intersection(estrazione_futura_fonte); numeri_vincenti = vincita_dest.union(vincita_fonte)
                if len(numeri_vincenti) >= 1: risultati_per_posizione[pos]['ambate'] += 1
                if len(numeri_vincenti) >= 2: risultati_per_posizione[pos]['ambi'] += 1
                if len(numeri_vincenti) >= 3: risultati_per_posizione[pos]['terni'] += 1
                if len(numeri_vincenti) >= 1: break
    output_string += "\n\n" + "="*80 + "\n"; output_string += "  RIEPILOGO FINALE - PERFORMANCE PER POSIZIONE\n"; output_string += "="*80 + "\n"
    header = f"{'Pos.':<5} | {'P. Testate':<10} | {'Ambate':<10} {'(% succ.)':<10} | {'Ambi':<8} | {'Terni':<8}\n"; output_string += header; output_string += "-" * len(header) + "\n"; tabella_dati = []
    for pos, dati in risultati_per_posizione.items():
        previsioni = dati['previsioni_totali']
        if previsioni > 0: perc_successo_ambata = (dati['ambate'] / previsioni) * 100; tabella_dati.append({'posizione': pos + 1, 'previsioni': previsioni, 'ambate': dati['ambate'], 'percentuale': perc_successo_ambata, 'ambi': dati['ambi'], 'terni': dati['terni']})
    tabella_ordinata = sorted(tabella_dati, key=lambda x: x['percentuale'], reverse=True)
    for res in tabella_ordinata: percentuale_str = f"({res['percentuale']:.2f}%)"; output_string += f"{res['posizione']:<5} | {res['previsioni']:<10} | {res['ambate']:<10} {percentuale_str:<10} | {res['ambi']:<8} | {res['terni']:<8}\n"
    output_string += "-" * len(header) + "\n"
    if tabella_ordinata: output_string += f"\n==> POSIZIONE MIGLIORE CONSIGLIATA (per Ambata): {tabella_ordinata[0]['posizione']}° ESTRATTO\n"
    else: output_string += "\nNessuna previsione valida generata.\n"
    return output_string

def esegui_backtest_armonico(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione, num_test=None):
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1]); output_string = "\n" + "#"*80 + "\n"; output_string += f"  AVVIO BACKTEST SUL METODO DELL'ANALISI ARMONICA\n"
    if is_incrociato: output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else: output_string += f"  RUOTA ANALIZZATA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Colpi: {colpi} | Finestra: {finestra} | Numeri Giocati: 5 (i più frequenti)\n" # Modificato per chiarezza
    if num_test: output_string += f"  Test limitato alle ultime {num_test} occasioni.\n"
    output_string += "#"*80 + "\n"; len_min = min(len(archivio_fonte), len(archivio_dest)); start_index = finestra
    if num_test and len_min > num_test + finestra + colpi: start_index = len_min - num_test - colpi
    risultati = defaultdict(int)
    for i in range(start_index, len_min - colpi):
        numeri_spia_sorgente = archivio_fonte[i]['numeri']
        
        # --- MODIFICA QUI ---
        # La chiamata ora ha 4 parametri invece di 5 (ho tolto 'finestra')
        previsione_armonica_tuples = genera_previsione_armonica(
            archivio_fonte[i-finestra:i], 
            archivio_dest[i-finestra:i], 
            num_previsione, # Questo è 'num_previsione_per_spia'
            numeri_spia_sorgente
        )
        # --- FINE MODIFICA ---

        if not previsione_armonica_tuples: continue
        previsione = [num for num, freq in previsione_armonica_tuples]; risultati['previsioni_totali'] += 1
        for colpo in range(colpi):
            if i + 1 + colpo >= len_min: break
            estrazione_futura_dest = set(archivio_dest[i + 1 + colpo]['numeri']); estrazione_futura_fonte = set(archivio_fonte[i + 1 + colpo]['numeri']) if is_incrociato else estrazione_futura_dest
            numeri_vincenti = set(previsione).intersection(estrazione_futura_dest.union(estrazione_futura_fonte))
            if len(numeri_vincenti) >= 1: risultati['ambate'] += 1
            if len(numeri_vincenti) >= 2: risultati['ambi'] += 1
            if len(numeri_vincenti) >= 3: risultati['terni'] += 1
            if len(numeri_vincenti) >= 1: break
    output_string += "\nRIEPILOGO STATISTICO DEL METODO ARMONICO:\n\n"; previsioni_tot = risultati['previsioni_totali']
    if previsioni_tot == 0: output_string += "Nessuna previsione armonica valida è stata generata nel periodo di test.\n"; return output_string
    perc_ambata = (risultati['ambate'] / previsioni_tot) * 100
    output_string += f"  Previsioni Valide Generate: {previsioni_tot}\n"; output_string += f"  Successi (Ambata):          {risultati['ambate']} ({perc_ambata:.2f}%)\n"
    output_string += f"  Successi (Ambo):            {risultati['ambi']}\n"; output_string += f"  Successi (Terno):           {risultati['terni']}\n"; output_string += "-"*50 + "\n"
    output_string += "Questo test dimostra la validità statistica del metodo con i parametri forniti.\n"; return output_string

def trova_previsioni_in_gioco(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione):
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1]); output_string = ""; output_string += "\n" + "#"*70 + "\n"
    if is_incrociato: output_string += f"  RICERCA PREVISIONI IN GIOCO (GIOCO SU DOPPIA RUOTA)\n"; output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else: output_string += f"  RICERCA PREVISIONI ANCORA IN GIOCO - RUOTA SINGOLA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Colpi Max: {colpi} | Finestra: {finestra} | Numeri Giocati: {num_previsione}\n"; output_string += "#"*70 + "\n"
    statistiche_dest = calcola_statistiche_complete(archivio_dest); previsioni_attive_per_posizione = {pos: [] for pos in range(5)}; tutti_i_numeri_attivi = []; len_min = min(len(archivio_fonte), len(archivio_dest))
    for pos in range(5):
        for i in range(len_min - colpi, len_min):
            if i < finestra: continue
            numero_spia = archivio_fonte[i]['numeri'][pos]; data_generazione = archivio_fonte[i]['data']
            previsione = genera_previsione_spia_incrociata(archivio_fonte[i-finestra:i], archivio_dest[i-finestra:i], numero_spia, num_previsione) if is_incrociato else genera_previsione_spia(archivio_fonte[i-finestra:i], numero_spia, num_previsione)
            if not previsione: continue
            esito_trovato = False; colpi_trascorsi = len_min - 1 - i
            for colpo in range(colpi_trascorsi):
                if i + 1 + colpo < len_min:
                    estrazione_futura_dest = set(archivio_dest[i + 1 + colpo]['numeri']); estrazione_futura_fonte = set(archivio_fonte[i + 1 + colpo]['numeri']) if is_incrociato else estrazione_futura_dest
                    if set(previsione).intersection(estrazione_futura_dest.union(estrazione_futura_fonte)): esito_trovato = True; break
            if not esito_trovato:
                previsione_info = {'data': data_generazione, 'previsione': previsione, 'spia': numero_spia, 'pos_seme': pos + 1, 'colpi_rimanenti': colpi - (colpi_trascorsi + 1)}
                previsioni_attive_per_posizione[pos].append(previsione_info); tutti_i_numeri_attivi.extend(previsione)
    has_active_predictions = False
    for pos, previsioni in previsioni_attive_per_posizione.items():
        if previsioni:
            has_active_predictions = True; output_string += f"\n--- Previsioni Attive per POSIZIONE SEME {pos + 1} ---\n"
            for p in previsioni:
                output_string += "-" * 70 + "\n"; output_string += f"> Generata il: {p['data']} (da Spia: {p['spia']}) - Colpi Rimasti: {p['colpi_rimanenti']}\n"
                ambata_suggerita = max(p['previsione'], key=lambda x: statistiche_dest.get(x, {}).get('ic', 0))
                header = f"{'Numero':<8} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"; output_string += header; output_string += "-"*70 + "\n"
                for num in p['previsione']:
                    stats_num = statistiche_dest.get(num, {}); tag = " <<< NUMERO FORTE" if num == ambata_suggerita else "" # MODIFICATO QUI
                    output_string += f"{num:<8} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}{tag}\n"
    if not has_active_predictions: output_string += "\nNessuna previsione posizionale risulta ancora in gioco.\n"
    output_string += "\n\n" + "="*70 + "\n"; output_string += "  ANALISI CONVERGENZE SUI NUMERI IN GIOCO (POSIZIONALI)\n"; output_string += "="*70 + "\n"
    if not tutti_i_numeri_attivi: output_string += "Nessun numero in gioco.\n"
    else:
        frequenze = defaultdict(int);
        for num in tutti_i_numeri_attivi: frequenze[num] += 1
        convergenze = {num: freq for num, freq in frequenze.items() if freq > 1}
        if not convergenze: output_string += "Nessuna convergenza trovata.\n"
        else:
            output_string += "I seguenti numeri sono apparsi in più previsioni attive (ordinati per forza):\n\n"; header = f"{'Numero':<8} | {'Presente in':<12} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
            output_string += header; output_string += "-"*70 + "\n"; convergenze_ordinate = sorted(convergenze.items(), key=lambda x: (x[1], statistiche_dest.get(x[0], {}).get('ic', 0)), reverse=True)
            for numero, freq in convergenze_ordinate:
                stats_num = statistiche_dest.get(numero, {}); output_string += f"{numero:<8} | {str(freq)+' previsioni':<12} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}\n"
            if convergenze_ordinate: numero_top = convergenze_ordinate[0][0]; output_string += f"\n==> NUMERO PIÙ FORTE PER CONVERGENZA: {numero_top}\n"
    output_string += "="*70 + "\n"; return output_string


def esegui_analisi_armonica(archivio_fonte, archivio_dest, nomi_ruote, finestra, num_previsione_per_spia):
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1])
    ultima_estrazione_fonte = archivio_fonte[-1] if archivio_fonte else None
    if not ultima_estrazione_fonte:
        return "Archivio fonte vuoto, impossibile generare previsione armonica."
        
    numeri_spia_armonici = ultima_estrazione_fonte['numeri']
    data_spia = ultima_estrazione_fonte['data']
    
    output_string = "\n" + "#"*80 + "\n"
    output_string += "  AVVIO ANALISI ARMONICA SU ULTIMA ESTRAZIONE\n"
    if is_incrociato:
        output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else:
        output_string += f"  RUOTA ANALIZZATA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Estrazione del {data_spia} ({nomi_ruote[0].upper()}) -> Spie: {sorted(numeri_spia_armonici)}\n"
    output_string += "#"*80 + "\n"
    
    archivio_fonte_finestra = archivio_fonte[-finestra:]
    archivio_dest_finestra = archivio_dest[-finestra:]
    convergenze_ordinate = genera_previsione_armonica(
        archivio_fonte_finestra, 
        archivio_dest_finestra, 
        num_previsione_per_spia, 
        numeri_spia_armonici
    )
    
    if not convergenze_ordinate:
        output_string += "Nessuna convergenza armonica trovata.\n"
        return output_string
        
    output_string += "RIEPILOGO CONVERGENZE ARMONICHE (ordinate per forza):\n\n"
    previsione_raccomandata = [num for num, freq in convergenze_ordinate]
    
    statistiche_dest = calcola_statistiche_complete(archivio_dest)
    ambata_suggerita = max(previsione_raccomandata, key=lambda x: statistiche_dest.get(x, {}).get('ic', 0)) if previsione_raccomandata else None
    
    header = f"{'Numero':<8} | {'Convergenza':<12} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
    output_string += header
    output_string += "-"*70 + "\n"
    
    for numero, freq in convergenze_ordinate:
        stats_num = statistiche_dest.get(numero, {})
        tag = " <<< NUMERO FORTE" if numero == ambata_suggerita else "" # MODIFICATO QUI
        freq_text = f"{freq} su {len(numeri_spia_armonici)}"
        output_string += f"{numero:<8} | {freq_text:<12} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}{tag}\n"
        
    output_string += "\n" + "="*70 + "\n"
    output_string += f"==> PREVISIONE ARMONICA CONSIGLIATA: {' - '.join(map(str, sorted(previsione_raccomandata)))}\n"
    output_string += "="*70 + "\n"
    return output_string

def trova_previsioni_armoniche_in_gioco(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione_per_spia):
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1])
    output_string = "\n" + "#"*70 + "\n"
    output_string += "  RICERCA PREVISIONI ARMONICHE ANCORA IN GIOCO\n"
    if is_incrociato:
        output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else:
        output_string += f"  RUOTA ANALIZZATA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Colpi Max: {colpi} | Finestra: {finestra} | Numeri per Spia: {num_previsione_per_spia}\n"
    output_string += "#"*70 + "\n\n"
    output_string += f"--- Analisi delle ultime {colpi} estrazioni per trovare previsioni attive ---\n"
    
    statistiche_dest = calcola_statistiche_complete(archivio_dest)
    previsioni_attive = []
    tutti_i_numeri_attivi = []
    len_min = min(len(archivio_fonte), len(archivio_dest))
    
    start_loop_index = max(finestra, len_min - colpi)
    
    for i in range(start_loop_index, len_min):
        archivio_storico_fonte = archivio_fonte[i - finestra : i]
        archivio_storico_dest = archivio_dest[i - finestra : i]
        
        numeri_spia_sorgente = archivio_fonte[i]['numeri']
        data_generazione = archivio_fonte[i]['data']
        
        output_string += f"\n> Analizzo estrazione del {data_generazione} (Indice {i}). Spie: {sorted(numeri_spia_sorgente)}\n"
        
        previsione_tuples = genera_previsione_armonica(archivio_storico_fonte, archivio_storico_dest, num_previsione_per_spia, numeri_spia_sorgente)
        
        if not previsione_tuples:
            output_string += "  - Nessuna previsione generata. Ignoro.\n"
            continue
        
        previsione = [num for num, freq in previsione_tuples]
        output_string += f"  - Previsione generata: {sorted(previsione)}\n"
        
        esito_trovato = False
        data_esito = ""
        esito_colpo_n = 0
        
        for colpo_idx in range(len_min - 1 - i):
            estrazione_futura_idx = i + 1 + colpo_idx
            estrazione_futura_dest_set = set(archivio_dest[estrazione_futura_idx]['numeri'])
            estrazione_futura_fonte_set = set(archivio_fonte[estrazione_futura_idx]['numeri']) if is_incrociato else estrazione_futura_dest_set
            numeri_vincenti = set(previsione).intersection(estrazione_futura_dest_set.union(estrazione_futura_fonte_set))
            
            if numeri_vincenti:
                esito_trovato = True
                data_esito = archivio_dest[estrazione_futura_idx]['data']
                esito_colpo_n = colpo_idx + 1
                output_string += f"  - ESITO TROVATO al colpo {esito_colpo_n} (data {data_esito}). Numero/i: {list(numeri_vincenti)}. Previsione chiusa.\n"
                break
        
        if not esito_trovato:
            colpi_trascorsi = len_min - 1 - i
            colpi_rimanenti = colpi - colpi_trascorsi
            if colpi_rimanenti > 0:
                output_string += f"  - ESITO NON TROVATO. La previsione è ATTIVA con {colpi_rimanenti} colpi rimanenti.\n"
                previsione_info = {'data': data_generazione, 'previsione': previsione, 'spie': numeri_spia_sorgente, 'colpi_rimanenti': colpi_rimanenti}
                previsioni_attive.append(previsione_info)
                tutti_i_numeri_attivi.extend(previsione)
            else:
                 output_string += f"  - ESITO NON TROVATO, ma i colpi sono esauriti. Previsione chiusa.\n"


    output_string += "\n\n" + "="*70 + "\n"
    output_string += "  RIEPILOGO PREVISIONI ARMONICHE ANCORA ATTIVE\n"
    output_string += "="*70 + "\n"
    if not previsioni_attive:
        output_string += "\nNessuna previsione armonica risulta ancora in gioco.\n"
    else:
        for p in previsioni_attive:
            output_string += "-" * 70 + "\n"
            output_string += f"> Generata il: {p['data']} (da spie {sorted(p['spie'])}) - Colpi Rimasti: {p['colpi_rimanenti']}\n"
            ambata_suggerita = max(p['previsione'], key=lambda x: statistiche_dest.get(x, {}).get('ic', 0)) if p['previsione'] else None
            header = f"{'Numero':<8} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
            output_string += header
            output_string += "-"*70 + "\n"
            for num in sorted(p['previsione']):
                stats_num = statistiche_dest.get(num, {})
                tag = " <<< NUMERO FORTE" if num == ambata_suggerita else "" # MODIFICATO QUI
                output_string += f"{num:<8} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}{tag}\n"
    
    # ... la parte sulle convergenze resta uguale e va bene ...
    output_string += "\n\n" + "="*70 + "\n"
    output_string += "  ANALISI CONVERGENZE SULLE PREVISIONI ARMONICHE IN GIOCO\n"
    output_string += "="*70 + "\n"
    if not tutti_i_numeri_attivi:
        output_string += "Nessun numero in gioco.\n"
    else:
        frequenze = defaultdict(int)
        for num in tutti_i_numeri_attivi:
            frequenze[num] += 1
        convergenze = {num: freq for num, freq in frequenze.items() if freq > 1}
        if not convergenze:
            output_string += "Nessuna convergenza trovata.\n"
        else:
            output_string += "I seguenti numeri sono apparsi in più previsioni armoniche attive (ordinati per forza):\n\n"
            header = f"{'Numero':<8} | {'Presente in':<12} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"
            output_string += header
            output_string += "-"*70 + "\n"
            convergenze_ordinate = sorted(convergenze.items(), key=lambda x: (x[1], statistiche_dest.get(x[0], {}).get('ic', 0)), reverse=True)
            for numero, freq in convergenze_ordinate:
                stats_num = statistiche_dest.get(numero, {})
                output_string += f"{numero:<8} | {str(freq)+' previsioni':<12} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}\n"
            if convergenze_ordinate:
                numero_top = convergenze_ordinate[0][0]
                output_string += f"\n==> NUMERO PIÙ FORTE PER CONVERGENZA: {numero_top}\n"
    output_string += "="*70 + "\n"
    return output_string

def esegui_backtest_numero_forte_armonico(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione_per_spia, num_test=None):
    """
    NUOVO BACKTEST: Confronta il successo della cinquina armonica con il successo del singolo 'Numero Forte'.
    """
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1])
    output_string = "\n" + "#"*80 + "\n"
    output_string += f"  AVVIO BACKTEST COMPARATIVO (CINQUINA vs NUMERO FORTE)\n"
    output_string += f"  METODO: ARMONICO\n"
    if is_incrociato:
        output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else:
        output_string += f"  RUOTA ANALIZZATA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Colpi: {colpi} | Finestra: {finestra}\n"
    if num_test:
        output_string += f"  Test limitato alle ultime {num_test} occasioni.\n"
    output_string += "#"*80 + "\n"

    len_min = min(len(archivio_fonte), len(archivio_dest))
    start_index = finestra
    if num_test and len_min > num_test + finestra + colpi:
        start_index = len_min - num_test - colpi

    risultati = defaultdict(int)
    for i in range(start_index, len_min - colpi):
        numeri_spia_sorgente = archivio_fonte[i]['numeri']
        
        # Genera previsione armonica
        previsione_tuples = genera_previsione_armonica(archivio_fonte[i-finestra:i], archivio_dest[i-finestra:i], num_previsione_per_spia, numeri_spia_sorgente)
        if not previsione_tuples:
            continue
        
        risultati['previsioni_totali'] += 1
        previsione_cinquina = [num for num, freq in previsione_tuples]

        # Identifica il 'Numero Forte' BASATO SULLO STORICO DI QUEL MOMENTO
        statistiche_storiche_dest = calcola_statistiche_complete(archivio_dest[:i])
        numero_forte = max(previsione_cinquina, key=lambda x: statistiche_storiche_dest.get(x, {}).get('ic', 0))

        vincita_cinquina = False
        vincita_forte = False

        for colpo in range(colpi):
            if i + 1 + colpo >= len_min: break
            
            estrazione_futura_dest = set(archivio_dest[i + 1 + colpo]['numeri'])
            estrazione_futura_fonte = set(archivio_fonte[i + 1 + colpo]['numeri']) if is_incrociato else estrazione_futura_dest
            estrazione_completa = estrazione_futura_dest.union(estrazione_futura_fonte)

            if not vincita_cinquina and set(previsione_cinquina).intersection(estrazione_completa):
                risultati['successi_cinquina'] += 1
                vincita_cinquina = True

            if not vincita_forte and numero_forte in estrazione_completa:
                risultati['successi_forte'] += 1
                vincita_forte = True
            
            # Se entrambi hanno vinto, possiamo passare alla prossima previsione
            if vincita_cinquina and vincita_forte:
                break
    
    output_string += "\nRIEPILOGO STATISTICO COMPARATIVO:\n\n"
    previsioni_tot = risultati['previsioni_totali']
    if previsioni_tot == 0:
        output_string += "Nessuna previsione valida generata nel periodo di test.\n"
        return output_string

    perc_cinquina = (risultati['successi_cinquina'] / previsioni_tot) * 100
    perc_forte = (risultati['successi_forte'] / previsioni_tot) * 100
    
    output_string += f"  Previsioni Valide Generate: {previsioni_tot}\n"
    output_string += "-"*50 + "\n"
    output_string += f"  Successi della Cinquina:    {risultati['successi_cinquina']} ({perc_cinquina:.2f}%)\n"
    output_string += f"  Successi del 'Numero Forte': {risultati['successi_forte']} ({perc_forte:.2f}%)\n"
    output_string += "-"*50 + "\n"
    output_string += "Questo test confronta la validità della previsione completa rispetto al singolo numero suggerito.\n"
    return output_string


def esegui_backtest_numero_forte_posizionale(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione, num_test=None):
    """
    NUOVO BACKTEST: Per ogni posizione seme, confronta il successo della cinquina con il successo del 'Numero Forte'.
    """
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1])
    output_string = "\n" + "#"*80 + "\n"
    output_string += f"  AVVIO BACKTEST COMPARATIVO (CINQUINA vs NUMERO FORTE)\n"
    output_string += f"  METODO: POSIZIONALE\n"
    if is_incrociato:
        output_string += f"  RUOTA SPIA: {nomi_ruote[0].upper()} | RUOTE GIOCO: {nomi_ruote[0].upper()} e {nomi_ruote[1].upper()}\n"
    else:
        output_string += f"  RUOTA ANALIZZATA: {nomi_ruote[0].upper()}\n"
    output_string += f"  Colpi: {colpi} | Finestra: {finestra}\n"
    if num_test:
        output_string += f"  Test limitato alle ultime {num_test} occasioni.\n"
    output_string += "#"*80 + "\n"

    len_min = min(len(archivio_fonte), len(archivio_dest))
    start_index = finestra
    if num_test and len_min > num_test + finestra + colpi:
        start_index = len_min - num_test - colpi

    risultati_per_posizione = {pos: defaultdict(int) for pos in range(5)}

    for pos in range(5):
        for i in range(start_index, len_min - colpi):
            numero_spia = archivio_fonte[i]['numeri'][pos]
            previsione_cinquina = genera_previsione_spia_incrociata(archivio_fonte[i-finestra:i], archivio_dest[i-finestra:i], numero_spia, num_previsione) if is_incrociato else genera_previsione_spia(archivio_fonte[i-finestra:i], numero_spia, num_previsione)
            
            if not previsione_cinquina:
                continue

            risultati_per_posizione[pos]['previsioni_totali'] += 1
            
            statistiche_storiche_dest = calcola_statistiche_complete(archivio_dest[:i])
            numero_forte = max(previsione_cinquina, key=lambda x: statistiche_storiche_dest.get(x, {}).get('ic', 0))

            vincita_cinquina = False
            vincita_forte = False

            for colpo in range(colpi):
                if i + 1 + colpo >= len_min: break
                
                estrazione_futura_dest = set(archivio_dest[i + 1 + colpo]['numeri'])
                estrazione_futura_fonte = set(archivio_fonte[i + 1 + colpo]['numeri']) if is_incrociato else estrazione_futura_dest
                estrazione_completa = estrazione_futura_dest.union(estrazione_futura_fonte)

                if not vincita_cinquina and set(previsione_cinquina).intersection(estrazione_completa):
                    risultati_per_posizione[pos]['successi_cinquina'] += 1
                    vincita_cinquina = True

                if not vincita_forte and numero_forte in estrazione_completa:
                    risultati_per_posizione[pos]['successi_forte'] += 1
                    vincita_forte = True
                
                if vincita_cinquina and vincita_forte:
                    break

    output_string += "\nRIEPILOGO FINALE - PERFORMANCE PER POSIZIONE\n"
    header = f"{'Pos.':<5} | {'P. Testate':<10} | {'Succ. Cinquina':<15} | {'Succ. N. Forte':<15}\n"
    output_string += header
    output_string += "-" * len(header) + "\n"
    
    for pos, dati in risultati_per_posizione.items():
        previsioni = dati['previsioni_totali']
        if previsioni > 0:
            perc_cinquina = (dati['successi_cinquina'] / previsioni) * 100
            perc_forte = (dati['successi_forte'] / previsioni) * 100
            succ_cinquina_str = f"{dati['successi_cinquina']} ({perc_cinquina:.2f}%)"
            succ_forte_str = f"{dati['successi_forte']} ({perc_forte:.2f}%)"
            output_string += f"{pos + 1:<5} | {previsioni:<10} | {succ_cinquina_str:<15} | {succ_forte_str:<15}\n"

    output_string += "-" * len(header) + "\n"
    return output_string

def trova_numeri_forti_in_gioco(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_previsione_per_spia, num_previsione_posizionale):
    """
    VERSIONE MIGLIORATA: Cerca i 'Numeri Forti' ancora attivi e li ordina per
    colpi rimanenti, rendendo la tabella più leggibile e strategica.
    """
    is_incrociato = (nomi_ruote[0] != nomi_ruote[1])
    output_string = "\n" + "#"*80 + "\n"
    output_string += f"  CRUSCOTTO DEI NUMERI FORTI ANCORA IN GIOCO\n"
    output_string += f"  (Ordinati per 'Colpi Rimanenti' e 'I.C.')\n"
    output_string += "#"*80 + "\n"

    statistiche_globali_dest = calcola_statistiche_complete(archivio_dest)
    numeri_forti_attivi = []
    len_min = min(len(archivio_fonte), len(archivio_dest))
    start_loop_index = max(finestra, len_min - colpi)

    # --- FASE 1 & 2: Ricerca combinata dei Numeri Forti ---
    for i in range(start_loop_index, len_min):
        # --- Metodo Armonico ---
        previsione_tuples_arm = genera_previsione_armonica(archivio_fonte[i-finestra:i], archivio_dest[i-finestra:i], num_previsione_per_spia, archivio_fonte[i]['numeri'])
        if previsione_tuples_arm:
            cinquina_arm = [num for num, freq in previsione_tuples_arm]
            stat_storiche = calcola_statistiche_complete(archivio_dest[:i])
            num_forte_arm = max(cinquina_arm, key=lambda x: stat_storiche.get(x, {}).get('ic', 0))
            
            esito_trovato_arm = False
            for colpo_idx in range(len_min - 1 - i):
                estrazione_completa = set(archivio_dest[i + 1 + colpo_idx]['numeri']).union(set(archivio_fonte[i + 1 + colpo_idx]['numeri']) if is_incrociato else set())
                if num_forte_arm in estrazione_completa:
                    esito_trovato_arm = True
                    break
            
            if not esito_trovato_arm:
                colpi_rimanenti = colpi - (len_min - 1 - i)
                if colpi_rimanenti > 0:
                    numeri_forti_attivi.append({'numero': num_forte_arm, 'data': archivio_fonte[i]['data'], 'colpi': colpi_rimanenti, 'metodo': 'Armonico'})

        # --- Metodo Posizionale ---
        for pos in range(5):
            numero_spia = archivio_fonte[i]['numeri'][pos]
            cinquina_pos = genera_previsione_spia_incrociata(archivio_fonte[i-finestra:i], archivio_dest[i-finestra:i], numero_spia, num_previsione_posizionale) if is_incrociato else genera_previsione_spia(archivio_fonte[i-finestra:i], numero_spia, num_previsione_posizionale)
            if not cinquina_pos: continue

            stat_storiche = calcola_statistiche_complete(archivio_dest[:i])
            num_forte_pos = max(cinquina_pos, key=lambda x: stat_storiche.get(x, {}).get('ic', 0))

            esito_trovato_pos = False
            for colpo_idx in range(len_min - 1 - i):
                estrazione_completa = set(archivio_dest[i + 1 + colpo_idx]['numeri']).union(set(archivio_fonte[i + 1 + colpo_idx]['numeri']) if is_incrociato else set())
                if num_forte_pos in estrazione_completa:
                    esito_trovato_pos = True
                    break
            
            if not esito_trovato_pos:
                colpi_rimanenti = colpi - (len_min - 1 - i)
                if colpi_rimanenti > 0:
                     numeri_forti_attivi.append({'numero': num_forte_pos, 'data': archivio_fonte[i]['data'], 'colpi': colpi_rimanenti, 'metodo': f'Posiz.(P{pos+1})'})

    # --- FASE 3: Analisi e Stampa ---
    if not numeri_forti_attivi:
        output_string += "\nNessun 'Numero Forte' risulta ancora in gioco.\n"
        return output_string

    # NUOVO ORDINAMENTO: Per colpi rimanenti (crescente) e poi per Indice di Convenienza (decrescente)
    numeri_forti_ordinati = sorted(
        numeri_forti_attivi, 
        key=lambda nf: (nf['colpi'], -statistiche_globali_dest.get(nf['numero'], {}).get('ic', 0))
    )
    
    output_string += "\nI seguenti 'Numeri Forti' sono ancora attivi:\n\n"
    header = f"{'Numero':<8} | {'Colpi Rim.':<12} | {'Rit.Att':<8} | {'I.C.':<8} | {'Origine'}\n"
    output_string += header
    output_string += "-" * (len(header) + 20) + "\n"

    for nf in numeri_forti_ordinati:
        stats = statistiche_globali_dest.get(nf['numero'], {})
        origine_str = f"{nf['metodo']} del {nf['data']}"
        output_string += f"{nf['numero']:<8} | {nf['colpi']:<12} | {stats.get('rit_att', 'N/A'):<8} | {stats.get('ic', 0):<8.2f} | {origine_str}\n"

    # Suggerimento del numero più "urgente"
    if numeri_forti_ordinati:
        numero_top = numeri_forti_ordinati[0]['numero']
        output_string += f"\n\n==> NUMERO PIU' 'URGENTE' (meno colpi rimasti): {numero_top}\n"

    return output_string

# ==============================================================================
# CLASSE PER L'INTERFACCIA GRAFICA (GUI)
# ==============================================================================
class LottoApp(tk.Tk):
    # --- METODO COSTRUTTORE ---
    def __init__(self):
        super().__init__()
        # Imposta il titolo di default iniziale
        self.title("Analizzatore Ciclo-Spazio - Convergenze Armoniche - Created by Max Lotto -")
        
        # Configura la finestra principale per espandersi
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Crea un frame principale che conterrà tutto
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Configura il main_frame per espandere i suoi contenuti
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(3, weight=1) # La riga 4 (output) si espanderà in altezza

        # Ora chiama i metodi per creare i widget
        self._crea_widgets_input()
        self._crea_widgets_output()
        
    def _crea_widgets_input(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="Parametri di Analisi", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        # Sezione Fonte Dati
        fonte_dati_frame = ttk.LabelFrame(input_frame, text="Fonte Dati", padding="5")
        fonte_dati_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        fonte_dati_frame.columnconfigure(2, weight=1)
        self.source_type_var = tk.StringVar(value="online")
        rb_online = ttk.Radiobutton(fonte_dati_frame, text="Online (GitHub)", variable=self.source_type_var, value="online", command=self._aggiorna_stato_sorgente_dati)
        rb_online.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        rb_locale = ttk.Radiobutton(fonte_dati_frame, text="Cartella Locale", variable=self.source_type_var, value="locale", command=self._aggiorna_stato_sorgente_dati)
        rb_locale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        local_path_frame = ttk.Frame(fonte_dati_frame)
        local_path_frame.grid(row=0, column=2, sticky="ew", padx=5)
        local_path_frame.columnconfigure(0, weight=1)
        self.local_path_var = tk.StringVar(value="")
        self.local_path_entry = ttk.Entry(local_path_frame, textvariable=self.local_path_var, width=50, state="disabled")
        self.local_path_entry.grid(row=0, column=0, sticky="ew")
        self.browse_button = ttk.Button(local_path_frame, text="Sfoglia...", command=self.browse_folder, state="disabled")
        self.browse_button.grid(row=0, column=1, padx=(5, 0))
        
        # Parametri Ruote e Gioco
        ttk.Label(input_frame, text="Ruota Fonte (Spia):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ruota_fonte_var = tk.StringVar()
        self.ruota_fonte_combobox = ttk.Combobox(input_frame, textvariable=self.ruota_fonte_var, values=list(RUOTE_MAP.keys()), state="readonly")
        self.ruota_fonte_combobox.current(0); self.ruota_fonte_combobox.grid(row=1, column=1, sticky="ew", pady=2)
        self.ruota_fonte_combobox.bind("<<ComboboxSelected>>", self.update_titles)
        ttk.Label(input_frame, text="Ruota Destinazione (Gioco):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.ruota_dest_var = tk.StringVar()
        self.ruota_dest_combobox = ttk.Combobox(input_frame, textvariable=self.ruota_dest_var, values=list(RUOTE_MAP.keys()), state="readonly")
        self.ruota_dest_combobox.current(1); self.ruota_dest_combobox.grid(row=2, column=1, sticky="ew", pady=2)
        self.ruota_dest_combobox.bind("<<ComboboxSelected>>", self.update_titles)
        ttk.Label(input_frame, text="Colpi di Gioco:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.colpi_var = tk.StringVar(value="10")
        ttk.Entry(input_frame, textvariable=self.colpi_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Label(input_frame, text="Finestra Storica:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.finestra_var = tk.StringVar(value="500")
        ttk.Entry(input_frame, textvariable=self.finestra_var, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)
        ttk.Label(input_frame, text="Numeri per Spia:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.num_numeri_var = tk.StringVar(value="5")
        ttk.Entry(input_frame, textvariable=self.num_numeri_var, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)
        
        # Pulsanti di Azione
        self.style = ttk.Style(self)
        self.style.configure("Accent.TButton", foreground="blue", font=('Helvetica', 9, 'bold'))

        armonic_frame = ttk.LabelFrame(self.main_frame, text="Metodo Analisi Armonica", padding="10")
        armonic_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        # --- MODIFICA: Rimossa espansione dei pulsanti ---
        self.btn_armonica_pred = ttk.Button(armonic_frame, text="1. Trova Previsione", command=self.run_analisi_armonica)
        self.btn_armonica_pred.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_armoniche_in_gioco = ttk.Button(armonic_frame, text="2. Previsioni in Gioco", command=self.run_armoniche_in_gioco)
        self.btn_armoniche_in_gioco.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_armonica_test = ttk.Button(armonic_frame, text="3. Backtest (Cinquina)", command=self.run_backtest_armonico)
        self.btn_armonica_test.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_armonica_test_forte = ttk.Button(armonic_frame, text="4. Backtest (N. Forte)", command=self.run_backtest_armonico_forte)
        self.btn_armonica_test_forte.pack(side=tk.LEFT, padx=5, pady=5)
        
        positional_frame = ttk.LabelFrame(self.main_frame, text="Metodo Analisi Posizionale (Classico)", padding="10")
        positional_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        # --- MODIFICA: Rimossa espansione dei pulsanti ---
        self.btn_previsione = ttk.Button(positional_frame, text="1. Genera Previsione", command=self.run_prediction)
        self.btn_previsione.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_in_gioco = ttk.Button(positional_frame, text="2. Previsioni in Gioco", command=self.run_in_gioco)
        self.btn_in_gioco.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_backtest = ttk.Button(positional_frame, text="3. Backtest (Cinquina)", command=self.run_backtest)
        self.btn_backtest.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_backtest_forte = ttk.Button(positional_frame, text="4. Backtest (N. Forte)", command=self.run_backtest_posizionale_forte)
        self.btn_backtest_forte.pack(side=tk.LEFT, padx=5, pady=5)

        # NUOVO RIQUADRO PER LA SINTESI FINALE
        sintesi_frame = ttk.LabelFrame(self.main_frame, text="Sintesi Finale", padding="10")
        sintesi_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        # --- MODIFICA: Rimossa espansione del pulsante ---
        self.btn_sintesi_numeri_forti = ttk.Button(sintesi_frame, text="Trova i 'Numeri Forti' più convergenti ancora in gioco", command=self.run_numeri_forti_in_gioco, style="Accent.TButton")
        self.btn_sintesi_numeri_forti.pack(side=tk.LEFT, padx=5, pady=5)
        
        self._aggiorna_stato_sorgente_dati()
        self.update_titles()
    
    def _crea_widgets_output(self):
        # Il riquadro Risultati ora va alla RIGA 4 per fare spazio alla Sintesi
        output_frame = ttk.LabelFrame(self.main_frame, text="Risultati", padding="10")
        output_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5) # <--- MODIFICA QUI: row=4
        
        # Questa riga che configura l'espansione, che abbiamo già corretto, ora ha senso
        self.main_frame.rowconfigure(4, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=100, height=25, font=("Courier New", 9))
        self.output_text.pack(expand=True, fill="both")

    def _aggiorna_stato_sorgente_dati(self):
        if self.source_type_var.get() == "locale":
            self.local_path_entry.config(state="normal")
            self.browse_button.config(state="normal")
        else:
            self.local_path_entry.config(state="disabled")
            self.browse_button.config(state="disabled")

    def browse_folder(self):
        from tkinter import filedialog
        folder_path = filedialog.askdirectory(title="Seleziona la cartella contenente gli archivi .txt")
        if folder_path:
            self.local_path_var.set(folder_path)

    def update_titles(self, event=None):
        fonte = self.ruota_fonte_var.get()
        dest = self.ruota_dest_var.get()
        
        # Questo è il tuo titolo base che verrà usato sempre
        titolo_base = "Analizzatore Ciclo-Spazio - Convergenze Armoniche - Created by Max Lotto -"

        if not fonte or not dest:
            self.title(titolo_base)
            return
            
        if fonte == dest:
            # Aggiunge solo l'indicazione della ruota al titolo base
            # Esempio: "Analizzatore...by Max Lotto - [Ruota Singola: BARI]"
            self.title(f"{titolo_base} [Ruota Singola: {fonte.upper()}]")
        else:
            # Aggiunge solo l'indicazione delle ruote al titolo base
            # Esempio: "Analizzatore...by Max Lotto - [Doppia Ruota: BARI > CAGLIARI]"
            self.title(f"{titolo_base} [Doppia Ruota: {fonte.upper()} > {dest.upper()}]")

    def _get_params(self):
        try:
            ruota_fonte = self.ruota_fonte_var.get()
            ruota_dest = self.ruota_dest_var.get()
            colpi = int(self.colpi_var.get())
            finestra = int(self.finestra_var.get())
            num_numeri = int(self.num_numeri_var.get())
            if not ruota_fonte or not ruota_dest: raise ValueError("Selezionare entrambe le ruote.")
            if colpi <= 0 or finestra <= 0 or not (1 <= num_numeri <= 10): raise ValueError("Parametri numerici non validi.")
            return ruota_fonte, ruota_dest, colpi, finestra, num_numeri
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Errore Parametri", str(e))
            return None, None, None, None, None

    def _carica_archivi(self):
        ruota_fonte, ruota_dest, colpi, finestra, num_numeri = self._get_params()
        if not ruota_fonte: return None, None, None
        sorgente_dati = ""
        if self.source_type_var.get() == "online":
            sorgente_dati = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
        else:
            sorgente_dati = self.local_path_var.get()
            if not sorgente_dati or not os.path.isdir(sorgente_dati):
                messagebox.showerror("Errore Percorso", "Selezionare una cartella locale valida.")
                return None, None, None
        archivio_fonte = carica_singola_estrazione(sorgente_dati, ruota_fonte, self.output_text)
        archivio_dest = archivio_fonte if ruota_fonte == ruota_dest else carica_singola_estrazione(sorgente_dati, ruota_dest, self.output_text)
        if not archivio_fonte or not archivio_dest:
            self.output_text.insert(tk.END, f"[ERRORE] Dati mancanti per una delle ruote.\n")
            return None, None, None
        len_min = min(len(archivio_fonte), len(archivio_dest))
        if len(archivio_fonte) != len(archivio_dest):
            self.output_text.insert(tk.END, f"[AVVISO] Archivi allineati alle ultime {len_min} estrazioni.\n")
            archivio_fonte = archivio_fonte[-len_min:]
            archivio_dest = archivio_dest[-len_min:]
        return (ruota_fonte, ruota_dest), (archivio_fonte, archivio_dest), (colpi, finestra, num_numeri)
    
    def run_backtest(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        user_input = simpledialog.askstring("Input", "Su quante delle ultime estrazioni vuoi eseguire il test?\n(es. 100, lascia vuoto per tutto l'archivio)", parent=self)
        if user_input is None: return
        num_test = int(user_input) if user_input and user_input.isdigit() else None
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, esegui_backtest_comparativo(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri, num_test))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti per l'analisi.\n")

    def run_prediction(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        ruota_fonte, ruota_dest = nomi_ruote; _, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        user_input = simpledialog.askstring("Input", "Quale posizione estratto usare come seme? (1-5)", parent=self)
        if user_input is None or not user_input.isdigit(): self.output_text.insert(tk.END, "[ERRORE] Input non valido.\n"); return
        pos_scelta = int(user_input) - 1
        if not (0 <= pos_scelta <= 4): self.output_text.insert(tk.END, "[ERRORE] Posizione non valida.\n"); return
        if len(archivio_fonte) > finestra:
            numero_spia = archivio_fonte[-1]['numeri'][pos_scelta]; is_incrociato = (ruota_fonte != ruota_dest)
            previsione = genera_previsione_spia_incrociata(archivio_fonte, archivio_dest, numero_spia, num_numeri) if is_incrociato else genera_previsione_spia(archivio_fonte, numero_spia, num_numeri)
            output_string = ""
            if previsione:
                statistiche = calcola_statistiche_complete(archivio_dest); ambata_suggerita = max(previsione, key=lambda x: statistiche.get(x, {}).get('ic', 0))
                output_string += f"\nPREVISIONE POSIZIONALE GENERATA{' (DOPPIA RUOTA)' if is_incrociato else ''}\n{'*'*70}\n"
                if is_incrociato: output_string += f"Ruota Spia: {ruota_fonte.upper()} | Ruote Gioco: {ruota_fonte.upper()} e {ruota_dest.upper()}\n"
                else: output_string += f"Ruota: {ruota_fonte.upper()}\n"
                output_string += f"Spia: {numero_spia} (da Pos. {pos_scelta+1} su {ruota_fonte.upper()})\n"; output_string += "-"*70 + "\n"; output_string += f"{'Numero':<8} | {'Rit.Att':<8} | {'Rit.Max':<8} | {'Freq.':<8} | {'I.C.':<8}\n"; output_string += "-"*70 + "\n"
                for num in previsione:
                    stats_num = statistiche.get(num, {}); tag = " <<< NUMERO FORTE" if num == ambata_suggerita else "" # MODIFICATO QUI
                    output_string += f"{num:<8} | {stats_num.get('rit_att', 'N/D'):<8} | {stats_num.get('rit_max', 'N/D'):<8} | {stats_num.get('freq', 'N/D'):<8} | {stats_num.get('ic', 0):.2f}{tag}\n"
            else: output_string += f"\nImpossibile generare una previsione per lo spia {numero_spia}.\n"
            self.output_text.insert(tk.END, output_string)
        else: self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti.\n")

    def run_in_gioco(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, trova_previsioni_in_gioco(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti.\n")

    def run_analisi_armonica(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        _, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        if len(archivio_fonte) > finestra:
            self.output_text.insert(tk.END, esegui_analisi_armonica(archivio_fonte, archivio_dest, nomi_ruote, finestra, num_numeri))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti (finestra troppo grande).\n")
        
    def run_backtest_armonico(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        user_input = simpledialog.askstring("Input", "Su quante delle ultime estrazioni vuoi eseguire il test?\n(es. 100, lascia vuoto per tutto l'archivio)", parent=self)
        if user_input is None: return
        num_test = int(user_input) if user_input and user_input.isdigit() else None
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, esegui_backtest_armonico(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri, num_test))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti per il backtest.\n")

    def run_armoniche_in_gioco(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, trova_previsioni_armoniche_in_gioco(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti per la ricerca.\n")

    def run_backtest_armonico_forte(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        user_input = simpledialog.askstring("Input", "Su quante delle ultime estrazioni vuoi eseguire il test?\n(es. 100, lascia vuoto per tutto l'archivio)", parent=self)
        if user_input is None: return
        num_test = int(user_input) if user_input and user_input.isdigit() else None
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, esegui_backtest_numero_forte_armonico(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri, num_test))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti per il backtest.\n")

    def run_backtest_posizionale_forte(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        user_input = simpledialog.askstring("Input", "Su quante delle ultime estrazioni vuoi eseguire il test?\n(es. 100, lascia vuoto per tutto l'archivio)", parent=self)
        if user_input is None: return
        num_test = int(user_input) if user_input and user_input.isdigit() else None
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, esegui_backtest_numero_forte_posizionale(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri, num_test))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti per il backtest.\n")

    def run_numeri_forti_in_gioco(self):
        self.output_text.delete('1.0', tk.END)
        nomi_ruote, archivi, params = self._carica_archivi()
        if not archivi: return
        colpi, finestra, num_numeri = params; archivio_fonte, archivio_dest = archivi
        if len(archivio_fonte) > finestra + colpi:
            self.output_text.insert(tk.END, trova_numeri_forti_in_gioco(archivio_fonte, archivio_dest, nomi_ruote, colpi, finestra, num_numeri, num_numeri))
        else:
            self.output_text.insert(tk.END, "[ERRORE] Dati insufficienti per la ricerca.\n")


# ==============================================================================
# BLOCCO DI ESECUZIONE PRINCIPALE
# ==============================================================================
if __name__ == "__main__":
    app = LottoApp()
    app.mainloop()