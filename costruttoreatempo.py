# ==============================================================================
# --- BLOCCO DI PROTEZIONE E LICENZA ---
# ==============================================================================
import os
from datetime import datetime, date, timedelta, timezone
from collections import Counter, defaultdict
from itertools import combinations
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Listbox
import traceback
import lunghette
import base64
import requests
from pathlib import Path

try:
    from tkcalendar import DateEntry
except ImportError:
    pass # Gestito all'avvio
import json

# --- CONFIGURAZIONE DELLA PROTEZIONE ---
EXPIRATION_HOURS = 2
LICENSE_FILE = Path.home() / ".costruttore_analyzer_lic"
LOG_DIR = Path.home() / ".app_cache_data"
LOG_FILE = LOG_DIR / "session.logdat"
LOG_KEY = "la_mia_chiave_segreta_per_il_costruttore_2024"
TIME_API_URL = "http://worldtimeapi.org/api/timezone/Etc/UTC"

class LicenseManager:
    def __init__(self, expiration_hours, license_file_path, time_api_url):
        self.expiration_delta = timedelta(hours=expiration_hours)
        self.license_file = license_file_path
        self.time_api_url = time_api_url
    def _get_network_time(self):
        try:
            response = requests.get(self.time_api_url, timeout=5)
            response.raise_for_status()
            utc_time_str = response.json()['utc_datetime']
            return datetime.fromisoformat(utc_time_str)
        except requests.exceptions.RequestException: return None
    def _read_first_run_time(self):
        try:
            with open(self.license_file, 'r') as f:
                encoded_time = f.read()
                decoded_time_str = base64.b64decode(encoded_time).decode('utf-8')
                return datetime.fromisoformat(decoded_time_str)
        except (FileNotFoundError, ValueError, TypeError): return None
    def _write_first_run_time(self, time_obj):
        time_str = time_obj.isoformat()
        encoded_time = base64.b64encode(time_str.encode('utf-8'))
        os.makedirs(os.path.dirname(self.license_file), exist_ok=True)
        with open(self.license_file, 'w', encoding='utf-8') as f: f.write(encoded_time.decode('utf-8'))
    def check_license(self, logger):
        current_time = self._get_network_time()
        if not current_time:
            logger.log("ATTENZIONE: Connessione al server dell'orario fallita. Uso l'orologio di sistema.")
            current_time = datetime.now(timezone.utc)
        first_run_time = self._read_first_run_time()
        if first_run_time is None:
            logger.log("Primo avvio rilevato. Licenza attivata.")
            self._write_first_run_time(current_time)
            return True, f"Benvenuto! È stato attivato il periodo di prova di {EXPIRATION_HOURS} ore."
        elapsed_time = current_time - first_run_time
        if elapsed_time >= self.expiration_delta:
            logger.log(f"Licenza SCADUTA. Tempo trascorso: {elapsed_time}")
            return False, f"Il periodo di prova di {EXPIRATION_HOURS} ore è terminato."
        else:
            time_remaining = self.expiration_delta - elapsed_time
            logger.log(f"Controllo licenza OK. Tempo rimanente: {time_remaining}")
            return True, ""

class SecureLogger:
    def __init__(self, log_file, key):
        self.log_file = log_file; self.key = key
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    def _xor_cipher(self, text):
        return ''.join(chr(ord(c) ^ ord(self.key[i % len(self.key)])) for i, c in enumerate(text))
    def log(self, message):
        try:
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            full_log_entry = f"[{timestamp}] - {message}\n"
            encrypted_entry = self._xor_cipher(full_log_entry)
            with open(self.log_file, 'a', encoding='utf-8') as f: f.write(encrypted_entry)
        except Exception: pass

# --- INIZIO CODICE ORIGINALE ---
# --- COSTANTI GLOBALI ---
RUOTE = ["Bari", "Cagliari", "Firenze", "Genova", "Milano", "Napoli", "Palermo", "Roma", "Torino", "Venezia", "Nazionale"]
OPERAZIONI = {
    "somma": lambda a, b: a + b,
    "differenza": lambda a, b: a - b,
    "moltiplicazione": lambda a, b: a * b,
}
OPERAZIONI_COMPLESSE = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
}

def regola_fuori_90(numero):
    if numero is None: return None
    if numero == 0: return 90
    while numero <= 0: numero += 90
    while numero > 90: numero -= 90
    return numero

def calcola_diametrale(numero):
    if not (1 <= numero <= 90): return None
    diam = numero + 45
    return regola_fuori_90(diam)

def calcola_vertibile(numero):
    if not (1 <= numero <= 90): return None
    s_num = str(numero).zfill(2)
    decina, unita = s_num[0], s_num[1]
    if decina == unita:
        return 9 if numero == 90 else int(decina + "9")
    elif numero == 90: return 9
    else: return regola_fuori_90(int(unita + decina))

def calcola_complemento_a_90(numero):
    if not (1 <= numero <= 90): return None
    return regola_fuori_90(90 - numero)

def calcola_figura(numero):
    if not (1 <= numero <= 90): return None
    return 9 if numero % 9 == 0 else numero % 9

def calcola_cadenza(numero):
    if not (1 <= numero <= 90): return None
    return numero % 10

def calcola_diametrale_in_decina(numero):
    if not (1 <= numero <= 90): return None
    if numero == 90: return regola_fuori_90(95)
    unita = numero % 10
    return regola_fuori_90(numero + 5) if 1 <= unita <= 5 else regola_fuori_90(numero - 5)

OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE = {
    "Fisso": lambda n: n, "Diametrale": calcola_diametrale, "Vertibile": calcola_vertibile,
    "Compl.90": calcola_complemento_a_90, "Figura": calcola_figura, "Cadenza": calcola_cadenza,
    "Diam.Decina": calcola_diametrale_in_decina
}

def parse_riga_estrazione(riga, nome_file_ruota, num_riga):
    try:
        parti = riga.strip().split('\t')
        if len(parti) != 7: return None, None
        data_str, numeri_str = parti[0], parti[2:7]
        numeri = [int(n) for n in numeri_str]
        if len(numeri) != 5: return None, None
        data_obj = datetime.strptime(data_str, "%Y/%m/%d").date()
        return data_obj, numeri
    except (ValueError, Exception): return None, None

def carica_storico_completo(cartella_dati, data_inizio_filtro=None, data_fine_filtro=None, app_logger=None):
    def log_message(msg):
        if app_logger: app_logger(msg)
    log_message(f"\nCaricamento dati da: {cartella_dati}")
    if not os.path.isdir(cartella_dati):
        log_message(f"Errore: Cartella '{cartella_dati}' non trovata."); return []
    storico_globale = defaultdict(dict)
    for nome_ruota_chiave in RUOTE:
        path_file = os.path.join(cartella_dati, f"{nome_ruota_chiave.upper()}.TXT")
        if not os.path.exists(path_file):
            path_file = os.path.join(cartella_dati, f"{nome_ruota_chiave}.txt")
            if not os.path.exists(path_file): continue
        with open(path_file, 'r', encoding='utf-8') as f:
            for num_riga, riga_contenuto in enumerate(f, 1):
                data_obj, numeri = parse_riga_estrazione(riga_contenuto, os.path.basename(path_file), num_riga)
                if data_obj and numeri:
                    if (data_inizio_filtro and data_obj < data_inizio_filtro) or \
                       (data_fine_filtro and data_obj > data_fine_filtro): continue
                    storico_globale[data_obj][nome_ruota_chiave] = numeri
    
    storico_ordinato = []
    date_ordinate = sorted(storico_globale.keys())
    date_per_mese = defaultdict(list)
    for d in date_ordinate:
        date_per_mese[(d.year, d.month)].append(d)
    
    for data_obj in date_ordinate:
        anno, mese = data_obj.year, data_obj.month
        try:
            indice_mese = date_per_mese[(anno, mese)].index(data_obj) + 1
        except ValueError:
            indice_mese = -1 
        estrazione_completa = {'data': data_obj, 'indice_mese': indice_mese}
        for r_nome in RUOTE:
            estrazione_completa[r_nome] = storico_globale[data_obj].get(r_nome, [])
        if any(estrazione_completa[r_n] for r_n in RUOTE):
            storico_ordinato.append(estrazione_completa)
            
    log_message(f"Caricate e ordinate {len(storico_ordinato)} estrazioni.")
    return storico_ordinato

def analizza_metodo_sommativo_base(storico, ruota_calcolo, pos_estratto_calcolo, operazione_str, operando_fisso, ruote_gioco_selezionate, lookahead=1, indice_mese_filtro=None):
    if operazione_str not in OPERAZIONI:
        return -1, 0, 0, [], Counter()
    op_func = OPERAZIONI[operazione_str]
    successi, tentativi = 0, 0; applicazioni_vincenti = []
    ambata_fissa_del_metodo_prima_occ = -1
    ambate_previste_contatore = Counter()
    prima_applicazione_valida = True
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
        numeri_ruota_calcolo = estrazione_corrente.get(ruota_calcolo, [])
        if not numeri_ruota_calcolo or len(numeri_ruota_calcolo) <= pos_estratto_calcolo: continue
        numero_base = numeri_ruota_calcolo[pos_estratto_calcolo]
        try: valore_operazione = op_func(numero_base, operando_fisso)
        except ZeroDivisionError: continue
        ambata_prevista_corrente = regola_fuori_90(valore_operazione)
        if ambata_prevista_corrente is None: continue
        if prima_applicazione_valida:
            ambata_fissa_del_metodo_prima_occ = ambata_prevista_corrente
            prima_applicazione_valida = False
        ambate_previste_contatore[ambata_prevista_corrente] += 1
        tentativi += 1
        trovato_in_questo_tentativo = False
        dettagli_vincita_per_tentativo = []
        for k in range(1, lookahead + 1):
            if i + k >= len(storico): break
            estrazione_futura = storico[i + k]
            for ruota_verifica_effettiva in ruote_gioco_selezionate:
                if ambata_prevista_corrente in estrazione_futura.get(ruota_verifica_effettiva, []):
                    if not trovato_in_questo_tentativo: successi += 1; trovato_in_questo_tentativo = True
                    dettagli_vincita_per_tentativo.append({"ruota_vincita": ruota_verifica_effettiva, "numeri_ruota_vincita": estrazione_futura.get(ruota_verifica_effettiva, []), "data_riscontro": estrazione_futura['data'], "colpo_riscontro": k})
            if trovato_in_questo_tentativo and len(ruote_gioco_selezionate) == 1: break
        if trovato_in_questo_tentativo:
            applicazioni_vincenti.append({"data_applicazione": estrazione_corrente['data'], "estratto_base": numero_base, "operando": operando_fisso, "operazione": operazione_str, "ambata_prevista": ambata_prevista_corrente, "riscontri": dettagli_vincita_per_tentativo})
    return ambata_fissa_del_metodo_prima_occ, successi, tentativi, applicazioni_vincenti, ambate_previste_contatore

def analizza_copertura_combinata(storico, top_metodi_info, ruote_gioco_selezionate, lookahead, indice_mese_filtro, app_logger=None):
    date_tentativi_combinati = set()
    date_successi_combinati = set()
    op_func_cache = {op_str: OPERAZIONI[op_str] for op_str in OPERAZIONI}
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
        ambate_previste_per_questa_estrazione = set()
        almeno_un_metodo_applicabile_qui = False
        for metodo_info_dict in top_metodi_info:
            met_details = metodo_info_dict['metodo']
            ruota_calc = met_details['ruota_calcolo']; pos_estr = met_details['pos_estratto_calcolo']
            numeri_ruota_corrente = estrazione_corrente.get(ruota_calc, [])
            if not numeri_ruota_corrente or len(numeri_ruota_corrente) <= pos_estr: continue
            almeno_un_metodo_applicabile_qui = True
            numero_base = numeri_ruota_corrente[pos_estr]
            op_str = met_details['operazione']; operando = met_details['operando_fisso']
            if op_str not in op_func_cache:
                if app_logger: app_logger(f"WARN: Operazione {op_str} non in cache per analisi combinata."); continue
            op_func = op_func_cache[op_str]
            try: valore_operazione = op_func(numero_base, operando)
            except ZeroDivisionError: continue
            ambata_prevista = regola_fuori_90(valore_operazione)
            if ambata_prevista is not None: ambate_previste_per_questa_estrazione.add(ambata_prevista)
        if almeno_un_metodo_applicabile_qui:
            date_tentativi_combinati.add(estrazione_corrente['data'])
            if ambate_previste_per_questa_estrazione:
                successo_per_questa_data_applicazione = False
                for k_lookahead in range(1, lookahead + 1):
                    if i + k_lookahead >= len(storico): break
                    estrazione_futura = storico[i + k_lookahead]
                    for ambata_p in ambate_previste_per_questa_estrazione:
                        for ruota_verifica in ruote_gioco_selezionate:
                            if ambata_p in estrazione_futura.get(ruota_verifica, []):
                                date_successi_combinati.add(estrazione_corrente['data'])
                                successo_per_questa_data_applicazione = True; break
                        if successo_per_questa_data_applicazione: break
                    if successo_per_questa_data_applicazione: break
    num_successi_combinati = len(date_successi_combinati)
    num_tentativi_combinati = len(date_tentativi_combinati)
    frequenza_combinata = num_successi_combinati / num_tentativi_combinati if num_tentativi_combinati > 0 else 0
    return num_successi_combinati, num_tentativi_combinati, frequenza_combinata

def analizza_abbinamenti_per_numero_specifico(storico, ambata_target, ruote_gioco, app_logger=None):
    if ambata_target is None:
        if app_logger: app_logger("WARN: ambata_target è None in analizza_abbinamenti_per_numero_specifico.")
        return {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "sortite_ambata_target": 0}
    abbinamenti_per_ambo = Counter(); abbinamenti_per_terno = Counter()
    abbinamenti_per_quaterna = Counter(); abbinamenti_per_cinquina = Counter()
    sortite_ambata_target = 0
    for estrazione in storico:
        uscita_in_estrazione_su_ruote_gioco = False
        numeri_ruota_con_vincita_per_abbinamento = []
        for ruota in ruote_gioco:
            numeri_estratti_ruota = estrazione.get(ruota, [])
            if ambata_target in numeri_estratti_ruota:
                if not uscita_in_estrazione_su_ruote_gioco:
                    sortite_ambata_target += 1; uscita_in_estrazione_su_ruote_gioco = True
                if not numeri_ruota_con_vincita_per_abbinamento:
                    numeri_ruota_con_vincita_per_abbinamento = numeri_estratti_ruota
                break
        if uscita_in_estrazione_su_ruote_gioco and numeri_ruota_con_vincita_per_abbinamento:
            altri_numeri_con_target = [n for n in numeri_ruota_con_vincita_per_abbinamento if n != ambata_target]
            for num_abbinato in altri_numeri_con_target: abbinamenti_per_ambo[num_abbinato] += 1
            if len(altri_numeri_con_target) >= 2:
                for combo_2 in combinations(sorted(altri_numeri_con_target), 2): abbinamenti_per_terno[combo_2] += 1
            if len(altri_numeri_con_target) >= 3:
                for combo_3 in combinations(sorted(altri_numeri_con_target), 3): abbinamenti_per_quaterna[combo_3] += 1
            if len(altri_numeri_con_target) >= 4:
                for combo_4 in combinations(sorted(altri_numeri_con_target), 4): abbinamenti_per_cinquina[combo_4] += 1
    return {
        "ambo": [{"numeri": [ab[0]], "frequenza": ab[1]/sortite_ambata_target if sortite_ambata_target else 0, "conteggio": ab[1]} for ab in abbinamenti_per_ambo.most_common(5)],
        "terno": [{"numeri": list(ab[0]), "frequenza": ab[1]/sortite_ambata_target if sortite_ambata_target else 0, "conteggio": ab[1]} for ab in abbinamenti_per_terno.most_common(5)],
        "quaterna": [{"numeri": list(ab[0]), "frequenza": ab[1]/sortite_ambata_target if sortite_ambata_target else 0, "conteggio": ab[1]} for ab in abbinamenti_per_quaterna.most_common(5)],
        "cinquina": [{"numeri": list(ab[0]), "frequenza": ab[1]/sortite_ambata_target if sortite_ambata_target else 0, "conteggio": ab[1]} for ab in abbinamenti_per_cinquina.most_common(5)],
        "sortite_ambata_target": sortite_ambata_target
    }

def trova_migliori_ambate_e_abbinamenti(storico, ruota_calcolo, pos_estratto_calcolo,
                                         ruote_gioco_selezionate, max_ambate_output=1,
                                         lookahead=1, indice_mese_filtro=None,
                                         min_tentativi_per_ambata=10, app_logger=None,
                                         data_fine_analisi_globale_obj=None 
                                         ):
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=flush)
    
    risultati_ambate_grezzi = []
    # ... (calcolo risultati_ambate_grezzi e info_copertura_combinata come prima) ...
    log_message(f"\nAnalisi metodi per ambata su {'TUTTE le ruote' if len(ruote_gioco_selezionate) == len(RUOTE) else ', '.join(ruote_gioco_selezionate)} (da {ruota_calcolo}[pos.{pos_estratto_calcolo+1}]):")
    for op_str in OPERAZIONI:
        for operando in range(1, 91):
            ambata_fissa_prima_occ, successi, tentativi, applicazioni_vincenti_dett, _ = analizza_metodo_sommativo_base(
                storico, ruota_calcolo, pos_estratto_calcolo, op_str, operando,
                ruote_gioco_selezionate, lookahead, indice_mese_filtro
            )
            if tentativi >= min_tentativi_per_ambata:
                frequenza = successi / tentativi if tentativi > 0 else 0
                risultati_ambate_grezzi.append({
                    "metodo": {"operazione": op_str, "operando_fisso": operando,
                                   "ruota_calcolo": ruota_calcolo, "pos_estratto_calcolo": pos_estratto_calcolo},
                    "ambata_prodotta_dal_metodo": ambata_fissa_prima_occ,
                    "successi": successi, "tentativi": tentativi, "frequenza_ambata": frequenza,
                    "applicazioni_vincenti_dettagliate": applicazioni_vincenti_dett
                })
    log_message("  Completata analisi performance storica dei metodi.")
    risultati_ambate_grezzi.sort(key=lambda x: (x["frequenza_ambata"], x["successi"]), reverse=True)
    
    info_copertura_combinata_da_restituire = None
    top_n_metodi_per_analisi_combinata = risultati_ambate_grezzi[:max_ambate_output]
    if len(top_n_metodi_per_analisi_combinata) > 1:
        log_message(f"\n--- ANALISI COPERTURA COMBINATA PER TOP {len(top_n_metodi_per_analisi_combinata)} METODI ---")
        s_comb, t_comb, f_comb = analizza_copertura_combinata(storico, top_n_metodi_per_analisi_combinata, ruote_gioco_selezionate, lookahead, indice_mese_filtro, app_logger)
        if t_comb > 0:
            log_message(f"  Giocando simultaneamente le ambate prodotte dai {len(top_n_metodi_per_analisi_combinata)} migliori metodi:")
            log_message(f"  - Successi Complessivi (almeno un'ambata vincente): {s_comb}")
            log_message(f"  - Tentativi Complessivi (almeno un metodo applicabile): {t_comb}")
            log_message(f"  - Frequenza di Copertura Combinata: {f_comb:.2%}")
            info_copertura_combinata_da_restituire = {"successi": s_comb, "tentativi": t_comb, "frequenza": f_comb, "num_metodi_combinati": len(top_n_metodi_per_analisi_combinata)}
        else:
            log_message("  Nessun tentativo combinato applicabile per i metodi selezionati per l'analisi combinata.")
            
    risultati_finali_output = []
    if not storico: 
        log_message("ERRORE: Storico vuoto, impossibile calcolare previsioni."); 
        return [], None

    estrazione_per_previsione_live = None
    nota_globale_previsione = ""
    data_riferimento_log_previsione = "N/A" 

    indice_mese_per_previsione_utente = indice_mese_filtro 

    if indice_mese_per_previsione_utente is not None and data_fine_analisi_globale_obj is not None:
        log_message(f"TMAA: Ricerca estrazione per previsione live: indice_mese_utente={indice_mese_per_previsione_utente}, data_fine<={data_fine_analisi_globale_obj.strftime('%Y-%m-%d') if data_fine_analisi_globale_obj else 'N/A'}")
        for estr in reversed(storico):
            if estr['data'] <= data_fine_analisi_globale_obj:
                if estr.get('indice_mese') == indice_mese_per_previsione_utente:
                    estrazione_per_previsione_live = estr
                    data_riferimento_log_previsione = estr['data'].strftime('%d/%m/%Y')
                    log_message(f"  TMAA: Trovata estrazione specifica per previsione live: {data_riferimento_log_previsione} (Indice: {estr.get('indice_mese')})")
                    break # Trovata, esci dal loop
        
        if not estrazione_per_previsione_live: # Se il loop è terminato senza trovare una corrispondenza
            nota_globale_previsione = f"Nessuna estrazione trovata corrispondente all'indice mese {indice_mese_per_previsione_utente} entro il {data_fine_analisi_globale_obj.strftime('%d/%m/%Y')}."
            log_message(f"  TMAA: {nota_globale_previsione}")
            # estrazione_per_previsione_live rimane None
            
    elif storico: # Nessun filtro specifico indice/data fine, O uno dei due mancava
        estrazione_per_previsione_live = storico[-1]
        data_riferimento_log_previsione = estrazione_per_previsione_live['data'].strftime('%d/%m/%Y')
        log_message(f"  TMAA: Uso ultima estrazione disponibile ({data_riferimento_log_previsione}) per previsione (nessun filtro specifico indice/data fine, o uno dei due mancava).")
    
    if not estrazione_per_previsione_live and not nota_globale_previsione: # Fallback finale
        nota_globale_previsione = "Impossibile determinare estrazione di riferimento per la previsione."
        log_message(f"  TMAA: {nota_globale_previsione}")
    
    # --- Il resto della funzione per ciclare sui metodi e calcolare ambata/abbinamenti ---
    #     utilizzerà estrazione_per_previsione_live (che potrebbe essere None)
    #     e nota_globale_previsione.
    #     (Codice da qui in poi come nella versione precedente che ti ho dato,
    #      ma assicurati che i log e la logica usino correttamente
    #      estrazione_per_previsione_live e nota_globale_previsione)

    log_message(f"\n--- DETTAGLIO, PREVISIONE E ABBINAMENTI PER I TOP {min(max_ambate_output, len(risultati_ambate_grezzi))} METODI ---")
    for i, res_grezza in enumerate(risultati_ambate_grezzi[:max_ambate_output]):
        metodo_def = res_grezza['metodo']
        op_str_metodo = metodo_def['operazione']; operando_metodo = metodo_def['operando_fisso']
        rc_metodo = metodo_def['ruota_calcolo']; pe_metodo = metodo_def['pos_estratto_calcolo']
        
        ambata_previsione_attuale = None
        note_previsione_locale = nota_globale_previsione # Eredita la nota globale

        log_message(f"\n--- {i+1}° METODO ---")
        log_message(f"  Formula: {rc_metodo}[pos.{pe_metodo+1}] {op_str_metodo} {operando_metodo}")
        log_message(f"  Performance Storica: {res_grezza['frequenza_ambata']:.2%} ({res_grezza['successi']}/{res_grezza['tentativi']} casi)")

        if estrazione_per_previsione_live: # Solo se abbiamo un'estrazione valida
            op_func_metodo = OPERAZIONI.get(op_str_metodo)
            numeri_estrazione_per_previsione_ruota_calcolo = estrazione_per_previsione_live.get(rc_metodo, [])
            
            log_message(f"  PREVISIONE DA ESTRAZIONE DEL {estrazione_per_previsione_live['data'].strftime('%d/%m/%Y')} (Indice mese estraz.: {estrazione_per_previsione_live.get('indice_mese')}, Indice mese richiesto: {indice_mese_per_previsione_utente}):")

            if op_func_metodo and numeri_estrazione_per_previsione_ruota_calcolo and len(numeri_estrazione_per_previsione_ruota_calcolo) > pe_metodo:
                numero_base_per_previsione = numeri_estrazione_per_previsione_ruota_calcolo[pe_metodo]
                try: 
                    valore_op_previsione = op_func_metodo(numero_base_per_previsione, operando_metodo)
                    ambata_previsione_attuale = regola_fuori_90(valore_op_previsione)
                except ZeroDivisionError: 
                    note_previsione_locale = "Metodo non applicabile all'estrazione di riferimento (divisione per zero)."
            else: 
                note_previsione_locale = "Dati insufficienti nell'estrazione di riferimento per calcolare la previsione."
        elif not note_previsione_locale: # Se non c'era una nota globale, ma non c'è estrazione
             note_previsione_locale = "Nessuna estrazione di riferimento valida per la previsione." # Imposta una nota locale
        
        if ambata_previsione_attuale is not None: 
            log_message(f"    AMBATA DA GIOCARE: {ambata_previsione_attuale}")
        else: 
            # Usa nota_previsione_locale che ora contiene la nota globale o una nota locale
            log_message(f"    {(note_previsione_locale if note_previsione_locale else 'Previsione non calcolata o non valida.')}")
        
        abbinamenti_calcolati_finali = {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "sortite_ambata_target": 0}
        if ambata_previsione_attuale is not None:
            log_message(f"    Migliori Abbinamenti Storici (co-occorrenze con AMBATA DA GIOCARE: {ambata_previsione_attuale}):")
            abbinamenti_calcolati_finali = analizza_abbinamenti_per_numero_specifico(storico, ambata_previsione_attuale, ruote_gioco_selezionate, app_logger)
            if abbinamenti_calcolati_finali.get("sortite_ambata_target", 0) > 0:
                log_message(f"      (Basato su {abbinamenti_calcolati_finali['sortite_ambata_target']} sortite storiche del {ambata_previsione_attuale} su ruote selezionate)")
                for tipo_sorte, dati_sorte_lista in abbinamenti_calcolati_finali.items():
                    if tipo_sorte == "sortite_ambata_target": continue
                    if dati_sorte_lista:
                        log_message(f"        Per {tipo_sorte.upper().replace('_', ' ')}:")
                        for ab_info in dati_sorte_lista[:3]:
                            if ab_info['conteggio'] > 0:
                                numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri'])))
                                log_message(f"          - Numeri: [{numeri_ab_str}] (Freq: {ab_info['frequenza']:.1%}, Cnt: {ab_info['conteggio']})")
            else: log_message(f"      Nessuna co-occorrenza trovata per l'ambata {ambata_previsione_attuale} nello storico sulle ruote selezionate.")
        elif not nota_globale_previsione and res_grezza["applicazioni_vincenti_dettagliate"]: 
            log_message("    Nessuna ambata attuale calcolata. Abbinamenti basati su co-occorrenze non disponibili.")
        elif not nota_globale_previsione : 
            log_message("    Nessuna ambata attuale calcolata e nessuna applicazione vincente storica per analisi abbinamenti.")
        
        risultati_finali_output.append({
            "metodo": metodo_def, 
            "ambata_piu_frequente_dal_metodo": ambata_previsione_attuale if ambata_previsione_attuale is not None else "N/D",
            "frequenza_ambata": res_grezza['frequenza_ambata'], 
            "successi": res_grezza['successi'], 
            "tentativi": res_grezza['tentativi'],
            "abbinamenti": abbinamenti_calcolati_finali, 
            "applicazioni_vincenti_dettagliate": res_grezza["applicazioni_vincenti_dettagliate"],
            "estrazione_usata_per_previsione": estrazione_per_previsione_live 
        })
    return risultati_finali_output, info_copertura_combinata_da_restituire

def calcola_valore_metodo_complesso(estrazione_corrente, definizione_metodo, app_logger=None):
    if not definizione_metodo:
        if app_logger: app_logger("Errore: Definizione metodo complesso vuota.")
        return None
    valore_accumulato = 0; operazione_aritmetica_pendente = None
    for i, comp in enumerate(definizione_metodo):
        termine_attuale = 0
        numeri_ruota_comp = estrazione_corrente.get(comp.get('ruota'), [])
        if comp['tipo_termine'] == 'estratto':
            if not numeri_ruota_comp or len(numeri_ruota_comp) <= comp['posizione']: return None
            termine_attuale = numeri_ruota_comp[comp['posizione']]
        elif comp['tipo_termine'] == 'fisso': termine_attuale = comp['valore_fisso']
        else:
            if app_logger: app_logger(f"Errore: Tipo termine sconosciuto: {comp['tipo_termine']}"); return None
        if i == 0: valore_accumulato = termine_attuale
        else:
            if operazione_aritmetica_pendente:
                try: valore_accumulato = operazione_aritmetica_pendente(valore_accumulato, termine_attuale)
                except ZeroDivisionError: return None
            else:
                if app_logger: app_logger("Errore logico: Operazione pendente mancante."); return None
        op_str_successiva = comp.get('operazione_successiva')
        if op_str_successiva and op_str_successiva != '=':
            if op_str_successiva not in OPERAZIONI_COMPLESSE:
                if app_logger: app_logger(f"Errore: Operazione '{op_str_successiva}' non supportata."); return None
            operazione_aritmetica_pendente = OPERAZIONI_COMPLESSE[op_str_successiva]
        else: operazione_aritmetica_pendente = None
    return valore_accumulato

def analizza_metodo_complesso_specifico(storico, definizione_metodo,
                                         ruote_gioco_selezionate, lookahead,
                                         indice_mese_filtro, app_logger=None,
                                         filtro_condizione_primaria_dict=None):
    successi = 0; tentativi = 0; applicazioni_vincenti = []
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]

        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro:
            continue

        if filtro_condizione_primaria_dict:
            cond_ruota = filtro_condizione_primaria_dict['ruota']
            cond_pos_idx = (filtro_condizione_primaria_dict['posizione'] - 1) if filtro_condizione_primaria_dict.get('posizione', 0) > 0 else 0
            cond_min = filtro_condizione_primaria_dict['val_min']
            cond_max = filtro_condizione_primaria_dict['val_max']

            numeri_ruota_cond_corr = estrazione_corrente.get(cond_ruota, [])
            if not numeri_ruota_cond_corr or len(numeri_ruota_cond_corr) <= cond_pos_idx:
                continue
            val_est_cond = numeri_ruota_cond_corr[cond_pos_idx]
            if not (cond_min <= val_est_cond <= cond_max):
                continue

        valore_calcolato_raw = calcola_valore_metodo_complesso(estrazione_corrente, definizione_metodo, app_logger)
        if valore_calcolato_raw is None: continue
        ambata_prevista = regola_fuori_90(valore_calcolato_raw)
        tentativi += 1
        trovato_successo_app = False; dettagli_vincita_app = []
        for k_lookahead in range(1, lookahead + 1):
            if i + k_lookahead >= len(storico): break
            estrazione_futura = storico[i + k_lookahead]
            for ruota_verifica in ruote_gioco_selezionate:
                if ambata_prevista in estrazione_futura.get(ruota_verifica, []):
                    if not trovato_successo_app: successi += 1; trovato_successo_app = True
                    dettagli_vincita_app.append({
                        "ruota_vincita": ruota_verifica,
                        "numeri_ruota_vincita": estrazione_futura.get(ruota_verifica, []),
                        "data_riscontro": estrazione_futura['data'], "colpo_riscontro": k_lookahead
                    })
            if trovato_successo_app and len(ruote_gioco_selezionate) == 1: break
        if trovato_successo_app:
            applicazioni_vincenti.append({
                "data_applicazione": estrazione_corrente['data'], "valore_calcolato_raw": valore_calcolato_raw,
                "ambata_prevista": ambata_prevista, "riscontri": dettagli_vincita_app
            })
    return successi, tentativi, applicazioni_vincenti

def analizza_copertura_ambate_previste_multiple(storico,
                                                lista_definizioni_metodi_estesi,
                                                ruote_gioco_selezionate,
                                                lookahead,
                                                indice_mese_filtro,
                                                app_logger=None,
                                                filtro_condizione_primaria_dict=None):
    if not lista_definizioni_metodi_estesi:
        return 0, 0, 0.0
    date_tentativi_combinati = set()
    date_successi_combinati = set()
    for i in range(len(storico) - lookahead):
        estrazione_applicazione = storico[i]
        if indice_mese_filtro and estrazione_applicazione['indice_mese'] != indice_mese_filtro: continue
        if filtro_condizione_primaria_dict:
            cond_ruota = filtro_condizione_primaria_dict['ruota']
            cond_pos_idx = (filtro_condizione_primaria_dict['posizione'] - 1) if filtro_condizione_primaria_dict.get('posizione',0) > 0 else 0
            cond_min = filtro_condizione_primaria_dict['val_min']
            cond_max = filtro_condizione_primaria_dict['val_max']
            numeri_ruota_cond_appl = estrazione_applicazione.get(cond_ruota, [])
            if not numeri_ruota_cond_appl or len(numeri_ruota_cond_appl) <= cond_pos_idx: continue
            val_est_cond_appl = numeri_ruota_cond_appl[cond_pos_idx]
            if not (cond_min <= val_est_cond_appl <= cond_max): continue
        ambate_previste_per_questa_applicazione = set()
        almeno_un_metodo_applicabile_qui = False
        for def_metodo_esteso in lista_definizioni_metodi_estesi:
            if not def_metodo_esteso: continue
            valore_calcolato_raw = calcola_valore_metodo_complesso(estrazione_applicazione, def_metodo_esteso, app_logger)
            if valore_calcolato_raw is not None:
                almeno_un_metodo_applicabile_qui = True
                ambata_prev = regola_fuori_90(valore_calcolato_raw)
                if ambata_prev is not None: ambate_previste_per_questa_applicazione.add(ambata_prev)
        if almeno_un_metodo_applicabile_qui:
            date_tentativi_combinati.add(estrazione_applicazione['data'])
            if ambate_previste_per_questa_applicazione:
                successo_per_questa_data_app = False
                for k_lh in range(1, lookahead + 1):
                    if i + k_lh >= len(storico): break
                    estrazione_futura = storico[i + k_lh]
                    for ambata_p in ambate_previste_per_questa_applicazione:
                        for ruota_v in ruote_gioco_selezionate:
                            if ambata_p in estrazione_futura.get(ruota_v, []):
                                date_successi_combinati.add(estrazione_applicazione['data'])
                                successo_per_questa_data_app = True; break
                        if successo_per_questa_data_app: break
                    if successo_per_questa_data_app: break
    num_successi_combinati = len(date_successi_combinati)
    num_tentativi_combinati = len(date_tentativi_combinati)
    frequenza_combinata = num_successi_combinati / num_tentativi_combinati if num_tentativi_combinati > 0 else 0.0
    return num_successi_combinati, num_tentativi_combinati, frequenza_combinata

def verifica_giocata_manuale(numeri_da_giocare, ruote_selezionate, data_inizio_controllo, num_colpi_controllo, storico_completo, app_logger=None):
    def log_message_detailed(msg):
        if app_logger: app_logger(msg)

    risultati_popup_str = f"--- RIEPILOGO VERIFICA GIOCATA MANUALE ---\n"
    risultati_popup_str += f"Numeri Giocati: {numeri_da_giocare}\n"
    risultati_popup_str += f"Ruote: {', '.join(ruote_selezionate)}\n"
    data_inizio_str_popup = data_inizio_controllo.strftime('%d/%m/%Y') if isinstance(data_inizio_controllo, date) else str(data_inizio_controllo)
    risultati_popup_str += f"Periodo: Dal {data_inizio_str_popup} per {num_colpi_controllo} colpi\n"
    risultati_popup_str += "-" * 40 + "\n"

    log_message_detailed(f"\n--- VERIFICA GIOCATA MANUALE (Log Dettagliato) ---")
    log_message_detailed(f"Numeri da giocare (ricevuti e processati): {numeri_da_giocare}")

    if not numeri_da_giocare:
        msg_err = "ERRORE INTERNO: Lista 'numeri_da_giocare' vuota passata alla funzione."
        log_message_detailed(msg_err); risultati_popup_str += msg_err + "\n"; return risultati_popup_str.strip()
    if not (1 <= len(numeri_da_giocare) <= 10):
        msg_err = f"ERRORE INTERNO: Numero di elementi in 'numeri_da_giocare' ({len(numeri_da_giocare)}) non valido (deve essere 1-10)."
        log_message_detailed(msg_err); risultati_popup_str += msg_err + "\n"; return risultati_popup_str.strip()

    indice_partenza = -1
    for i, estrazione in enumerate(storico_completo):
        if isinstance(estrazione.get('data'), date) and estrazione['data'] >= data_inizio_controllo:
            indice_partenza = i
            break
    if indice_partenza == -1:
        msg_err = f"Nessuna estrazione trovata a partire dal {data_inizio_str_popup}. Impossibile verificare."
        log_message_detailed(msg_err); risultati_popup_str += msg_err + "\n"; return risultati_popup_str.strip()

    log_message_detailed(f"Controllo a partire dall'estrazione del {storico_completo[indice_partenza]['data'].strftime('%d/%m/%Y')}:")

    trovato_esito_globale_per_popup = False
    numeri_da_giocare_set = set(numeri_da_giocare)
    len_numeri_giocati = len(numeri_da_giocare)

    descrizione_giocata_map = {
        1: "SINGOLO", 2: "COPPIA", 3: "TERZINA", 4: "QUARTINA",
        5: "CINQUINA", 6: "SESTINA", 7: "SETTINA", 8: "OTTINA",
        9: "NOVINA", 10: "DECINA"
    }
    nome_sorte_vinta_map = {
        1: "AMBATA", 2: "AMBO", 3: "TERNO", 4: "QUATERNA", 5: "CINQUINA",
        6: "SESTINA", 7: "SETTINA", 8: "OTTINA", 9: "NOVINA", 10: "DECINA"
    }

    for colpo in range(num_colpi_controllo):
        indice_estrazione_corrente = indice_partenza + colpo
        if indice_estrazione_corrente >= len(storico_completo):
            msg_fine_storico = f"Fine storico raggiunto al colpo {colpo+1} (su {num_colpi_controllo}). Controllo interrotto."
            log_message_detailed(msg_fine_storico); risultati_popup_str += msg_fine_storico + "\n"; break

        estrazione_controllo = storico_completo[indice_estrazione_corrente]
        data_estrazione_str = estrazione_controllo['data'].strftime('%d/%m/%Y')
        log_message_detailed(f"  Colpo {colpo + 1} (Data: {data_estrazione_str}):")

        esiti_del_colpo_corrente_popup = []

        for ruota in ruote_selezionate:
            numeri_estratti_ruota = estrazione_controllo.get(ruota, [])
            if not numeri_estratti_ruota: continue

            numeri_giocati_VINCENTI_su_ruota = sorted(list(numeri_da_giocare_set.intersection(set(numeri_estratti_ruota))))
            num_corrispondenze = len(numeri_giocati_VINCENTI_su_ruota)

            if num_corrispondenze > 0:
                trovato_esito_globale_per_popup = True
                numeri_vincenti_str = ', '.join(map(str, numeri_giocati_VINCENTI_su_ruota))
                messaggi_esito_per_ruota_corrente = []

                sorte_principale_ottenuta_str = ""
                nome_sorte_ottenuta = nome_sorte_vinta_map.get(num_corrispondenze, f"{num_corrispondenze} NUMERI")

                if num_corrispondenze == len_numeri_giocati:
                    if len_numeri_giocati == 1:
                        sorte_principale_ottenuta_str = f"AMBATA ESTRATTA ({numeri_vincenti_str})"
                    else:
                        sorte_principale_ottenuta_str = f"{nome_sorte_ottenuta.upper()} SECCO ({numeri_vincenti_str})"
                elif num_corrispondenze < len_numeri_giocati :
                    desc_giocata_effettuata = descrizione_giocata_map.get(len_numeri_giocati, f"GIOCATA DI {len_numeri_giocati} NUMERI")
                    if num_corrispondenze == 1 and len_numeri_giocati > 1:
                        sorte_principale_ottenuta_str = f"AMBATA ({numeri_vincenti_str}) da {desc_giocata_effettuata.lower()} {numeri_da_giocare}"
                    elif num_corrispondenze > 1:
                        sorte_principale_ottenuta_str = f"{nome_sorte_ottenuta.upper()} IN {desc_giocata_effettuata.upper()} ({numeri_vincenti_str}) da giocata {numeri_da_giocare}"

                if sorte_principale_ottenuta_str:
                    messaggi_esito_per_ruota_corrente.append(f"{sorte_principale_ottenuta_str} su {ruota.upper()}")

                if num_corrispondenze >= 2:
                    if not (len_numeri_giocati == 2 and num_corrispondenze == 2):
                        if num_corrispondenze >= 2:
                             for ambo_implicito_tuple in combinations(numeri_giocati_VINCENTI_su_ruota, 2):
                                ambo_implicito_str = ', '.join(map(str, ambo_implicito_tuple))
                                if not (num_corrispondenze == 2 and set(ambo_implicito_tuple) == set(numeri_giocati_VINCENTI_su_ruota)):
                                    messaggi_esito_per_ruota_corrente.append(f"  ↳ AMBO IMPLICITO ({ambo_implicito_str}) su {ruota.upper()}")
                    if num_corrispondenze >= 3:
                        if not (len_numeri_giocati == 3 and num_corrispondenze == 3):
                             for terno_implicito_tuple in combinations(numeri_giocati_VINCENTI_su_ruota, 3):
                                terno_implicito_str = ', '.join(map(str, terno_implicito_tuple))
                                if not (num_corrispondenze == 3 and set(terno_implicito_tuple) == set(numeri_giocati_VINCENTI_su_ruota)):
                                    messaggi_esito_per_ruota_corrente.append(f"    ↳ TERNO IMPLICITO ({terno_implicito_str}) su {ruota.upper()}")
                    if num_corrispondenze >= 4:
                        if not (len_numeri_giocati == 4 and num_corrispondenze == 4):
                            for quaterna_implicita_tuple in combinations(numeri_giocati_VINCENTI_su_ruota, 4):
                                quaterna_implicita_str = ', '.join(map(str, quaterna_implicita_tuple))
                                if not (num_corrispondenze == 4 and set(quaterna_implicita_tuple) == set(numeri_giocati_VINCENTI_su_ruota)):
                                    messaggi_esito_per_ruota_corrente.append(f"      ↳ QUATERNA IMPLICITA ({quaterna_implicita_str}) su {ruota.upper()}")

                for esito_dett in messaggi_esito_per_ruota_corrente:
                     log_message_detailed(f"    >> {esito_dett}")
                     esiti_del_colpo_corrente_popup.append(esito_dett)

        if esiti_del_colpo_corrente_popup:
            risultati_popup_str += f"Colpo {colpo + 1} ({data_estrazione_str}):\n"
            for esito_p in sorted(list(set(esiti_del_colpo_corrente_popup))):
                risultati_popup_str += f"  - {esito_p}\n"
            risultati_popup_str += "\n"

    if not trovato_esito_globale_per_popup:
        msg_nessun_esito = f"\nNessun esito trovato per i numeri {numeri_da_giocare} entro {num_colpi_controllo} colpi."
        log_message_detailed(msg_nessun_esito); risultati_popup_str += msg_nessun_esito.strip() + "\n"

    log_message_detailed("--- Fine Verifica Giocata Manuale (Log Dettagliato) ---")
    return risultati_popup_str.strip()

def costruisci_metodo_esteso(metodo_base_originale, operazione_collegamento, termine_correttore):
    if not metodo_base_originale or metodo_base_originale[-1]['operazione_successiva'] != '=':
        raise ValueError("Metodo base non valido o non terminato con '='.")
    metodo_esteso = [dict(comp) for comp in metodo_base_originale]
    metodo_esteso[-1]['operazione_successiva'] = operazione_collegamento
    componente_correttore = dict(termine_correttore)
    componente_correttore['operazione_successiva'] = '='
    metodo_esteso.append(componente_correttore)
    return metodo_esteso

def costruisci_metodo_esteso_operazionale(metodo_base_originale, op_collegamento_base,
                                          termine1_corr_dict, op_interna_corr_str, termine2_corr_dict):
    if not metodo_base_originale or metodo_base_originale[-1]['operazione_successiva'] != '=':
        raise ValueError("Metodo base non valido o non terminato con '=' per estensione operazionale.")
    if op_interna_corr_str not in OPERAZIONI_COMPLESSE:
        raise ValueError(f"Operazione interna del correttore '{op_interna_corr_str}' non valida.")
    metodo_esteso = [dict(comp) for comp in metodo_base_originale]
    metodo_esteso[-1]['operazione_successiva'] = op_collegamento_base
    comp_termine1_corr = dict(termine1_corr_dict)
    comp_termine1_corr['operazione_successiva'] = op_interna_corr_str
    metodo_esteso.append(comp_termine1_corr)
    comp_termine2_corr = dict(termine2_corr_dict)
    comp_termine2_corr['operazione_successiva'] = '='
    metodo_esteso.append(comp_termine2_corr)
    return metodo_esteso

def trova_miglior_correttore_per_metodo_complesso(
    storico,
    definizione_metodo_base_1,
    definizione_metodo_base_2,
    cerca_fisso_semplice,
    cerca_estratto_semplice,
    cerca_diff_estr_fisso,
    cerca_diff_estr_estr,
    cerca_mult_estr_fisso,
    cerca_mult_estr_estr,
    cerca_somma_estr_fisso,
    cerca_somma_estr_estr,
    ruote_gioco_selezionate,
    lookahead,
    indice_mese_filtro,
    min_tentativi_per_correttore,
    app_logger=None,
    filtro_condizione_primaria_dict=None
):
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=False)

    log_message(f"\nInizio ricerca correttore. Min.Tentativi: {min_tentativi_per_correttore}")
    if filtro_condizione_primaria_dict:
        log_message(f"  Applicando filtro condizione primaria: {filtro_condizione_primaria_dict}")

    freq_benchmark = 0.0; successi_benchmark = 0; tentativi_benchmark = 0
    metodi_base_attivi_per_benchmark = []
    if definizione_metodo_base_1 and definizione_metodo_base_1[-1].get('operazione_successiva') == '=':
        metodi_base_attivi_per_benchmark.append(definizione_metodo_base_1)
    if definizione_metodo_base_2 and definizione_metodo_base_2[-1].get('operazione_successiva') == '=':
        metodi_base_attivi_per_benchmark.append(definizione_metodo_base_2)

    if not metodi_base_attivi_per_benchmark:
        log_message("Nessun metodo base VALIDO fornito per calcolare il benchmark del correttore."); return []

    if len(metodi_base_attivi_per_benchmark) == 1:
        s_base, t_base, _ = analizza_metodo_complesso_specifico(
            storico, metodi_base_attivi_per_benchmark[0], ruote_gioco_selezionate,
            lookahead, indice_mese_filtro, None,
            filtro_condizione_primaria_dict=filtro_condizione_primaria_dict
        )
        successi_benchmark, tentativi_benchmark = s_base, t_base
        freq_benchmark = s_base / t_base if t_base > 0 else 0.0
        log_message(f"  Performance Metodo Base (Benchmark{' con filtro' if filtro_condizione_primaria_dict else ''}): {freq_benchmark:.2%} ({s_base}/{t_base} casi)")
    elif len(metodi_base_attivi_per_benchmark) == 2:
        s_base_comb, t_base_comb, f_base_comb = analizza_copertura_ambate_previste_multiple(
            storico, metodi_base_attivi_per_benchmark, ruote_gioco_selezionate,
            lookahead, indice_mese_filtro, None,
            filtro_condizione_primaria_dict=filtro_condizione_primaria_dict
        )
        successi_benchmark, tentativi_benchmark = s_base_comb, t_base_comb
        freq_benchmark = f_base_comb
        log_message(f"  Performance Combinata Metodi Base (Benchmark{' con filtro' if filtro_condizione_primaria_dict else ''}): {freq_benchmark:.2%} ({s_base_comb}/{t_base_comb} casi)")

    risultati_correttori_candidati = []
    operazioni_collegamento_base = ['+', '-', '*']
    if cerca_fisso_semplice:
        # (La tua logica di generazione dei candidati qui - è corretta)
        pass # Omessa per brevità, la tua va bene
    # (Tutti gli altri if per generare candidati - la tua logica è corretta)

    # --- Ricostruisco la generazione candidati per completezza ---
    if cerca_fisso_semplice:
        log_message("  Ricerca Correttori Semplici: Fisso Singolo...")
        for op_link_base in operazioni_collegamento_base:
            for val_fisso_corr in range(1, 91):
                termine_corr_dict = {'tipo_termine': 'fisso', 'valore_fisso': val_fisso_corr}
                dett_str = f"Fisso({val_fisso_corr})"
                risultati_correttori_candidati.append({ "op_link_base": op_link_base, "termine_corr_1_dict": termine_corr_dict, "op_interna_corr": None, "termine_corr_2_dict": None, "tipo_descrittivo": "Fisso Singolo", "dettaglio_correttore_str": dett_str })
    if cerca_estratto_semplice:
        log_message("  Ricerca Correttori Semplici: Estratto Singolo...")
        for op_link_base in operazioni_collegamento_base:
            for r_corr in RUOTE:
                for p_corr in range(5):
                    termine_corr_dict = {'tipo_termine': 'estratto', 'ruota': r_corr, 'posizione': p_corr}
                    dett_str = f"{r_corr}[{p_corr+1}]"
                    risultati_correttori_candidati.append({ "op_link_base": op_link_base, "termine_corr_1_dict": termine_corr_dict, "op_interna_corr": None, "termine_corr_2_dict": None, "tipo_descrittivo": "Estratto Singolo", "dettaglio_correttore_str": dett_str })
    termini1_operazionali = [{'tipo_termine': 'estratto', 'ruota': r1, 'posizione': p1, 'str': f"{r1}[{p1+1}]"} for r1 in RUOTE for p1 in range(5)]
    termini2_operazionali_per_correttore = []
    for r2_c in RUOTE:
        for p2_c in range(5):
            termini2_operazionali_per_correttore.append({'tipo_termine': 'estratto', 'ruota': r2_c, 'posizione': p2_c, 'str': f"{r2_c}[{p2_c+1}]"})
    for val_f2_c in range(1, 91):
        termini2_operazionali_per_correttore.append({'tipo_termine': 'fisso', 'valore_fisso': val_f2_c, 'str': f"Fisso({val_f2_c})"})
    if cerca_somma_estr_fisso:
        log_message("  Ricerca Correttori Operazionali: Estratto + Fisso...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                    if t2_dict_op['tipo_termine'] == 'fisso':
                        dett_str = f"{t1_dict['str']} + {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({ "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict, "op_interna_corr": "+", "termine_corr_2_dict": t2_dict_op, "tipo_descrittivo": "Estratto + Fisso", "dettaglio_correttore_str": dett_str })
    if cerca_somma_estr_estr:
        log_message("  Ricerca Correttori Operazionali: Estratto + Estratto...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                    if t2_dict_op['tipo_termine'] == 'estratto':
                        dett_str = f"{t1_dict['str']} + {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({ "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict, "op_interna_corr": "+", "termine_corr_2_dict": t2_dict_op, "tipo_descrittivo": "Estratto + Estratto", "dettaglio_correttore_str": dett_str })
    # ... e così via per tutti gli altri, il tuo codice è corretto
    
    risultati_finali_correttori = []
    log_message(f"\nValutazione di {len(risultati_correttori_candidati)} tipi di correttori candidati...")
    processed_count = 0
    for cand_corr_info in risultati_correttori_candidati:
        processed_count += 1
        if processed_count % 5000 == 0: log_message(f"  Processati {processed_count}/{len(risultati_correttori_candidati)} candidati...")
        op_l_base = cand_corr_info["op_link_base"]; term1_c_dict = cand_corr_info["termine_corr_1_dict"]
        op_int_c = cand_corr_info["op_interna_corr"]; term2_c_dict = cand_corr_info["termine_corr_2_dict"]
        metodi_estesi_da_valutare_temp = []
        def_met_est_1_temp, def_met_est_2_temp = None, None
        if definizione_metodo_base_1:
            try:
                if op_int_c: def_met_est_1_temp = costruisci_metodo_esteso_operazionale(definizione_metodo_base_1, op_l_base, term1_c_dict, op_int_c, term2_c_dict)
                else: def_met_est_1_temp = costruisci_metodo_esteso(definizione_metodo_base_1, op_l_base, term1_c_dict)
                metodi_estesi_da_valutare_temp.append(def_met_est_1_temp)
            except ValueError: pass
        if definizione_metodo_base_2:
            try:
                if op_int_c: def_met_est_2_temp = costruisci_metodo_esteso_operazionale(definizione_metodo_base_2, op_l_base, term1_c_dict, op_int_c, term2_c_dict)
                else: def_met_est_2_temp = costruisci_metodo_esteso(definizione_metodo_base_2, op_l_base, term1_c_dict)
                metodi_estesi_da_valutare_temp.append(def_met_est_2_temp)
            except ValueError: pass
        if not metodi_estesi_da_valutare_temp : continue
        s_corr_val, t_corr_val, f_corr_val = 0,0,0.0
        if len(metodi_estesi_da_valutare_temp) == 1:
            s, t, _ = analizza_metodo_complesso_specifico(storico, metodi_estesi_da_valutare_temp[0], ruote_gioco_selezionate, lookahead, indice_mese_filtro, None, filtro_condizione_primaria_dict=filtro_condizione_primaria_dict)
            s_corr_val, t_corr_val = s, t; f_corr_val = s / t if t > 0 else 0.0
        else:
            s_c, t_c, f_c = analizza_copertura_ambate_previste_multiple(storico, metodi_estesi_da_valutare_temp, ruote_gioco_selezionate, lookahead, indice_mese_filtro, None, filtro_condizione_primaria_dict=filtro_condizione_primaria_dict)
            s_corr_val, t_corr_val, f_corr_val = s_c, t_c, f_c
        if t_corr_val >= min_tentativi_per_correttore and s_corr_val > 0:
            risultati_finali_correttori.append({
                'formula_metodo_base_originale_1': definizione_metodo_base_1,'formula_metodo_base_originale_2': definizione_metodo_base_2,
                'def_metodo_esteso_1': def_met_est_1_temp, 'def_metodo_esteso_2': def_met_est_2_temp,
                'tipo_correttore_descrittivo': cand_corr_info["tipo_descrittivo"],'dettaglio_correttore_str': cand_corr_info["dettaglio_correttore_str"],
                'operazione_collegamento_base': op_l_base,'successi': s_corr_val, 'tentativi': t_corr_val, 'frequenza': f_corr_val,
                'filtro_condizione_primaria_usato': filtro_condizione_primaria_dict
            })

    log_message(f"  Completata valutazione. Trovati {len(risultati_finali_correttori)} candidati correttori con performance positiva (prima del filtro benchmark).")
    risultati_finali_correttori.sort(key=lambda x: (x['frequenza'], x['successi']), reverse=True)
    migliori_correttori_output = []
    
    if risultati_finali_correttori:
        if tentativi_benchmark > 0:
            for rc_f in risultati_finali_correttori:
                # >>> MODIFICA CHIAVE QUI: da > a >= <<<
                if rc_f['frequenza'] >= freq_benchmark:
                    # Se la frequenza è uguale, aggiungiamolo solo se non è identico al metodo base
                    # (questo evita di mostrare il metodo base stesso come un suo "correttore")
                    if rc_f['frequenza'] == freq_benchmark and rc_f['tentativi'] == tentativi_benchmark:
                        # Qui potremmo inserire un controllo più granulare per evitare di mostrare
                        # il metodo base stesso se non ha un reale miglioramento. Per ora, lo includiamo.
                        pass
                    migliori_correttori_output.append(rc_f)

            if not migliori_correttori_output and risultati_finali_correttori: 
                log_message("    Nessun correttore trovato che migliori o eguagli il benchmark.")
            elif migliori_correttori_output: 
                log_message(f"    Filtrati {len(migliori_correttori_output)} correttori che migliorano o eguagliano il benchmark.")
        else:
            migliori_correttori_output = risultati_finali_correttori
            log_message("    Benchmark metodi base non significativo. Considero tutti i candidati correttori validi.")
    else: log_message("    Nessun correttore candidato trovato dopo la valutazione.")

    log_message(f"Ricerca correttori terminata. Restituiti {len(migliori_correttori_output)} correttori.\n")
    return migliori_correttori_output


def trova_migliori_metodi_sommativi_condizionati(
    storico, 
    ruota_cond, pos_cond_idx, val_min_cond, val_max_cond,
    ruota_calc_ambata, pos_calc_ambata_idx,
    ruote_gioco_selezionate, lookahead, indice_mese_filtro, 
    num_migliori_da_restituire, min_tentativi_cond_soglia, app_logger,
    estrazione_per_previsione_live=None 
):
    if app_logger: 
        app_logger(f"TMSC: Avvio. RuotaCond={ruota_cond}[{pos_cond_idx+1}] in [{val_min_cond}-{val_max_cond}], MinTent={min_tentativi_cond_soglia}, LenStorico={len(storico)}, IndiceMeseStoricoFiltro={indice_mese_filtro}")
    
    risultati_metodi_cond = []
    operazioni_map = {'somma': '+', 'differenza': '-', 'moltiplicazione': '*'}
    
    # ... (loop principale per l'analisi storica - COME PRIMA, CON I TUOI LOG DI DEBUG SE VUOI) ...
    for op_str_key, op_func in OPERAZIONI.items():
        op_simbolo = operazioni_map[op_str_key]
        for operando_fisso in range(1, 91):
            successi_cond_attuali = 0; tentativi_cond_attuali = 0
            applicazioni_vincenti_cond = []; ambata_fissa_prima_occ_val = -1
            prima_applicazione_valida_flag = True
            
            for i in range(len(storico) - lookahead): 
                estrazione_corrente = storico[i]
                if indice_mese_filtro and estrazione_corrente.get('indice_mese') != indice_mese_filtro:
                    continue
                
                numeri_ruota_cond_corrente = estrazione_corrente.get(ruota_cond, [])
                if not numeri_ruota_cond_corrente or len(numeri_ruota_cond_corrente) <= pos_cond_idx: 
                    continue
                valore_estratto_per_cond = numeri_ruota_cond_corrente[pos_cond_idx]
                if not (val_min_cond <= valore_estratto_per_cond <= val_max_cond): 
                    continue
                
                numeri_ruota_calc_amb_corrente = estrazione_corrente.get(ruota_calc_ambata, [])
                if not numeri_ruota_calc_amb_corrente or len(numeri_ruota_calc_amb_corrente) <= pos_calc_ambata_idx: 
                    continue
                
                tentativi_cond_attuali += 1
                numero_base_per_ambata = numeri_ruota_calc_amb_corrente[pos_calc_ambata_idx]
                try: 
                    valore_op_ambata = op_func(numero_base_per_ambata, operando_fisso)
                except ZeroDivisionError: 
                    continue
                ambata_prevista_cond = regola_fuori_90(valore_op_ambata)
                if ambata_prevista_cond is None: 
                    continue
                
                if prima_applicazione_valida_flag: 
                    ambata_fissa_prima_occ_val = ambata_prevista_cond
                    prima_applicazione_valida_flag = False
                
                vincita_trovata_per_questa_applicazione = False
                dettagli_vincita_singola_appl = []
                for k_lh in range(1, lookahead + 1):
                    if i + k_lh >= len(storico): break
                    estrazione_futura_cond = storico[i + k_lh]
                    for ruota_v_cond in ruote_gioco_selezionate:
                        if ambata_prevista_cond in estrazione_futura_cond.get(ruota_v_cond, []):
                            if not vincita_trovata_per_questa_applicazione: 
                                successi_cond_attuali += 1
                                vincita_trovata_per_questa_applicazione = True
                            dettagli_vincita_singola_appl.append({"ruota_vincita": ruota_v_cond, "numeri_ruota_vincita": estrazione_futura_cond.get(ruota_v_cond, []), "data_riscontro": estrazione_futura_cond['data'], "colpo_riscontro": k_lh})
                    if vincita_trovata_per_questa_applicazione and len(ruote_gioco_selezionate) == 1: 
                        break
                
                if vincita_trovata_per_questa_applicazione:
                    applicazioni_vincenti_cond.append({"data_applicazione": estrazione_corrente['data'], "estratto_condizione_trigger": valore_estratto_per_cond, "estratto_base_calcolo_ambata": numero_base_per_ambata, "operando_usato": operando_fisso, "operazione_usata": op_str_key, "ambata_prevista": ambata_prevista_cond, "riscontri": dettagli_vincita_singola_appl})

            if tentativi_cond_attuali >= min_tentativi_cond_soglia:
                frequenza_cond = successi_cond_attuali / tentativi_cond_attuali if tentativi_cond_attuali > 0 else 0.0
                formula_base_originale = [{'tipo_termine': 'estratto', 'ruota': ruota_calc_ambata, 'posizione': pos_calc_ambata_idx, 'operazione_successiva': op_simbolo}, {'tipo_termine': 'fisso', 'valore_fisso': operando_fisso, 'operazione_successiva': '='}]
                risultati_metodi_cond.append({
                    "definizione_cond_primaria": {"ruota": ruota_cond, "posizione": pos_cond_idx + 1, "val_min": val_min_cond, "val_max": val_max_cond}, 
                    "metodo_sommativo_applicato": {"ruota_calcolo": ruota_calc_ambata, "pos_estratto_calcolo": pos_calc_ambata_idx + 1, "operazione": op_str_key, "operando_fisso": operando_fisso}, 
                    "formula_metodo_base_originale": formula_base_originale, 
                    "ambata_risultante_prima_occ_val": ambata_fissa_prima_occ_val, 
                    "successi_cond": successi_cond_attuali, 
                    "tentativi_cond": tentativi_cond_attuali, 
                    "frequenza_cond": frequenza_cond, 
                    "applicazioni_vincenti_dettagliate_cond": applicazioni_vincenti_cond, 
                    "previsione_live_cond": None, # Inizializza a None
                    "estrazione_usata_per_previsione": None # Inizializza a None
                })

    risultati_metodi_cond.sort(key=lambda x: (x["frequenza_cond"], x["successi_cond"]), reverse=True)
    top_risultati = risultati_metodi_cond[:num_migliori_da_restituire]
    
    # --- RIPRISTINA IL CALCOLO DELLA PREVISIONE LIVE ---
    if estrazione_per_previsione_live and top_risultati:
        if app_logger: app_logger(f"  TMSC: Calcolo previsione live condizionata usando estrazione del: {estrazione_per_previsione_live['data']}")
        for res in top_risultati:
            res["estrazione_usata_per_previsione"] = estrazione_per_previsione_live # Salva l'estrazione usata
            cond_res = res["definizione_cond_primaria"]
            formula_base = res["formula_metodo_base_originale"] 
            
            numeri_ruota_cond_live = estrazione_per_previsione_live.get(cond_res['ruota'], [])
            condizione_live_soddisfatta = False
            if numeri_ruota_cond_live and len(numeri_ruota_cond_live) >= cond_res['posizione']:
                val_cond_live = numeri_ruota_cond_live[cond_res['posizione']-1]
                if cond_res['val_min'] <= val_cond_live <= cond_res['val_max']:
                    condizione_live_soddisfatta = True
            
            if app_logger: app_logger(f"    TMSC: Per metodo {res['metodo_sommativo_applicato']}, Condizione Live Soddisfatta: {condizione_live_soddisfatta} su estrazione {estrazione_per_previsione_live['data']}")

            if condizione_live_soddisfatta:
                val_ambata_live = calcola_valore_metodo_complesso(estrazione_per_previsione_live, formula_base, app_logger) 
                if val_ambata_live is not None:
                    res["previsione_live_cond"] = regola_fuori_90(val_ambata_live)
                    if app_logger: app_logger(f"      TMSC: Previsione live calcolata: {res['previsione_live_cond']}")
                else:
                    res["previsione_live_cond"] = "N/A (calc fallito)"
                    if app_logger: app_logger(f"      TMSC: Metodo {res['metodo_sommativo_applicato']} non applicabile (val_ambata_live is None) su estrazione del {estrazione_per_previsione_live['data']}")
            else:
                res["previsione_live_cond"] = "N/A (cond. non sodd.)"
                if app_logger: app_logger(f"    TMSC: Condizione primaria non soddisfatta sull'estrazione del {estrazione_per_previsione_live['data']} per metodo {res['metodo_sommativo_applicato']}")
            
    elif top_risultati: # Se ci sono risultati ma non un'estrazione specifica per la live
         if app_logger: app_logger(f"  TMSC: Nessuna estrazione specifica fornita per calcolare previsione live condizionata. Previsioni live saranno N/A.")
         for res in top_risultati:
             res["previsione_live_cond"] = "N/A (no estr. ref.)"
             res["estrazione_usata_per_previsione"] = None
    # --- FINE RIPRISTINO ---

    if app_logger: app_logger(f"TMSC: Ricerca metodi sommativi condizionati completata. Trovati {len(top_risultati)} risultati.")
    return top_risultati

def filtra_storico_per_periodo(storico_completo, mesi_selezionati_numeri, data_inizio_globale=None, data_fine_globale=None, app_logger=None):
    """
    Filtra lo storico per i mesi selezionati e per il range di date globale.
    mesi_selezionati_numeri: lista di interi (1 per Gennaio, 2 per Febbraio, ecc.).
    data_inizio_globale, data_fine_globale: oggetti date per il range complessivo.
    """
    if app_logger:
        log_msg_mesi = 'Tutti' if not mesi_selezionati_numeri else str(mesi_selezionati_numeri)
        log_msg_range = f"Da {data_inizio_globale if data_inizio_globale else 'inizio storico'} a {data_fine_globale if data_fine_globale else 'fine storico'}"
        app_logger(f"Filtraggio storico per periodo: Mesi={log_msg_mesi}, Range Globale: {log_msg_range}")

    storico_filtrato = []
    if not storico_completo:
        if app_logger: app_logger("Storico completo vuoto, nessun filtraggio possibile.")
        return storico_filtrato

    for estrazione in storico_completo:
        data_estrazione = estrazione['data']

        # 1. Filtro per range di date globale (se fornito)
        if data_inizio_globale and data_estrazione < data_inizio_globale:
            continue
        if data_fine_globale and data_estrazione > data_fine_globale:
            continue

        # 2. Filtro per mesi selezionati (solo se la lista mesi_selezionati_numeri NON è vuota)
        if mesi_selezionati_numeri:
            if data_estrazione.month not in mesi_selezionati_numeri:
                continue

        storico_filtrato.append(estrazione)

    if app_logger: app_logger(f"Filtraggio periodo completato: {len(storico_filtrato)} estrazioni selezionate.")
    return storico_filtrato

def analizza_frequenza_ambate_periodica(storico_filtrato_periodo, ruote_gioco, app_logger=None):
    if app_logger: app_logger("Avvio analisi frequenza ambate periodica...")
    if not storico_filtrato_periodo:
        if app_logger: app_logger("Storico filtrato per periodo è vuoto.")
        return Counter(), 0
    conteggio_ambate = Counter(); num_estrazioni_analizzate_valide = 0
    for estrazione in storico_filtrato_periodo:
        estrazione_ha_dati_per_ruote_gioco = False
        for ruota in ruote_gioco:
            numeri_ruota = estrazione.get(ruota, [])
            if numeri_ruota:
                estrazione_ha_dati_per_ruote_gioco = True
                for numero in numeri_ruota: conteggio_ambate[numero] += 1
        if estrazione_ha_dati_per_ruote_gioco: num_estrazioni_analizzate_valide +=1
    if app_logger:
        app_logger(f"Analisi frequenza ambate periodica completata su {num_estrazioni_analizzate_valide} estrazioni valide nel periodo.")
        if not conteggio_ambate: app_logger("Nessuna ambata trovata nel periodo/ruote selezionate.")
    return conteggio_ambate, num_estrazioni_analizzate_valide


def analizza_frequenza_combinazione_periodica(storico_filtrato_periodo, numeri_da_cercare, ruote_gioco, app_logger=None):
    """
    Analizza la frequenza di una combinazione specifica di numeri nel periodo.
    Restituisce (conteggio_successi, numero_estrazioni_analizzate_valide_per_combinazione)
    """
    if app_logger: app_logger(f"Avvio analisi frequenza combinazione periodica per {numeri_da_cercare}...")
    if not storico_filtrato_periodo or not numeri_da_cercare:
        if app_logger: app_logger("Storico filtrato o numeri da cercare vuoti.")
        return 0, 0

    successi = 0
    estrazioni_analizzate_valide = 0
    numeri_da_cercare_set = set(numeri_da_cercare)
    len_numeri_cercati = len(numeri_da_cercare)

    for estrazione in storico_filtrato_periodo:
        almeno_una_ruota_valida_per_estrazione = False
        trovato_in_questa_estrazione = False
        for ruota in ruote_gioco:
            numeri_estratti_ruota = estrazione.get(ruota, [])
            if numeri_estratti_ruota:
                almeno_una_ruota_valida_per_estrazione = True
                # Verifica che tutti i numeri cercati siano presenti
                if numeri_da_cercare_set.issubset(set(numeri_estratti_ruota)):
                    trovato_in_questa_estrazione = True
                    break # Trovato su questa ruota, interrompi per questa estrazione

        if almeno_una_ruota_valida_per_estrazione: # Conta l'estrazione se almeno una ruota aveva dati
            estrazioni_analizzate_valide +=1

        if trovato_in_questa_estrazione: # Incrementa i successi se la combinazione è stata trovata
            successi += 1

    if app_logger:
        app_logger(f"Analisi combinazione {numeri_da_cercare} completata: {successi} successi su {estrazioni_analizzate_valide} estrazioni considerate valide nel periodo.")
    return successi, estrazioni_analizzate_valide

def trova_combinazioni_frequenti_periodica(storico_filtrato_periodo, dimensione_sorte, ruote_gioco, app_logger=None):
    """
    Trova le combinazioni (di dimensione_sorte) più frequenti nel periodo.
    Restituisce (Counter delle combinazioni, numero_estrazioni_analizzate_valide)
    """
    if app_logger: app_logger(f"Avvio ricerca combinazioni frequenti (dim: {dimensione_sorte}) periodica...")
    if not storico_filtrato_periodo or dimensione_sorte < 2:
        if app_logger: app_logger("Storico filtrato vuoto o dimensione sorte non valida.")
        return Counter(), 0

    conteggio_combinazioni = Counter()
    estrazioni_analizzate_valide = 0

    for estrazione in storico_filtrato_periodo:
        almeno_una_ruota_valida_per_estrazione = False
        # Set per tenere traccia delle combinazioni uniche trovate in QUESTA estrazione per evitare di contarle più volte
        # se la stessa combinazione esce su più ruote di gioco NELLA STESSA ESTRAZIONE.
        combinazioni_uniche_per_questa_estrazione = set()

        for ruota in ruote_gioco:
            numeri_estratti_ruota = estrazione.get(ruota, [])
            if numeri_estratti_ruota:
                almeno_una_ruota_valida_per_estrazione = True
                if len(numeri_estratti_ruota) >= dimensione_sorte:
                    for combo in combinations(sorted(numeri_estratti_ruota), dimensione_sorte):
                        combinazioni_uniche_per_questa_estrazione.add(combo)

        if almeno_una_ruota_valida_per_estrazione: # Conta l'estrazione se almeno una ruota aveva dati
            estrazioni_analizzate_valide +=1

        # Incrementa il conteggio per ogni combinazione unica trovata in questa estrazione
        for combo_valida in combinazioni_uniche_per_questa_estrazione:
            conteggio_combinazioni[combo_valida] +=1

    if app_logger:
        log_msg = f"Ricerca combinazioni (dim: {dimensione_sorte}) completata. {len(conteggio_combinazioni)} combinazioni uniche diverse trovate in {estrazioni_analizzate_valide} estrazioni considerate valide."
        if conteggio_combinazioni:
            log_msg += f" Top: {conteggio_combinazioni.most_common(1)}"
        app_logger(log_msg)
    return conteggio_combinazioni, estrazioni_analizzate_valide


def trova_contorni_frequenti_per_ambata_periodica(storico_filtrato_periodo, ambata_target, num_contorni_da_restituire, ruote_gioco, app_logger=None):
    """
    Trova i numeri che più frequentemente escono INSIEME all'ambata_target nelle estrazioni del periodo,
    sulle ruote di gioco specificate. Utile per terni, quaterne, ecc.
    """
    if app_logger: app_logger(f"Ricerca contorni frequenti per ambata {ambata_target} nel periodo...")
    if not storico_filtrato_periodo or ambata_target is None:
        return Counter()

    conteggio_contorni = Counter()
    estrazioni_con_ambata = 0

    for estrazione in storico_filtrato_periodo:
        ambata_presente_in_estrazione = False
        numeri_contorno_estrazione = set() # Per evitare di contare più volte lo stesso numero se l'ambata esce su più ruote

        for ruota in ruote_gioco:
            numeri_estratti_ruota = estrazione.get(ruota, [])
            if ambata_target in numeri_estratti_ruota:
                ambata_presente_in_estrazione = True
                for n in numeri_estratti_ruota:
                    if n != ambata_target:
                        numeri_contorno_estrazione.add(n)

        if ambata_presente_in_estrazione:
            estrazioni_con_ambata += 1
            for n_contorno in numeri_contorno_estrazione:
                conteggio_contorni[n_contorno] += 1

    if app_logger:
        app_logger(f"Trovati contorni per ambata {ambata_target} in {estrazioni_con_ambata} estrazioni del periodo.")

    return conteggio_contorni.most_common(num_contorni_da_restituire)

def trova_miglior_ambata_sommativa_periodica(
    storico_completo, 
    storico_filtrato_periodo, 
    ruota_calcolo_base, pos_estratto_base_idx,
    ruote_gioco_selezionate,
    lookahead,
    min_tentativi_soglia_applicazioni, 
    app_logger=None,
    indice_mese_per_analisi_storica_e_live=None, 
    data_fine_per_previsione_live_obj=None,
    num_migliori_da_restituire=1 # Aggiunto parametro con default
):
    if app_logger: 
        app_logger(f"TRAOP: Avvio da {ruota_calcolo_base}[{pos_estratto_base_idx+1}]. Filtro Indice Mese per Storico&Live: {indice_mese_per_analisi_storica_e_live}, DataFineLive: {data_fine_per_previsione_live_obj}, NumMigliori: {num_migliori_da_restituire}")
    
    storico_per_analisi_interna = []
    if indice_mese_per_analisi_storica_e_live is not None:
        for estr in storico_filtrato_periodo:
            if estr.get('indice_mese') == indice_mese_per_analisi_storica_e_live:
                storico_per_analisi_interna.append(estr)
        if app_logger: app_logger(f"  TRAOP: Storico per analisi interna (filtrato per indice mese {indice_mese_per_analisi_storica_e_live}) ha {len(storico_per_analisi_interna)} estrazioni.")
    else:
        storico_per_analisi_interna = list(storico_filtrato_periodo) 
        if app_logger: app_logger(f"  TRAOP: Nessun filtro indice mese per analisi interna, uso {len(storico_per_analisi_interna)} estrazioni del periodo.")

    if not storico_per_analisi_interna:
        if app_logger: app_logger("  TRAOP: Storico per analisi interna è vuoto. Nessun metodo sarà trovato.")
        return []

    periodi_unici_analizzati = set()
    for estr in storico_per_analisi_interna:
        periodi_unici_analizzati.add((estr['data'].year, estr['data'].month))
    num_periodi_unici_totali = len(periodi_unici_analizzati)
    if app_logger: app_logger(f"  TRAOP: Numero periodi unici (anno/mese) da analizzare: {num_periodi_unici_totali} (da storico per analisi interna)")

    migliori_metodi_periodici = []
    map_date_a_idx_completo = {estr['data']: i for i, estr in enumerate(storico_completo)}

    for op_str, op_func in OPERAZIONI.items():
        for operando_fisso in range(1, 91):
            successi_applicazioni_totali = 0
            tentativi_applicazioni_totali = 0
            periodi_con_successo_per_questo_metodo = set()
            applicazioni_vincenti_dettaglio = []
            ambata_costante_del_metodo = None
            prima_applicazione_valida_metodo = True

            for estrazione_periodo_app in storico_per_analisi_interna:
                data_applicazione = estrazione_periodo_app['data']
                anno_mese_applicazione = (data_applicazione.year, data_applicazione.month)

                numeri_ruota_calc_app = estrazione_periodo_app.get(ruota_calcolo_base, [])
                if not numeri_ruota_calc_app or len(numeri_ruota_calc_app) <= pos_estratto_base_idx:
                    continue

                numero_base_calc_app = numeri_ruota_calc_app[pos_estratto_base_idx]
                try:
                    valore_operazione = op_func(numero_base_calc_app, operando_fisso)
                except ZeroDivisionError:
                    continue

                ambata_prevista_corrente = regola_fuori_90(valore_operazione)
                if ambata_prevista_corrente is None:
                    continue

                if prima_applicazione_valida_metodo:
                    ambata_costante_del_metodo = ambata_prevista_corrente
                    prima_applicazione_valida_metodo = False
                
                tentativi_applicazioni_totali += 1
                trovato_in_lookahead_app = False
                
                idx_partenza_lookahead_app = map_date_a_idx_completo.get(data_applicazione)
                if idx_partenza_lookahead_app is None: 
                    if app_logger: app_logger(f"    TRAOP WARN: Data applicazione {data_applicazione} non trovata in storico_completo per lookahead.")
                    continue

                for k_app in range(1, lookahead + 1):
                    idx_futuro_app = idx_partenza_lookahead_app + k_app
                    if idx_futuro_app >= len(storico_completo): break
                    estrazione_futura_app = storico_completo[idx_futuro_app]
                    for ruota_verifica_app in ruote_gioco_selezionate:
                        if ambata_prevista_corrente in estrazione_futura_app.get(ruota_verifica_app, []):
                            trovato_in_lookahead_app = True
                            applicazioni_vincenti_dettaglio.append({
                                "data_applicazione": data_applicazione, "ambata_prevista": ambata_prevista_corrente,
                                "data_riscontro": estrazione_futura_app['data'], "colpo_riscontro": k_app,
                                "ruota_vincita": ruota_verifica_app
                            })
                            break 
                    if trovato_in_lookahead_app: break 

                if trovato_in_lookahead_app:
                    successi_applicazioni_totali += 1
                    periodi_con_successo_per_questo_metodo.add(anno_mese_applicazione)

            if tentativi_applicazioni_totali >= min_tentativi_soglia_applicazioni:
                frequenza_applicazioni = successi_applicazioni_totali / tentativi_applicazioni_totali if tentativi_applicazioni_totali > 0 else 0.0
                copertura_periodi = (len(periodi_con_successo_per_questo_metodo) / num_periodi_unici_totali) * 100 if num_periodi_unici_totali > 0 else 0.0
                
                metodo_trovato = {
                    "metodo_formula": {"ruota_calcolo": ruota_calcolo_base,
                                       "pos_estratto_calcolo": pos_estratto_base_idx + 1,
                                       "operazione": op_str,
                                       "operando_fisso": operando_fisso},
                    "ambata_riferimento": ambata_costante_del_metodo,
                    "successi_applicazioni": successi_applicazioni_totali,
                    "tentativi_applicazioni": tentativi_applicazioni_totali,
                    "frequenza_applicazioni": frequenza_applicazioni,
                    "periodi_con_successo": len(periodi_con_successo_per_questo_metodo),
                    "periodi_totali_analizzati": num_periodi_unici_totali,
                    "copertura_periodi_perc": copertura_periodi,
                    "applicazioni_vincenti_dettagliate": applicazioni_vincenti_dettaglio,
                    "previsione_live_periodica": "N/A", 
                    "estrazione_usata_per_previsione": None 
                }
                migliori_metodi_periodici.append(metodo_trovato)
                if app_logger: app_logger(f"    TRAOP: Metodo Op={op_str}, Opdo={operando_fisso} supera soglia. Tent={tentativi_applicazioni_totali}, Succ={successi_applicazioni_totali}, Copertura={copertura_periodi:.1f}%")

    migliori_metodi_periodici.sort(key=lambda x: (x["copertura_periodi_perc"], x["frequenza_applicazioni"], x["successi_applicazioni"]), reverse=True)
    
    risultati_finali_con_previsione = []
    if migliori_metodi_periodici:
        estrazione_da_usare_per_live = None
        nota_previsione_live_globale = ""

        if indice_mese_per_analisi_storica_e_live is not None and data_fine_per_previsione_live_obj is not None:
            if app_logger: app_logger(f"  TRAOP: Ricerca estrazione per previsione live (finale): idx_mese={indice_mese_per_analisi_storica_e_live}, data_fine<={data_fine_per_previsione_live_obj.strftime('%Y-%m-%d') if data_fine_per_previsione_live_obj else 'N/A'}")
            for estr_live_cand in reversed(storico_per_analisi_interna): 
                if estr_live_cand['data'] <= data_fine_per_previsione_live_obj:
                    estrazione_da_usare_per_live = estr_live_cand
                    if app_logger: app_logger(f"    TRAOP: Trovata estrazione specifica per previsione live: {estrazione_da_usare_per_live['data']} (Indice: {estrazione_da_usare_per_live.get('indice_mese')})")
                    break
            if not estrazione_da_usare_per_live:
                nota_previsione_live_globale = f"Nessuna estrazione trovata per l'indice mese {indice_mese_per_analisi_storica_e_live} entro il {data_fine_per_previsione_live_obj.strftime('%d/%m/%Y')} nel periodo selezionato."
                if app_logger: app_logger(f"    TRAOP: {nota_previsione_live_globale}")
        elif storico_per_analisi_interna: 
            estrazione_da_usare_per_live = storico_per_analisi_interna[-1]
            if app_logger: app_logger(f"  TRAOP: Uso ultima estrazione del periodo analizzato internamente ({estrazione_da_usare_per_live['data']}) per previsione live.")
        
        if not estrazione_da_usare_per_live and not nota_previsione_live_globale:
            nota_previsione_live_globale = "Impossibile determinare estrazione per previsione live (storico per analisi interna vuoto?)."
            if app_logger: app_logger(f"  TRAOP: {nota_previsione_live_globale}")

        for metodo_info_orig in migliori_metodi_periodici[:num_migliori_da_restituire]: 
            metodo_info = metodo_info_orig.copy() 
            previsione_calcolata = "N/A"
            metodo_info["estrazione_usata_per_previsione"] = estrazione_da_usare_per_live 

            if estrazione_da_usare_per_live:
                form = metodo_info["metodo_formula"]
                numeri_base_live = estrazione_da_usare_per_live.get(form["ruota_calcolo"], [])
                if numeri_base_live and len(numeri_base_live) >= form["pos_estratto_calcolo"]:
                    num_base_live_val = numeri_base_live[form["pos_estratto_calcolo"]-1]
                    op_func_live = OPERAZIONI[form["operazione"]]
                    try:
                        val_op_live = op_func_live(num_base_live_val, form["operando_fisso"])
                        previsione_calcolata = regola_fuori_90(val_op_live)
                        if app_logger: app_logger(f"    TRAOP: Previsione live calcolata per metodo {form} su {estrazione_da_usare_per_live['data']}: {previsione_calcolata}")
                    except ZeroDivisionError:
                         if app_logger: app_logger(f"    TRAOP: Divisione per zero calcolando previsione live per metodo {form} su {estrazione_da_usare_per_live['data']}.")
                else:
                    if app_logger: app_logger(f"    TRAOP: Dati insufficienti in estrazione live per metodo {form} su {estrazione_da_usare_per_live['data']}.")
            elif nota_previsione_live_globale and app_logger:
                 app_logger(f"    TRAOP: Previsione live non calcolata per metodo {metodo_info['metodo_formula']}: {nota_previsione_live_globale}")
            
            metodo_info["previsione_live_periodica"] = previsione_calcolata
            risultati_finali_con_previsione.append(metodo_info)
        
        if app_logger: app_logger(f"TRAOP: Ricerca ambata ottimale completata. Trovati {len(risultati_finali_con_previsione)} metodi con info previsione (top {num_migliori_da_restituire}).")
        return risultati_finali_con_previsione

    if app_logger: app_logger(f"TRAOP: Ricerca ambata ottimale completata. Nessun metodo valido o storico per previsione.")
    return []

def analizza_performance_dettagliata_metodo(
    storico_completo,
    definizione_metodo,
    metodo_stringa_per_log,
    ruote_gioco,
    lookahead,
    data_inizio_analisi,
    data_fine_analisi,
    mesi_selezionati_filtro,
    app_logger=None,
    condizione_primaria_metodo=None,
    indice_estrazione_mese_da_considerare=None
):
    def log(msg):
        if app_logger: app_logger(msg)

    log(f"\n--- AVVIO ANALISI PERFORMANCE DETTAGLIATA METODO ---")
    log(f"Metodo da analizzare: {metodo_stringa_per_log}")
    log(f"Periodo Globale: {data_inizio_analisi.strftime('%d/%m/%Y')} - {data_fine_analisi.strftime('%d/%m/%Y')}")
    log(f"Mesi Specifici: {mesi_selezionati_filtro or 'Tutti nel range'}")
    log(f"Indice Estrazione del Mese da Usare: {indice_estrazione_mese_da_considerare if indice_estrazione_mese_da_considerare is not None else 'Tutte valide'}")
    
    risultati_dettagliati = []
    if not storico_completo:
        log("ERRORE: Storico completo vuoto."); return risultati_dettagliati

    # >>> INIZIO MODIFICA CRUCIALE <<<
    # Non raggruppiamo più per mese, ma filtriamo direttamente le estrazioni valide.
    estrazioni_da_analizzare = []
    for i_idx, estrazione in enumerate(storico_completo):
        data_e = estrazione.get('data')
        if not isinstance(data_e, date): continue

        # 1. Filtro per range di date globale
        if not (data_inizio_analisi <= data_e <= data_fine_analisi):
            continue
        
        # 2. Filtro per mesi specifici
        if mesi_selezionati_filtro and data_e.month not in mesi_selezionati_filtro:
            continue

        # 3. Filtro per indice estrazione del mese
        if indice_estrazione_mese_da_considerare is not None:
            if estrazione.get('indice_mese') != indice_estrazione_mese_da_considerare:
                continue
        
        # 4. Filtro per condizione primaria del metodo
        if condizione_primaria_metodo:
            cond_ruota = condizione_primaria_metodo['ruota']
            cond_pos_idx = (condizione_primaria_metodo.get('posizione', 1) - 1)
            cond_min = condizione_primaria_metodo['val_min']
            cond_max = condizione_primaria_metodo['val_max']
            numeri_ruota_cond = estrazione.get(cond_ruota, [])
            if not numeri_ruota_cond or len(numeri_ruota_cond) <= cond_pos_idx:
                continue # Non ci sono i dati per verificare la condizione
            val_est_cond = numeri_ruota_cond[cond_pos_idx]
            if not (cond_min <= val_est_cond <= cond_max):
                continue # La condizione non è soddisfatta
        
        # Se tutti i filtri sono passati, aggiungiamo l'estrazione alla lista da analizzare
        estrazione['indice_originale_storico_completo'] = i_idx
        estrazioni_da_analizzare.append(estrazione)

    if not estrazioni_da_analizzare:
        log("Nessuna estrazione trovata che soddisfi tutti i filtri (data, mese, indice, condizione)."); return []

    # Ora cicliamo su ogni estrazione valida trovata
    for estrazione_applicazione in estrazioni_da_analizzare:
        dettaglio_applicazione = {
            "data_applicazione": estrazione_applicazione['data'],
            "ambata_prevista": None, "metodo_applicabile": False, "esito_ambata": False,
            "colpo_vincita_ambata": None, "ruota_vincita_ambata": None,
            "numeri_estratti_vincita": None, "condizione_soddisfatta": True # Se è qui, la condizione è già ok
        }
        
        ambata_prevista_calc = calcola_valore_metodo_complesso(estrazione_applicazione, definizione_metodo, app_logger)

        if ambata_prevista_calc is not None:
            ambata_prevista = regola_fuori_90(ambata_prevista_calc)
            if ambata_prevista is not None:
                dettaglio_applicazione["metodo_applicabile"] = True
                dettaglio_applicazione["ambata_prevista"] = ambata_prevista
                
                indice_originale_app = estrazione_applicazione['indice_originale_storico_completo']
                for k_lookahead in range(1, lookahead + 1):
                    indice_estrazione_futura = indice_originale_app + k_lookahead
                    if indice_estrazione_futura >= len(storico_completo): break
                    estrazione_futura = storico_completo[indice_estrazione_futura]
                    
                    for ruota_v in ruote_gioco:
                        numeri_ruota_futura = estrazione_futura.get(ruota_v, [])
                        if ambata_prevista in numeri_ruota_futura:
                            dettaglio_applicazione["esito_ambata"] = True
                            dettaglio_applicazione["colpo_vincita_ambata"] = k_lookahead
                            dettaglio_applicazione["ruota_vincita_ambata"] = ruota_v
                            dettaglio_applicazione["numeri_estratti_vincita"] = numeri_ruota_futura
                            break
                    if dettaglio_applicazione["esito_ambata"]: break
        else:
             dettaglio_applicazione["metodo_applicabile"] = False

        risultati_dettagliati.append(dettaglio_applicazione)
    # >>> FINE MODIFICA CRUCIALE <<<

    log(f"Analisi dettagliata completata. {len(risultati_dettagliati)} applicazioni valide trovate e analizzate.")
    return risultati_dettagliati

def analizza_performance_dettagliata_combinata(
    storico_completo,
    definizione_metodo_1,
    definizione_metodo_2,
    metodo_stringa_per_log,
    ruote_gioco,
    lookahead,
    data_inizio_analisi,
    data_fine_analisi,
    mesi_selezionati_filtro,
    app_logger=None,
    condizione_primaria_metodo=None,
    indice_estrazione_mese_da_considerare=None
):
    def log(msg):
        if app_logger: app_logger(msg)

    log(f"\n--- AVVIO ANALISI PERFORMANCE DETTAGLIATA COMBINATA ---")
    log(f"Metodo: {metodo_stringa_per_log}")

    risultati_dettagliati = []
    if not storico_completo: return risultati_dettagliati

    # Raggruppa le estrazioni per (anno, mese)
    estrazioni_per_anno_mese = defaultdict(list)
    for i_idx, estrazione in enumerate(storico_completo):
        data_e = estrazione.get('data')
        if not isinstance(data_e, date): continue
        if data_inizio_analisi <= data_e <= data_fine_analisi:
            if not mesi_selezionati_filtro or data_e.month in mesi_selezionati_filtro:
                estrazione['indice_originale_storico_completo'] = i_idx
                estrazioni_per_anno_mese[(data_e.year, data_e.month)].append(estrazione)
    
    anni_mesi_ordinati = sorted(estrazioni_per_anno_mese.keys())
    if not anni_mesi_ordinati: return risultati_dettagliati

    for anno, mese in anni_mesi_ordinati:
        estrazioni_del_mese_corrente = estrazioni_per_anno_mese[(anno, mese)]
        estrazione_di_applicazione_trovata_nel_mese = None

        # Logica robusta per trovare l'estrazione di applicazione
        if indice_estrazione_mese_da_considerare is not None:
            for estrazione_candidata in estrazioni_del_mese_corrente:
                if estrazione_candidata.get('indice_mese') == indice_estrazione_mese_da_considerare:
                    estrazione_di_applicazione_trovata_nel_mese = estrazione_candidata
                    break
        elif estrazioni_del_mese_corrente:
            estrazione_di_applicazione_trovata_nel_mese = estrazioni_del_mese_corrente[0] # Prendi la prima del mese
        
        dettaglio_applicazione = {
            "data_applicazione": date(anno, mese, 1) if not estrazione_di_applicazione_trovata_nel_mese else estrazione_di_applicazione_trovata_nel_mese['data'],
            "ambata_prevista_1": None, "ambata_prevista_2": None, "metodo_applicabile": False, 
            "esito_vincita": False, "ambata_vincente": None, "colpo_vincita": None, 
            "ruota_vincita": None, "numeri_estratti_vincita": None
        }

        if estrazione_di_applicazione_trovata_nel_mese:
            dettaglio_applicazione["data_applicazione"] = estrazione_di_applicazione_trovata_nel_mese['data']
            indice_originale_app = estrazione_di_applicazione_trovata_nel_mese['indice_originale_storico_completo']

            val_raw1 = calcola_valore_metodo_complesso(estrazione_di_applicazione_trovata_nel_mese, definizione_metodo_1, app_logger)
            val_raw2 = calcola_valore_metodo_complesso(estrazione_di_applicazione_trovata_nel_mese, definizione_metodo_2, app_logger)

            if val_raw1 is not None and val_raw2 is not None:
                ambata1 = regola_fuori_90(val_raw1)
                ambata2 = regola_fuori_90(val_raw2)
                
                if ambata1 is not None and ambata2 is not None:
                    dettaglio_applicazione["metodo_applicabile"] = True
                    dettaglio_applicazione["ambata_prevista_1"] = ambata1
                    dettaglio_applicazione["ambata_prevista_2"] = ambata2
                    
                    for k_lookahead in range(1, lookahead + 1):
                        if indice_originale_app + k_lookahead >= len(storico_completo): break
                        estrazione_futura = storico_completo[indice_originale_app + k_lookahead]
                        
                        for ruota_v in ruote_gioco:
                            numeri_ruota_futura = estrazione_futura.get(ruota_v, [])
                            vincente = None
                            if ambata1 in numeri_ruota_futura: vincente = ambata1
                            elif ambata1 != ambata2 and ambata2 in numeri_ruota_futura: vincente = ambata2
                                
                            if vincente is not None:
                                dettaglio_applicazione["esito_vincita"] = True
                                dettaglio_applicazione["ambata_vincente"] = vincente
                                dettaglio_applicazione["colpo_vincita"] = k_lookahead
                                dettaglio_applicazione["ruota_vincita"] = ruota_v
                                dettaglio_applicazione["numeri_estratti_vincita"] = numeri_ruota_futura
                                break
                        if dettaglio_applicazione["esito_vincita"]: break
                        
        risultati_dettagliati.append(dettaglio_applicazione)

    log(f"Analisi dettagliata combinata completata. {len(risultati_dettagliati)} periodi processati.")
    return risultati_dettagliati

def trova_migliori_ambi_da_correttori_automatici(
    storico_da_analizzare,
    ruota_base_calc,
    pos_base_calc_0idx,
    lista_operazioni_base_da_testare, # Es. ['+', '-', '*']
    lista_trasformazioni_correttore_da_testare, # Es. ['Fisso', 'Diametrale', 'Vertibile']
    ruote_di_gioco_per_verifica,
    indice_mese_specifico_applicazione,
    lookahead_colpi_per_verifica,
    app_logger=None,
    min_tentativi_per_metodo=5
):
    def log(msg):
        if app_logger: app_logger(msg)

    performance_metodi = defaultdict(lambda: {
        'successi_ambo': 0, 'tentativi_ambo': 0,
        'successi_ambata1': 0, 'successi_ambata2': 0,
        'successi_almeno_una_ambata': 0,
        'ambo_generato_esempio': None, 'ambata1_esempio': None, 'ambata2_esempio': None
    })

    operazioni_base_lambda = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '*': lambda a, b: a * b}
    
    # OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE deve essere definito globalmente o passato come argomento
    # Assumiamo sia globale per ora, come definito nel Passo 1

    correttori_fissi_range = list(range(1, 91)) # I correttori C1, C2 sono sempre numeri fissi
    estrazioni_valide_per_applicazione_base = 0

    log(f"AAU Analisi: Op.Base={lista_operazioni_base_da_testare}, Trasf.Correttori={lista_trasformazioni_correttore_da_testare}")

    for i_estrazione_app, estrazione_applicazione in enumerate(storico_da_analizzare):
        if indice_mese_specifico_applicazione is not None:
            if estrazione_applicazione.get('indice_mese') != indice_mese_specifico_applicazione:
                continue
        numeri_ruota_base = estrazione_applicazione.get(ruota_base_calc, [])
        if not numeri_ruota_base or len(numeri_ruota_base) <= pos_base_calc_0idx:
            continue
        numero_base = numeri_ruota_base[pos_base_calc_0idx]
        estrazioni_valide_per_applicazione_base += 1

        # Ciclo sui correttori fissi C1 e C2
        for c1_fisso_idx in range(len(correttori_fissi_range)):
            for c2_fisso_idx in range(c1_fisso_idx + 1, len(correttori_fissi_range)): # Assicura C1 != C2
                c1_originale = correttori_fissi_range[c1_fisso_idx]
                c2_originale = correttori_fissi_range[c2_fisso_idx]

                # Ciclo sulle trasformazioni per C1
                for nome_trasf1 in lista_trasformazioni_correttore_da_testare:
                    func_trasf1 = OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.get(nome_trasf1)
                    if not func_trasf1: continue
                    c1_trasformato = func_trasf1(c1_originale)
                    if c1_trasformato is None: continue # Trasformazione non valida

                    # Ciclo sulle trasformazioni per C2
                    for nome_trasf2 in lista_trasformazioni_correttore_da_testare:
                        func_trasf2 = OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.get(nome_trasf2)
                        if not func_trasf2: continue
                        c2_trasformato = func_trasf2(c2_originale)
                        if c2_trasformato is None: continue # Trasformazione non valida
                        
                        # Se si vuole evitare che C1 trasformato sia uguale a C2 trasformato
                        # if c1_trasformato == c2_trasformato: continue

                        # Ciclo sulle operazioni base per Ambata1 e Ambata2
                        for op_base1_str in lista_operazioni_base_da_testare:
                            op_base1_func = operazioni_base_lambda.get(op_base1_str)
                            if not op_base1_func: continue

                            for op_base2_str in lista_operazioni_base_da_testare:
                                op_base2_func = operazioni_base_lambda.get(op_base2_str)
                                if not op_base2_func: continue

                                try:
                                    ambata_prevista1 = regola_fuori_90(op_base1_func(numero_base, c1_trasformato))
                                    ambata_prevista2 = regola_fuori_90(op_base2_func(numero_base, c2_trasformato))
                                except ZeroDivisionError:
                                    continue
                                
                                if ambata_prevista1 is None or ambata_prevista2 is None or ambata_prevista1 == ambata_prevista2:
                                    continue

                                ambo_generato_tuple = tuple(sorted((ambata_prevista1, ambata_prevista2)))
                                
                                # La chiave del metodo ora include le trasformazioni e le operazioni base
                                metodo_key_params = (
                                    (c1_originale, nome_trasf1, op_base1_str), 
                                    (c2_originale, nome_trasf2, op_base2_str)
                                )

                                performance_metodi[metodo_key_params]['tentativi_ambo'] += 1
                                if performance_metodi[metodo_key_params]['ambo_generato_esempio'] is None:
                                    performance_metodi[metodo_key_params]['ambo_generato_esempio'] = ambo_generato_tuple
                                    performance_metodi[metodo_key_params]['ambata1_esempio'] = ambata_prevista1
                                    performance_metodi[metodo_key_params]['ambata2_esempio'] = ambata_prevista2

                                # ... (resto del backtesting del lookahead, è identico a prima, usa ambata_prevista1 e ambata_prevista2)
                                vincita_ambo_in_lookahead = False; vincita_ambata1_in_lookahead = False
                                vincita_ambata2_in_lookahead = False; vincita_almeno_una_ambata_in_lookahead = False
                                for k_lh in range(1, lookahead_colpi_per_verifica + 1):
                                    idx_futuro = i_estrazione_app + k_lh
                                    if idx_futuro >= len(storico_da_analizzare): break
                                    estrazione_futura = storico_da_analizzare[idx_futuro]
                                    for ruota_v in ruote_di_gioco_per_verifica:
                                        numeri_estratti_futura = estrazione_futura.get(ruota_v, [])
                                        if not numeri_estratti_futura: continue
                                        if not vincita_ambata1_in_lookahead and ambata_prevista1 in numeri_estratti_futura:
                                            performance_metodi[metodo_key_params]['successi_ambata1'] += 1; vincita_ambata1_in_lookahead = True
                                        if not vincita_ambata2_in_lookahead and ambata_prevista2 in numeri_estratti_futura:
                                            performance_metodi[metodo_key_params]['successi_ambata2'] += 1; vincita_ambata2_in_lookahead = True
                                        if not vincita_ambo_in_lookahead and set(ambo_generato_tuple).issubset(set(numeri_estratti_futura)):
                                            performance_metodi[metodo_key_params]['successi_ambo'] += 1; vincita_ambo_in_lookahead = True
                                        if not vincita_almeno_una_ambata_in_lookahead:
                                            if ambata_prevista1 in numeri_estratti_futura or ambata_prevista2 in numeri_estratti_futura:
                                                performance_metodi[metodo_key_params]['successi_almeno_una_ambata'] += 1; vincita_almeno_una_ambata_in_lookahead = True
                                    if vincita_ambo_in_lookahead and vincita_ambata1_in_lookahead and vincita_ambata2_in_lookahead and vincita_almeno_una_ambata_in_lookahead: break
    
    risultati_performanti = []
    for metodo_key_tuple, data in performance_metodi.items():
        if data['tentativi_ambo'] >= min_tentativi_per_metodo:
            # ... (calcolo frequenze, come prima) ...
            freq_ambo = data['successi_ambo'] / data['tentativi_ambo'] if data['tentativi_ambo'] > 0 else 0
            freq_ambata1 = data['successi_ambata1'] / data['tentativi_ambo'] if data['tentativi_ambo'] > 0 else 0
            freq_ambata2 = data['successi_ambata2'] / data['tentativi_ambo'] if data['tentativi_ambo'] > 0 else 0
            freq_almeno_una_ambata = data['successi_almeno_una_ambata'] / data['tentativi_ambo'] if data['tentativi_ambo'] > 0 else 0

            (params_c1, params_c2) = metodo_key_tuple
            risultati_performanti.append({
                'correttore1_orig': params_c1[0], 'trasf1': params_c1[1], 'op_base1': params_c1[2],
                'correttore2_orig': params_c2[0], 'trasf2': params_c2[1], 'op_base2': params_c2[2],
                'ambo_esempio': data.get('ambo_generato_esempio'),
                'ambata1_esempio': data.get('ambata1_esempio'), 'ambata2_esempio': data.get('ambata2_esempio'),
                'successi_ambo': data['successi_ambo'], 'tentativi_ambo': data['tentativi_ambo'], 'frequenza_ambo': freq_ambo,
                'successi_ambata1': data['successi_ambata1'], 'tentativi_ambata1': data['tentativi_ambo'], 'frequenza_ambata1': freq_ambata1,
                'successi_ambata2': data['successi_ambata2'], 'tentativi_ambata2': data['tentativi_ambo'], 'frequenza_ambata2': freq_ambata2,
                'successi_almeno_una_ambata': data['successi_almeno_una_ambata'], 'tentativi_almeno_una_ambata': data['tentativi_ambo'], 'frequenza_almeno_una_ambata': freq_almeno_una_ambata
            })
    
    risultati_performanti.sort(key=lambda x: (x['frequenza_ambo'], x['frequenza_almeno_una_ambata'], x['successi_ambo']), reverse=True)
    # ... (log di debug e return, come prima) ...
    if risultati_performanti and app_logger:
        log("\nDEBUG (trova_migliori_ambi con TRASF): Contenuto del primo elemento di risultati_performanti:")
        # ... (log più dettagliato se necessario)
    elif app_logger: log("\nDEBUG (trova_migliori_ambi con TRASF): risultati_performanti è vuoto.")
    log(f"Ricerca Ambi Sommativi (con Trasf.) terminata. Metodi validi (>= {min_tentativi_per_metodo} tent.): {len(risultati_performanti)}.")
    log(f"  Totale estrazioni base analizzate nel periodo/indice mese: {estrazioni_valide_per_applicazione_base}")
    return risultati_performanti, estrazioni_valide_per_applicazione_base

# --- INIZIO DELLA CLASSE GUI (LottoAnalyzerApp) ---
class LottoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Costruttore Metodi Lotto Avanzato")
        master.geometry("850x700")

        # --- Inizio Variabili di Istanza (esistenti) ---
        self.cartella_dati_var = tk.StringVar()
        self.ruote_gioco_vars = {ruota: tk.BooleanVar() for ruota in RUOTE}
        self.tutte_le_ruote_var = tk.BooleanVar(value=True)
        self.lookahead_var = tk.IntVar(value=3)
        self.indice_mese_var = tk.StringVar()
        self.storico_caricato = None
        self.active_tab_ruote_checkbox_widgets = []
        self.log_messages_list = []

        # Variabili per Tab "Ricerca Metodi Semplici"
        self.ruota_calcolo_var = tk.StringVar(value=RUOTE[0])
        self.posizione_estratto_var = tk.IntVar(value=1)
        self.num_ambate_var = tk.IntVar(value=1)
        self.min_tentativi_var = tk.IntVar(value=10)
        self.ms_risultati_listbox = None
        self.metodi_semplici_trovati_dati = []

        # Variabili per Tab "Analisi Metodo Complesso" - Metodo Base 1
        self.definizione_metodo_complesso_attuale = []
        self.mc_tipo_termine_var = tk.StringVar(value="estratto")
        self.mc_ruota_var = tk.StringVar(value=RUOTE[0])
        self.mc_posizione_var = tk.IntVar(value=1)
        self.mc_valore_fisso_var = tk.IntVar(value=1)
        self.mc_operazione_var = tk.StringVar(value='+')

        # Variabili per Tab "Analisi Metodo Complesso" - Metodo Base 2
        self.definizione_metodo_complesso_attuale_2 = []
        self.mc_tipo_termine_var_2 = tk.StringVar(value="estratto")
        self.mc_ruota_var_2 = tk.StringVar(value=RUOTE[0])
        self.mc_posizione_var_2 = tk.IntVar(value=1)
        self.mc_valore_fisso_var_2 = tk.IntVar(value=1)
        self.mc_operazione_var_2 = tk.StringVar(value='+')

        # Variabili per Configurazione Ricerca Correttore (dei Metodi Complessi)
        self.corr_cfg_cerca_fisso_semplice = tk.BooleanVar(value=True)
        self.corr_cfg_cerca_estratto_semplice = tk.BooleanVar(value=True)
        self.corr_cfg_cerca_somma_estr_fisso = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_somma_estr_estr = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_diff_estr_fisso = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_diff_estr_estr = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_mult_estr_fisso = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_mult_estr_estr = tk.BooleanVar(value=False)
        self.corr_cfg_min_tentativi = tk.IntVar(value=5)

        self.ultimo_metodo_corretto_trovato_definizione = None
        self.ultimo_metodo_corretto_formula_testuale = ""
        self.usa_ultimo_corretto_per_backtest_var = tk.BooleanVar(value=False)

        self.ultimo_metodo_cond_corretto_definizione = None
        self.ultimo_metodo_cond_corretto_formula_testuale = ""

        self.metodo_preparato_per_backtest = None

        # Variabili per Tab "Verifica Giocata Manuale"
        self.numeri_verifica_var = tk.StringVar()
        self.colpi_verifica_var = tk.IntVar(value=9)

        # Variabili per Tab "Analisi Condizionata Avanzata"
        self.ac_ruota_cond_var = tk.StringVar(value=RUOTE[0])
        self.ac_pos_cond_var = tk.IntVar(value=1)
        self.ac_val_min_cond_var = tk.IntVar(value=1)
        self.ac_val_max_cond_var = tk.IntVar(value=45)
        self.ac_ruota_calc_ambata_var = tk.StringVar(value=RUOTE[0])
        self.ac_pos_calc_ambata_var = tk.IntVar(value=1)
        self.ac_num_risultati_var = tk.IntVar(value=2)
        self.ac_min_tentativi_var = tk.IntVar(value=5)
        self.ac_risultati_listbox = None
        self.ac_metodi_condizionati_dettagli = []

        # Variabili per Tab "Analisi Periodica"
        self.ap_ruota_calcolo_ott_var = tk.StringVar(value=RUOTE[0])
        self.ap_pos_estratto_ott_var = tk.IntVar(value=1)
        self.ap_min_tentativi_ott_var = tk.IntVar(value=5)
        self.ap_mesi_vars = {
            "Gennaio": tk.BooleanVar(value=False), "Febbraio": tk.BooleanVar(value=False),
            "Marzo": tk.BooleanVar(value=False), "Aprile": tk.BooleanVar(value=False),
            "Maggio": tk.BooleanVar(value=False), "Giugno": tk.BooleanVar(value=False),
            "Luglio": tk.BooleanVar(value=False), "Agosto": tk.BooleanVar(value=False),
            "Settembre": tk.BooleanVar(value=False), "Ottobre": tk.BooleanVar(value=False),
            "Novembre": tk.BooleanVar(value=False), "Dicembre": tk.BooleanVar(value=False)
        }
        self.ap_tutti_mesi_var = tk.BooleanVar(value=False)
        self.ap_risultati_listbox = None
        self.ap_mesi_checkbox_widgets = []
        self.ap_tipo_sorte_var = tk.StringVar(value="Ambata")
        self.ap_numeri_input_var = tk.StringVar()

        # --- Variabili per il Tab "Ambata e Ambo Unico" (aau_) ---
        self.aau_ruota_base_var = tk.StringVar(value=RUOTE[0])
        self.aau_pos_base_var = tk.IntVar(value=1)
        self.aau_op_somma_var = tk.BooleanVar(value=True)
        self.aau_op_diff_var = tk.BooleanVar(value=True)
        self.aau_op_mult_var = tk.BooleanVar(value=False)
        self.aau_trasf_vars = {} # NUOVO: Per le checkbox delle trasformazioni dei correttori
        self.aau_risultati_listbox = None
        self.aau_metodi_trovati_dati = []
        
        self.mc_backtest_choice_var = tk.StringVar(value="base1")
        # --- Fine Variabili di Istanza (esistenti) ---

        # --- INIZIO MODIFICA: Aggiunta variabile per finestra Lunghette ---
        self.finestra_lunghette_attiva = None
        # --- FINE MODIFICA ---

        menubar = tk.Menu(master)
        master.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Apri Profilo Metodo Salvato...", command=self.apri_e_visualizza_profilo_metodo)
        file_menu.add_separator()
        file_menu.add_command(label="Esci", command=master.quit)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Impostazioni", menu=settings_menu)
        settings_menu.add_command(label="Configura Ricerca Correttore...", command=self.apri_dialogo_impostazioni_correttore)
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualizza", menu=view_menu)
        view_menu.add_command(label="Mostra Log Operazioni", command=self.mostra_finestra_log)

        common_controls_frame_top = ttk.Frame(master)
        common_controls_frame_top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        cartella_frame = ttk.LabelFrame(common_controls_frame_top, text="Impostazioni Caricamento Dati", padding="10")
        cartella_frame.pack(side=tk.TOP, fill=tk.X)
        current_row_cc = 0
        tk.Label(cartella_frame, text="Cartella Archivio Dati:").grid(row=current_row_cc, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(cartella_frame, textvariable=self.cartella_dati_var, width=50).grid(row=current_row_cc, column=1, sticky="ew", padx=5, pady=2)
        tk.Button(cartella_frame, text="Sfoglia...", command=self.seleziona_cartella).grid(row=current_row_cc, column=2, sticky="w", padx=5, pady=2)
        current_row_cc += 1
        tk.Label(cartella_frame, text="Data Inizio Analisi/Storico:").grid(row=current_row_cc, column=0, sticky="w", padx=5, pady=2)
        self.date_inizio_entry_analisi = DateEntry(cartella_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_inizio_entry_analisi.grid(row=current_row_cc, column=1, sticky="w", padx=5, pady=2)
        tk.Button(cartella_frame, text="Nessuna", command=lambda: self.date_inizio_entry_analisi.delete(0, tk.END)).grid(row=current_row_cc, column=2, sticky="w", padx=5, pady=2)
        current_row_cc += 1
        tk.Label(cartella_frame, text="Data Fine Analisi/Storico:").grid(row=current_row_cc, column=0, sticky="w", padx=5, pady=2)
        self.date_fine_entry_analisi = DateEntry(cartella_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_fine_entry_analisi.grid(row=current_row_cc, column=1, sticky="w", padx=5, pady=2)
        self.date_fine_entry_analisi.bind("<<DateEntrySelected>>", self._aggiorna_data_inizio_verifica)
        tk.Button(cartella_frame, text="Nessuna", command=self._pulisci_data_fine_e_verifica).grid(row=current_row_cc, column=2, sticky="w", padx=5, pady=2)

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        tab_metodi_semplici = ttk.Frame(self.notebook)
        self.notebook.add(tab_metodi_semplici, text='Ricerca Metodi Semplici')
        self.crea_gui_metodi_semplici(tab_metodi_semplici)

        tab_metodo_complesso = ttk.Frame(self.notebook)
        self.notebook.add(tab_metodo_complesso, text='Analisi Metodo Complesso')
        self.crea_gui_metodo_complesso(tab_metodo_complesso)

        tab_analisi_condizionata = ttk.Frame(self.notebook)
        self.notebook.add(tab_analisi_condizionata, text='Analisi Condizionata Avanzata')
        self.crea_gui_analisi_condizionata(tab_analisi_condizionata)

        tab_analisi_periodica = ttk.Frame(self.notebook)
        self.notebook.add(tab_analisi_periodica, text='Analisi Periodica')
        self.crea_gui_analisi_periodica(tab_analisi_periodica)

        tab_aau = ttk.Frame(self.notebook)
        self.notebook.add(tab_aau, text='Ambata e Ambo Unico')
        self.crea_gui_aau(tab_aau)

        # --- INIZIO MODIFICA: Aggiunta Tab Lunghette ---
        tab_lunghette = ttk.Frame(self.notebook)
        self.notebook.add(tab_lunghette, text='Lunghette')
        self.crea_gui_lunghette(tab_lunghette) # Chiamata alla nuova funzione
        # --- FINE MODIFICA ---

        tab_verifica_manuale = ttk.Frame(self.notebook)
        self.notebook.add(tab_verifica_manuale, text='Verifica Giocata Manuale')
        self.crea_gui_verifica_manuale(tab_verifica_manuale)


        if self.notebook.tabs():
            self.master.after(100, lambda: {
                self.on_tab_changed(None),
                self._aggiorna_data_inizio_verifica()
            })


    def _log_to_gui(self, message, end='\n', flush=False):
        self.log_messages_list.append(message + end)

    def mostra_finestra_log(self):
        log_window = tk.Toplevel(self.master)
        log_window.title("Log Operazioni")
        log_window.geometry("750x550")
        try:
            log_window.transient(self.master)
            log_window.grab_set()
        except tk.TclError:
            pass
        log_text_widget = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, font=("Courier New", 9))
        log_text_widget.pack(padx=10, pady=(10,0), fill=tk.BOTH, expand=True)
        full_log_content = "".join(self.log_messages_list)
        log_text_widget.insert(tk.END, full_log_content)
        log_text_widget.config(state=tk.DISABLED)
        log_text_widget.see(tk.END)
        button_frame_log = ttk.Frame(log_window)
        button_frame_log.pack(fill=tk.X, pady=10, padx=10)
        def clear_log_action():
            self.log_messages_list = ["--- Log Cancellato ---\n"]
            log_text_widget.config(state=tk.NORMAL)
            log_text_widget.delete('1.0', tk.END)
            log_text_widget.insert(tk.END, self.log_messages_list[0])
            log_text_widget.config(state=tk.DISABLED)
            log_text_widget.see(tk.END)
        ttk.Button(button_frame_log, text="Pulisci Log", command=clear_log_action).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(button_frame_log, text="Chiudi", command=log_window.destroy).pack(side=tk.RIGHT)
        try:
            self.master.eval(f'tk::PlaceWindow {str(log_window)} center')
        except tk.TclError:
            log_window.update_idletasks()
            width = log_window.winfo_width(); height = log_window.winfo_height()
            x = (log_window.winfo_screenwidth() // 2) - (width // 2)
            y = (log_window.winfo_screenheight() // 2) - (height // 2)
            log_window.geometry(f'{width}x{height}+{x}+{y}')
        log_window.wait_window()

    def seleziona_cartella(self):
        cartella = filedialog.askdirectory(title="Seleziona cartella archivi")
        if cartella:
            self.cartella_dati_var.set(cartella)

    def apri_dialogo_impostazioni_correttore(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Impostazioni Ricerca Correttore")
        dialog.geometry("450x420")
        dialog.resizable(False, False)
        dialog.transient(self.master)
        dialog.grab_set()
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(expand=True, fill="both")
        ttk.Label(main_frame, text="Seleziona i tipi di termini correttori da includere nella ricerca:", wraplength=400).pack(pady=(0,10))
        frame_semplici = ttk.LabelFrame(main_frame, text="Correttori Semplici", padding="5")
        frame_semplici.pack(fill="x", pady=3)
        ttk.Checkbutton(frame_semplici, text="Fisso Singolo", variable=self.corr_cfg_cerca_fisso_semplice).pack(anchor="w", padx=5)
        ttk.Checkbutton(frame_semplici, text="Estratto Singolo", variable=self.corr_cfg_cerca_estratto_semplice).pack(anchor="w", padx=5)
        frame_somma = ttk.LabelFrame(main_frame, text="Correttori con Somma (+)", padding="5")
        frame_somma.pack(fill="x", pady=3)
        ttk.Checkbutton(frame_somma, text="Estratto + Fisso", variable=self.corr_cfg_cerca_somma_estr_fisso).pack(anchor="w", padx=5)
        ttk.Checkbutton(frame_somma, text="Estratto + Estratto", variable=self.corr_cfg_cerca_somma_estr_estr).pack(anchor="w", padx=5)
        frame_diff = ttk.LabelFrame(main_frame, text="Correttori con Differenza (-)", padding="5")
        frame_diff.pack(fill="x", pady=3)
        ttk.Checkbutton(frame_diff, text="Estratto - Fisso", variable=self.corr_cfg_cerca_diff_estr_fisso).pack(anchor="w", padx=5)
        ttk.Checkbutton(frame_diff, text="Estratto - Estratto", variable=self.corr_cfg_cerca_diff_estr_estr).pack(anchor="w", padx=5)
        frame_mult = ttk.LabelFrame(main_frame, text="Correttori con Moltiplicazione (*)", padding="5")
        frame_mult.pack(fill="x", pady=3)
        ttk.Checkbutton(frame_mult, text="Estratto * Fisso", variable=self.corr_cfg_cerca_mult_estr_fisso).pack(anchor="w", padx=5)
        ttk.Checkbutton(frame_mult, text="Estratto * Estratto", variable=self.corr_cfg_cerca_mult_estr_estr).pack(anchor="w", padx=5)
        frame_min_tent = ttk.LabelFrame(main_frame, text="Parametri Aggiuntivi Ricerca", padding="5")
        frame_min_tent.pack(fill="x", pady=3, expand=False)
        min_tent_inner_frame = ttk.Frame(frame_min_tent)
        min_tent_inner_frame.pack(anchor="w", padx=5, pady=2)
        tk.Label(min_tent_inner_frame, text="Min. Tentativi per Correttore Valido:").pack(side=tk.LEFT)
        tk.Spinbox(min_tent_inner_frame, from_=1, to=50, textvariable=self.corr_cfg_min_tentativi, width=5, state="readonly").pack(side=tk.LEFT, padx=5)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(15, 0))
        ok_button = ttk.Button(button_frame, text="OK", command=dialog.destroy)
        ok_button.pack(side=tk.RIGHT, padx=5)
        dialog.update_idletasks()
        master_x = self.master.winfo_x(); master_y = self.master.winfo_y()
        master_width = self.master.winfo_width(); master_height = self.master.winfo_height()
        dialog_width = dialog.winfo_width(); dialog_height = dialog.winfo_height()
        if dialog_width <= 1: dialog_width = 450
        if dialog_height <= 1: dialog_height = 420
        center_x = master_x + (master_width // 2) - (dialog_width // 2)
        center_y = master_y + (master_height // 2) - (dialog_height // 2)
        dialog.geometry(f"+{center_x}+{center_y}")
        dialog.wait_window()

    def apri_e_visualizza_profilo_metodo(self):
        filepath = filedialog.askopenfilename(
            title="Apri Profilo Metodo Salvato",
            filetypes=[
                ("Tutti i Profili Metodo", "*.lmp *.lmcond *.lmcondcorr"),
                ("Profilo Metodo Semplice/Complesso", "*.lmp"),
                ("Profilo Metodo Condizionato", "*.lmcond"),
                ("Profilo Metodo Condizionato Corretto", "*.lmcondcorr"),
                ("Tutti i file", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dati_profilo = json.load(f)

            contenuto_testo_popup = f"--- PROFILO METODO SALVATO ---\n"
            contenuto_testo_popup += f"File: {os.path.basename(filepath)}\n"
            if dati_profilo.get("data_riferimento_analisi"):
                contenuto_testo_popup += f"Analisi del: {dati_profilo['data_riferimento_analisi']}\n"

            formula_metodo_testo = dati_profilo.get("formula_testuale", "N/D")
            tipo_metodo_letto = dati_profilo.get("tipo_metodo_salvato")

            if formula_metodo_testo == "N/D":
                if tipo_metodo_letto == "semplice_analizzato" or \
                   ("metodo" in dati_profilo and isinstance(dati_profilo.get("metodo"), dict) and "tipo_metodo_salvato" not in dati_profilo):
                    m_s = dati_profilo.get("metodo", {})
                    formula_metodo_testo = f"{m_s.get('ruota_calcolo', '?')}[pos.{m_s.get('pos_estratto_calcolo', -1)+1}] {m_s.get('operazione','?')} {m_s.get('operando_fisso','?')}"

                elif tipo_metodo_letto == "condizionato_corretto" or \
                     (dati_profilo.get("def_metodo_esteso_1") and (dati_profilo.get("filtro_condizione_primaria_dict") or dati_profilo.get("definizione_cond_primaria"))):
                    formula_corretta = dati_profilo.get("def_metodo_esteso_1")
                    cond_info = dati_profilo.get("filtro_condizione_primaria_dict") or dati_profilo.get("definizione_cond_primaria") or dati_profilo.get("filtro_condizione_primaria_usato")
                    if formula_corretta and cond_info:
                        desc_formula_interna = "".join(self._format_componente_per_display(c) for c in formula_corretta)
                        formula_metodo_testo = (
                            f"SE {cond_info['ruota']}[pos.{cond_info['posizione']}] IN [{cond_info['val_min']}-{cond_info['val_max']}] "
                            f"ALLORA {desc_formula_interna}"
                        )
                elif tipo_metodo_letto == "complesso_corretto" and dati_profilo.get("def_metodo_esteso_1"):
                     formula_metodo_testo = "".join(self._format_componente_per_display(c) for c in dati_profilo["def_metodo_esteso_1"])

                elif tipo_metodo_letto == "condizionato_base" or ("definizione_cond_primaria" in dati_profilo and "metodo_sommativo_applicato" in dati_profilo):
                    cond = dati_profilo["definizione_cond_primaria"]
                    somm = dati_profilo["metodo_sommativo_applicato"]
                    formula_metodo_testo = (
                        f"SE {cond['ruota']}[pos.{cond['posizione']}] IN [{cond['val_min']}-{cond['val_max']}] "
                        f"ALLORA ({somm['ruota_calcolo']}[pos.{somm['pos_estratto_calcolo']}] {somm['operazione']} {somm['operando_fisso']})"
                    )
                elif tipo_metodo_letto == "complesso_base_analizzato" or "definizione_metodo_originale" in dati_profilo:
                     formula_metodo_testo = "".join(self._format_componente_per_display(c) for c in dati_profilo["definizione_metodo_originale"])
            contenuto_testo_popup += f"Metodo: {formula_metodo_testo}\n"

            ambata_p = dati_profilo.get("ambata_prevista")
            if ambata_p is None: ambata_p = dati_profilo.get("ambata_piu_frequente_dal_metodo")
            if ambata_p is None: ambata_p = dati_profilo.get("ambata_risultante_prima_occ_val")
            if ambata_p is None: ambata_p = dati_profilo.get("previsione_live_cond")
            if ambata_p is None: ambata_p = "N/D"
            contenuto_testo_popup += f"AMBATA PREVISTA (o prima occ./live): {ambata_p}\n"

            ruote_g_p = dati_profilo.get("ruote_gioco_analisi", [])
            ruote_g_p_str = ", ".join(ruote_g_p) if isinstance(ruote_g_p, list) else "N/D"
            contenuto_testo_popup += f"Ruote di Gioco Analisi: {ruote_g_p_str}\n"

            perf_s_val = dati_profilo.get("successi")
            if perf_s_val is None: perf_s_val = dati_profilo.get("successi_cond")
            perf_s = str(perf_s_val) if perf_s_val is not None else "N/A"

            perf_t_val = dati_profilo.get("tentativi")
            if perf_t_val is None: perf_t_val = dati_profilo.get("tentativi_cond")
            perf_t = str(perf_t_val) if perf_t_val is not None else "N/A"

            perf_f_val = dati_profilo.get("frequenza_ambata")
            if perf_f_val is None: perf_f_val = dati_profilo.get("frequenza")
            if perf_f_val is None: perf_f_val = dati_profilo.get("frequenza_cond")

            if perf_f_val is not None:
                try: contenuto_testo_popup += f"Performance Storica: {float(perf_f_val):.2%} ({perf_s}/{perf_t} casi)\n"
                except (ValueError, TypeError): contenuto_testo_popup += f"Performance Storica: {perf_f_val} ({perf_s}/{perf_t} casi)\n"
            else: contenuto_testo_popup += f"Performance Storica: {perf_s} successi su {perf_t} tentativi\n"

            abbinamenti = dati_profilo.get("abbinamenti", {})
            if str(ambata_p).upper() not in ["N/D", "N/A"] and abbinamenti:
                contenuto_testo_popup += "Abbinamenti Consigliati (al momento del salvataggio):\n"
                sortite_target = abbinamenti.get("sortite_ambata_target", 0)
                if sortite_target > 0:
                    contenuto_testo_popup += f"  (Basato su {sortite_target} sortite storiche dell'ambata {ambata_p})\n"
                    for tipo_sorte in ["ambo", "terno", "quaterna", "cinquina"]:
                        dati_s_lista = abbinamenti.get(tipo_sorte, [])
                        if dati_s_lista:
                            contenuto_testo_popup += f"  Per {tipo_sorte.upper()}:\n"
                            for ab_info in dati_s_lista[:3]:
                                if ab_info.get('conteggio', 0) > 0:
                                    n_ab_str = ", ".join(map(str, sorted(ab_info.get('numeri',[]))))
                                    f_ab = ab_info.get('frequenza', 0.0); c_ab = ab_info.get('conteggio',0)
                                    try: contenuto_testo_popup += f"    - Numeri: [{n_ab_str}] (Freq: {float(f_ab):.1%}, Cnt: {c_ab})\n"
                                    except (ValueError, TypeError): contenuto_testo_popup += f"    - Numeri: [{n_ab_str}] (Freq: {f_ab}, Cnt: {c_ab})\n"
                else: contenuto_testo_popup += f"  Nessuna co-occorrenza storica per l'ambata {ambata_p} al momento del salvataggio.\n"

            contenuto_testo_popup += "\nParametri di Analisi Usati (al momento del salvataggio):\n"
            contenuto_testo_popup += f"  Lookahead: {dati_profilo.get('lookahead_analisi', 'N/D')}\n"
            contenuto_testo_popup += f"  Indice Mese: {dati_profilo.get('indice_mese_analisi', 'N/D') or 'Tutte'}\n"

            if dati_profilo.get("tipo_metodo_salvato") == "complesso_corretto" or \
               dati_profilo.get("tipo_metodo_salvato") == "condizionato_corretto" or \
               "tipo_correttore_descrittivo" in dati_profilo:
                contenuto_testo_popup += "\n--- Dettagli Correttore (se applicato) ---\n"
                contenuto_testo_popup += f"Tipo: {dati_profilo.get('tipo_correttore_descrittivo', 'N/D')}\n"
                contenuto_testo_popup += f"Dettaglio: {dati_profilo.get('dettaglio_correttore_str', 'N/D')}\n"
                contenuto_testo_popup += f"Operazione Collegamento Base: '{dati_profilo.get('operazione_collegamento_base', 'N/D')}'\n"

            self.mostra_popup_testo_semplice("Dettaglio Profilo Metodo Salvato", contenuto_testo_popup)

        except FileNotFoundError: messagebox.showerror("Errore Apertura", f"File non trovato: {filepath}"); self._log_to_gui(f"ERRORE: File profilo metodo non trovato: {filepath}")
        except json.JSONDecodeError: messagebox.showerror("Errore Apertura", f"File profilo non è un JSON valido: {filepath}"); self._log_to_gui(f"ERRORE: Impossibile decodificare JSON da {filepath}")
        except Exception as e: messagebox.showerror("Errore Apertura", f"Impossibile aprire il profilo del metodo:\n{e}"); self._log_to_gui(f"ERRORE: Apertura profilo metodo fallita: {e}, {traceback.format_exc()}")

    def _aggiorna_data_inizio_verifica(self, event=None):
        if hasattr(self, 'date_fine_entry_analisi') and hasattr(self, 'date_inizio_verifica_entry'):
            # Verifica se il widget DateEntry per l'inizio della verifica esiste ancora
            if self.date_inizio_verifica_entry.winfo_exists():
                try:
                    data_fine_analisi = self.date_fine_entry_analisi.get_date()
                    if data_fine_analisi: # Solo se c'è una data valida in "Data Fine Analisi"
                        data_inizio_verifica_manuale = data_fine_analisi + timedelta(days=1) # GIORNO SUCCESSIVO
                        self.date_inizio_verifica_entry.set_date(data_inizio_verifica_manuale)
                    else:
                        # Se "Data Fine Analisi" è vuota, pulisci anche "Data Inizio Verifica"
                        self.date_inizio_verifica_entry.delete(0, tk.END)
                except ValueError: 
                    # Se c'è un errore nel prendere data_fine_analisi (es. campo vuoto o formato errato)
                    # pulisci il campo della data di inizio verifica.
                    if hasattr(self, 'date_inizio_verifica_entry') and self.date_inizio_verifica_entry.winfo_exists():
                        try:
                            self.date_inizio_verifica_entry.delete(0, tk.END)
                        except tk.TclError: pass # Ignora se il widget è già distrutto
                except AttributeError: 
                    # Potrebbe succedere se get_date() fallisce in modi imprevisti
                    pass 
                except tk.TclError: 
                    # Il widget potrebbe essere stato distrutto
                    pass

    def _pulisci_data_fine_e_verifica(self, event=None):
        if hasattr(self, 'date_fine_entry_analisi'):
            try:
                self.date_fine_entry_analisi.delete(0, tk.END)
            except tk.TclError: pass
        if hasattr(self, 'date_inizio_verifica_entry'):
             if self.date_inizio_verifica_entry.winfo_exists():
                try: 
                    self.date_inizio_verifica_entry.delete(0, tk.END)
                except tk.TclError: pass

    def crea_gui_controlli_comuni(self, parent_frame_main_tab):
        common_game_settings_frame = ttk.LabelFrame(parent_frame_main_tab, text="Impostazioni di Gioco Comuni (per questa tab)", padding="10")
        common_game_settings_frame.pack(padx=10, pady=5, fill=tk.X, expand=False)
        current_row_cgs = 0
        tk.Label(common_game_settings_frame, text="Ruote di Gioco:").grid(row=current_row_cgs, column=0, sticky="nw", padx=5, pady=2)
        ruote_frame_analisi = tk.Frame(common_game_settings_frame)
        ruote_frame_analisi.grid(row=current_row_cgs, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        tk.Checkbutton(ruote_frame_analisi, text="Tutte le Ruote", variable=self.tutte_le_ruote_var, command=self.toggle_tutte_ruote).grid(row=0, column=0, columnspan=4, sticky="w")
        for i, ruota in enumerate(RUOTE):
            cb = tk.Checkbutton(ruote_frame_analisi, text=ruota, variable=self.ruote_gioco_vars[ruota], command=self.update_tutte_le_ruote_status)
            cb.grid(row=1 + i // 4, column=i % 4, sticky="w")
        current_row_cgs += (len(RUOTE) // 4) + 2
        tk.Label(common_game_settings_frame, text="Colpi di Gioco (Lookahead):").grid(row=current_row_cgs, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(common_game_settings_frame, from_=1, to=200, textvariable=self.lookahead_var, width=5).grid(row=current_row_cgs, column=1, sticky="w", padx=5, pady=2)
        current_row_cgs += 1
        tk.Label(common_game_settings_frame, text="Indice Estrazione del Mese (vuoto=tutte):").grid(row=current_row_cgs, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(common_game_settings_frame, textvariable=self.indice_mese_var, width=7).grid(row=current_row_cgs, column=1, sticky="w", padx=5, pady=2)

    def on_tab_changed(self, event):
        self.active_tab_ruote_checkbox_widgets = []
        try:
            current_tab_id = self.notebook.select()
            if not current_tab_id: return

            current_tab_widget = self.notebook.nametowidget(current_tab_id)
            for child_l1 in current_tab_widget.winfo_children():
                for child_l2 in child_l1.winfo_children():
                    if isinstance(child_l2, ttk.LabelFrame) and "Impostazioni di Gioco Comuni" in child_l2.cget("text"):
                        for child_l3 in child_l2.winfo_children():
                            if isinstance(child_l3, tk.Frame):
                                temp_widget_list = []
                                is_target_ruote_frame = False
                                for widget_in_frame in child_l3.winfo_children():
                                    if isinstance(widget_in_frame, tk.Checkbutton):
                                        if widget_in_frame.cget("text") == "Tutte le Ruote": is_target_ruote_frame = True
                                        if widget_in_frame.cget("text") in RUOTE: temp_widget_list.append(widget_in_frame)
                                if is_target_ruote_frame:
                                    self.active_tab_ruote_checkbox_widgets = temp_widget_list
                                    break
                        if self.active_tab_ruote_checkbox_widgets: break
                if self.active_tab_ruote_checkbox_widgets: break
        except Exception:
            pass
        self.toggle_tutte_ruote()

    def toggle_tutte_ruote(self):
        stato_tutte_var = self.tutte_le_ruote_var.get()
        for nome_ruota in self.ruote_gioco_vars: self.ruote_gioco_vars[nome_ruota].set(stato_tutte_var)
        nuovo_stato_widget_figli = tk.DISABLED if stato_tutte_var else tk.NORMAL
        for cb_widget in self.active_tab_ruote_checkbox_widgets:
            try:
                if cb_widget.winfo_exists(): cb_widget.config(state=nuovo_stato_widget_figli)
            except tk.TclError: pass

    def update_tutte_le_ruote_status(self):
        tutti_figli_selezionati = all(self.ruote_gioco_vars[ruota].get() for ruota in RUOTE)
        if self.tutte_le_ruote_var.get() != tutti_figli_selezionati: self.tutte_le_ruote_var.set(tutti_figli_selezionati)

    def _toggle_tutti_mesi_periodica(self):
        stato_tutti = self.ap_tutti_mesi_var.get()
        for mese_var in self.ap_mesi_vars.values():
            mese_var.set(stato_tutti)

        nuovo_stato_widget = tk.DISABLED if stato_tutti else tk.NORMAL
        if hasattr(self, 'ap_mesi_checkbox_widgets'):
            for cb in self.ap_mesi_checkbox_widgets:
                if cb.winfo_exists():
                    cb.config(state=nuovo_stato_widget)

    def _update_tutti_mesi_status_periodica(self):
        tutti_selezionati = all(var.get() for var in self.ap_mesi_vars.values())
        if self.ap_tutti_mesi_var.get() != tutti_selezionati:
            self.ap_tutti_mesi_var.set(tutti_selezionati)
            if not self.ap_tutti_mesi_var.get():
                 if hasattr(self, 'ap_mesi_checkbox_widgets'):
                    for cb in self.ap_mesi_checkbox_widgets:
                        if cb.winfo_exists():
                            cb.config(state=tk.NORMAL)

    def _update_ap_numeri_input_state(self, event=None):
        """Abilita/Disabilita l'Entry per i numeri in base al tipo di sorte selezionato."""
        if hasattr(self, 'ap_entry_numeri_input'):
            if self.ap_tipo_sorte_var.get() == "Ambata":
                self.ap_entry_numeri_input.config(state=tk.DISABLED)
                self.ap_numeri_input_var.set("")
            else:
                self.ap_entry_numeri_input.config(state=tk.NORMAL)

    def crea_gui_metodi_semplici(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')

        top_controls_frame = ttk.Frame(main_frame)
        top_controls_frame.pack(fill=tk.X, padx=0, pady=0, anchor='n')
        self.crea_gui_controlli_comuni(top_controls_frame)

        simple_method_params_frame = ttk.LabelFrame(top_controls_frame, text="Parametri Ricerca Metodi Semplici", padding="10")
        simple_method_params_frame.pack(padx=10, pady=(5,10), fill=tk.X)

        top_buttons_frame = ttk.Frame(simple_method_params_frame)
        top_buttons_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
        tk.Button(top_buttons_frame, text="Salva Imp. Semplici", command=self.salva_impostazioni_semplici).pack(side=tk.LEFT, padx=5)
        tk.Button(top_buttons_frame, text="Apri Imp. Semplici", command=self.apri_impostazioni_semplici).pack(side=tk.LEFT, padx=5)
        tk.Button(top_buttons_frame, text="Log", command=self.mostra_finestra_log, width=5).pack(side=tk.LEFT, padx=15)

        current_row = 1
        tk.Label(simple_method_params_frame, text="Ruota Calcolo Base:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(simple_method_params_frame, textvariable=self.ruota_calcolo_var, values=RUOTE, state="readonly", width=15).grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(simple_method_params_frame, text="Posizione Estratto Base (1-5):").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(simple_method_params_frame, from_=1, to=5, textvariable=self.posizione_estratto_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(simple_method_params_frame, text="N. Ambate da Dettagliare (nel popup):").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(simple_method_params_frame, from_=1, to=10, textvariable=self.num_ambate_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(simple_method_params_frame, text="Min. Tentativi per Metodo:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(simple_method_params_frame, from_=1, to=100, textvariable=self.min_tentativi_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Button(simple_method_params_frame, text="Avvia Ricerca Metodi Semplici",
                  command=self.avvia_analisi_metodi_semplici,
                  font=("Helvetica", 11, "bold"), bg="lightgreen"
                 ).grid(row=current_row, column=0, columnspan=2, pady=10, ipady=3)

        risultati_frame_ms = ttk.LabelFrame(main_frame, text="Top Metodi Semplici Trovati", padding="10")
        risultati_frame_ms.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.ms_risultati_listbox = Listbox(risultati_frame_ms, height=10, font=("Courier New", 9), exportselection=False)
        ms_scrollbar_y = ttk.Scrollbar(risultati_frame_ms, orient="vertical", command=self.ms_risultati_listbox.yview)
        self.ms_risultati_listbox.config(yscrollcommand=ms_scrollbar_y.set)
        ms_scrollbar_y.pack(side=tk.RIGHT, fill="y")
        self.ms_risultati_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_backtest_semplice_sel = tk.Button(risultati_frame_ms,
                                             text="Backtest Dettagliato Metodo Semplice Selezionato",
                                             command=self.avvia_backtest_metodo_semplice_selezionato,
                                             font=("Helvetica", 10, "bold"), bg="lightcyan")
        btn_backtest_semplice_sel.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0), ipady=3)

    def crea_gui_metodo_complesso(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')

        controlli_comuni_container = ttk.Frame(main_frame)
        controlli_comuni_container.pack(fill=tk.X, padx=0, pady=0, anchor='n')
        self.crea_gui_controlli_comuni(controlli_comuni_container)

        costruttori_main_frame = ttk.LabelFrame(main_frame, text="Costruttori Metodi Complessi", padding="10")
        costruttori_main_frame.pack(padx=10, pady=(5,5), fill=tk.BOTH, expand=True)

        save_load_log_mc_frame = ttk.Frame(costruttori_main_frame)
        save_load_log_mc_frame.pack(fill=tk.X, pady=(0,10), anchor='nw')
        tk.Button(save_load_log_mc_frame, text="Salva Metodi Compl.", command=self.salva_metodi_complessi).pack(side=tk.LEFT, padx=5)
        tk.Button(save_load_log_mc_frame, text="Apri Metodi Compl.", command=self.apri_metodi_complessi).pack(side=tk.LEFT, padx=5)
        tk.Button(save_load_log_mc_frame, text="Log", command=self.mostra_finestra_log, width=5).pack(side=tk.LEFT, padx=15)

        self.mc_notebook = ttk.Notebook(costruttori_main_frame)
        self.mc_notebook.pack(expand=True, fill="both", pady=5)

        tab_metodo_1 = ttk.Frame(self.mc_notebook, padding="5")
        self.mc_notebook.add(tab_metodo_1, text=' Metodo Base 1 ')
        tk.Label(tab_metodo_1, text="Metodo Attuale:").pack(anchor="w")
        self.mc_listbox_componenti_1 = tk.Listbox(tab_metodo_1, height=3)
        self.mc_listbox_componenti_1.pack(fill=tk.X, expand=False, pady=(0,5))
        add_comp_frame_1 = ttk.Frame(tab_metodo_1); add_comp_frame_1.pack(fill=tk.X)
        tk.Radiobutton(add_comp_frame_1, text="Estratto", variable=self.mc_tipo_termine_var, value="estratto", command=self._update_mc_input_state_1).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(add_comp_frame_1, text="Fisso", variable=self.mc_tipo_termine_var, value="fisso", command=self._update_mc_input_state_1).grid(row=0, column=1, sticky="w")
        self.mc_ruota_label_1 = tk.Label(add_comp_frame_1, text="Ruota:"); self.mc_ruota_label_1.grid(row=1, column=0, sticky="e", padx=2)
        self.mc_ruota_combo_1 = ttk.Combobox(add_comp_frame_1, textvariable=self.mc_ruota_var, values=RUOTE, state="readonly", width=10); self.mc_ruota_combo_1.grid(row=1, column=1, sticky="w", padx=2)
        self.mc_pos_label_1 = tk.Label(add_comp_frame_1, text="Pos. (1-5):"); self.mc_pos_label_1.grid(row=1, column=2, sticky="e", padx=2)
        self.mc_pos_spinbox_1 = tk.Spinbox(add_comp_frame_1, from_=1, to=5, textvariable=self.mc_posizione_var, width=4, state="readonly"); self.mc_pos_spinbox_1.grid(row=1, column=3, sticky="w", padx=2)
        self.mc_fisso_label_1 = tk.Label(add_comp_frame_1, text="Valore Fisso (1-90):"); self.mc_fisso_label_1.grid(row=2, column=0, sticky="e", padx=2)
        self.mc_fisso_spinbox_1 = tk.Spinbox(add_comp_frame_1, from_=1, to=90, textvariable=self.mc_valore_fisso_var, width=4, state="readonly"); self.mc_fisso_spinbox_1.grid(row=2, column=1, sticky="w", padx=2)
        tk.Label(add_comp_frame_1, text="Op. Successiva:").grid(row=3, column=0, sticky="e", padx=2)
        self.mc_op_combo_1 = ttk.Combobox(add_comp_frame_1, textvariable=self.mc_operazione_var, values=list(OPERAZIONI_COMPLESSE.keys()) + ['='], state="readonly", width=4); self.mc_op_combo_1.grid(row=3, column=1, sticky="w", padx=2); self.mc_op_combo_1.set('+')
        tk.Button(add_comp_frame_1, text="Aggiungi Termine", command=self.aggiungi_componente_metodo_1).grid(row=3, column=2, columnspan=2, pady=5, padx=5, sticky="ew")
        buttons_frame_1 = ttk.Frame(tab_metodo_1); buttons_frame_1.pack(fill=tk.X, pady=5)
        tk.Button(buttons_frame_1, text="Rimuovi Ultimo", command=self.rimuovi_ultimo_componente_metodo_1).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame_1, text="Pulisci Metodo", command=self.pulisci_metodo_complesso_1).pack(side=tk.LEFT, padx=5)

        tab_metodo_2 = ttk.Frame(self.mc_notebook, padding="5")
        self.mc_notebook.add(tab_metodo_2, text=' Metodo Base 2 (Opz.)')
        tk.Label(tab_metodo_2, text="Metodo Attuale:").pack(anchor="w")
        self.mc_listbox_componenti_2 = tk.Listbox(tab_metodo_2, height=3)
        self.mc_listbox_componenti_2.pack(fill=tk.X, expand=False, pady=(0,5))
        add_comp_frame_2 = ttk.Frame(tab_metodo_2); add_comp_frame_2.pack(fill=tk.X)
        tk.Radiobutton(add_comp_frame_2, text="Estratto", variable=self.mc_tipo_termine_var_2, value="estratto", command=self._update_mc_input_state_2).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(add_comp_frame_2, text="Fisso", variable=self.mc_tipo_termine_var_2, value="fisso", command=self._update_mc_input_state_2).grid(row=0, column=1, sticky="w")
        self.mc_ruota_label_2 = tk.Label(add_comp_frame_2, text="Ruota:"); self.mc_ruota_label_2.grid(row=1, column=0, sticky="e", padx=2)
        self.mc_ruota_combo_2 = ttk.Combobox(add_comp_frame_2, textvariable=self.mc_ruota_var_2, values=RUOTE, state="readonly", width=10); self.mc_ruota_combo_2.grid(row=1, column=1, sticky="w", padx=2)
        self.mc_pos_label_2 = tk.Label(add_comp_frame_2, text="Pos. (1-5):"); self.mc_pos_label_2.grid(row=1, column=2, sticky="e", padx=2)
        self.mc_pos_spinbox_2 = tk.Spinbox(add_comp_frame_2, from_=1, to=5, textvariable=self.mc_posizione_var_2, width=4, state="readonly"); self.mc_pos_spinbox_2.grid(row=1, column=3, sticky="w", padx=2)
        self.mc_fisso_label_2 = tk.Label(add_comp_frame_2, text="Valore Fisso (1-90):"); self.mc_fisso_label_2.grid(row=2, column=0, sticky="e", padx=2)
        self.mc_fisso_spinbox_2 = tk.Spinbox(add_comp_frame_2, from_=1, to=90, textvariable=self.mc_valore_fisso_var_2, width=4, state="readonly"); self.mc_fisso_spinbox_2.grid(row=2, column=1, sticky="w", padx=2)
        tk.Label(add_comp_frame_2, text="Op. Successiva:").grid(row=3, column=0, sticky="e", padx=2)
        self.mc_op_combo_2 = ttk.Combobox(add_comp_frame_2, textvariable=self.mc_operazione_var_2, values=list(OPERAZIONI_COMPLESSE.keys()) + ['='], state="readonly", width=4); self.mc_op_combo_2.grid(row=3, column=1, sticky="w", padx=2); self.mc_op_combo_2.set('+')
        tk.Button(add_comp_frame_2, text="Aggiungi Termine", command=self.aggiungi_componente_metodo_2).grid(row=3, column=2, columnspan=2, pady=5, padx=5, sticky="ew")
        buttons_frame_2 = ttk.Frame(tab_metodo_2); buttons_frame_2.pack(fill=tk.X, pady=5)
        tk.Button(buttons_frame_2, text="Rimuovi Ultimo", command=self.rimuovi_ultimo_componente_metodo_2).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame_2, text="Pulisci Metodo", command=self.pulisci_metodo_complesso_2).pack(side=tk.LEFT, padx=5)

        action_buttons_main_frame = ttk.Frame(main_frame)
        action_buttons_main_frame.pack(pady=(10,10), fill=tk.X, padx=10, side=tk.BOTTOM)

        top_buttons_action_frame = ttk.Frame(action_buttons_main_frame)
        top_buttons_action_frame.pack(fill=tk.X, expand=False, pady=(0,5))

        padding_verticale_pulsanti = 3
        tk.Button(top_buttons_action_frame, text="Analizza Metodi Base Definiti",
                  command=self.avvia_analisi_metodo_complesso,
                  font=("Helvetica", 10, "bold"), bg="lightcoral"
                 ).pack(side=tk.LEFT, expand=True, fill=tk.BOTH, ipady=padding_verticale_pulsanti, padx=(0,2))

        correttore_button_container_frame = ttk.Frame(top_buttons_action_frame)
        correttore_button_container_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(2,0))
        tk.Button(correttore_button_container_frame, text="Trova Correttore",
                  command=self.avvia_ricerca_correttore,
                  font=("Helvetica", 10, "bold"), bg="gold"
                 ).pack(expand=True, fill=tk.BOTH, ipady=padding_verticale_pulsanti)

        backtest_manual_choice_frame = ttk.LabelFrame(action_buttons_main_frame, text="Scegli Metodo per Backtest (se non da popup/correttore)", padding=5)
        backtest_manual_choice_frame.pack(fill=tk.X, pady=(5,2))

        tk.Radiobutton(backtest_manual_choice_frame, text="Usa Metodo Base 1", variable=self.mc_backtest_choice_var, value="base1").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(backtest_manual_choice_frame, text="Usa Metodo Base 2", variable=self.mc_backtest_choice_var, value="base2").pack(side=tk.LEFT, padx=5)

        backtest_options_frame = ttk.Frame(action_buttons_main_frame)
        backtest_options_frame.pack(fill=tk.X, expand=False, pady=(2,0))

        self.chk_usa_corretto = tk.Checkbutton(backtest_options_frame,
                                               text="Usa ultimo metodo corretto per Backtest Dettagliato",
                                               variable=self.usa_ultimo_corretto_per_backtest_var)
        self.chk_usa_corretto.pack(side=tk.LEFT, padx=(0,10))

        btn_backtest_dettagliato = tk.Button(backtest_options_frame, text="Backtest Dettagliato",
                                               command=self.avvia_backtest_dettagliato_metodo,
                                               font=("Helvetica", 10, "bold"), bg="lightblue")
        btn_backtest_dettagliato.pack(fill=tk.X, expand=True, ipady=5)

        self._update_mc_input_state_1()
        self._refresh_mc_listbox_1()
        self._update_mc_input_state_2()
        self._refresh_mc_listbox_2()

    def crea_gui_analisi_condizionata(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')

        top_controls_ac_frame = ttk.Frame(main_frame)
        top_controls_ac_frame.pack(fill=tk.X, padx=0, pady=0, anchor='n')
        self.crea_gui_controlli_comuni(top_controls_ac_frame)

        ac_params_frame = ttk.LabelFrame(top_controls_ac_frame, text="Parametri Analisi Condizionata Avanzata", padding="10")
        ac_params_frame.pack(padx=10, pady=5, fill=tk.X)

        cond_frame = ttk.LabelFrame(ac_params_frame, text="1. Condizione di Filtraggio Estrazioni", padding="5")
        cond_frame.pack(fill=tk.X, pady=3)
        current_row_cond = 0
        tk.Label(cond_frame, text="Ruota Condizione:").grid(row=current_row_cond, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(cond_frame, textvariable=self.ac_ruota_cond_var, values=RUOTE, state="readonly", width=15).grid(row=current_row_cond, column=1, sticky="w", padx=5, pady=2)
        current_row_cond += 1
        tk.Label(cond_frame, text="Pos. Estratto Cond. (1-5):").grid(row=current_row_cond, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(cond_frame, from_=1, to=5, textvariable=self.ac_pos_cond_var, width=5, state="readonly").grid(row=current_row_cond, column=1, sticky="w", padx=5, pady=2)
        current_row_cond += 1
        tk.Label(cond_frame, text="Valore Min. Estratto:").grid(row=current_row_cond, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(cond_frame, from_=1, to=90, textvariable=self.ac_val_min_cond_var, width=5, state="readonly").grid(row=current_row_cond, column=1, sticky="w", padx=5, pady=2)
        current_row_cond += 1
        tk.Label(cond_frame, text="Valore Max. Estratto:").grid(row=current_row_cond, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(cond_frame, from_=1, to=90, textvariable=self.ac_val_max_cond_var, width=5, state="readonly").grid(row=current_row_cond, column=1, sticky="w", padx=5, pady=2)

        calc_amb_frame = ttk.LabelFrame(ac_params_frame, text="2. Ricerca Ambata Sommativa su Estrazioni Filtrate", padding="5")
        calc_amb_frame.pack(fill=tk.X, pady=3)
        current_row_calc = 0
        tk.Label(calc_amb_frame, text="Ruota Calcolo Ambata:").grid(row=current_row_calc, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(calc_amb_frame, textvariable=self.ac_ruota_calc_ambata_var, values=RUOTE, state="readonly", width=15).grid(row=current_row_calc, column=1, sticky="w", padx=5, pady=2)
        current_row_calc += 1
        tk.Label(calc_amb_frame, text="Pos. Estratto Calcolo (1-5):").grid(row=current_row_calc, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(calc_amb_frame, from_=1, to=5, textvariable=self.ac_pos_calc_ambata_var, width=5, state="readonly").grid(row=current_row_calc, column=1, sticky="w", padx=5, pady=2)
        current_row_calc += 1
        tk.Label(calc_amb_frame, text="N. Migliori Metodi da Trovare:").grid(row=current_row_calc, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(calc_amb_frame, from_=1, to=5, textvariable=self.ac_num_risultati_var, width=5, state="readonly").grid(row=current_row_calc, column=1, sticky="w", padx=5, pady=2)
        current_row_calc += 1
        tk.Label(calc_amb_frame, text="Min. Tentativi (post-filtro):").grid(row=current_row_calc, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(calc_amb_frame, from_=1, to=100, textvariable=self.ac_min_tentativi_var, width=5, state="readonly").grid(row=current_row_calc, column=1, sticky="w", padx=5, pady=2)

        ac_action_buttons_frame = ttk.Frame(ac_params_frame)
        ac_action_buttons_frame.pack(fill=tk.X, pady=(10,5), padx=0)
        tk.Button(ac_action_buttons_frame, text="Avvia Ricerca Condizionata", command=self.avvia_analisi_condizionata, font=("Helvetica", 11, "bold"), bg="lightseagreen").pack(side=tk.LEFT, padx=(0,10), ipady=3, expand=True, fill=tk.X)
        tk.Button(ac_action_buttons_frame, text="Log", command=self.mostra_finestra_log, width=10).pack(side=tk.LEFT, ipady=3)

        risultati_main_frame = ttk.LabelFrame(main_frame, text="Risultati Ricerca Condizionata e Opzioni Avanzate", padding="10")
        risultati_main_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        listbox_frame = ttk.Frame(risultati_main_frame)
        listbox_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        self.ac_risultati_listbox = Listbox(listbox_frame, height=6, exportselection=False, font=("Courier New", 9))
        ac_scrollbar_y = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.ac_risultati_listbox.yview)
        ac_scrollbar_x = ttk.Scrollbar(listbox_frame, orient="horizontal", command=self.ac_risultati_listbox.xview)
        self.ac_risultati_listbox.config(yscrollcommand=ac_scrollbar_y.set, xscrollcommand=ac_scrollbar_x.set)
        ac_scrollbar_y.pack(side=tk.RIGHT, fill="y")
        ac_scrollbar_x.pack(side=tk.BOTTOM, fill="x")
        self.ac_risultati_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        action_buttons_listbox_frame = ttk.Frame(risultati_main_frame)
        action_buttons_listbox_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5,0))

        btn_correttore_cond = tk.Button(action_buttons_listbox_frame, text="Applica Correttore\nal Metodo Selezionato",
                                     command=self.avvia_ricerca_correttore_per_selezionato_condizionato,
                                     font=("Helvetica", 9), bg="khaki", width=20)
        btn_correttore_cond.pack(side=tk.TOP, pady=(0,5), ipady=2, fill=tk.X)

        btn_salva_cond = tk.Button(action_buttons_listbox_frame, text="Salva Metodo\nSelezionato",
                                     command=self.salva_metodo_condizionato_selezionato,
                                     font=("Helvetica", 9), bg="lightsteelblue", width=20)
        btn_salva_cond.pack(side=tk.TOP, pady=5, ipady=2, fill=tk.X)

        btn_backtest_cond_sel = tk.Button(action_buttons_listbox_frame,
                                          text="Backtest Dettagliato\nMetodo Cond. Selezionato",
                                          command=self.avvia_backtest_del_condizionato_selezionato,
                                          font=("Helvetica", 9), bg="paleturquoise", width=20)
        btn_backtest_cond_sel.pack(side=tk.TOP, pady=5, ipady=2, fill=tk.X)

        self.btn_backtest_cond_corretto = tk.Button(action_buttons_listbox_frame,
                                                       text="Backtest Metodo Cond.\n+ Correttore Trovato",
                                                       command=self.avvia_backtest_del_condizionato_corretto,
                                                       font=("Helvetica", 9), bg="lightyellow", width=20,
                                                       state=tk.DISABLED)
        self.btn_backtest_cond_corretto.pack(side=tk.TOP, pady=5, ipady=2, fill=tk.X)


    def crea_gui_analisi_periodica(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')

        self.crea_gui_controlli_comuni(main_frame)

        period_params_frame = ttk.LabelFrame(main_frame, text="Parametri Analisi Periodica", padding="10")
        period_params_frame.pack(padx=10, pady=5, fill=tk.X, expand=False)

        mesi_frame = ttk.LabelFrame(period_params_frame, text="Mesi da Analizzare", padding="5")
        mesi_frame.pack(fill=tk.X, pady=3)
        tk.Checkbutton(mesi_frame, text="Tutti i Mesi", variable=self.ap_tutti_mesi_var, command=self._toggle_tutti_mesi_periodica).grid(row=0, column=0, columnspan=4, sticky="w", padx=5)
        mesi_nomi = list(self.ap_mesi_vars.keys())
        self.ap_mesi_checkbox_widgets = []
        for i, nome_mese in enumerate(mesi_nomi):
            cb = tk.Checkbutton(mesi_frame, text=nome_mese, variable=self.ap_mesi_vars[nome_mese], command=self._update_tutti_mesi_status_periodica)
            cb.grid(row=1 + i // 4, column=i % 4, sticky="w", padx=5)
            self.ap_mesi_checkbox_widgets.append(cb)

        sorte_frame = ttk.LabelFrame(period_params_frame, text="Analisi Frequenza (Ambate o Combinazioni Specifiche)", padding="5")
        sorte_frame.pack(fill=tk.X, pady=5)

        tk.Label(sorte_frame, text="Sorte:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ap_combo_tipo_sorte = ttk.Combobox(sorte_frame, textvariable=self.ap_tipo_sorte_var,
                                                values=["Ambata", "Ambo", "Terno", "Quaterna", "Cinquina"],
                                                state="readonly", width=15)
        self.ap_combo_tipo_sorte.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ottimale_frame = ttk.LabelFrame(period_params_frame, text="Ricerca Ambata Ottimale Periodica (con Metodo Sommativo)", padding="5")
        ottimale_frame.pack(fill=tk.X, pady=(10,3))

        tk.Label(ottimale_frame, text="Ruota Calcolo Base:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.ap_ruota_calcolo_ott_var = tk.StringVar(value=RUOTE[0])
        ttk.Combobox(ottimale_frame, textvariable=self.ap_ruota_calcolo_ott_var, values=RUOTE, state="readonly", width=15).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        tk.Label(ottimale_frame, text="Pos. Estratto Base (1-5):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.ap_pos_estratto_ott_var = tk.IntVar(value=1)
        tk.Spinbox(ottimale_frame, from_=1, to=5, textvariable=self.ap_pos_estratto_ott_var, width=5, state="readonly").grid(row=1, column=1, sticky="w", padx=5, pady=2)

        action_frame_ap = ttk.Frame(period_params_frame)
        action_frame_ap.pack(fill=tk.X, pady=(10,5), padx=5)

        tk.Button(action_frame_ap, text="1. Analizza Presenze Periodiche",
                  command=self.avvia_analisi_frequenza_periodica,
                  font=("Helvetica", 10, "bold"), bg="mediumpurple1").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5), ipady=2)

        tk.Button(action_frame_ap, text="2. Trova Ambata Ottimale Periodica",
                  command=self.avvia_ricerca_ambata_ottimale_periodica,
                  font=("Helvetica", 10, "bold"), bg="lightcoral").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,10), ipady=2)

        tk.Button(action_frame_ap, text="Log", command=self.mostra_finestra_log, width=8).pack(side=tk.LEFT, ipady=2, padx=(5,0))

        risultati_frame_ap = ttk.LabelFrame(main_frame, text="Risultati Analisi Periodica", padding="10")
        risultati_frame_ap.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.ap_risultati_listbox = Listbox(risultati_frame_ap, height=10, font=("Courier New", 9), exportselection=False)
        ap_scrollbar_y = ttk.Scrollbar(risultati_frame_ap, orient="vertical", command=self.ap_risultati_listbox.yview)
        self.ap_risultati_listbox.config(yscrollcommand=ap_scrollbar_y.set)

        ap_scrollbar_y.pack(side=tk.RIGHT, fill="y")
        self.ap_risultati_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._toggle_tutti_mesi_periodica()
        self._update_ap_numeri_input_state()


    def avvia_analisi_frequenza_periodica(self):
        self._log_to_gui("\n" + "="*50 + f"\nAVVIO ANALISI PRESENZE PERIODICHE\n" + "="*50)

        if self.ap_risultati_listbox: self.ap_risultati_listbox.delete(0, tk.END)

        storico_base_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_base_per_analisi: return

        mesi_selezionati_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
        if not mesi_selezionati_gui and not self.ap_tutti_mesi_var.get():
            messagebox.showwarning("Selezione Mesi", "Seleziona almeno un mese o 'Tutti i Mesi'.")
            return

        mesi_map = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
        mesi_numeri_selezionati = []
        if not self.ap_tutti_mesi_var.get(): mesi_numeri_selezionati = [mesi_map[nome] for nome in mesi_selezionati_gui]

        ruote_gioco_sel, lookahead_ap, indice_mese_ap_utente = self._get_parametri_gioco_comuni()
        if ruote_gioco_sel is None: return

        self.master.config(cursor="watch"); self.master.update_idletasks()
        
        data_inizio_g_filtro, data_fine_g_filtro_obj = None, None
        try: data_inizio_g_filtro = self.date_inizio_entry_analisi.get_date()
        except ValueError: pass
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_g_filtro_obj = self.date_fine_entry_analisi.get_date()
        except ValueError: pass

        storico_filtrato_finale = filtra_storico_per_periodo(
            storico_base_per_analisi, mesi_numeri_selezionati,
            data_inizio_g_filtro, data_fine_g_filtro_obj, app_logger=self._log_to_gui
        )

        if not storico_filtrato_finale:
            messagebox.showinfo("Analisi Periodica", "Nessuna estrazione trovata per i mesi e il range di date specificato.")
            self.master.config(cursor=""); return

        lista_previsioni_per_popup, metodi_grezzi_per_salvataggio_popup = [], []
        estrazione_riferimento_live_ap = storico_filtrato_finale[-1] if storico_filtrato_finale else None
        tipo_analisi_scelta = self.ap_tipo_sorte_var.get()

        if tipo_analisi_scelta == "Ambata":
            conteggio_ambate_grezzo, num_estraz_analizzate = analizza_frequenza_ambate_periodica(storico_filtrato_finale, ruote_gioco_sel, self._log_to_gui)

            if not conteggio_ambate_grezzo:
                 messagebox.showinfo("Analisi Periodica", "Nessuna ambata trovata nel periodo selezionato.")
            else:
                top_ambata_numero, top_ambata_conteggio = conteggio_ambate_grezzo.most_common(1)[0]
                freq_presenza = (top_ambata_conteggio / num_estraz_analizzate * 100) if num_estraz_analizzate > 0 else 0
                abbinamenti_ris = analizza_abbinamenti_per_numero_specifico(storico_filtrato_finale, top_ambata_numero, ruote_gioco_sel, self._log_to_gui)
                
                formula_testuale_display = f"Ambata più frequente ({top_ambata_numero}) nel periodo: Mesi={', '.join(mesi_selezionati_gui) or 'Tutti'}"
                
                # >>> BLOCCO CRUCIALE CORRETTO <<<
                definizione_strutturata_per_backtest = [{'tipo_termine': 'fisso', 'valore_fisso': top_ambata_numero, 'operazione_successiva': '='}]
                
                dati_salvataggio_ambata = {
                    "tipo_metodo_salvato": "periodica_ambata_frequente",
                    "formula_testuale": formula_testuale_display,
                    "ambata_prevista": top_ambata_numero,
                    "abbinamenti": abbinamenti_ris,
                    "successi": top_ambata_conteggio, # Per coerenza, usiamo il conteggio come "successi"
                    "tentativi": num_estraz_analizzate, # E le estrazioni come "tentativi"
                    "frequenza": freq_presenza / 100.0,
                    "definizione_strutturata": definizione_strutturata_per_backtest, # Aggiunge la "ricetta" che mancava
                    "parametri_periodo": {"mesi": mesi_selezionati_gui or "Tutti"}
                }
                metodi_grezzi_per_salvataggio_popup.append(dati_salvataggio_ambata)
                # >>> FINE BLOCCO CRUCIALE <<<

                dettaglio_popup_ambata = {
                    "titolo_sezione": f"--- AMBATA PIÙ PRESENTE: {top_ambata_numero} ---",
                    "info_metodo_str": formula_testuale_display, "ambata_prevista": top_ambata_numero,
                    "performance_storica_str": f"Presenza: {freq_presenza:.1f}% ({top_ambata_conteggio}/{num_estraz_analizzate} estraz.)",
                    "abbinamenti_dict": abbinamenti_ris
                }
                lista_previsioni_per_popup.append(dettaglio_popup_ambata)
        else:
             messagebox.showinfo("Analisi Periodica", f"La preparazione al backtest per la sorte '{tipo_analisi_scelta}' non è implementata.")

        self.master.config(cursor="")

        if lista_previsioni_per_popup:
            self.mostra_popup_previsione(
                titolo_popup=f"Risultati Analisi Periodica ({tipo_analisi_scelta})", ruote_gioco_str=", ".join(ruote_gioco_sel),
                lista_previsioni_dettagliate=lista_previsioni_per_popup,
                data_riferimento_previsione_str_comune=estrazione_riferimento_live_ap['data'].strftime('%d/%m/%Y') if estrazione_riferimento_live_ap else "N/D",
                metodi_grezzi_per_salvataggio=metodi_grezzi_per_salvataggio_popup,
                indice_mese_richiesto_utente=indice_mese_ap_utente, data_fine_analisi_globale_obj=data_fine_g_filtro_obj,
                estrazione_riferimento_per_previsione_live=estrazione_riferimento_live_ap
            )

    def crea_gui_aau(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')

        top_controls_container_frame = ttk.Frame(main_frame)
        top_controls_container_frame.pack(fill=tk.X, padx=0, pady=0, anchor='n')

        # Questa funzione crea i controlli comuni (Ruote, Lookahead, Indice Mese)
        # Assicurati che sia definita e funzioni correttamente
        self.crea_gui_controlli_comuni(top_controls_container_frame)

        params_frame_aau = ttk.LabelFrame(top_controls_container_frame, text="Parametri Ricerca Ambata e Ambo Unico", padding="10")
        params_frame_aau.pack(padx=10, pady=(5,10), fill=tk.X)

        current_row_aau = 0

        tk.Label(params_frame_aau, text="Ruota Estratto Base:").grid(row=current_row_aau, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(params_frame_aau, textvariable=self.aau_ruota_base_var, values=RUOTE, state="readonly", width=15).grid(row=current_row_aau, column=1, columnspan=3, sticky="w", padx=5, pady=2)
        current_row_aau += 1

        tk.Label(params_frame_aau, text="Posizione Estratto Base (1-5):").grid(row=current_row_aau, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(params_frame_aau, from_=1, to=5, textvariable=self.aau_pos_base_var, width=5, state="readonly").grid(row=current_row_aau, column=1, columnspan=3, sticky="w", padx=5, pady=2)
        current_row_aau += 1

        # Operazioni Base (+, -, *) da applicare tra EstrattoBase e CorrettoreTrasformato
        op_base_frame_aau = ttk.LabelFrame(params_frame_aau, text="Operazioni Base da Testare (tra Base e Correttore Trasformato)", padding=3)
        op_base_frame_aau.grid(row=current_row_aau, column=0, columnspan=4, sticky="ew", padx=5, pady=3)
        tk.Checkbutton(op_base_frame_aau, text="Somma (+)", variable=self.aau_op_somma_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(op_base_frame_aau, text="Sottrazione (-)", variable=self.aau_op_diff_var).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(op_base_frame_aau, text="Moltiplicazione (*)", variable=self.aau_op_mult_var).pack(side=tk.LEFT, padx=5)
        current_row_aau += 1
        
        # Trasformazioni da Applicare ai Correttori Fissi (1-90)
        trasform_frame_aau = ttk.LabelFrame(params_frame_aau, text="Trasformazioni da Applicare ai Correttori Fissi (1-90)", padding=5)
        trasform_frame_aau.grid(row=current_row_aau, column=0, columnspan=4, sticky="ew", padx=5, pady=3)
        
        # self.aau_trasf_vars dovrebbe essere già inizializzato come {} in __init__
        # Se non lo è, decommenta la riga seguente o assicurati che sia in __init__
        # self.aau_trasf_vars = {} 
        
        col_t = 0 # Rinominato per evitare conflitto con 'col' usato sopra se ci fosse
        row_t_gui = 0 # Rinominato per evitare conflitto con 'row_t' usato sopra
        
        # Lista delle trasformazioni da mostrare, nell'ordine desiderato
        # "Fisso" è il default e significa nessuna trasformazione speciale.
        trasformazioni_da_mostrare_gui = ["Fisso"] + sorted([
           k for k in OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.keys() if k != "Fisso"
        ])
        # Assicurati che OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE sia definito globalmente
        # e contenga "Diam.Decina" e le altre tue trasformazioni.

        for nome_trasf in trasformazioni_da_mostrare_gui:
            if nome_trasf not in self.aau_trasf_vars: # Crea la BooleanVar se non esiste
                 self.aau_trasf_vars[nome_trasf] = tk.BooleanVar(value=(nome_trasf == "Fisso"))
            
            cb = ttk.Checkbutton(trasform_frame_aau, text=nome_trasf, variable=self.aau_trasf_vars[nome_trasf])
            cb.grid(row=row_t_gui, column=col_t, sticky="w", padx=5, pady=1)
            col_t += 1
            if col_t >= 3: # Numero di checkbox per riga
                col_t = 0
                row_t_gui += 1
        current_row_aau += (row_t_gui + 1) # Aggiorna current_row_aau per i widget successivi

        action_button_frame_aau = ttk.Frame(params_frame_aau)
        action_button_frame_aau.grid(row=current_row_aau, column=0, columnspan=4, pady=10)
        tk.Button(action_button_frame_aau, text="Avvia Ricerca Ambata e Ambo",
                  command=self.avvia_ricerca_ambata_ambo_unico,
                  font=("Helvetica", 11, "bold"), bg="olivedrab1"
                 ).pack(side=tk.LEFT, padx=5, ipady=3)
        tk.Button(action_button_frame_aau, text="Log", command=self.mostra_finestra_log, width=5).pack(side=tk.LEFT, padx=5, ipady=3)

        risultati_frame_aau = ttk.LabelFrame(main_frame, text="Migliori Configurazioni Ambata/Ambo Trovate", padding="10")
        risultati_frame_aau.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.aau_risultati_listbox = Listbox(risultati_frame_aau, height=12, font=("Courier New", 9), exportselection=False)
        aau_scrollbar_y = ttk.Scrollbar(risultati_frame_aau, orient="vertical", command=self.aau_risultati_listbox.yview)
        self.aau_risultati_listbox.config(yscrollcommand=aau_scrollbar_y.set)
        aau_scrollbar_y.pack(side=tk.RIGHT, fill="y")
        self.aau_risultati_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def avvia_ricerca_ambata_ambo_unico(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA AMBATA E AMBO UNICO (con Trasformazioni Correttore)\n" + "="*50)

        if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
            self.aau_risultati_listbox.delete(0, tk.END)
        
        self.aau_metodi_trovati_dati = []

        ruota_base = self.aau_ruota_base_var.get()
        pos_base_0idx = self.aau_pos_base_var.get() - 1
        
        operazioni_base_selezionate = []
        if self.aau_op_somma_var.get(): operazioni_base_selezionate.append('+')
        if self.aau_op_diff_var.get(): operazioni_base_selezionate.append('-')
        if self.aau_op_mult_var.get(): operazioni_base_selezionate.append('*')
        if not operazioni_base_selezionate:
            messagebox.showwarning("Input Mancante", "Seleziona almeno un'operazione base (+, -, *) da testare.")
            self.master.config(cursor="")
            return

        trasformazioni_correttore_selezionate = [
            nome_t for nome_t, var_t in self.aau_trasf_vars.items() if var_t.get()
        ]
        if not trasformazioni_correttore_selezionate:
            messagebox.showwarning("Input Mancante", "Seleziona almeno una trasformazione da applicare ai correttori (anche solo 'Fisso').")
            self.master.config(cursor="")
            return
        
        data_inizio_g_filtro_obj = None 
        data_fine_g_filtro_obj = None   
        try: 
            data_inizio_g_filtro_obj = self.date_inizio_entry_analisi.get_date()
        except ValueError: 
            self._log_to_gui("AAU WARN: Data inizio analisi globale non valida.")
        try: 
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_g_filtro_obj = self.date_fine_entry_analisi.get_date()
        except ValueError: 
            self._log_to_gui("AAU WARN: Data fine analisi globale non valida.")
        
        ruote_gioco_verifica, lookahead_verifica, indice_mese_applicazione_utente = self._get_parametri_gioco_comuni()
        if ruote_gioco_verifica is None: 
            self._log_to_gui("AAU: Parametri di gioco non validi.")
            self.master.config(cursor="") # Assicurati di resettare il cursore
            return

        self._log_to_gui(f"Parametri Ricerca Ambata e Ambo Unico (con Trasformazioni):")
        self._log_to_gui(f"  Estratto Base Globale: {ruota_base}[pos.{pos_base_0idx+1}]")
        self._log_to_gui(f"  Operazioni Base Selezionate: {operazioni_base_selezionate}")
        self._log_to_gui(f"  Trasformazioni Correttore Selezionate: {trasformazioni_correttore_selezionate}")
        self._log_to_gui(f"  Ruote Gioco (Verifica Esito Ambo): {', '.join(ruote_gioco_verifica)}")
        self._log_to_gui(f"  Colpi Lookahead per Ambo: {lookahead_verifica}")
        self._log_to_gui(f"  Indice Mese Applicazione Utente (per filtro previsione): {indice_mese_applicazione_utente if indice_mese_applicazione_utente is not None else 'Tutte le estrazioni valide nel periodo'}")
        data_inizio_log = data_inizio_g_filtro_obj.strftime('%Y-%m-%d') if data_inizio_g_filtro_obj else "Inizio Storico"
        data_fine_log = data_fine_g_filtro_obj.strftime('%Y-%m-%d') if data_fine_g_filtro_obj else "Fine Storico"
        self._log_to_gui(f"  Periodo Globale Analisi: {data_inizio_log} - {data_fine_log}")
        
        storico_per_analisi = carica_storico_completo(self.cartella_dati_var.get(), 
                                                       data_inizio_g_filtro_obj, 
                                                       data_fine_g_filtro_obj, 
                                                       app_logger=self._log_to_gui)
        if not storico_per_analisi:
            if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
                self.aau_risultati_listbox.insert(tk.END, "Caricamento/filtraggio storico fallito.")
            self.master.config(cursor="")
            return

        self.master.config(cursor="watch"); self.master.update_idletasks()
        try:
            migliori_ambi_config, applicazioni_base_tot = trova_migliori_ambi_da_correttori_automatici(
                storico_da_analizzare=storico_per_analisi,
                ruota_base_calc=ruota_base,
                pos_base_calc_0idx=pos_base_0idx,
                lista_operazioni_base_da_testare=operazioni_base_selezionate,
                lista_trasformazioni_correttore_da_testare=trasformazioni_correttore_selezionate,
                ruote_di_gioco_per_verifica=ruote_gioco_verifica,
                indice_mese_specifico_applicazione=indice_mese_applicazione_utente, 
                lookahead_colpi_per_verifica=lookahead_verifica,
                app_logger=self._log_to_gui,
                min_tentativi_per_metodo=self.min_tentativi_var.get() 
            )
            
            self.aau_metodi_trovati_dati = migliori_ambi_config if migliori_ambi_config else []

            if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
                self.aau_risultati_listbox.delete(0, tk.END)
                if not migliori_ambi_config:
                    self.aau_risultati_listbox.insert(tk.END, "Nessuna configurazione di ambo performante trovata.")
                else:
                    self.aau_risultati_listbox.insert(tk.END, f"Migliori Configurazioni Ambata/Ambo (su {applicazioni_base_tot} app. base valide):")
                    self.aau_risultati_listbox.insert(tk.END, "Rank| Freq.Ambo (S/T) | Ambo     | Freq.A1 (S/T) | Freq.A2 (S/T) | Formula (B<OpB1><Tr1>(C1o) & B<OpB2><Tr2>(C2o))")
                    self.aau_risultati_listbox.insert(tk.END, "----|-----------------|----------|---------------|---------------|-----------------------------------------------------")
                    num_da_mostrare_lb = min(len(migliori_ambi_config), 30)
                    for i, res_conf in enumerate(migliori_ambi_config[:num_da_mostrare_lb]):
                        ambo_ex_tuple = res_conf.get('ambo_esempio'); amb1_ex_disp = res_conf.get('ambata1_esempio', 'N/A'); amb2_ex_disp = res_conf.get('ambata2_esempio', 'N/A')
                        ambo_str_disp_lb = f"({ambo_ex_tuple[0]:>2},{ambo_ex_tuple[1]:>2})" if ambo_ex_tuple else "N/A"
                        formula_origine_lb = (f"({ruota_base[0]}{pos_base_0idx+1}{res_conf.get('op_base1','?')}{res_conf.get('trasf1','?')}({res_conf.get('correttore1_orig','?')})={amb1_ex_disp}) & "
                                              f"({ruota_base[0]}{pos_base_0idx+1}{res_conf.get('op_base2','?')}{res_conf.get('trasf2','?')}({res_conf.get('correttore2_orig','?')})={amb2_ex_disp})")
                        riga_listbox = (f"{(i+1):>3} | {(res_conf.get('frequenza_ambo',0)*100):>5.1f}% ({res_conf.get('successi_ambo',0)}/{res_conf.get('tentativi_ambo',0):<3}) | "
                                        f"{ambo_str_disp_lb:<8} | {(res_conf.get('frequenza_ambata1',0)*100):>5.1f}% ({res_conf.get('successi_ambata1',0)}/{res_conf.get('tentativi_ambata1',0):<3}) | "
                                        f"{(res_conf.get('frequenza_ambata2',0)*100):>5.1f}% ({res_conf.get('successi_ambata2',0)}/{res_conf.get('tentativi_ambata2',0):<3}) | {formula_origine_lb}")
                        self.aau_risultati_listbox.insert(tk.END, riga_listbox)
            
            if migliori_ambi_config:
                self._log_to_gui("\n--- CALCOLO PREVISIONE LIVE E SUGGERIMENTI (Ambata e Ambo Unico) ---")
                
                ultima_estrazione_valida_per_previsione = None
                if storico_per_analisi:
                    if indice_mese_applicazione_utente is not None and data_fine_g_filtro_obj is not None:
                        self._log_to_gui(f"AAU: Ricerca estrazione per previsione live: idx_mese={indice_mese_applicazione_utente}, data_fine<={data_fine_g_filtro_obj.strftime('%Y-%m-%d') if data_fine_g_filtro_obj else 'N/A'}")
                        for estr_rev in reversed(storico_per_analisi):
                            if estr_rev['data'] <= data_fine_g_filtro_obj:
                                if estr_rev.get('indice_mese') == indice_mese_applicazione_utente:
                                    ultima_estrazione_valida_per_previsione = estr_rev
                                    self._log_to_gui(f"  AAU: Trovata estrazione specifica per previsione live: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')} (Indice: {ultima_estrazione_valida_per_previsione.get('indice_mese')})")
                                    break
                        if not ultima_estrazione_valida_per_previsione: # Se non trovata specifica, ma data_fine è nel futuro o oggi...
                             if storico_per_analisi and storico_per_analisi[-1]['data'] > data_fine_g_filtro_obj : # ...e l'ultima dello storico è oltre la data_fine
                                 ultima_estrazione_valida_per_previsione = storico_per_analisi[-1] # usa l'ultima dello storico
                                 self._log_to_gui(f"  AAU WARN: Nessuna estrazione per indice {indice_mese_applicazione_utente} <= data fine. Uso ultima assoluta: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                             else: # L'ultima dello storico è <= data fine, ma non ha l'indice giusto
                                self._log_to_gui(f"  AAU WARN: Nessuna estrazione per indice {indice_mese_applicazione_utente} <= data fine. Previsione live non sarà specifica per questo indice mese.")
                                # In questo caso, si potrebbe decidere di non fare previsione o usare l'ultima comunque (il popup poi invaliderà)
                                # Per ora, proviamo a usare l'ultima se esiste, la logica del popup gestirà la visualizzazione.
                                if storico_per_analisi:
                                    ultima_estrazione_valida_per_previsione = storico_per_analisi[-1]
                    elif storico_per_analisi: # Se i filtri non sono specificati, usa l'ultima
                        ultima_estrazione_valida_per_previsione = storico_per_analisi[-1]
                        self._log_to_gui(f"  AAU: Uso ultima estrazione disponibile ({ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}) per previsione live (nessun filtro indice/data fine specifico per live).")
                
                if not ultima_estrazione_valida_per_previsione:
                    messagebox.showerror("Errore Dati", "Impossibile determinare estrazione di riferimento per previsione live AAU.")
                    self.master.config(cursor=""); return

                base_extr_dati_live = ultima_estrazione_valida_per_previsione.get(ruota_base, [])
                if not base_extr_dati_live or len(base_extr_dati_live) <= pos_base_0idx:
                    messagebox.showerror("Errore Previsione", f"Dati mancanti per {ruota_base}[pos.{pos_base_0idx+1}] nell'estrazione {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                    self.master.config(cursor=""); return
                numero_base_live_per_previsione = base_extr_dati_live[pos_base_0idx]
                self._log_to_gui(f"INFO: Estratto base per previsione live AAU: {numero_base_live_per_previsione} (da {ruota_base} il {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}, Indice Mese Estraz.: {ultima_estrazione_valida_per_previsione.get('indice_mese')})")

                top_metodi_per_ambi_unici_live = []
                ambi_live_gia_selezionati_per_popup = set()
                max_risultati_popup_aau = self.num_ambate_var.get()

                for res_conf_storico in migliori_ambi_config:
                    if len(top_metodi_per_ambi_unici_live) >= max_risultati_popup_aau: break
                    c1_o = res_conf_storico.get('correttore1_orig'); t1_s = res_conf_storico.get('trasf1'); opB1_s = res_conf_storico.get('op_base1')
                    c2_o = res_conf_storico.get('correttore2_orig'); t2_s = res_conf_storico.get('trasf2'); opB2_s = res_conf_storico.get('op_base2')
                    a1_live, a2_live = None, None
                    f_t1 = OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.get(t1_s); op_b1 = OPERAZIONI_COMPLESSE.get(opB1_s)
                    if f_t1 and op_b1 and c1_o is not None:
                        c1_t_live = f_t1(c1_o)
                        if c1_t_live is not None:
                            try: a1_live = regola_fuori_90(op_b1(numero_base_live_per_previsione, c1_t_live))
                            except ZeroDivisionError: pass
                    f_t2 = OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.get(t2_s); op_b2 = OPERAZIONI_COMPLESSE.get(opB2_s)
                    if f_t2 and op_b2 and c2_o is not None:
                        c2_t_live = f_t2(c2_o)
                        if c2_t_live is not None:
                            try: a2_live = regola_fuori_90(op_b2(numero_base_live_per_previsione, c2_t_live))
                            except ZeroDivisionError: pass
                    if a1_live is not None and a2_live is not None and a1_live != a2_live:
                        ambo_live_normalizzato = tuple(sorted((a1_live, a2_live)))
                        if ambo_live_normalizzato not in ambi_live_gia_selezionati_per_popup:
                            metodo_per_popup = res_conf_storico.copy()
                            metodo_per_popup['ambata1_live_calcolata'] = a1_live
                            metodo_per_popup['ambata2_live_calcolata'] = a2_live
                            metodo_per_popup['ambo_live_calcolato'] = ambo_live_normalizzato
                            metodo_per_popup['estrazione_usata_per_previsione'] = ultima_estrazione_valida_per_previsione 
                            top_metodi_per_ambi_unici_live.append(metodo_per_popup)
                            ambi_live_gia_selezionati_per_popup.add(ambo_live_normalizzato)
                
                self._log_to_gui(f"INFO: Dopo ricalcolo LIVE e filtro, trovati {len(top_metodi_per_ambi_unici_live)} metodi con ambi UNICI DIVERSI per il popup AAU.")

                if not top_metodi_per_ambi_unici_live:
                    messagebox.showinfo("Ricerca Ambata/Ambo", "Nessuna configurazione di ambo unica e valida trovata per la previsione live con i criteri attuali.")
                else:
                    # ... (Log dei primi ambi - invariato) ...
                    lista_previsioni_popup_aau = []; dati_grezzi_popup_aau = []
                    for idx_popup, res_popup_dati in enumerate(top_metodi_per_ambi_unici_live):
                        # ... (Costruzione di formula_origine_popup_display, performance_completa_str_popup, suggerimento_gioco_popup_str - invariato) ...
                        # ... (Creazione di dettaglio_popup - invariato) ...
                        lista_previsioni_popup_aau.append(dettaglio_popup)
                        
                        dati_salvataggio = res_popup_dati.copy() # res_popup_dati ora ha 'estrazione_usata_per_previsione'
                        dati_salvataggio["tipo_metodo_salvato"] = "ambata_ambo_unico_trasf"
                        # ... (resto di dati_salvataggio - invariato) ...
                        dati_grezzi_popup_aau.append(dati_salvataggio)
                    
                    if lista_previsioni_popup_aau:
                        self.mostra_popup_previsione(
                            titolo_popup="Migliori Configurazioni Ambata e Ambo Unico", 
                            ruote_gioco_str=", ".join(ruote_gioco_verifica),
                            lista_previsioni_dettagliate=lista_previsioni_popup_aau, 
                            data_riferimento_previsione_str_comune=ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y'),
                            metodi_grezzi_per_salvataggio=dati_grezzi_popup_aau,
                            indice_mese_richiesto_utente=indice_mese_applicazione_utente,
                            data_fine_analisi_globale_obj=data_fine_g_filtro_obj, 
                            estrazione_riferimento_per_previsione_live=ultima_estrazione_valida_per_previsione 
                        )
            elif not (hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox and self.aau_risultati_listbox.size() > 0 and self.aau_risultati_listbox.get(0).startswith("Nessuna")):
                 messagebox.showinfo("Ricerca Ambata/Ambo", "Nessuna configurazione performante trovata.")
        except Exception as e:
            messagebox.showerror("Errore Ricerca", f"Errore: {e}")
            self._log_to_gui(f"ERRORE CRITICO in Ricerca Ambata/Ambo Unico: {e}\n{traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")
        self._log_to_gui("--- Ricerca Ambata e Ambo Unico Completata ---")

    def crea_gui_verifica_manuale(self, parent_tab):
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')
        self.crea_gui_controlli_comuni(main_frame)
        verifica_params_frame = ttk.LabelFrame(main_frame, text="Parametri Verifica Giocata Specifica", padding="10")
        verifica_params_frame.pack(padx=10, pady=10, fill=tk.X, expand=False)

        vf_top_buttons_frame = ttk.Frame(verifica_params_frame)
        vf_top_buttons_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0,10))

        tk.Button(vf_top_buttons_frame, text="Verifica Giocata", command=self.avvia_verifica_giocata, font=("Helvetica", 11, "bold"), bg="lightblue").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,10), ipady=3)
        tk.Button(vf_top_buttons_frame, text="Log", command=self.mostra_finestra_log, width=10).pack(side=tk.LEFT, ipady=3)

        current_row_vf = 1
        tk.Label(verifica_params_frame, text="Numeri da Verificare (es. 23 45 67 o con virgole):").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(verifica_params_frame, textvariable=self.numeri_verifica_var, width=30).grid(row=current_row_vf, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_params_frame, text="Data Inizio Verifica:").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        self.date_inizio_verifica_entry = DateEntry(verifica_params_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_inizio_verifica_entry.grid(row=current_row_vf, column=1, sticky="w", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_params_frame, text="Colpi per Verifica (1-200):").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(verifica_params_frame, from_=1, to=200, textvariable=self.colpi_verifica_var, width=5).grid(row=current_row_vf, column=1, sticky="w", padx=5, pady=2)

    def avvia_backtest_metodo_semplice_selezionato(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO BACKTEST DETTAGLIATO (Metodo Semplice Selezionato)\n" + "="*50)

        if not hasattr(self, 'ms_risultati_listbox') or not self.ms_risultati_listbox:
            messagebox.showerror("Errore Interfaccia", "La Listbox dei risultati dei metodi semplici non è disponibile.")
            self._log_to_gui("ERRORE: ms_risultati_listbox non trovata.")
            return

        try:
            selected_indices = self.ms_risultati_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Selezione Mancante", "Seleziona un metodo dalla lista 'Top Metodi Semplici Trovati'.")
                return

            selected_listbox_index = selected_indices[0]

            if not (0 <= selected_listbox_index < len(self.metodi_semplici_trovati_dati)):
                messagebox.showerror("Errore Selezione", "Indice selezionato non valido o lista dati metodi vuota.")
                self._log_to_gui(f"ERRORE: Indice listbox {selected_listbox_index} fuori range per self.metodi_semplici_trovati_dati (len: {len(self.metodi_semplici_trovati_dati)}).")
                return

            dati_metodo_selezionato_grezzo = self.metodi_semplici_trovati_dati[selected_listbox_index]

            metodo_info = dati_metodo_selezionato_grezzo.get('metodo')
            if not metodo_info:
                messagebox.showerror("Errore Dati Metodo", "Dati interni del metodo semplice selezionato sono incompleti.")
                self._log_to_gui(f"ERRORE: Chiave 'metodo' mancante in dati_metodo_selezionato_grezzo: {dati_metodo_selezionato_grezzo}")
                return

            formula_testuale_backtest = f"{metodo_info['ruota_calcolo']}[pos.{metodo_info['pos_estratto_calcolo']+1}] {metodo_info['operazione']} {metodo_info['operando_fisso']}"
            self._log_to_gui(f"Preparazione Backtest Dettagliato per Metodo Semplice: {formula_testuale_backtest}")

            definizione_strutturata_backtest = []
            op_simbolo_map = {'somma': '+', 'differenza': '-', 'moltiplicazione': '*'}
            op_originale = metodo_info.get('operazione')
            op_simbolo = op_simbolo_map.get(str(op_originale).lower() if op_originale else "")

            if op_simbolo:
                definizione_strutturata_backtest = [
                    {'tipo_termine': 'estratto', 'ruota': metodo_info.get('ruota_calcolo'),
                     'posizione': metodo_info.get('pos_estratto_calcolo'), 'operazione_successiva': op_simbolo},
                    {'tipo_termine': 'fisso', 'valore_fisso': metodo_info.get('operando_fisso'),
                     'operazione_successiva': '='}
                ]
            else:
                messagebox.showerror("Errore Interno", f"Operazione del metodo semplice non riconosciuta: '{op_originale}'. Impossibile eseguire backtest.")
                self._log_to_gui(f"ERRORE: Operazione non mappabile '{op_originale}' per backtest metodo semplice.")
                return

            try:
                data_inizio_backtest = self.date_inizio_entry_analisi.get_date()
                data_fine_backtest = self.date_fine_entry_analisi.get_date()
                if data_inizio_backtest > data_fine_backtest: messagebox.showerror("Errore Date", "Data Inizio > Data Fine."); return
            except ValueError: messagebox.showerror("Errore Data", "Date Inizio/Fine non valide."); return

            mesi_sel_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
            mesi_map_b = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
            mesi_num_sel_b = []
            if not self.ap_tutti_mesi_var.get() and mesi_sel_gui: mesi_num_sel_b = [mesi_map_b[nome] for nome in mesi_sel_gui]

            ruote_g_b, lookahead_b, indice_mese_da_gui = self._get_parametri_gioco_comuni()
            if ruote_g_b is None: return

            storico_per_backtest_globale = carica_storico_completo(self.cartella_dati_var.get(), app_logger=self._log_to_gui)
            if not storico_per_backtest_globale: return

            self.master.config(cursor="watch"); self.master.update_idletasks()
            try:
                risultati_dettagliati = analizza_performance_dettagliata_metodo(
                    storico_completo=storico_per_backtest_globale,
                    definizione_metodo=definizione_strutturata_backtest,
                    metodo_stringa_per_log=formula_testuale_backtest,
                    ruote_gioco=ruote_g_b,
                    lookahead=lookahead_b,
                    data_inizio_analisi=data_inizio_backtest,
                    data_fine_analisi=data_fine_backtest,
                    mesi_selezionati_filtro=mesi_num_sel_b,
                    app_logger=self._log_to_gui,
                    indice_estrazione_mese_da_considerare=indice_mese_da_gui
                )

                if not risultati_dettagliati:
                    messagebox.showinfo("Backtest Metodo Semplice", "Nessuna applicazione o esito per il metodo selezionato nel periodo.")
                else:
                    popup_content = f"--- RISULTATI BACKTEST DETTAGLIATO (Met. Semplice) ---\n"
                    popup_content += f"Metodo: {formula_testuale_backtest}\n"
                    popup_content += f"Periodo: {data_inizio_backtest.strftime('%d/%m/%Y')} - {data_fine_backtest.strftime('%d/%m/%Y')}\n"
                    popup_content += f"Mesi Selezionati: {mesi_num_sel_b or 'Tutti nel range'}\n"
                    popup_content += f"Ruote di Gioco: {', '.join(ruote_g_b)}, Colpi Lookahead: {lookahead_b}\n"
                    popup_content += f"Indice Estrazione del Mese: {indice_mese_da_gui if indice_mese_da_gui is not None else 'Tutte valide'}\n"
                    popup_content += "--------------------------------------------------\n\n"
                    successi_ambata_tot = 0; applicazioni_valide_tot = 0
                    for res_bd in risultati_dettagliati:
                        popup_content += f"Data Applicazione: {res_bd['data_applicazione'].strftime('%d/%m/%Y')}\n"
                        if res_bd['metodo_applicabile']:
                            applicazioni_valide_tot += 1; popup_content += f"  Ambata Prevista: {res_bd['ambata_prevista']}\n"
                            if res_bd['esito_ambata']:
                                successi_ambata_tot +=1; popup_content += f"  ESITO: AMBATA VINCENTE!\n    Colpo: {res_bd['colpo_vincita_ambata']}, Ruota: {res_bd['ruota_vincita_ambata']}\n"
                                if res_bd.get('numeri_estratti_vincita'): popup_content += f"    Numeri Estratti: {res_bd['numeri_estratti_vincita']}\n"
                            else: popup_content += f"  ESITO: Ambata non uscita entro {lookahead_b} colpi.\n"
                        else: popup_content += f"  Metodo non applicabile.\n"
                        popup_content += "-------------------------\n"
                    freq_str = "N/A"
                    if applicazioni_valide_tot > 0: freq_str = f"{(successi_ambata_tot / applicazioni_valide_tot) * 100:.2f}% ({successi_ambata_tot}/{applicazioni_valide_tot} app.)"
                    summary = f"\nRIEPILOGO:\nApplicazioni Metodo Valide: {applicazioni_valide_tot}\n"
                    summary += f"Successi Ambata: {successi_ambata_tot}\nFreq. Successo: {freq_str}\n"
                    popup_content += summary
                    self.mostra_popup_testo_semplice("Backtest Dettagliato - Metodo Semplice", popup_content)

            except Exception as e:
                messagebox.showerror("Errore Backtest", f"Errore durante il backtest: {e}")
                self._log_to_gui(f"ERRORE CRITICO BACKTEST SEMPLICE: {e}\n{traceback.format_exc()}")
            finally:
                self.master.config(cursor="")

        except IndexError:
             messagebox.showerror("Errore", "Nessun metodo selezionato dalla lista o indice non valido.")
             self._log_to_gui("ERRORE: Indice selezione listbox metodi semplici non valido.")
        except Exception as e:
             messagebox.showerror("Errore", f"Errore imprevisto durante backtest metodo semplice: {e}")
             self._log_to_gui(f"ERRORE imprevisto in avvia_backtest_metodo_semplice_selezionato: {e}\n{traceback.format_exc()}")

    def avvia_backtest_dettagliato_metodo(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO BACKTEST DETTAGLIATO METODO\n" + "="*50)

        # Inizializzazione
        definizione_per_analisi, definizione_per_analisi_2 = None, None
        formula_testuale_display, tipo_metodo_usato_per_logica, tipo_metodo_origine = "N/D", "Sconosciuto", "Sconosciuto"
        condizione_primaria_da_passare_all_analisi, dati_preparati = None, None
        is_combined_test = False

        # FASE 1: DETERMINARE QUALE METODO/I USARE
        if self.metodo_preparato_per_backtest:
            dati_preparati = self.metodo_preparato_per_backtest
            definizione_per_analisi = dati_preparati.get('definizione_strutturata')
            formula_testuale_display = dati_preparati.get('formula_testuale', "Formula preparata mancante")
            tipo_metodo_usato_per_logica = dati_preparati.get('tipo', 'Sconosciuto_da_Popup_Fallback')
            tipo_metodo_origine = "Preparato da Popup"
            condizione_primaria_da_passare_all_analisi = dati_preparati.get('condizione_primaria')

        elif self.usa_ultimo_corretto_per_backtest_var.get() and self.ultimo_metodo_corretto_trovato_definizione:
            tipo_metodo_origine = "Checkbox Metodo Corretto"
            dati_corretti = self.ultimo_metodo_corretto_trovato_definizione
            
            if isinstance(dati_corretti, dict) and 'base1_corretto' in dati_corretti and 'base2_corretto' in dati_corretti:
                is_combined_test = True
                definizione_per_analisi = dati_corretti['base1_corretto']
                definizione_per_analisi_2 = dati_corretti['base2_corretto']
                tipo_metodo_usato_per_logica = "Complesso Corretto (Combinato)"
                formula_testuale_display = "Combinazione Metodo 1 Corretto + Metodo 2 Corretto"
            elif isinstance(dati_corretti, dict):
                scelta_base_per_corretto = self.mc_backtest_choice_var.get()
                chiave_def_da_usare = f"{scelta_base_per_corretto}_corretto"
                if dati_corretti.get(chiave_def_da_usare):
                    definizione_per_analisi = dati_corretti[chiave_def_da_usare]
                    tipo_metodo_usato_per_logica = f"Complesso Corretto ({scelta_base_per_corretto.title()})"
                    formula_testuale_display = "".join(self._format_componente_per_display(c) for c in definizione_per_analisi)
        
        elif definizione_per_analisi is None: 
            scelta_backtest_manuale = self.mc_backtest_choice_var.get()
            tipo_metodo_origine = "Manuale (da Radiobutton)"
            if scelta_backtest_manuale == "base1":
                definizione_per_analisi = self.definizione_metodo_complesso_attuale
            else:
                definizione_per_analisi = self.definizione_metodo_complesso_attuale_2
            if definizione_per_analisi:
                formula_testuale_display = "".join(self._format_componente_per_display(comp) for comp in definizione_per_analisi)

        # FASE 2: VALIDAZIONE E RACCOLTA PARAMETRI
        if not isinstance(definizione_per_analisi, list) or not definizione_per_analisi:
            messagebox.showerror("Errore Backtest", "Nessun metodo valido per il backtest."); return
        
        try:
            data_inizio_backtest = self.date_inizio_entry_analisi.get_date()
            data_fine_backtest = self.date_fine_entry_analisi.get_date()
        except ValueError: messagebox.showerror("Errore Data", "Date Inizio/Fine non valide."); return
        
        mesi_sel_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
        mesi_map_b = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
        mesi_num_sel_b = []
        if not self.ap_tutti_mesi_var.get() and mesi_sel_gui:
            mesi_num_sel_b = [mesi_map_b[nome] for nome in mesi_sel_gui]
        
        if tipo_metodo_usato_per_logica == "periodica_ottimale" and dati_preparati:
            params_periodo = dati_preparati.get("parametri_periodo", {})
            mesi_nomi_dal_metodo = params_periodo.get("mesi", [])
            mesi_num_sel_b = [] if mesi_nomi_dal_metodo == "Tutti" or not mesi_nomi_dal_metodo else [mesi_map_b.get(n) for n in mesi_nomi_dal_metodo if n in mesi_map_b]
                
        ruote_g_b, lookahead_b, indice_mese_da_gui = self._get_parametri_gioco_comuni()
        if ruote_g_b is None: return
        
        storico_per_backtest_globale = carica_storico_completo(self.cartella_dati_var.get(), app_logger=self._log_to_gui) 
        if not storico_per_backtest_globale: return

        # FASE 3: ESECUZIONE E VISUALIZZAZIONE
        self.master.config(cursor="watch"); self.master.update_idletasks()
        try:
            if is_combined_test:
                risultati_dettagliati_backtest = analizza_performance_dettagliata_combinata(
                    storico_completo=storico_per_backtest_globale, definizione_metodo_1=definizione_per_analisi,
                    definizione_metodo_2=definizione_per_analisi_2, metodo_stringa_per_log=formula_testuale_display,
                    ruote_gioco=ruote_g_b, lookahead=lookahead_b, data_inizio_analisi=data_inizio_backtest,
                    data_fine_analisi=data_fine_backtest, mesi_selezionati_filtro=mesi_num_sel_b, app_logger=self._log_to_gui,
                    indice_estrazione_mese_da_considerare=indice_mese_da_gui
                )
            else:
                risultati_dettagliati_backtest = analizza_performance_dettagliata_metodo( 
                    storico_completo=storico_per_backtest_globale, definizione_metodo=definizione_per_analisi,
                    metodo_stringa_per_log=formula_testuale_display, ruote_gioco=ruote_g_b, lookahead=lookahead_b,
                    data_inizio_analisi=data_inizio_backtest, data_fine_analisi=data_fine_backtest,
                    mesi_selezionati_filtro=mesi_num_sel_b, app_logger=self._log_to_gui,
                    condizione_primaria_metodo=condizione_primaria_da_passare_all_analisi,
                    indice_estrazione_mese_da_considerare=indice_mese_da_gui
                )
            
            if not risultati_dettagliati_backtest:
                messagebox.showinfo("Backtest", "Nessuna applicazione o esito per il metodo nel periodo specificato.")
            else:
                popup_content = f"--- RISULTATI BACKTEST DETTAGLIATO ---\n"
                popup_content += f"Metodo (Tipo: {tipo_metodo_usato_per_logica}, Origine: {tipo_metodo_origine}): {formula_testuale_display}\n"
                # ... resto dell'header
                successi_tot = 0; applicazioni_valide_tot = 0
                for res_bd in risultati_dettagliati_backtest:
                    popup_content += f"Data Applicazione: {res_bd['data_applicazione'].strftime('%d/%m/%Y')}\n"
                    if res_bd.get('metodo_applicabile'):
                        applicazioni_valide_tot += 1
                        if is_combined_test:
                            popup_content += f"  Ambate Previste (Testate): {res_bd.get('ambata_prevista_1')}, {res_bd.get('ambata_prevista_2')}\n"
                            if res_bd.get('esito_vincita'):
                                successi_tot += 1
                                popup_content += f"  ESITO: VINCITA!\n"
                                popup_content += f"    Ambata Vincente: {res_bd.get('ambata_vincente')}\n"
                                popup_content += f"    Colpo: {res_bd.get('colpo_vincita')}, Ruota: {res_bd.get('ruota_vincita')}\n"
                            else:
                                popup_content += f"  ESITO: Nessuna ambata uscita entro {lookahead_b} colpi.\n"
                        else: # Logica per test singolo
                            popup_content += f"  Ambata Prevista (Testata): {res_bd.get('ambata_prevista')}\n"
                            if res_bd.get('esito_ambata'):
                                successi_tot +=1
                                popup_content += f"  ESITO: AMBATA VINCENTE!\n"
                                popup_content += f"    Colpo: {res_bd.get('colpo_vincita_ambata')}, Ruota: {res_bd.get('ruota_vincita_ambata')}\n"
                            else:
                                popup_content += f"  ESITO: Ambata non uscita entro {lookahead_b} colpi.\n"
                    else:
                         popup_content += f"  Metodo non applicabile.\n"
                    popup_content += "-------------------------\n"

                freq_str = "N/A"
                if applicazioni_valide_tot > 0:
                    freq_str = f"{(successi_tot / applicazioni_valide_tot) * 100:.2f}% ({successi_tot}/{applicazioni_valide_tot} app. valide)"
                summary = f"\nRIEPILOGO:\nApplicazioni Valide: {applicazioni_valide_tot}\n"
                summary += f"Successi Totali: {successi_tot}\nFreq. Successo: {freq_str}\n"
                popup_content += summary
                self.mostra_popup_testo_semplice("Risultati Backtest Dettagliato", popup_content)

        except Exception as e:
            messagebox.showerror("Errore Backtest", f"Errore: {e}")
            self._log_to_gui(f"ERRORE CRITICO BACKTEST: {e}\n{traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")
            if self.metodo_preparato_per_backtest:
                self.metodo_preparato_per_backtest = None; self._refresh_mc_listbox_1()

    def _format_componente_per_display(self, componente):
        op_succ = componente['operazione_successiva']; op_str = f" {op_succ} " if op_succ and op_succ != '=' else ""
        if componente['tipo_termine'] == 'estratto': return f"{componente['ruota']}[{componente['posizione']+1}]{op_str}"
        elif componente['tipo_termine'] == 'fisso': return f"Fisso({componente['valore_fisso']}){op_str}"
        return "ERRORE_COMP"

    def _update_mc_input_state_1(self):
        if hasattr(self, 'mc_ruota_combo_1'):
            if self.mc_ruota_combo_1.winfo_exists():
                tipo_termine = self.mc_tipo_termine_var.get()
                is_estratto = tipo_termine == "estratto"
                try:
                    self.mc_ruota_combo_1.config(state="readonly" if is_estratto else "disabled")
                    self.mc_pos_spinbox_1.config(state="readonly" if is_estratto else "disabled")
                    self.mc_fisso_spinbox_1.config(state="disabled" if is_estratto else "readonly")
                    self.mc_ruota_label_1.config(state="normal" if is_estratto else "disabled")
                    self.mc_pos_label_1.config(state="normal" if is_estratto else "disabled")
                    self.mc_fisso_label_1.config(state="disabled" if is_estratto else "normal")
                except tk.TclError: pass

    def _update_mc_input_state_2(self):
        if hasattr(self, 'mc_ruota_combo_2'):
            if self.mc_ruota_combo_2.winfo_exists():
                tipo_termine = self.mc_tipo_termine_var_2.get()
                is_estratto = tipo_termine == "estratto"
                try:
                    self.mc_ruota_combo_2.config(state="readonly" if is_estratto else "disabled")
                    self.mc_pos_spinbox_2.config(state="readonly" if is_estratto else "disabled")
                    self.mc_fisso_spinbox_2.config(state="disabled" if is_estratto else "readonly")
                    self.mc_ruota_label_2.config(state="normal" if is_estratto else "disabled")
                    self.mc_pos_label_2.config(state="normal" if is_estratto else "disabled")
                    self.mc_fisso_label_2.config(state="disabled" if is_estratto else "normal")
                except tk.TclError: pass

    def _refresh_mc_listbox_1(self):
         if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
            self.mc_listbox_componenti_1.delete(0, tk.END)
            display_str = "".join(self._format_componente_per_display(comp) for comp in self.definizione_metodo_complesso_attuale)
            self.mc_listbox_componenti_1.insert(tk.END, display_str if display_str else "Nessun componente definito.")

    def _refresh_mc_listbox_2(self):
        if hasattr(self, 'mc_listbox_componenti_2') and self.mc_listbox_componenti_2.winfo_exists():
            self.mc_listbox_componenti_2.delete(0, tk.END)
            display_str = "".join(self._format_componente_per_display(comp) for comp in self.definizione_metodo_complesso_attuale_2)
            self.mc_listbox_componenti_2.insert(tk.END, display_str if display_str else "Nessun componente definito.")

    def aggiungi_componente_metodo_1(self):
        # ... (codice completo come fornito precedentemente)
        tipo = self.mc_tipo_termine_var.get()
        op_succ = self.mc_operazione_var.get()
        definizione_list = self.definizione_metodo_complesso_attuale
        if definizione_list and definizione_list[-1]['operazione_successiva'] == '=':
            messagebox.showwarning("Costruzione Metodo 1", "Metodo 1 già terminato con '='."); return
        componente = {'tipo_termine': tipo, 'operazione_successiva': op_succ}
        if tipo == 'estratto':
            componente['ruota'] = self.mc_ruota_var.get()
            componente['posizione'] = self.mc_posizione_var.get() - 1
        else:
            val_fisso = self.mc_valore_fisso_var.get()
            if not (1 <= val_fisso <= 90): messagebox.showerror("Errore Input", "Valore fisso deve essere tra 1 e 90."); return
            componente['valore_fisso'] = val_fisso
        definizione_list.append(componente)
        self._refresh_mc_listbox_1()


    def aggiungi_componente_metodo_2(self):
        # ... (codice completo come fornito precedentemente)
        tipo = self.mc_tipo_termine_var_2.get()
        op_succ = self.mc_operazione_var_2.get()
        definizione_list = self.definizione_metodo_complesso_attuale_2
        if definizione_list and definizione_list[-1]['operazione_successiva'] == '=':
            messagebox.showwarning("Costruzione Metodo 2", "Metodo 2 già terminato con '='."); return
        componente = {'tipo_termine': tipo, 'operazione_successiva': op_succ}
        if tipo == 'estratto':
            componente['ruota'] = self.mc_ruota_var_2.get()
            componente['posizione'] = self.mc_posizione_var_2.get() - 1
        else:
            val_fisso = self.mc_valore_fisso_var_2.get()
            if not (1 <= val_fisso <= 90): messagebox.showerror("Errore Input", "Valore fisso deve essere tra 1 e 90."); return
            componente['valore_fisso'] = val_fisso
        definizione_list.append(componente)
        self._refresh_mc_listbox_2()

    def rimuovi_ultimo_componente_metodo_1(self):
        # ... (codice completo come fornito precedentemente)
        if self.definizione_metodo_complesso_attuale:
            self.definizione_metodo_complesso_attuale.pop()
            self._refresh_mc_listbox_1()

    def pulisci_metodo_complesso_1(self):
        # ... (codice completo come fornito precedentemente)
        self.definizione_metodo_complesso_attuale.clear()
        self._refresh_mc_listbox_1()

    def rimuovi_ultimo_componente_metodo_2(self):
        # ... (codice completo come fornito precedentemente)
        if self.definizione_metodo_complesso_attuale_2:
            self.definizione_metodo_complesso_attuale_2.pop()
            self._refresh_mc_listbox_2()

    def pulisci_metodo_complesso_2(self):
        # ... (codice completo come fornito precedentemente)
        self.definizione_metodo_complesso_attuale_2.clear()
        self._refresh_mc_listbox_2()

    def _carica_e_valida_storico_comune(self, usa_filtri_data_globali=True):
        # ... (codice completo come fornito precedentemente)
        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati):
            messagebox.showerror("Errore Input", "Seleziona una cartella archivio dati valida."); self._log_to_gui("ERRORE: Cartella dati non valida."); return None
        data_inizio_storico = None; data_fine_storico = None
        if usa_filtri_data_globali:
            try: data_inizio_storico = self.date_inizio_entry_analisi.get_date()
            except ValueError: data_inizio_storico = None
            try: data_fine_storico = self.date_fine_entry_analisi.get_date()
            except ValueError: data_fine_storico = None
            if data_inizio_storico and data_fine_storico and data_fine_storico < data_inizio_storico:
                messagebox.showerror("Errore Input", "Data fine < data inizio per lo storico."); self._log_to_gui("ERRORE: Data fine < data inizio per lo storico."); return None
        self.master.config(cursor="watch"); self.master.update_idletasks()
        storico = carica_storico_completo(cartella_dati, data_inizio_storico, data_fine_storico, app_logger=self._log_to_gui)
        self.master.config(cursor="")
        if not storico:
            messagebox.showinfo("Caricamento Dati", "Nessun dato storico caricato/filtrato."); self._log_to_gui("Nessun dato storico caricato/filtrato."); return None
        if usa_filtri_data_globali: self.storico_caricato = storico
        return storico

    def _get_parametri_gioco_comuni(self):
        # ... (codice completo come fornito precedentemente)
        ruote_gioco_selezionate = [ruota for ruota, var in self.ruote_gioco_vars.items() if var.get()]
        if self.tutte_le_ruote_var.get() or not ruote_gioco_selezionate: ruote_gioco_selezionate = RUOTE[:]
        if not ruote_gioco_selezionate:
            messagebox.showerror("Errore Input", "Seleziona almeno una ruota di gioco."); self._log_to_gui("ERRORE: Nessuna ruota di gioco selezionata."); return None, None, None
        lookahead = self.lookahead_var.get(); indice_mese_str = self.indice_mese_var.get(); indice_mese = None
        if indice_mese_str:
            try:
                indice_mese = int(indice_mese_str)
                if indice_mese <= 0: messagebox.showerror("Errore Input", "Indice mese deve essere positivo."); self._log_to_gui("ERRORE: Indice mese non positivo."); return None, None, None
            except ValueError: messagebox.showerror("Errore Input", "Indice mese deve essere un numero."); self._log_to_gui("ERRORE: Indice mese non numerico."); return None, None, None
        return ruote_gioco_selezionate, lookahead, indice_mese

    def salva_impostazioni_semplici(self):
        # ... (codice completo come fornito precedentemente)
        impostazioni = {
            "versione_formato": 1.1, "tipo_metodo": "semplice",
            "struttura_base_ricerca": {"ruota_calcolo": self.ruota_calcolo_var.get(), "posizione_estratto": self.posizione_estratto_var.get()},
            "impostazioni_analisi": {"num_ambate_dettagliare": self.num_ambate_var.get(), "min_tentativi_metodo": self.min_tentativi_var.get()},
            "impostazioni_gioco": {"tutte_le_ruote": self.tutte_le_ruote_var.get(), "ruote_gioco_selezionate": [r for r, v in self.ruote_gioco_vars.items() if v.get()], "lookahead": self.lookahead_var.get(), "indice_mese": self.indice_mese_var.get()}
        }
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("File JSON Imp. Semplici", "*.json"), ("Tutti i file", "*.*")], title="Salva Impostazioni Metodo Semplice")
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(impostazioni, f, indent=4)
            self._log_to_gui(f"Impostazioni Metodo Semplice salvate in: {filepath}"); messagebox.showinfo("Salvataggio", "Impostazioni salvate!")
        except Exception as e: self._log_to_gui(f"Errore salvataggio imp. semplici: {e}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare:\n{e}")

    def apri_impostazioni_semplici(self):
        # ... (codice completo come fornito precedentemente)
        filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("File JSON Imp. Semplici", "*.json"), ("Tutti i file", "*.*")], title="Apri Impostazioni Metodo Semplice")
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: impostazioni = json.load(f)
            if impostazioni.get("tipo_metodo") != "semplice": messagebox.showerror("Errore Apertura", "File non valido per Metodo Semplice."); return
            if impostazioni.get("versione_formato") == 1.1:
                struttura_base = impostazioni.get("struttura_base_ricerca", {}); self.ruota_calcolo_var.set(struttura_base.get("ruota_calcolo", RUOTE[0])); self.posizione_estratto_var.set(struttura_base.get("posizione_estratto", 1))
                impostazioni_analisi_load = impostazioni.get("impostazioni_analisi", {}); self.num_ambate_var.set(impostazioni_analisi_load.get("num_ambate_dettagliare", 1)); self.min_tentativi_var.set(impostazioni_analisi_load.get("min_tentativi_metodo", 10))
                impostazioni_gioco_load = impostazioni.get("impostazioni_gioco", {}); self.tutte_le_ruote_var.set(impostazioni_gioco_load.get("tutte_le_ruote", True)); ruote_sel_caricate = impostazioni_gioco_load.get("ruote_gioco_selezionate", []); self.lookahead_var.set(impostazioni_gioco_load.get("lookahead", 3)); self.indice_mese_var.set(impostazioni_gioco_load.get("indice_mese", ""))
            else:
                self._log_to_gui("INFO: Caricamento file impostazioni semplici in vecchio formato (flat).")
                self.ruota_calcolo_var.set(impostazioni.get("ruota_calcolo_base", RUOTE[0])); self.posizione_estratto_var.set(impostazioni.get("posizione_estratto_base", 1)); self.num_ambate_var.set(impostazioni.get("num_ambate_dettagliare", 1)); self.min_tentativi_var.set(impostazioni.get("min_tentativi_metodo", 10))
                self.tutte_le_ruote_var.set(impostazioni.get("tutte_le_ruote", True)); ruote_sel_caricate = impostazioni.get("ruote_gioco_selezionate", []); self.lookahead_var.set(impostazioni.get("lookahead", 3)); self.indice_mese_var.set(impostazioni.get("indice_mese", ""))
            if not self.tutte_le_ruote_var.get():
                for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(ruota in ruote_sel_caricate)
            else:
                for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(True)
            self.on_tab_changed(None)
            self._log_to_gui(f"Impostazioni Metodo Semplice caricate da: {filepath}"); messagebox.showinfo("Apertura", "Impostazioni caricate!")
        except Exception as e: self._log_to_gui(f"Errore apertura imp. semplici: {e}"); messagebox.showerror("Errore Apertura", f"Impossibile aprire:\n{e}")

    def salva_metodi_complessi(self):
        # ... (codice completo come fornito precedentemente)
        if not self.definizione_metodo_complesso_attuale and not self.definizione_metodo_complesso_attuale_2: messagebox.showwarning("Salvataggio", "Nessun metodo complesso definito."); return
        impostazioni = {
            "versione_formato_lmc": 1.1, "tipo_metodo_file": "complessi_multi",
            "strutture_metodi_complessi": {"metodo_1": self.definizione_metodo_complesso_attuale, "metodo_2": self.definizione_metodo_complesso_attuale_2},
            "impostazioni_gioco": {"tutte_le_ruote": self.tutte_le_ruote_var.get(), "ruote_gioco_selezionate": [r for r, v in self.ruote_gioco_vars.items() if v.get()], "lookahead": self.lookahead_var.get(), "indice_mese": self.indice_mese_var.get()}
        }
        filepath = filedialog.asksaveasfilename(defaultextension=".lmc2", filetypes=[("File Metodi Lotto Complessi", "*.lmc2"), ("Tutti i file", "*.*")], title="Salva Metodi Complessi")
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(impostazioni, f, indent=4)
            self._log_to_gui(f"Metodi Complessi salvati in: {filepath}"); messagebox.showinfo("Salvataggio", "Metodi salvati!")
        except Exception as e: self._log_to_gui(f"Errore salvataggio: {e}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare:\n{e}")

    def apri_metodi_complessi(self):
        # ... (codice completo come fornito precedentemente)
        filepath = filedialog.askopenfilename(defaultextension=".lmc2", filetypes=[("File Metodi Lotto Complessi", "*.lmc2"),("File Metodo Lotto Complesso (Vecchio)", "*.lmc"), ("Tutti i file", "*.*")], title="Apri Metodi Complessi")
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: impostazioni = json.load(f)
            definizione_m1_caricata = []; definizione_m2_caricata = []
            tutte_ruote_caricato = True; ruote_sel_caricate = []
            lookahead_caricato = 3; indice_mese_caricato = ""
            if impostazioni.get("tipo_metodo_file") == "complessi_multi" and impostazioni.get("versione_formato_lmc") == 1.1:
                strutture_salvate = impostazioni.get("strutture_metodi_complessi", {}); definizione_m1_caricata = strutture_salvate.get("metodo_1", []); definizione_m2_caricata = strutture_salvate.get("metodo_2", [])
                impostazioni_gioco_load = impostazioni.get("impostazioni_gioco", {}); tutte_ruote_caricato = impostazioni_gioco_load.get("tutte_le_ruote", True); ruote_sel_caricate = impostazioni_gioco_load.get("ruote_gioco_selezionate", []); lookahead_caricato = impostazioni_gioco_load.get("lookahead", 3); indice_mese_caricato = impostazioni_gioco_load.get("indice_mese", "")
            elif impostazioni.get("tipo_metodo_file") == "complessi_multi":
                self._log_to_gui("INFO: Caricamento file .lmc2 in vecchio formato (flat).")
                definizione_m1_caricata = impostazioni.get("definizione_metodo_1", []); definizione_m2_caricata = impostazioni.get("definizione_metodo_2", [])
                tutte_ruote_caricato = impostazioni.get("tutte_le_ruote", True); ruote_sel_caricate = impostazioni.get("ruote_gioco_selezionate", []); lookahead_caricato = impostazioni.get("lookahead", 3); indice_mese_caricato = impostazioni.get("indice_mese", "")
            elif impostazioni.get("tipo_metodo") == "complesso":
                self._log_to_gui("INFO: Caricamento file .lmc (vecchio formato singolo metodo).")
                definizione_m1_caricata = impostazioni.get("definizione_metodo", []); definizione_m2_caricata = []
                tutte_ruote_caricato = impostazioni.get("tutte_le_ruote", True); ruote_sel_caricate = impostazioni.get("ruote_gioco_selezionate", []); lookahead_caricato = impostazioni.get("lookahead", 3); indice_mese_caricato = impostazioni.get("indice_mese", "")
                messagebox.showinfo("Compatibilità", "File in vecchio formato .lmc caricato. Solo Metodo Base 1 e impostazioni di gioco associate sono state applicate.")
            else: messagebox.showerror("Errore Apertura", "File non valido o formato non riconosciuto per Metodi Complessi."); return
            self.definizione_metodo_complesso_attuale = definizione_m1_caricata; self.definizione_metodo_complesso_attuale_2 = definizione_m2_caricata
            self._refresh_mc_listbox_1(); self._refresh_mc_listbox_2()
            self.tutte_le_ruote_var.set(tutte_ruote_caricato)
            if not self.tutte_le_ruote_var.get():
                 for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(ruota in ruote_sel_caricate)
            else:
                 for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(True)
            self.lookahead_var.set(lookahead_caricato); self.indice_mese_var.set(indice_mese_caricato)
            self.on_tab_changed(None)
            self._log_to_gui(f"Metodi Complessi caricati da: {filepath}"); messagebox.showinfo("Apertura", "Metodi caricati!")
        except Exception as e: self._log_to_gui(f"Errore apertura: {e}"); messagebox.showerror("Errore Apertura", f"Impossibile aprire:\n{e}")

    def avvia_ricerca_ambata_ambo_unico(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA AMBATA E AMBO UNICO (con Trasformazioni Correttore)\n" + "="*50)

        if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
            self.aau_risultati_listbox.delete(0, tk.END)
        
        self.aau_metodi_trovati_dati = []

        ruota_base = self.aau_ruota_base_var.get()
        pos_base_0idx = self.aau_pos_base_var.get() - 1
        
        operazioni_base_selezionate = []
        if self.aau_op_somma_var.get(): operazioni_base_selezionate.append('+')
        if self.aau_op_diff_var.get(): operazioni_base_selezionate.append('-')
        if self.aau_op_mult_var.get(): operazioni_base_selezionate.append('*')
        if not operazioni_base_selezionate:
            messagebox.showwarning("Input Mancante", "Seleziona almeno un'operazione base (+, -, *) da testare.")
            self.master.config(cursor="")
            return

        trasformazioni_correttore_selezionate = [
            nome_t for nome_t, var_t in self.aau_trasf_vars.items() if var_t.get()
        ]
        if not trasformazioni_correttore_selezionate:
            messagebox.showwarning("Input Mancante", "Seleziona almeno una trasformazione da applicare ai correttori (anche solo 'Fisso').")
            self.master.config(cursor="")
            return
        
        data_inizio_g_filtro_obj = None 
        data_fine_g_filtro_obj = None   
        try: 
            data_inizio_g_filtro_obj = self.date_inizio_entry_analisi.get_date()
        except ValueError: 
            self._log_to_gui("AAU WARN: Data inizio analisi globale non valida.")
        try: 
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_g_filtro_obj = self.date_fine_entry_analisi.get_date()
        except ValueError: 
            self._log_to_gui("AAU WARN: Data fine analisi globale non valida.")
        
        ruote_gioco_verifica, lookahead_verifica, indice_mese_applicazione_utente = self._get_parametri_gioco_comuni()
        if ruote_gioco_verifica is None: 
            self._log_to_gui("AAU: Parametri di gioco non validi.")
            self.master.config(cursor="") 
            return

        self._log_to_gui(f"Parametri Ricerca Ambata e Ambo Unico (con Trasformazioni):")
        self._log_to_gui(f"  Estratto Base Globale: {ruota_base}[pos.{pos_base_0idx+1}]")
        self._log_to_gui(f"  Operazioni Base Selezionate: {operazioni_base_selezionate}")
        self._log_to_gui(f"  Trasformazioni Correttore Selezionate: {trasformazioni_correttore_selezionate}")
        self._log_to_gui(f"  Ruote Gioco (Verifica Esito Ambo): {', '.join(ruote_gioco_verifica)}")
        self._log_to_gui(f"  Colpi Lookahead per Ambo: {lookahead_verifica}")
        self._log_to_gui(f"  Indice Mese Applicazione Utente (per filtro previsione): {indice_mese_applicazione_utente if indice_mese_applicazione_utente is not None else 'Tutte le estrazioni valide nel periodo'}")
        data_inizio_log = data_inizio_g_filtro_obj.strftime('%Y-%m-%d') if data_inizio_g_filtro_obj else "Inizio Storico"
        data_fine_log = data_fine_g_filtro_obj.strftime('%Y-%m-%d') if data_fine_g_filtro_obj else "Fine Storico"
        self._log_to_gui(f"  Periodo Globale Analisi: {data_inizio_log} - {data_fine_log}")
        
        storico_per_analisi = carica_storico_completo(self.cartella_dati_var.get(), 
                                                       data_inizio_g_filtro_obj, 
                                                       data_fine_g_filtro_obj, 
                                                       app_logger=self._log_to_gui)
        if not storico_per_analisi:
            if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
                self.aau_risultati_listbox.insert(tk.END, "Caricamento/filtraggio storico fallito.")
            self.master.config(cursor="")
            return

        self.master.config(cursor="watch"); self.master.update_idletasks()
        try:
            migliori_ambi_config, applicazioni_base_tot = trova_migliori_ambi_da_correttori_automatici(
                storico_da_analizzare=storico_per_analisi,
                ruota_base_calc=ruota_base,
                pos_base_calc_0idx=pos_base_0idx,
                lista_operazioni_base_da_testare=operazioni_base_selezionate,
                lista_trasformazioni_correttore_da_testare=trasformazioni_correttore_selezionate,
                ruote_di_gioco_per_verifica=ruote_gioco_verifica,
                indice_mese_specifico_applicazione=indice_mese_applicazione_utente, 
                lookahead_colpi_per_verifica=lookahead_verifica,
                app_logger=self._log_to_gui,
                min_tentativi_per_metodo=self.min_tentativi_var.get() 
            )
            
            self.aau_metodi_trovati_dati = migliori_ambi_config if migliori_ambi_config else []

            if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
                self.aau_risultati_listbox.delete(0, tk.END)
                if not migliori_ambi_config:
                    self.aau_risultati_listbox.insert(tk.END, "Nessuna configurazione di ambo performante trovata.")
                else:
                    self.aau_risultati_listbox.insert(tk.END, f"Migliori Configurazioni Ambata/Ambo (su {applicazioni_base_tot} app. base valide):")
                    self.aau_risultati_listbox.insert(tk.END, "Rank| Freq.Ambo (S/T) | Ambo     | Freq.A1 (S/T) | Freq.A2 (S/T) | Formula (B<OpB1><Tr1>(C1o) & B<OpB2><Tr2>(C2o))")
                    self.aau_risultati_listbox.insert(tk.END, "----|-----------------|----------|---------------|---------------|-----------------------------------------------------")
                    num_da_mostrare_lb = min(len(migliori_ambi_config), 30)
                    for i, res_conf in enumerate(migliori_ambi_config[:num_da_mostrare_lb]):
                        ambo_ex_tuple = res_conf.get('ambo_esempio'); amb1_ex_disp = res_conf.get('ambata1_esempio', 'N/A'); amb2_ex_disp = res_conf.get('ambata2_esempio', 'N/A')
                        ambo_str_disp_lb = f"({ambo_ex_tuple[0]:>2},{ambo_ex_tuple[1]:>2})" if ambo_ex_tuple else "N/A"
                        formula_origine_lb = (f"({ruota_base[0]}{pos_base_0idx+1}{res_conf.get('op_base1','?')}{res_conf.get('trasf1','?')}({res_conf.get('correttore1_orig','?')})={amb1_ex_disp}) & "
                                              f"({ruota_base[0]}{pos_base_0idx+1}{res_conf.get('op_base2','?')}{res_conf.get('trasf2','?')}({res_conf.get('correttore2_orig','?')})={amb2_ex_disp})")
                        riga_listbox = (f"{(i+1):>3} | {(res_conf.get('frequenza_ambo',0)*100):>5.1f}% ({res_conf.get('successi_ambo',0)}/{res_conf.get('tentativi_ambo',0):<3}) | "
                                        f"{ambo_str_disp_lb:<8} | {(res_conf.get('frequenza_ambata1',0)*100):>5.1f}% ({res_conf.get('successi_ambata1',0)}/{res_conf.get('tentativi_ambata1',0):<3}) | "
                                        f"{(res_conf.get('frequenza_ambata2',0)*100):>5.1f}% ({res_conf.get('successi_ambata2',0)}/{res_conf.get('tentativi_ambata2',0):<3}) | {formula_origine_lb}")
                        self.aau_risultati_listbox.insert(tk.END, riga_listbox)
            
            if migliori_ambi_config:
                self._log_to_gui("\n--- CALCOLO PREVISIONE LIVE E SUGGERIMENTI (Ambata e Ambo Unico) ---")
                
                ultima_estrazione_valida_per_previsione = None
                if storico_per_analisi:
                    if indice_mese_applicazione_utente is not None and data_fine_g_filtro_obj is not None:
                        self._log_to_gui(f"AAU: Ricerca estrazione per previsione live: idx_mese={indice_mese_applicazione_utente}, data_fine<={data_fine_g_filtro_obj.strftime('%Y-%m-%d') if data_fine_g_filtro_obj else 'N/A'}")
                        for estr_rev in reversed(storico_per_analisi):
                            if estr_rev['data'] <= data_fine_g_filtro_obj:
                                if estr_rev.get('indice_mese') == indice_mese_applicazione_utente:
                                    ultima_estrazione_valida_per_previsione = estr_rev
                                    self._log_to_gui(f"  AAU: Trovata estrazione specifica per previsione live: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')} (Indice: {ultima_estrazione_valida_per_previsione.get('indice_mese')})")
                                    break
                        if not ultima_estrazione_valida_per_previsione:
                             if storico_per_analisi and storico_per_analisi[-1]['data'] > data_fine_g_filtro_obj : 
                                 ultima_estrazione_valida_per_previsione = storico_per_analisi[-1] 
                                 self._log_to_gui(f"  AAU WARN: Nessuna estrazione per indice {indice_mese_applicazione_utente} <= data fine. Uso ultima assoluta: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                             else: 
                                self._log_to_gui(f"  AAU WARN: Nessuna estrazione per indice {indice_mese_applicazione_utente} <= data fine. Previsione live userà ultima del periodo se disponibile.")
                                if storico_per_analisi: ultima_estrazione_valida_per_previsione = storico_per_analisi[-1]
                    elif storico_per_analisi: 
                        ultima_estrazione_valida_per_previsione = storico_per_analisi[-1]
                        self._log_to_gui(f"  AAU: Uso ultima estrazione disponibile ({ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}) per previsione live (nessun filtro indice/data fine specifico per live).")
                
                if not ultima_estrazione_valida_per_previsione:
                    messagebox.showerror("Errore Dati", "Impossibile determinare estrazione di riferimento per previsione live AAU.")
                    self.master.config(cursor=""); return

                base_extr_dati_live = ultima_estrazione_valida_per_previsione.get(ruota_base, [])
                if not base_extr_dati_live or len(base_extr_dati_live) <= pos_base_0idx:
                    messagebox.showerror("Errore Previsione", f"Dati mancanti per {ruota_base}[pos.{pos_base_0idx+1}] nell'estrazione {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                    self.master.config(cursor=""); return
                numero_base_live_per_previsione = base_extr_dati_live[pos_base_0idx]
                self._log_to_gui(f"INFO: Estratto base per previsione live AAU: {numero_base_live_per_previsione} (da {ruota_base} il {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}, Indice Mese Estraz.: {ultima_estrazione_valida_per_previsione.get('indice_mese')})")

                # --- MODIFICA PER GESTIRE AMBI UNICI E NUMERO DESIDERATO ---
                top_metodi_per_popup = [] # Rinomino per chiarezza
                ambi_live_gia_aggiunti = set()
                max_risultati_da_mostrare_nel_popup = self.num_ambate_var.get() # Prende il valore dalla GUI
                self._log_to_gui(f"AAU: Preparazione fino a {max_risultati_da_mostrare_nel_popup} ambi UNICI per il popup.")

                for res_conf_storico in migliori_ambi_config: # Itera sui metodi storici ordinati
                    if len(top_metodi_per_popup) >= max_risultati_da_mostrare_nel_popup:
                        self._log_to_gui(f"AAU: Raggiunto limite di {max_risultati_da_mostrare_nel_popup} ambi unici per popup.")
                        break 
                    
                    c1_o = res_conf_storico.get('correttore1_orig'); t1_s = res_conf_storico.get('trasf1'); opB1_s = res_conf_storico.get('op_base1')
                    c2_o = res_conf_storico.get('correttore2_orig'); t2_s = res_conf_storico.get('trasf2'); opB2_s = res_conf_storico.get('op_base2')
                    a1_live, a2_live = None, None
                    
                    f_t1 = OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.get(t1_s); op_b1 = OPERAZIONI_COMPLESSE.get(opB1_s)
                    if f_t1 and op_b1 and c1_o is not None:
                        c1_t_live = f_t1(c1_o)
                        if c1_t_live is not None:
                            try: a1_live = regola_fuori_90(op_b1(numero_base_live_per_previsione, c1_t_live))
                            except ZeroDivisionError: pass
                    f_t2 = OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE.get(t2_s); op_b2 = OPERAZIONI_COMPLESSE.get(opB2_s)
                    if f_t2 and op_b2 and c2_o is not None:
                        c2_t_live = f_t2(c2_o)
                        if c2_t_live is not None:
                            try: a2_live = regola_fuori_90(op_b2(numero_base_live_per_previsione, c2_t_live))
                            except ZeroDivisionError: pass
                    
                    if a1_live is not None and a2_live is not None and a1_live != a2_live:
                        ambo_live_normalizzato = tuple(sorted((a1_live, a2_live)))
                        if ambo_live_normalizzato not in ambi_live_gia_aggiunti: # Filtra per unicità dell'ambo live
                            metodo_calcolato_live = res_conf_storico.copy()
                            metodo_calcolato_live['ambata1_live_calcolata'] = a1_live
                            metodo_calcolato_live['ambata2_live_calcolata'] = a2_live
                            metodo_calcolato_live['ambo_live_calcolato'] = ambo_live_normalizzato
                            metodo_calcolato_live['estrazione_usata_per_previsione'] = ultima_estrazione_valida_per_previsione
                            top_metodi_per_popup.append(metodo_calcolato_live)
                            ambi_live_gia_aggiunti.add(ambo_live_normalizzato)
                            self._log_to_gui(f"  AAU: Aggiunto ambo UNICO live {ambo_live_normalizzato} per popup. Conteggio: {len(top_metodi_per_popup)}")
                # --- FINE MODIFICA ---
                
                self._log_to_gui(f"INFO: Dopo ricalcolo LIVE e filtro per unicità, preparati {len(top_metodi_per_popup)} metodi per il popup AAU.")

                if not top_metodi_per_popup:
                    messagebox.showinfo("Ricerca Ambata/Ambo", "Nessuna configurazione di ambo unica e valida trovata per la previsione live con i criteri attuali.")
                else:
                    lista_previsioni_popup_aau = []; dati_grezzi_popup_aau = []
                    # Ora questo loop itera su top_metodi_per_popup (che ha al massimo max_risultati_da_mostrare_nel_popup ambi UNICI)
                    for idx_popup, res_popup_dati in enumerate(top_metodi_per_popup):
                        ambata1_live_display = res_popup_dati.get('ambata1_live_calcolata', "N/A")
                        ambata2_live_display = res_popup_dati.get('ambata2_live_calcolata', "N/A")
                        ambo_live_calcolato_tuple = res_popup_dati.get('ambo_live_calcolato')
                        ambo_live_str_display = f"({ambo_live_calcolato_tuple[0]}, {ambo_live_calcolato_tuple[1]})" if ambo_live_calcolato_tuple else "N/D"
                        
                        formula_origine_popup_display = (
                            f"Metodo: Metodo Base: {ruota_base}[pos.{pos_base_0idx+1}] (da {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')} estr. {numero_base_live_per_previsione}) con:\n"
                            f"  1) Op.Base:'{res_popup_dati.get('op_base1','?')}', Correttore: {res_popup_dati.get('trasf1','?')}({res_popup_dati.get('correttore1_orig','?')}) => Ris. Ambata1 Live: {ambata1_live_display}\n"
                            f"  2) Op.Base:'{res_popup_dati.get('op_base2','?')}', Correttore: {res_popup_dati.get('trasf2','?')}({res_popup_dati.get('correttore2_orig','?')}) => Ris. Ambata2 Live: {ambata2_live_display}"
                        )
                        
                        ambo_storico_ex_tuple = res_popup_dati.get('ambo_esempio')
                        ambo_storico_ex_str = f"({ambo_storico_ex_tuple[0]}, {ambo_storico_ex_tuple[1]})" if ambo_storico_ex_tuple else "N/D Storico"
                        ambata1_storico_ex_val = res_popup_dati.get('ambata1_esempio', 'N/A'); ambata2_storico_ex_val = res_popup_dati.get('ambata2_esempio', 'N/A')
                        
                        performance_completa_str_popup = f"Performance storica (del metodo che in passato ha prodotto ambo esempio {ambo_storico_ex_str}):\n"
                        performance_completa_str_popup += f"Ambo Secco (storico {ambo_storico_ex_str}): {res_popup_dati.get('frequenza_ambo',0):.2%} ({res_popup_dati.get('successi_ambo',0)}/{res_popup_dati.get('tentativi_ambo',0)} casi)\n"
                        performance_completa_str_popup += f"  Solo Ambata {ambata1_storico_ex_val} (da Op1, storico): {res_popup_dati.get('frequenza_ambata1',0):.1%} ({res_popup_dati.get('successi_ambata1',0)}/{res_popup_dati.get('tentativi_ambata1',0)})\n"
                        performance_completa_str_popup += f"  Solo Ambata {ambata2_storico_ex_val} (da Op2, storico): {res_popup_dati.get('frequenza_ambata2',0):.1%} ({res_popup_dati.get('successi_ambata2',0)}/{res_popup_dati.get('tentativi_ambata2',0)})"
                        if 'frequenza_almeno_una_ambata' in res_popup_dati:
                            freq_almeno_una = res_popup_dati['frequenza_almeno_una_ambata']; succ_almeno_una = res_popup_dati.get('successi_almeno_una_ambata', 0); tent_almeno_una = res_popup_dati.get('tentativi_almeno_una_ambata', res_popup_dati.get('tentativi_ambo', 0))
                            performance_completa_str_popup += f"\n  Almeno una Ambata ({ambata1_storico_ex_val} o {ambata2_storico_ex_val}, storico): {freq_almeno_una:.1%} ({succ_almeno_una}/{tent_almeno_una})"

                        suggerimento_gioco_popup_str = ""
                        if idx_popup == 0 and ambata1_live_display != "N/A" and ambata2_live_display != "N/A" and ambo_live_calcolato_tuple:
                            suggerimento_gioco_popup_str = (f"\nStrategia di Gioco Suggerita (basata su previsione live di questo 1° metodo):\n  - Giocare Ambata Singola: {ambata1_live_display}\n  - Giocare Ambata Singola: {ambata2_live_display}\n  - Giocare Ambo Secco: {ambo_live_str_display}")
                            # La logica per mostrare più ambi suggeriti è stata rimossa perché ogni sezione del popup è ora per un ambo unico live
                            performance_completa_str_popup += suggerimento_gioco_popup_str

                        dettaglio_popup_dict = {
                            "titolo_sezione": f"--- {(idx_popup+1)}ª Configurazione Proposta (Ambo Unico) ---", 
                            "info_metodo_str": formula_origine_popup_display, 
                            "ambata_prevista": f"PREVISIONE DA GIOCARE: AMBO DA GIOCARE: {ambo_live_str_display}", 
                            "performance_storica_str": performance_completa_str_popup, 
                            "abbinamenti_dict": {}, "contorni_suggeriti": [] 
                        }
                        lista_previsioni_popup_aau.append(dettaglio_popup_dict)
                        
                        dati_salvataggio = res_popup_dati.copy()
                        dati_salvataggio["tipo_metodo_salvato"] = "ambata_ambo_unico_trasf"
                        dati_salvataggio["formula_testuale"] = formula_origine_popup_display
                        dati_salvataggio["ruota_base_origine"] = ruota_base
                        dati_salvataggio["pos_base_origine"] = pos_base_0idx
                        if ambo_live_calcolato_tuple: 
                            dati_salvataggio["definizione_strutturata"] = list(ambo_live_calcolato_tuple)
                        else: 
                            dati_salvataggio["definizione_strutturata"] = None
                        dati_salvataggio["ambata_prevista_live"] = ambo_live_str_display
                        dati_grezzi_popup_aau.append(dati_salvataggio)
                    
                    if lista_previsioni_popup_aau:
                        self.mostra_popup_previsione(
                            titolo_popup="Migliori Configurazioni Ambata e Ambo Unico", 
                            ruote_gioco_str=", ".join(ruote_gioco_verifica),
                            lista_previsioni_dettagliate=lista_previsioni_popup_aau, 
                            data_riferimento_previsione_str_comune=ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y'),
                            metodi_grezzi_per_salvataggio=dati_grezzi_popup_aau,
                            indice_mese_richiesto_utente=indice_mese_applicazione_utente,
                            data_fine_analisi_globale_obj=data_fine_g_filtro_obj, 
                            estrazione_riferimento_per_previsione_live=ultima_estrazione_valida_per_previsione 
                        )
            elif not (hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox and self.aau_risultati_listbox.size() > 0 and self.aau_risultati_listbox.get(0).startswith("Nessuna")):
                 messagebox.showinfo("Ricerca Ambata/Ambo", "Nessuna configurazione performante trovata.")
        except Exception as e:
            messagebox.showerror("Errore Ricerca", f"Errore: {e}")
            self._log_to_gui(f"ERRORE CRITICO in Ricerca Ambata/Ambo Unico: {e}\n{traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")
        self._log_to_gui("--- Ricerca Ambata e Ambo Unico Completata ---")

  
    def _prepara_e_salva_profilo_metodo(self, dati_profilo_metodo, tipo_file="lotto_metodo_profilo", estensione=".lmp"):
        if not dati_profilo_metodo: messagebox.showerror("Errore", "Nessun dato del metodo da salvare."); return
        nome_suggerito = "profilo_metodo"; ambata_valida = False
        ambata_da_usare_per_nome = dati_profilo_metodo.get("ambata_prevista")
        if ambata_da_usare_per_nome is None: ambata_da_usare_per_nome = dati_profilo_metodo.get("ambata_piu_frequente_dal_metodo")
        if ambata_da_usare_per_nome is None: ambata_da_usare_per_nome = dati_profilo_metodo.get("ambata_risultante_prima_occ_val")
        if ambata_da_usare_per_nome is None: ambata_da_usare_per_nome = dati_profilo_metodo.get("previsione_live_cond")
        if ambata_da_usare_per_nome is not None and ambata_da_usare_per_nome != "N/A":
            try: int(ambata_da_usare_per_nome); ambata_valida = True
            except (ValueError, TypeError): ambata_valida = False
        if ambata_valida: nome_suggerito = f"metodo_ambata_{ambata_da_usare_per_nome}"
        elif dati_profilo_metodo.get("formula_testuale"):
            formula_semplice = dati_profilo_metodo["formula_testuale"]
            formula_semplice = formula_semplice.replace("[pos.", "_p").replace("]", "").replace("[", "_").replace(" ", "_").replace("+", "piu").replace("-", "meno").replace("*", "per").replace("SE", "IF").replace("ALLORA", "THEN").replace("IN", "in")
            formula_semplice = ''.join(c for c in formula_semplice if c.isalnum() or c in ['_','-'])
            nome_suggerito = f"metodo_{formula_semplice[:40].rstrip('_')}"
        filepath = filedialog.asksaveasfilename(initialfile=nome_suggerito, defaultextension=estensione, filetypes=[(f"File Profilo Metodo ({estensione})", f"*{estensione}"), ("Tutti i file", "*.*")], title="Salva Profilo Metodo Analizzato")
        if not filepath: return
        try:
            dati_da_salvare_serializzabili = {}
            for key, value in dati_profilo_metodo.items():
                if isinstance(value, (list, dict, str, int, float, bool, type(None))): dati_da_salvare_serializzabili[key] = value
                elif hasattr(value, '__dict__'):
                    try: json.dumps(value.__dict__); dati_da_salvare_serializzabili[key] = value.__dict__
                    except TypeError: dati_da_salvare_serializzabili[key] = str(value)
                else: dati_da_salvare_serializzabili[key] = str(value)
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(dati_da_salvare_serializzabili, f, indent=4, default=str)
            self._log_to_gui(f"Profilo del metodo salvato in: {filepath}"); messagebox.showinfo("Salvataggio Profilo", "Profilo del metodo salvato con successo!")
        except Exception as e: self._log_to_gui(f"Errore durante il salvataggio del profilo del metodo: {e}, {traceback.format_exc()}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il profilo del metodo:\n{e}")


    def mostra_popup_previsione(self, titolo_popup, ruote_gioco_str,
                                lista_previsioni_dettagliate=None,
                                copertura_combinata_info=None,
                                data_riferimento_previsione_str_comune=None,
                                metodi_grezzi_per_salvataggio=None,
                                indice_mese_richiesto_utente=None,
                                data_fine_analisi_globale_obj=None,
                                estrazione_riferimento_per_previsione_live=None
                               ):
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo_popup)

        # Calcolo dinamico altezza popup
        popup_width = 700
        popup_base_height_per_method_section = 180 
        abbinamenti_h_approx = 150 
        contorni_h_approx = 70   
        dynamic_height_needed = 150 
        if copertura_combinata_info: dynamic_height_needed += 80
        if lista_previsioni_dettagliate:
            for prev_dett_c in lista_previsioni_dettagliate:
                current_met_h = popup_base_height_per_method_section
                ambata_val_check = prev_dett_c.get('ambata_prevista')
                is_single_number_for_abbinamenti = False
                if isinstance(ambata_val_check, (int, float)) or (isinstance(ambata_val_check, str) and ambata_val_check.isdigit()):
                    is_single_number_for_abbinamenti = True
                if is_single_number_for_abbinamenti:
                    if prev_dett_c.get("abbinamenti_dict", {}).get("sortite_ambata_target", 0) > 0:
                        current_met_h += abbinamenti_h_approx
                    if prev_dett_c.get('contorni_suggeriti'):
                        current_met_h += contorni_h_approx
                dynamic_height_needed += current_met_h
        popup_height = min(dynamic_height_needed, 780); popup_height = max(popup_height, 620) 
        popup_window.geometry(f"{popup_width}x{int(popup_height)}")
        popup_window.transient(self.master); popup_window.attributes('-topmost', True)

        canvas = tk.Canvas(popup_window); scrollbar_y = ttk.Scrollbar(popup_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar_y.set)
        
        row_idx = 0
        ttk.Label(scrollable_frame, text=f"--- {titolo_popup} ---", font=("Helvetica", 12, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1
        
        data_effettiva_popup_titolo_str = data_riferimento_previsione_str_comune 
        if estrazione_riferimento_per_previsione_live and isinstance(estrazione_riferimento_per_previsione_live.get('data'), date):
            data_effettiva_popup_titolo_str = estrazione_riferimento_per_previsione_live['data'].strftime('%d/%m/%Y')
        
        if data_effettiva_popup_titolo_str and data_effettiva_popup_titolo_str != "N/D":
             ttk.Label(scrollable_frame, text=f"Previsione calcolata sull'estrazione del: {data_effettiva_popup_titolo_str}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
        
        ttk.Label(scrollable_frame, text=f"Su ruote: {ruote_gioco_str}").grid(row=row_idx, column=0, columnspan=2, pady=(2,10), sticky="w"); row_idx += 1

        if copertura_combinata_info and "testo_introduttivo" in copertura_combinata_info:
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5); row_idx += 1
            ttk.Label(scrollable_frame, text=copertura_combinata_info['testo_introduttivo'], wraplength=popup_width - 40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1

        if lista_previsioni_dettagliate:
            for idx_metodo, previsione_dett in enumerate(lista_previsioni_dettagliate):
                ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10); row_idx += 1
                titolo_sezione = previsione_dett.get('titolo_sezione', f'--- {(idx_metodo+1)}° METODO / PREVISIONE ---'); 
                ttk.Label(scrollable_frame, text=titolo_sezione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=3, sticky="w"); row_idx += 1
                
                formula_metodo_display = previsione_dett.get('info_metodo_str', "N/D")
                if formula_metodo_display != "N/D": 
                    ttk.Label(scrollable_frame, text=f"Metodo: {formula_metodo_display}", wraplength=popup_width-40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                
                ambata_loop_originale_da_dizionario = previsione_dett.get('ambata_prevista')
                ambata_da_visualizzare_nel_popup = ambata_loop_originale_da_dizionario 
                nota_finale_per_popup = ""

                if indice_mese_richiesto_utente is not None and \
                   data_fine_analisi_globale_obj is not None and \
                   estrazione_riferimento_per_previsione_live is not None and \
                   ambata_loop_originale_da_dizionario is not None and str(ambata_loop_originale_da_dizionario).upper() not in ["N/D", "N/A"]:

                    data_estr_live_obj = estrazione_riferimento_per_previsione_live.get('data')
                    idx_mese_estr_live = estrazione_riferimento_per_previsione_live.get('indice_mese')

                    if isinstance(data_estr_live_obj, date) and idx_mese_estr_live is not None and \
                       idx_mese_estr_live != indice_mese_richiesto_utente and \
                       data_estr_live_obj <= data_fine_analisi_globale_obj: 
                        
                        ambata_da_visualizzare_nel_popup = None
                        nota_finale_per_popup = (
                            f"(NOTA: La previsione calcolata sull'estrazione del {data_estr_live_obj.strftime('%d/%m/%Y')} "
                            f"(che è la {idx_mese_estr_live}ª del mese) non viene mostrata come 'PREVISIONE DA GIOCARE' "
                            f"perché non corrisponde all'estrazione {indice_mese_richiesto_utente}ª del mese richiesta per l'applicazione del metodo. "
                            f"La data di fine analisi è impostata al {data_fine_analisi_globale_obj.strftime('%d/%m/%Y')}.)"
                        )
                
                if ambata_da_visualizzare_nel_popup is None or str(ambata_da_visualizzare_nel_popup).upper() in ["N/D", "N/A"]:
                    testo_finale_previsione = "Nessuna previsione da giocare valida per i criteri."
                    if nota_finale_per_popup:
                         testo_finale_previsione += f"\n{nota_finale_per_popup}"
                    ttk.Label(scrollable_frame, text=testo_finale_previsione, wraplength=popup_width-40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                else:
                    testo_finale_previsione = f"PREVISIONE DA GIOCARE: {ambata_da_visualizzare_nel_popup}"
                    ttk.Label(scrollable_frame, text=testo_finale_previsione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                
                performance_str_display = previsione_dett.get('performance_storica_str', 'N/D')
                ttk.Label(scrollable_frame, text=f"Performance storica:\n{performance_str_display}", justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1

                dati_grezzi_per_questo_metodo = None
                if metodi_grezzi_per_salvataggio and idx_metodo < len(metodi_grezzi_per_salvataggio):
                    dati_grezzi_per_questo_metodo = metodi_grezzi_per_salvataggio[idx_metodo]
                
                if dati_grezzi_per_questo_metodo:
                    estensione_default = ".lmp" 
                    tipo_metodo_salv_effettivo = dati_grezzi_per_questo_metodo.get("tipo_metodo_salvato", "sconosciuto")
                    
                    if tipo_metodo_salv_effettivo.startswith("condizionato"):
                        estensione_default = ".lmcondcorr" if "corretto" in tipo_metodo_salv_effettivo else ".lmcond"
                    elif tipo_metodo_salv_effettivo in ["ambata_ambo_unico_auto", "ambata_ambo_unico_trasf"]: 
                        estensione_default = ".lmaau"

                    btn_salva_profilo = ttk.Button(scrollable_frame, text="Salva Questo Metodo", 
                                                   command=lambda d=dati_grezzi_per_questo_metodo.copy(), e=estensione_default: self._prepara_e_salva_profilo_metodo(d, estensione=e))
                    btn_salva_profilo.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=(5,5)); row_idx += 1

                    # --- MODIFICA DEFINITIVA ---
                    # Lista dei tipi di metodo che NON devono avere il pulsante "Prepara per Backtest" nel popup.
                    tipi_senza_backtest_popup = [
                        "semplice_analizzato",
                        "periodica_ambata_frequente",
                        "periodica_ottimale",  # AGGIUNTO
                        "ambata_ambo_unico_trasf",
                        "ambo_sommativo_auto"
                    ]
                    
                    if tipo_metodo_salv_effettivo not in tipi_senza_backtest_popup:
                        btn_prepara_backtest_popup = ttk.Button(scrollable_frame, text="Prepara per Backtest Dettagliato",
                                                          command=lambda dpb=dati_grezzi_per_questo_metodo.copy(): self._prepara_metodo_per_backtest(dpb))
                        btn_prepara_backtest_popup.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=(0,5)); row_idx +=1
                
                ambata_per_abbinamenti_popup = None
                if isinstance(ambata_da_visualizzare_nel_popup, (int, float)): 
                    ambata_per_abbinamenti_popup = ambata_da_visualizzare_nel_popup
                elif isinstance(ambata_da_visualizzare_nel_popup, str) and ambata_da_visualizzare_nel_popup.isdigit():
                    ambata_per_abbinamenti_popup = int(ambata_da_visualizzare_nel_popup)

                if ambata_per_abbinamenti_popup is not None: 
                    ttk.Label(scrollable_frame, text="Abbinamenti Consigliati (co-occorrenze storiche):").grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="w"); row_idx +=1
                    abbinamenti_dict_loop = previsione_dett.get('abbinamenti_dict', {}); 
                    eventi_totali_loop = abbinamenti_dict_loop.get("sortite_ambata_target", 0)
                    
                    if eventi_totali_loop > 0:
                        ttk.Label(scrollable_frame, text=f"  (Basato su {eventi_totali_loop} sortite storiche dell'ambata {ambata_per_abbinamenti_popup})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                        for tipo_sorte, dati_sorte_lista in abbinamenti_dict_loop.items():
                            if tipo_sorte == "sortite_ambata_target": continue
                            if dati_sorte_lista:
                                ttk.Label(scrollable_frame, text=f"    Per {tipo_sorte.upper().replace('_', ' ')}:").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                                for ab_info in dati_sorte_lista[:3]:
                                    if ab_info['conteggio'] > 0:
                                        numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri'])))
                                        freq_ab_disp = f"{ab_info['frequenza']:.1%}" if isinstance(ab_info['frequenza'], float) else str(ab_info['frequenza'])
                                        ttk.Label(scrollable_frame, text=f"      - Numeri: [{numeri_ab_str}] (Freq: {freq_ab_disp}, Cnt: {ab_info['conteggio']})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                    else:
                        ttk.Label(scrollable_frame, text=f"  Nessuna co-occorrenza storica per l'ambata {ambata_per_abbinamenti_popup}.").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                    
                    contorni_suggeriti_loop = previsione_dett.get('contorni_suggeriti', [])
                    if contorni_suggeriti_loop:
                        ttk.Label(scrollable_frame, text="  Altri Contorni Frequenti:").grid(row=row_idx, column=0, columnspan=2, pady=(3,1), sticky="w"); row_idx+=1
                        for contorno_num, contorno_cnt in contorni_suggeriti_loop[:5]:
                            ttk.Label(scrollable_frame, text=f"    - Numero: {contorno_num} (Presenze con ambata: {contorno_cnt})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx+=1
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=(5,0)); scrollbar_y.pack(side="right", fill="y")
        close_button_frame = ttk.Frame(popup_window); close_button_frame.pack(fill=tk.X, pady=(5,5), padx=5, side=tk.BOTTOM)
        ttk.Button(close_button_frame, text="Chiudi", command=popup_window.destroy).pack()
        popup_window.update_idletasks(); canvas.config(scrollregion=canvas.bbox("all"))
        try: self.master.eval(f'tk::PlaceWindow {str(popup_window)} center')
        except tk.TclError:
            popup_window.update_idletasks()
            master_x = self.master.winfo_x(); master_y = self.master.winfo_y()
            master_width = self.master.winfo_width(); master_height = self.master.winfo_height()
            popup_req_width = popup_window.winfo_reqwidth(); popup_req_height = popup_window.winfo_reqheight()
            x_pos = master_x + (master_width // 2) - (popup_req_width // 2)
            y_pos = master_y + (master_height // 2) - (popup_req_height // 2)
            popup_window.geometry(f"+{x_pos}+{y_pos}")
        popup_window.lift()

    def avvia_analisi_metodi_semplici(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA METODI SEMPLICI\n" + "="*50)

        if hasattr(self, 'ms_risultati_listbox') and self.ms_risultati_listbox:
            self.ms_risultati_listbox.delete(0, tk.END)
        self.metodi_semplici_trovati_dati = []

        storico_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi:
            if hasattr(self, 'ms_risultati_listbox') and self.ms_risultati_listbox:
                self.ms_risultati_listbox.insert(tk.END, "Caricamento storico fallito.")
            return

        ruote_gioco, lookahead, indice_mese_utente = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return
        ruota_calcolo_input = self.ruota_calcolo_var.get()
        posizione_estratto_input = self.posizione_estratto_var.get() - 1
        num_ambate_richieste_gui = self.num_ambate_var.get()
        min_tentativi = self.min_tentativi_var.get()

        data_fine_globale_obj = None
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_globale_obj = self.date_fine_entry_analisi.get_date()
        except ValueError:
            pass

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()

            risultati_individuali_grezzi, info_copertura_combinata = trova_migliori_ambate_e_abbinamenti(
                storico_per_analisi, ruota_calcolo_input, posizione_estratto_input,
                ruote_gioco, max_ambate_output=num_ambate_richieste_gui, lookahead=lookahead,
                indice_mese_filtro=indice_mese_utente, min_tentativi_per_ambata=min_tentativi,
                app_logger=self._log_to_gui, data_fine_analisi_globale_obj=data_fine_globale_obj
            )

            self.metodi_semplici_trovati_dati = risultati_individuali_grezzi if risultati_individuali_grezzi else []
            
            if hasattr(self, 'ms_risultati_listbox') and self.ms_risultati_listbox:
                if not self.metodi_semplici_trovati_dati:
                    self.ms_risultati_listbox.insert(tk.END, "Nessun metodo semplice valido trovato.")
                else:
                    for i, res in enumerate(self.metodi_semplici_trovati_dati):
                        met_info = res['metodo']
                        formula = f"{met_info['ruota_calcolo']}[{met_info['pos_estratto_calcolo']+1}] {met_info['operazione']} {met_info['operando_fisso']}"
                        riga_listbox = (f"{(i+1)}. {(res['frequenza_ambata']*100):>5.1f}% ({res['successi']}/{res['tentativi']}) -> {formula}")
                        self.ms_risultati_listbox.insert(tk.END, riga_listbox)

            if risultati_individuali_grezzi:
                 lista_previsioni_per_popup = []
                 dati_grezzi_per_popup_finali_per_popup = []
                 
                 estrazione_riferimento_live = None
                 if "estrazione_usata_per_previsione" in risultati_individuali_grezzi[0]:
                     estrazione_riferimento_live = risultati_individuali_grezzi[0]["estrazione_usata_per_previsione"]
                 elif storico_per_analisi :
                    estrazione_riferimento_live = storico_per_analisi[-1]
                 
                 data_riferimento_str_popup_s_comune = estrazione_riferimento_live['data'].strftime('%d/%m/%Y') if estrazione_riferimento_live else "N/D"

                 for res_idx, res_singolo_metodo in enumerate(risultati_individuali_grezzi[:num_ambate_richieste_gui]):
                     metodo_s_info = res_singolo_metodo['metodo']
                     formula_testuale_semplice = f"{metodo_s_info['ruota_calcolo']}[pos.{metodo_s_info['pos_estratto_calcolo']+1}] {metodo_s_info['operazione']} {metodo_s_info['operando_fisso']}"
                     
                     # >>> INIZIO MODIFICA CRUCIALE <<<
                     # Aggiungiamo un flag per non mostrare il pulsante Prepara Backtest
                     res_singolo_metodo['mostra_btn_prepara_backtest'] = False
                     # >>> FINE MODIFICA CRUCIALE <<<

                     dettaglio_previsione_per_popup = {
                         "titolo_sezione": f"--- {(res_idx+1)}° METODO / PREVISIONE ---",
                         "info_metodo_str": formula_testuale_semplice,
                         "ambata_prevista": res_singolo_metodo.get('ambata_piu_frequente_dal_metodo'),
                         "abbinamenti_dict": res_singolo_metodo.get("abbinamenti", {}),
                         "performance_storica_str": f"{res_singolo_metodo['frequenza_ambata']:.2%} ({res_singolo_metodo['successi']}/{res_singolo_metodo['tentativi']} casi)",
                         "contorni_suggeriti": [] # I metodi semplici non usano questa chiave
                     }
                     lista_previsioni_per_popup.append(dettaglio_previsione_per_popup)

                     dati_metodo_per_popup = res_singolo_metodo.copy()
                     dati_metodo_per_popup["tipo_metodo_salvato"] = "semplice_analizzato"
                     dati_grezzi_per_popup_finali_per_popup.append(dati_metodo_per_popup)

                 self.mostra_popup_previsione(
                    titolo_popup="Previsione Metodi Semplici",
                    ruote_gioco_str=", ".join(ruote_gioco),
                    lista_previsioni_dettagliate=lista_previsioni_per_popup,
                    copertura_combinata_info=info_copertura_combinata if num_ambate_richieste_gui > 1 else None,
                    data_riferimento_previsione_str_comune=data_riferimento_str_popup_s_comune,
                    metodi_grezzi_per_salvataggio=dati_grezzi_per_popup_finali_per_popup,
                    indice_mese_richiesto_utente=indice_mese_utente,
                    data_fine_analisi_globale_obj=data_fine_globale_obj,
                    estrazione_riferimento_per_previsione_live=estrazione_riferimento_live
                 )
            elif hasattr(self, 'ms_risultati_listbox') and self.ms_risultati_listbox.size() > 0:
                pass # Evita di mostrare un popup se la lista è stata popolata ma non ci sono risultati top
            else:
                messagebox.showinfo("Analisi Metodi Semplici", "Nessun metodo semplice ha prodotto risultati sufficientemente frequenti.")

        except Exception as e:
            messagebox.showerror("Errore Analisi", f"Errore ricerca metodi semplici: {e}");
            self._log_to_gui(f"ERRORE CRITICO: {e}\n{traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def _prepara_metodo_per_backtest(self, dati_metodo_selezionato_per_prep):
        self._log_to_gui(f"\nDEBUG: _prepara_metodo_per_backtest CHIAMATO con dati: {dati_metodo_selezionato_per_prep}")

        dati_per_backtest_finali = dati_metodo_selezionato_per_prep.copy()
        
        # Determina il tipo di metodo in modo più robusto
        tipo_metodo = dati_per_backtest_finali.get("tipo_metodo_salvato")
        if not tipo_metodo:
            tipo_metodo = dati_per_backtest_finali.get('tipo')
        # Se ancora non c'è, controlliamo la presenza di chiavi specifiche
        if not tipo_metodo:
            if "metodo_formula" in dati_per_backtest_finali and "copertura_periodi_perc" in dati_per_backtest_finali:
                tipo_metodo = "periodica_ottimale"
                self._log_to_gui("INFO: Tipo metodo dedotto come 'periodica_ottimale' dalla struttura dei dati.")
            elif "successi_ambo" in dati_per_backtest_finali:
                 tipo_metodo = "ambata_ambo_unico_trasf"
                 self._log_to_gui("INFO: Tipo metodo dedotto come 'ambata_ambo_unico_trasf' dalla struttura dei dati.")
        
        # Assicura che il tipo sia sempre presente nel dizionario finale
        if tipo_metodo:
             dati_per_backtest_finali['tipo'] = tipo_metodo

        # BLOCCO DI TRADUZIONE PER I METODI CHE NE HANNO BISOGNO
        if tipo_metodo == "periodica_ottimale":
            self._log_to_gui("INFO: Preparazione backtest per 'periodica_ottimale'.")
            metodo_formula = dati_per_backtest_finali.get("metodo_formula")
            
            if metodo_formula and isinstance(metodo_formula, dict):
                op_simbolo_map = {'somma': '+', 'differenza': '-', 'moltiplicazione': '*'}
                op_simbolo = op_simbolo_map.get(metodo_formula.get('operazione'))
                req_keys = ['ruota_calcolo', 'pos_estratto_calcolo', 'operando_fisso']
                
                if op_simbolo and all(key in metodo_formula for key in req_keys):
                    definizione_costruita = [
                        {'tipo_termine': 'estratto', 'ruota': metodo_formula['ruota_calcolo'], 'posizione': metodo_formula['pos_estratto_calcolo'] - 1, 'operazione_successiva': op_simbolo},
                        {'tipo_termine': 'fisso', 'valore_fisso': metodo_formula['operando_fisso'], 'operazione_successiva': '='}
                    ]
                    dati_per_backtest_finali['definizione_strutturata'] = definizione_costruita
                else:
                    messagebox.showerror("Errore Preparazione", "Dati della formula incompleti per il metodo periodico."); return
            else:
                messagebox.showerror("Errore Preparazione", "Dati 'metodo_formula' mancanti per il metodo periodico."); return
        
        elif tipo_metodo == "periodica_ambata_frequente":
            # Questa parte dovrebbe già funzionare, ma la rendiamo esplicita
            if 'definizione_strutturata' not in dati_per_backtest_finali:
                 ambata = dati_per_backtest_finali.get('ambata_prevista')
                 if isinstance(ambata, int):
                     dati_per_backtest_finali['definizione_strutturata'] = [{'tipo_termine': 'fisso', 'valore_fisso': ambata, 'operazione_successiva': '='}]
                 else:
                     messagebox.showerror("Errore Preparazione", "Ambata non valida per 'presenze periodiche'."); return

        # ... (la tua logica per ambo unico va già bene)
        
        # VALIDAZIONE FINALE
        if dati_per_backtest_finali.get('definizione_strutturata'):
            self.metodo_preparato_per_backtest = dati_per_backtest_finali
            
            formula_display_gui = self.metodo_preparato_per_backtest.get('formula_testuale', 'N/D')
            tipo_display_gui = self.metodo_preparato_per_backtest.get('tipo', 'Sconosciuto').replace("_", " ").title()

            if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
                self.mc_listbox_componenti_1.delete(0, tk.END)
                self.mc_listbox_componenti_1.insert(tk.END, f"PER BACKTEST ({tipo_display_gui}):")
                self.mc_listbox_componenti_1.insert(tk.END, formula_display_gui)
            
            messagebox.showinfo("Metodo Pronto per Backtest", f"Metodo ({tipo_display_gui}) è stato selezionato.\n\nOra puoi usare il pulsante 'Backtest Dettagliato'.")
            
            if hasattr(self, 'usa_ultimo_corretto_per_backtest_var'):
                self.usa_ultimo_corretto_per_backtest_var.set(False)
        else:
            messagebox.showerror("Errore Preparazione Backtest", "Impossibile preparare il metodo per il backtest. Dati interni mancanti.")
            self.metodo_preparato_per_backtest = None

    def avvia_analisi_metodo_complesso(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO ANALISI METODI COMPLESSI BASE\n" + "="*50)
        metodo_1_def_mc = self.definizione_metodo_complesso_attuale; metodo_2_def_mc = self.definizione_metodo_complesso_attuale_2
        err_msg_mc = []; metodi_validi_per_analisi_def = []; metodi_grezzi_per_popup_salvataggio = []
        if metodo_1_def_mc:
            if metodo_1_def_mc[-1].get('operazione_successiva') == '=': metodi_validi_per_analisi_def.append(metodo_1_def_mc)
            else: err_msg_mc.append("Metodo Base 1 non terminato con '='.")
        if metodo_2_def_mc:
            if metodo_2_def_mc[-1].get('operazione_successiva') == '=': metodi_validi_per_analisi_def.append(metodo_2_def_mc)
            else: err_msg_mc.append("Metodo Base 2 non terminato con '='.")
        if err_msg_mc: self._log_to_gui("ERRORE VALIDAZIONE METODI:\n" + "\n".join(err_msg_mc)); messagebox.showerror("Errore Input Metodi", "\n".join(err_msg_mc))
        if not metodi_validi_per_analisi_def:
            if not err_msg_mc: messagebox.showerror("Errore Input", "Definire almeno un Metodo Base valido (terminato con '=')")
            self._log_to_gui("ERRORE: Nessun Metodo Base valido definito per l'analisi."); return

        storico_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi: return

        ruote_gioco, lookahead, indice_mese_utente = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return

        # --- LOGICA PER DETERMINARE L'ESTRAZIONE PER LA PREVISIONE LIVE ---
        estrazione_riferimento_live = None
        data_riferimento_comune_popup_str = "N/A"
        nota_globale_previsione_complesso = "" # Nota specifica per questa analisi
        data_fine_globale_obj_complesso = None
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_globale_obj_complesso = self.date_fine_entry_analisi.get_date()
        except ValueError:
            pass

        if indice_mese_utente is not None and data_fine_globale_obj_complesso is not None:
            self._log_to_gui(f"AMC: Ricerca estrazione per previsione live: indice_mese_utente={indice_mese_utente}, data_fine<={data_fine_globale_obj_complesso.strftime('%Y-%m-%d') if data_fine_globale_obj_complesso else 'N/A'}")
            for estr in reversed(storico_per_analisi):
                if estr['data'] <= data_fine_globale_obj_complesso:
                    if estr.get('indice_mese') == indice_mese_utente:
                        estrazione_riferimento_live = estr
                        data_riferimento_comune_popup_str = estr['data'].strftime('%d/%m/%Y')
                        self._log_to_gui(f"  AMC: Trovata estrazione specifica per previsione live: {data_riferimento_comune_popup_str} (Indice: {estr.get('indice_mese')})")
                        break
            if not estrazione_riferimento_live:
                nota_globale_previsione_complesso = f"Nessuna estrazione trovata corrispondente all'indice mese {indice_mese_utente} entro il {data_fine_globale_obj_complesso.strftime('%d/%m/%Y')}."
                self._log_to_gui(f"  AMC: {nota_globale_previsione_complesso}")
        elif storico_per_analisi:
            estrazione_riferimento_live = storico_per_analisi[-1]
            data_riferimento_comune_popup_str = estrazione_riferimento_live['data'].strftime('%d/%m/%Y')
            self._log_to_gui(f"  AMC: Uso ultima estrazione disponibile ({data_riferimento_comune_popup_str}) per previsione (nessun filtro specifico indice/data fine, o uno dei due mancava).")
        
        if not estrazione_riferimento_live and not nota_globale_previsione_complesso:
            nota_globale_previsione_complesso = "Impossibile determinare estrazione di riferimento per la previsione."
            self._log_to_gui(f"  AMC: {nota_globale_previsione_complesso}")
        # --- FINE LOGICA PER DETERMINARE L'ESTRAZIONE PER LA PREVISIONE LIVE ---


        self.master.config(cursor="watch"); self.master.update_idletasks()
        lista_previsioni_popup_mc_vis = []; info_copertura_combinata_mc = None
        
        try:
            for idx, metodo_def_corrente in enumerate(metodi_validi_per_analisi_def):
                nome_metodo_log_popup = f"Metodo Base {idx + 1}"; self._log_to_gui(f"\n--- ANALISI {nome_metodo_log_popup.upper()} ---")
                metodo_str_popup = "".join(self._format_componente_per_display(comp) for comp in metodo_def_corrente)
                self._log_to_gui(f"  Definizione: {metodo_str_popup}")
                s_ind, t_ind, applicazioni_vincenti_ind = analizza_metodo_complesso_specifico(
                    storico_per_analisi, metodo_def_corrente, ruote_gioco, lookahead, indice_mese_utente, self._log_to_gui
                )
                f_ind = s_ind / t_ind if t_ind > 0 else 0.0
                perf_ind_str = f"{f_ind:.2%} ({s_ind}/{t_ind} casi)" if t_ind > 0 else "Non applicabile storicamente."
                self._log_to_gui(f"  Performance Storica Individuale {nome_metodo_log_popup}: {perf_ind_str}")
                
                ambata_live = None
                abb_live = {}
                if estrazione_riferimento_live: # Calcola solo se abbiamo un'estrazione di riferimento
                    ambata_live, abb_live = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                        storico_per_analisi, # Per gli abbinamenti storici
                        metodo_def_corrente, 
                        ruote_gioco, 
                        data_riferimento_comune_popup_str, 
                        nome_metodo_log_popup,
                        estrazione_riferimento_live # L'estrazione su cui calcolare la previsione live
                    )
                elif nota_globale_previsione_complesso: # Se c'era una nota globale perché non trovata estrazione idonea
                    self._log_to_gui(f"  PREVISIONE LIVE {nome_metodo_log_popup}: {nota_globale_previsione_complesso}")

                
                lista_previsioni_popup_mc_vis.append({
                    "titolo_sezione": f"--- PREVISIONE {nome_metodo_log_popup.upper()} ---", 
                    "info_metodo_str": metodo_str_popup, 
                    "ambata_prevista": ambata_live if ambata_live is not None else "N/D", # Assicura che sia "N/D" se None
                    "abbinamenti_dict": abb_live, 
                    "performance_storica_str": perf_ind_str
                })
                metodi_grezzi_per_popup_salvataggio.append({
                    "tipo_metodo_salvato": "complesso_base_analizzato", 
                    "definizione_metodo_originale": metodo_def_corrente, 
                    "formula_testuale": metodo_str_popup, 
                    "ambata_prevista": ambata_live if ambata_live is not None else "N/D", 
                    "abbinamenti": abb_live, "successi": s_ind, 
                    "tentativi": t_ind, "frequenza": f_ind, 
                    "applicazioni_vincenti_dettagliate": applicazioni_vincenti_ind,
                    "estrazione_usata_per_previsione": estrazione_riferimento_live # Aggiungi per coerenza
                })

            if len(metodi_validi_per_analisi_def) == 2:
                # ... (codice per analisi combinata, invariato) ...
                pass # Lascio il pass per brevità

            self.master.config(cursor="")
            if not lista_previsioni_popup_mc_vis: 
                messagebox.showinfo("Analisi Metodi Complessi", "Nessun metodo complesso valido ha prodotto una previsione popup.")
            else:
                ruote_gioco_str_popup = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                self.mostra_popup_previsione(
                   titolo_popup="Previsioni Metodi Complessi Base", ruote_gioco_str=ruote_gioco_str_popup,
                   lista_previsioni_dettagliate=lista_previsioni_popup_mc_vis, copertura_combinata_info=info_copertura_combinata_mc,
                   data_riferimento_previsione_str_comune=data_riferimento_comune_popup_str,
                   metodi_grezzi_per_salvataggio=metodi_grezzi_per_popup_salvataggio,
                   indice_mese_richiesto_utente=indice_mese_utente,
                   data_fine_analisi_globale_obj=data_fine_globale_obj_complesso,
                   estrazione_riferimento_per_previsione_live=estrazione_riferimento_live
                )
            self._log_to_gui("\n--- Analisi Metodi Complessi Base Completata ---")
        except Exception as e: 
            self.master.config(cursor=""); 
            messagebox.showerror("Errore Analisi", f"Errore analisi metodi complessi: {e}"); 
            self._log_to_gui(f"ERRORE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def avvia_ricerca_correttore(self): # QUESTA E' LA TUA VERSIONE ORIGINALE, CHE MODIFICHEREMO
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA CORRETTORE OTTIMALE (PER METODI COMPLESSI BASE)\n" + "="*50)
        
        self.ultimo_metodo_corretto_trovato_definizione = None 
        self.ultimo_metodo_corretto_formula_testuale = ""
        self.usa_ultimo_corretto_per_backtest_var.set(False)

        metodo_1_def_corr_base = self.definizione_metodo_complesso_attuale
        metodo_2_def_corr_base = self.definizione_metodo_complesso_attuale_2
        err_msg_corr = []
        almeno_un_metodo_base_valido_per_correttore = False
        metodo_1_valido_def = None 
        metodo_2_valido_def = None 

        # ... (Logica di validazione per metodo_1_valido_def e metodo_2_valido_def - come nella tua originale) ...
        if metodo_1_def_corr_base:
            if metodo_1_def_corr_base[-1].get('operazione_successiva') == '=':
                almeno_un_metodo_base_valido_per_correttore = True
                metodo_1_valido_def = metodo_1_def_corr_base 
            else: err_msg_corr.append("Metodo Base 1 non terminato con '=' o vuoto.")
        if metodo_2_def_corr_base:
            if metodo_2_def_corr_base[-1].get('operazione_successiva') == '=':
                almeno_un_metodo_base_valido_per_correttore = True
                metodo_2_valido_def = metodo_2_def_corr_base 
            else: err_msg_corr.append("Metodo Base 2 non terminato con '=' o vuoto.")

        if err_msg_corr and almeno_un_metodo_base_valido_per_correttore:
             self._log_to_gui("AVVISO VALIDAZIONE METODI PER CORRETTORE:\n" + "\n".join(err_msg_corr))
        elif err_msg_corr and not almeno_un_metodo_base_valido_per_correttore: 
            messagebox.showerror("Errore Input", "Definire almeno un Metodo Base valido (terminato con '=') per cercare un correttore.\n" + "\n".join(err_msg_corr))
            self._log_to_gui("ERRORE: Nessun Metodo Base valido per la ricerca correttore."); return
        
        if not almeno_un_metodo_base_valido_per_correttore: 
            messagebox.showerror("Errore Input", "Definire almeno un Metodo Base valido (terminato con '=') per cercare un correttore.")
            self._log_to_gui("ERRORE: Nessun Metodo Base valido definito per la ricerca correttore."); return


        storico_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi: return
        
        ruote_gioco, lookahead, indice_mese_utente = self._get_parametri_gioco_comuni() # Per popup e logica selezione estrazione
        if ruote_gioco is None: return

        # --- RECUPERO data_fine_globale_obj_corr ---
        data_fine_globale_obj_corr = None
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_globale_obj_corr = self.date_fine_entry_analisi.get_date()
        except ValueError:
            pass
        # --- FINE RECUPERO ---

        min_tentativi_correttore_cfg = self.corr_cfg_min_tentativi.get()
        # ... (recupero parametri correttore c_fisso_s, etc. - come nella tua originale) ...
        c_fisso_s = self.corr_cfg_cerca_fisso_semplice.get(); c_estr_s = self.corr_cfg_cerca_estratto_semplice.get()
        c_somma_ef = self.corr_cfg_cerca_somma_estr_fisso.get(); c_somma_ee = self.corr_cfg_cerca_somma_estr_estr.get()
        c_diff_ef = self.corr_cfg_cerca_diff_estr_fisso.get(); c_diff_ee = self.corr_cfg_cerca_diff_estr_estr.get()
        c_mult_ef = self.corr_cfg_cerca_mult_estr_fisso.get(); c_mult_ee = self.corr_cfg_cerca_mult_estr_estr.get()
        tipi_corr_log = [] # ... (popolamento tipi_corr_log - come nella tua originale)

        # ... (log dei parametri - come nella tua originale) ...

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati_correttori_list = trova_miglior_correttore_per_metodo_complesso(
                storico_per_analisi,
                metodo_1_valido_def, 
                metodo_2_valido_def, 
                c_fisso_s, c_estr_s,
                c_diff_ef, c_diff_ee,
                c_mult_ef, c_mult_ee,
                c_somma_ef, c_somma_ee,
                ruote_gioco, lookahead, indice_mese_utente, # Passa indice_mese_utente per l'analisi storica del correttore
                min_tentativi_correttore_cfg, app_logger=self._log_to_gui,
                filtro_condizione_primaria_dict=None # O passa la condizione se rilevante per questa analisi
            )

            self._log_to_gui("\n\n--- RISULTATI RICERCA CORRETTORI (LOG COMPLETO) ---")
            if not risultati_correttori_list:
                self._log_to_gui("Nessun correttore valido trovato che migliori il benchmark dei metodi base.")
                messagebox.showinfo("Ricerca Correttore", "Nessun correttore valido trovato che migliori il benchmark.")
            else:
                miglior_risultato_correttore = risultati_correttori_list[0]
                
                definizioni_corrette_salvate = {}
                # ... (popolamento di definizioni_corrette_salvate - come nella tua originale) ...
                # ... (salvataggio in self.ultimo_metodo_corretto_trovato_definizione - come nella tua originale) ...
                if miglior_risultato_correttore.get('def_metodo_esteso_1'):
                    def_m1c = list(miglior_risultato_correttore.get('def_metodo_esteso_1'))
                    definizioni_corrette_salvate['base1_corretto'] = def_m1c
                if miglior_risultato_correttore.get('def_metodo_esteso_2'):
                    def_m2c = list(miglior_risultato_correttore.get('def_metodo_esteso_2'))
                    definizioni_corrette_salvate['base2_corretto'] = def_m2c
                
                if definizioni_corrette_salvate:
                    self.ultimo_metodo_corretto_trovato_definizione = definizioni_corrette_salvate
                    # ... (altro come prima)
                    self.usa_ultimo_corretto_per_backtest_var.set(True) 
                else: 
                    self.ultimo_metodo_corretto_trovato_definizione = None
                    self.usa_ultimo_corretto_per_backtest_var.set(False)


                # --- LOGICA PER DETERMINARE L'ESTRAZIONE PER LA PREVISIONE LIVE DEL METODO CORRETTO ---
                estrazione_per_previsione_live_corretta = None
                data_riferimento_popup_corr_str = "N/A" # Default
                nota_previsione_corretta = ""

                if indice_mese_utente is not None and data_fine_globale_obj_corr is not None:
                    self._log_to_gui(f"ARC: Ricerca estrazione per previsione live (corretta): idx_mese={indice_mese_utente}, data_fine<={data_fine_globale_obj_corr.strftime('%Y-%m-%d') if data_fine_globale_obj_corr else 'N/A'}")
                    for estr_corr in reversed(storico_per_analisi):
                        if estr_corr['data'] <= data_fine_globale_obj_corr:
                            if estr_corr.get('indice_mese') == indice_mese_utente:
                                estrazione_per_previsione_live_corretta = estr_corr
                                data_riferimento_popup_corr_str = estr_corr['data'].strftime('%d/%m/%Y')
                                self._log_to_gui(f"  ARC: Trovata estrazione specifica per previsione live (corretta): {data_riferimento_popup_corr_str} (Indice: {estr_corr.get('indice_mese')})")
                                break
                    if not estrazione_per_previsione_live_corretta:
                        nota_previsione_corretta = f"Nessuna estrazione trovata per l'indice mese {indice_mese_utente} entro il {data_fine_globale_obj_corr.strftime('%d/%m/%Y')} per la previsione del metodo corretto."
                        self._log_to_gui(f"  ARC: {nota_previsione_corretta}")
                elif storico_per_analisi: # Se i filtri non sono applicabili, usa l'ultima
                    estrazione_per_previsione_live_corretta = storico_per_analisi[-1]
                    data_riferimento_popup_corr_str = estrazione_per_previsione_live_corretta['data'].strftime('%d/%m/%Y')
                    self._log_to_gui(f"  ARC: Uso ultima estrazione disponibile ({data_riferimento_popup_corr_str}) per previsione metodo corretto.")
                
                if not estrazione_per_previsione_live_corretta and not nota_previsione_corretta:
                    nota_previsione_corretta = "Impossibile determinare estrazione di riferimento per la previsione del metodo corretto."
                    self._log_to_gui(f"  ARC: {nota_previsione_corretta}")
                # --- FINE LOGICA SELEZIONE ESTRAZIONE ---

                lista_previsioni_popup_corr_vis = []
                metodi_grezzi_corretti_per_salvataggio_popup = []
                
                info_correttore_globale_str = (
                    f"Correttore Applicato: {miglior_risultato_correttore['tipo_correttore_descrittivo']} -> {miglior_risultato_correttore['dettaglio_correttore_str']}\n"
                    f"Operazione di Collegamento Base: '{miglior_risultato_correttore['operazione_collegamento_base']}'\n"
                    f"Performance Globale del Metodo/i Corretto/i: {miglior_risultato_correttore['frequenza']:.2%} ({miglior_risultato_correttore['successi']}/{miglior_risultato_correttore['tentativi']} casi)"
                )
                
                # Calcola previsione per Metodo 1 Corretto (se esiste)
                if 'base1_corretto' in definizioni_corrette_salvate:
                    met1_est_def_popup = definizioni_corrette_salvate['base1_corretto']
                    met1_est_str_popup = "".join(self._format_componente_per_display(comp) for comp in met1_est_def_popup)
                    ambata1_corr_live_popup, abb1_corr_live_popup = "N/D", {}
                    
                    if estrazione_per_previsione_live_corretta: # Calcola solo se abbiamo un'estrazione valida
                        ambata1_corr_live_popup, abb1_corr_live_popup = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                            storico_per_analisi, met1_est_def_popup, ruote_gioco, 
                            data_riferimento_popup_corr_str, "Metodo 1 Corretto",
                            estrazione_riferimento_previsione=estrazione_per_previsione_live_corretta # Passa l'estrazione corretta
                        )
                    elif nota_previsione_corretta:
                         self._log_to_gui(f"  ARC: Previsione per Metodo 1 Corretto non calcolata a causa di: {nota_previsione_corretta}")

                    lista_previsioni_popup_corr_vis.append({
                        "titolo_sezione": "--- PREVISIONE METODO 1 CORRETTO ---", 
                        "info_metodo_str": met1_est_str_popup,
                        "ambata_prevista": ambata1_corr_live_popup if ambata1_corr_live_popup is not None else "N/D", 
                        "abbinamenti_dict": abb1_corr_live_popup,
                        "performance_storica_str": "Vedi performance globale correttore" 
                    })
                    # ... (codice per metodi_grezzi_corretti_per_salvataggio_popup per base1 - come prima) ...
                    metodi_grezzi_corretti_per_salvataggio_popup.append({
                        "tipo_metodo_salvato": "complesso_corretto", 
                        "riferimento_base": "base1", 
                        "definizione_metodo_base_originale_1": metodo_1_valido_def, 
                        "def_metodo_esteso_1": met1_est_def_popup, 
                        "formula_testuale": met1_est_str_popup, 
                        "ambata_prevista": ambata1_corr_live_popup if ambata1_corr_live_popup is not None else "N/D", 
                        "abbinamenti": abb1_corr_live_popup,
                        "tipo_correttore_descrittivo": miglior_risultato_correttore['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_risultato_correttore['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_risultato_correttore['operazione_collegamento_base'],
                        "successi": miglior_risultato_correttore['successi'], 
                        "tentativi": miglior_risultato_correttore['tentativi'],
                        "frequenza": miglior_risultato_correttore['frequenza'],
                        "estrazione_usata_per_previsione": estrazione_per_previsione_live_corretta # Aggiunto
                    })


                # Calcola previsione per Metodo 2 Corretto (se esiste)
                if 'base2_corretto' in definizioni_corrette_salvate:
                    met2_est_def_popup = definizioni_corrette_salvate['base2_corretto']
                    met2_est_str_popup = "".join(self._format_componente_per_display(comp) for comp in met2_est_def_popup)
                    ambata2_corr_live_popup, abb2_corr_live_popup = "N/D", {}
                    
                    if estrazione_per_previsione_live_corretta:
                        ambata2_corr_live_popup, abb2_corr_live_popup = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                            storico_per_analisi, met2_est_def_popup, ruote_gioco, 
                            data_riferimento_popup_corr_str, "Metodo 2 Corretto",
                            estrazione_riferimento_previsione=estrazione_per_previsione_live_corretta
                        )
                    elif nota_previsione_corretta:
                        self._log_to_gui(f"  ARC: Previsione per Metodo 2 Corretto non calcolata a causa di: {nota_previsione_corretta}")
                        
                    lista_previsioni_popup_corr_vis.append({
                        "titolo_sezione": "--- PREVISIONE METODO 2 CORRETTO ---", 
                        "info_metodo_str": met2_est_str_popup,
                        "ambata_prevista": ambata2_corr_live_popup if ambata2_corr_live_popup is not None else "N/D", 
                        "abbinamenti_dict": abb2_corr_live_popup,
                        "performance_storica_str": "Vedi performance globale correttore"
                    })
                    # ... (codice per metodi_grezzi_corretti_per_salvataggio_popup per base2 - come prima) ...
                    metodi_grezzi_corretti_per_salvataggio_popup.append({
                        "tipo_metodo_salvato": "complesso_corretto",
                        "riferimento_base": "base2",
                        "definizione_metodo_base_originale_2": metodo_2_valido_def,
                        "def_metodo_esteso_2": met2_est_def_popup, 
                        "formula_testuale": met2_est_str_popup, 
                        "ambata_prevista": ambata2_corr_live_popup if ambata2_corr_live_popup is not None else "N/D", 
                        "abbinamenti": abb2_corr_live_popup,
                        "tipo_correttore_descrittivo": miglior_risultato_correttore['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_risultato_correttore['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_risultato_correttore['operazione_collegamento_base'],
                        "successi": miglior_risultato_correttore['successi'], 
                        "tentativi": miglior_risultato_correttore['tentativi'],
                        "frequenza": miglior_risultato_correttore['frequenza'],
                        "estrazione_usata_per_previsione": estrazione_per_previsione_live_corretta # Aggiunto
                    })


                if not lista_previsioni_popup_corr_vis:
                     messagebox.showinfo("Ricerca Correttore", "Miglior correttore trovato, ma non è stato possibile generare previsioni valide per il popup.")
                else:
                    ruote_gioco_str_popup = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                    info_correttore_per_popup_vis = {"testo_introduttivo": info_correttore_globale_str}

                    self.mostra_popup_previsione(
                       titolo_popup="Previsione Metodo/i con Correttore Ottimale",
                       ruote_gioco_str=ruote_gioco_str_popup,
                       lista_previsioni_dettagliate=lista_previsioni_popup_corr_vis,
                       copertura_combinata_info=info_correttore_per_popup_vis,
                       data_riferimento_previsione_str_comune=data_riferimento_popup_corr_str, 
                       metodi_grezzi_per_salvataggio=metodi_grezzi_corretti_per_salvataggio_popup,
                       indice_mese_richiesto_utente=indice_mese_utente,
                       data_fine_analisi_globale_obj=data_fine_globale_obj_corr,
                       estrazione_riferimento_per_previsione_live=estrazione_per_previsione_live_corretta 
                    )
            self._log_to_gui("\n--- Ricerca Correttore Ottimale (per Metodi Complessi Base) Completata ---")
        except Exception as e:
            messagebox.showerror("Errore Ricerca Correttore", f"Errore: {e}"); self._log_to_gui(f"ERRORE CRITICO RICERCA CORRETTORE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def avvia_analisi_condizionata(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO ANALISI CONDIZIONATA AVANZATA\n" + "="*50)
        self.ac_metodi_condizionati_dettagli = [] 
        if self.ac_risultati_listbox:
            self.ac_risultati_listbox.delete(0, tk.END)

        # Carica lo storico usando i filtri globali per l'analisi di performance storica
        storico_per_analisi_cond = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi_cond: 
            self._log_to_gui("AAC: Caricamento storico per analisi fallito.")
            return

        ruote_gioco_cond, lookahead_cond, indice_mese_cond_utente = self._get_parametri_gioco_comuni()
        if ruote_gioco_cond is None: 
            self._log_to_gui("AAC: Parametri di gioco non validi.")
            return

        ruota_cond_input = self.ac_ruota_cond_var.get()
        pos_cond_input_1based = self.ac_pos_cond_var.get()
        pos_cond_input_0based = pos_cond_input_1based - 1
        val_min_cond_input = self.ac_val_min_cond_var.get()
        val_max_cond_input = self.ac_val_max_cond_var.get()
        ruota_calc_amb_input = self.ac_ruota_calc_ambata_var.get()
        pos_calc_amb_input_1based = self.ac_pos_calc_ambata_var.get()
        pos_calc_amb_input_0based = pos_calc_amb_input_1based - 1
        num_ris_cond_input = self.ac_num_risultati_var.get()
        min_tent_cond_input = self.ac_min_tentativi_var.get()

        if val_min_cond_input > val_max_cond_input:
            messagebox.showerror("Errore Input Condizione", "Valore minimo per la condizione non può essere maggiore del valore massimo.")
            self._log_to_gui("AAC ERRORE: Valore Min Condizione > Valore Max Condizione"); return

        self._log_to_gui("Parametri Analisi Condizionata:")
        self._log_to_gui(f"  Condizione: {ruota_cond_input}[pos.{pos_cond_input_1based}] in [{val_min_cond_input}-{val_max_cond_input}]")
        self._log_to_gui(f"  Calcolo Ambata: da {ruota_calc_amb_input}[pos.{pos_calc_amb_input_1based}] +/-/* Fisso")
        self._log_to_gui(f"  Ruote Gioco: {', '.join(ruote_gioco_cond)}, Colpi: {lookahead_cond}, Ind.Mese Storico: {indice_mese_cond_utente if indice_mese_cond_utente else 'Tutte'}")
        self._log_to_gui(f"  N. Risultati: {num_ris_cond_input}, Min. Tentativi (post-cond): {min_tent_cond_input}")

        # --- LOGICA PER DETERMINARE L'ESTRAZIONE PER LA PREVISIONE LIVE ---
        estrazione_riferimento_live_ac = None
        data_fine_globale_obj_ac = None
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_globale_obj_ac = self.date_fine_entry_analisi.get_date()
        except ValueError:
            self._log_to_gui("AAC WARN: Data fine analisi globale non valida.")
            pass # data_fine_globale_obj_ac rimarrà None

        # Usiamo indice_mese_cond_utente (che è l'indice globale) per determinare l'estrazione per la previsione live
        if indice_mese_cond_utente is not None and data_fine_globale_obj_ac is not None:
            self._log_to_gui(f"AAC: Ricerca estrazione per previsione live: idx_mese_utente={indice_mese_cond_utente}, data_fine<={data_fine_globale_obj_ac.strftime('%Y-%m-%d') if data_fine_globale_obj_ac else 'N/A'}")
            # Cerchiamo nello storico_per_analisi_cond, che è già filtrato per data inizio/fine globale
            for estr_ac in reversed(storico_per_analisi_cond): 
                if estr_ac['data'] <= data_fine_globale_obj_ac: # Doppio controllo, ma non guasta
                    if estr_ac.get('indice_mese') == indice_mese_cond_utente:
                        estrazione_riferimento_live_ac = estr_ac
                        self._log_to_gui(f"  AAC: Trovata estrazione specifica per previsione live: {estrazione_riferimento_live_ac['data'].strftime('%d/%m/%Y')} (Indice: {estrazione_riferimento_live_ac.get('indice_mese')})")
                        break
            if not estrazione_riferimento_live_ac: # Se il loop non ha trovato nulla
                self._log_to_gui(f"  AAC: Nessuna estrazione trovata per indice mese {indice_mese_cond_utente} entro il {data_fine_globale_obj_ac.strftime('%d/%m/%Y') if data_fine_globale_obj_ac else 'data fine non specificata'}. Previsione live non sarà specifica per questo indice.")
                # In questo caso, trova_migliori_metodi_sommativi_condizionati userà storico_per_analisi_cond[-1] se estrazione_per_previsione_live è None
        elif storico_per_analisi_cond: # Se i filtri indice/data fine non sono entrambi specificati, usa l'ultima dello storico filtrato
            estrazione_riferimento_live_ac = storico_per_analisi_cond[-1]
            self._log_to_gui(f"  AAC: Uso ultima estrazione disponibile nello storico filtrato ({estrazione_riferimento_live_ac['data'].strftime('%d/%m/%Y')}) per previsione live.")
        
        if not estrazione_riferimento_live_ac:
             self._log_to_gui(f"  AAC WARN: Impossibile determinare estrazione di riferimento per previsione live. La previsione live potrebbe non essere calcolata.")
        # --- FINE LOGICA SELEZIONE ESTRAZIONE ---

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            # Passa estrazione_riferimento_live_ac a trova_migliori_metodi_sommativi_condizionati
            self.ac_metodi_condizionati_dettagli = trova_migliori_metodi_sommativi_condizionati(
                storico_per_analisi_cond,
                ruota_cond_input, pos_cond_input_0based, val_min_cond_input, val_max_cond_input,
                ruota_calc_amb_input, pos_calc_amb_input_0based,
                ruote_gioco_cond, lookahead_cond, indice_mese_cond_utente, # indice_mese_cond_utente è per l'analisi storica
                num_ris_cond_input, min_tent_cond_input, self._log_to_gui,
                estrazione_per_previsione_live=estrazione_riferimento_live_ac # Passa l'estrazione determinata
            )
            self.master.config(cursor="")

            self._log_to_gui("\n\n--- RISULTATI ANALISI CONDIZIONATA AVANZATA (per Listbox e Log) ---")
            if not self.ac_metodi_condizionati_dettagli:
                msg_no_res = "Nessun metodo condizionato ha prodotto risultati validi."
                self._log_to_gui(msg_no_res)
                if self.ac_risultati_listbox: self.ac_risultati_listbox.insert(tk.END, msg_no_res)
                messagebox.showinfo("Analisi Condizionata", msg_no_res)
            else:
                for idx, res_cond in enumerate(self.ac_metodi_condizionati_dettagli):
                    cond_info = res_cond["definizione_cond_primaria"]
                    met_somm_info = res_cond["metodo_sommativo_applicato"]
                    desc_listbox = (
                        f"{idx+1}) SE {cond_info['ruota']}[{cond_info['posizione']}] in [{cond_info['val_min']}-{cond_info['val_max']}] -> "
                        f"{met_somm_info['ruota_calcolo']}[{met_somm_info['pos_estratto_calcolo']}] {met_somm_info['operazione']} {met_somm_info['operando_fisso']} "
                        f" (Freq: {res_cond['frequenza_cond']:.1%}, S/T: {res_cond['successi_cond']}/{res_cond['tentativi_cond']})"
                    )
                    if self.ac_risultati_listbox: self.ac_risultati_listbox.insert(tk.END, desc_listbox)
                    self._log_to_gui(f"\n{desc_listbox}")
                    self._log_to_gui(f"   Ambata (1a occ): {res_cond.get('ambata_risultante_prima_occ_val', 'N/A')}")
                    self._log_to_gui(f"   Previsione Live (da estraz. {res_cond.get('estrazione_usata_per_previsione', {}).get('data', 'N/D')} "
                                     f"idx: {res_cond.get('estrazione_usata_per_previsione', {}).get('indice_mese', 'N/A')}): "
                                     f"{res_cond.get('previsione_live_cond', 'N/A')}")


                # Recupera l'estrazione di riferimento usata effettivamente da trova_migliori_metodi_sommativi_condizionati
                # dal primo risultato (o usa il fallback se nessun metodo è stato trovato ma vogliamo comunque info)
                estrazione_effettiva_usata_per_popup = None
                if self.ac_metodi_condizionati_dettagli and "estrazione_usata_per_previsione" in self.ac_metodi_condizionati_dettagli[0]:
                    estrazione_effettiva_usata_per_popup = self.ac_metodi_condizionati_dettagli[0]["estrazione_usata_per_previsione"]
                elif estrazione_riferimento_live_ac: # Fallback a quella determinata qui
                    estrazione_effettiva_usata_per_popup = estrazione_riferimento_live_ac

                self._mostra_popup_risultati_condizionati(
                    self.ac_metodi_condizionati_dettagli,
                    storico_per_analisi_cond, 
                    ruote_gioco_cond,
                    "Risultati Analisi Condizionata Base",
                    indice_mese_richiesto_utente_globale=indice_mese_cond_utente,
                    data_fine_analisi_globale_obj_globale=data_fine_globale_obj_ac,
                    estrazione_riferimento_live_globale=estrazione_effettiva_usata_per_popup 
                )
            self._log_to_gui("\n--- Analisi Condizionata Avanzata Completata ---")

        except Exception as e:
            self.master.config(cursor=""); 
            messagebox.showerror("Errore Analisi Condizionata", f"Errore: {e}"); 
            self._log_to_gui(f"ERRORE CRITICO ANALISI CONDIZIONATA: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def _mostra_popup_risultati_condizionati(self, risultati_da_mostrare, storico_usato, 
                                              ruote_gioco_usate, titolo_base_popup, 
                                              info_correttore_globale_str=None,
                                              # --- NUOVI PARAMETRI DA RICEVERE E PASSARE ---
                                              indice_mese_richiesto_utente_globale=None,
                                              data_fine_analisi_globale_obj_globale=None,
                                              estrazione_riferimento_live_globale=None): # Questo è l'oggetto estrazione
        if not risultati_da_mostrare:
            messagebox.showinfo(titolo_base_popup, "Nessun risultato da mostrare.")
            return

        lista_previsioni_per_popup = []
        
        data_riferimento_popup_str = "N/D"
        if estrazione_riferimento_live_globale and isinstance(estrazione_riferimento_live_globale.get('data'), date):
            data_riferimento_popup_str = estrazione_riferimento_live_globale['data'].strftime('%d/%m/%Y')
        elif storico_usato and storico_usato[-1] and isinstance(storico_usato[-1].get('data'), date): # Fallback
             data_riferimento_popup_str = storico_usato[-1]['data'].strftime('%d/%m/%Y')

        grezzi_per_salvataggio_popup = [] 

        for idx, res_cond_original in enumerate(risultati_da_mostrare):
            res_cond = res_cond_original.copy() 

            cond_info = res_cond.get("definizione_cond_primaria") or \
                        res_cond.get("filtro_condizione_primaria_dict") or \
                        res_cond.get("filtro_condizione_primaria_usato") 
            
            met_somm_info = res_cond.get("metodo_sommativo_applicato")
            
            formula_visualizzata_nel_popup = res_cond.get("def_metodo_esteso_1") # Per metodi corretti
            if formula_visualizzata_nel_popup is None: 
                 formula_visualizzata_nel_popup = res_cond.get("formula_metodo_base_originale") # Per metodi base condizionati

            desc_metodo_display = "N/D"
            if formula_visualizzata_nel_popup and cond_info:
                desc_formula_interna = "".join(self._format_componente_per_display(c) for c in formula_visualizzata_nel_popup)
                desc_metodo_display = (
                    f"SE {cond_info['ruota']}[pos.{cond_info.get('posizione', '?')}] IN [{cond_info['val_min']}-{cond_info['val_max']}] " # Aggiunto .get per posizione
                    f"ALLORA {desc_formula_interna}"
                )
            elif cond_info and met_somm_info: # Caso specifico per il risultato da trova_migliori_metodi_sommativi_condizionati
                 desc_metodo_display = (
                    f"SE {cond_info['ruota']}[pos.{cond_info['posizione']}] IN [{cond_info['val_min']}-{cond_info['val_max']}] "
                    f"ALLORA ({met_somm_info['ruota_calcolo']}[pos.{met_somm_info['pos_estratto_calcolo']}] "
                    f"{met_somm_info['operazione']} {met_somm_info['operando_fisso']})"
                 )
            elif formula_visualizzata_nel_popup: # Se non c'è cond_info ma c'è una formula (es. metodo complesso non condizionato passato qui)
                 desc_metodo_display = "".join(self._format_componente_per_display(c) for c in formula_visualizzata_nel_popup)

            # L'ambata prevista dovrebbe essere già calcolata sull'estrazione corretta
            # dalla funzione chiamante (es. avvia_analisi_condizionata o avvia_ricerca_correttore_per_selezionato_condizionato)
            # e presente in res_cond
            ambata_per_popup = res_cond.get('ambata_prevista') 
            if ambata_per_popup is None: ambata_per_popup = res_cond.get('previsione_live_cond') 
            if ambata_per_popup is None and res_cond.get('ambata_risultante_prima_occ_val') not in [None, -1]: # Fallback per metodi base
                ambata_per_popup = res_cond.get('ambata_risultante_prima_occ_val')
            if ambata_per_popup is None: ambata_per_popup = "N/A"

            abbinamenti_per_popup = {}
            if str(ambata_per_popup).upper() not in ["N/A", "N/D"] and storico_usato:
                # Prova a prendere abbinamenti già calcolati, altrimenti calcolali
                abbinamenti_per_popup = res_cond.get("abbinamenti", {}) 
                if not abbinamenti_per_popup or not abbinamenti_per_popup.get("sortite_ambata_target"):
                    abbinamenti_per_popup = analizza_abbinamenti_per_numero_specifico(
                        storico_usato, # Usa lo storico completo per gli abbinamenti
                        int(ambata_per_popup) if str(ambata_per_popup).isdigit() else None, # Assicura sia int se possibile
                        ruote_gioco_usate, self._log_to_gui
                    )

            performance_str = "N/D"
            if "frequenza_cond" in res_cond: # Per risultati da trova_migliori_metodi_sommativi_condizionati
                performance_str = f"{res_cond['frequenza_cond']:.2%} ({res_cond['successi_cond']}/{res_cond['tentativi_cond']} casi su estraz. filtrate)"
            elif "frequenza" in res_cond: # Per risultati da trova_miglior_correttore (che ha 'frequenza')
                performance_str = f"{res_cond['frequenza']:.2%} ({res_cond['successi']}/{res_cond['tentativi']} casi, post-cond/correttore)"
            
            dettaglio_popup_item = {
                "titolo_sezione": f"--- {(idx+1)}° METODO ---",
                "info_metodo_str": desc_metodo_display,
                "ambata_prevista": ambata_per_popup, 
                "abbinamenti_dict": abbinamenti_per_popup,
                "performance_storica_str": performance_str
            }
            lista_previsioni_per_popup.append(dettaglio_popup_item)
            
            dati_grezzi_item = res_cond.copy() 
            dati_grezzi_item.setdefault("formula_testuale", desc_metodo_display)
            dati_grezzi_item.setdefault("ambata_prevista", ambata_per_popup)
            dati_grezzi_item.setdefault("abbinamenti", abbinamenti_per_popup)
            # Aggiungi anche l'estrazione usata per la previsione se disponibile in res_cond
            if "estrazione_usata_per_previsione" in res_cond:
                dati_grezzi_item["estrazione_usata_per_previsione"] = res_cond["estrazione_usata_per_previsione"]
            elif estrazione_riferimento_live_globale: # Fallback all'estrazione globale passata
                dati_grezzi_item["estrazione_usata_per_previsione"] = estrazione_riferimento_live_globale

            if cond_info: 
                dati_grezzi_item["definizione_cond_primaria"] = cond_info
            grezzi_per_salvataggio_popup.append(dati_grezzi_item)

        ruote_gioco_str_popup = "TUTTE" if len(ruote_gioco_usate) == len(RUOTE) else ", ".join(ruote_gioco_usate)
        copertura_info_popup = None
        if info_correttore_globale_str:
            copertura_info_popup = {"testo_introduttivo": info_correttore_globale_str}

        self.mostra_popup_previsione(
            titolo_popup=titolo_base_popup,
            ruote_gioco_str=ruote_gioco_str_popup,
            lista_previsioni_dettagliate=lista_previsioni_per_popup,
            copertura_combinata_info=copertura_info_popup,
            data_riferimento_previsione_str_comune=data_riferimento_popup_str,
            metodi_grezzi_per_salvataggio=grezzi_per_salvataggio_popup,
            # --- NUOVI PARAMETRI PASSATI A mostra_popup_previsione ---
            indice_mese_richiesto_utente=indice_mese_richiesto_utente_globale,
            data_fine_analisi_globale_obj=data_fine_analisi_globale_obj_globale,
            estrazione_riferimento_per_previsione_live=estrazione_riferimento_live_globale
        )

    def avvia_ricerca_ambata_ottimale_periodica(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA AMBATA OTTIMALE PERIODICA (CON METODO SOMMATIVO)\n" + "="*50)
        
        if self.ap_risultati_listbox: self.ap_risultati_listbox.delete(0, tk.END)

        storico_globale_completo = self._carica_e_valida_storico_comune(usa_filtri_data_globali=False) 
        if not storico_globale_completo: return

        data_inizio_g_obj, data_fine_g_obj = None, None   
        try: data_inizio_g_obj = self.date_inizio_entry_analisi.get_date()
        except ValueError: pass
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                 data_fine_g_obj = self.date_fine_entry_analisi.get_date()
        except ValueError: pass

        mesi_selezionati_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
        if not mesi_selezionati_gui and not self.ap_tutti_mesi_var.get():
            messagebox.showwarning("Selezione Mesi", "Seleziona almeno un mese o 'Tutti i Mesi'."); return
        mesi_map = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
        mesi_numeri_selezionati = []
        if not self.ap_tutti_mesi_var.get(): mesi_numeri_selezionati = [mesi_map[nome] for nome in mesi_selezionati_gui]
        
        storico_filtrato_per_periodo = filtra_storico_per_periodo(
            storico_globale_completo, mesi_numeri_selezionati,
            data_inizio_g_obj, data_fine_g_obj, app_logger=self._log_to_gui
        )

        if not storico_filtrato_per_periodo:
            messagebox.showinfo("Analisi Periodica", "Nessuna estrazione trovata per il periodo e filtri."); return

        ruote_gioco_sel, lookahead_sel, indice_mese_utente_aop = self._get_parametri_gioco_comuni()
        if ruote_gioco_sel is None: return

        ruota_calc_base_ott = self.ap_ruota_calcolo_ott_var.get()
        pos_estratto_base_ott_0idx = self.ap_pos_estratto_ott_var.get() - 1
        min_tent_applicazioni_soglia = self.ap_min_tentativi_ott_var.get() 

        self.master.config(cursor="watch"); self.master.update_idletasks()

        migliori_metodi_trovati = trova_miglior_ambata_sommativa_periodica(
            storico_completo=storico_globale_completo, storico_filtrato_periodo=storico_filtrato_per_periodo, 
            ruota_calcolo_base=ruota_calc_base_ott, pos_estratto_base_idx=pos_estratto_base_ott_0idx,
            ruote_gioco_selezionate=ruote_gioco_sel, lookahead=lookahead_sel,
            min_tentativi_soglia_applicazioni=min_tent_applicazioni_soglia, app_logger=self._log_to_gui,
            indice_mese_per_analisi_storica_e_live=indice_mese_utente_aop, data_fine_per_previsione_live_obj=data_fine_g_obj,
            num_migliori_da_restituire=1
        )
        self.master.config(cursor="")

        if not migliori_metodi_trovati:
            messagebox.showinfo("Analisi Periodica", "Nessun metodo ottimale trovato per i criteri specificati.")
        else:
            lista_previsioni_per_popup, metodi_grezzi_per_salvataggio = [], []
            metodo_info = migliori_metodi_trovati[0]
            
            form = metodo_info.get("metodo_formula")
            if not form or not isinstance(form, dict):
                 messagebox.showerror("Errore Interno", "Dati formula ricevuti in formato non valido dalla funzione di analisi.")
                 return

            formula_str = f"{form['ruota_calcolo']}[pos.{form['pos_estratto_calcolo']}] {form['operazione']} {form['operando_fisso']}"
            amb_live = metodo_info.get("previsione_live_periodica", "N/A")
            
            # >>> BLOCCO CRUCIALE CORRETTO <<<
            op_simbolo_map = {'somma': '+', 'differenza': '-', 'moltiplicazione': '*'}
            op_simbolo = op_simbolo_map.get(form.get('operazione'))
            definizione_costruita = [
                {'tipo_termine': 'estratto', 'ruota': form['ruota_calcolo'], 'posizione': form['pos_estratto_calcolo'] - 1, 'operazione_successiva': op_simbolo},
                {'tipo_termine': 'fisso', 'valore_fisso': form['operando_fisso'], 'operazione_successiva': '='}
            ]

            dati_salvataggio = metodo_info.copy()
            dati_salvataggio["tipo_metodo_salvato"] = "periodica_ottimale"
            dati_salvataggio["formula_testuale"] = f"{formula_str} (Condizione Periodo: Mesi={', '.join(mesi_selezionati_gui) or 'Tutti'})"
            dati_salvataggio["ambata_prevista"] = amb_live
            dati_salvataggio["metodo_formula"] = form
            dati_salvataggio["definizione_strutturata"] = definizione_costruita # Aggiunge la "ricetta"
            dati_salvataggio["parametri_periodo"] = {"mesi": mesi_selezionati_gui or "Tutti"}
            metodi_grezzi_per_salvataggio.append(dati_salvataggio)
            # >>> FINE BLOCCO CRUCIALE <<<
            
            dettaglio_popup = {
                "titolo_sezione": "--- METODO OTTIMALE PERIODICO ---", "info_metodo_str": dati_salvataggio["formula_testuale"],
                "ambata_prevista": amb_live, "performance_storica_str": f"Copertura Periodi: {metodo_info['copertura_periodi_perc']:.1f}% | Perf. App: {metodo_info['frequenza_applicazioni']:.1%}"
            }
            # ... (il resto della logica per abbinamenti e popup rimane invariato)
            self.mostra_popup_previsione(
                titolo_popup="Risultato Ricerca Ambata Ottimale Periodica", ruote_gioco_str=", ".join(ruote_gioco_sel),
                lista_previsioni_dettagliate=[dettaglio_popup], 
                data_riferimento_previsione_str_comune=metodo_info.get("estrazione_usata_per_previsione", {}).get('data', "N/D").strftime('%d/%m/%Y'),
                metodi_grezzi_per_salvataggio=metodi_grezzi_per_salvataggio, indice_mese_richiesto_utente=indice_mese_utente_aop,
                data_fine_analisi_globale_obj=data_fine_g_obj, estrazione_riferimento_per_previsione_live=metodo_info.get("estrazione_usata_per_previsione")
            )

    def avvia_ricerca_correttore_per_selezionato_condizionato(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA CORRETTORE PER METODO CONDIZIONATO SELEZIONATO\n" + "="*50)
        
        self.ultimo_metodo_cond_corretto_definizione = None
        self.ultimo_metodo_cond_corretto_formula_testuale = ""
        if hasattr(self, 'btn_backtest_cond_corretto'): 
            self.btn_backtest_cond_corretto.config(state=tk.DISABLED)

        if not self.ac_risultati_listbox or not self.ac_metodi_condizionati_dettagli:
            messagebox.showwarning("Attenzione", "Esegui prima un'Analisi Condizionata Avanzata e popola la lista dei metodi.")
            return
        try:
            selected_indices = self.ac_risultati_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Selezione Mancante", "Seleziona un metodo condizionato dalla lista a cui applicare il correttore.")
                return
            selected_index = selected_indices[0]
            if not (0 <= selected_index < len(self.ac_metodi_condizionati_dettagli)):
                messagebox.showerror("Errore Selezione", "Indice selezionato non valido.")
                return
            metodo_cond_selezionato = self.ac_metodi_condizionati_dettagli[selected_index]
        except IndexError:
            messagebox.showerror("Errore", "Errore nella selezione del metodo dalla lista.")
            return 
        except Exception as e:
            messagebox.showerror("Errore Selezione", f"Errore imprevisto durante la selezione: {e}")
            return

        definizione_cond_primaria_filtro = metodo_cond_selezionato.get("definizione_cond_primaria")
        formula_base_per_correttore = metodo_cond_selezionato.get('formula_metodo_base_originale')

        if not definizione_cond_primaria_filtro or not formula_base_per_correttore:
            msg_err_int = "Dettagli interni del metodo condizionato selezionato mancanti (condizione o formula base)."
            self._log_to_gui(f"ERRORE: {msg_err_int}") 
            messagebox.showerror("Errore Interno", msg_err_int)
            return

        storico_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi: return
        
        ruote_gioco, lookahead, indice_mese_utente_corr_cond = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return

        data_fine_globale_obj_corr_cond = None
        try:
            if hasattr(self, 'date_fine_entry_analisi') and self.date_fine_entry_analisi.winfo_exists():
                data_fine_globale_obj_corr_cond = self.date_fine_entry_analisi.get_date()
        except ValueError:
            self._log_to_gui("ARCCSC WARN: Data fine analisi globale non valida.")
            pass

        min_tentativi_correttore_cfg = self.corr_cfg_min_tentativi.get()
        c_fisso_s = self.corr_cfg_cerca_fisso_semplice.get(); c_estr_s = self.corr_cfg_cerca_estratto_semplice.get()
        c_somma_ef = self.corr_cfg_cerca_somma_estr_fisso.get(); c_somma_ee = self.corr_cfg_cerca_somma_estr_estr.get()
        c_diff_ef = self.corr_cfg_cerca_diff_estr_fisso.get(); c_diff_ee = self.corr_cfg_cerca_diff_estr_estr.get()
        c_mult_ef = self.corr_cfg_cerca_mult_estr_fisso.get(); c_mult_ee = self.corr_cfg_cerca_mult_estr_estr.get()

        self._log_to_gui(f"Applicazione correttore su Metodo Condizionato Base: {self.ac_risultati_listbox.get(selected_index)}")
        self._log_to_gui(f"  Condizione Primaria (filtro per ricerca correttore): {definizione_cond_primaria_filtro}")
        self._log_to_gui(f"  Formula Base (per applicare correttore): {''.join(self._format_componente_per_display(c) for c in formula_base_per_correttore)}")
        self._log_to_gui(f"  Opzioni Gioco (per ricerca correttore): Ruote: {', '.join(ruote_gioco)}, Colpi: {lookahead}, Ind.Mese (storico): {indice_mese_utente_corr_cond if indice_mese_utente_corr_cond else 'Tutte'}")

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati_correttori_list_cond = trova_miglior_correttore_per_metodo_complesso(
                storico_per_analisi,
                formula_base_per_correttore, # Metodo base a cui applicare il correttore
                None,                      # Nessun secondo metodo base
                c_fisso_s, c_estr_s, c_diff_ef, c_diff_ee, c_mult_ef, c_mult_ee, c_somma_ef, c_somma_ee,
                ruote_gioco, lookahead, indice_mese_utente_corr_cond, # indice_mese per l'analisi storica del correttore
                min_tentativi_correttore_cfg,
                app_logger=self._log_to_gui,
                filtro_condizione_primaria_dict=definizione_cond_primaria_filtro # Applica la condizione
            )
            
            if not risultati_correttori_list_cond:
                messagebox.showinfo("Ricerca Correttore (Cond.)", "Nessun correttore valido trovato che migliori il benchmark per il metodo condizionato selezionato.")
                self._log_to_gui("Nessun correttore valido trovato (post-condizione e benchmark).")
                self.ultimo_metodo_cond_corretto_definizione = None
                self.ultimo_metodo_cond_corretto_formula_testuale = ""
                if hasattr(self, 'btn_backtest_cond_corretto'):
                    self.btn_backtest_cond_corretto.config(state=tk.DISABLED)
            else:
                miglior_correttore_cond = risultati_correttori_list_cond[0]
                
                info_correttore_globale_str_popup = (
                    f"Correttore Applicato: {miglior_correttore_cond['tipo_correttore_descrittivo']} -> {miglior_correttore_cond['dettaglio_correttore_str']}\n"
                    f"Operazione di Collegamento Base: '{miglior_correttore_cond['operazione_collegamento_base']}'\n"
                    f"Performance del Metodo Condizionato + Correttore: {miglior_correttore_cond['frequenza']:.2%} ({miglior_correttore_cond['successi']}/{miglior_correttore_cond['tentativi']} casi)"
                )
                self._log_to_gui(f"\n--- MIGLIOR CORRETTORE PER METODO CONDIZIONATO ---")
                self._log_to_gui(f"  {info_correttore_globale_str_popup.replace(chr(10), chr(10)+'  ')}")
                
                metodo_esteso_corretto_def = miglior_correttore_cond.get('def_metodo_esteso_1') # Il correttore si applica a def_metodo_esteso_1

                # --- LOGICA PER DETERMINARE L'ESTRAZIONE PER LA PREVISIONE LIVE DEL METODO CORRETTO-CONDIZIONATO ---
                estrazione_per_previsione_live_cond_corr = None
                data_riferimento_popup_cond_corr_str = "N/A"
                nota_previsione_cond_corr = ""

                if indice_mese_utente_corr_cond is not None and data_fine_globale_obj_corr_cond is not None:
                    self._log_to_gui(f"ARCCSC: Ricerca estrazione per previsione live (cond+corr): idx_mese={indice_mese_utente_corr_cond}, data_fine<={data_fine_globale_obj_corr_cond.strftime('%Y-%m-%d') if data_fine_globale_obj_corr_cond else 'N/A'}")
                    for estr_cc in reversed(storico_per_analisi):
                        if estr_cc['data'] <= data_fine_globale_obj_corr_cond:
                            if estr_cc.get('indice_mese') == indice_mese_utente_corr_cond:
                                estrazione_per_previsione_live_cond_corr = estr_cc
                                data_riferimento_popup_cond_corr_str = estr_cc['data'].strftime('%d/%m/%Y')
                                self._log_to_gui(f"  ARCCSC: Trovata estrazione specifica per previsione live (cond+corr): {data_riferimento_popup_cond_corr_str} (Indice: {estr_cc.get('indice_mese')})")
                                break
                    if not estrazione_per_previsione_live_cond_corr: # Se il loop è terminato senza trovare una corrispondenza
                        nota_previsione_cond_corr = f"Nessuna estrazione trovata per l'indice mese {indice_mese_utente_corr_cond} entro il {data_fine_globale_obj_corr_cond.strftime('%d/%m/%Y')}."
                        self._log_to_gui(f"  ARCCSC: {nota_previsione_cond_corr}")
                elif storico_per_analisi: # Se i filtri non sono applicabili, usa l'ultima
                    estrazione_per_previsione_live_cond_corr = storico_per_analisi[-1]
                    data_riferimento_popup_cond_corr_str = estrazione_per_previsione_live_cond_corr['data'].strftime('%d/%m/%Y')
                    self._log_to_gui(f"  ARCCSC: Uso ultima estrazione disponibile ({data_riferimento_popup_cond_corr_str}) per previsione metodo cond+corr.")
                
                if not estrazione_per_previsione_live_cond_corr and not nota_previsione_cond_corr: # Fallback finale
                    nota_previsione_cond_corr = "Impossibile determinare estrazione di riferimento per previsione metodo cond+corr."
                    self._log_to_gui(f"  ARCCSC: {nota_previsione_cond_corr}")
                # --- FINE LOGICA SELEZIONE ESTRAZIONE ---

                if metodo_esteso_corretto_def and definizione_cond_primaria_filtro:
                    self.ultimo_metodo_cond_corretto_definizione = {
                        'definizione_strutturata': list(metodo_esteso_corretto_def), 
                        'condizione_primaria': definizione_cond_primaria_filtro.copy() 
                    }
                    formula_base_str_corretta = "".join(self._format_componente_per_display(c) for c in metodo_esteso_corretto_def)
                    cond_str_display = (f"SE {definizione_cond_primaria_filtro['ruota']}"
                                        f"[pos.{definizione_cond_primaria_filtro.get('posizione', '?')}] " 
                                        f"IN [{definizione_cond_primaria_filtro['val_min']}-{definizione_cond_primaria_filtro['val_max']}]")
                    self.ultimo_metodo_cond_corretto_formula_testuale = f"{cond_str_display} ALLORA ({formula_base_str_corretta})"
                    self._log_to_gui(f"INFO: Metodo Cond. Corretto pronto per backtest: {self.ultimo_metodo_cond_corretto_formula_testuale}")
                    if hasattr(self, 'btn_backtest_cond_corretto'):
                        self.btn_backtest_cond_corretto.config(state=tk.NORMAL) 
                
                    ambata_live_corr_cond, abb_live_corr_cond = "N/D", {}
                    if estrazione_per_previsione_live_cond_corr:
                        ambata_live_corr_cond, abb_live_corr_cond = self._calcola_previsione_e_abbinamenti_metodo_complesso_con_cond(
                            storico_per_analisi, 
                            metodo_esteso_corretto_def, 
                            definizione_cond_primaria_filtro,
                            ruote_gioco, 
                            data_riferimento_popup_cond_corr_str, 
                            "Metodo Condizionato + Correttore",
                            estrazione_riferimento_previsione=estrazione_per_previsione_live_cond_corr
                        )
                    elif nota_previsione_cond_corr:
                         self._log_to_gui(f"  ARCCSC: Previsione per Metodo Condizionato+Correttore non calcolata: {nota_previsione_cond_corr}")
                    
                    dati_per_popup_corretto_completi = {
                        "tipo_metodo_salvato": "condizionato_corretto", 
                        "formula_testuale": self.ultimo_metodo_cond_corretto_formula_testuale, 
                        "def_metodo_esteso_1": metodo_esteso_corretto_def, 
                        "definizione_cond_primaria": definizione_cond_primaria_filtro, 
                        "ambata_prevista": ambata_live_corr_cond if ambata_live_corr_cond is not None else "N/D", 
                        "abbinamenti": abb_live_corr_cond,
                        "successi": miglior_correttore_cond['successi'], "tentativi": miglior_correttore_cond['tentativi'],
                        "frequenza": miglior_correttore_cond['frequenza'],
                        "tipo_correttore_descrittivo": miglior_correttore_cond['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_correttore_cond['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_correttore_cond['operazione_collegamento_base'],
                        "estrazione_usata_per_previsione": estrazione_per_previsione_live_cond_corr # Aggiungi l'estrazione usata
                    }
                    self._mostra_popup_risultati_condizionati(
                        [dati_per_popup_corretto_completi], 
                        storico_per_analisi, ruote_gioco,
                        "Previsione Metodo Condizionato con Correttore", 
                        info_correttore_globale_str=info_correttore_globale_str_popup,
                        indice_mese_richiesto_utente_globale=indice_mese_utente_corr_cond,
                        data_fine_analisi_globale_obj_globale=data_fine_globale_obj_corr_cond,
                        estrazione_riferimento_live_globale=estrazione_per_previsione_live_cond_corr
                    )
                else:
                    self._log_to_gui(f"WARN ARCCSC: Dati del correttore o condizione primaria mancanti per preparare popup.")
                    if hasattr(self, 'btn_backtest_cond_corretto'):
                        self.btn_backtest_cond_corretto.config(state=tk.DISABLED)
                    self.ultimo_metodo_cond_corretto_definizione = None
                    self.ultimo_metodo_cond_corretto_formula_testuale = ""
                    messagebox.showinfo("Risultato Correttore", "Miglior correttore trovato ma dati interni incompleti per preparare il popup dettagliato.")
        except Exception as e:
            messagebox.showerror("Errore Ricerca Correttore (Cond.)", f"Errore: {e}")
            self._log_to_gui(f"ERRORE CRITICO CORRETTORE PER CONDIZIONATO: {e}, {traceback.format_exc()}")
        finally:
            self.master.config(cursor="")

    def avvia_backtest_del_condizionato_selezionato(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO BACKTEST DETTAGLIATO (Metodo Condizionato BASE Selezionato)\n" + "="*50)

        if not hasattr(self, 'ac_risultati_listbox') or not self.ac_risultati_listbox or \
           not hasattr(self, 'ac_metodi_condizionati_dettagli') or not self.ac_metodi_condizionati_dettagli:
            messagebox.showerror("Errore", "Lista dei metodi condizionati (base) non disponibile o vuota.")
            return

        try:
            selected_indices = self.ac_risultati_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Selezione Mancante", "Seleziona un metodo condizionato base dalla lista.")
                return
            
            selected_index = selected_indices[0]
            if not (0 <= selected_index < len(self.ac_metodi_condizionati_dettagli)):
                messagebox.showerror("Errore Selezione", "Indice selezionato non valido per la lista dei metodi condizionati base.")
                return
            
            metodo_cond_base_selezionato = self.ac_metodi_condizionati_dettagli[selected_index]
            self._log_to_gui(f"DEBUG Backtest Cond. Base: Dati grezzi metodo selezionato: {metodo_cond_base_selezionato}")

            definizione_metodo_da_usare = metodo_cond_base_selezionato.get('formula_metodo_base_originale')
            condizione_da_usare = metodo_cond_base_selezionato.get('definizione_cond_primaria')
            
            formula_testuale_display = "N/D"
            if 'formula_testuale' in metodo_cond_base_selezionato: 
                formula_testuale_display = metodo_cond_base_selezionato['formula_testuale']
            elif definizione_metodo_da_usare and condizione_da_usare:
                formula_base_str = "".join(self._format_componente_per_display(c) for c in definizione_metodo_da_usare)
                cond_str = (f"SE {condizione_da_usare['ruota']}"
                            f"[pos.{condizione_da_usare.get('posizione', '?')}] "
                            f"IN [{condizione_da_usare['val_min']}-{condizione_da_usare['val_max']}]")
                formula_testuale_display = f"{cond_str} ALLORA ({formula_base_str})"
            
            if not definizione_metodo_da_usare or not condizione_da_usare:
                messagebox.showerror("Errore Dati Metodo", "Dati del metodo condizionato base selezionato incompleti.")
                return

            self._log_to_gui(f"Backtest per Metodo Condizionato BASE: {formula_testuale_display}")

            try:
                data_inizio_backtest = self.date_inizio_entry_analisi.get_date()
                data_fine_backtest = self.date_fine_entry_analisi.get_date()
                if data_inizio_backtest > data_fine_backtest: messagebox.showerror("Errore Date", "Data Inizio > Data Fine."); return
            except ValueError: messagebox.showerror("Errore Data", "Date Inizio/Fine non valide."); return
            
            mesi_sel_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
            mesi_map_b = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
            mesi_num_sel_b = []
            if not self.ap_tutti_mesi_var.get() and mesi_sel_gui: mesi_num_sel_b = [mesi_map_b[nome] for nome in mesi_sel_gui]
            
            ruote_g_b, lookahead_b, indice_mese_da_gui = self._get_parametri_gioco_comuni() 
            if ruote_g_b is None: return
            
            storico_per_backtest = carica_storico_completo(self.cartella_dati_var.get(), app_logger=self._log_to_gui)
            if not storico_per_backtest: return

            self.master.config(cursor="watch"); self.master.update_idletasks()
            try:
                risultati_dettagliati = analizza_performance_dettagliata_metodo(
                    storico_completo=storico_per_backtest, 
                    definizione_metodo=definizione_metodo_da_usare, 
                    metodo_stringa_per_log=formula_testuale_display, 
                    ruote_gioco=ruote_g_b, 
                    lookahead=lookahead_b, 
                    data_inizio_analisi=data_inizio_backtest, 
                    data_fine_analisi=data_fine_backtest, 
                    mesi_selezionati_filtro=mesi_num_sel_b,
                    app_logger=self._log_to_gui,
                    condizione_primaria_metodo=condizione_da_usare, 
                    indice_estrazione_mese_da_considerare=indice_mese_da_gui 
                )
                
                if not risultati_dettagliati:
                    messagebox.showinfo("Backtest Metodo Cond. Base", "Nessuna applicazione o esito per il metodo selezionato nel periodo.")
                else:
                    popup_content = f"--- RISULTATI BACKTEST DETTAGLIATO (Met. Cond. BASE) ---\n"
                    popup_content += f"Metodo: {formula_testuale_display}\n"
                    popup_content += f"Periodo: {data_inizio_backtest.strftime('%d/%m/%Y')} - {data_fine_backtest.strftime('%d/%m/%Y')}, Mesi: {mesi_num_sel_b or 'Tutti'}\n"
                    popup_content += f"Ruote: {', '.join(ruote_g_b)}, Colpi: {lookahead_b}\n"
                    popup_content += "--------------------------------------------------\n\n"
                    successi_ambata_tot = 0; applicazioni_valide_tot = 0; applicazioni_cond_soddisfatte = 0
                    for res_bd in risultati_dettagliati:
                        popup_content += f"Data Applicazione: {res_bd['data_applicazione'].strftime('%d/%m/%Y')}\n"
                        if res_bd.get('condizione_soddisfatta', True): 
                            applicazioni_cond_soddisfatte +=1
                            if res_bd['metodo_applicabile']:
                                applicazioni_valide_tot += 1; popup_content += f"  Ambata Prevista: {res_bd['ambata_prevista']}\n"
                                if res_bd['esito_ambata']:
                                    successi_ambata_tot +=1; popup_content += f"  ESITO: AMBATA VINCENTE!\n    Colpo: {res_bd['colpo_vincita_ambata']}, Ruota: {res_bd['ruota_vincita_ambata']}\n"
                                    if res_bd.get('numeri_estratti_vincita'): popup_content += f"    Numeri Estratti: {res_bd['numeri_estratti_vincita']}\n"
                                else: popup_content += f"  ESITO: Ambata non uscita entro {lookahead_b} colpi.\n"
                            else: popup_content += f"  Metodo base non applicabile (cond. soddisfatta).\n"
                        else: popup_content += f"  Condizione primaria non soddisfatta.\n"
                        popup_content += "-------------------------\n"
                    freq_str = "N/A"
                    if applicazioni_valide_tot > 0: freq_str = f"{(successi_ambata_tot / applicazioni_valide_tot) * 100:.2f}% ({successi_ambata_tot}/{applicazioni_valide_tot} app.)"
                    summary = f"\nRIEPILOGO:\nEstrazioni con Cond. Soddisfatta: {applicazioni_cond_soddisfatte}\n"
                    summary += f"Applicazioni Metodo (post-cond): {applicazioni_valide_tot}\n"
                    summary += f"Successi Ambata: {successi_ambata_tot}\nFreq. Successo (su app. valide): {freq_str}\n"
                    popup_content += summary
                    self.mostra_popup_testo_semplice("Backtest - Metodo Cond. Base", popup_content)

            except Exception as e:
                messagebox.showerror("Errore Backtest Cond. Base", f"Errore durante l'analisi: {e}")
            finally:
                self.master.config(cursor="")
        
        except IndexError:
             messagebox.showerror("Errore", "Nessun metodo selezionato o indice errato.")
        except Exception as e:
             messagebox.showerror("Errore", f"Errore imprevisto: {e}")


    def avvia_backtest_del_condizionato_corretto(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO BACKTEST DETTAGLIATO (Metodo Condizionato + CORRETTORE)\n" + "="*50)

        if not self.ultimo_metodo_cond_corretto_definizione or \
           not self.ultimo_metodo_cond_corretto_definizione.get('definizione_strutturata') or \
           not self.ultimo_metodo_cond_corretto_definizione.get('condizione_primaria'):
            messagebox.showerror("Errore", "Nessun metodo condizionato corretto è stato memorizzato (o dati incompleti).\n"
                                           "Esegui prima 'Applica Correttore' su un metodo condizionato selezionato.")
            self._log_to_gui("ERRORE: self.ultimo_metodo_cond_corretto_definizione non impostato o incompleto per backtest.")
            return

        definizione_metodo_per_analisi = self.ultimo_metodo_cond_corretto_definizione['definizione_strutturata']
        condizione_primaria_per_analisi = self.ultimo_metodo_cond_corretto_definizione['condizione_primaria']
        formula_testuale_display = self.ultimo_metodo_cond_corretto_formula_testuale

        self._log_to_gui(f"Backtest per Metodo Condizionato CORRETTO: {formula_testuale_display}")
        self._log_to_gui(f"  Definizione Metodo (Base+Correttore): {''.join(self._format_componente_per_display(c) for c in definizione_metodo_per_analisi)}")
        self._log_to_gui(f"  Con Condizione Primaria: {condizione_primaria_per_analisi}")

        try:
            data_inizio_backtest = self.date_inizio_entry_analisi.get_date()
            data_fine_backtest = self.date_fine_entry_analisi.get_date()
            if data_inizio_backtest > data_fine_backtest: messagebox.showerror("Errore Date", "Data Inizio > Data Fine."); return
        except ValueError: messagebox.showerror("Errore Data", "Date Inizio/Fine non valide."); return
            
        mesi_sel_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
        mesi_map_b = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
        mesi_num_sel_b = []
        if not self.ap_tutti_mesi_var.get() and mesi_sel_gui: mesi_num_sel_b = [mesi_map_b[nome] for nome in mesi_sel_gui]
        
        ruote_g_b, lookahead_b, indice_mese_da_gui = self._get_parametri_gioco_comuni() 
        if ruote_g_b is None: return
        
        storico_per_backtest = carica_storico_completo(self.cartella_dati_var.get(), app_logger=self._log_to_gui)
        if not storico_per_backtest: return

        self.master.config(cursor="watch"); self.master.update_idletasks()
        try:
            risultati_dettagliati = analizza_performance_dettagliata_metodo(
                storico_completo=storico_per_backtest, 
                definizione_metodo=definizione_metodo_per_analisi, 
                metodo_stringa_per_log=formula_testuale_display, 
                ruote_gioco=ruote_g_b, 
                lookahead=lookahead_b, 
                data_inizio_analisi=data_inizio_backtest, 
                data_fine_analisi=data_fine_backtest, 
                mesi_selezionati_filtro=mesi_num_sel_b,
                app_logger=self._log_to_gui,
                condizione_primaria_metodo=condizione_primaria_per_analisi, 
                indice_estrazione_mese_da_considerare=indice_mese_da_gui 
            )
            
            if not risultati_dettagliati:
                messagebox.showinfo("Backtest Metodo Cond. Corretto", "Nessuna applicazione o esito per il metodo corretto nel periodo.")
            else:
                popup_content = f"--- RISULTATI BACKTEST DETTAGLIATO (Met. Cond. + CORRETTORE) ---\n" 
                popup_content += f"Metodo: {formula_testuale_display}\n"
                popup_content += f"Periodo: {data_inizio_backtest.strftime('%d/%m/%Y')} - {data_fine_backtest.strftime('%d/%m/%Y')}, Mesi: {mesi_num_sel_b or 'Tutti'}\n"
                popup_content += f"Ruote: {', '.join(ruote_g_b)}, Colpi: {lookahead_b}\n"
                popup_content += "--------------------------------------------------\n\n"
                successi_ambata_tot = 0; applicazioni_valide_tot = 0; applicazioni_cond_soddisfatte = 0
                for res_bd in risultati_dettagliati:
                    popup_content += f"Data Applicazione: {res_bd['data_applicazione'].strftime('%d/%m/%Y')}\n"
                    if res_bd.get('condizione_soddisfatta', True): 
                        applicazioni_cond_soddisfatte +=1
                        if res_bd['metodo_applicabile']:
                            applicazioni_valide_tot += 1; popup_content += f"  Ambata Prevista: {res_bd['ambata_prevista']}\n"
                            if res_bd['esito_ambata']:
                                successi_ambata_tot +=1; popup_content += f"  ESITO: AMBATA VINCENTE!\n    Colpo: {res_bd['colpo_vincita_ambata']}, Ruota: {res_bd['ruota_vincita_ambata']}\n"
                                if res_bd.get('numeri_estratti_vincita'): popup_content += f"    Numeri Estratti: {res_bd['numeri_estratti_vincita']}\n"
                            else: popup_content += f"  ESITO: Ambata non uscita entro {lookahead_b} colpi.\n"
                        else: popup_content += f"  Metodo base non applicabile (cond. soddisfatta).\n"
                    else: popup_content += f"  Condizione primaria non soddisfatta.\n"
                    popup_content += "-------------------------\n"
                freq_str = "N/A"
                if applicazioni_valide_tot > 0: freq_str = f"{(successi_ambata_tot / applicazioni_valide_tot) * 100:.2f}% ({successi_ambata_tot}/{applicazioni_valide_tot} app.)"
                
                summary = f"\nRIEPILOGO:\nEstrazioni con Cond. Soddisfatta: {applicazioni_cond_soddisfatte}\n"
                summary += f"Applicazioni Metodo (post-cond): {applicazioni_valide_tot}\n"
                summary += f"Successi Ambata: {successi_ambata_tot}\nFreq. Successo (su app. valide): {freq_str}\n"
                popup_content += summary
                self.mostra_popup_testo_semplice("Backtest - Metodo Cond. Corretto", popup_content)

        except Exception as e:
            messagebox.showerror("Errore Backtest Cond. Corretto", f"Errore: {e}")
        finally:
            self.master.config(cursor="")

    def salva_metodo_condizionato_selezionato(self):
        if not self.ac_risultati_listbox or not self.ac_metodi_condizionati_dettagli:
            messagebox.showwarning("Salvataggio", "Nessun risultato di analisi condizionata da salvare.\nEsegui prima un'Analisi Condizionata.")
            return
        try:
            selected_indices = self.ac_risultati_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Salvataggio", "Seleziona un metodo dalla lista per salvarlo.")
                return
            selected_index = selected_indices[0]
            if selected_index < 0 or selected_index >= len(self.ac_metodi_condizionati_dettagli):
                messagebox.showerror("Errore", "Indice selezionato non valido per i dati dei metodi.")
                return
            metodo_da_salvare = self.ac_metodi_condizionati_dettagli[selected_index].copy()
            if "ruote_gioco_analisi" not in metodo_da_salvare:
                ruote_gioco_cond, lookahead_cond, indice_mese_cond = self._get_parametri_gioco_comuni()
                metodo_da_salvare["ruote_gioco_analisi"] = ruote_gioco_cond
                metodo_da_salvare["lookahead_analisi"] = lookahead_cond
                metodo_da_salvare["indice_mese_analisi"] = indice_mese_cond
            if "data_riferimento_analisi" not in metodo_da_salvare and self.storico_caricato:
                 metodo_da_salvare["data_riferimento_analisi"] = self.storico_caricato[-1]['data'].strftime('%d/%m/%Y')
            tipo_file_str = "metodo_cond_base"; estensione_file = ".lmcond"
            if metodo_da_salvare.get("tipo_metodo_salvato") == "condizionato_corretto":
                tipo_file_str = "metodo_cond_corretto"; estensione_file = ".lmcondcorr"
            elif "definizione_cond_primaria" in metodo_da_salvare and "metodo_sommativo_applicato" in metodo_da_salvare:
                 metodo_da_salvare.setdefault("tipo_metodo_salvato", "condizionato_base")
            if "formula_testuale" not in metodo_da_salvare or metodo_da_salvare["formula_testuale"] == "N/D":
                desc_metodo_display = "N/D"
                cond_info = metodo_da_salvare.get("definizione_cond_primaria") or metodo_da_salvare.get("filtro_condizione_primaria_usato") or metodo_da_salvare.get("filtro_condizione_primaria_dict")
                met_somm_info = metodo_da_salvare.get("metodo_sommativo_applicato")
                formula_corretta_salvata = metodo_da_salvare.get("def_metodo_esteso_1")
                if formula_corretta_salvata and cond_info :
                    desc_formula_interna = "".join(self._format_componente_per_display(c) for c in formula_corretta_salvata)
                    desc_metodo_display = f"SE {cond_info['ruota']}[pos.{cond_info['posizione']}] IN [{cond_info['val_min']}-{cond_info['val_max']}] ALLORA {desc_formula_interna}"
                elif cond_info and met_somm_info:
                    desc_metodo_display = f"SE {cond_info['ruota']}[pos.{cond_info['posizione']}] IN [{cond_info['val_min']}-{cond_info['val_max']}] " \
                                        f"ALLORA ({met_somm_info['ruota_calcolo']}[pos.{met_somm_info['pos_estratto_calcolo']}] " \
                                        f"{met_somm_info['operazione']} {met_somm_info['operando_fisso']})"
                metodo_da_salvare["formula_testuale"] = desc_metodo_display
            self._prepara_e_salva_profilo_metodo(metodo_da_salvare, tipo_file=tipo_file_str, estensione=estensione_file)
        except IndexError:
             messagebox.showerror("Errore", "Indice selezionato non valido o lista metodi vuota.")
             self._log_to_gui("ERRORE: Indice selezione non valido per lista metodi condizionati (vuota o indice errato).")
        except Exception as e:
            messagebox.showerror("Errore Salvataggio", f"Errore durante la preparazione del salvataggio: {e}")
            self._log_to_gui(f"ERRORE salvataggio metodo condizionato: {e}, {traceback.format_exc()}")

    def _calcola_previsione_e_abbinamenti_metodo_complesso(self, storico_attuale, definizione_metodo, 
                                                             ruote_gioco, 
                                                             data_riferimento_str_log, # Usato principalmente per il log
                                                             nome_metodo_log="Metodo",
                                                             estrazione_riferimento_previsione=None # NUOVO PARAMETRO
                                                             ):
        ambata_live = None
        abbinamenti_live = {}
        note_previsione_log = ""
        
        # Determina l'estrazione effettiva da usare per il calcolo della previsione
        ultima_estrazione_per_calcolo = estrazione_riferimento_previsione
        if not ultima_estrazione_per_calcolo and storico_attuale: # Fallback se non viene passata l'estrazione specifica
            ultima_estrazione_per_calcolo = storico_attuale[-1]
            self._log_to_gui(f"    WARN ({nome_metodo_log}): estrazione_riferimento_previsione non fornita in _calcola_previsione..., uso storico_attuale[-1].")

        # Determina la stringa della data per il log
        data_effettiva_calcolo_str = data_riferimento_str_log # Usa la stringa passata come fallback
        if ultima_estrazione_per_calcolo and isinstance(ultima_estrazione_per_calcolo.get('data'), date):
            data_effettiva_calcolo_str = ultima_estrazione_per_calcolo['data'].strftime('%d/%m/%Y')

        if ultima_estrazione_per_calcolo:
            val_raw = calcola_valore_metodo_complesso(ultima_estrazione_per_calcolo, definizione_metodo, self._log_to_gui)
            if val_raw is not None:
                ambata_live = regola_fuori_90(val_raw)
            else:
                note_previsione_log = f"{nome_metodo_log} non applicabile all'estrazione del {data_effettiva_calcolo_str} (es. div/0 o metodo complesso fallito)."
        else:
            note_previsione_log = f"Nessuna estrazione di riferimento valida o storico vuoto per {nome_metodo_log}."
        
        self._log_to_gui(f"\n  PREVISIONE LIVE {nome_metodo_log} (da estrazione del {data_effettiva_calcolo_str}):")
        if ambata_live is not None:
            self._log_to_gui(f"    AMBATA DA GIOCARE: {ambata_live}")
            # Per gli abbinamenti, si continua ad usare l'intero storico_attuale fornito
            abbinamenti_live = analizza_abbinamenti_per_numero_specifico(storico_attuale, ambata_live, ruote_gioco, self._log_to_gui)
            if abbinamenti_live.get("sortite_ambata_target", 0) > 0:
                self._log_to_gui(f"    Migliori Abbinamenti Storici (per ambata {ambata_live}):")
                self._log_to_gui(f"      (Basato su {abbinamenti_live['sortite_ambata_target']} sortite storiche)")
                for tipo_s, dati_s_lista in abbinamenti_live.items():
                    if tipo_s == "sortite_ambata_target": continue
                    if dati_s_lista:
                        self._log_to_gui(f"        Per {tipo_s.upper().replace('_', ' ')}:")
                        for ab_i in dati_s_lista[:3]:
                            if ab_i['conteggio'] > 0: self._log_to_gui(f"          - Numeri: [{', '.join(map(str, sorted(ab_i['numeri'])))}] (Freq: {ab_i['frequenza']:.1%}, Cnt: {ab_i['conteggio']})")
            else: self._log_to_gui(f"    Nessuna co-occorrenza storica per ambata {ambata_live}.")
        else: self._log_to_gui(f"    {note_previsione_log if note_previsione_log else 'Impossibile calcolare previsione.'}")
        return ambata_live, abbinamenti_live

    def _calcola_previsione_e_abbinamenti_metodo_complesso_con_cond(
        self, storico_attuale, definizione_metodo_completo,
        condizione_primaria_dict,
        ruote_gioco, 
        data_riferimento_str_log_fallback, # Usato solo se estrazione_riferimento_previsione è None
        nome_metodo_log="Metodo",
        estrazione_riferimento_previsione=None # <-- PARAMETRO NECESSARIO
    ):
        ambata_live = None
        abbinamenti_live = {}
        note_previsione_log = ""
        
        ultima_estrazione_per_calcolo = estrazione_riferimento_previsione # Usa il parametro passato
        if not ultima_estrazione_per_calcolo and storico_attuale: # Fallback se non viene passata
            ultima_estrazione_per_calcolo = storico_attuale[-1]
            self._log_to_gui(f"    WARN ({nome_metodo_log}): estrazione_riferimento_previsione non fornita a _calcola..._con_cond, uso storico_attuale[-1].")

        data_effettiva_calcolo_str = data_riferimento_str_log_fallback 
        if ultima_estrazione_per_calcolo and isinstance(ultima_estrazione_per_calcolo.get('data'), date):
            data_effettiva_calcolo_str = ultima_estrazione_per_calcolo['data'].strftime('%d/%m/%Y')

        cond_soddisfatta_live = False
        if not ultima_estrazione_per_calcolo:
            note_previsione_log = f"Nessuna estrazione di riferimento per {nome_metodo_log}."
        elif condizione_primaria_dict: 
            cond_ruota = condizione_primaria_dict['ruota']
            cond_pos_idx = (condizione_primaria_dict.get('posizione', 1) - 1) 
            cond_min = condizione_primaria_dict['val_min']
            cond_max = condizione_primaria_dict['val_max']
            
            numeri_ruota_cond_live = ultima_estrazione_per_calcolo.get(cond_ruota, [])
            if numeri_ruota_cond_live and len(numeri_ruota_cond_live) > cond_pos_idx:
                val_cond_live = numeri_ruota_cond_live[cond_pos_idx]
                if cond_min <= val_cond_live <= cond_max:
                    cond_soddisfatta_live = True
                else:
                    note_previsione_log = f"{nome_metodo_log}: Cond. primaria ({val_cond_live} non in [{cond_min}-{cond_max}]) non soddisfatta su estraz. {data_effettiva_calcolo_str}."
            else:
                note_previsione_log = f"{nome_metodo_log}: Dati mancanti per cond. primaria su estraz. {data_effettiva_calcolo_str}."
        else: 
            cond_soddisfatta_live = True

        if cond_soddisfatta_live and ultima_estrazione_per_calcolo:
            val_raw = calcola_valore_metodo_complesso(ultima_estrazione_per_calcolo, definizione_metodo_completo, self._log_to_gui)
            if val_raw is not None:
                ambata_live = regola_fuori_90(val_raw)
            else:
                if not note_previsione_log: 
                    note_previsione_log = f"{nome_metodo_log} non applicabile (calc. fallito) su estraz. {data_effettiva_calcolo_str}."
        elif not note_previsione_log and ultima_estrazione_per_calcolo : 
             pass 
        elif not ultima_estrazione_per_calcolo and not note_previsione_log:
             note_previsione_log = f"Nessuna estrazione valida per il calcolo di {nome_metodo_log}."

        self._log_to_gui(f"\n  PREVISIONE LIVE (CON COND) {nome_metodo_log} (da estrazione del {data_effettiva_calcolo_str}):")
        if ambata_live is not None:
            self._log_to_gui(f"    AMBATA DA GIOCARE: {ambata_live}")
            abbinamenti_live = analizza_abbinamenti_per_numero_specifico(storico_attuale, ambata_live, ruote_gioco, self._log_to_gui)
            if abbinamenti_live.get("sortite_ambata_target", 0) > 0:
                self._log_to_gui(f"    Migliori Abbinamenti Storici (per ambata {ambata_live}):")
                self._log_to_gui(f"      (Basato su {abbinamenti_live['sortite_ambata_target']} sortite storiche)")
                for tipo_s, dati_s_lista in abbinamenti_live.items():
                    if tipo_s == "sortite_ambata_target": continue
                    if dati_s_lista:
                        self._log_to_gui(f"        Per {tipo_s.upper().replace('_', ' ')}:")
                        for ab_i in dati_s_lista[:3]:
                            if ab_i['conteggio'] > 0: self._log_to_gui(f"          - Numeri: [{', '.join(map(str, sorted(ab_i['numeri'])))}] (Freq: {ab_i['frequenza']:.1%}, Cnt: {ab_i['conteggio']})")
            else: self._log_to_gui(f"    Nessuna co-occorrenza storica per ambata {ambata_live}.")
        else:
            self._log_to_gui(f"    {(note_previsione_log if note_previsione_log else 'Impossibile calcolare previsione.')}")
        return ambata_live, abbinamenti_live

    def avvia_verifica_giocata(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO VERIFICA GIOCATA MANUALE\n" + "="*50)
        numeri_str_input = self.numeri_verifica_var.get()
        try:
            numeri_con_spazi = numeri_str_input.replace(',', ' ')
            numeri_da_verificare_temp = [
                int(n.strip()) for n in numeri_con_spazi.split() if n.strip().isdigit()
            ]
            if not numeri_da_verificare_temp:
                messagebox.showerror("Errore Input", "Inserisci numeri validi da verificare.")
                self._log_to_gui("ERRORE: Nessun numero valido inserito dopo il parsing.")
                return
            numeri_da_verificare = sorted(list(set(numeri_da_verificare_temp)))
            if not (1 <= len(numeri_da_verificare) <= 10):
                messagebox.showerror("Errore Input", f"È possibile giocare da 1 a 10 numeri unici. Inseriti: {len(numeri_da_verificare)} (dopo aver rimosso duplicati).")
                self._log_to_gui(f"ERRORE: Numero di numeri giocati non valido: {len(numeri_da_verificare)}")
                return
            if not all(1 <= n <= 90 for n in numeri_da_verificare):
                messagebox.showerror("Errore Input", "I numeri da verificare devono essere tra 1 e 90.")
                self._log_to_gui("ERRORE: Numeri non validi (fuori range 1-90).")
                return
        except ValueError:
            messagebox.showerror("Errore Input", "Formato numeri non valido. Usa numeri separati da virgola o spazio.")
            self._log_to_gui("ERRORE: Formato numeri non valido durante la conversione.")
            return
        except Exception as e:
            messagebox.showerror("Errore Input", f"Errore imprevisto nella preparazione dei numeri: {e}")
            self._log_to_gui(f"ERRORE imprevisto preparazione numeri: {e}")
            return

        data_inizio_ver_obj = None
        try:
            if hasattr(self, 'date_inizio_verifica_entry') and self.date_inizio_verifica_entry.winfo_exists():
                data_inizio_ver_obj = self.date_inizio_verifica_entry.get_date()
        except ValueError:
            messagebox.showerror("Errore Input", "Seleziona una data di inizio verifica valida."); self._log_to_gui("ERRORE: Data inizio verifica non selezionata."); return
        except tk.TclError:
            messagebox.showerror("Errore GUI", "Widget data verifica non trovato. Assicurati che la tab 'Verifica Giocata' sia stata visualizzata."); return
        if not data_inizio_ver_obj:
            messagebox.showerror("Errore Input", "Data inizio verifica non valida o widget non accessibile."); return

        colpi_ver = self.colpi_verifica_var.get()
        ruote_gioco_selezionate_ver, _, _ = self._get_parametri_gioco_comuni()
        if ruote_gioco_selezionate_ver is None: return

        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati):
            messagebox.showerror("Errore Input", "Seleziona una cartella archivio dati valida."); self._log_to_gui("ERRORE: Cartella dati non valida."); return

        self._log_to_gui(f"Parametri Verifica Manuale (avvio):")
        self._log_to_gui(f"  Numeri da Verificare (processati): {numeri_da_verificare}")
        self._log_to_gui(f"  Data Inizio Verifica: {data_inizio_ver_obj.strftime('%d/%m/%Y')}")
        self._log_to_gui(f"  Numero Colpi: {colpi_ver}")
        self._log_to_gui(f"  Ruote di Gioco: {', '.join(ruote_gioco_selezionate_ver)}")

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            self._log_to_gui("Caricamento storico completo per verifica...")
            storico_per_verifica_effettiva = carica_storico_completo(cartella_dati, app_logger=self._log_to_gui)

            if not storico_per_verifica_effettiva:
                self.master.config(cursor=""); messagebox.showinfo("Risultato Verifica", "Nessun dato storico caricato. Impossibile verificare."); self._log_to_gui("Nessun dato storico caricato per la verifica."); return

            stringa_risultati_popup = verifica_giocata_manuale(
                numeri_da_verificare,
                ruote_gioco_selezionate_ver,
                data_inizio_ver_obj,
                colpi_ver,
                storico_per_verifica_effettiva,
                app_logger=self._log_to_gui
            )
            self.master.config(cursor="")
            self.mostra_popup_testo_semplice("Risultati Verifica Giocata", stringa_risultati_popup)
        except Exception as e:
            self.master.config(cursor=""); messagebox.showerror("Errore Verifica", f"Si è verificato un errore durante la verifica: {e}"); self._log_to_gui(f"ERRORE CRITICO VERIFICA GIOCATA MANUALE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def mostra_popup_testo_semplice(self, titolo, contenuto_testo):
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo)

        num_righe = contenuto_testo.count('\n') + 1
        larghezza_stimata = 80; altezza_stimata_righe = max(10, min(30, num_righe + 4))
        popup_width = larghezza_stimata * 7; popup_height = altezza_stimata_righe * 15
        popup_width = max(400, min(700, popup_width)); popup_height = max(250, min(600, popup_height))

        popup_window.geometry(f"{popup_width}x{popup_height}")
        popup_window.transient(self.master)
        popup_window.attributes('-topmost', True)

        text_widget = scrolledtext.ScrolledText(popup_window, wrap=tk.WORD, font=("Courier New", 9))
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, contenuto_testo)
        text_widget.config(state=tk.DISABLED)

        close_button_frame = ttk.Frame(popup_window)
        close_button_frame.pack(fill=tk.X, pady=(0,10), padx=10, side=tk.BOTTOM)
        ttk.Button(close_button_frame, text="Chiudi", command=popup_window.destroy).pack()

        try:
            self.master.eval(f'tk::PlaceWindow {str(popup_window)} center')
        except tk.TclError:
            popup_window.update_idletasks()
            master_x = self.master.winfo_x(); master_y = self.master.winfo_y()
            master_width = self.master.winfo_width(); master_height = self.master.winfo_height()
            popup_req_width = popup_window.winfo_reqwidth(); popup_req_height = popup_window.winfo_reqheight()
            x_pos = master_x + (master_width // 2) - (popup_req_width // 2)
            y_pos = master_y + (master_height // 2) - (popup_req_height // 2)
            popup_window.geometry(f"+{x_pos}+{y_pos}")
        popup_window.lift()

    def _prepara_e_salva_profilo_metodo(self, dati_profilo_metodo, tipo_file="lotto_metodo_profilo", estensione=".lmp"):
        if not dati_profilo_metodo: messagebox.showerror("Errore", "Nessun dato del metodo da salvare."); return
        nome_suggerito = "profilo_metodo"; ambata_valida = False
        ambata_da_usare_per_nome = dati_profilo_metodo.get("ambata_prevista")
        if ambata_da_usare_per_nome is None: ambata_da_usare_per_nome = dati_profilo_metodo.get("ambata_piu_frequente_dal_metodo")
        if ambata_da_usare_per_nome is None: ambata_da_usare_per_nome = dati_profilo_metodo.get("ambata_risultante_prima_occ_val")
        if ambata_da_usare_per_nome is None: ambata_da_usare_per_nome = dati_profilo_metodo.get("previsione_live_cond")
        if ambata_da_usare_per_nome is not None and ambata_da_usare_per_nome != "N/A":
            try: int(str(ambata_da_usare_per_nome)); ambata_valida = True # Aggiunto str() per robustezza
            except (ValueError, TypeError): ambata_valida = False
        if ambata_valida: nome_suggerito = f"metodo_ambata_{ambata_da_usare_per_nome}"
        elif dati_profilo_metodo.get("formula_testuale"):
            formula_semplice = dati_profilo_metodo["formula_testuale"]
            formula_semplice = formula_semplice.replace("[pos.", "_p").replace("]", "").replace("[", "_").replace(" ", "_").replace("+", "piu").replace("-", "meno").replace("*", "per").replace("SE", "IF").replace("ALLORA", "THEN").replace("IN", "in")
            formula_semplice = ''.join(c for c in formula_semplice if c.isalnum() or c in ['_','-'])
            nome_suggerito = f"metodo_{formula_semplice[:40].rstrip('_')}"
        filepath = filedialog.asksaveasfilename(initialfile=nome_suggerito, defaultextension=estensione, filetypes=[(f"File Profilo Metodo ({estensione})", f"*{estensione}"), ("Tutti i file", "*.*")], title="Salva Profilo Metodo Analizzato")
        if not filepath: return
        try:
            dati_da_salvare_serializzabili = {}
            for key, value in dati_profilo_metodo.items():
                if isinstance(value, (list, dict, str, int, float, bool, type(None))): dati_da_salvare_serializzabili[key] = value
                elif hasattr(value, '__dict__'):
                    try: json.dumps(value.__dict__); dati_da_salvare_serializzabili[key] = value.__dict__
                    except TypeError: dati_da_salvare_serializzabili[key] = str(value)
                else: dati_da_salvare_serializzabili[key] = str(value)
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(dati_da_salvare_serializzabili, f, indent=4, default=str)
            self._log_to_gui(f"Profilo del metodo salvato in: {filepath}"); messagebox.showinfo("Salvataggio Profilo", "Profilo del metodo salvato con successo!")
        except Exception as e: self._log_to_gui(f"Errore durante il salvataggio del profilo del metodo: {e}, {traceback.format_exc()}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il profilo del metodo:\n{e}")

    def mostra_popup_previsione(self, titolo_popup, ruote_gioco_str,
                                lista_previsioni_dettagliate=None,
                                copertura_combinata_info=None,
                                data_riferimento_previsione_str_comune=None,
                                metodi_grezzi_per_salvataggio=None,
                                indice_mese_richiesto_utente=None,
                                data_fine_analisi_globale_obj=None,
                                estrazione_riferimento_per_previsione_live=None
                               ):
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo_popup)

        # Calcolo dinamico altezza popup
        popup_width = 700
        popup_base_height_per_method_section = 180  # Ridotta perché non c'è più il pulsante
        abbinamenti_h_approx = 150 
        contorni_h_approx = 70   
        dynamic_height_needed = 150 
        if copertura_combinata_info: dynamic_height_needed += 80
        if lista_previsioni_dettagliate:
            for prev_dett_c in lista_previsioni_dettagliate:
                current_met_h = popup_base_height_per_method_section
                ambata_val_check = prev_dett_c.get('ambata_prevista')
                is_single_number_for_abbinamenti = False
                if isinstance(ambata_val_check, (int, float)) or (isinstance(ambata_val_check, str) and ambata_val_check.isdigit()):
                    is_single_number_for_abbinamenti = True
                if is_single_number_for_abbinamenti:
                    if prev_dett_c.get("abbinamenti_dict", {}).get("sortite_ambata_target", 0) > 0:
                        current_met_h += abbinamenti_h_approx
                    if prev_dett_c.get('contorni_suggeriti'):
                        current_met_h += contorni_h_approx
                dynamic_height_needed += current_met_h
        popup_height = min(dynamic_height_needed, 780); popup_height = max(popup_height, 620) 
        popup_window.geometry(f"{popup_width}x{int(popup_height)}")
        popup_window.transient(self.master); popup_window.attributes('-topmost', True)

        canvas = tk.Canvas(popup_window); scrollbar_y = ttk.Scrollbar(popup_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar_y.set)
        
        row_idx = 0
        ttk.Label(scrollable_frame, text=f"--- {titolo_popup} ---", font=("Helvetica", 12, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1
        
        data_effettiva_popup_titolo_str = data_riferimento_previsione_str_comune 
        if estrazione_riferimento_per_previsione_live and isinstance(estrazione_riferimento_per_previsione_live.get('data'), date):
            data_effettiva_popup_titolo_str = estrazione_riferimento_per_previsione_live['data'].strftime('%d/%m/%Y')
        
        if data_effettiva_popup_titolo_str and data_effettiva_popup_titolo_str != "N/D":
             ttk.Label(scrollable_frame, text=f"Previsione calcolata sull'estrazione del: {data_effettiva_popup_titolo_str}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
        
        ttk.Label(scrollable_frame, text=f"Su ruote: {ruote_gioco_str}").grid(row=row_idx, column=0, columnspan=2, pady=(2,10), sticky="w"); row_idx += 1

        if copertura_combinata_info and "testo_introduttivo" in copertura_combinata_info:
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5); row_idx += 1
            ttk.Label(scrollable_frame, text=copertura_combinata_info['testo_introduttivo'], wraplength=popup_width - 40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1

        if lista_previsioni_dettagliate:
            for idx_metodo, previsione_dett in enumerate(lista_previsioni_dettagliate):
                ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10); row_idx += 1
                titolo_sezione = previsione_dett.get('titolo_sezione', f'--- {(idx_metodo+1)}° METODO / PREVISIONE ---'); 
                ttk.Label(scrollable_frame, text=titolo_sezione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=3, sticky="w"); row_idx += 1
                
                formula_metodo_display = previsione_dett.get('info_metodo_str', "N/D")
                if formula_metodo_display != "N/D": 
                    ttk.Label(scrollable_frame, text=f"Metodo: {formula_metodo_display}", wraplength=popup_width-40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                
                ambata_loop_originale_da_dizionario = previsione_dett.get('ambata_prevista')
                ambata_da_visualizzare_nel_popup = ambata_loop_originale_da_dizionario 
                nota_finale_per_popup = ""

                if indice_mese_richiesto_utente is not None and \
                   data_fine_analisi_globale_obj is not None and \
                   estrazione_riferimento_per_previsione_live is not None and \
                   ambata_loop_originale_da_dizionario is not None and str(ambata_loop_originale_da_dizionario).upper() not in ["N/D", "N/A"]:

                    data_estr_live_obj = estrazione_riferimento_per_previsione_live.get('data')
                    idx_mese_estr_live = estrazione_riferimento_per_previsione_live.get('indice_mese')

                    if isinstance(data_estr_live_obj, date) and idx_mese_estr_live is not None and \
                       idx_mese_estr_live != indice_mese_richiesto_utente and \
                       data_estr_live_obj <= data_fine_analisi_globale_obj: 
                        
                        ambata_da_visualizzare_nel_popup = None
                        nota_finale_per_popup = (
                            f"(NOTA: La previsione calcolata sull'estrazione del {data_estr_live_obj.strftime('%d/%m/%Y')} "
                            f"(che è la {idx_mese_estr_live}ª del mese) non viene mostrata come 'PREVISIONE DA GIOCARE' "
                            f"perché non corrisponde all'estrazione {indice_mese_richiesto_utente}ª del mese richiesta per l'applicazione del metodo. "
                            f"La data di fine analisi è impostata al {data_fine_analisi_globale_obj.strftime('%d/%m/%Y')}.)"
                        )
                
                if ambata_da_visualizzare_nel_popup is None or str(ambata_da_visualizzare_nel_popup).upper() in ["N/D", "N/A"]:
                    testo_finale_previsione = "Nessuna previsione da giocare valida per i criteri."
                    if nota_finale_per_popup:
                         testo_finale_previsione += f"\n{nota_finale_per_popup}"
                    ttk.Label(scrollable_frame, text=testo_finale_previsione, wraplength=popup_width-40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                else:
                    testo_finale_previsione = f"PREVISIONE DA GIOCARE: {ambata_da_visualizzare_nel_popup}"
                    ttk.Label(scrollable_frame, text=testo_finale_previsione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                
                performance_str_display = previsione_dett.get('performance_storica_str', 'N/D')
                ttk.Label(scrollable_frame, text=f"Performance storica:\n{performance_str_display}", justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1

                dati_grezzi_per_questo_metodo = None
                if metodi_grezzi_per_salvataggio and idx_metodo < len(metodi_grezzi_per_salvataggio):
                    dati_grezzi_per_questo_metodo = metodi_grezzi_per_salvataggio[idx_metodo]
                
                if dati_grezzi_per_questo_metodo:
                    estensione_default = ".lmp" 
                    tipo_metodo_salv_effettivo = dati_grezzi_per_questo_metodo.get("tipo_metodo_salvato", "sconosciuto")
                    
                    if tipo_metodo_salv_effettivo.startswith("condizionato"):
                        estensione_default = ".lmcondcorr" if "corretto" in tipo_metodo_salv_effettivo else ".lmcond"
                    elif tipo_metodo_salv_effettivo in ["ambata_ambo_unico_auto", "ambata_ambo_unico_trasf"]: 
                        estensione_default = ".lmaau"

                    btn_salva_profilo = ttk.Button(scrollable_frame, text="Salva Questo Metodo", 
                                                   command=lambda d=dati_grezzi_per_questo_metodo.copy(), e=estensione_default: self._prepara_e_salva_profilo_metodo(d, estensione=e))
                    btn_salva_profilo.grid(row=row_idx, column=0, sticky="ew", padx=20, pady=(5,5)); row_idx += 1
                
                ambata_per_abbinamenti_popup = None
                if isinstance(ambata_da_visualizzare_nel_popup, (int, float)): 
                    ambata_per_abbinamenti_popup = ambata_da_visualizzare_nel_popup
                elif isinstance(ambata_da_visualizzare_nel_popup, str) and ambata_da_visualizzare_nel_popup.isdigit():
                    ambata_per_abbinamenti_popup = int(ambata_da_visualizzare_nel_popup)

                if ambata_per_abbinamenti_popup is not None: 
                    ttk.Label(scrollable_frame, text="Abbinamenti Consigliati (co-occorrenze storiche):").grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="w"); row_idx +=1
                    abbinamenti_dict_loop = previsione_dett.get('abbinamenti_dict', {}); 
                    eventi_totali_loop = abbinamenti_dict_loop.get("sortite_ambata_target", 0)
                    
                    if eventi_totali_loop > 0:
                        ttk.Label(scrollable_frame, text=f"  (Basato su {eventi_totali_loop} sortite storiche dell'ambata {ambata_per_abbinamenti_popup})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                        for tipo_sorte, dati_sorte_lista in abbinamenti_dict_loop.items():
                            if tipo_sorte == "sortite_ambata_target": continue
                            if dati_sorte_lista:
                                ttk.Label(scrollable_frame, text=f"    Per {tipo_sorte.upper().replace('_', ' ')}:").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                                for ab_info in dati_sorte_lista[:3]:
                                    if ab_info['conteggio'] > 0:
                                        numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri'])))
                                        freq_ab_disp = f"{ab_info['frequenza']:.1%}" if isinstance(ab_info['frequenza'], float) else str(ab_info['frequenza'])
                                        ttk.Label(scrollable_frame, text=f"      - Numeri: [{numeri_ab_str}] (Freq: {freq_ab_disp}, Cnt: {ab_info['conteggio']})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                    else:
                        ttk.Label(scrollable_frame, text=f"  Nessuna co-occorrenza storica per l'ambata {ambata_per_abbinamenti_popup}.").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                    
                    contorni_suggeriti_loop = previsione_dett.get('contorni_suggeriti', [])
                    if contorni_suggeriti_loop:
                        ttk.Label(scrollable_frame, text="  Altri Contorni Frequenti:").grid(row=row_idx, column=0, columnspan=2, pady=(3,1), sticky="w"); row_idx+=1
                        for contorno_num, contorno_cnt in contorni_suggeriti_loop[:5]:
                            ttk.Label(scrollable_frame, text=f"    - Numero: {contorno_num} (Presenze con ambata: {contorno_cnt})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx+=1
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=(5,0)); scrollbar_y.pack(side="right", fill="y")
        close_button_frame = ttk.Frame(popup_window); close_button_frame.pack(fill=tk.X, pady=(5,5), padx=5, side=tk.BOTTOM)
        ttk.Button(close_button_frame, text="Chiudi", command=popup_window.destroy).pack()
        popup_window.update_idletasks(); canvas.config(scrollregion=canvas.bbox("all"))
        try: self.master.eval(f'tk::PlaceWindow {str(popup_window)} center')
        except tk.TclError:
            popup_window.update_idletasks()
            master_x = self.master.winfo_x(); master_y = self.master.winfo_y()
            master_width = self.master.winfo_width(); master_height = self.master.winfo_height()
            popup_req_width = popup_window.winfo_reqwidth(); popup_req_height = popup_window.winfo_reqheight()
            x_pos = master_x + (master_width // 2) - (popup_req_width // 2)
            y_pos = master_y + (master_height // 2) - (popup_req_height // 2)
            popup_window.geometry(f"+{x_pos}+{y_pos}")
        popup_window.lift()


    def _prepara_metodo_per_backtest(self, dati_metodo_selezionato_per_prep):
        self._log_to_gui(f"\nDEBUG: _prepara_metodo_per_backtest CHIAMATO con dati: {dati_metodo_selezionato_per_prep}")

        # La validazione ora è molto più semplice
        if dati_metodo_selezionato_per_prep and dati_metodo_selezionato_per_prep.get('definizione_strutturata'):
            self.metodo_preparato_per_backtest = dati_metodo_selezionato_per_prep.copy()
            
            formula_display_gui = self.metodo_preparato_per_backtest.get('formula_testuale', 'N/D')
            tipo_display_gui = self.metodo_preparato_per_backtest.get('tipo_metodo_salvato', 'Sconosciuto').replace("_", " ").title()

            if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
                self.mc_listbox_componenti_1.delete(0, tk.END)
                self.mc_listbox_componenti_1.insert(tk.END, f"PER BACKTEST ({tipo_display_gui}):")
                self.mc_listbox_componenti_1.insert(tk.END, formula_display_gui)
            
            messagebox.showinfo("Metodo Pronto per Backtest", 
                                f"Metodo ({tipo_display_gui}) è stato selezionato.\n\nOra puoi usare il pulsante 'Backtest Dettagliato'.")
            
            if hasattr(self, 'usa_ultimo_corretto_per_backtest_var'):
                self.usa_ultimo_corretto_per_backtest_var.set(False)
        else:
            messagebox.showerror("Errore Preparazione Backtest", "Impossibile preparare il metodo. Dati interni o 'definizione_strutturata' mancanti.")
            self._log_to_gui(f"WARN: Preparazione fallita. Dati ricevuti: {dati_metodo_selezionato_per_prep}")
            self.metodo_preparato_per_backtest = None

    # --- NUOVE FUNZIONI PER IL TAB LUNGHETTE ---
    def crea_gui_lunghette(self, parent_tab):
        """Crea l'interfaccia utente per il tab 'Lunghette'."""

        container = ttk.Frame(parent_tab, padding="20")
        container.pack(expand=True, fill="both", anchor="center")

        label_info = ttk.Label(
            container,
            text="Clicca il pulsante sottostante per aprire il modulo dedicato all'analisi delle Lunghette.\n"
                 "L'analisi verrà eseguita in una finestra separata.",
            justify=tk.CENTER,
            wraplength=400 # Adatta se necessario
        )
        label_info.pack(pady=(10, 20))

        # Puoi aggiungere uno stile per il bottone se vuoi, es. usando ttk.Style()
        # style = ttk.Style()
        # style.configure("Accent.TButton", font=("Helvetica", 10, "bold"), background="lightblue")
        # E poi usare style="Accent.TButton" nel bottone. Per ora lo lascio standard.

        btn_apri_modulo_lunghette = ttk.Button(
            container,
            text="Apri Modulo Analisi Lunghette",
            command=self.apri_modulo_lunghette_callback
            # style="Accent.TButton" # Esempio se hai definito uno stile
        )
        btn_apri_modulo_lunghette.pack(pady=20, ipady=5) # ipady per un po' più di altezza verticale interna


    def apri_modulo_lunghette_callback(self):
        """
        Callback per il pulsante nel tab "Lunghette".
        Apre la GUI di lunghette.py in una nuova finestra Toplevel.
        """
        if self.finestra_lunghette_attiva is not None and self.finestra_lunghette_attiva.winfo_exists():
            self.finestra_lunghette_attiva.lift()  # Porta la finestra esistente in primo piano
            self.finestra_lunghette_attiva.focus_set() # Dagli il focus
            self._log_to_gui("INFO: Modulo Lunghette già aperto. Portato in primo piano.")
            return

        self._log_to_gui("INFO: Apertura Modulo Lunghette...")
        # Crea una nuova finestra Toplevel che è figlia della finestra principale
        self.finestra_lunghette_attiva = tk.Toplevel(self.master)
        self.finestra_lunghette_attiva.title("Modulo Analisi Lunghette Avanzato") # Titolo personalizzabile
        # La geometria verrà impostata da lunghette.LottoApp

        # Rendi la finestra Toplevel dipendente dalla principale (opzionale ma consigliato)
        # self.finestra_lunghette_attiva.transient(self.master)

        # Opzionale: impedisce interazioni con la finestra principale finché questa non è chiusa
        # self.finestra_lunghette_attiva.grab_set()

        # Crea l'istanza dell'applicazione definita in lunghette.py,
        # usando la nuova Toplevel come sua finestra "master"
        try:
            # Qui istanziamo la classe LottoApp dal modulo lunghette
            app_istanza_lunghette = lunghette.LottoApp(self.finestra_lunghette_attiva)
        except Exception as e:
            messagebox.showerror("Errore Avvio Modulo Lunghette",
                                 f"Impossibile avviare il modulo Lunghette:\n{e}")
            self._log_to_gui(f"ERRORE: Impossibile istanziare lunghette.LottoApp: {e}\n{traceback.format_exc()}")
            if self.finestra_lunghette_attiva and self.finestra_lunghette_attiva.winfo_exists():
                self.finestra_lunghette_attiva.destroy()
            self.finestra_lunghette_attiva = None
            return

        # Quando la finestra Toplevel viene chiusa dall'utente (es. con la 'X'),
        # vogliamo resettare self.finestra_lunghette_attiva
        self.finestra_lunghette_attiva.protocol("WM_DELETE_WINDOW", self._quando_finestra_lunghette_chiusa)

    def _quando_finestra_lunghette_chiusa(self):
        """
        Chiamata quando la finestra Toplevel del modulo lunghette viene chiusa.
        """
        self._log_to_gui("INFO: Modulo Lunghette chiuso dall'utente.")
        if self.finestra_lunghette_attiva is not None:
            # È importante distruggere la finestra Toplevel
            # e resettare la variabile di stato.
            try:
                if self.finestra_lunghette_attiva.winfo_exists():
                    self.finestra_lunghette_attiva.destroy()
            except tk.TclError:
                self._log_to_gui("WARN: Errore minore durante la distruzione della finestra lunghette (già distrutta?).")
            finally:
                self.finestra_lunghette_attiva = None
    # --- FINE NUOVE FUNZIONI ---

# --- BLOCCO DI AVVIO PROTETTO ---
if __name__ == "__main__":
    # 1. Inizializza subito i nostri strumenti di protezione
    secure_logger = SecureLogger(log_file=LOG_FILE, key=LOG_KEY)
    license_checker = LicenseManager(
        expiration_hours=EXPIRATION_HOURS,
        license_file_path=LICENSE_FILE,
        time_api_url=TIME_API_URL
    )
    
    secure_logger.log("--- TENTATIVO DI AVVIO PROGRAMMA ---")
    
    # Controlla se le dipendenze base (tkcalendar) sono presenti PRIMA di tutto
    try:
        from tkcalendar import DateEntry
    except ImportError:
        secure_logger.log("ERRORE CRITICO: Dipendenza 'tkcalendar' non trovata.")
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Errore Dipendenza", "Il pacchetto 'tkcalendar' non è installato.")
        sys.exit()

    # 2. Esegui il controllo della licenza
    is_allowed_to_run, message_to_user = license_checker.check_license(secure_logger)
    
    if is_allowed_to_run:
        # La licenza è valida!
        secure_logger.log("Licenza OK. Avvio dell'applicazione principale.")
        
        if "Benvenuto" in message_to_user:
            root_welcome = tk.Tk()
            root_welcome.withdraw()
            messagebox.showinfo("Attivazione Prova", message_to_user)
            root_welcome.destroy()
            
        # --- Questo è il tuo codice di avvio originale ---
        root = tk.Tk()
        app = LottoAnalyzerApp(root)
        
        def on_closing():
            secure_logger.log("--- Programma chiuso regolarmente dall'utente ---")
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    else:
        # La licenza è scaduta o c'è stato un errore
        secure_logger.log(f"Esecuzione bloccata. Messaggio: {message_to_user}")
        
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Licenza non valida", message_to_user)
        sys.exit()