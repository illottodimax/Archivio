import os
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
from itertools import combinations
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Listbox
import traceback
import lunghette
try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Errore Dipendenza", "Il pacchetto 'tkcalendar' non è installato.\nPer favore, installalo con: pip install tkcalendar")
    sys.exit()
import json

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

# --- FUNZIONI LOGICHE (aggiungi queste alle tue esistenti) ---

def regola_fuori_90(numero): # Già esistente, assicurati che sia lì
    if numero is None: return None
    if numero == 0: return 90
    while numero <= 0: numero += 90
    while numero > 90: numero -= 90
    return numero

def calcola_diametrale(numero):
    """Calcola il diametrale di un numero (differenza 45)."""
    if not (1 <= numero <= 90): return None # O gestisci diversamente
    diam = numero + 45
    return regola_fuori_90(diam)

def calcola_vertibile(numero):
    """
    Calcola il vertibile di un numero secondo le regole lottologiche:
    - Numeri a cifra singola X (0X): vertibile 0X -> X0 (se X0 <=90)
    - Numeri con cifre D U (diverse): vertibile DU -> UD
    - Numeri gemelli XX (11, 22..88): vertibile XX -> X9
    - Numero 90: vertibile 90 -> 09 (cioè 9)
    - I numeri che terminano in 0 (10, 20..80) non hanno un vertibile classico se non il 90.
      Tradizionalmente, il vertibile di X0 (con X!=9) è spesso considerato X stesso o non definito.
      Qui, se non è un gemello o 90, e finisce per 0, lo restituiamo inalterato.
      Potresti voler cambiare questo comportamento se hai una regola diversa per X0.
    """
    if not (1 <= numero <= 90):
        return None

    s_num = str(numero).zfill(2) # Es. 5 -> "05", 11 -> "11", 90 -> "90"
    decina = s_num[0]
    unita = s_num[1]

    if decina == unita:  # Numeri gemelli (00 non è possibile, 99 non è nel lotto)
        if numero == 90: # Caso speciale per 90, anche se non è tecnicamente un gemello per questa logica
            return 9
        # Per 11, 22, ..., 88
        try:
            # Il vertibile di XX è X9
            vertibile_val = int(decina + "9")
            return regola_fuori_90(vertibile_val) # regola_fuori_90 per sicurezza, anche se non dovrebbe servire qui
        except ValueError: # Non dovrebbe accadere con decina da 0 a 8
            return numero # Fallback
    elif numero == 90: # Vertibile di 90 è 9
        return 9
    elif unita == '0' and decina != '0': # Numeri come 10, 20, ..., 80
        # La tradizione qui varia. Alcuni non definiscono un vertibile,
        # altri lo considerano il numero stesso, altri 0X (X).
        # Per ora, restituiamo il numero stesso se non è 90.
        # Oppure potresti voler restituire int(unita + decina) -> int("0" + decina)
        # Esempio: vertibile di 10 -> 01 (1)
        # return regola_fuori_90(int(unita + decina))
        return numero # Comportamento attuale: 10->10, 20->20. Modifica se necessario.
    else: # Tutti gli altri casi (cifre diverse, o numeri a singola cifra non zero)
        vertibile_str = unita + decina
        return regola_fuori_90(int(vertibile_str))

def calcola_complemento_a_90(numero):
    """Calcola il complemento a 90."""
    if not (1 <= numero <= 90): return None
    return regola_fuori_90(90 - numero)

def calcola_figura(numero):
    """Calcola la figura di un numero."""
    if not (1 <= numero <= 90): return None
    if numero % 9 == 0:
        return 9
    else:
        return numero % 9

def calcola_cadenza(numero):
    """Calcola la cadenza (unità) di un numero. La cadenza 0 è per i numeri che terminano in 0."""
    if not (1 <= numero <= 90): return None
    return numero % 10

def calcola_diametrale_in_decina(numero):
    """
    Calcola il diametrale in decina di un numero secondo la tabella fornita:
    - Se l'unità è 1,2,3,4,5: numero + 5
    - Se l'unità è 6,7,8,9: numero - 5
    - Se l'unità è 0 (numeri 10,20..80): numero - 5
    - Per il numero 90 (considerato con unità "speciale" 0 nella prima decina): numero + 5 (che diventa 95 -> 5)
    """
    if not (1 <= numero <= 90):
        return None

    if numero == 90: # Caso speciale per il 90 come da tua tabella (90 -> 05)
        risultato = numero + 5 # 90 + 5 = 95
    else:
        unita = numero % 10
        if 1 <= unita <= 5: # Unità 1, 2, 3, 4, 5
            risultato = numero + 5
        elif unita == 0 or (6 <= unita <= 9): # Unità 0 (per 10..80), 6, 7, 8, 9
            risultato = numero - 5
        else: # Non dovrebbe accadere per numeri validi, ma per sicurezza
            return numero 

    return regola_fuori_90(risultato)

OPERAZIONI_SPECIALI_TRASFORMAZIONE_CORRETTORE = {
    "Fisso": lambda n: n, # Identità, il correttore è usato così com'è
    "Diametrale": calcola_diametrale,
    "Vertibile": calcola_vertibile,
    "Compl.90": calcola_complemento_a_90,
    "Figura": calcola_figura,
    "Cadenza": calcola_cadenza,
    "Diam.Decina": calcola_diametrale_in_decina # NUOVA AGGIUNTA
}

def parse_riga_estrazione(riga, nome_file_ruota, num_riga):
    try:
        parti = riga.strip().split('\t')
        if len(parti) != 7: return None, None
        data_str = parti[0]
        numeri_str = parti[2:7]
        numeri = [int(n) for n in numeri_str]
        if len(numeri) != 5: return None, None
        data_obj = datetime.strptime(data_str, "%Y/%m/%d").date()
        return data_obj, numeri
    except ValueError: return None, None
    except Exception: return None, None

def carica_storico_completo(cartella_dati, data_inizio_filtro=None, data_fine_filtro=None, app_logger=None):
    def log_message(msg):
        if app_logger: app_logger(msg)
        else: print(msg)
    log_message(f"\nCaricamento dati da: {cartella_dati}")
    if data_inizio_filtro or data_fine_filtro: log_message(f"Filtro date: Da {data_inizio_filtro or 'inizio'} a {data_fine_filtro or 'fine'}")
    if not os.path.isdir(cartella_dati): log_message(f"Errore: Cartella '{cartella_dati}' non trovata."); return []
    storico_globale = defaultdict(dict)
    file_trovati_cont = 0; righe_valide_tot = 0; righe_processate_tot = 0
    for nome_ruota_chiave in RUOTE:
        nome_file_da_cercare = f"{nome_ruota_chiave.upper()}.TXT"
        path_file = os.path.join(cartella_dati, nome_file_da_cercare)
        if not os.path.exists(path_file):
            path_file_fallback = os.path.join(cartella_dati, f"{nome_ruota_chiave}.txt")
            if os.path.exists(path_file_fallback): path_file = path_file_fallback
            else: continue
        file_trovati_cont += 1; righe_nel_file = 0; righe_valide_nel_file = 0
        try:
            with open(path_file, 'r', encoding='utf-8') as f:
                for num_riga, riga_contenuto in enumerate(f, 1):
                    righe_nel_file += 1
                    if not riga_contenuto.strip(): continue
                    data_obj, numeri = parse_riga_estrazione(riga_contenuto, os.path.basename(path_file), num_riga)
                    if data_obj and numeri:
                        if data_inizio_filtro and data_obj < data_inizio_filtro: continue
                        if data_fine_filtro and data_obj > data_fine_filtro: continue
                        righe_valide_nel_file +=1
                        if nome_ruota_chiave in storico_globale[data_obj] and storico_globale[data_obj][nome_ruota_chiave] != numeri:
                            log_message(f"Attenzione: Dati discordanti per {nome_ruota_chiave} il {data_obj}.")
                        elif nome_ruota_chiave not in storico_globale[data_obj]:
                             storico_globale[data_obj][nome_ruota_chiave] = numeri
            righe_processate_tot += righe_nel_file; righe_valide_tot += righe_valide_nel_file
        except Exception as e: log_message(f"Errore grave leggendo {path_file}: {e}")
    if file_trovati_cont == 0: log_message("Nessun file archivio valido trovato."); return []
    log_message(f"Processate {righe_valide_tot}/{righe_processate_tot} righe valide da {file_trovati_cont} file.")
    storico_ordinato = []
    date_ordinate = sorted(storico_globale.keys())
    estrazioni_mese_corrente = 0; mese_precedente, anno_precedente = None, None
    for data_obj in date_ordinate:
        if anno_precedente != data_obj.year or mese_precedente != data_obj.month:
            estrazioni_mese_corrente = 1; mese_precedente, anno_precedente = data_obj.month, data_obj.year
        else: estrazioni_mese_corrente += 1
        estrazione_completa = {'data': data_obj, 'indice_mese': estrazioni_mese_corrente}
        for r_nome in RUOTE: estrazione_completa[r_nome] = storico_globale[data_obj].get(r_nome, [])
        if any(estrazione_completa[r_n] for r_n in RUOTE): storico_ordinato.append(estrazione_completa)
    log_message(f"Caricate e ordinate {len(storico_ordinato)} estrazioni complessive valide.")
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
                                         min_tentativi_per_ambata=10, app_logger=None):
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=flush)
    risultati_ambate_grezzi = []
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
    if not storico: log_message("ERRORE: Storico vuoto, impossibile calcolare previsioni."); return [], None
    ultima_estrazione_disponibile = storico[-1]
    log_message(f"\n--- DETTAGLIO, PREVISIONE E ABBINAMENTI PER I TOP {min(max_ambate_output, len(risultati_ambate_grezzi))} METODI ---")
    for i, res_grezza in enumerate(risultati_ambate_grezzi[:max_ambate_output]):
        metodo_def = res_grezza['metodo']
        op_str_metodo = metodo_def['operazione']; operando_metodo = metodo_def['operando_fisso']
        rc_metodo = metodo_def['ruota_calcolo']; pe_metodo = metodo_def['pos_estratto_calcolo']
        log_message(f"\n--- {i+1}° METODO ---")
        log_message(f"  Formula: {rc_metodo}[pos.{pe_metodo+1}] {op_str_metodo} {operando_metodo}")
        log_message(f"  Performance Storica: {res_grezza['frequenza_ambata']:.2%} ({res_grezza['successi']}/{res_grezza['tentativi']} casi)")
        ambata_previsione_attuale = None; note_previsione = ""
        op_func_metodo = OPERAZIONI.get(op_str_metodo)
        numeri_ultima_ruota_calcolo = ultima_estrazione_disponibile.get(rc_metodo, [])
        if op_func_metodo and numeri_ultima_ruota_calcolo and len(numeri_ultima_ruota_calcolo) > pe_metodo:
            numero_base_ultima = numeri_ultima_ruota_calcolo[pe_metodo]
            try: valore_op_ultima = op_func_metodo(numero_base_ultima, operando_metodo); ambata_previsione_attuale = regola_fuori_90(valore_op_ultima)
            except ZeroDivisionError: note_previsione = "Metodo non applicabile all'ultima estrazione (divisione per zero)."
        else: note_previsione = "Dati insufficienti nell'ultima estrazione per calcolare la previsione."
        log_message(f"  PREVISIONE DA ULTIMA ESTRAZIONE ({ultima_estrazione_disponibile['data'].strftime('%d/%m/%Y')}):")
        if ambata_previsione_attuale is not None: log_message(f"    AMBATA DA GIOCARE: {ambata_previsione_attuale}")
        else: log_message(f"    {note_previsione if note_previsione else 'Impossibile calcolare previsione.'}")
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
        elif res_grezza["applicazioni_vincenti_dettagliate"]: log_message("    Nessuna ambata attuale calcolata. Abbinamenti basati su co-occorrenze non disponibili.")
        else: log_message("    Nessuna ambata attuale calcolata e nessuna applicazione vincente storica per analisi abbinamenti.")
        risultati_finali_output.append({
            "metodo": metodo_def, "ambata_piu_frequente_dal_metodo": ambata_previsione_attuale if ambata_previsione_attuale is not None else "N/D",
            "frequenza_ambata": res_grezza['frequenza_ambata'], "successi": res_grezza['successi'], "tentativi": res_grezza['tentativi'],
            "abbinamenti": abbinamenti_calcolati_finali, "applicazioni_vincenti_dettagliate": res_grezza["applicazioni_vincenti_dettagliate"]
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
        if app_logger: app_logger(msg, end=end, flush=flush)

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
        log_message("  Ricerca Correttori Semplici: Fisso Singolo...")
        for op_link_base in operazioni_collegamento_base:
            for val_fisso_corr in range(1, 91):
                termine_corr_dict = {'tipo_termine': 'fisso', 'valore_fisso': val_fisso_corr}
                dett_str = f"Fisso({val_fisso_corr})"
                risultati_correttori_candidati.append({
                    "op_link_base": op_link_base, "termine_corr_1_dict": termine_corr_dict,
                    "op_interna_corr": None, "termine_corr_2_dict": None,
                    "tipo_descrittivo": "Fisso Singolo", "dettaglio_correttore_str": dett_str
                })
    if cerca_estratto_semplice:
        log_message("  Ricerca Correttori Semplici: Estratto Singolo...")
        for op_link_base in operazioni_collegamento_base:
            for r_corr in RUOTE:
                for p_corr in range(5):
                    termine_corr_dict = {'tipo_termine': 'estratto', 'ruota': r_corr, 'posizione': p_corr}
                    dett_str = f"{r_corr}[{p_corr+1}]"
                    risultati_correttori_candidati.append({
                        "op_link_base": op_link_base, "termine_corr_1_dict": termine_corr_dict,
                        "op_interna_corr": None, "termine_corr_2_dict": None,
                        "tipo_descrittivo": "Estratto Singolo", "dettaglio_correttore_str": dett_str
                    })
    termini1_operazionali = []
    for r1 in RUOTE:
        for p1 in range(5):
            termini1_operazionali.append({'tipo_termine': 'estratto', 'ruota': r1, 'posizione': p1, 'str': f"{r1}[{p1+1}]"})
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
                        risultati_correttori_candidati.append({
                            "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                            "op_interna_corr": "+", "termine_corr_2_dict": t2_dict_op,
                            "tipo_descrittivo": "Estratto + Fisso", "dettaglio_correttore_str": dett_str
                        })
    if cerca_somma_estr_estr:
        log_message("  Ricerca Correttori Operazionali: Estratto + Estratto...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                    if t2_dict_op['tipo_termine'] == 'estratto':
                        dett_str = f"{t1_dict['str']} + {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({
                            "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                            "op_interna_corr": "+", "termine_corr_2_dict": t2_dict_op,
                            "tipo_descrittivo": "Estratto + Estratto", "dettaglio_correttore_str": dett_str
                        })
    if cerca_diff_estr_fisso:
        log_message("  Ricerca Correttori Operazionali: Estratto - Fisso...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                    if t2_dict_op['tipo_termine'] == 'fisso':
                        dett_str = f"{t1_dict['str']} - {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({
                            "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                            "op_interna_corr": "-", "termine_corr_2_dict": t2_dict_op,
                            "tipo_descrittivo": "Estratto - Fisso", "dettaglio_correttore_str": dett_str
                        })
    if cerca_diff_estr_estr:
        log_message("  Ricerca Correttori Operazionali: Estratto - Estratto...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                    if t2_dict_op['tipo_termine'] == 'estratto':
                        dett_str = f"{t1_dict['str']} - {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({
                            "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                            "op_interna_corr": "-", "termine_corr_2_dict": t2_dict_op,
                            "tipo_descrittivo": "Estratto - Estratto", "dettaglio_correttore_str": dett_str
                        })
    if cerca_mult_estr_fisso:
        log_message("  Ricerca Correttori Operazionali: Estratto * Fisso...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                    if t2_dict_op['tipo_termine'] == 'fisso' and t2_dict_op['valore_fisso'] != 0 :
                        dett_str = f"{t1_dict['str']} * {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({
                            "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                            "op_interna_corr": "*", "termine_corr_2_dict": t2_dict_op,
                            "tipo_descrittivo": "Estratto * Fisso", "dettaglio_correttore_str": dett_str
                        })
    if cerca_mult_estr_estr:
        log_message("  Ricerca Correttori Operazionali: Estratto * Estratto...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali_per_correttore:
                     if t2_dict_op['tipo_termine'] == 'estratto':
                        dett_str = f"{t1_dict['str']} * {t2_dict_op['str']}"
                        risultati_correttori_candidati.append({
                            "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                            "op_interna_corr": "*", "termine_corr_2_dict": t2_dict_op,
                            "tipo_descrittivo": "Estratto * Estratto", "dettaglio_correttore_str": dett_str
                        })
    risultati_finali_correttori = []
    log_message(f"\nValutazione di {len(risultati_correttori_candidati)} tipi di correttori candidati...")
    processed_count = 0
    for cand_corr_info in risultati_correttori_candidati:
        processed_count += 1
        if processed_count % 500 == 0: log_message(f"  Processati {processed_count}/{len(risultati_correttori_candidati)} candidati correttore...")
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
                'formula_metodo_base_originale_1': definizione_metodo_base_1,
                'formula_metodo_base_originale_2': definizione_metodo_base_2,
                'def_metodo_esteso_1': def_met_est_1_temp,
                'def_metodo_esteso_2': def_met_est_2_temp,
                'tipo_correttore_descrittivo': cand_corr_info["tipo_descrittivo"],
                'dettaglio_correttore_str': cand_corr_info["dettaglio_correttore_str"],
                'operazione_collegamento_base': op_l_base,
                'successi': s_corr_val, 'tentativi': t_corr_val, 'frequenza': f_corr_val,
                'filtro_condizione_primaria_usato': filtro_condizione_primaria_dict
            })
    log_message(f"  Completata valutazione. Trovati {len(risultati_finali_correttori)} candidati correttori con performance positiva (prima del filtro benchmark).")
    risultati_finali_correttori.sort(key=lambda x: (x['frequenza'], x['successi']), reverse=True)
    migliori_correttori_output = []
    if risultati_finali_correttori:
        if tentativi_benchmark > 0:
            for rc_f in risultati_finali_correttori:
                if rc_f['frequenza'] > freq_benchmark: migliori_correttori_output.append(rc_f)
                elif rc_f['frequenza'] == freq_benchmark and rc_f['successi'] > successi_benchmark: migliori_correttori_output.append(rc_f)
            if not migliori_correttori_output and risultati_finali_correttori: log_message("    Nessun correttore trovato è STRETTAMENTE MIGLIORE del benchmark.")
            elif migliori_correttori_output: log_message(f"    Filtrati {len(migliori_correttori_output)} correttori che migliorano il benchmark.")
        else:
            migliori_correttori_output = risultati_finali_correttori
            log_message("    Benchmark metodi base non significativo (o 0 tentativi), considero tutti i candidati correttori validi come migliorativi.")
    else: log_message("    Nessun correttore candidato trovato dopo la valutazione (min. tentativi/successi).")
    log_message(f"Ricerca correttori terminata. Restituiti {len(migliori_correttori_output)} correttori migliorativi.\n")
    return migliori_correttori_output


def trova_migliori_metodi_sommativi_condizionati(
    storico,
    ruota_cond, pos_cond_idx, val_min_cond, val_max_cond,
    ruota_calc_ambata, pos_calc_ambata_idx,
    ruote_gioco_selezionate, lookahead, indice_mese_filtro,
    num_migliori_da_restituire, min_tentativi_cond_soglia, app_logger
):
    if app_logger: app_logger(f"Avvio ricerca metodi sommativi condizionati...")
    risultati_metodi_cond = []
    operazioni_map = {'somma': '+', 'differenza': '-', 'moltiplicazione': '*'}
    for op_str_key, op_func in OPERAZIONI.items():
        op_simbolo = operazioni_map[op_str_key]
        for operando_fisso in range(1, 91):
            successi_cond_attuali = 0; tentativi_cond_attuali = 0
            applicazioni_vincenti_cond = []; ambata_fissa_prima_occ_val = -1
            prima_applicazione_valida_flag = True
            for i in range(len(storico) - lookahead):
                estrazione_corrente = storico[i]
                if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
                numeri_ruota_cond_corrente = estrazione_corrente.get(ruota_cond, [])
                if not numeri_ruota_cond_corrente or len(numeri_ruota_cond_corrente) <= pos_cond_idx: continue
                valore_estratto_per_cond = numeri_ruota_cond_corrente[pos_cond_idx]
                if not (val_min_cond <= valore_estratto_per_cond <= val_max_cond): continue
                numeri_ruota_calc_amb_corrente = estrazione_corrente.get(ruota_calc_ambata, [])
                if not numeri_ruota_calc_amb_corrente or len(numeri_ruota_calc_amb_corrente) <= pos_calc_ambata_idx: continue
                tentativi_cond_attuali += 1
                numero_base_per_ambata = numeri_ruota_calc_amb_corrente[pos_calc_ambata_idx]
                try: valore_op_ambata = op_func(numero_base_per_ambata, operando_fisso)
                except ZeroDivisionError: continue
                ambata_prevista_cond = regola_fuori_90(valore_op_ambata)
                if ambata_prevista_cond is None: continue
                if prima_applicazione_valida_flag: ambata_fissa_prima_occ_val = ambata_prevista_cond; prima_applicazione_valida_flag = False
                vincita_trovata_per_questa_applicazione = False; dettagli_vincita_singola_appl = []
                for k_lh in range(1, lookahead + 1):
                    if i + k_lh >= len(storico): break
                    estrazione_futura_cond = storico[i + k_lh]
                    for ruota_v_cond in ruote_gioco_selezionate:
                        if ambata_prevista_cond in estrazione_futura_cond.get(ruota_v_cond, []):
                            if not vincita_trovata_per_questa_applicazione: successi_cond_attuali += 1; vincita_trovata_per_questa_applicazione = True
                            dettagli_vincita_singola_appl.append({"ruota_vincita": ruota_v_cond, "numeri_ruota_vincita": estrazione_futura_cond.get(ruota_v_cond, []), "data_riscontro": estrazione_futura_cond['data'], "colpo_riscontro": k_lh})
                    if vincita_trovata_per_questa_applicazione and len(ruote_gioco_selezionate) == 1: break
                if vincita_trovata_per_questa_applicazione:
                    applicazioni_vincenti_cond.append({"data_applicazione": estrazione_corrente['data'], "estratto_condizione_trigger": valore_estratto_per_cond, "estratto_base_calcolo_ambata": numero_base_per_ambata, "operando_usato": operando_fisso, "operazione_usata": op_str_key, "ambata_prevista": ambata_prevista_cond, "riscontri": dettagli_vincita_singola_appl})
            if tentativi_cond_attuali >= min_tentativi_cond_soglia:
                frequenza_cond = successi_cond_attuali / tentativi_cond_attuali if tentativi_cond_attuali > 0 else 0.0
                formula_base_originale = [{'tipo_termine': 'estratto', 'ruota': ruota_calc_ambata, 'posizione': pos_calc_ambata_idx, 'operazione_successiva': op_simbolo}, {'tipo_termine': 'fisso', 'valore_fisso': operando_fisso, 'operazione_successiva': '='}]
                risultati_metodi_cond.append({"definizione_cond_primaria": {"ruota": ruota_cond, "posizione": pos_cond_idx + 1, "val_min": val_min_cond, "val_max": val_max_cond}, "metodo_sommativo_applicato": {"ruota_calcolo": ruota_calc_ambata, "pos_estratto_calcolo": pos_calc_ambata_idx + 1, "operazione": op_str_key, "operando_fisso": operando_fisso}, "formula_metodo_base_originale": formula_base_originale, "ambata_risultante_prima_occ_val": ambata_fissa_prima_occ_val, "successi_cond": successi_cond_attuali, "tentativi_cond": tentativi_cond_attuali, "frequenza_cond": frequenza_cond, "applicazioni_vincenti_dettagliate_cond": applicazioni_vincenti_cond, "previsione_live_cond": None})
    risultati_metodi_cond.sort(key=lambda x: (x["frequenza_cond"], x["successi_cond"]), reverse=True)
    top_risultati = risultati_metodi_cond[:num_migliori_da_restituire]
    if storico and top_risultati:
        ultima_estrazione_globale = storico[-1]
        for res in top_risultati:
            cond_res = res["definizione_cond_primaria"]; formula_base = res["formula_metodo_base_originale"]
            numeri_ruota_cond_ultima = ultima_estrazione_globale.get(cond_res['ruota'], [])
            if numeri_ruota_cond_ultima and len(numeri_ruota_cond_ultima) >= cond_res['posizione']:
                val_cond_ultima = numeri_ruota_cond_ultima[cond_res['posizione']-1]
                if cond_res['val_min'] <= val_cond_ultima <= cond_res['val_max']:
                    val_ambata_live = calcola_valore_metodo_complesso(ultima_estrazione_globale, formula_base)
                    if val_ambata_live is not None: res["previsione_live_cond"] = regola_fuori_90(val_ambata_live)
    if app_logger: app_logger(f"Ricerca metodi sommativi condizionati completata. Trovati {len(top_risultati)} risultati validi.")
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
    min_tentativi_soglia_applicazioni, # Minimo applicazioni del metodo nel periodo
    app_logger=None
):
    if app_logger: app_logger(f"Avvio ricerca miglior ambata sommativa periodica da {ruota_calcolo_base}[{pos_estratto_base_idx+1}]...")

    migliori_metodi_periodici = []

    # Identifica i periodi unici (anno, mese) presenti nello storico_filtrato_periodo
    periodi_unici_analizzati = set()
    for estr in storico_filtrato_periodo:
        periodi_unici_analizzati.add((estr['data'].year, estr['data'].month))
    num_periodi_unici_totali = len(periodi_unici_analizzati)
    if app_logger: app_logger(f"  Numero di periodi unici (anno/mese) da analizzare: {num_periodi_unici_totali}")


    for op_str, op_func in OPERAZIONI.items():
        for operando_fisso in range(1, 91):
            successi_applicazioni_totali = 0 # Conteggio grezzo dei successi
            tentativi_applicazioni_totali = 0 # Conteggio grezzo delle applicazioni

            # Per la copertura periodi
            periodi_con_successo_per_questo_metodo = set()

            applicazioni_vincenti_dettaglio = []
            ambata_costante_del_metodo = None # Per riferimento, se il metodo ne produce una costante

            map_date_a_idx_completo = {estr['data']: i for i, estr in enumerate(storico_completo)}

            for estrazione_periodo in storico_filtrato_periodo:
                data_applicazione = estrazione_periodo['data']
                anno_mese_applicazione = (data_applicazione.year, data_applicazione.month)

                numeri_ruota_calc = estrazione_periodo.get(ruota_calcolo_base, [])
                if not numeri_ruota_calc or len(numeri_ruota_calc) <= pos_estratto_base_idx:
                    continue

                numero_base_calc = numeri_ruota_calc[pos_estratto_base_idx]
                try:
                    valore_operazione = op_func(numero_base_calc, operando_fisso)
                except ZeroDivisionError:
                    continue

                ambata_prevista_corrente = regola_fuori_90(valore_operazione)
                if ambata_prevista_corrente is None:
                    continue

                if ambata_costante_del_metodo is None:
                    ambata_costante_del_metodo = ambata_prevista_corrente

                tentativi_applicazioni_totali += 1
                trovato_in_lookahead = False

                idx_partenza_lookahead = map_date_a_idx_completo.get(data_applicazione)
                if idx_partenza_lookahead is None: continue

                for k in range(1, lookahead + 1):
                    idx_futuro = idx_partenza_lookahead + k
                    if idx_futuro >= len(storico_completo): break
                    estrazione_futura = storico_completo[idx_futuro]
                    for ruota_verifica in ruote_gioco_selezionate:
                        if ambata_prevista_corrente in estrazione_futura.get(ruota_verifica, []):
                            trovato_in_lookahead = True
                            applicazioni_vincenti_dettaglio.append({
                                "data_applicazione": data_applicazione, "ambata_prevista": ambata_prevista_corrente,
                                "data_riscontro": estrazione_futura['data'], "colpo_riscontro": k,
                                "ruota_vincita": ruota_verifica
                            })
                            break
                    if trovato_in_lookahead: break

                if trovato_in_lookahead:
                    successi_applicazioni_totali += 1
                    periodi_con_successo_per_questo_metodo.add(anno_mese_applicazione) # Aggiungi il periodo (anno,mese)

            if tentativi_applicazioni_totali >= min_tentativi_soglia_applicazioni:
                frequenza_applicazioni = successi_applicazioni_totali / tentativi_applicazioni_totali if tentativi_applicazioni_totali > 0 else 0.0

                copertura_periodi = 0.0
                if num_periodi_unici_totali > 0:
                    copertura_periodi = (len(periodi_con_successo_per_questo_metodo) / num_periodi_unici_totali) * 100

                migliori_metodi_periodici.append({
                    "metodo_formula": {"ruota_calcolo": ruota_calcolo_base,
                                       "pos_estratto_calcolo": pos_estratto_base_idx + 1,
                                       "operazione": op_str,
                                       "operando_fisso": operando_fisso},
                    "ambata_riferimento": ambata_costante_del_metodo,
                    "successi_applicazioni": successi_applicazioni_totali,
                    "tentativi_applicazioni": tentativi_applicazioni_totali,
                    "frequenza_applicazioni": frequenza_applicazioni, # Performance sulle singole applicazioni
                    "periodi_con_successo": len(periodi_con_successo_per_questo_metodo),
                    "periodi_totali_analizzati": num_periodi_unici_totali,
                    "copertura_periodi_perc": copertura_periodi, # Nuova metrica
                    "applicazioni_vincenti_dettagliate": applicazioni_vincenti_dettaglio
                })

    # Ordina per copertura periodi, poi per frequenza applicazioni, poi per successi
    migliori_metodi_periodici.sort(key=lambda x: (x["copertura_periodi_perc"], x["frequenza_applicazioni"], x["successi_applicazioni"]), reverse=True)

    if app_logger: app_logger(f"Ricerca ambata ottimale periodica completata. Trovati {len(migliori_metodi_periodici)} metodi validi.")

    risultati_con_previsione = []
    if migliori_metodi_periodici and storico_completo:
        if storico_filtrato_periodo: # Per calcolare la previsione live
            ultima_estrazione_valida_nel_periodo = storico_filtrato_periodo[-1]
            for metodo_info in migliori_metodi_periodici[:1]: # Solo per il migliore per ora
                previsione_live = "N/A"
                form = metodo_info["metodo_formula"]
                numeri_base_live = ultima_estrazione_valida_nel_periodo.get(form["ruota_calcolo"], [])
                if numeri_base_live and len(numeri_base_live) >= form["pos_estratto_calcolo"]:
                    num_base_live_val = numeri_base_live[form["pos_estratto_calcolo"]-1]
                    op_func_live = OPERAZIONI[form["operazione"]]
                    try:
                        val_op_live = op_func_live(num_base_live_val, form["operando_fisso"])
                        previsione_live = regola_fuori_90(val_op_live)
                    except ZeroDivisionError: pass
                metodo_info["previsione_live_periodica"] = previsione_live
                risultati_con_previsione.append(metodo_info)
            return risultati_con_previsione # Restituisce solo i metodi con previsione calcolata

    return migliori_metodi_periodici[:1] # Fallback se non si può calcolare previsione live

def analizza_performance_dettagliata_metodo(
    storico_completo,
    definizione_metodo,
    metodo_stringa_per_log,
    ruote_gioco,
    lookahead,
    data_inizio_analisi,
    data_fine_analisi,
    mesi_selezionati_filtro, # NOME CORRETTO
    app_logger=None,
    condizione_primaria_metodo=None,
    indice_estrazione_mese_da_considerare=None
):
    def log(msg):
        if app_logger: app_logger(msg)

    log(f"\n--- AVVIO ANALISI PERFORMANCE DETTAGLIATA METODO ---") # Modificato il titolo per chiarezza
    log(f"Metodo da analizzare: {metodo_stringa_per_log}")
    log(f"Periodo Globale: {data_inizio_analisi.strftime('%d/%m/%Y')} - {data_fine_analisi.strftime('%d/%m/%Y')}")
    log(f"Mesi Specifici (se lista non vuota): {mesi_selezionati_filtro or 'Tutti nel range'}")
    log(f"Indice Estrazione del Mese da Usare: {indice_estrazione_mese_da_considerare if indice_estrazione_mese_da_considerare is not None else 'Non specificato (o tutte valide nel mese)'}")
    if condizione_primaria_metodo:
        log(f"Applicando Condizione Primaria: {condizione_primaria_metodo}")
    log(f"Ruote Gioco Verifica Esito: {', '.join(ruote_gioco)}, Lookahead: {lookahead}")

    risultati_dettagliati = []
    if not storico_completo:
        log("ERRORE: Storico completo vuoto.")
        return risultati_dettagliati

    estrazioni_per_anno_mese = defaultdict(list)
    for i_idx, estrazione in enumerate(storico_completo):
        data_e = estrazione.get('data')
        if not isinstance(data_e, date): continue
        if data_inizio_analisi <= data_e <= data_fine_analisi:
            if not mesi_selezionati_filtro or data_e.month in mesi_selezionati_filtro:
                estrazione_con_idx = estrazione.copy()
                estrazione_con_idx['indice_originale_storico_completo'] = i_idx
                estrazioni_per_anno_mese[(data_e.year, data_e.month)].append(estrazione_con_idx)

    anni_mesi_ordinati = sorted(estrazioni_per_anno_mese.keys())

    if not anni_mesi_ordinati:
        log("Nessuna estrazione trovata nel periodo e mesi specificati per l'analisi.")
        return risultati_dettagliati

    for anno, mese in anni_mesi_ordinati:
        estrazioni_del_mese_corrente = estrazioni_per_anno_mese[(anno, mese)]
        estrazione_di_applicazione_trovata_nel_mese = None

        for idx_estrazione_nel_mese, estrazione_candidata in enumerate(estrazioni_del_mese_corrente):
            # Usa 'indice_mese' fornito da carica_storico_completo se esiste, altrimenti calcola
            indice_estrazione_nel_mese_1_based = estrazione_candidata.get('indice_mese', idx_estrazione_nel_mese + 1)

            if indice_estrazione_mese_da_considerare is not None:
                if indice_estrazione_nel_mese_1_based != indice_estrazione_mese_da_considerare:
                    continue

            condizione_soddisfatta_per_candidata = False
            if condizione_primaria_metodo:
                cond_ruota = condizione_primaria_metodo['ruota']
                cond_pos_idx = (condizione_primaria_metodo.get('posizione', 1) - 1)
                cond_min = condizione_primaria_metodo['val_min']
                cond_max = condizione_primaria_metodo['val_max']
                numeri_ruota_cond = estrazione_candidata.get(cond_ruota, [])
                if numeri_ruota_cond and len(numeri_ruota_cond) > cond_pos_idx:
                    val_est_cond = numeri_ruota_cond[cond_pos_idx]
                    if cond_min <= val_est_cond <= cond_max:
                        condizione_soddisfatta_per_candidata = True
            else:
                condizione_soddisfatta_per_candidata = True

            if condizione_soddisfatta_per_candidata:
                estrazione_di_applicazione_trovata_nel_mese = estrazione_candidata
                break

        dettaglio_applicazione = {
            "data_applicazione": date(anno, mese, 1) if not estrazione_di_applicazione_trovata_nel_mese else estrazione_di_applicazione_trovata_nel_mese['data'],
            "ambata_prevista": None, "metodo_applicabile": False, "esito_ambata": False,
            "colpo_vincita_ambata": None, "ruota_vincita_ambata": None,
            "numeri_estratti_vincita": None, "condizione_soddisfatta": False
        }

        if not estrazione_di_applicazione_trovata_nel_mese:
            log(f"INFO: Nessuna estrazione di applicazione per {mese:02d}/{anno} (IndiceMese: {indice_estrazione_mese_da_considerare}, CondOK?: No)")
        else:
            dettaglio_applicazione["condizione_soddisfatta"] = True
            data_app = estrazione_di_applicazione_trovata_nel_mese['data']
            dettaglio_applicazione["data_applicazione"] = data_app
            indice_originale_app = estrazione_di_applicazione_trovata_nel_mese['indice_originale_storico_completo']

            ambata_prevista_calc = calcola_valore_metodo_complesso(estrazione_di_applicazione_trovata_nel_mese, definizione_metodo, app_logger)

            if ambata_prevista_calc is not None:
                ambata_prevista = regola_fuori_90(ambata_prevista_calc)
                if ambata_prevista is not None:
                    dettaglio_applicazione["metodo_applicabile"] = True
                    dettaglio_applicazione["ambata_prevista"] = ambata_prevista
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

    log(f"Analisi dettagliata periodica specifica completata. {len(risultati_dettagliati)} periodi (anno/mese) processati.")
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
        tipo_analisi_scelta = self.ap_tipo_sorte_var.get()
        self._log_to_gui("\n" + "="*50 + f"\nAVVIO ANALISI PRESENZE PERIODICHE ({tipo_analisi_scelta.upper()})\n" + "="*50)

        if self.ap_risultati_listbox:
            self.ap_risultati_listbox.delete(0, tk.END)

        storico_base_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_base_per_analisi:
            self._log_to_gui("Nessun dato storico base caricato. Analisi periodica interrotta.")
            return

        mesi_selezionati_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
        if not mesi_selezionati_gui and not self.ap_tutti_mesi_var.get():
            messagebox.showwarning("Selezione Mesi", "Seleziona almeno un mese o 'Tutti i Mesi'.")
            return

        mesi_map = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
        mesi_numeri_selezionati = []
        if not self.ap_tutti_mesi_var.get():
            mesi_numeri_selezionati = [mesi_map[nome] for nome in mesi_selezionati_gui]

        ruote_gioco_sel, _, _ = self._get_parametri_gioco_comuni()
        if ruote_gioco_sel is None: return

        self._log_to_gui(f"Parametri Analisi Presenze Periodiche ({tipo_analisi_scelta}):")
        log_mesi_str = 'Tutti (nel range date globale)' if not mesi_numeri_selezionati else ', '.join(mesi_selezionati_gui)
        self._log_to_gui(f"  Mesi: {log_mesi_str}")
        data_inizio_g_str = self.date_inizio_entry_analisi.get() or "Inizio Storico"
        data_fine_g_str = self.date_fine_entry_analisi.get() or "Fine Storico"
        self._log_to_gui(f"  Range Date Globale Applicato: Da {data_inizio_g_str} a {data_fine_g_str}")
        self._log_to_gui(f"  Ruote di Gioco: {', '.join(ruote_gioco_sel)}")

        numeri_input_utente_str = self.ap_numeri_input_var.get().strip()
        numeri_da_verificare_utente = []
        if tipo_analisi_scelta != "Ambata" and numeri_input_utente_str:
            try:
                numeri_da_verificare_utente = sorted([int(n.strip()) for n in numeri_input_utente_str.split(',') if n.strip()])
                num_numeri_attesi = {"Ambo": 2, "Terno": 3, "Quaterna": 4, "Cinquina": 5}.get(tipo_analisi_scelta)
                if len(numeri_da_verificare_utente) != num_numeri_attesi:
                    messagebox.showerror("Errore Input Numeri", f"Per la sorte '{tipo_analisi_scelta}' sono necessari {num_numeri_attesi} numeri. Ne hai forniti {len(numeri_da_verificare_utente)}.")
                    return
                if not all(1 <= n <= 90 for n in numeri_da_verificare_utente):
                    messagebox.showerror("Errore Input Numeri", "I numeri devono essere tra 1 e 90."); return
                self._log_to_gui(f"  Verifica combinazione specifica: {numeri_da_verificare_utente}")
            except ValueError:
                messagebox.showerror("Errore Input Numeri", "Formato numeri non valido. Usa numeri separati da virgola (es. 10,20,30)."); return

        self.master.config(cursor="watch"); self.master.update_idletasks()

        data_inizio_g_filtro = None
        data_fine_g_filtro = None
        try:
            data_inizio_g_filtro = self.date_inizio_entry_analisi.get_date()
        except ValueError:
            pass
        try:
            data_fine_g_filtro = self.date_fine_entry_analisi.get_date()
        except ValueError:
            pass

        storico_filtrato_finale = filtra_storico_per_periodo(
            storico_base_per_analisi,
            mesi_numeri_selezionati,
            data_inizio_g_filtro,
            data_fine_g_filtro,
            app_logger=self._log_to_gui
        )

        if not storico_filtrato_finale:
            msg = "Nessuna estrazione trovata per i mesi selezionati nel range di date specificato."
            self._log_to_gui(msg)
            if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, msg)
            messagebox.showinfo("Analisi Periodica", msg)
            self.master.config(cursor="")
            return

        lista_previsioni_per_popup = []
        metodi_grezzi_per_salvataggio_popup = []
        data_riferimento_popup_periodica = storico_filtrato_finale[-1]['data'].strftime('%d/%m/%Y') if storico_filtrato_finale else "N/D"
        ruote_gioco_str_popup_periodica = ", ".join(ruote_gioco_sel)

        if tipo_analisi_scelta == "Ambata":
            conteggio_ambate_grezzo, num_estraz_analizzate_periodo_totali = analizza_frequenza_ambate_periodica(
                storico_filtrato_finale, ruote_gioco_sel, app_logger=self._log_to_gui
            )
            self._log_to_gui("\n--- RISULTATI PRESENZE AMBATE PERIODICA (FREQUENZA GREZZA) ---")
            if self.ap_risultati_listbox:
                self.ap_risultati_listbox.insert(tk.END, f"Presenze Ambate (su {num_estraz_analizzate_periodo_totali} estraz. nel periodo):")

            if not conteggio_ambate_grezzo:
                msg = "Nessuna ambata trovata nel periodo selezionato."
                self._log_to_gui(msg)
                if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, msg)
            else:
                risultati_display = []
                for numero, conteggio in conteggio_ambate_grezzo.most_common():
                    freq_presenza = (conteggio / num_estraz_analizzate_periodo_totali * 100) if num_estraz_analizzate_periodo_totali > 0 else 0
                    risultati_display.append({"numero": numero, "conteggio": conteggio, "freq_presenza": freq_presenza})
                risultati_display.sort(key=lambda x: (x["freq_presenza"], x["conteggio"]), reverse=True)

                for res in risultati_display[:10]:
                    riga = f"  Num: {res['numero']:>2} - Pres: {res['conteggio']:<3} ({res['freq_presenza']:.1f}%)"
                    if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, riga)
                    self._log_to_gui(riga)

                if risultati_display:
                    ambata_principale_info = risultati_display[0]
                    ambata_principale = ambata_principale_info["numero"]
                    conteggio_grezzo_ambata_principale = ambata_principale_info["conteggio"]
                    freq_presenza_grezza_amb_princ = ambata_principale_info["freq_presenza"]

                    anni_con_uscita_ambata_principale = set()
                    anni_analizzati_distinti_per_copertura = set()

                    for estrazione_periodo in storico_filtrato_finale:
                        anni_analizzati_distinti_per_copertura.add(estrazione_periodo['data'].year)
                        for ruota_g in ruote_gioco_sel:
                            if ambata_principale in estrazione_periodo.get(ruota_g, []):
                                anni_con_uscita_ambata_principale.add(estrazione_periodo['data'].year)
                                break

                    num_anni_con_uscita = len(anni_con_uscita_ambata_principale)
                    num_anni_totali_analizzati_cop = len(anni_analizzati_distinti_per_copertura)
                    copertura_periodica_perc = 0.0
                    if num_anni_totali_analizzati_cop > 0:
                        copertura_periodica_perc = (num_anni_con_uscita / num_anni_totali_analizzati_cop) * 100

                    self._log_to_gui(f"\n--- DETTAGLI COPERTURA PER AMBATA PIÙ PRESENTE: {ambata_principale} ---")
                    self._log_to_gui(f"  Uscita in {num_anni_con_uscita} anni distinti su {num_anni_totali_analizzati_cop} anni analizzati nel periodo.")
                    self._log_to_gui(f"  Copertura Periodica (Anni): {copertura_periodica_perc:.1f}%")
                    if self.ap_risultati_listbox:
                        self.ap_risultati_listbox.insert(tk.END, "")
                        self.ap_risultati_listbox.insert(tk.END, f"Dettaglio Copertura Ambata {ambata_principale}:")
                        self.ap_risultati_listbox.insert(tk.END, f"  Copertura Periodica (Anni): {copertura_periodica_perc:.1f}% ({num_anni_con_uscita}/{num_anni_totali_analizzati_cop})")

                    abbinamenti_ambo = analizza_abbinamenti_per_numero_specifico(storico_filtrato_finale, ambata_principale, ruote_gioco_sel, self._log_to_gui)
                    contorni_frequenti_lista = trova_contorni_frequenti_per_ambata_periodica(
                        storico_filtrato_finale, ambata_principale, 10, ruote_gioco_sel, self._log_to_gui
                    )

                    performance_str_popup = (
                        f"Presenza grezza: {freq_presenza_grezza_amb_princ:.1f}% ({conteggio_grezzo_ambata_principale}/{num_estraz_analizzate_periodo_totali} estraz.)\n"
                        f"Copertura Periodica (Anni): {copertura_periodica_perc:.1f}% ({num_anni_con_uscita}/{num_anni_totali_analizzati_cop} anni)"
                    )

                    dettaglio_popup_ambata = {
                        "titolo_sezione": f"--- AMBATA PIÙ PRESENTE: {ambata_principale} ---",
                        "info_metodo_str": f"Analisi frequenza nel periodo: Mesi={log_mesi_str}, Range Date: {data_inizio_g_str} a {data_fine_g_str}",
                        "ambata_prevista": ambata_principale,
                        "performance_storica_str": performance_str_popup,
                        "abbinamenti_dict": abbinamenti_ambo,
                        "contorni_suggeriti": contorni_frequenti_lista
                    }
                    lista_previsioni_per_popup.append(dettaglio_popup_ambata)

                    dati_salvataggio_ambata = {
                        "tipo_metodo_salvato": "periodica_ambata_frequente",
                        "formula_testuale": f"Ambata più frequente ({ambata_principale}) nel periodo ({log_mesi_str}, Range: {data_inizio_g_str}-{data_fine_g_str})",
                        "ambata_prevista": ambata_principale,
                        "successi_grezzi": conteggio_grezzo_ambata_principale,
                        "tentativi_grezzi": num_estraz_analizzate_periodo_totali,
                        "frequenza_grezza": freq_presenza_grezza_amb_princ / 100.0 if freq_presenza_grezza_amb_princ is not None else 0.0,
                        "anni_con_uscita": num_anni_con_uscita,
                        "anni_totali_analizzati_copertura": num_anni_totali_analizzati_cop,
                        "copertura_periodica_anni_perc": copertura_periodica_perc,
                        "abbinamenti": abbinamenti_ambo,
                        "contorni_suggeriti_extra": contorni_frequenti_lista,
                        "parametri_periodo": {"mesi": mesi_selezionati_gui or "Tutti", "range_date": f"{data_inizio_g_str} a {data_fine_g_str}"}
                    }
                    metodi_grezzi_per_salvataggio_popup.append(dati_salvataggio_ambata)


        elif numeri_da_verificare_utente:
            successi_comb, estraz_analizzate_comb = analizza_frequenza_combinazione_periodica(
                storico_filtrato_finale, numeri_da_verificare_utente, ruote_gioco_sel, app_logger=self._log_to_gui
            )
            self._log_to_gui(f"\n--- RISULTATI PRESENZE {tipo_analisi_scelta.upper()} PERIODICA (SPECIFICA) ---")
            self._log_to_gui(f"  Combinazione: {numeri_da_verificare_utente}")
            ris_text_log = ""
            ris_text_listbox = ""
            if estraz_analizzate_comb > 0:
                freq_percentuale_comb = (successi_comb / estraz_analizzate_comb) * 100
                ris_text_log = f"  Trovata {successi_comb} volte su {estraz_analizzate_comb} estraz. valide (Presenza: {freq_percentuale_comb:.1f}%)."
                ris_text_listbox = f"Trovata {successi_comb} su {estraz_analizzate_comb} estraz. (Pres: {freq_percentuale_comb:.1f}%)"
            else:
                ris_text_log = f"  Trovata {successi_comb} volte (0 estrazioni analizzabili nel periodo)."
                ris_text_listbox = f"Trovata {successi_comb} volte (0 estraz. analizzabili)"
            self._log_to_gui(ris_text_log)
            if self.ap_risultati_listbox:
                self.ap_risultati_listbox.insert(tk.END, f"Presenze per {tipo_analisi_scelta} {numeri_da_verificare_utente}:")
                self.ap_risultati_listbox.insert(tk.END, ris_text_listbox)

            if estraz_analizzate_comb > 0 or successi_comb > 0:
                dettaglio_popup_comb_spec = {
                    "titolo_sezione": f"--- FREQUENZA {tipo_analisi_scelta.upper()} {numeri_da_verificare_utente} ---",
                    "info_metodo_str": f"Analisi nel periodo: Mesi={log_mesi_str}, Range Date: {data_inizio_g_str} a {data_fine_g_str}",
                    "ambata_prevista": ", ".join(map(str, numeri_da_verificare_utente)),
                    "performance_storica_str": ris_text_log.strip(),
                    "abbinamenti_dict": {}
                }
                lista_previsioni_per_popup.append(dettaglio_popup_comb_spec)
                metodi_grezzi_per_salvataggio_popup.append({
                    "tipo_metodo_salvato": f"periodica_combinazione_specifica_{tipo_analisi_scelta.lower()}",
                    "formula_testuale": f"{tipo_analisi_scelta} {numeri_da_verificare_utente} nel periodo ({log_mesi_str}, Range: {data_inizio_g_str}-{data_fine_g_str})",
                    "numeri_cercati": numeri_da_verificare_utente,
                    "successi_cond": successi_comb, "tentativi_cond": estraz_analizzate_comb,
                    "frequenza_cond": (successi_comb / estraz_analizzate_comb) if estraz_analizzate_comb > 0 else 0.0,
                    "parametri_periodo": {"mesi": mesi_selezionati_gui or "Tutti", "range_date": f"{data_inizio_g_str} a {data_fine_g_str}"}
                })


        else: # Ricerca automatica delle combinazioni più frequenti
            dimensione_sorte = {"Ambo": 2, "Terno": 3, "Quaterna": 4, "Cinquina": 5}.get(tipo_analisi_scelta)
            if dimensione_sorte:
                conteggio_items_comb, num_estraz_analizzate_periodo_comb_auto = trova_combinazioni_frequenti_periodica(
                    storico_filtrato_finale, dimensione_sorte, ruote_gioco_sel, app_logger=self._log_to_gui
                )
                self._log_to_gui(f"\n--- TOP {tipo_analisi_scelta.upper()} PIÙ PRESENTI NEL PERIODO ---")
                if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, f"Top {tipo_analisi_scelta} (su {num_estraz_analizzate_periodo_comb_auto} estraz. nel periodo):")
                if not conteggio_items_comb:
                    msg = f"Nessun {tipo_analisi_scelta.lower()} trovato."
                    self._log_to_gui(msg);
                    if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, msg)
                else:
                    risultati_display_comb_auto = []
                    for combo, conteggio in conteggio_items_comb.most_common():
                        freq_presenza = (conteggio / num_estraz_analizzate_periodo_comb_auto * 100) if num_estraz_analizzate_periodo_comb_auto > 0 else 0
                        risultati_display_comb_auto.append({"combo": combo, "conteggio": conteggio, "freq_presenza": freq_presenza})
                    risultati_display_comb_auto.sort(key=lambda x: (x["freq_presenza"], x["conteggio"]), reverse=True)

                    for res_idx, res in enumerate(risultati_display_comb_auto[:10]):
                        combo_str = ", ".join(map(str, res['combo']))
                        riga = f"  {tipo_analisi_scelta}: [{combo_str}] - Cnt: {res['conteggio']:<3} (Pres: {res['freq_presenza']:.1f}%)"
                        self._log_to_gui(riga)
                        if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, riga)

                    if risultati_display_comb_auto:
                        top_combo_info = risultati_display_comb_auto[0]
                        combo_principale = top_combo_info["combo"]
                        conteggio_principale = top_combo_info["conteggio"]
                        freq_presenza_principale = top_combo_info["freq_presenza"]

                        dettaglio_popup_top_comb = {
                            "titolo_sezione": f"--- TOP {tipo_analisi_scelta.upper()} PIÙ PRESENTE: [{', '.join(map(str, combo_principale))}] ---",
                            "info_metodo_str": f"Analisi nel periodo: Mesi={log_mesi_str}, Range Date: {data_inizio_g_str} a {data_fine_g_str}",
                            "ambata_prevista": ", ".join(map(str, combo_principale)),
                            "performance_storica_str": f"Conteggio: {conteggio_principale} (Presenza: {freq_presenza_principale:.1f}% su {num_estraz_analizzate_periodo_comb_auto} estraz.)",
                            "abbinamenti_dict": {}
                        }
                        lista_previsioni_per_popup.append(dettaglio_popup_top_comb)
                        metodi_grezzi_per_salvataggio_popup.append({
                            "tipo_metodo_salvato": f"periodica_top_{tipo_analisi_scelta.lower()}",
                            "formula_testuale": f"Top {tipo_analisi_scelta} [{', '.join(map(str, combo_principale))}] nel periodo ({log_mesi_str}, Range: {data_inizio_g_str}-{data_fine_g_str})",
                            "combinazione_trovata": list(combo_principale),
                            "successi_cond": conteggio_principale,
                            "tentativi_cond": num_estraz_analizzate_periodo_comb_auto,
                            "frequenza_cond": freq_presenza_principale / 100.0 if freq_presenza_principale is not None else 0.0,
                            "parametri_periodo": {"mesi": mesi_selezionati_gui or "Tutti", "range_date": f"{data_inizio_g_str} a {data_fine_g_str}"}
                        })
            else:
                self._log_to_gui(f"Tipo di analisi {tipo_analisi_scelta} non ancora implementato per ricerca automatica se non è Ambata.")

        self.master.config(cursor="")

        if lista_previsioni_per_popup:
            self.mostra_popup_previsione(
                titolo_popup=f"Risultati Analisi Periodica ({tipo_analisi_scelta})",
                ruote_gioco_str=ruote_gioco_str_popup_periodica,
                lista_previsioni_dettagliate=lista_previsioni_per_popup,
                data_riferimento_previsione_str_comune=data_riferimento_popup_periodica,
                metodi_grezzi_per_salvataggio=metodi_grezzi_per_salvataggio_popup
            )
        elif not (self.ap_risultati_listbox and self.ap_risultati_listbox.size() > 0 and self.ap_risultati_listbox.get(0).startswith("Nessun")):
            messagebox.showinfo("Analisi Periodica", "Nessun risultato specifico da mostrare nel popup per i criteri selezionati (controllare la lista per i dettagli).")

        self._log_to_gui(f"--- Analisi Presenze Periodiche ({tipo_analisi_scelta}) Completata ---")

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
        
        data_inizio_g_filtro = None; data_fine_g_filtro = None
        try: data_inizio_g_filtro = self.date_inizio_entry_analisi.get_date()
        except ValueError: pass
        try: data_fine_g_filtro = self.date_fine_entry_analisi.get_date()
        except ValueError: pass
        
        ruote_gioco_verifica, lookahead_verifica, indice_mese_applicazione = self._get_parametri_gioco_comuni()
        if ruote_gioco_verifica is None: 
            self.master.config(cursor="")
            return

        self._log_to_gui(f"Parametri Ricerca Ambata e Ambo Unico (con Trasformazioni):")
        self._log_to_gui(f"  Estratto Base Globale: {ruota_base}[pos.{pos_base_0idx+1}]")
        self._log_to_gui(f"  Operazioni Base Selezionate: {operazioni_base_selezionate}")
        self._log_to_gui(f"  Trasformazioni Correttore Selezionate: {trasformazioni_correttore_selezionate}")
        self._log_to_gui(f"  Ruote Gioco (Verifica Esito Ambo): {', '.join(ruote_gioco_verifica)}")
        self._log_to_gui(f"  Colpi Lookahead per Ambo: {lookahead_verifica}")
        self._log_to_gui(f"  Indice Mese Applicazione Metodo (per backtest): {indice_mese_applicazione if indice_mese_applicazione is not None else 'Tutte le estrazioni valide nel periodo'}")
        data_inizio_log = data_inizio_g_filtro.strftime('%Y-%m-%d') if data_inizio_g_filtro else "Inizio Storico"
        data_fine_log = data_fine_g_filtro.strftime('%Y-%m-%d') if data_fine_g_filtro else "Fine Storico"
        self._log_to_gui(f"  Periodo Globale Analisi: {data_inizio_log} - {data_fine_log}")
        
        storico_per_analisi = carica_storico_completo(self.cartella_dati_var.get(), 
                                                       data_inizio_g_filtro, 
                                                       data_fine_g_filtro, 
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
                indice_mese_specifico_applicazione=indice_mese_applicazione,
                lookahead_colpi_per_verifica=lookahead_verifica,
                app_logger=self._log_to_gui,
                min_tentativi_per_metodo=self.min_tentativi_var.get() 
            )
            
            self.aau_metodi_trovati_dati = migliori_ambi_config if migliori_ambi_config else []

            if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
                self.aau_risultati_listbox.delete(0, tk.END)
                if not migliori_ambi_config:
                    self.aau_risultati_listbox.insert(tk.END, "Nessuna configurazione di ambo performante trovata.")
                    self._log_to_gui("Nessuna configurazione di ambo performante trovata.")
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
                    if indice_mese_applicazione is not None:
                        for estr_rev in reversed(storico_per_analisi):
                            if estr_rev.get('indice_mese') == indice_mese_applicazione:
                                ultima_estrazione_valida_per_previsione = estr_rev
                                break
                        if ultima_estrazione_valida_per_previsione:
                            self._log_to_gui(f"INFO: Previsione live AAU sarà basata sull'ultima {indice_mese_applicazione}a estr. del mese valida: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                        else:
                            self._log_to_gui(f"WARN: Non trovata un'estrazione che sia la {indice_mese_applicazione}a del mese per la previsione live AAU. Nessuna previsione live possibile con questo filtro.")
                            messagebox.showwarning("Previsione Live AAU", f"Nessuna estrazione trovata che sia la {indice_mese_applicazione}a del mese per calcolare la previsione live.\nControlla i filtri o lo storico.\n(Verranno mostrate solo le performance storiche nel popup).")
                            self.master.config(cursor=""); return 
                    else:
                        ultima_estrazione_valida_per_previsione = storico_per_analisi[-1]
                        self._log_to_gui(f"INFO: Previsione live AAU sarà basata sull'ultima estrazione disponibile: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                else:
                    messagebox.showerror("Errore Dati", "Storico per analisi vuoto, impossibile generare previsione live.")
                    self.master.config(cursor=""); return

                base_extr_dati_live = ultima_estrazione_valida_per_previsione.get(ruota_base, [])
                if not base_extr_dati_live or len(base_extr_dati_live) <= pos_base_0idx:
                    messagebox.showerror("Errore Previsione", f"Dati mancanti per {ruota_base}[pos.{pos_base_0idx+1}] nell'ultima estrazione ({ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}) valida per la previsione AAU.")
                    self.master.config(cursor=""); return
                numero_base_live_per_previsione = base_extr_dati_live[pos_base_0idx]
                self._log_to_gui(f"INFO: Estratto base per previsione live AAU: {numero_base_live_per_previsione} (da {ruota_base} il {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')})")

                top_metodi_per_ambi_unici_live = []
                ambi_live_gia_selezionati_per_popup = set()
                max_risultati_unici_desiderati = self.num_ambate_var.get()

                for res_conf_storico in migliori_ambi_config:
                    if len(top_metodi_per_ambi_unici_live) >= max_risultati_unici_desiderati: break
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
                            top_metodi_per_ambi_unici_live.append(metodo_per_popup)
                            ambi_live_gia_selezionati_per_popup.add(ambo_live_normalizzato)
                
                self._log_to_gui(f"INFO: Dopo ricalcolo LIVE e filtro, trovati {len(top_metodi_per_ambi_unici_live)} metodi con ambi UNICI DIVERSI per il popup AAU.")

                if not top_metodi_per_ambi_unici_live:
                    self._log_to_gui("Nessun metodo con ambi unici trovato dopo ricalcolo live e filtro per il popup.")
                    messagebox.showinfo("Ricerca Ambata/Ambo", "Nessuna configurazione valida trovata o solo ambi duplicati/non validi per la previsione live.")
                else:
                    primo_metodo_live_log = top_metodi_per_ambi_unici_live[0]
                    a1_sugg_live_log = primo_metodo_live_log.get('ambata1_live_calcolata'); a2_sugg_live_log = primo_metodo_live_log.get('ambata2_live_calcolata')
                    if a1_sugg_live_log is not None and a2_sugg_live_log is not None:
                        self._log_to_gui(f"  Ambate Singole Consigliate (LIVE): {a1_sugg_live_log}, {a2_sugg_live_log}")
                    self._log_to_gui(f"  Ambi Secchi Consigliati (LIVE):")
                    for i_log, met_log_live in enumerate(top_metodi_per_ambi_unici_live[:3]):
                        ambo_log_live = met_log_live.get('ambo_live_calcolato')
                        self._log_to_gui(f"    {i_log+1}°) Ambo LIVE: {ambo_log_live} (Performance storica ambo: {met_log_live.get('frequenza_ambo',0):.2%})")

                    lista_previsioni_popup_aau = []; dati_grezzi_popup_aau = []
                    for idx_popup, res_popup_dati in enumerate(top_metodi_per_ambi_unici_live):
                        ambata1_live_display = res_popup_dati.get('ambata1_live_calcolata', "N/A"); ambata2_live_display = res_popup_dati.get('ambata2_live_calcolata', "N/A")
                        ambo_live_calcolato_tuple = res_popup_dati.get('ambo_live_calcolato'); ambo_live_str_display = f"({ambo_live_calcolato_tuple[0]}, {ambo_live_calcolato_tuple[1]})" if ambo_live_calcolato_tuple else "N/D (calcolo live fallito)"
                        
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
                            if len(top_metodi_per_ambi_unici_live) > 1:
                                ambo2_sugg_live = top_metodi_per_ambi_unici_live[1].get('ambo_live_calcolato'); ambo2_sugg_str = f"({ambo2_sugg_live[0]}, {ambo2_sugg_live[1]})" if ambo2_sugg_live else "N/D"
                                suggerimento_gioco_popup_str += f"\n  - Giocare 2° Ambo Secco (LIVE): {ambo2_sugg_str}"
                            if len(top_metodi_per_ambi_unici_live) > 2:
                                ambo3_sugg_live = top_metodi_per_ambi_unici_live[2].get('ambo_live_calcolato'); ambo3_sugg_str = f"({ambo3_sugg_live[0]}, {ambo3_sugg_live[1]})" if ambo3_sugg_live else "N/D"
                                suggerimento_gioco_popup_str += f"\n  - Giocare 3° Ambo Secco (LIVE): {ambo3_sugg_str}"
                            performance_completa_str_popup += suggerimento_gioco_popup_str

                        dettaglio_popup = {"titolo_sezione": f"--- {(idx_popup+1)}ª Configurazione Proposta (Ambo Unico) ---", "info_metodo_str": formula_origine_popup_display, "ambata_prevista": f"PREVISIONE DA GIOCARE: AMBO DA GIOCARE: {ambo_live_str_display}", "performance_storica_str": performance_completa_str_popup, "abbinamenti_dict": {}, "contorni_suggeriti": [] }
                        lista_previsioni_popup_aau.append(dettaglio_popup)
                        
                        dati_salvataggio = res_popup_dati.copy(); dati_salvataggio["tipo_metodo_salvato"] = "ambata_ambo_unico_trasf"; dati_salvataggio["formula_testuale"] = formula_origine_popup_display; dati_salvataggio["ruota_base_origine"] = ruota_base; dati_salvataggio["pos_base_origine"] = pos_base_0idx
                        if ambo_live_calcolato_tuple: dati_salvataggio["definizione_strutturata"] = list(ambo_live_calcolato_tuple)
                        else: dati_salvataggio["definizione_strutturata"] = None
                        dati_salvataggio["ambata_prevista_live"] = ambo_live_str_display
                        dati_grezzi_popup_aau.append(dati_salvataggio)
                    
                    if lista_previsioni_popup_aau:
                        self.mostra_popup_previsione(
                            titolo_popup="Migliori Configurazioni Ambata e Ambo Unico", ruote_gioco_str=", ".join(ruote_gioco_verifica),
                            lista_previsioni_dettagliate=lista_previsioni_popup_aau, data_riferimento_previsione_str_comune=ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y'),
                            metodi_grezzi_per_salvataggio=dati_grezzi_popup_aau
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

        definizione_per_analisi = None
        formula_testuale_display = "N/D"
        tipo_metodo_usato_display = "Sconosciuto" 
        tipo_metodo_origine = "Sconosciuto"     
        condizione_primaria_da_passare_all_analisi = None

        self._log_to_gui(f"DEBUG Backtest: Inizio. metodo_preparato_per_backtest: {self.metodo_preparato_per_backtest is not None}, usa_ultimo_corretto: {self.usa_ultimo_corretto_per_backtest_var.get()}, ultimo_metodo_corretto_def: {self.ultimo_metodo_corretto_trovato_definizione is not None}")

        # Priorità 1: Metodo esplicitamente preparato da un popup
        if self.metodo_preparato_per_backtest:
            dati_preparati = self.metodo_preparato_per_backtest
            definizione_per_analisi = dati_preparati.get('definizione_strutturata')
            formula_testuale_display = dati_preparati.get('formula_testuale', "Formula preparata mancante")
            tipo_metodo_usato_display = dati_preparati.get('tipo', 'Sconosciuto_da_Popup')
            tipo_metodo_origine = "Preparato da Popup"
            condizione_primaria_da_passare_all_analisi = dati_preparati.get('condizione_primaria')
            self._log_to_gui(f"INFO: Utilizzo del METODO PREPARATO DA POPUP (Tipo Originale: {tipo_metodo_usato_display}): {formula_testuale_display}")
            self._log_to_gui(f"  Definizione preparata: {definizione_per_analisi}")
            self._log_to_gui(f"  Condizione primaria preparata: {condizione_primaria_da_passare_all_analisi}")

        # Priorità 2: Ultimo metodo corretto (se checkbox è spuntato)
        elif self.usa_ultimo_corretto_per_backtest_var.get() and self.ultimo_metodo_corretto_trovato_definizione:
            scelta_base_per_corretto = self.mc_backtest_choice_var.get()
            self._log_to_gui(f"DEBUG Backtest (Checkbox Corretto): Scelta utente per base corretta: {scelta_base_per_corretto}")

            if isinstance(self.ultimo_metodo_corretto_trovato_definizione, dict):
                chiave_definizione_scelta = None
                temp_tipo_metodo = "Sconosciuto"

                if scelta_base_per_corretto == "base1" and 'base1_corretto' in self.ultimo_metodo_corretto_trovato_definizione:
                    chiave_definizione_scelta = 'base1_corretto'
                    temp_tipo_metodo = "Complesso Corretto (Base 1)"
                elif scelta_base_per_corretto == "base2" and 'base2_corretto' in self.ultimo_metodo_corretto_trovato_definizione:
                    chiave_definizione_scelta = 'base2_corretto'
                    temp_tipo_metodo = "Complesso Corretto (Base 2)"
                
                if chiave_definizione_scelta:
                    definizione_per_analisi = self.ultimo_metodo_corretto_trovato_definizione.get(chiave_definizione_scelta)
                    tipo_metodo_usato_display = temp_tipo_metodo
                    if definizione_per_analisi:
                        formula_testuale_display = "".join(self._format_componente_per_display(comp) for comp in definizione_per_analisi)
                else: # Fallback se la scelta non è disponibile
                    if 'base1_corretto' in self.ultimo_metodo_corretto_trovato_definizione:
                        definizione_per_analisi = self.ultimo_metodo_corretto_trovato_definizione['base1_corretto']
                        tipo_metodo_usato_display = "Complesso Corretto (Base 1 - Fallback)"
                        formula_testuale_display = "".join(self._format_componente_per_display(comp) for comp in definizione_per_analisi)
                        self._log_to_gui(f"WARN: Scelta '{scelta_base_per_corretto}' per metodo corretto non trovata, fallback su Base 1 corretto.")
                    elif 'base2_corretto' in self.ultimo_metodo_corretto_trovato_definizione:
                        definizione_per_analisi = self.ultimo_metodo_corretto_trovato_definizione['base2_corretto']
                        tipo_metodo_usato_display = "Complesso Corretto (Base 2 - Fallback)"
                        formula_testuale_display = "".join(self._format_componente_per_display(comp) for comp in definizione_per_analisi)
                        self._log_to_gui(f"WARN: Scelta '{scelta_base_per_corretto}' e Base 1 corretto non trovati, fallback su Base 2 corretto.")
                    else:
                        definizione_per_analisi = None
                
                condizione_primaria_da_passare_all_analisi = None 
                tipo_metodo_origine = f"Checkbox ({tipo_metodo_usato_display})" if definizione_per_analisi else "Checkbox (Nessuna def. valida trovata)"
            
            elif isinstance(self.ultimo_metodo_corretto_trovato_definizione, list): # Gestione legacy o condizionato corretto non come dict
                definizione_per_analisi = list(self.ultimo_metodo_corretto_trovato_definizione)
                tipo_metodo_usato_display = "Complesso Corretto (Legacy/Lista)"
                tipo_metodo_origine = "Checkbox (Legacy/Lista)"
                formula_testuale_display = self.ultimo_metodo_corretto_formula_testuale
                self._log_to_gui(f"WARN: self.ultimo_metodo_corretto_trovato_definizione è una lista. Se era un condizionato corretto, la condizione primaria potrebbe mancare.")
            else:
                 self._log_to_gui(f"WARN: Struttura di self.ultimo_metodo_corretto_trovato_definizione non gestita: {self.ultimo_metodo_corretto_trovato_definizione}")
                 definizione_per_analisi = None

            if definizione_per_analisi:
                self._log_to_gui(f"INFO: Utilizzo dell'ULTIMO METODO CORRETTO ({tipo_metodo_origine} - Tipo Metodo: {tipo_metodo_usato_display}): {formula_testuale_display}")
                self._log_to_gui(f"  Definizione da checkbox: {definizione_per_analisi}")
            else:
                 self._log_to_gui(f"WARN: Non è stato possibile determinare una definizione valida per il metodo corretto selezionato tramite checkbox (Scelta Radiobutton: {scelta_base_per_corretto}).")

        # Priorità 3: Metodo Base definito manualmente
        if definizione_per_analisi is None:
            tipo_metodo_origine = "Manuale (da Radiobutton)"
            scelta_backtest_manuale = self.mc_backtest_choice_var.get()

            if self.usa_ultimo_corretto_per_backtest_var.get() and not self.ultimo_metodo_corretto_trovato_definizione:
                 messagebox.showwarning("Attenzione Backtest",
                                       "L'opzione 'Usa ultimo metodo corretto' è selezionata, ma nessun metodo corretto è memorizzato.\n"
                                       f"Verrà usato il Metodo Base scelto ({scelta_backtest_manuale}), se valido.")
                 self._log_to_gui(f"WARN: ({tipo_metodo_origine}) Richiesto backtest metodo corretto (da checkbox), ma 'self.ultimo_metodo_corretto_trovato_definizione' è vuoto. Tento con Metodo Base scelto: {scelta_backtest_manuale}.")

            metodo_da_usare_temp = None
            
            if scelta_backtest_manuale == "base1":
                tipo_metodo_origine = "Manuale (Base 1)"
                if not self.definizione_metodo_complesso_attuale or \
                   not (self.definizione_metodo_complesso_attuale[-1].get('operazione_successiva') == '='):
                    messagebox.showerror("Errore Metodo", "Metodo Base 1 non definito o non terminato con '='.")
                    self._log_to_gui(f"ERRORE: ({tipo_metodo_origine}) Metodo Base 1 non valido.")
                    return
                metodo_da_usare_temp = self.definizione_metodo_complesso_attuale
                tipo_metodo_usato_display = "Complesso Base Manuale (Metodo 1)"
            elif scelta_backtest_manuale == "base2":
                tipo_metodo_origine = "Manuale (Base 2)"
                if not self.definizione_metodo_complesso_attuale_2 or \
                   not (self.definizione_metodo_complesso_attuale_2[-1].get('operazione_successiva') == '='):
                    messagebox.showerror("Errore Metodo", "Metodo Base 2 non definito o non terminato con '='.")
                    self._log_to_gui(f"ERRORE: ({tipo_metodo_origine}) Metodo Base 2 non valido.")
                    return
                metodo_da_usare_temp = self.definizione_metodo_complesso_attuale_2
                tipo_metodo_usato_display = "Complesso Base Manuale (Metodo 2)"
            else:
                messagebox.showerror("Errore Interno", "Scelta backtest manuale non riconosciuta.")
                self._log_to_gui(f"ERRORE: ({tipo_metodo_origine}) Scelta backtest manuale non valida: {scelta_backtest_manuale}")
                return

            if metodo_da_usare_temp:
                definizione_per_analisi = list(metodo_da_usare_temp)
                formula_testuale_display = "".join(self._format_componente_per_display(comp) for comp in definizione_per_analisi)
                condizione_primaria_da_passare_all_analisi = None
                self._log_to_gui(f"INFO: Utilizzo del Metodo Base scelto (Origine: {tipo_metodo_origine}, Tipo Metodo: {tipo_metodo_usato_display}): {formula_testuale_display}")
                self._log_to_gui(f"  Definizione: {definizione_per_analisi}")
            else:
                messagebox.showerror("Errore Metodo", "Impossibile determinare il metodo base manuale da usare per il backtest.")
                self._log_to_gui(f"ERRORE: ({tipo_metodo_origine}) Nessun metodo base manuale valido identificato per {scelta_backtest_manuale}.")
                return

        # Controllo finale robustezza definizione e determinazione se usare analizza_performance_dettagliata_metodo
        usa_analizza_performance = False
        
        tipi_validi_per_analisi_dettagliata = [
            "Semplice Analizzato",
            "complesso_base_analizzato",
            "complesso_corretto",
            "Complesso Corretto",             # Nome generico per i corretti
            "Complesso Corretto (Base 1)",
            "Complesso Corretto (Base 2)",
            "Complesso Corretto (Base 1 - Fallback)",
            "Complesso Corretto (Base 2 - Fallback)",
            "Complesso Corretto (Legacy/Lista)",
            "Complesso Base Manuale (Metodo 1)",
            "Complesso Base Manuale (Metodo 2)",
            "Condizionato Base",
            "Condizionato Corretto"
        ]
        
        effective_type_to_check_for_logic = tipo_metodo_usato_display
        if tipo_metodo_origine == "Preparato da Popup" and self.metodo_preparato_per_backtest:
            effective_type_to_check_for_logic = self.metodo_preparato_per_backtest.get('tipo', tipo_metodo_usato_display)
        
        if effective_type_to_check_for_logic in tipi_validi_per_analisi_dettagliata:
            if isinstance(definizione_per_analisi, list) and definizione_per_analisi:
                usa_analizza_performance = True
            else:
                messagebox.showerror("Errore Backtest", f"Definizione strutturata del metodo (tipo effettivo: {effective_type_to_check_for_logic}, origine: {tipo_metodo_origine}) è mancante o non valida.")
                self._log_to_gui(f"ERRORE: Definizione per analisi (lista) è vuota/non valida per tipo effettivo '{effective_type_to_check_for_logic}' (origine: {tipo_metodo_origine}). Valore: {definizione_per_analisi}")
                return
        elif effective_type_to_check_for_logic == "ambo_sommativo_auto":
            if not (isinstance(definizione_per_analisi, list) and len(definizione_per_analisi) == 2 and all(isinstance(n, int) for n in definizione_per_analisi)):
                messagebox.showerror("Errore Backtest", f"Definizione per 'Ambo Sommativo Auto' non valida. Ricevuto: {definizione_per_analisi}")
                self._log_to_gui(f"ERRORE: Definizione per 'ambo_sommativo_auto' non valida: {definizione_per_analisi}")
                return
            self._log_to_gui(f"INFO: Riconosciuto tipo 'ambo_sommativo_auto' con numeri: {definizione_per_analisi}. Non usa 'analizza_performance_dettagliata_metodo'.")

        elif effective_type_to_check_for_logic.startswith("periodica_"):
             self._log_to_gui(f"INFO: Il tipo '{effective_type_to_check_for_logic}' non usa `analizza_performance_dettagliata_metodo` in questo flusso.")
        
        else: 
            messagebox.showerror("Errore Backtest", f"Tipo di metodo effettivo '{effective_type_to_check_for_logic}' (origine: {tipo_metodo_origine}) non gestito per il backtest.")
            self._log_to_gui(f"ERRORE: Tipo di metodo effettivo non gestito '{effective_type_to_check_for_logic}' (origine: {tipo_metodo_origine}). Definizione: {definizione_per_analisi}")
            return
        
        # --- Recupero Parametri di Periodo e Gioco ---
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
        if not storico_per_backtest_globale: messagebox.showerror("Errore Dati", "Impossibile caricare storico per backtest."); return


        self.master.config(cursor="watch"); self.master.update_idletasks()
        try:
            risultati_dettagliati = []

            if usa_analizza_performance:
                self._log_to_gui(f"INFO: Chiamata a analizza_performance_dettagliata_metodo con:")
                self._log_to_gui(f"  definizione_metodo: {definizione_per_analisi}")
                self._log_to_gui(f"  metodo_stringa_per_log: {formula_testuale_display}")
                self._log_to_gui(f"  condizione_primaria_metodo: {condizione_primaria_da_passare_all_analisi}")
                self._log_to_gui(f"  indice_estrazione_mese_da_considerare: {indice_mese_da_gui}")

                risultati_dettagliati = analizza_performance_dettagliata_metodo(
                    storico_completo=storico_per_backtest_globale,
                    definizione_metodo=definizione_per_analisi,
                    metodo_stringa_per_log=formula_testuale_display,
                    ruote_gioco=ruote_g_b,
                    lookahead=lookahead_b,
                    data_inizio_analisi=data_inizio_backtest,
                    data_fine_analisi=data_fine_backtest,
                    mesi_selezionati_filtro=mesi_num_sel_b,
                    app_logger=self._log_to_gui,
                    condizione_primaria_metodo=condizione_primaria_da_passare_all_analisi,
                    indice_estrazione_mese_da_considerare=indice_mese_da_gui
                )
            elif effective_type_to_check_for_logic == "ambo_sommativo_auto":
                 messagebox.showwarning("Funzionalità Limitata", "Backtest dettagliato per 'Ambo Sommativo Auto' non ancora implementato per l'analisi di performance. Verranno mostrati i parametri di input.")
                 self._log_to_gui(f"INFO: Backtest per 'ambo_sommativo_auto' con numeri {definizione_per_analisi}. Formula: {formula_testuale_display}. Periodo: {data_inizio_backtest} - {data_fine_backtest}. Mesi: {mesi_num_sel_b or 'Tutti'}. Ruote: {ruote_g_b}. Lookahead: {lookahead_b}. Indice Mese: {indice_mese_da_gui}")
                 popup_content = f"--- INFO BACKTEST (Ambo Sommativo Auto) ---\n"
                 popup_content += f"Metodo: {formula_testuale_display}\n"
                 popup_content += f"Numeri da giocare (Ambo): {definizione_per_analisi}\n"
                 popup_content += f"Periodo Analisi: {data_inizio_backtest.strftime('%d/%m/%Y')} - {data_fine_backtest.strftime('%d/%m/%Y')}\n"
                 popup_content += f"Mesi Selezionati: {mesi_num_sel_b or 'Tutti nel range'}\n"
                 popup_content += f"Ruote di Gioco: {', '.join(ruote_g_b)}\n"
                 popup_content += f"Colpi di Lookahead: {lookahead_b}\n"
                 popup_content += f"Indice Estrazione del Mese: {indice_mese_da_gui if indice_mese_da_gui is not None else 'Tutte valide'}\n\n"
                 popup_content += "NOTA: L'analisi di performance specifica per questa tipologia non è ancora implementata in questo flusso unificato."
                 self.mostra_popup_testo_semplice("Info Backtest - Ambo Sommativo Auto", popup_content)

            else:
                messagebox.showwarning("Funzionalità Limitata", f"Il backtest dettagliato con analisi di performance per il tipo effettivo '{effective_type_to_check_for_logic}' (origine: {tipo_metodo_origine}) non è attualmente supportato.")
                self._log_to_gui(f"WARN: Backtest di performance non eseguito per tipo effettivo '{effective_type_to_check_for_logic}' (origine: {tipo_metodo_origine}).")

            # --- Visualizzazione Risultati ---
            if risultati_dettagliati:
                popup_content = f"--- RISULTATI BACKTEST DETTAGLIATO ---\n"
                popup_content += f"Metodo (Tipo Effettivo: {effective_type_to_check_for_logic}, Origine: {tipo_metodo_origine}): {formula_testuale_display}\n"
                popup_content += f"Periodo: {data_inizio_backtest.strftime('%d/%m/%Y')} - {data_fine_backtest.strftime('%d/%m/%Y')}\n"
                popup_content += f"Mesi Selezionati: {mesi_num_sel_b or 'Tutti nel range'}\n"
                popup_content += f"Ruote di Gioco: {', '.join(ruote_g_b)}, Colpi Lookahead: {lookahead_b}\n"
                popup_content += f"Indice Estrazione del Mese: {indice_mese_da_gui if indice_mese_da_gui is not None else 'Tutte valide'}\n"
                if condizione_primaria_da_passare_all_analisi:
                     popup_content += f"Condizione Primaria Applicata: {condizione_primaria_da_passare_all_analisi}\n"
                popup_content += "--------------------------------------------------\n\n"

                successi_ambata_tot = 0
                applicazioni_valide_tot = 0
                applicazioni_cond_soddisfatte_tot = 0

                for res_bd in risultati_dettagliati:
                    popup_content += f"Data Applicazione: {res_bd['data_applicazione'].strftime('%d/%m/%Y')}\n"
                    
                    cond_soddisfatta_display = res_bd.get('condizione_soddisfatta', True)
                    if condizione_primaria_da_passare_all_analisi:
                        popup_content += f"  Condizione Primaria Soddisfatta: {'Sì' if cond_soddisfatta_display else 'No'}\n"
                        if cond_soddisfatta_display:
                            applicazioni_cond_soddisfatte_tot +=1
                    else:
                        applicazioni_cond_soddisfatte_tot +=1

                    if cond_soddisfatta_display:
                        if res_bd['metodo_applicabile']:
                            applicazioni_valide_tot += 1
                            popup_content += f"  Ambata Prevista: {res_bd['ambata_prevista']}\n"
                            if res_bd['esito_ambata']:
                                successi_ambata_tot +=1
                                popup_content += f"  ESITO: AMBATA VINCENTE!\n"
                                popup_content += f"    Colpo: {res_bd['colpo_vincita_ambata']}, Ruota: {res_bd['ruota_vincita_ambata']}\n"
                                if res_bd.get('numeri_estratti_vincita'):
                                    popup_content += f"    Numeri Estratti: {res_bd['numeri_estratti_vincita']}\n"
                            else:
                                popup_content += f"  ESITO: Ambata non uscita entro {lookahead_b} colpi.\n"
                        else:
                             popup_content += f"  Metodo base non applicabile (es. div/0) nonostante la condizione (se presente) fosse soddisfatta.\n"
                    popup_content += "-------------------------\n"

                freq_str = "N/A"
                if applicazioni_valide_tot > 0:
                    freq_str = f"{(successi_ambata_tot / applicazioni_valide_tot) * 100:.2f}% ({successi_ambata_tot}/{applicazioni_valide_tot} app. valide)"
                
                summary = f"\nRIEPILOGO:\n"
                if condizione_primaria_da_passare_all_analisi:
                    summary += f"Estrazioni con Cond. Primaria Soddisfatta: {applicazioni_cond_soddisfatte_tot}\n"
                summary += f"Applicazioni Metodo Valide (Cond. OK + Met. OK): {applicazioni_valide_tot}\n"
                summary += f"Successi Ambata: {successi_ambata_tot}\n"
                summary += f"Frequenza Successo (su app. valide): {freq_str}\n"
                popup_content += summary
                self.mostra_popup_testo_semplice("Risultati Backtest Dettagliato Metodo", popup_content)
            
            elif not usa_analizza_performance and tipo_metodo_usato_display != "ambo_sommativo_auto":
                messagebox.showinfo("Backtest Metodo", "Nessuna analisi di performance eseguita o nessun risultato applicabile per il tipo di metodo selezionato.")

        except Exception as e:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")
            messagebox.showerror("Errore Backtest", f"Errore durante il backtest: {e}")
            self._log_to_gui(f"ERRORE CRITICO BACKTEST: {e}\n{traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")
            if self.metodo_preparato_per_backtest:
                self._log_to_gui(f"INFO: Resetto self.metodo_preparato_per_backtest (era: {self.metodo_preparato_per_backtest.get('formula_testuale', 'N/A')})")
                self.metodo_preparato_per_backtest = None
                if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
                    current_listbox_text_line1 = self.mc_listbox_componenti_1.get(0) if self.mc_listbox_componenti_1.size() > 0 else ""
                    if "PER BACKTEST" in current_listbox_text_line1:
                        self._refresh_mc_listbox_1()
                        if not self.definizione_metodo_complesso_attuale:
                             self.mc_listbox_componenti_1.insert(tk.END, "Nessun metodo base definito.")

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
            self.master.config(cursor="") # Assicura reset cursore
            return

        trasformazioni_correttore_selezionate = [
            nome_t for nome_t, var_t in self.aau_trasf_vars.items() if var_t.get()
        ]
        if not trasformazioni_correttore_selezionate:
            messagebox.showwarning("Input Mancante", "Seleziona almeno una trasformazione da applicare ai correttori (anche solo 'Fisso').")
            self.master.config(cursor="") # Assicura reset cursore
            return
        
        data_inizio_g_filtro = None; data_fine_g_filtro = None
        try: data_inizio_g_filtro = self.date_inizio_entry_analisi.get_date()
        except ValueError: pass
        try: data_fine_g_filtro = self.date_fine_entry_analisi.get_date()
        except ValueError: pass
        
        ruote_gioco_verifica, lookahead_verifica, indice_mese_applicazione = self._get_parametri_gioco_comuni()
        if ruote_gioco_verifica is None: 
            self.master.config(cursor="") # Assicura reset cursore
            return

        self._log_to_gui(f"Parametri Ricerca Ambata e Ambo Unico (con Trasformazioni):")
        self._log_to_gui(f"  Estratto Base Globale: {ruota_base}[pos.{pos_base_0idx+1}]")
        self._log_to_gui(f"  Operazioni Base Selezionate: {operazioni_base_selezionate}")
        self._log_to_gui(f"  Trasformazioni Correttore Selezionate: {trasformazioni_correttore_selezionate}")
        self._log_to_gui(f"  Ruote Gioco (Verifica Esito Ambo): {', '.join(ruote_gioco_verifica)}")
        self._log_to_gui(f"  Colpi Lookahead per Ambo: {lookahead_verifica}")
        self._log_to_gui(f"  Indice Mese Applicazione Metodo (per backtest): {indice_mese_applicazione if indice_mese_applicazione is not None else 'Tutte le estrazioni valide nel periodo'}")
        data_inizio_log = data_inizio_g_filtro.strftime('%Y-%m-%d') if data_inizio_g_filtro else "Inizio Storico"
        data_fine_log = data_fine_g_filtro.strftime('%Y-%m-%d') if data_fine_g_filtro else "Fine Storico"
        self._log_to_gui(f"  Periodo Globale Analisi: {data_inizio_log} - {data_fine_log}")
        
        storico_per_analisi = carica_storico_completo(self.cartella_dati_var.get(), 
                                                       data_inizio_g_filtro, 
                                                       data_fine_g_filtro, 
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
                indice_mese_specifico_applicazione=indice_mese_applicazione,
                lookahead_colpi_per_verifica=lookahead_verifica,
                app_logger=self._log_to_gui,
                min_tentativi_per_metodo=self.min_tentativi_var.get() 
            )
            
            self.aau_metodi_trovati_dati = migliori_ambi_config if migliori_ambi_config else []

            # --- Visualizzazione nella Listbox (come prima) ---
            if hasattr(self, 'aau_risultati_listbox') and self.aau_risultati_listbox:
                self.aau_risultati_listbox.delete(0, tk.END)
                if not migliori_ambi_config:
                    self.aau_risultati_listbox.insert(tk.END, "Nessuna configurazione di ambo performante trovata.")
                    self._log_to_gui("Nessuna configurazione di ambo performante trovata.")
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
            
            # --- Preparazione per previsione LIVE e suggerimenti ---
            if migliori_ambi_config:
                self._log_to_gui("\n--- CALCOLO PREVISIONE LIVE E SUGGERIMENTI (Ambata e Ambo Unico) ---")
                
                ultima_estrazione_valida_per_previsione = None
                if storico_per_analisi:
                    if indice_mese_applicazione is not None: # Filtro per indice mese ANCHE per previsione live
                        for estr_rev in reversed(storico_per_analisi):
                            if estr_rev.get('indice_mese') == indice_mese_applicazione:
                                ultima_estrazione_valida_per_previsione = estr_rev
                                break
                        if ultima_estrazione_valida_per_previsione:
                            self._log_to_gui(f"INFO: Previsione live AAU sarà basata sull'ultima {indice_mese_applicazione}a estr. del mese valida: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                        else:
                            self._log_to_gui(f"WARN: Non trovata un'estrazione che sia la {indice_mese_applicazione}a del mese per la previsione live AAU. Impossibile procedere con previsione live.")
                            messagebox.showwarning("Previsione Live AAU", f"Nessuna estrazione trovata che sia la {indice_mese_applicazione}a del mese per calcolare la previsione live.\nControlla i filtri o lo storico.\n(Verranno mostrate solo le performance storiche nel popup).")
                            # Se non troviamo l'estrazione base per la live, potremmo decidere di mostrare comunque
                            # il popup con i dati storici e "N/A" per la previsione live.
                            # Per ora, interrompiamo se non c'è la base per la live.
                            self.master.config(cursor=""); return 
                    else:
                        ultima_estrazione_valida_per_previsione = storico_per_analisi[-1]
                        self._log_to_gui(f"INFO: Previsione live AAU sarà basata sull'ultima estrazione disponibile: {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}")
                else:
                    messagebox.showerror("Errore Dati", "Storico per analisi vuoto, impossibile generare previsione live.")
                    self.master.config(cursor=""); return

                base_extr_dati_live = ultima_estrazione_valida_per_previsione.get(ruota_base, [])
                if not base_extr_dati_live or len(base_extr_dati_live) <= pos_base_0idx:
                    messagebox.showerror("Errore Previsione", f"Dati mancanti per {ruota_base}[pos.{pos_base_0idx+1}] nell'ultima estrazione ({ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')}) valida per la previsione AAU.")
                    self.master.config(cursor=""); return
                numero_base_live_per_previsione = base_extr_dati_live[pos_base_0idx]
                self._log_to_gui(f"INFO: Estratto base per previsione live AAU: {numero_base_live_per_previsione} (da {ruota_base} il {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')})")

                top_metodi_per_ambi_unici_live = []
                ambi_live_gia_selezionati_per_popup = set()
                max_risultati_unici_desiderati = self.num_ambate_var.get()

                for res_conf_storico in migliori_ambi_config:
                    if len(top_metodi_per_ambi_unici_live) >= max_risultati_unici_desiderati: break

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
                            metodo_per_popup['ambata1_live_calcolata'] = a1_live # Salva le ambate live calcolate
                            metodo_per_popup['ambata2_live_calcolata'] = a2_live
                            metodo_per_popup['ambo_live_calcolato'] = ambo_live_normalizzato
                            top_metodi_per_ambi_unici_live.append(metodo_per_popup)
                            ambi_live_gia_selezionati_per_popup.add(ambo_live_normalizzato)
                
                self._log_to_gui(f"INFO: Dopo ricalcolo LIVE e filtro, trovati {len(top_metodi_per_ambi_unici_live)} metodi con ambi UNICI DIVERSI per il popup AAU.")

                if not top_metodi_per_ambi_unici_live:
                    self._log_to_gui("Nessun metodo con ambi unici trovato dopo ricalcolo live e filtro per il popup.")
                    messagebox.showinfo("Ricerca Ambata/Ambo", "Nessuna configurazione valida trovata o solo ambi duplicati/non validi per la previsione live.")
                else:
                    # Log dei suggerimenti basati sui risultati live
                    primo_metodo_live_log = top_metodi_per_ambi_unici_live[0]
                    a1_sugg_live_log = primo_metodo_live_log.get('ambata1_live_calcolata')
                    a2_sugg_live_log = primo_metodo_live_log.get('ambata2_live_calcolata')
                    if a1_sugg_live_log is not None and a2_sugg_live_log is not None:
                        self._log_to_gui(f"  Ambate Singole Consigliate (LIVE): {a1_sugg_live_log}, {a2_sugg_live_log}")
                    self._log_to_gui(f"  Ambi Secchi Consigliati (LIVE):")
                    for i_log, met_log_live in enumerate(top_metodi_per_ambi_unici_live[:3]):
                        ambo_log_live = met_log_live.get('ambo_live_calcolato')
                        self._log_to_gui(f"    {i_log+1}°) Ambo LIVE: {ambo_log_live} (Performance storica ambo: {met_log_live.get('frequenza_ambo',0):.2%})")

                    lista_previsioni_popup_aau = []; dati_grezzi_popup_aau = []
                    for idx_popup, res_popup_dati in enumerate(top_metodi_per_ambi_unici_live):
                        # Usa le ambate e l'ambo LIVE calcolati e memorizzati in res_popup_dati
                        ambata1_live_display = res_popup_dati.get('ambata1_live_calcolata', "N/A")
                        ambata2_live_display = res_popup_dati.get('ambata2_live_calcolata', "N/A")
                        ambo_live_calcolato_tuple = res_popup_dati.get('ambo_live_calcolato')
                        ambo_live_str_display = f"({ambo_live_calcolato_tuple[0]}, {ambo_live_calcolato_tuple[1]})" if ambo_live_calcolato_tuple else "N/D (calcolo live fallito)"
                        
                        formula_origine_popup_display = (
                            f"Metodo Base: {ruota_base}[pos.{pos_base_0idx+1}] (da {ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y')} estr. {numero_base_live_per_previsione}) con:\n"
                            f"  1) Op.Base:'{res_popup_dati.get('op_base1','?')}', Correttore: {res_popup_dati.get('trasf1','?')}({res_popup_dati.get('correttore1_orig','?')}) => Ris. Ambata1 Live: {ambata1_live_display}\n"
                            f"  2) Op.Base:'{res_popup_dati.get('op_base2','?')}', Correttore: {res_popup_dati.get('trasf2','?')}({res_popup_dati.get('correttore2_orig','?')}) => Ris. Ambata2 Live: {ambata2_live_display}"
                        )
                        
                        perf_ambo_storico_str = f"Ambo Secco (storico {res_popup_dati.get('ambo_esempio')}): {res_popup_dati.get('frequenza_ambo',0):.2%} ({res_popup_dati.get('successi_ambo',0)}/{res_popup_dati.get('tentativi_ambo',0)} casi)"
                        ambata1_storico_ex = res_popup_dati.get('ambata1_esempio', 'N/A'); ambata2_storico_ex = res_popup_dati.get('ambata2_esempio', 'N/A')
                        perf_amb1_storico_str = f"  Solo Ambata {ambata1_storico_ex} (da Op1, storico): {res_popup_dati.get('frequenza_ambata1',0):.1%} ({res_popup_dati.get('successi_ambata1',0)}/{res_popup_dati.get('tentativi_ambata1',0)})"
                        perf_amb2_storico_str = f"  Solo Ambata {ambata2_storico_ex} (da Op2, storico): {res_popup_dati.get('frequenza_ambata2',0):.1%} ({res_popup_dati.get('successi_ambata2',0)}/{res_popup_dati.get('tentativi_ambata2',0)})"
                        performance_completa_str_popup = f"{perf_ambo_storico_str}\n{perf_amb1_storico_str}\n{perf_amb2_storico_str}"
                        if 'frequenza_almeno_una_ambata' in res_popup_dati:
                            freq_almeno_una = res_popup_dati['frequenza_almeno_una_ambata']; succ_almeno_una = res_popup_dati.get('successi_almeno_una_ambata', 0); tent_almeno_una = res_popup_dati.get('tentativi_almeno_una_ambata', res_popup_dati.get('tentativi_ambo', 0))
                            performance_completa_str_popup += f"\n  Almeno una Ambata ({ambata1_storico_ex} o {ambata2_storico_ex}, storico): {freq_almeno_una:.1%} ({succ_almeno_una}/{tent_almeno_una})"

                        suggerimento_gioco_popup_str = ""
                        if idx_popup == 0 and ambata1_live_display != "N/A" and ambata2_live_display != "N/A" and ambo_live_calcolato_tuple:
                            suggerimento_gioco_popup_str = (f"\nStrategia di Gioco Suggerita (basata su previsione live di questo 1° metodo):\n  - Giocare Ambata Singola: {ambata1_live_display}\n  - Giocare Ambata Singola: {ambata2_live_display}\n  - Giocare Ambo Secco: {ambo_live_str_display}")
                            if len(top_metodi_per_ambi_unici_live) > 1:
                                ambo2_sugg_live = top_metodi_per_ambi_unici_live[1].get('ambo_live_calcolato'); ambo2_sugg_str = f"({ambo2_sugg_live[0]}, {ambo2_sugg_live[1]})" if ambo2_sugg_live else "N/D"
                                suggerimento_gioco_popup_str += f"\n  - Giocare 2° Ambo Secco (LIVE): {ambo2_sugg_str}"
                            if len(top_metodi_per_ambi_unici_live) > 2:
                                ambo3_sugg_live = top_metodi_per_ambi_unici_live[2].get('ambo_live_calcolato'); ambo3_sugg_str = f"({ambo3_sugg_live[0]}, {ambo3_sugg_live[1]})" if ambo3_sugg_live else "N/D"
                                suggerimento_gioco_popup_str += f"\n  - Giocare 3° Ambo Secco (LIVE): {ambo3_sugg_str}"
                            performance_completa_str_popup += suggerimento_gioco_popup_str

                        dettaglio_popup = {"titolo_sezione": f"--- {(idx_popup+1)}ª Configurazione Proposta (Ambo Unico) ---", "info_metodo_str": formula_origine_popup_display, "ambata_prevista": f"PREVISIONE DA GIOCARE: AMBO DA GIOCARE: {ambo_live_str_display}", "performance_storica_str": performance_completa_str_popup, "abbinamenti_dict": {}, "contorni_suggeriti": [] }
                        lista_previsioni_popup_aau.append(dettaglio_popup)
                        
                        dati_salvataggio = res_popup_dati.copy(); dati_salvataggio["tipo_metodo_salvato"] = "ambata_ambo_unico_trasf"; dati_salvataggio["formula_testuale"] = formula_origine_popup_display; dati_salvataggio["ruota_base_origine"] = ruota_base; dati_salvataggio["pos_base_origine"] = pos_base_0idx
                        if ambo_live_calcolato_tuple: dati_salvataggio["definizione_strutturata"] = list(ambo_live_calcolato_tuple) # Salva l'ambo live
                        else: dati_salvataggio["definizione_strutturata"] = None
                        dati_salvataggio["ambata_prevista_live"] = ambo_live_str_display # Esplicita l'ambo live nel salvataggio
                        dati_grezzi_popup_aau.append(dati_salvataggio)
                    
                    if lista_previsioni_popup_aau:
                        self.mostra_popup_previsione(
                            titolo_popup="Migliori Configurazioni Ambata e Ambo Unico", ruote_gioco_str=", ".join(ruote_gioco_verifica),
                            lista_previsioni_dettagliate=lista_previsioni_popup_aau, data_riferimento_previsione_str_comune=ultima_estrazione_valida_per_previsione['data'].strftime('%d/%m/%Y'),
                            metodi_grezzi_per_salvataggio=dati_grezzi_popup_aau
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


    def mostra_popup_previsione(self, titolo_popup, ruote_gioco_str, lista_previsioni_dettagliate=None, copertura_combinata_info=None, data_riferimento_previsione_str_comune=None, metodi_grezzi_per_salvataggio=None ):
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo_popup)
        
        popup_width = 700
        popup_base_height_per_method_section = 240 # Aumentato per dare più spazio
        abbinamenti_h_approx = 150 
        contorni_h_approx = 70   
        
        dynamic_height_needed = 150 
        if copertura_combinata_info: dynamic_height_needed += 80

        if lista_previsioni_dettagliate:
            for prev_dett_c in lista_previsioni_dettagliate:
                current_met_h = popup_base_height_per_method_section
                ambata_val_check = prev_dett_c.get('ambata_prevista')
                is_single_number_for_abbinamenti = False
                if isinstance(ambata_val_check, (int, float)):
                    is_single_number_for_abbinamenti = True
                elif isinstance(ambata_val_check, str) and ambata_val_check.isdigit():
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
        
        self._log_to_gui(f"DEBUG POPUP (mostra_popup_previsione): Titolo: {titolo_popup}") 

        row_idx = 0
        ttk.Label(scrollable_frame, text=f"--- {titolo_popup} ---", font=("Helvetica", 12, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1
        if data_riferimento_previsione_str_comune: ttk.Label(scrollable_frame, text=f"Previsione del: {data_riferimento_previsione_str_comune}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
        ttk.Label(scrollable_frame, text=f"Su ruote: {ruote_gioco_str}").grid(row=row_idx, column=0, columnspan=2, pady=(2,10), sticky="w"); row_idx += 1

        if copertura_combinata_info and "testo_introduttivo" in copertura_combinata_info:
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5); row_idx += 1
            ttk.Label(scrollable_frame, text=copertura_combinata_info['testo_introduttivo'], wraplength=popup_width - 40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1

        if lista_previsioni_dettagliate:
            for idx_metodo, previsione_dett in enumerate(lista_previsioni_dettagliate):
                self._log_to_gui(f"DEBUG POPUP (mostra_popup): Processando previsione_dett #{idx_metodo}: {previsione_dett.get('titolo_sezione', 'N/A')}")
                ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10); row_idx += 1
                titolo_sezione = previsione_dett.get('titolo_sezione', '--- PREVISIONE ---'); ttk.Label(scrollable_frame, text=titolo_sezione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=3, sticky="w"); row_idx += 1
                formula_metodo_display = previsione_dett.get('info_metodo_str', "N/D")
                if formula_metodo_display != "N/D": ttk.Label(scrollable_frame, text=f"Metodo: {formula_metodo_display}", wraplength=popup_width-40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                
                ambata_loop = previsione_dett.get('ambata_prevista')
                self._log_to_gui(f"DEBUG POPUP (mostra_popup): ambata_loop (o previsione) per sezione '{titolo_sezione}' = {ambata_loop} (tipo: {type(ambata_loop)})") 
                
                if ambata_loop is None or str(ambata_loop).upper() in ["N/D", "N/A"]:
                    ttk.Label(scrollable_frame, text="Nessuna previsione valida.").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                else:
                    testo_previsione_popup = f"PREVISIONE DA GIOCARE: {ambata_loop}"
                    ttk.Label(scrollable_frame, text=testo_previsione_popup, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                
                performance_str_display = previsione_dett.get('performance_storica_str', 'N/D')
                ttk.Label(scrollable_frame, text=f"Performance storica:\n{performance_str_display}", justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1

                dati_grezzi_per_questo_metodo = None
                if metodi_grezzi_per_salvataggio and idx_metodo < len(metodi_grezzi_per_salvataggio):
                    dati_grezzi_per_questo_metodo = metodi_grezzi_per_salvataggio[idx_metodo]
                
                if dati_grezzi_per_questo_metodo:
                    estensione_default = ".lmp" 
                    tipo_metodo_salv = dati_grezzi_per_questo_metodo.get("tipo_metodo_salvato", "sconosciuto")
                    if tipo_metodo_salv.startswith("condizionato"):
                        estensione_default = ".lmcondcorr" if "corretto" in tipo_metodo_salv else ".lmcond"
                    elif tipo_metodo_salv == "ambata_ambo_unico_auto": 
                        estensione_default = ".lmaau"

                    btn_salva_profilo = ttk.Button(scrollable_frame, text="Salva Questo Metodo", 
                                                   command=lambda d=dati_grezzi_per_questo_metodo.copy(), e=estensione_default: self._prepara_e_salva_profilo_metodo(d, estensione=e))
                    btn_salva_profilo.grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="ew"); row_idx += 1
                
                ambata_per_abbinamenti_popup = None
                if isinstance(ambata_loop, (int, float)):
                    ambata_per_abbinamenti_popup = ambata_loop
                elif isinstance(ambata_loop, str) and ambata_loop.isdigit():
                    ambata_per_abbinamenti_popup = int(ambata_loop)
                
                self._log_to_gui(f"DEBUG POPUP (mostra_popup) Sezione Abbinamenti: ambata_per_abbinamenti_popup='{ambata_per_abbinamenti_popup}', tipo={type(ambata_per_abbinamenti_popup)}, previsione_dett['abbinamenti_dict'] esiste? {'abbinamenti_dict' in previsione_dett}")

                if ambata_per_abbinamenti_popup is not None: 
                    ttk.Label(scrollable_frame, text="Abbinamenti Consigliati (co-occorrenze storiche):").grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="w"); row_idx +=1
                    abbinamenti_dict_loop = previsione_dett.get('abbinamenti_dict', {}); 
                    eventi_totali_loop = abbinamenti_dict_loop.get("sortite_ambata_target", 0)
                    self._log_to_gui(f"DEBUG POPUP (mostra_popup): eventi_totali_loop (sortite ambata target per abbinamenti) = {eventi_totali_loop}")
                    
                    if eventi_totali_loop > 0:
                        # ... (logica per mostrare abbinamenti come prima) ...
                        pass 
                    else:
                        ttk.Label(scrollable_frame, text=f"  Nessuna co-occorrenza storica per l'ambata {ambata_per_abbinamenti_popup} (eventi_totali_loop={eventi_totali_loop}).").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                    
                    contorni_suggeriti_loop = previsione_dett.get('contorni_suggeriti', [])
                    # ... (logica per mostrare contorni come prima) ...
                else:
                    self._log_to_gui(f"DEBUG POPUP (mostra_popup): Nessun abbinamento mostrato perché ambata_loop ('{ambata_loop}') non è un singolo numero.")

        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=(5,0)); scrollbar_y.pack(side="right", fill="y")
        close_button_frame = ttk.Frame(popup_window); close_button_frame.pack(fill=tk.X, pady=(5,5), padx=5, side=tk.BOTTOM)
        ttk.Button(close_button_frame, text="Chiudi", command=popup_window.destroy).pack()
        popup_window.update_idletasks(); canvas.config(scrollregion=canvas.bbox("all")) 
        try: self.master.eval(f'tk::PlaceWindow {str(popup_window)} center')
        except tk.TclError: 
            # ... (fallback per centrare la finestra)
            pass
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
        
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return
        ruota_calcolo_input = self.ruota_calcolo_var.get()
        posizione_estratto_input = self.posizione_estratto_var.get() - 1 
        num_ambate_richieste_gui = self.num_ambate_var.get()
        min_tentativi = self.min_tentativi_var.get()

        self._log_to_gui(f"Parametri Ricerca Metodi Semplici:\n  Ruota Base: {ruota_calcolo_input}, Posizione: {posizione_estratto_input+1}")
        self._log_to_gui(f"  Ruote Gioco: {', '.join(ruote_gioco)}, Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        self._log_to_gui(f"  Output Ambate Popup: {num_ambate_richieste_gui}, Min. Tentativi: {min_tentativi}")

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            
            risultati_individuali_grezzi, info_copertura_combinata = trova_migliori_ambate_e_abbinamenti(
                storico_per_analisi,             
                ruota_calcolo_input,             
                posizione_estratto_input,        
                ruote_gioco,                     
                max_ambate_output=num_ambate_richieste_gui, 
                lookahead=lookahead,
                indice_mese_filtro=indice_mese, 
                min_tentativi_per_ambata=min_tentativi, 
                app_logger=self._log_to_gui
            )
            
            self.metodi_semplici_trovati_dati = risultati_individuali_grezzi if risultati_individuali_grezzi else []
            self._log_to_gui(f"DEBUG: Numero metodi semplici grezzi salvati in self.metodi_semplici_trovati_dati: {len(self.metodi_semplici_trovati_dati)}")

            if hasattr(self, 'ms_risultati_listbox') and self.ms_risultati_listbox:
                if not self.metodi_semplici_trovati_dati: 
                    self.ms_risultati_listbox.insert(tk.END, "Nessun metodo semplice valido trovato.")
                else:
                    for i, res in enumerate(self.metodi_semplici_trovati_dati): 
                        met_info = res['metodo']
                        formula = f"{met_info['ruota_calcolo']}[{met_info['pos_estratto_calcolo']+1}] {met_info['operazione']} {met_info['operando_fisso']}"
                        riga_listbox = (f"{(i+1)}. {(res['frequenza_ambata']*100):>5.1f}% ({res['successi']}/{res['tentativi']}) "
                                        f"-> {formula}")
                        self.ms_risultati_listbox.insert(tk.END, riga_listbox)
            
            if risultati_individuali_grezzi: 
                 lista_previsioni_per_popup = []
                 dati_grezzi_per_popup_finali_per_popup = [] 
                 data_riferimento_str_popup_s_comune = storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else "N/D"
                
                 for res_idx, res_singolo_metodo in enumerate(risultati_individuali_grezzi[:num_ambate_richieste_gui]):
                     metodo_s_info = res_singolo_metodo['metodo']
                     formula_testuale_semplice = f"{metodo_s_info['ruota_calcolo']}[pos.{metodo_s_info['pos_estratto_calcolo']+1}] {metodo_s_info['operazione']} {metodo_s_info['operando_fisso']}"
                     dettaglio_previsione_per_popup = {
                         "titolo_sezione": f"--- {(res_idx+1)}° METODO / PREVISIONE ---",
                         "info_metodo_str": formula_testuale_semplice,
                         "ambata_prevista": res_singolo_metodo.get('ambata_piu_frequente_dal_metodo'), 
                         "abbinamenti_dict": res_singolo_metodo.get("abbinamenti", {}),
                         "performance_storica_str": f"{res_singolo_metodo['frequenza_ambata']:.2%} ({res_singolo_metodo['successi']}/{res_singolo_metodo['tentativi']} casi)",
                         "contorni_suggeriti": res_singolo_metodo.get("abbinamenti", {}).get("contorni_per_popup_simple", []) 
                     }
                     lista_previsioni_per_popup.append(dettaglio_previsione_per_popup)
                     
                     dati_metodo_per_popup = res_singolo_metodo.copy() 
                     dati_metodo_per_popup["tipo_metodo_salvato"] = "semplice_analizzato" 
                     dati_grezzi_per_popup_finali_per_popup.append(dati_metodo_per_popup)

                 ruote_gioco_str_popup_s = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                
                 self.mostra_popup_previsione(
                    titolo_popup="Previsione Metodi Semplici", 
                    ruote_gioco_str=ruote_gioco_str_popup_s,
                    lista_previsioni_dettagliate=lista_previsioni_per_popup,
                    copertura_combinata_info=info_copertura_combinata if num_ambate_richieste_gui > 1 else None,
                    data_riferimento_previsione_str_comune=data_riferimento_str_popup_s_comune,
                    metodi_grezzi_per_salvataggio=dati_grezzi_per_popup_finali_per_popup
                 )
            elif not (hasattr(self, 'ms_risultati_listbox') and self.ms_risultati_listbox and self.ms_risultati_listbox.size() > 0 and self.ms_risultati_listbox.get(0).startswith("Nessun")):
                messagebox.showinfo("Analisi Metodi Semplici", "Nessun metodo semplice ha prodotto risultati sufficientemente frequenti.")
            
            self._log_to_gui("\n--- Ricerca Metodi Semplici Completata ---")
        except Exception as e: 
            messagebox.showerror("Errore Analisi", f"Errore ricerca metodi semplici: {e}"); 
            self._log_to_gui(f"ERRORE CRITICO: {e}\n{traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def _prepara_metodo_per_backtest(self, dati_metodo_selezionato_per_prep):
        self._log_to_gui(f"DEBUG: _prepara_metodo_per_backtest CHIAMATO con dati: {dati_metodo_selezionato_per_prep}") 

        tipo_metodo = dati_metodo_selezionato_per_prep.get('tipo') if dati_metodo_selezionato_per_prep else None
        formula_ok = bool(dati_metodo_selezionato_per_prep.get('formula_testuale')) if dati_metodo_selezionato_per_prep else False
        
        def_strutturata_presente = 'definizione_strutturata' in dati_metodo_selezionato_per_prep if dati_metodo_selezionato_per_prep else False
        condizione_speciale_tipo = False
        if tipo_metodo:
            condizione_speciale_tipo = tipo_metodo.startswith("periodica_") or \
                                       (tipo_metodo.startswith("condizionato_") and 'condizione_primaria' in dati_metodo_selezionato_per_prep)

        condizione_valida_per_salvataggio = tipo_metodo and formula_ok and (def_strutturata_presente or condizione_speciale_tipo)

        if condizione_valida_per_salvataggio:
            self.metodo_preparato_per_backtest = dati_metodo_selezionato_per_prep.copy() 
            
            formula_display = self.metodo_preparato_per_backtest['formula_testuale']
            tipo_display = self.metodo_preparato_per_backtest['tipo'].replace("_", " ").title()

            if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
                self.mc_listbox_componenti_1.delete(0, tk.END)
                self.mc_listbox_componenti_1.insert(tk.END, f"PER BACKTEST ({tipo_display}):")
                self.mc_listbox_componenti_1.insert(tk.END, formula_display)
            
            messagebox.showinfo("Metodo Pronto per Backtest", 
                                f"Metodo ({tipo_display}):\n{formula_display}\n"
                                "è stato selezionato.\n\n"
                                "Ora puoi usare il pulsante 'Backtest Dettagliato'.")
            self._log_to_gui(f"INFO: Metodo selezionato da popup per backtest dettagliato ({tipo_display}): {formula_display}")
            self._log_to_gui(f"DEBUG: Dati completi del metodo preparato in self.metodo_preparato_per_backtest: {self.metodo_preparato_per_backtest}")
            
            if hasattr(self, 'usa_ultimo_corretto_per_backtest_var'):
                self.usa_ultimo_corretto_per_backtest_var.set(False) 
                self._log_to_gui("INFO: Checkbox 'Usa ultimo metodo corretto' deselezionato perché un metodo è stato preparato da popup.")
        else:
            messaggio_errore_dett = "Dati del metodo selezionato non validi o definizione per backtest mancante."
            if dati_metodo_selezionato_per_prep:
                messaggio_errore_dett += f"\nDati ricevuti: Tipo='{tipo_metodo}', FormulaOK={formula_ok}, " \
                                         f"DefStrutturataPresente={def_strutturata_presente}, CondSpecialeTipoOK={condizione_speciale_tipo}"
            messagebox.showerror("Errore Preparazione Backtest", messaggio_errore_dett)
            self._log_to_gui(f"WARN: _prepara_metodo_per_backtest chiamato con dati metodo non validi. {messaggio_errore_dett}")
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
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return
        self.master.config(cursor="watch"); self.master.update_idletasks()
        lista_previsioni_popup_mc_vis = []; info_copertura_combinata_mc = None
        data_riferimento_comune_popup = storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else None
        try:
            for idx, metodo_def_corrente in enumerate(metodi_validi_per_analisi_def):
                nome_metodo_log_popup = f"Metodo Base {idx + 1}"; self._log_to_gui(f"\n--- ANALISI {nome_metodo_log_popup.upper()} ---")
                metodo_str_popup = "".join(self._format_componente_per_display(comp) for comp in metodo_def_corrente)
                self._log_to_gui(f"  Definizione: {metodo_str_popup}")
                s_ind, t_ind, applicazioni_vincenti_ind = analizza_metodo_complesso_specifico(storico_per_analisi, metodo_def_corrente, ruote_gioco, lookahead, indice_mese, self._log_to_gui)
                f_ind = s_ind / t_ind if t_ind > 0 else 0.0
                perf_ind_str = f"{f_ind:.2%} ({s_ind}/{t_ind} casi)" if t_ind > 0 else "Non applicabile storicamente."
                self._log_to_gui(f"  Performance Storica Individuale {nome_metodo_log_popup}: {perf_ind_str}")
                ambata_live, abb_live = self._calcola_previsione_e_abbinamenti_metodo_complesso(storico_per_analisi, metodo_def_corrente, ruote_gioco, data_riferimento_comune_popup, nome_metodo_log_popup)
                lista_previsioni_popup_mc_vis.append({"titolo_sezione": f"--- PREVISIONE {nome_metodo_log_popup.upper()} ---", "info_metodo_str": metodo_str_popup, "ambata_prevista": ambata_live, "abbinamenti_dict": abb_live, "performance_storica_str": perf_ind_str})
                metodi_grezzi_per_popup_salvataggio.append({"tipo_metodo_salvato": "complesso_base_analizzato", "definizione_metodo_originale": metodo_def_corrente, "formula_testuale": metodo_str_popup, "ambata_prevista": ambata_live, "abbinamenti": abb_live, "successi": s_ind, "tentativi": t_ind, "frequenza": f_ind, "applicazioni_vincenti_dettagliate": applicazioni_vincenti_ind})
            if len(metodi_validi_per_analisi_def) == 2:
                self._log_to_gui("\n--- ANALISI PERFORMANCE COMBINATA METODI COMPLESSI BASE ---")
                s_comb_mc, t_comb_mc, f_comb_mc = analizza_copertura_ambate_previste_multiple(storico_per_analisi, metodi_validi_per_analisi_def, ruote_gioco,lookahead, indice_mese, self._log_to_gui)
                if t_comb_mc > 0:
                    self._log_to_gui(f"  Successi Combinati (almeno un'ambata vincente): {s_comb_mc}"); self._log_to_gui(f"  Tentativi Combinati (almeno un metodo applicabile): {t_comb_mc}"); self._log_to_gui(f"  Frequenza di Copertura Combinata Metodi Complessi: {f_comb_mc:.2%}")
                    info_copertura_combinata_mc = {"successi": s_comb_mc, "tentativi": t_comb_mc, "frequenza": f_comb_mc, "num_metodi_combinati": len(metodi_validi_per_analisi_def)}
                else: self._log_to_gui("  Nessun tentativo combinato applicabile.")
            self.master.config(cursor="")
            if not lista_previsioni_popup_mc_vis: messagebox.showinfo("Analisi Metodi Complessi", "Nessun metodo complesso valido ha prodotto una previsione popup.")
            else:
                ruote_gioco_str_popup = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                self.mostra_popup_previsione(
                   titolo_popup="Previsioni Metodi Complessi Base", ruote_gioco_str=ruote_gioco_str_popup,
                   lista_previsioni_dettagliate=lista_previsioni_popup_mc_vis, copertura_combinata_info=info_copertura_combinata_mc,
                   data_riferimento_previsione_str_comune=data_riferimento_comune_popup, metodi_grezzi_per_salvataggio=metodi_grezzi_per_popup_salvataggio
                )
            self._log_to_gui("\n--- Analisi Metodi Complessi Base Completata ---")
        except Exception as e: self.master.config(cursor=""); messagebox.showerror("Errore Analisi", f"Errore analisi metodi complessi: {e}"); self._log_to_gui(f"ERRORE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def _calcola_previsione_e_abbinamenti_metodo_complesso(self, storico_attuale, definizione_metodo, ruote_gioco, data_riferimento_str, nome_metodo_log="Metodo"):
        ambata_live = None; abbinamenti_live = {}; note_previsione_log = ""
        if storico_attuale:
            ultima_estrazione = storico_attuale[-1]
            val_raw = calcola_valore_metodo_complesso(ultima_estrazione, definizione_metodo, self._log_to_gui)
            if val_raw is not None: ambata_live = regola_fuori_90(val_raw)
            else: note_previsione_log = f"{nome_metodo_log} non applicabile all'ultima estrazione."
        else: note_previsione_log = f"Storico vuoto per {nome_metodo_log}."
        self._log_to_gui(f"\n  PREVISIONE LIVE {nome_metodo_log} (da estrazione del {data_riferimento_str or 'N/D'}):")
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
        else: self._log_to_gui(f"    {note_previsione_log if note_previsione_log else 'Impossibile calcolare previsione.'}")
        return ambata_live, abbinamenti_live


    def avvia_ricerca_correttore(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA CORRETTORE OTTIMALE (PER METODI COMPLESSI BASE)\n" + "="*50)
        
        # Resetta sempre le variabili del metodo corretto all'inizio di una nuova ricerca
        self.ultimo_metodo_corretto_trovato_definizione = None 
        self.ultimo_metodo_corretto_formula_testuale = ""
        self.usa_ultimo_corretto_per_backtest_var.set(False)

        metodo_1_def_corr_base = self.definizione_metodo_complesso_attuale
        metodo_2_def_corr_base = self.definizione_metodo_complesso_attuale_2
        err_msg_corr = []
        almeno_un_metodo_base_valido_per_correttore = False
        metodo_1_valido_def = None # Inizializza a None
        metodo_2_valido_def = None # Inizializza a None

        if metodo_1_def_corr_base:
            if metodo_1_def_corr_base[-1].get('operazione_successiva') == '=':
                almeno_un_metodo_base_valido_per_correttore = True
                metodo_1_valido_def = metodo_1_def_corr_base # Assegna la lista
            else: err_msg_corr.append("Metodo Base 1 non terminato con '=' o vuoto.")
        if metodo_2_def_corr_base:
            if metodo_2_def_corr_base[-1].get('operazione_successiva') == '=':
                almeno_un_metodo_base_valido_per_correttore = True
                metodo_2_valido_def = metodo_2_def_corr_base # Assegna la lista
            else: err_msg_corr.append("Metodo Base 2 non terminato con '=' o vuoto.")

        if err_msg_corr and almeno_un_metodo_base_valido_per_correttore:
             self._log_to_gui("AVVISO VALIDAZIONE METODI PER CORRETTORE:\n" + "\n".join(err_msg_corr))
        elif err_msg_corr and not almeno_un_metodo_base_valido_per_correttore: # Se ci sono errori e nessun metodo è valido
            messagebox.showerror("Errore Input", "Definire almeno un Metodo Base valido (terminato con '=') per cercare un correttore.\n" + "\n".join(err_msg_corr))
            self._log_to_gui("ERRORE: Nessun Metodo Base valido per la ricerca correttore."); return
        
        if not almeno_un_metodo_base_valido_per_correttore: # Se non ci sono errori ma comunque nessun metodo valido
            messagebox.showerror("Errore Input", "Definire almeno un Metodo Base valido (terminato con '=') per cercare un correttore.")
            self._log_to_gui("ERRORE: Nessun Metodo Base valido definito per la ricerca correttore."); return

        storico_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi: return
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return

        min_tentativi_correttore_cfg = self.corr_cfg_min_tentativi.get()
        c_fisso_s = self.corr_cfg_cerca_fisso_semplice.get(); c_estr_s = self.corr_cfg_cerca_estratto_semplice.get()
        c_somma_ef = self.corr_cfg_cerca_somma_estr_fisso.get(); c_somma_ee = self.corr_cfg_cerca_somma_estr_estr.get()
        c_diff_ef = self.corr_cfg_cerca_diff_estr_fisso.get(); c_diff_ee = self.corr_cfg_cerca_diff_estr_estr.get()
        c_mult_ef = self.corr_cfg_cerca_mult_estr_fisso.get(); c_mult_ee = self.corr_cfg_cerca_mult_estr_estr.get()

        tipi_corr_log = []
        if c_fisso_s: tipi_corr_log.append("Fisso Singolo")
        if c_estr_s: tipi_corr_log.append("Estratto Singolo")
        if c_somma_ef: tipi_corr_log.append("Estratto+Fisso (Somma)")
        if c_somma_ee: tipi_corr_log.append("Estratto+Estratto (Somma)")
        if c_diff_ef: tipi_corr_log.append("Estratto-Fisso (Diff)")
        if c_diff_ee: tipi_corr_log.append("Estratto-Estratto (Diff)")
        if c_mult_ef: tipi_corr_log.append("Estratto*Fisso (Mult)")
        if c_mult_ee: tipi_corr_log.append("Estratto*Estratto (Mult)")

        self._log_to_gui("Parametri Ricerca Correttore (per Metodi Complessi Base):")
        if metodo_1_valido_def: self._log_to_gui(f"  Metodo Base 1: {''.join(self._format_componente_per_display(c) for c in metodo_1_valido_def)}")
        if metodo_2_valido_def: self._log_to_gui(f"  Metodo Base 2: {''.join(self._format_componente_per_display(c) for c in metodo_2_valido_def)}")
        self._log_to_gui(f"  Opzioni Gioco: Ruote: {', '.join(ruote_gioco)}, Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        self._log_to_gui(f"  Min. Tentativi Correttore: {min_tentativi_correttore_cfg}")
        self._log_to_gui(f"  Tipi Correttore Selezionati: {', '.join(tipi_corr_log) if tipi_corr_log else 'Nessuno (controllare impostazioni)'}")

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati_correttori_list = trova_miglior_correttore_per_metodo_complesso(
                storico_per_analisi,
                metodo_1_valido_def, # Passa la lista o None
                metodo_2_valido_def, # Passa la lista o None
                c_fisso_s, c_estr_s,
                c_diff_ef, c_diff_ee,
                c_mult_ef, c_mult_ee,
                c_somma_ef, c_somma_ee,
                ruote_gioco, lookahead, indice_mese,
                min_tentativi_correttore_cfg, app_logger=self._log_to_gui,
                filtro_condizione_primaria_dict=None
            )

            self._log_to_gui("\n\n--- RISULTATI RICERCA CORRETTORI (LOG COMPLETO) ---")
            if not risultati_correttori_list:
                self._log_to_gui("Nessun correttore valido trovato che migliori il benchmark dei metodi base.")
                messagebox.showinfo("Ricerca Correttore", "Nessun correttore valido trovato che migliori il benchmark.")
            else:
                miglior_risultato_correttore = risultati_correttori_list[0]
                
                # --- MODIFICA CHIAVE: Salva le definizioni corrette in un dizionario ---
                definizioni_corrette_salvate = {}
                formula_testuale_complessiva_display = "N/A" 

                if miglior_risultato_correttore.get('def_metodo_esteso_1'):
                    def_m1c = list(miglior_risultato_correttore.get('def_metodo_esteso_1'))
                    definizioni_corrette_salvate['base1_corretto'] = def_m1c
                    formula_testuale_complessiva_display = "".join(self._format_componente_per_display(comp) for comp in def_m1c)

                if miglior_risultato_correttore.get('def_metodo_esteso_2'):
                    def_m2c = list(miglior_risultato_correttore.get('def_metodo_esteso_2'))
                    definizioni_corrette_salvate['base2_corretto'] = def_m2c
                    if formula_testuale_complessiva_display == "N/A": # Se il metodo 1 non c'era
                        formula_testuale_complessiva_display = "".join(self._format_componente_per_display(comp) for comp in def_m2c)
                
                if definizioni_corrette_salvate:
                    self.ultimo_metodo_corretto_trovato_definizione = definizioni_corrette_salvate # Ora è un dizionario
                    self.ultimo_metodo_corretto_formula_testuale = formula_testuale_complessiva_display # Una formula rappresentativa
                    self._log_to_gui(f"INFO: Metodi corretti aggiornati in memoria: Formula rappresentativa: '{self.ultimo_metodo_corretto_formula_testuale}', Dettagli def: {self.ultimo_metodo_corretto_trovato_definizione}")
                    self.usa_ultimo_corretto_per_backtest_var.set(True) 
                else: 
                    self.ultimo_metodo_corretto_trovato_definizione = None
                    self.ultimo_metodo_corretto_formula_testuale = ""
                    self.usa_ultimo_corretto_per_backtest_var.set(False)
                # --- FINE MODIFICA CHIAVE ---

                lista_previsioni_popup_corr_vis = []
                metodi_grezzi_corretti_per_salvataggio_popup = [] # Per il pulsante "Salva Metodo" nel popup
                data_riferimento_comune_popup_corr = storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else "N/D"
                
                info_correttore_globale_str = (
                    f"Correttore Applicato: {miglior_risultato_correttore['tipo_correttore_descrittivo']} -> {miglior_risultato_correttore['dettaglio_correttore_str']}\n"
                    f"Operazione di Collegamento Base: '{miglior_risultato_correttore['operazione_collegamento_base']}'\n"
                    f"Performance Globale del Metodo/i Corretto/i: {miglior_risultato_correttore['frequenza']:.2%} ({miglior_risultato_correttore['successi']}/{miglior_risultato_correttore['tentativi']} casi)"
                )
                
                # Prepara i dati per il popup, mostrando entrambi i metodi corretti se esistono
                if 'base1_corretto' in definizioni_corrette_salvate:
                    met1_est_def_popup = definizioni_corrette_salvate['base1_corretto']
                    met1_est_str_popup = "".join(self._format_componente_per_display(comp) for comp in met1_est_def_popup)
                    ambata1_corr_live_popup, abb1_corr_live_popup = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                        storico_per_analisi, met1_est_def_popup, ruote_gioco, data_riferimento_comune_popup_corr, "Metodo 1 Corretto"
                    )
                    lista_previsioni_popup_corr_vis.append({
                        "titolo_sezione": "--- PREVISIONE METODO 1 CORRETTO ---", 
                        "info_metodo_str": met1_est_str_popup,
                        "ambata_prevista": ambata1_corr_live_popup, 
                        "abbinamenti_dict": abb1_corr_live_popup,
                        "performance_storica_str": "Vedi performance globale correttore" # La performance è aggregata
                    })
                    metodi_grezzi_corretti_per_salvataggio_popup.append({
                        "tipo_metodo_salvato": "complesso_corretto", # Tipo specifico per il salvataggio
                        "riferimento_base": "base1", # Per sapere a quale base si riferisce
                        "definizione_metodo_base_originale_1": metodo_1_valido_def, # Salva l'originale
                        "def_metodo_esteso_1": met1_est_def_popup, 
                        "formula_testuale": met1_est_str_popup, 
                        "ambata_prevista": ambata1_corr_live_popup, 
                        "abbinamenti": abb1_corr_live_popup,
                        "tipo_correttore_descrittivo": miglior_risultato_correttore['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_risultato_correttore['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_risultato_correttore['operazione_collegamento_base'],
                        "successi": miglior_risultato_correttore['successi'], # Performance aggregata
                        "tentativi": miglior_risultato_correttore['tentativi'],
                        "frequenza": miglior_risultato_correttore['frequenza'],
                    })

                if 'base2_corretto' in definizioni_corrette_salvate:
                    met2_est_def_popup = definizioni_corrette_salvate['base2_corretto']
                    met2_est_str_popup = "".join(self._format_componente_per_display(comp) for comp in met2_est_def_popup)
                    ambata2_corr_live_popup, abb2_corr_live_popup = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                        storico_per_analisi, met2_est_def_popup, ruote_gioco, data_riferimento_comune_popup_corr, "Metodo 2 Corretto"
                    )
                    lista_previsioni_popup_corr_vis.append({
                        "titolo_sezione": "--- PREVISIONE METODO 2 CORRETTO ---", 
                        "info_metodo_str": met2_est_str_popup,
                        "ambata_prevista": ambata2_corr_live_popup, 
                        "abbinamenti_dict": abb2_corr_live_popup,
                        "performance_storica_str": "Vedi performance globale correttore"
                    })
                    metodi_grezzi_corretti_per_salvataggio_popup.append({
                        "tipo_metodo_salvato": "complesso_corretto",
                        "riferimento_base": "base2",
                        "definizione_metodo_base_originale_2": metodo_2_valido_def,
                        "def_metodo_esteso_2": met2_est_def_popup, # Nota: chiave diversa per distinguerlo
                        "formula_testuale": met2_est_str_popup, 
                        "ambata_prevista": ambata2_corr_live_popup, 
                        "abbinamenti": abb2_corr_live_popup,
                        "tipo_correttore_descrittivo": miglior_risultato_correttore['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_risultato_correttore['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_risultato_correttore['operazione_collegamento_base'],
                        "successi": miglior_risultato_correttore['successi'], 
                        "tentativi": miglior_risultato_correttore['tentativi'],
                        "frequenza": miglior_risultato_correttore['frequenza'],
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
                       data_riferimento_previsione_str_comune=data_riferimento_comune_popup_corr,
                       metodi_grezzi_per_salvataggio=metodi_grezzi_corretti_per_salvataggio_popup
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

        storico_per_analisi_cond = self._carica_e_valida_storico_comune()
        if not storico_per_analisi_cond: return

        ruote_gioco_cond, lookahead_cond, indice_mese_cond = self._get_parametri_gioco_comuni()
        if ruote_gioco_cond is None: return

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
            self._log_to_gui("ERRORE: Valore Min Condizione > Valore Max Condizione"); return

        self._log_to_gui("Parametri Analisi Condizionata:")
        self._log_to_gui(f"  Condizione: {ruota_cond_input}[pos.{pos_cond_input_1based}] in [{val_min_cond_input}-{val_max_cond_input}]")
        self._log_to_gui(f"  Calcolo Ambata: da {ruota_calc_amb_input}[pos.{pos_calc_amb_input_1based}] +/-/* Fisso")
        self._log_to_gui(f"  Ruote Gioco: {', '.join(ruote_gioco_cond)}, Colpi: {lookahead_cond}, Ind.Mese: {indice_mese_cond if indice_mese_cond else 'Tutte'}")
        self._log_to_gui(f"  N. Risultati: {num_ris_cond_input}, Min. Tentativi (post-cond): {min_tent_cond_input}")

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            self.ac_metodi_condizionati_dettagli = trova_migliori_metodi_sommativi_condizionati(
                storico_per_analisi_cond,
                ruota_cond_input, pos_cond_input_0based, val_min_cond_input, val_max_cond_input,
                ruota_calc_amb_input, pos_calc_amb_input_0based,
                ruote_gioco_cond, lookahead_cond, indice_mese_cond,
                num_ris_cond_input, min_tent_cond_input, self._log_to_gui
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
                    self._log_to_gui(f"   Previsione Live (se cond. ultima estraz.): {res_cond.get('previsione_live_cond', 'N/A')}")

                self._mostra_popup_risultati_condizionati(
                    self.ac_metodi_condizionati_dettagli,
                    storico_per_analisi_cond,
                    ruote_gioco_cond,
                    "Risultati Analisi Condizionata Base"
                )
            self._log_to_gui("\n--- Analisi Condizionata Avanzata Completata ---")

        except Exception as e:
            self.master.config(cursor=""); messagebox.showerror("Errore Analisi Condizionata", f"Errore: {e}"); self._log_to_gui(f"ERRORE CRITICO ANALISI CONDIZIONATA: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")
   
    def _mostra_popup_risultati_condizionati(self, risultati_da_mostrare, storico_usato, ruote_gioco_usate, titolo_base_popup, info_correttore_globale_str=None):
        if not risultati_da_mostrare:
            messagebox.showinfo(titolo_base_popup, "Nessun risultato da mostrare.")
            return

        lista_previsioni_per_popup = []
        data_riferimento_popup = storico_usato[-1]['data'].strftime('%d/%m/%Y') if storico_usato and storico_usato[-1] else "N/D" 
        grezzi_per_salvataggio_popup = [] 

        for idx, res_cond_original in enumerate(risultati_da_mostrare):
            res_cond = res_cond_original.copy() 

            cond_info = res_cond.get("definizione_cond_primaria") or \
                        res_cond.get("filtro_condizione_primaria_dict") or \
                        res_cond.get("filtro_condizione_primaria_usato") 
            
            met_somm_info = res_cond.get("metodo_sommativo_applicato")
            
            formula_visualizzata_nel_popup = res_cond.get("def_metodo_esteso_1") 
            if formula_visualizzata_nel_popup is None: 
                 formula_visualizzata_nel_popup = res_cond.get("formula_metodo_base_originale")

            desc_metodo_display = "N/D"
            if formula_visualizzata_nel_popup and cond_info:
                desc_formula_interna = "".join(self._format_componente_per_display(c) for c in formula_visualizzata_nel_popup)
                desc_metodo_display = (
                    f"SE {cond_info['ruota']}[pos.{cond_info['posizione']}] IN [{cond_info['val_min']}-{cond_info['val_max']}] "
                    f"ALLORA {desc_formula_interna}"
                )
            elif cond_info and met_somm_info: 
                 desc_metodo_display = (
                    f"SE {cond_info['ruota']}[pos.{cond_info['posizione']}] IN [{cond_info['val_min']}-{cond_info['val_max']}] "
                    f"ALLORA ({met_somm_info['ruota_calcolo']}[pos.{met_somm_info['pos_estratto_calcolo']}] "
                    f"{met_somm_info['operazione']} {met_somm_info['operando_fisso']})"
                 )
            elif formula_visualizzata_nel_popup: 
                 desc_metodo_display = "".join(self._format_componente_per_display(c) for c in formula_visualizzata_nel_popup)

            ambata_per_popup = res_cond.get('ambata_prevista')
            if ambata_per_popup is None: ambata_per_popup = res_cond.get('previsione_live_cond')
            if ambata_per_popup is None and res_cond.get('ambata_risultante_prima_occ_val') not in [None, -1]:
                ambata_per_popup = res_cond.get('ambata_risultante_prima_occ_val')
            if ambata_per_popup is None: ambata_per_popup = "N/A"

            abbinamenti_per_popup = {}
            if str(ambata_per_popup).upper() not in ["N/A", "N/D"] and storico_usato:
                abbinamenti_per_popup = res_cond.get("abbinamenti", {})
                if not abbinamenti_per_popup:
                    abbinamenti_per_popup = analizza_abbinamenti_per_numero_specifico(
                        storico_usato, ambata_per_popup, ruote_gioco_usate, self._log_to_gui
                    )

            performance_str = "N/D"
            if "frequenza_cond" in res_cond:
                performance_str = f"{res_cond['frequenza_cond']:.2%} ({res_cond['successi_cond']}/{res_cond['tentativi_cond']} casi su estraz. filtrate)"
            elif "frequenza" in res_cond:
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
            data_riferimento_previsione_str_comune=data_riferimento_popup,
            metodi_grezzi_per_salvataggio=grezzi_per_salvataggio_popup
        )

    def avvia_ricerca_ambata_ottimale_periodica(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA AMBATA OTTIMALE PERIODICA (CON METODO SOMMATIVO)\n" + "="*50)
        
        if self.ap_risultati_listbox:
            self.ap_risultati_listbox.delete(0, tk.END)

        storico_globale_completo = self._carica_e_valida_storico_comune(usa_filtri_data_globali=False) 
        if not storico_globale_completo:
            self._log_to_gui("Nessun dato storico globale caricato. Analisi interrotta.")
            return

        data_inizio_g = None; data_fine_g = None
        try: data_inizio_g = self.date_inizio_entry_analisi.get_date()
        except ValueError: pass
        try: data_fine_g = self.date_fine_entry_analisi.get_date()
        except ValueError: pass

        mesi_selezionati_gui = [nome for nome, var in self.ap_mesi_vars.items() if var.get()]
        if not mesi_selezionati_gui and not self.ap_tutti_mesi_var.get():
            messagebox.showwarning("Selezione Mesi", "Seleziona almeno un mese o 'Tutti i Mesi' per questa analisi.")
            return
        mesi_map = {nome: i+1 for i, nome in enumerate(list(self.ap_mesi_vars.keys()))}
        mesi_numeri_selezionati = []
        if not self.ap_tutti_mesi_var.get():
            mesi_numeri_selezionati = [mesi_map[nome] for nome in mesi_selezionati_gui]
        
        storico_filtrato_per_periodo = filtra_storico_per_periodo(
            storico_globale_completo, mesi_numeri_selezionati,
            data_inizio_g, data_fine_g, app_logger=self._log_to_gui
        )

        if not storico_filtrato_per_periodo:
            msg = "Nessuna estrazione trovata per il periodo e i filtri temporali selezionati."
            self._log_to_gui(msg); messagebox.showinfo("Analisi Periodica", msg)
            if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, msg)
            self.master.config(cursor="")
            return

        ruote_gioco_sel, lookahead_sel, _ = self._get_parametri_gioco_comuni()
        if ruote_gioco_sel is None: return

        ruota_calc_base_ott = self.ap_ruota_calcolo_ott_var.get()
        pos_estratto_base_ott_0idx = self.ap_pos_estratto_ott_var.get() - 1
        min_tent_applicazioni_soglia = self.min_tentativi_var.get() 

        self._log_to_gui(f"Parametri Ricerca Ambata Ottimale Periodica:")
        log_mesi_str_ott = 'Tutti' if not mesi_numeri_selezionati and self.ap_tutti_mesi_var.get() else (', '.join(mesi_selezionati_gui) if mesi_numeri_selezionati else 'Tutti (implicito)')
        self._log_to_gui(f"  Mesi: {log_mesi_str_ott}")
        data_inizio_g_str = self.date_inizio_entry_analisi.get() or "Inizio Storico"
        data_fine_g_str = self.date_fine_entry_analisi.get() or "Fine Storico"
        self._log_to_gui(f"  Range Date Globale Applicato: Da {data_inizio_g_str} a {data_fine_g_str}")
        self._log_to_gui(f"  Ruota/Pos Base Calcolo: {ruota_calc_base_ott}[{pos_estratto_base_ott_0idx+1}]")
        self._log_to_gui(f"  Ruote di Gioco Verifica: {', '.join(ruote_gioco_sel)}")
        self._log_to_gui(f"  Colpi Lookahead: {lookahead_sel}")
        self._log_to_gui(f"  Min. Applicazioni Metodo nel Periodo: {min_tent_applicazioni_soglia}")


        self.master.config(cursor="watch"); self.master.update_idletasks()

        migliori_metodi_trovati = trova_miglior_ambata_sommativa_periodica(
            storico_globale_completo, storico_filtrato_per_periodo, 
            ruota_calc_base_ott, pos_estratto_base_ott_0idx,
            ruote_gioco_sel, lookahead_sel,
            min_tent_applicazioni_soglia,
            app_logger=self._log_to_gui
        )
        self.master.config(cursor="")

        self._log_to_gui("\n--- RISULTATO RICERCA AMBATA OTTIMALE PERIODICA ---")
        if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, "--- Ambata Ottimale Periodica (Metodo Sommativo) ---")

        if not migliori_metodi_trovati:
            msg = "Nessun metodo/ambata ottimale trovato per i criteri specificati nel periodo."
            self._log_to_gui(msg)
            if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, msg)
            messagebox.showinfo("Analisi Periodica", msg) 
        else:
            lista_previsioni_per_popup_ott = []
            metodi_grezzi_per_salvataggio_ott = []
            data_riferimento_popup_ott = storico_filtrato_per_periodo[-1]['data'].strftime('%d/%m/%Y') if storico_filtrato_per_periodo else "N/D"
            ruote_gioco_str_popup_ott = ", ".join(ruote_gioco_sel)


            for metodo_info in migliori_metodi_trovati: 
                form = metodo_info["metodo_formula"]
                formula_str = f"{form['ruota_calcolo']}[{form['pos_estratto_calcolo']}] {form['operazione']} {form['operando_fisso']}"
                amb_live = metodo_info.get("previsione_live_periodica", "N/A")
                
                riga_lb0 = f"Metodo: {formula_str}"
                riga_lb1 = f"  Ambata Live (da ultima estraz. periodo): {amb_live}"
                riga_lb2 = f"  Copertura Periodi: {metodo_info['copertura_periodi_perc']:.1f}% ({metodo_info['periodi_con_successo']}/{metodo_info['periodi_totali_analizzati']} periodi)"
                riga_lb3 = f"  Perf. Applicazioni: {metodo_info['frequenza_applicazioni']:.1%} ({metodo_info['successi_applicazioni']}/{metodo_info['tentativi_applicazioni']} app.)"
                if self.ap_risultati_listbox:
                    for r in [riga_lb0, riga_lb1, riga_lb2, riga_lb3]: self.ap_risultati_listbox.insert(tk.END, r)

                abbinamenti_periodici_ott = {}
                contorni_ottimali_lista = []
                if str(amb_live).upper() not in ["N/A", "N/D"] and amb_live is not None:
                    abbinamenti_periodici_ott = analizza_abbinamenti_per_numero_specifico(storico_filtrato_per_periodo, amb_live, ruote_gioco_sel, self._log_to_gui)
                    contorni_ottimali_lista = trova_contorni_frequenti_per_ambata_periodica(storico_filtrato_per_periodo, amb_live, 10, ruote_gioco_sel, self._log_to_gui)
                    if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, f"  Abbinamenti per Ambata {amb_live}:")
                    # ... (logica di visualizzazione listbox per abbinamenti e contorni)

                dettaglio_popup_ott = {
                    "titolo_sezione": f"--- METODO OTTIMALE PERIODICO ---",
                    "info_metodo_str": formula_str + f" (Condizione Periodo: Mesi={log_mesi_str_ott})",
                    "ambata_prevista": amb_live,
                    "performance_storica_str": f"Copertura Periodi: {metodo_info['copertura_periodi_perc']:.1f}% ({metodo_info['periodi_con_successo']}/{metodo_info['periodi_totali_analizzati']}) | Perf. App: {metodo_info['frequenza_applicazioni']:.1%}",
                    "abbinamenti_dict": abbinamenti_periodici_ott,
                    "contorni_suggeriti": contorni_ottimali_lista
                }
                lista_previsioni_per_popup_ott.append(dettaglio_popup_ott)
                
                dati_salvataggio_ott = metodo_info.copy() 
                dati_salvataggio_ott["tipo_metodo_salvato"] = "periodica_ottimale"
                dati_salvataggio_ott["formula_testuale"] = dettaglio_popup_ott["info_metodo_str"] 
                dati_salvataggio_ott["ambata_prevista"] = amb_live
                dati_salvataggio_ott["abbinamenti"] = abbinamenti_periodici_ott
                dati_salvataggio_ott["contorni_suggeriti_extra"] = contorni_ottimali_lista
                dati_salvataggio_ott["parametri_periodo"] = {"mesi": mesi_selezionati_gui or "Tutti", "range_date": f"{data_inizio_g_str} a {data_fine_g_str}"}
                metodi_grezzi_per_salvataggio_ott.append(dati_salvataggio_ott)

                if self.ap_risultati_listbox: self.ap_risultati_listbox.insert(tk.END, "-"*30)
            
            if lista_previsioni_per_popup_ott:
                self.mostra_popup_previsione(
                    titolo_popup="Risultato Ricerca Ambata Ottimale Periodica",
                    ruote_gioco_str=ruote_gioco_str_popup_ott,
                    lista_previsioni_dettagliate=lista_previsioni_per_popup_ott,
                    data_riferimento_previsione_str_comune=data_riferimento_popup_ott,
                    metodi_grezzi_per_salvataggio=metodi_grezzi_per_salvataggio_ott
                )

        self._log_to_gui("--- Ricerca Ambata Ottimale Periodica Completata ---")

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
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return

        min_tentativi_correttore_cfg = self.corr_cfg_min_tentativi.get()
        c_fisso_s = self.corr_cfg_cerca_fisso_semplice.get(); c_estr_s = self.corr_cfg_cerca_estratto_semplice.get()
        c_somma_ef = self.corr_cfg_cerca_somma_estr_fisso.get(); c_somma_ee = self.corr_cfg_cerca_somma_estr_estr.get()
        c_diff_ef = self.corr_cfg_cerca_diff_estr_fisso.get(); c_diff_ee = self.corr_cfg_cerca_diff_estr_estr.get()
        c_mult_ef = self.corr_cfg_cerca_mult_estr_fisso.get(); c_mult_ee = self.corr_cfg_cerca_mult_estr_estr.get()

        self._log_to_gui(f"Applicazione correttore su Metodo Condizionato Base: {self.ac_risultati_listbox.get(selected_index)}")
        self._log_to_gui(f"  Condizione Primaria (filtro): {definizione_cond_primaria_filtro}")
        self._log_to_gui(f"  Formula Base per Correttore: {''.join(self._format_componente_per_display(c) for c in formula_base_per_correttore)}")

        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati_correttori_list_cond = trova_miglior_correttore_per_metodo_complesso(
                storico_per_analisi,
                formula_base_per_correttore, None,
                c_fisso_s, c_estr_s, c_diff_ef, c_diff_ee, c_mult_ef, c_mult_ee, c_somma_ef, c_somma_ee,
                ruote_gioco, lookahead, indice_mese,
                min_tentativi_correttore_cfg,
                app_logger=self._log_to_gui,
                filtro_condizione_primaria_dict=definizione_cond_primaria_filtro
            )
            
            if not risultati_correttori_list_cond:
                messagebox.showinfo("Ricerca Correttore (Cond.)", "Nessun correttore valido trovato che migliori il benchmark.")
                self._log_to_gui("Nessun correttore valido trovato (post-condizione).")
                if hasattr(self, 'btn_backtest_cond_corretto'):
                    self.btn_backtest_cond_corretto.config(state=tk.DISABLED)
                ### AGGIUNTA ###
                # Assicurati che le variabili siano resettate anche in questo caso
                self.ultimo_metodo_cond_corretto_definizione = None
                self.ultimo_metodo_cond_corretto_formula_testuale = ""
                ### FINE AGGIUNTA ###

            else:
                miglior_correttore_cond = risultati_correttori_list_cond[0]
                self._log_to_gui(f"DEBUG CORR_COND: Dati del miglior correttore trovato: {miglior_correttore_cond}")
                
                info_correttore_globale_str_popup = (
                    f"Correttore Applicato: {miglior_correttore_cond['tipo_correttore_descrittivo']} -> {miglior_correttore_cond['dettaglio_correttore_str']}\n"
                    f"Operazione di Collegamento Base: '{miglior_correttore_cond['operazione_collegamento_base']}'\n"
                    f"Performance del Metodo Condizionato + Correttore: {miglior_correttore_cond['frequenza']:.2%} ({miglior_correttore_cond['successi']}/{miglior_correttore_cond['tentativi']} casi)"
                )
                self._log_to_gui(f"\n--- MIGLIOR CORRETTORE PER METODO CONDIZIONATO ---")
                self._log_to_gui(f"  {info_correttore_globale_str_popup.replace(chr(10), chr(10)+'  ')}")
                
                metodo_esteso_corretto_def = miglior_correttore_cond.get('def_metodo_esteso_1')
                self._log_to_gui(f"DEBUG CORR_COND: 'def_metodo_esteso_1' estratta: {metodo_esteso_corretto_def}")
                self._log_to_gui(f"DEBUG CORR_COND: 'definizione_cond_primaria_filtro' (usata per l'analisi): {definizione_cond_primaria_filtro}")

                if metodo_esteso_corretto_def and definizione_cond_primaria_filtro:
                    self._log_to_gui("DEBUG CORR_COND: Entrato nel blocco if per salvare dati e abilitare il pulsante.")
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
                        self._log_to_gui("DEBUG CORR_COND: Pulsante btn_backtest_cond_corretto ABILITATO.")
                    else:
                        self._log_to_gui("WARN CORR_COND: Attributo btn_backtest_cond_corretto non trovato per abilitazione.")

                    ambata_live_corr_cond, abb_live_corr_cond = self._calcola_previsione_e_abbinamenti_metodo_complesso_con_cond(
                        storico_per_analisi, metodo_esteso_corretto_def, definizione_cond_primaria_filtro,
                        ruote_gioco, storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else "N/D",
                        "Metodo Condizionato + Correttore"
                    )
                    dati_per_popup_corretto_completi = {
                        "tipo_metodo_salvato": "condizionato_corretto", 
                        "formula_testuale": self.ultimo_metodo_cond_corretto_formula_testuale, 
                        "def_metodo_esteso_1": metodo_esteso_corretto_def, 
                        "definizione_cond_primaria": definizione_cond_primaria_filtro, 
                        "ambata_prevista": ambata_live_corr_cond, "abbinamenti": abb_live_corr_cond,
                        "successi": miglior_correttore_cond['successi'], "tentativi": miglior_correttore_cond['tentativi'],
                        "frequenza": miglior_correttore_cond['frequenza'],
                        "tipo_correttore_descrittivo": miglior_correttore_cond['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_correttore_cond['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_correttore_cond['operazione_collegamento_base'],
                    }
                    self._mostra_popup_risultati_condizionati(
                        [dati_per_popup_corretto_completi], 
                        storico_per_analisi, ruote_gioco,
                        "Previsione Metodo Condizionato con Correttore", info_correttore_globale_str_popup
                    )
                else:
                    self._log_to_gui(f"WARN CORR_COND: Pulsante backtest corretto NON abilitato. "
                                     f"metodo_esteso_corretto_def è {'valido' if metodo_esteso_corretto_def else 'NON valido/None'}. "
                                     f"definizione_cond_primaria_filtro è {'valida' if definizione_cond_primaria_filtro else 'NON valida/None'}.")
                    if hasattr(self, 'btn_backtest_cond_corretto'):
                        self.btn_backtest_cond_corretto.config(state=tk.DISABLED)
                    ### AGGIUNTA ###
                    self.ultimo_metodo_cond_corretto_definizione = None
                    self.ultimo_metodo_cond_corretto_formula_testuale = ""
                    ### FINE AGGIUNTA ###
                    messagebox.showinfo("Risultato Correttore", "Miglior correttore trovato ma dati interni incompleti per preparare il backtest.")
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

    def _calcola_previsione_e_abbinamenti_metodo_complesso_con_cond(
        self, storico_attuale, definizione_metodo_completo,
        condizione_primaria_dict,
        ruote_gioco, data_riferimento_str, nome_metodo_log="Metodo"
    ):
        ambata_live = None; abbinamenti_live = {}; note_previsione_log = ""
        if not storico_attuale:
            note_previsione_log = f"Storico vuoto per {nome_metodo_log}."
            self._log_to_gui(f"\n  PREVISIONE LIVE {nome_metodo_log} (da estrazione del {data_riferimento_str or 'N/D'}):")
            self._log_to_gui(f"    {note_previsione_log}")
            return ambata_live, abbinamenti_live
        ultima_estrazione = storico_attuale[-1]; cond_soddisfatta = False
        if condizione_primaria_dict:
            cond_ruota = condizione_primaria_dict['ruota']
            cond_pos_idx = (condizione_primaria_dict['posizione'] - 1) if condizione_primaria_dict.get('posizione',0) > 0 else 0
            cond_min = condizione_primaria_dict['val_min']; cond_max = condizione_primaria_dict['val_max']
            numeri_ruota_cond_ultima = ultima_estrazione.get(cond_ruota, [])
            if numeri_ruota_cond_ultima and len(numeri_ruota_cond_ultima) > cond_pos_idx:
                val_cond_ultima = numeri_ruota_cond_ultima[cond_pos_idx]
                if cond_min <= val_cond_ultima <= cond_max: cond_soddisfatta = True
                else: note_previsione_log = f"{nome_metodo_log}: Cond. primaria non soddisfatta dall'ultima estraz. ({val_cond_ultima} non in [{cond_min}-{cond_max}] su {cond_ruota})."
            else: note_previsione_log = f"{nome_metodo_log}: Impossibile verif. cond. primaria su ultima estraz. (dati mancanti per {cond_ruota})."
        else: cond_soddisfatta = True
        if cond_soddisfatta:
            val_raw = calcola_valore_metodo_complesso(ultima_estrazione, definizione_metodo_completo, self._log_to_gui)
            if val_raw is not None: ambata_live = regola_fuori_90(val_raw)
            else: note_previsione_log = f"{nome_metodo_log} (con cond. soddisfatta) non applicabile all'ultima estrazione (es. div/0)."
        self._log_to_gui(f"\n  PREVISIONE LIVE {nome_metodo_log} (da estrazione del {data_riferimento_str or 'N/D'}):")
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
        else: self._log_to_gui(f"    {note_previsione_log if note_previsione_log else 'Impossibile calcolare previsione.'}")
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

    def mostra_popup_previsione(self, titolo_popup, ruote_gioco_str, lista_previsioni_dettagliate=None, copertura_combinata_info=None, data_riferimento_previsione_str_comune=None, metodi_grezzi_per_salvataggio=None ):
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo_popup)

        popup_width = 700
        popup_base_height_per_method_section = 240
        abbinamenti_h_approx = 150
        contorni_h_approx = 70

        dynamic_height_needed = 150
        if copertura_combinata_info: dynamic_height_needed += 80

        if lista_previsioni_dettagliate:
            for prev_dett_c in lista_previsioni_dettagliate:
                current_met_h = popup_base_height_per_method_section
                ambata_val_check = prev_dett_c.get('ambata_prevista')
                is_single_number_for_abbinamenti = False
                if isinstance(ambata_val_check, (int, float)):
                    is_single_number_for_abbinamenti = True
                elif isinstance(ambata_val_check, str) and ambata_val_check.isdigit():
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

        self._log_to_gui(f"DEBUG POPUP (mostra_popup_previsione): Titolo: {titolo_popup}")

        row_idx = 0
        ttk.Label(scrollable_frame, text=f"--- {titolo_popup} ---", font=("Helvetica", 12, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1
        if data_riferimento_previsione_str_comune: ttk.Label(scrollable_frame, text=f"Previsione del: {data_riferimento_previsione_str_comune}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
        ttk.Label(scrollable_frame, text=f"Su ruote: {ruote_gioco_str}").grid(row=row_idx, column=0, columnspan=2, pady=(2,10), sticky="w"); row_idx += 1

        if copertura_combinata_info and "testo_introduttivo" in copertura_combinata_info:
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5); row_idx += 1
            ttk.Label(scrollable_frame, text=copertura_combinata_info['testo_introduttivo'], wraplength=popup_width - 40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w"); row_idx += 1

        if lista_previsioni_dettagliate:
            for idx_metodo, previsione_dett in enumerate(lista_previsioni_dettagliate):
                self._log_to_gui(f"DEBUG POPUP (mostra_popup): Processando previsione_dett #{idx_metodo}: {previsione_dett.get('titolo_sezione', 'N/A')}")
                ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10); row_idx += 1
                titolo_sezione = previsione_dett.get('titolo_sezione', '--- PREVISIONE ---'); ttk.Label(scrollable_frame, text=titolo_sezione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=3, sticky="w"); row_idx += 1
                formula_metodo_display = previsione_dett.get('info_metodo_str', "N/D")
                if formula_metodo_display != "N/D": ttk.Label(scrollable_frame, text=f"Metodo: {formula_metodo_display}", wraplength=popup_width-40, justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1

                ambata_loop = previsione_dett.get('ambata_prevista')
                self._log_to_gui(f"DEBUG POPUP (mostra_popup): ambata_loop (o previsione) per sezione '{titolo_sezione}' = {ambata_loop} (tipo: {type(ambata_loop)})")

                if ambata_loop is None or str(ambata_loop).upper() in ["N/D", "N/A"]:
                    ttk.Label(scrollable_frame, text="Nessuna previsione valida.").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1
                else:
                    testo_previsione_popup = f"PREVISIONE DA GIOCARE: {ambata_loop}"
                    ttk.Label(scrollable_frame, text=testo_previsione_popup, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1

                performance_str_display = previsione_dett.get('performance_storica_str', 'N/D')
                ttk.Label(scrollable_frame, text=f"Performance storica:\n{performance_str_display}", justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w"); row_idx += 1

                dati_grezzi_per_questo_metodo = None
                if metodi_grezzi_per_salvataggio and idx_metodo < len(metodi_grezzi_per_salvataggio):
                    dati_grezzi_per_questo_metodo = metodi_grezzi_per_salvataggio[idx_metodo]

                if dati_grezzi_per_questo_metodo:
                    estensione_default = ".lmp"
                    tipo_metodo_salv = dati_grezzi_per_questo_metodo.get("tipo_metodo_salvato", "sconosciuto")
                    if tipo_metodo_salv.startswith("condizionato"):
                        estensione_default = ".lmcondcorr" if "corretto" in tipo_metodo_salv else ".lmcond"
                    elif tipo_metodo_salv == "ambata_ambo_unico_auto":
                        estensione_default = ".lmaau"

                    btn_salva_profilo = ttk.Button(scrollable_frame, text="Salva Questo Metodo",
                                                   command=lambda d=dati_grezzi_per_questo_metodo.copy(), e=estensione_default: self._prepara_e_salva_profilo_metodo(d, estensione=e))
                    btn_salva_profilo.grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="ew"); row_idx += 1

                ambata_per_abbinamenti_popup = None
                if isinstance(ambata_loop, (int, float)):
                    ambata_per_abbinamenti_popup = ambata_loop
                elif isinstance(ambata_loop, str) and ambata_loop.isdigit():
                    ambata_per_abbinamenti_popup = int(ambata_loop)

                self._log_to_gui(f"DEBUG POPUP (mostra_popup) Sezione Abbinamenti: ambata_per_abbinamenti_popup='{ambata_per_abbinamenti_popup}', tipo={type(ambata_per_abbinamenti_popup)}, previsione_dett['abbinamenti_dict'] esiste? {'abbinamenti_dict' in previsione_dett}")

                if ambata_per_abbinamenti_popup is not None:
                    ttk.Label(scrollable_frame, text="Abbinamenti Consigliati (co-occorrenze storiche):").grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="w"); row_idx +=1
                    abbinamenti_dict_loop = previsione_dett.get('abbinamenti_dict', {});
                    eventi_totali_loop = abbinamenti_dict_loop.get("sortite_ambata_target", 0)
                    self._log_to_gui(f"DEBUG POPUP (mostra_popup): eventi_totali_loop (sortite ambata target per abbinamenti) = {eventi_totali_loop}")

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
                        ttk.Label(scrollable_frame, text=f"  Nessuna co-occorrenza storica per l'ambata {ambata_per_abbinamenti_popup} (eventi_totali_loop={eventi_totali_loop}).").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1

                    contorni_suggeriti_loop = previsione_dett.get('contorni_suggeriti', [])
                    if contorni_suggeriti_loop:
                        ttk.Label(scrollable_frame, text="  Altri Contorni Frequenti:").grid(row=row_idx, column=0, columnspan=2, pady=(3,1), sticky="w"); row_idx+=1
                        for contorno_num, contorno_cnt in contorni_suggeriti_loop[:5]:
                            ttk.Label(scrollable_frame, text=f"    - Numero: {contorno_num} (Presenze con ambata: {contorno_cnt})").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx+=1

                else:
                    self._log_to_gui(f"DEBUG POPUP (mostra_popup): Nessun abbinamento mostrato perché ambata_loop ('{ambata_loop}') non è un singolo numero.")

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
        self._log_to_gui(f"DEBUG: _prepara_metodo_per_backtest CHIAMATO con dati: {dati_metodo_selezionato_per_prep}")

        tipo_metodo = dati_metodo_selezionato_per_prep.get('tipo') if dati_metodo_selezionato_per_prep else None
        formula_ok = bool(dati_metodo_selezionato_per_prep.get('formula_testuale')) if dati_metodo_selezionato_per_prep else False

        def_strutturata_presente = 'definizione_strutturata' in dati_metodo_selezionato_per_prep if dati_metodo_selezionato_per_prep else False
        condizione_speciale_tipo = False
        if tipo_metodo:
            condizione_speciale_tipo = tipo_metodo.startswith("periodica_") or \
                                       (tipo_metodo.startswith("condizionato_") and 'condizione_primaria' in dati_metodo_selezionato_per_prep)

        condizione_valida_per_salvataggio = tipo_metodo and formula_ok and (def_strutturata_presente or condizione_speciale_tipo)

        if condizione_valida_per_salvataggio:
            self.metodo_preparato_per_backtest = dati_metodo_selezionato_per_prep.copy()

            formula_display = self.metodo_preparato_per_backtest['formula_testuale']
            tipo_display = self.metodo_preparato_per_backtest['tipo'].replace("_", " ").title()

            if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
                self.mc_listbox_componenti_1.delete(0, tk.END)
                self.mc_listbox_componenti_1.insert(tk.END, f"PER BACKTEST ({tipo_display}):")
                self.mc_listbox_componenti_1.insert(tk.END, formula_display)

            messagebox.showinfo("Metodo Pronto per Backtest",
                                f"Metodo ({tipo_display}):\n{formula_display}\n"
                                "è stato selezionato.\n\n"
                                "Ora puoi usare il pulsante 'Backtest Dettagliato'.")
            self._log_to_gui(f"INFO: Metodo selezionato da popup per backtest dettagliato ({tipo_display}): {formula_display}")
            self._log_to_gui(f"DEBUG: Dati completi del metodo preparato in self.metodo_preparato_per_backtest: {self.metodo_preparato_per_backtest}")

            if hasattr(self, 'usa_ultimo_corretto_per_backtest_var'):
                self.usa_ultimo_corretto_per_backtest_var.set(False)
                self._log_to_gui("INFO: Checkbox 'Usa ultimo metodo corretto' deselezionato perché un metodo è stato preparato da popup.")
        else:
            messaggio_errore_dett = "Dati del metodo selezionato non validi o definizione per backtest mancante."
            if dati_metodo_selezionato_per_prep:
                messaggio_errore_dett += f"\nDati ricevuti: Tipo='{tipo_metodo}', FormulaOK={formula_ok}, " \
                                         f"DefStrutturataPresente={def_strutturata_presente}, CondSpecialeTipoOK={condizione_speciale_tipo}"
            messagebox.showerror("Errore Preparazione Backtest", messaggio_errore_dett)
            self._log_to_gui(f"WARN: _prepara_metodo_per_backtest chiamato con dati metodo non validi. {messaggio_errore_dett}")
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

# --- BLOCCO PRINCIPALE DI ESECUZIONE ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LottoAnalyzerApp(root)
    root.mainloop()