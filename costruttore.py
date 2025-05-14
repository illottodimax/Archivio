
import os
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
from itertools import combinations
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Listbox
import traceback # Aggiunto per un logging più dettagliato degli errori
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

# --- FUNZIONI LOGICHE --- (Queste rimangono fuori dalla classe)
def regola_fuori_90(numero):
    if numero is None:
        return None
    if numero == 0:
        return 90
    while numero <= 0:
        numero += 90
    while numero > 90:
        numero -= 90
    return numero

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
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro:
            continue

        ambate_previste_per_questa_estrazione = set()
        almeno_un_metodo_applicabile_qui = False

        for metodo_info_dict in top_metodi_info:
            met_details = metodo_info_dict['metodo']
            ruota_calc = met_details['ruota_calcolo']
            pos_estr = met_details['pos_estratto_calcolo']

            numeri_ruota_corrente = estrazione_corrente.get(ruota_calc, [])
            if not numeri_ruota_corrente or len(numeri_ruota_corrente) <= pos_estr:
                continue

            almeno_un_metodo_applicabile_qui = True
            numero_base = numeri_ruota_corrente[pos_estr]
            op_str = met_details['operazione']
            operando = met_details['operando_fisso']

            if op_str not in op_func_cache:
                if app_logger: app_logger(f"WARN: Operazione {op_str} non in cache per analisi combinata.");
                continue
            op_func = op_func_cache[op_str]

            try:
                valore_operazione = op_func(numero_base, operando)
            except ZeroDivisionError:
                continue

            ambata_prevista = regola_fuori_90(valore_operazione)
            if ambata_prevista is not None:
                ambate_previste_per_questa_estrazione.add(ambata_prevista)

        if almeno_un_metodo_applicabile_qui:
            date_tentativi_combinati.add(estrazione_corrente['data'])
            if ambate_previste_per_questa_estrazione:
                successo_per_questa_data_applicazione = False
                for k_lookahead in range(1, lookahead + 1):
                    if i + k_lookahead >= len(storico):
                        break
                    estrazione_futura = storico[i + k_lookahead]
                    for ambata_p in ambate_previste_per_questa_estrazione:
                        for ruota_verifica in ruote_gioco_selezionate:
                            if ambata_p in estrazione_futura.get(ruota_verifica, []):
                                date_successi_combinati.add(estrazione_corrente['data'])
                                successo_per_questa_data_applicazione = True
                                break
                        if successo_per_questa_data_applicazione:
                            break
                    if successo_per_questa_data_applicazione:
                        break

    num_successi_combinati = len(date_successi_combinati)
    num_tentativi_combinati = len(date_tentativi_combinati)
    frequenza_combinata = num_successi_combinati / num_tentativi_combinati if num_tentativi_combinati > 0 else 0
    return num_successi_combinati, num_tentativi_combinati, frequenza_combinata

def analizza_abbinamenti_per_numero_specifico(storico, ambata_target, ruote_gioco, app_logger=None):
    if ambata_target is None:
        if app_logger: app_logger("WARN: ambata_target è None in analizza_abbinamenti_per_numero_specifico.")
        return {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "sortite_ambata_target": 0}

    abbinamenti_per_ambo = Counter()
    abbinamenti_per_terno = Counter()
    abbinamenti_per_quaterna = Counter()
    abbinamenti_per_cinquina = Counter()
    sortite_ambata_target = 0

    for estrazione in storico:
        uscita_in_estrazione_su_ruote_gioco = False
        numeri_ruota_con_vincita_per_abbinamento = []

        for ruota in ruote_gioco:
            numeri_estratti_ruota = estrazione.get(ruota, [])
            if ambata_target in numeri_estratti_ruota:
                if not uscita_in_estrazione_su_ruote_gioco:
                    sortite_ambata_target += 1
                    uscita_in_estrazione_su_ruote_gioco = True
                if not numeri_ruota_con_vincita_per_abbinamento: 
                    numeri_ruota_con_vincita_per_abbinamento = numeri_estratti_ruota
                break 

        if uscita_in_estrazione_su_ruote_gioco and numeri_ruota_con_vincita_per_abbinamento:
            altri_numeri_con_target = [n for n in numeri_ruota_con_vincita_per_abbinamento if n != ambata_target]

            for num_abbinato in altri_numeri_con_target:
                abbinamenti_per_ambo[num_abbinato] += 1

            if len(altri_numeri_con_target) >= 2:
                for combo_2 in combinations(sorted(altri_numeri_con_target), 2):
                    abbinamenti_per_terno[combo_2] += 1
            if len(altri_numeri_con_target) >= 3:
                for combo_3 in combinations(sorted(altri_numeri_con_target), 3):
                    abbinamenti_per_quaterna[combo_3] += 1
            if len(altri_numeri_con_target) >= 4:
                for combo_4 in combinations(sorted(altri_numeri_con_target), 4):
                    abbinamenti_per_cinquina[combo_4] += 1
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
                    "successi": successi,
                    "tentativi": tentativi,
                    "frequenza_ambata": frequenza,
                    "applicazioni_vincenti_dettagliate": applicazioni_vincenti_dett
                })
    log_message("  Completata analisi performance storica dei metodi.")
    risultati_ambate_grezzi.sort(key=lambda x: (x["frequenza_ambata"], x["successi"]), reverse=True)

    info_copertura_combinata_da_restituire = None
    top_n_metodi_per_analisi_combinata = risultati_ambate_grezzi[:max_ambate_output]

    if len(top_n_metodi_per_analisi_combinata) > 1:
        log_message(f"\n--- ANALISI COPERTURA COMBINATA PER TOP {len(top_n_metodi_per_analisi_combinata)} METODI ---")
        s_comb, t_comb, f_comb = analizza_copertura_combinata(
            storico, top_n_metodi_per_analisi_combinata,
            ruote_gioco_selezionate, lookahead, indice_mese_filtro, app_logger
        )
        if t_comb > 0:
            log_message(f"  Giocando simultaneamente le ambate prodotte dai {len(top_n_metodi_per_analisi_combinata)} migliori metodi:")
            log_message(f"  - Successi Complessivi (almeno un'ambata vincente): {s_comb}")
            log_message(f"  - Tentativi Complessivi (almeno un metodo applicabile): {t_comb}")
            log_message(f"  - Frequenza di Copertura Combinata: {f_comb:.2%}")
            info_copertura_combinata_da_restituire = {
                "successi": s_comb, "tentativi": t_comb, "frequenza": f_comb,
                "num_metodi_combinati": len(top_n_metodi_per_analisi_combinata)
            }
        else:
            log_message("  Nessun tentativo combinato applicabile per i metodi selezionati per l'analisi combinata.")

    risultati_finali_output = []
    if not storico:
        log_message("ERRORE: Storico vuoto, impossibile calcolare previsioni.")
        return [], None

    ultima_estrazione_disponibile = storico[-1]
    log_message(f"\n--- DETTAGLIO, PREVISIONE E ABBINAMENTI PER I TOP {min(max_ambate_output, len(risultati_ambate_grezzi))} METODI ---")

    for i, res_grezza in enumerate(risultati_ambate_grezzi[:max_ambate_output]):
        metodo_def = res_grezza['metodo']
        op_str_metodo = metodo_def['operazione']
        operando_metodo = metodo_def['operando_fisso']
        rc_metodo = metodo_def['ruota_calcolo']
        pe_metodo = metodo_def['pos_estratto_calcolo']

        log_message(f"\n--- {i+1}° METODO ---")
        log_message(f"  Formula: {rc_metodo}[pos.{pe_metodo+1}] {op_str_metodo} {operando_metodo}")
        log_message(f"  Performance Storica: {res_grezza['frequenza_ambata']:.2%} ({res_grezza['successi']}/{res_grezza['tentativi']} casi)")

        ambata_previsione_attuale = None
        note_previsione = ""
        op_func_metodo = OPERAZIONI.get(op_str_metodo)

        numeri_ultima_ruota_calcolo = ultima_estrazione_disponibile.get(rc_metodo, [])
        if op_func_metodo and numeri_ultima_ruota_calcolo and len(numeri_ultima_ruota_calcolo) > pe_metodo:
            numero_base_ultima = numeri_ultima_ruota_calcolo[pe_metodo]
            try:
                valore_op_ultima = op_func_metodo(numero_base_ultima, operando_metodo)
                ambata_previsione_attuale = regola_fuori_90(valore_op_ultima)
            except ZeroDivisionError:
                note_previsione = "Metodo non applicabile all'ultima estrazione (divisione per zero)."
        else:
            note_previsione = "Dati insufficienti nell'ultima estrazione per calcolare la previsione."

        log_message(f"  PREVISIONE DA ULTIMA ESTRAZIONE ({ultima_estrazione_disponibile['data'].strftime('%d/%m/%Y')}):")
        if ambata_previsione_attuale is not None:
            log_message(f"    AMBATA DA GIOCARE: {ambata_previsione_attuale}")
        else:
            log_message(f"    {note_previsione if note_previsione else 'Impossibile calcolare previsione.'}")

        abbinamenti_calcolati_finali = {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "sortite_ambata_target": 0}
        if ambata_previsione_attuale is not None:
            log_message(f"    Migliori Abbinamenti Storici (co-occorrenze con AMBATA DA GIOCARE: {ambata_previsione_attuale}):")
            abbinamenti_calcolati_finali = analizza_abbinamenti_per_numero_specifico(
                storico, ambata_previsione_attuale, ruote_gioco_selezionate, app_logger
            )
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
            else:
                log_message(f"      Nessuna co-occorrenza trovata per l'ambata {ambata_previsione_attuale} nello storico sulle ruote selezionate.")
        elif res_grezza["applicazioni_vincenti_dettagliate"]:
             log_message("    Nessuna ambata attuale calcolata. Abbinamenti basati su co-occorrenze non disponibili.")
        else:
            log_message("    Nessuna ambata attuale calcolata e nessuna applicazione vincente storica per analisi abbinamenti.")

        risultati_finali_output.append({
            "metodo": metodo_def, # Questa è la "formula" per metodi semplici
            "ambata_piu_frequente_dal_metodo": ambata_previsione_attuale if ambata_previsione_attuale is not None else "N/D",
            "frequenza_ambata": res_grezza['frequenza_ambata'],
            "successi": res_grezza['successi'],
            "tentativi": res_grezza['tentativi'],
            "abbinamenti": abbinamenti_calcolati_finali,
            "applicazioni_vincenti_dettagliate": res_grezza["applicazioni_vincenti_dettagliate"]
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
                                         indice_mese_filtro, app_logger=None):
    successi = 0; tentativi = 0; applicazioni_vincenti = []
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
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
                                                app_logger=None):
    if not lista_definizioni_metodi_estesi:
        return 0, 0, 0.0

    date_tentativi_combinati = set()
    date_successi_combinati = set()

    for i in range(len(storico) - lookahead):
        estrazione_applicazione = storico[i]
        if indice_mese_filtro and estrazione_applicazione['indice_mese'] != indice_mese_filtro:
            continue

        ambate_previste_per_questa_applicazione = set()
        almeno_un_metodo_applicabile_qui = False

        for def_metodo_esteso in lista_definizioni_metodi_estesi:
            if not def_metodo_esteso: continue 

            valore_calcolato_raw = calcola_valore_metodo_complesso(estrazione_applicazione, def_metodo_esteso, app_logger)
            if valore_calcolato_raw is not None:
                almeno_un_metodo_applicabile_qui = True
                ambata_prev = regola_fuori_90(valore_calcolato_raw)
                if ambata_prev is not None:
                    ambate_previste_per_questa_applicazione.add(ambata_prev)

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
                                successo_per_questa_data_app = True
                                break 
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
    risultati_popup_str += f"Periodo: Dal {data_inizio_controllo} per {num_colpi_controllo} colpi\n"
    risultati_popup_str += "-" * 40 + "\n"

    log_message_detailed(f"\n--- VERIFICA GIOCATA MANUALE (Log Dettagliato) ---") 
    log_message_detailed(f"Numeri da giocare: {numeri_da_giocare}")
    # ... (resto della funzione invariato) ...
    if not numeri_da_giocare:
        msg_err = "ERRORE: Nessun numero inserito per la verifica."
        log_message_detailed(msg_err)
        return msg_err 
    if not ruote_selezionate:
        msg_err = "ERRORE: Nessuna ruota selezionata per la verifica."
        log_message_detailed(msg_err)
        return msg_err
    if not data_inizio_controllo:
        msg_err = "ERRORE: Data inizio controllo non specificata."
        log_message_detailed(msg_err)
        return msg_err

    indice_partenza = -1
    for i, estrazione in enumerate(storico_completo):
        if estrazione['data'] >= data_inizio_controllo:
            indice_partenza = i
            break

    if indice_partenza == -1:
        msg_err = f"Nessuna estrazione trovata a partire dal {data_inizio_controllo}. Impossibile verificare."
        log_message_detailed(msg_err)
        risultati_popup_str += msg_err + "\n"
        return risultati_popup_str

    log_message_detailed(f"Controllo a partire dall'estrazione del {storico_completo[indice_partenza]['data']}:")
    trovato_esito_globale_per_popup = False

    for colpo in range(num_colpi_controllo):
        indice_estrazione_corrente = indice_partenza + colpo
        if indice_estrazione_corrente >= len(storico_completo):
            msg_fine_storico = f"Fine storico raggiunto al colpo {colpo+1} (su {num_colpi_controllo}). Controllo interrotto."
            log_message_detailed(msg_fine_storico)
            risultati_popup_str += msg_fine_storico + "\n"
            break

        estrazione_controllo = storico_completo[indice_estrazione_corrente]
        data_estrazione_str = estrazione_controllo['data'].strftime('%d/%m/%Y')
        log_message_detailed(f"  Colpo {colpo + 1} (Data: {data_estrazione_str}):")
        
        esito_trovato_in_questo_colpo_su_ruota_secca = False
        esiti_colpo_per_popup = [] 

        for ruota in ruote_selezionate:
            numeri_estratti_ruota = estrazione_controllo.get(ruota, [])
            if not numeri_estratti_ruota:
                continue

            vincenti_ambata = [n for n in numeri_da_giocare if n in numeri_estratti_ruota]
            if vincenti_ambata:
                msg_ambata = f"    >> AMBATA SU {ruota.upper()}! Numeri usciti: {vincenti_ambata}"
                log_message_detailed(msg_ambata)
                esiti_colpo_per_popup.append(f"AMBATA su {ruota.upper()}: {vincenti_ambata}")
                trovato_esito_globale_per_popup = True
                if len(ruote_selezionate) == 1:
                    esito_trovato_in_questo_colpo_su_ruota_secca = True

            if len(numeri_da_giocare) >= 2:
                numeri_giocati_presenti_nella_ruota = [n for n in numeri_da_giocare if n in numeri_estratti_ruota]
                num_corrispondenze = len(numeri_giocati_presenti_nella_ruota)
                sorte_trovata_in_questo_passaggio = False
                sorte_msg_log = ""
                sorte_msg_popup = ""

                if num_corrispondenze == 2:
                    sorte_msg_log = f"    >> AMBO SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_msg_popup = f"AMBO su {ruota.upper()}: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_trovata_in_questo_passaggio = True
                elif num_corrispondenze == 3:
                    sorte_msg_log = f"    >> TERNO SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_msg_popup = f"TERNO su {ruota.upper()}: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_trovata_in_questo_passaggio = True
                elif num_corrispondenze == 4:
                    sorte_msg_log = f"    >> QUATERNA SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_msg_popup = f"QUATERNA su {ruota.upper()}: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_trovata_in_questo_passaggio = True
                elif num_corrispondenze >= 5: 
                    sorte_msg_log = f"    >> CINQUINA (o sup.) SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_msg_popup = f"CINQUINA (o sup.) su {ruota.upper()}: {sorted(numeri_giocati_presenti_nella_ruota)}"
                    sorte_trovata_in_questo_passaggio = True
                
                if sorte_trovata_in_questo_passaggio:
                    log_message_detailed(sorte_msg_log)
                    esiti_colpo_per_popup.append(sorte_msg_popup)
                    trovato_esito_globale_per_popup = True
                    if len(ruote_selezionate) == 1:
                        esito_trovato_in_questo_colpo_su_ruota_secca = True
        
        if esiti_colpo_per_popup:
            risultati_popup_str += f"Colpo {colpo + 1} ({data_estrazione_str}):\n"
            for esito_p in esiti_colpo_per_popup:
                risultati_popup_str += f"  - {esito_p}\n"
            risultati_popup_str += "\n"

        if esito_trovato_in_questo_colpo_su_ruota_secca:
            msg_fine_secca = f"--- Esito trovato su ruota secca al colpo {colpo + 1}. Interruzione verifica colpi successivi. ---"
            log_message_detailed(msg_fine_secca)
            break
            
    if not trovato_esito_globale_per_popup:
        msg_nessun_esito = f"\nNessun esito trovato per i numeri {numeri_da_giocare} entro {num_colpi_controllo} colpi."
        log_message_detailed(msg_nessun_esito)
        risultati_popup_str += msg_nessun_esito.strip() + "\n" 
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
    ruote_gioco_selezionate,
    lookahead,
    indice_mese_filtro,
    min_tentativi_per_correttore,
    app_logger=None
):
    # ... (corpo della funzione invariato) ...
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=flush)

    log_message(f"\nInizio ricerca correttore. Min.Tentativi: {min_tentativi_per_correttore}")
    freq_benchmark = 0.0; successi_benchmark = 0; tentativi_benchmark = 0
    metodi_base_attivi = []
    if definizione_metodo_base_1 and definizione_metodo_base_1[-1].get('operazione_successiva') == '=':
        metodi_base_attivi.append(definizione_metodo_base_1)
    if definizione_metodo_base_2 and definizione_metodo_base_2[-1].get('operazione_successiva') == '=':
        metodi_base_attivi.append(definizione_metodo_base_2)

    if not metodi_base_attivi:
        log_message("Nessun metodo base VALIDO fornito per la ricerca correttore."); return []

    if len(metodi_base_attivi) == 1:
        s_base, t_base, _ = analizza_metodo_complesso_specifico(storico, metodi_base_attivi[0], ruote_gioco_selezionate, lookahead, indice_mese_filtro, None)
        successi_benchmark, tentativi_benchmark = s_base, t_base
        freq_benchmark = s_base / t_base if t_base > 0 else 0.0
        log_message(f"  Performance Metodo Base Singolo (Benchmark): {freq_benchmark:.2%} ({s_base}/{t_base} casi)")
    else:
        s_base_comb, t_base_comb, f_base_comb = analizza_copertura_ambate_previste_multiple(storico, metodi_base_attivi, ruote_gioco_selezionate, lookahead, indice_mese_filtro, None)
        successi_benchmark, tentativi_benchmark = s_base_comb, t_base_comb
        freq_benchmark = f_base_comb
        log_message(f"  Performance Combinata Metodi Base (Benchmark): {freq_benchmark:.2%} ({s_base_comb}/{t_base_comb} casi)")

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

    termini2_operazionali = []
    for r2 in RUOTE: 
        for p2 in range(5):
            termini2_operazionali.append({'tipo_termine': 'estratto', 'ruota': r2, 'posizione': p2, 'str': f"{r2}[{p2+1}]"})
    for val_f2 in range(1, 91): 
        termini2_operazionali.append({'tipo_termine': 'fisso', 'valore_fisso': val_f2, 'str': f"F({val_f2})"})

    if cerca_diff_estr_fisso:
        log_message("  Ricerca Correttori Operazionali: Estratto - Fisso...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for val_f_corr in range(1, 91):
                    t2_dict = {'tipo_termine': 'fisso', 'valore_fisso': val_f_corr}
                    dett_str = f"{t1_dict['str']} - {val_f_corr}"
                    risultati_correttori_candidati.append({
                        "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                        "op_interna_corr": "-", "termine_corr_2_dict": t2_dict,
                        "tipo_descrittivo": "Estratto - Fisso", "dettaglio_correttore_str": dett_str
                    })
    if cerca_diff_estr_estr:
        log_message("  Ricerca Correttori Operazionali: Estratto - Estratto...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali: 
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
                for val_f_corr in range(1, 91): 
                    if val_f_corr == 0: continue 
                    t2_dict = {'tipo_termine': 'fisso', 'valore_fisso': val_f_corr}
                    dett_str = f"{t1_dict['str']} * {val_f_corr}"
                    risultati_correttori_candidati.append({
                        "op_link_base": op_link_base, "termine_corr_1_dict": t1_dict,
                        "op_interna_corr": "*", "termine_corr_2_dict": t2_dict,
                        "tipo_descrittivo": "Estratto * Fisso", "dettaglio_correttore_str": dett_str
                    })
    if cerca_mult_estr_estr:
        log_message("  Ricerca Correttori Operazionali: Estratto * Estratto...")
        for op_link_base in operazioni_collegamento_base:
            for t1_dict in termini1_operazionali:
                for t2_dict_op in termini2_operazionali:
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
        if processed_count % 500 == 0: 
            log_message(f"  Processati {processed_count}/{len(risultati_correttori_candidati)} candidati correttore...")

        op_l_base = cand_corr_info["op_link_base"]
        term1_c_dict = cand_corr_info["termine_corr_1_dict"]
        op_int_c = cand_corr_info["op_interna_corr"]
        term2_c_dict = cand_corr_info["termine_corr_2_dict"]

        metodi_estesi_da_valutare = []
        def_met_est_1_curr, def_met_est_2_curr = None, None

        if definizione_metodo_base_1:
            try:
                if op_int_c: 
                    def_met_est_1_curr = costruisci_metodo_esteso_operazionale(
                        definizione_metodo_base_1, op_l_base, term1_c_dict, op_int_c, term2_c_dict
                    )
                else: 
                    def_met_est_1_curr = costruisci_metodo_esteso(
                        definizione_metodo_base_1, op_l_base, term1_c_dict
                    )
                metodi_estesi_da_valutare.append(def_met_est_1_curr)
            except ValueError: pass 

        if definizione_metodo_base_2:
            try:
                if op_int_c:
                    def_met_est_2_curr = costruisci_metodo_esteso_operazionale(
                        definizione_metodo_base_2, op_l_base, term1_c_dict, op_int_c, term2_c_dict
                    )
                else:
                    def_met_est_2_curr = costruisci_metodo_esteso(
                        definizione_metodo_base_2, op_l_base, term1_c_dict
                    )
                metodi_estesi_da_valutare.append(def_met_est_2_curr)
            except ValueError: pass

        if not metodi_estesi_da_valutare : continue 

        s_corr, t_corr, f_corr = 0,0,0.0
        if len(metodi_estesi_da_valutare) == 1:
            s, t, _ = analizza_metodo_complesso_specifico(
                storico, metodi_estesi_da_valutare[0], ruote_gioco_selezionate,
                lookahead, indice_mese_filtro, None
            )
            s_corr, t_corr = s, t
            f_corr = s / t if t > 0 else 0.0
        else: 
            s_c, t_c, f_c = analizza_copertura_ambate_previste_multiple(
                storico, metodi_estesi_da_valutare, ruote_gioco_selezionate,
                lookahead, indice_mese_filtro, None
            )
            s_corr, t_corr, f_corr = s_c, t_c, f_c

        if t_corr >= min_tentativi_per_correttore and s_corr > 0:
            risultati_finali_correttori.append({
                'def_metodo_esteso_1': def_met_est_1_curr,
                'def_metodo_esteso_2': def_met_est_2_curr,
                'tipo_correttore_descrittivo': cand_corr_info["tipo_descrittivo"], 
                'dettaglio_correttore_str': cand_corr_info["dettaglio_correttore_str"], 
                'operazione_collegamento_base': op_l_base, 
                'successi': s_corr, 'tentativi': t_corr, 'frequenza': f_corr
            })

    log_message(f"  Completata valutazione. Trovati {len(risultati_finali_correttori)} candidati correttori con performance positiva.")
    risultati_finali_correttori.sort(key=lambda x: (x['frequenza'], x['successi']), reverse=True)

    migliori_correttori_output = []
    if risultati_finali_correttori:
        if tentativi_benchmark > 0:
            for rc_f in risultati_finali_correttori:
                if rc_f['frequenza'] > freq_benchmark:
                    migliori_correttori_output.append(rc_f)
                elif rc_f['frequenza'] == freq_benchmark and rc_f['successi'] > successi_benchmark:
                    migliori_correttori_output.append(rc_f)
            if not migliori_correttori_output and risultati_finali_correttori:
                 log_message("    Nessun correttore trovato è STRETTAMENTE MIGLIORE del benchmark.")
            elif migliori_correttori_output:
                 log_message(f"    Filtrati {len(migliori_correttori_output)} correttori che migliorano il benchmark.")
        else:
            migliori_correttori_output = risultati_finali_correttori
            log_message("    Benchmark metodi base non significativo, considero tutti i candidati correttori validi come migliorativi.")
    else:
        log_message("    Nessun correttore candidato trovato dopo la valutazione (min. tentativi/successi).")

    log_message(f"Ricerca correttori terminata. Restituiti {len(migliori_correttori_output)} correttori migliorativi.\n")
    return migliori_correttori_output

class LottoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Costruttore Metodi Lotto Avanzato")
        master.geometry("850x750")

        # --- Variabili d'Istanza Comuni ---
        self.cartella_dati_var = tk.StringVar()
        self.ruote_gioco_vars = {ruota: tk.BooleanVar() for ruota in RUOTE}
        self.tutte_le_ruote_var = tk.BooleanVar(value=True)
        self.lookahead_var = tk.IntVar(value=3)
        self.indice_mese_var = tk.StringVar()
        self.storico_caricato = None
        self.active_tab_ruote_checkbox_widgets = []

        # --- Variabili per Metodi Semplici ---
        self.ruota_calcolo_var = tk.StringVar(value=RUOTE[0])
        self.posizione_estratto_var = tk.IntVar(value=1)
        self.num_ambate_var = tk.IntVar(value=1)
        self.min_tentativi_var = tk.IntVar(value=10)

        # --- Variabili per Metodo Complesso 1 ---
        self.definizione_metodo_complesso_attuale = []
        self.mc_tipo_termine_var = tk.StringVar(value="estratto")
        self.mc_ruota_var = tk.StringVar(value=RUOTE[0])
        self.mc_posizione_var = tk.IntVar(value=1)
        self.mc_valore_fisso_var = tk.IntVar(value=1)
        self.mc_operazione_var = tk.StringVar(value='+')

        # --- Variabili per Metodo Complesso 2 ---
        self.definizione_metodo_complesso_attuale_2 = []
        self.mc_tipo_termine_var_2 = tk.StringVar(value="estratto")
        self.mc_ruota_var_2 = tk.StringVar(value=RUOTE[0])
        self.mc_posizione_var_2 = tk.IntVar(value=1)
        self.mc_valore_fisso_var_2 = tk.IntVar(value=1)
        self.mc_operazione_var_2 = tk.StringVar(value='+')

        # --- Variabili per Impostazioni Correttore ---
        self.corr_cfg_cerca_fisso_semplice = tk.BooleanVar(value=True)
        self.corr_cfg_cerca_estratto_semplice = tk.BooleanVar(value=True)
        self.corr_cfg_cerca_diff_estr_fisso = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_diff_estr_estr = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_mult_estr_fisso = tk.BooleanVar(value=False)
        self.corr_cfg_cerca_mult_estr_estr = tk.BooleanVar(value=False)
        self.corr_cfg_min_tentativi = tk.IntVar(value=5)

        # --- Variabili per Verifica Giocata Manuale ---
        self.numeri_verifica_var = tk.StringVar()
        self.colpi_verifica_var = tk.IntVar(value=9)

        # --- CREAZIONE DELLA BARRA DEL MENU ---
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
        
        # --- GUI: Widget Superiori (Caricamento Dati) ---
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

        # --- GUI: Notebook Principale per le Tab ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        tab_metodi_semplici = ttk.Frame(self.notebook)
        self.notebook.add(tab_metodi_semplici, text='Ricerca Metodi Semplici')
        self.crea_gui_metodi_semplici(tab_metodi_semplici)
        
        tab_metodo_complesso = ttk.Frame(self.notebook) 
        self.notebook.add(tab_metodo_complesso, text='Analisi Metodo Complesso')
        self.crea_gui_metodo_complesso(tab_metodo_complesso) 
        
        tab_verifica_manuale = ttk.Frame(self.notebook)
        self.notebook.add(tab_verifica_manuale, text='Verifica Giocata Manuale')
        self.crea_gui_verifica_manuale(tab_verifica_manuale)
        
        # --- GUI: Frame per Controlli Output e Area di Testo ---
        output_controls_frame = ttk.Frame(master)
        output_controls_frame.pack(fill=tk.X, padx=10, pady=(5,0)) 
        output_label = tk.Label(output_controls_frame, text="Log e Risultati:", font=("Helvetica", 10, "bold"))
        output_label.pack(side=tk.LEFT, anchor="w") 
        self.btn_pulisci_output = tk.Button(output_controls_frame, text="Pulisci Output", command=self._clear_output_area_manual)
        self.btn_pulisci_output.pack(side=tk.LEFT, padx=10) 
        
        self.output_text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=90, height=25, font=("Courier New", 9))
        self.output_text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.output_text_area.config(state=tk.DISABLED)

        if self.notebook.tabs(): 
            self.master.after(100, lambda: {
                self.on_tab_changed(None),
                self._aggiorna_data_inizio_verifica()
            })

    def apri_dialogo_impostazioni_correttore(self):
        # ... (corpo della funzione invariato) ...
        dialog = tk.Toplevel(self.master)
        dialog.title("Impostazioni Ricerca Correttore")
        dialog.geometry("450x350") 
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
        if dialog_height <= 1: dialog_height = 350 
        center_x = master_x + (master_width // 2) - (dialog_width // 2)
        center_y = master_y + (master_height // 2) - (dialog_height // 2)
        dialog.geometry(f"+{center_x}+{center_y}")
        dialog.wait_window()

    def _aggiorna_data_inizio_verifica(self, event=None):
        # ... (corpo della funzione invariato) ...
        if hasattr(self, 'date_fine_entry_analisi') and hasattr(self, 'date_inizio_verifica_entry'):
            try:
                data_fine = self.date_fine_entry_analisi.get_date()
                self.date_inizio_verifica_entry.set_date(data_fine)
            except ValueError:
                pass
            except AttributeError:
                pass

    def _pulisci_data_fine_e_verifica(self, event=None):
        # ... (corpo della funzione invariato) ...
        if hasattr(self, 'date_fine_entry_analisi'):
            self.date_fine_entry_analisi.delete(0, tk.END)
        if hasattr(self, 'date_inizio_verifica_entry'):
            self.date_inizio_verifica_entry.delete(0, tk.END)

    def crea_gui_controlli_comuni(self, parent_frame_main_tab):
        # ... (corpo della funzione invariato) ...
        common_game_settings_frame = ttk.LabelFrame(parent_frame_main_tab, text="Impostazioni di Gioco Comuni (per questa tab)", padding="10")
        common_game_settings_frame.pack(padx=10, pady=5, fill=tk.X, expand=False)
        current_row_cgs = 0
        tk.Label(common_game_settings_frame, text="Ruote di Gioco:").grid(row=current_row_cgs, column=0, sticky="nw", padx=5, pady=2)
        ruote_frame_analisi = tk.Frame(common_game_settings_frame)
        ruote_frame_analisi.grid(row=current_row_cgs, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        tk.Checkbutton(ruote_frame_analisi, text="Tutte le Ruote", variable=self.tutte_le_ruote_var, command=self.toggle_tutte_ruote).grid(row=0, column=0, columnspan=4, sticky="w")
        for i, ruota in enumerate(RUOTE):
            tk.Checkbutton(ruote_frame_analisi, text=ruota, variable=self.ruote_gioco_vars[ruota], command=self.update_tutte_le_ruote_status).grid(row=1 + i // 4, column=i % 4, sticky="w")
        current_row_cgs += (len(RUOTE) // 4) + 2 
        tk.Label(common_game_settings_frame, text="Colpi di Gioco (Lookahead):").grid(row=current_row_cgs, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(common_game_settings_frame, from_=1, to=18, textvariable=self.lookahead_var, width=5, state="readonly").grid(row=current_row_cgs, column=1, sticky="w", padx=5, pady=2)
        current_row_cgs += 1
        tk.Label(common_game_settings_frame, text="Indice Estrazione del Mese (vuoto=tutte):").grid(row=current_row_cgs, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(common_game_settings_frame, textvariable=self.indice_mese_var, width=7).grid(row=current_row_cgs, column=1, sticky="w", padx=5, pady=2)

    def on_tab_changed(self, event):
        # ... (corpo della funzione invariato) ...
        self.active_tab_ruote_checkbox_widgets = []
        try:
            current_tab_id = self.notebook.select()
            if not current_tab_id: return 
            current_tab_frame = self.notebook.nametowidget(current_tab_id)
            for child_l1 in current_tab_frame.winfo_children(): 
                if isinstance(child_l1, ttk.LabelFrame) and "Impostazioni di Gioco Comuni" in child_l1.cget("text"):
                    for child_l2 in child_l1.winfo_children(): 
                        if isinstance(child_l2, tk.Frame): 
                            is_target_ruote_frame = False; temp_widget_list = []
                            for widget_in_frame in child_l2.winfo_children():
                                if isinstance(widget_in_frame, tk.Checkbutton):
                                    if widget_in_frame.cget("text") == "Tutte le Ruote":
                                        is_target_ruote_frame = True 
                                    if widget_in_frame.cget("text") in RUOTE: 
                                        temp_widget_list.append(widget_in_frame)
                            if is_target_ruote_frame: 
                                self.active_tab_ruote_checkbox_widgets = temp_widget_list
                                break 
                    if self.active_tab_ruote_checkbox_widgets or (is_target_ruote_frame if 'is_target_ruote_frame' in locals() else False) : break 
        except Exception as e:
            print(f"Errore in on_tab_changed: {e}")
        self.toggle_tutte_ruote()

    def toggle_tutte_ruote(self):
        # ... (corpo della funzione invariato) ...
        stato_tutte_var = self.tutte_le_ruote_var.get()
        for nome_ruota in self.ruote_gioco_vars:
            self.ruote_gioco_vars[nome_ruota].set(stato_tutte_var)
        nuovo_stato_widget_figli = tk.DISABLED if stato_tutte_var else tk.NORMAL
        for cb_widget in self.active_tab_ruote_checkbox_widgets:
            cb_widget.config(state=nuovo_stato_widget_figli)

    def update_tutte_le_ruote_status(self):
        # ... (corpo della funzione invariato) ...
        tutti_figli_selezionati = all(self.ruote_gioco_vars[ruota].get() for ruota in RUOTE)
        if self.tutte_le_ruote_var.get() != tutti_figli_selezionati:
            self.tutte_le_ruote_var.set(tutti_figli_selezionati)

    def crea_gui_metodi_semplici(self, parent_tab):
        # ... (corpo della funzione invariato) ...
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')
        self.crea_gui_controlli_comuni(main_frame)
        simple_method_params_frame = ttk.LabelFrame(main_frame, text="Parametri Ricerca Metodi Semplici", padding="10")
        simple_method_params_frame.pack(padx=10, pady=5, fill=tk.X, expand=False)
        save_load_buttons_frame = ttk.Frame(simple_method_params_frame)
        save_load_buttons_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
        tk.Button(save_load_buttons_frame, text="Salva Imp. Semplici", command=self.salva_impostazioni_semplici).pack(side=tk.LEFT, padx=5)
        tk.Button(save_load_buttons_frame, text="Apri Imp. Semplici", command=self.apri_impostazioni_semplici).pack(side=tk.LEFT, padx=5)
        current_row = 1
        tk.Label(simple_method_params_frame, text="Ruota Calcolo Base:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(simple_method_params_frame, textvariable=self.ruota_calcolo_var, values=RUOTE, state="readonly", width=15).grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(simple_method_params_frame, text="Posizione Estratto Base (1-5):").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(simple_method_params_frame, from_=1, to=5, textvariable=self.posizione_estratto_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(simple_method_params_frame, text="N. Ambate da Dettagliare:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(simple_method_params_frame, from_=1, to=10, textvariable=self.num_ambate_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Label(simple_method_params_frame, text="Min. Tentativi per Metodo:").grid(row=current_row, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(simple_method_params_frame, from_=1, to=100, textvariable=self.min_tentativi_var, width=5, state="readonly").grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        current_row += 1
        tk.Button(simple_method_params_frame, text="Avvia Ricerca Metodi Semplici", command=self.avvia_analisi_metodi_semplici, font=("Helvetica", 11, "bold"), bg="lightgreen").grid(row=current_row, column=0, columnspan=2, pady=10, ipady=3)

    def crea_gui_metodo_complesso(self, parent_tab):
        # ... (corpo della funzione invariato) ...
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')
        self.crea_gui_controlli_comuni(main_frame)
        costruttori_main_frame = ttk.LabelFrame(main_frame, text="Costruttori Metodi Complessi", padding="10")
        costruttori_main_frame.pack(padx=10, pady=(10,5), fill=tk.X, expand=False)
        save_load_mc_frame = ttk.Frame(costruttori_main_frame)
        save_load_mc_frame.pack(fill=tk.X, pady=(0,10), anchor='nw')
        tk.Button(save_load_mc_frame, text="Salva Metodi Compl.", command=self.salva_metodi_complessi).pack(side=tk.LEFT, padx=5)
        tk.Button(save_load_mc_frame, text="Apri Metodi Compl.", command=self.apri_metodi_complessi).pack(side=tk.LEFT, padx=5)
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
        tk.Button(main_frame, text="Analizza Metodi Base Definiti", command=self.avvia_analisi_metodo_complesso, font=("Helvetica", 10, "bold"), bg="lightcoral" ).pack(pady=(5,5), ipady=2, fill=tk.X, padx=10)
        correttore_frame = ttk.LabelFrame(main_frame, text="Ricerca Correttore Ottimale", padding="10")
        correttore_frame.pack(padx=10, pady=(0,10), fill=tk.X, expand=False)
        tk.Button(correttore_frame, text="Trova Correttore", command=self.avvia_ricerca_correttore, font=("Helvetica", 11, "bold"), bg="gold" ).pack(pady=5, ipady=3)
        self._update_mc_input_state_1()
        self._refresh_mc_listbox_1()
        self._update_mc_input_state_2()
        self._refresh_mc_listbox_2()

    def crea_gui_verifica_manuale(self, parent_tab):
        # ... (corpo della funzione invariato) ...
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')
        self.crea_gui_controlli_comuni(main_frame)
        verifica_params_frame = ttk.LabelFrame(main_frame, text="Parametri Verifica Giocata Specifica", padding="10")
        verifica_params_frame.pack(padx=10, pady=10, fill=tk.X, expand=False)
        current_row_vf = 0
        tk.Label(verifica_params_frame, text="Numeri da Verificare (es. 23 o 23,45,67):").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        tk.Entry(verifica_params_frame, textvariable=self.numeri_verifica_var, width=30).grid(row=current_row_vf, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_params_frame, text="Data Inizio Verifica:").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        self.date_inizio_verifica_entry = DateEntry(verifica_params_frame, width=12, date_pattern='yyyy-mm-dd', state="readonly")
        self.date_inizio_verifica_entry.grid(row=current_row_vf, column=1, sticky="w", padx=5, pady=2)
        current_row_vf += 1
        tk.Label(verifica_params_frame, text="Colpi per Verifica (1-18):").grid(row=current_row_vf, column=0, sticky="w", padx=5, pady=2)
        tk.Spinbox(verifica_params_frame, from_=1, to=18, textvariable=self.colpi_verifica_var, width=5, state="readonly").grid(row=current_row_vf, column=1, sticky="w", padx=5, pady=2)
        current_row_vf += 1
        tk.Button(verifica_params_frame, text="Verifica Giocata", command=self.avvia_verifica_giocata, font=("Helvetica", 11, "bold"), bg="lightblue").grid(row=current_row_vf, column=0, columnspan=3, pady=10, ipady=3)

    def _format_componente_per_display(self, componente):
        # ... (corpo della funzione invariato) ...
        op_succ = componente['operazione_successiva']; op_str = f" {op_succ} " if op_succ and op_succ != '=' else ""
        if componente['tipo_termine'] == 'estratto': return f"{componente['ruota']}[{componente['posizione']+1}]{op_str}"
        elif componente['tipo_termine'] == 'fisso': return f"Fisso({componente['valore_fisso']}){op_str}"
        return "ERRORE_COMP"

    def _update_mc_input_state_1(self):
        # ... (corpo della funzione invariato) ...
        if hasattr(self, 'mc_ruota_combo_1'):
            tipo_termine = self.mc_tipo_termine_var.get()
            is_estratto = tipo_termine == "estratto"
            self.mc_ruota_combo_1.config(state="readonly" if is_estratto else "disabled")
            self.mc_pos_spinbox_1.config(state="readonly" if is_estratto else "disabled")
            self.mc_fisso_spinbox_1.config(state="disabled" if is_estratto else "readonly")
            self.mc_ruota_label_1.config(state="normal" if is_estratto else "disabled")
            self.mc_pos_label_1.config(state="normal" if is_estratto else "disabled")
            self.mc_fisso_label_1.config(state="disabled" if is_estratto else "normal")

    def _update_mc_input_state_2(self):
        # ... (corpo della funzione invariato) ...
        if hasattr(self, 'mc_ruota_combo_2'):
            tipo_termine = self.mc_tipo_termine_var_2.get()
            is_estratto = tipo_termine == "estratto"
            self.mc_ruota_combo_2.config(state="readonly" if is_estratto else "disabled")
            self.mc_pos_spinbox_2.config(state="readonly" if is_estratto else "disabled")
            self.mc_fisso_spinbox_2.config(state="disabled" if is_estratto else "readonly")
            self.mc_ruota_label_2.config(state="normal" if is_estratto else "disabled")
            self.mc_pos_label_2.config(state="normal" if is_estratto else "disabled")
            self.mc_fisso_label_2.config(state="disabled" if is_estratto else "normal")

    def _refresh_mc_listbox_1(self):
        # ... (corpo della funzione invariato) ...
         if hasattr(self, 'mc_listbox_componenti_1') and self.mc_listbox_componenti_1.winfo_exists():
            self.mc_listbox_componenti_1.delete(0, tk.END)
            display_str = "".join(self._format_componente_per_display(comp) for comp in self.definizione_metodo_complesso_attuale)
            self.mc_listbox_componenti_1.insert(tk.END, display_str if display_str else "Nessun componente definito.")

    def _refresh_mc_listbox_2(self):
        # ... (corpo della funzione invariato) ...
        if hasattr(self, 'mc_listbox_componenti_2') and self.mc_listbox_componenti_2.winfo_exists():
            self.mc_listbox_componenti_2.delete(0, tk.END)
            display_str = "".join(self._format_componente_per_display(comp) for comp in self.definizione_metodo_complesso_attuale_2)
            self.mc_listbox_componenti_2.insert(tk.END, display_str if display_str else "Nessun componente definito.")
            
    def aggiungi_componente_metodo_1(self):
        # ... (corpo della funzione invariato) ...
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
        # ... (corpo della funzione invariato) ...
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
        # ... (corpo della funzione invariato) ...
        if self.definizione_metodo_complesso_attuale:
            self.definizione_metodo_complesso_attuale.pop()
            self._refresh_mc_listbox_1()

    def pulisci_metodo_complesso_1(self):
        # ... (corpo della funzione invariato) ...
        self.definizione_metodo_complesso_attuale.clear()
        self._refresh_mc_listbox_1()

    def rimuovi_ultimo_componente_metodo_2(self):
        # ... (corpo della funzione invariato) ...
        if self.definizione_metodo_complesso_attuale_2:
            self.definizione_metodo_complesso_attuale_2.pop()
            self._refresh_mc_listbox_2()

    def pulisci_metodo_complesso_2(self):
        # ... (corpo della funzione invariato) ...
        self.definizione_metodo_complesso_attuale_2.clear()
        self._refresh_mc_listbox_2()

    def _log_to_gui(self, message, end='\n', flush=False):
        # ... (corpo della funzione invariato) ...
        self.output_text_area.config(state=tk.NORMAL)
        self.output_text_area.insert(tk.END, message + end)
        if flush or "\r" in message: self.output_text_area.see(tk.END); self.output_text_area.update_idletasks()
        self.output_text_area.config(state=tk.DISABLED); self.output_text_area.see(tk.END)

    def seleziona_cartella(self):
        # ... (corpo della funzione invariato) ...
        cartella = filedialog.askdirectory(title="Seleziona cartella archivi")
        if cartella: self.cartella_dati_var.set(cartella)

    def _clear_output_area_manual(self):
        # ... (corpo della funzione invariato) ...
        self.output_text_area.config(state=tk.NORMAL)
        self.output_text_area.delete('1.0', tk.END)
        self.output_text_area.config(state=tk.DISABLED)

    def _carica_e_valida_storico_comune(self, usa_filtri_data_globali=True):
        # ... (corpo della funzione invariato) ...
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
        # ... (corpo della funzione invariato) ...
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
        # ... (corpo della funzione invariato, usa il formato "strutturato" che hai già) ...
        impostazioni = {
            "versione_formato": 1.1, 
            "tipo_metodo": "semplice",
            "struttura_base_ricerca": {  
                "ruota_calcolo": self.ruota_calcolo_var.get(),
                "posizione_estratto": self.posizione_estratto_var.get(),
            },
            "impostazioni_analisi": {
                "num_ambate_dettagliare": self.num_ambate_var.get(),
                "min_tentativi_metodo": self.min_tentativi_var.get(),
            },
            "impostazioni_gioco": {
                "tutte_le_ruote": self.tutte_le_ruote_var.get(),
                "ruote_gioco_selezionate": [r for r, v in self.ruote_gioco_vars.items() if v.get()],
                "lookahead": self.lookahead_var.get(),
                "indice_mese": self.indice_mese_var.get(),
            }
        }
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("File JSON Imp. Semplici", "*.json"), ("Tutti i file", "*.*")],
            title="Salva Impostazioni Metodo Semplice"
        )
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(impostazioni, f, indent=4)
            self._log_to_gui(f"Impostazioni Metodo Semplice (strutturate) salvate in: {filepath}"); messagebox.showinfo("Salvataggio", "Impostazioni salvate!")
        except Exception as e: self._log_to_gui(f"Errore salvataggio imp. semplici: {e}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare:\n{e}")

    def apri_impostazioni_semplici(self):
        # ... (corpo della funzione invariato, gestisce sia il formato nuovo che quello vecchio) ...
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("File JSON Imp. Semplici", "*.json"), ("Tutti i file", "*.*")],
            title="Apri Impostazioni Metodo Semplice"
        )
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: impostazioni = json.load(f)
            
            if impostazioni.get("tipo_metodo") != "semplice":
                messagebox.showerror("Errore Apertura", "File non valido per Metodo Semplice."); return

            if impostazioni.get("versione_formato") == 1.1: 
                struttura_base = impostazioni.get("struttura_base_ricerca", {})
                self.ruota_calcolo_var.set(struttura_base.get("ruota_calcolo", RUOTE[0]))
                self.posizione_estratto_var.set(struttura_base.get("posizione_estratto", 1))
                impostazioni_analisi_load = impostazioni.get("impostazioni_analisi", {})
                self.num_ambate_var.set(impostazioni_analisi_load.get("num_ambate_dettagliare", 1))
                self.min_tentativi_var.set(impostazioni_analisi_load.get("min_tentativi_metodo", 10))
                impostazioni_gioco_load = impostazioni.get("impostazioni_gioco", {})
                self.tutte_le_ruote_var.set(impostazioni_gioco_load.get("tutte_le_ruote", True))
                ruote_sel_caricate = impostazioni_gioco_load.get("ruote_gioco_selezionate", [])
                self.lookahead_var.set(impostazioni_gioco_load.get("lookahead", 3))
                self.indice_mese_var.set(impostazioni_gioco_load.get("indice_mese", ""))
            else: 
                self._log_to_gui("INFO: Caricamento file impostazioni semplici in vecchio formato (flat).")
                self.ruota_calcolo_var.set(impostazioni.get("ruota_calcolo_base", RUOTE[0]))
                self.posizione_estratto_var.set(impostazioni.get("posizione_estratto_base", 1))
                self.num_ambate_var.set(impostazioni.get("num_ambate_dettagliare", 1))
                self.min_tentativi_var.set(impostazioni.get("min_tentativi_metodo", 10))
                self.tutte_le_ruote_var.set(impostazioni.get("tutte_le_ruote", True))
                ruote_sel_caricate = impostazioni.get("ruote_gioco_selezionate", [])
                self.lookahead_var.set(impostazioni.get("lookahead", 3))
                self.indice_mese_var.set(impostazioni.get("indice_mese", ""))

            if not self.tutte_le_ruote_var.get():
                for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(ruota in ruote_sel_caricate)
            else:
                for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(True)
            
            self.on_tab_changed(None) 
            self._log_to_gui(f"Impostazioni Metodo Semplice caricate da: {filepath}"); messagebox.showinfo("Apertura", "Impostazioni caricate!")
        except Exception as e: self._log_to_gui(f"Errore apertura imp. semplici: {e}"); messagebox.showerror("Errore Apertura", f"Impossibile aprire:\n{e}")

    def salva_metodi_complessi(self):
        # ... (corpo della funzione invariato, usa il formato "strutturato" che hai già) ...
        if not self.definizione_metodo_complesso_attuale and not self.definizione_metodo_complesso_attuale_2:
            messagebox.showwarning("Salvataggio", "Nessun metodo complesso definito."); return
        
        impostazioni = {
            "versione_formato_lmc": 1.1, 
            "tipo_metodo_file": "complessi_multi", 
            "strutture_metodi_complessi": { 
                "metodo_1": self.definizione_metodo_complesso_attuale,
                "metodo_2": self.definizione_metodo_complesso_attuale_2,
            },
            "impostazioni_gioco": {
                "tutte_le_ruote": self.tutte_le_ruote_var.get(),
                "ruote_gioco_selezionate": [r for r, v in self.ruote_gioco_vars.items() if v.get()],
                "lookahead": self.lookahead_var.get(),
                "indice_mese": self.indice_mese_var.get()
            }
        }
        filepath = filedialog.asksaveasfilename(
            defaultextension=".lmc2", filetypes=[("File Metodi Lotto Complessi", "*.lmc2"), ("Tutti i file", "*.*")],
            title="Salva Metodi Complessi"
        )
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(impostazioni, f, indent=4)
            self._log_to_gui(f"Metodi Complessi (strutturati) salvati in: {filepath}"); messagebox.showinfo("Salvataggio", "Metodi salvati!")
        except Exception as e: self._log_to_gui(f"Errore salvataggio: {e}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare:\n{e}")

    def apri_metodi_complessi(self):
        # ... (corpo della funzione invariato, gestisce sia il formato nuovo che quello vecchio) ...
        filepath = filedialog.askopenfilename(
            defaultextension=".lmc2",
            filetypes=[("File Metodi Lotto Complessi", "*.lmc2"),("File Metodo Lotto Complesso (Vecchio)", "*.lmc"), ("Tutti i file", "*.*")],
            title="Apri Metodi Complessi"
        )
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: impostazioni = json.load(f)

            definizione_m1_caricata = []
            definizione_m2_caricata = []
            tutte_ruote_caricato = True
            ruote_sel_caricate = []
            lookahead_caricato = 3
            indice_mese_caricato = ""

            if impostazioni.get("tipo_metodo_file") == "complessi_multi" and impostazioni.get("versione_formato_lmc") == 1.1:
                strutture_salvate = impostazioni.get("strutture_metodi_complessi", {})
                definizione_m1_caricata = strutture_salvate.get("metodo_1", [])
                definizione_m2_caricata = strutture_salvate.get("metodo_2", [])
                impostazioni_gioco_load = impostazioni.get("impostazioni_gioco", {})
                tutte_ruote_caricato = impostazioni_gioco_load.get("tutte_le_ruote", True)
                ruote_sel_caricate = impostazioni_gioco_load.get("ruote_gioco_selezionate", [])
                lookahead_caricato = impostazioni_gioco_load.get("lookahead", 3)
                indice_mese_caricato = impostazioni_gioco_load.get("indice_mese", "")
                
            elif impostazioni.get("tipo_metodo_file") == "complessi_multi": 
                self._log_to_gui("INFO: Caricamento file .lmc2 in vecchio formato (flat).")
                definizione_m1_caricata = impostazioni.get("definizione_metodo_1", [])
                definizione_m2_caricata = impostazioni.get("definizione_metodo_2", [])
                tutte_ruote_caricato = impostazioni.get("tutte_le_ruote", True)
                ruote_sel_caricate = impostazioni.get("ruote_gioco_selezionate", [])
                lookahead_caricato = impostazioni.get("lookahead", 3)
                indice_mese_caricato = impostazioni.get("indice_mese", "")
                
            elif impostazioni.get("tipo_metodo") == "complesso": 
                self._log_to_gui("INFO: Caricamento file .lmc (vecchio formato singolo metodo).")
                definizione_m1_caricata = impostazioni.get("definizione_metodo", [])
                definizione_m2_caricata = [] 
                tutte_ruote_caricato = impostazioni.get("tutte_le_ruote", True)
                ruote_sel_caricate = impostazioni.get("ruote_gioco_selezionate", [])
                lookahead_caricato = impostazioni.get("lookahead", 3)
                indice_mese_caricato = impostazioni.get("indice_mese", "")
                messagebox.showinfo("Compatibilità", "File in vecchio formato .lmc caricato. Solo Metodo Base 1 e impostazioni di gioco associate sono state applicate.")
            else:
                messagebox.showerror("Errore Apertura", "File non valido o formato non riconosciuto per Metodi Complessi."); return

            self.definizione_metodo_complesso_attuale = definizione_m1_caricata
            self.definizione_metodo_complesso_attuale_2 = definizione_m2_caricata
            self._refresh_mc_listbox_1()
            self._refresh_mc_listbox_2()

            self.tutte_le_ruote_var.set(tutte_ruote_caricato)
            if not self.tutte_le_ruote_var.get():
                 for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(ruota in ruote_sel_caricate)
            else:
                 for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(True)
            self.lookahead_var.set(lookahead_caricato)
            self.indice_mese_var.set(indice_mese_caricato)
            
            self.on_tab_changed(None) 
            self._log_to_gui(f"Metodi Complessi caricati da: {filepath}"); messagebox.showinfo("Apertura", "Metodi caricati!")
        except Exception as e: self._log_to_gui(f"Errore apertura: {e}"); messagebox.showerror("Errore Apertura", f"Impossibile aprire:\n{e}")

    def _prepara_e_salva_profilo_metodo(self, dati_profilo_metodo, tipo_file="lotto_metodo_profilo", estensione=".lmp"):
        if not dati_profilo_metodo:
            messagebox.showerror("Errore", "Nessun dato del metodo da salvare.")
            return

        nome_suggerito = "profilo_metodo"
        ambata_valida = False
        if dati_profilo_metodo.get("ambata_prevista") is not None:
            try:
                int(dati_profilo_metodo["ambata_prevista"]) # Verifica se è un numero o stringa numerica
                ambata_valida = True
            except (ValueError, TypeError):
                ambata_valida = False
        
        if ambata_valida:
             nome_suggerito = f"metodo_ambata_{dati_profilo_metodo['ambata_prevista']}"
        elif dati_profilo_metodo.get("formula_testuale"): # Usa la formula testuale se l'ambata non è valida/presente
            formula_semplice = dati_profilo_metodo["formula_testuale"]
            formula_semplice = formula_semplice.replace("[", "").replace("]", "").replace(" ", "_").replace("+", "piu").replace("-", "meno").replace("*", "per").replace("=", "")
            nome_suggerito = f"metodo_{formula_semplice[:30]}" # Limita lunghezza per nome file

        filepath = filedialog.asksaveasfilename(
            initialfile=nome_suggerito,
            defaultextension=estensione,
            filetypes=[(f"File Profilo Metodo ({estensione})", f"*{estensione}"), ("Tutti i file", "*.*")],
            title="Salva Profilo Metodo Analizzato"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dati_profilo_metodo, f, indent=4, default=str) # default=str per date
            self._log_to_gui(f"Profilo del metodo salvato in: {filepath}")
            messagebox.showinfo("Salvataggio Profilo", "Profilo del metodo salvato con successo!")
        except Exception as e:
            self._log_to_gui(f"Errore durante il salvataggio del profilo del metodo: {e}")
            messagebox.showerror("Errore Salvataggio", f"Impossibile salvare il profilo del metodo:\n{e}")

    def mostra_popup_previsione(self, titolo_popup,
                                   ruote_gioco_str, 
                                   lista_previsioni_dettagliate=None, 
                                   copertura_combinata_info=None, 
                                   data_riferimento_previsione_str_comune=None,
                                   metodi_grezzi_per_salvataggio=None 
                                  ):
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo_popup)

        popup_width = 700
        popup_height = 550
        if lista_previsioni_dettagliate and len(lista_previsioni_dettagliate) > 0:
            base_h_per_met = 220 
            head_foot_h = 150
            comb_h = 0
            if copertura_combinata_info and ("num_metodi_combinati" in copertura_combinata_info or "testo_introduttivo" in copertura_combinata_info):
                comb_h = 100
            num_m = len(lista_previsioni_dettagliate)
            popup_height = min(head_foot_h + (num_m * base_h_per_met) + comb_h, 780)
            popup_height = max(popup_height, 550) # Altezza minima


        popup_window.geometry(f"{popup_width}x{int(popup_height)}")
        # popup_window.grab_set()
        popup_window.transient(self.master)

        canvas = tk.Canvas(popup_window)
        scrollbar_y = ttk.Scrollbar(popup_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set)

        row_idx = 0
        ttk.Label(scrollable_frame, text=f"--- {titolo_popup} ---", font=("Helvetica", 12, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w")
        row_idx += 1

        if data_riferimento_previsione_str_comune:
            ttk.Label(scrollable_frame, text=f"Previsione del: {data_riferimento_previsione_str_comune}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
            row_idx += 1

        ttk.Label(scrollable_frame, text=f"Su ruote: {ruote_gioco_str}").grid(row=row_idx, column=0, columnspan=2, pady=(2,10), sticky="w")
        row_idx += 1

        if copertura_combinata_info and "testo_introduttivo" in copertura_combinata_info:
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=5)
            row_idx += 1
            testo_intro_corr = copertura_combinata_info['testo_introduttivo']
            ttk.Label(scrollable_frame, text=testo_intro_corr, wraplength=popup_width - 40).grid(row=row_idx, column=0, columnspan=2, pady=5, sticky="w")
            row_idx += 1

        if lista_previsioni_dettagliate:
            for idx_metodo, previsione_dett in enumerate(lista_previsioni_dettagliate):
                ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10)
                row_idx += 1

                titolo_sezione = previsione_dett.get('titolo_sezione', '--- PREVISIONE ---')
                ttk.Label(scrollable_frame, text=titolo_sezione, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=3, sticky="w")
                row_idx += 1

                if previsione_dett.get('info_metodo_str'):
                    ttk.Label(scrollable_frame, text=f"Metodo: {previsione_dett['info_metodo_str']}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
                    row_idx += 1

                ambata_loop = previsione_dett.get('ambata_prevista')
                if ambata_loop is None or str(ambata_loop).upper() == "N/D":
                    ttk.Label(scrollable_frame, text="Nessuna ambata valida prevista per questo metodo.").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
                    row_idx += 1
                else:
                    ttk.Label(scrollable_frame, text=f"AMBATA DA GIOCARE: {ambata_loop}", font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
                    row_idx += 1

                ttk.Label(scrollable_frame, text=f"Performance storica individuale: {previsione_dett.get('performance_storica_str', 'N/D')}").grid(row=row_idx, column=0, columnspan=2, pady=2, sticky="w")
                row_idx += 1
                
                dati_grezzi_per_questo_metodo = None
                if metodi_grezzi_per_salvataggio and idx_metodo < len(metodi_grezzi_per_salvataggio):
                    dati_grezzi_per_questo_metodo = metodi_grezzi_per_salvataggio[idx_metodo]
                
                if dati_grezzi_per_questo_metodo:
                    dati_da_salvare = dati_grezzi_per_questo_metodo.copy()
                    dati_da_salvare["ruote_gioco_analisi"] = ruote_gioco_str.split(", ") if ruote_gioco_str != "TUTTE" else RUOTE
                    dati_da_salvare["data_riferimento_analisi"] = data_riferimento_previsione_str_comune
                    dati_da_salvare["lookahead_analisi"] = self.lookahead_var.get() 
                    dati_da_salvare["indice_mese_analisi"] = self.indice_mese_var.get()
                    
                    if 'formula_testuale' not in dati_da_salvare and previsione_dett.get('info_metodo_str'):
                         dati_da_salvare['formula_testuale'] = previsione_dett['info_metodo_str']
                    if 'ambata_prevista' not in dati_da_salvare:
                         dati_da_salvare['ambata_prevista'] = ambata_loop
                    if 'abbinamenti' not in dati_da_salvare and previsione_dett.get('abbinamenti_dict'):
                         dati_da_salvare['abbinamenti'] = previsione_dett.get('abbinamenti_dict')

                    btn_salva_profilo = ttk.Button(
                        scrollable_frame,
                        text="Salva Questo Metodo",
                        command=lambda d=dati_da_salvare: self._prepara_e_salva_profilo_metodo(d)
                    )
                    btn_salva_profilo.grid(row=row_idx, column=0, pady=(5,0), sticky="w")
                    row_idx += 1

                if ambata_loop is not None and str(ambata_loop).upper() != "N/D":
                    ttk.Label(scrollable_frame, text="Abbinamenti Consigliati (co-occorrenze storiche):").grid(row=row_idx, column=0, columnspan=2, pady=(5,2), sticky="w")
                    row_idx +=1
                    abbinamenti_dict_loop = previsione_dett.get('abbinamenti_dict', {})
                    eventi_totali_loop = abbinamenti_dict_loop.get("sortite_ambata_target", 0)
                    if eventi_totali_loop > 0:
                        ttk.Label(scrollable_frame, text=f"  (Basato su {eventi_totali_loop} sortite storiche dell'ambata {ambata_loop} su ruote selezionate)").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w")
                        row_idx += 1
                        abbinamenti_mostrati_loop = False
                        for tipo_sorte in ["ambo", "terno", "quaterna", "cinquina"]:
                            dati_sorte_lista_loop = abbinamenti_dict_loop.get(tipo_sorte, [])
                            if dati_sorte_lista_loop:
                                testo_sorte_temp = ""
                                for ab_info_loop in dati_sorte_lista_loop[:3]:
                                    if ab_info_loop.get('conteggio', 0) > 0:
                                        numeri_ab_str_loop = ", ".join(map(str, sorted(ab_info_loop['numeri'])))
                                        freq_ab_loop = ab_info_loop.get('frequenza', 0.0)
                                        testo_sorte_temp += f"    - Numeri: [{numeri_ab_str_loop}] (Freq: {freq_ab_loop:.1%}, Cnt: {ab_info_loop['conteggio']})\n"
                                if testo_sorte_temp:
                                    ttk.Label(scrollable_frame, text=f"  Per {tipo_sorte.upper()}:").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                                    ttk.Label(scrollable_frame, text=testo_sorte_temp.strip(), justify=tk.LEFT).grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx +=1
                                    abbinamenti_mostrati_loop = True
                        if not abbinamenti_mostrati_loop:
                            ttk.Label(scrollable_frame, text="  Nessun abbinamento significativo trovato.").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w")
                            row_idx += 1
                    else:
                        ttk.Label(scrollable_frame, text=f"  Nessuna co-occorrenza storica per l'ambata {ambata_loop}.").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w")
                        row_idx += 1
        
        if copertura_combinata_info and "num_metodi_combinati" in copertura_combinata_info:
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10)
            row_idx += 1
            titolo_comb = f"--- PERFORMANCE COMBINATA DEI {copertura_combinata_info.get('num_metodi_combinati', 'N')} METODI ELENCATI ---"
            ttk.Label(scrollable_frame, text=titolo_comb, font=("Helvetica", 10, "bold")).grid(row=row_idx, column=0, columnspan=2, pady=3, sticky="w")
            row_idx += 1
            if copertura_combinata_info.get("tentativi", 0) > 0:
                ttk.Label(scrollable_frame, text="  Giocando simultaneamente le ambate prodotte:").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                ttk.Label(scrollable_frame, text=f"  - Successi Complessivi: {copertura_combinata_info['successi']}").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                ttk.Label(scrollable_frame, text=f"  - Tentativi Complessivi: {copertura_combinata_info['tentativi']}").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
                ttk.Label(scrollable_frame, text=f"  - Frequenza di Copertura: {copertura_combinata_info['frequenza']:.2%}").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w"); row_idx += 1
            else:
                ttk.Label(scrollable_frame, text="  Nessun tentativo combinato applicabile.").grid(row=row_idx, column=0, columnspan=2, pady=1, sticky="w")
                row_idx += 1

        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=(5,0)) 
        scrollbar_y.pack(side="right", fill="y")

        close_button_frame = ttk.Frame(popup_window) 
        close_button_frame.pack(fill=tk.X, pady=(0,5), padx=5, side=tk.BOTTOM) 
        ttk.Button(close_button_frame, text="Chiudi", command=popup_window.destroy).pack()

        popup_window.update_idletasks() 
        canvas.config(scrollregion=canvas.bbox("all"))

        self.master.eval(f'tk::PlaceWindow {str(popup_window)} center')
        # popup_window.wait_window()

    def apri_e_visualizza_profilo_metodo(self):
        filepath = filedialog.askopenfilename(
            defaultextension=".lmp",
            filetypes=[("File Profilo Metodo Lotto", "*.lmp"), ("Tutti i file", "*.*")],
            title="Apri Profilo Metodo Salvato"
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
            if formula_metodo_testo == "N/D" and "metodo" in dati_profilo and isinstance(dati_profilo["metodo"], dict): 
                m_s = dati_profilo["metodo"]
                formula_metodo_testo = f"{m_s.get('ruota_calcolo', '?')}[pos.{m_s.get('pos_estratto_calcolo', -1)+1}] {m_s.get('operazione','?')} {m_s.get('operando_fisso','?')}"
            elif formula_metodo_testo == "N/D": 
                if "definizione_metodo_esteso_1" in dati_profilo and dati_profilo["definizione_metodo_esteso_1"]:
                     formula_metodo_testo = "".join(self._format_componente_per_display(c) for c in dati_profilo["definizione_metodo_esteso_1"])
                elif "definizione_metodo_originale" in dati_profilo and dati_profilo["definizione_metodo_originale"]: 
                     formula_metodo_testo = "".join(self._format_componente_per_display(c) for c in dati_profilo["definizione_metodo_originale"])


            contenuto_testo_popup += f"Metodo: {formula_metodo_testo}\n"
            ambata_p = dati_profilo.get("ambata_prevista", "N/D")
            contenuto_testo_popup += f"AMBATA PREVISTA: {ambata_p}\n"

            ruote_g_p = dati_profilo.get("ruote_gioco_analisi", [])
            ruote_g_p_str = ", ".join(ruote_g_p) if isinstance(ruote_g_p, list) else "N/D"
            contenuto_testo_popup += f"Ruote di Gioco Analisi: {ruote_g_p_str}\n"
            
            perf_s = dati_profilo.get("successi", "N/A")
            perf_t = dati_profilo.get("tentativi", "N/A")
            perf_f_val = dati_profilo.get("frequenza_ambata") 
            if perf_f_val is None: perf_f_val = dati_profilo.get("frequenza")

            if perf_f_val is not None:
                try:
                    contenuto_testo_popup += f"Performance Storica: {float(perf_f_val):.2%} ({perf_s}/{perf_t} casi)\n"
                except ValueError:
                    contenuto_testo_popup += f"Performance Storica: {perf_f_val} ({perf_s}/{perf_t} casi)\n"
            else:
                contenuto_testo_popup += f"Performance Storica: {perf_s} successi su {perf_t} tentativi\n"

            abbinamenti = dati_profilo.get("abbinamenti", {})
            if ambata_p != "N/D" and abbinamenti:
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
                                    f_ab = ab_info.get('frequenza', 0.0)
                                    c_ab = ab_info.get('conteggio',0)
                                    try:
                                        contenuto_testo_popup += f"    - Numeri: [{n_ab_str}] (Freq: {float(f_ab):.1%}, Cnt: {c_ab})\n"
                                    except ValueError:
                                         contenuto_testo_popup += f"    - Numeri: [{n_ab_str}] (Freq: {f_ab}, Cnt: {c_ab})\n"
                else:
                    contenuto_testo_popup += f"  Nessuna co-occorrenza storica per l'ambata {ambata_p} al momento del salvataggio.\n"
            
            contenuto_testo_popup += "\nParametri di Analisi Usati (al momento del salvataggio):\n"
            contenuto_testo_popup += f"  Lookahead: {dati_profilo.get('lookahead_analisi', 'N/D')}\n"
            contenuto_testo_popup += f"  Indice Mese: {dati_profilo.get('indice_mese_analisi', 'N/D') or 'Tutte'}\n"

            if "tipo_correttore_descrittivo" in dati_profilo:
                contenuto_testo_popup += "\n--- Dettagli Correttore (se applicato) ---\n"
                contenuto_testo_popup += f"Tipo: {dati_profilo['tipo_correttore_descrittivo']}\n"
                contenuto_testo_popup += f"Dettaglio: {dati_profilo['dettaglio_correttore_str']}\n"
                contenuto_testo_popup += f"Operazione Collegamento Base: '{dati_profilo['operazione_collegamento_base']}'\n"

            self.mostra_popup_testo_semplice("Dettaglio Profilo Metodo Salvato", contenuto_testo_popup)

        except FileNotFoundError:
            messagebox.showerror("Errore Apertura", f"File non trovato: {filepath}")
            self._log_to_gui(f"ERRORE: File profilo metodo non trovato: {filepath}")
        except json.JSONDecodeError:
            messagebox.showerror("Errore Apertura", f"File profilo non è un JSON valido: {filepath}")
            self._log_to_gui(f"ERRORE: Impossibile decodificare JSON da {filepath}")
        except Exception as e:
            messagebox.showerror("Errore Apertura", f"Impossibile aprire il profilo del metodo:\n{e}")
            self._log_to_gui(f"ERRORE: Apertura profilo metodo fallita: {e}, {traceback.format_exc()}")

    def avvia_analisi_metodi_semplici(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA METODI SEMPLICI\n" + "="*50)
        storico_per_analisi = self._carica_e_valida_storico_comune()
        if not storico_per_analisi: return
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return
        ruota_calcolo_input = self.ruota_calcolo_var.get()
        posizione_estratto_input = self.posizione_estratto_var.get() - 1
        num_ambate_richieste_gui = self.num_ambate_var.get()
        min_tentativi = self.min_tentativi_var.get()
        # ... (log parametri) ...
        self._log_to_gui(f"Parametri Ricerca Metodi Semplici:\n  Ruota Base: {ruota_calcolo_input}, Posizione: {posizione_estratto_input+1}")
        self._log_to_gui(f"  Ruote Gioco: {', '.join(ruote_gioco)}\n  Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        self._log_to_gui(f"  Output Ambate Richieste: {num_ambate_richieste_gui}, Min. Tentativi per Metodo: {min_tentativi}")
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati_individuali, info_copertura_combinata = trova_migliori_ambate_e_abbinamenti(
                storico_per_analisi, ruota_calcolo_input, posizione_estratto_input, ruote_gioco,
                max_ambate_output=num_ambate_richieste_gui, lookahead=lookahead,
                indice_mese_filtro=indice_mese, min_tentativi_per_ambata=min_tentativi, app_logger=self._log_to_gui
            )
            # ... (log risultati grezzi) ...
            self._log_to_gui("\n\n--- RISULTATI FINALI RICERCA METODI SEMPLICI (LOG COMPLETO) ---")
            if not risultati_individuali: self._log_to_gui("Nessun metodo semplice ha prodotto risultati sufficientemente frequenti.")
            else:
                for i, res_log in enumerate(risultati_individuali):
                    metodo_log = res_log['metodo']
                    self._log_to_gui(f"\n--- {(i+1)}° METODO SEMPLICE TROVATO ---")
                    self._log_to_gui(f"  Metodo: {metodo_log['ruota_calcolo']}[pos.{metodo_log['pos_estratto_calcolo']+1}] {metodo_log['operazione']} {metodo_log['operando_fisso']}")
                    self._log_to_gui(f"  Ambata da previsione live: {res_log.get('ambata_piu_frequente_dal_metodo', 'N/D')}")
                    self._log_to_gui(f"  Frequenza successo Ambata (metodo): {res_log['frequenza_ambata']:.2%} ({res_log['successi']}/{res_log['tentativi']} casi)")

            self.master.config(cursor="")
            if risultati_individuali:
                lista_previsioni_per_popup = []
                data_riferimento_str_popup_s_comune = storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else None
                
                for res_idx, res_singolo_metodo in enumerate(risultati_individuali):
                    metodo_s_info = res_singolo_metodo['metodo']
                    dettaglio_previsione_per_popup = {
                        "ambata_prevista": res_singolo_metodo.get('ambata_piu_frequente_dal_metodo'),
                        "abbinamenti_dict": res_singolo_metodo.get("abbinamenti", {}),
                        "performance_storica_str": f"{res_singolo_metodo['frequenza_ambata']:.2%} ({res_singolo_metodo['successi']}/{res_singolo_metodo['tentativi']} casi)",
                        "info_metodo_str": f"{metodo_s_info['ruota_calcolo']}[pos.{metodo_s_info['pos_estratto_calcolo']+1}] {metodo_s_info['operazione']} {metodo_s_info['operando_fisso']}",
                        "titolo_sezione": f"--- {(res_idx+1)}° METODO / PREVISIONE ---"
                    }
                    lista_previsioni_per_popup.append(dettaglio_previsione_per_popup)
                
                ruote_gioco_str_popup_s = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                
                self.mostra_popup_previsione(
                   titolo_popup="Previsione Metodi Semplici", ruote_gioco_str=ruote_gioco_str_popup_s,
                   lista_previsioni_dettagliate=lista_previsioni_per_popup,
                   copertura_combinata_info=info_copertura_combinata if num_ambate_richieste_gui > 1 else None,
                   data_riferimento_previsione_str_comune=data_riferimento_str_popup_s_comune,
                   metodi_grezzi_per_salvataggio=risultati_individuali 
                )
            else: messagebox.showinfo("Analisi Metodi Semplici", "Nessun metodo semplice ha prodotto risultati sufficientemente frequenti.")
            self._log_to_gui("\n--- Ricerca Metodi Semplici Completata ---")
        except Exception as e:
            self.master.config(cursor=""); messagebox.showerror("Errore Analisi", f"Errore ricerca metodi semplici: {e}"); self._log_to_gui(f"ERRORE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def avvia_analisi_metodo_complesso(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO ANALISI METODI COMPLESSI BASE\n" + "="*50)
        metodo_1_def_mc = self.definizione_metodo_complesso_attuale
        metodo_2_def_mc = self.definizione_metodo_complesso_attuale_2
        err_msg_mc = []
        metodi_validi_per_analisi_def = [] 
        metodi_grezzi_per_popup_salvataggio = [] 

        if metodo_1_def_mc:
            if metodo_1_def_mc[-1].get('operazione_successiva') == '=':
                metodi_validi_per_analisi_def.append(metodo_1_def_mc)
            else: err_msg_mc.append("Metodo Base 1 non terminato con '='.")
        if metodo_2_def_mc:
            if metodo_2_def_mc[-1].get('operazione_successiva') == '=':
                metodi_validi_per_analisi_def.append(metodo_2_def_mc)
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
        lista_previsioni_popup_mc_vis = []
        info_copertura_combinata_mc = None
        data_riferimento_comune_popup = storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else None

        try:
            for idx, metodo_def_corrente in enumerate(metodi_validi_per_analisi_def):
                nome_metodo_log_popup = f"Metodo Base {idx + 1}"
                self._log_to_gui(f"\n--- ANALISI {nome_metodo_log_popup.upper()} ---")
                metodo_str_popup = "".join(self._format_componente_per_display(comp) for comp in metodo_def_corrente)
                self._log_to_gui(f"  Definizione: {metodo_str_popup}")
                
                s_ind, t_ind, applicazioni_vincenti_ind = analizza_metodo_complesso_specifico(
                    storico_per_analisi, metodo_def_corrente, ruote_gioco, lookahead, indice_mese, self._log_to_gui
                )
                f_ind = s_ind / t_ind if t_ind > 0 else 0.0
                perf_ind_str = f"{f_ind:.2%} ({s_ind}/{t_ind} casi)" if t_ind > 0 else "Non applicabile storicamente."
                self._log_to_gui(f"  Performance Storica Individuale {nome_metodo_log_popup}: {perf_ind_str}")

                ambata_live, abb_live = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                    storico_per_analisi, metodo_def_corrente, ruote_gioco, data_riferimento_comune_popup, nome_metodo_log_popup
                )
                
                lista_previsioni_popup_mc_vis.append({
                    "titolo_sezione": f"--- PREVISIONE {nome_metodo_log_popup.upper()} ---",
                    "info_metodo_str": metodo_str_popup,
                    "ambata_prevista": ambata_live,
                    "abbinamenti_dict": abb_live,
                    "performance_storica_str": perf_ind_str
                })
                metodi_grezzi_per_popup_salvataggio.append({
                    "tipo_metodo_salvato": "complesso_base",
                    "definizione_metodo_originale": metodo_def_corrente,
                    "formula_testuale": metodo_str_popup,
                    "ambata_prevista": ambata_live,
                    "abbinamenti": abb_live,
                    "successi": s_ind,
                    "tentativi": t_ind,
                    "frequenza": f_ind,
                    "applicazioni_vincenti_dettagliate": applicazioni_vincenti_ind
                })

            if len(metodi_validi_per_analisi_def) == 2:
                # ... (calcolo copertura combinata come prima) ...
                self._log_to_gui("\n--- ANALISI PERFORMANCE COMBINATA METODI COMPLESSI BASE ---")
                s_comb_mc, t_comb_mc, f_comb_mc = analizza_copertura_ambate_previste_multiple(
                    storico_per_analisi, metodi_validi_per_analisi_def, ruote_gioco,
                    lookahead, indice_mese, self._log_to_gui
                )
                if t_comb_mc > 0:
                    self._log_to_gui(f"  Successi Combinati (almeno un'ambata vincente): {s_comb_mc}")
                    self._log_to_gui(f"  Tentativi Combinati (almeno un metodo applicabile): {t_comb_mc}")
                    self._log_to_gui(f"  Frequenza di Copertura Combinata Metodi Complessi: {f_comb_mc:.2%}")
                    info_copertura_combinata_mc = {
                        "successi": s_comb_mc, "tentativi": t_comb_mc, "frequenza": f_comb_mc,
                        "num_metodi_combinati": len(metodi_validi_per_analisi_def)
                    }
                else: self._log_to_gui("  Nessun tentativo combinato applicabile.")


            self.master.config(cursor="")
            if not lista_previsioni_popup_mc_vis:
                messagebox.showinfo("Analisi Metodi Complessi", "Nessun metodo complesso valido ha prodotto una previsione popup.")
            else:
                ruote_gioco_str_popup = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                self.mostra_popup_previsione(
                   titolo_popup="Previsioni Metodi Complessi Base",
                   ruote_gioco_str=ruote_gioco_str_popup,
                   lista_previsioni_dettagliate=lista_previsioni_popup_mc_vis,
                   copertura_combinata_info=info_copertura_combinata_mc,
                   data_riferimento_previsione_str_comune=data_riferimento_comune_popup,
                   metodi_grezzi_per_salvataggio=metodi_grezzi_per_popup_salvataggio 
                )
            self._log_to_gui("\n--- Analisi Metodi Complessi Base Completata ---")
        except Exception as e:
             self.master.config(cursor=""); messagebox.showerror("Errore Analisi", f"Errore analisi metodi complessi: {e}"); self._log_to_gui(f"ERRORE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def _calcola_previsione_e_abbinamenti_metodo_complesso(self, storico_attuale, definizione_metodo, ruote_gioco, data_riferimento_str, nome_metodo_log="Metodo"):
        # ... (corpo della funzione invariato) ...
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
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA CORRETTORE OTTIMALE\n" + "="*50)
        metodo_1_def_corr_base = self.definizione_metodo_complesso_attuale 
        metodo_2_def_corr_base = self.definizione_metodo_complesso_attuale_2
        # ... (validazione e logica come prima) ...
        err_msg_corr = []
        almeno_un_metodo_base_valido_per_correttore = False
        metodo_1_valido_def = [] 
        metodo_2_valido_def = []

        if metodo_1_def_corr_base:
            if metodo_1_def_corr_base[-1].get('operazione_successiva') == '=':
                almeno_un_metodo_base_valido_per_correttore = True
                metodo_1_valido_def = metodo_1_def_corr_base
            else: err_msg_corr.append("Metodo Base 1 non terminato con '='.")
        if metodo_2_def_corr_base:
            if metodo_2_def_corr_base[-1].get('operazione_successiva') == '=':
                almeno_un_metodo_base_valido_per_correttore = True
                metodo_2_valido_def = metodo_2_def_corr_base
            else: err_msg_corr.append("Metodo Base 2 non terminato con '='.")
        
        if err_msg_corr: self._log_to_gui("AVVISO VALIDAZIONE METODI PER CORRETTORE:\n" + "\n".join(err_msg_corr))
        if not almeno_un_metodo_base_valido_per_correttore:
            messagebox.showerror("Errore Input", "Definire almeno un Metodo Base valido (terminato con '=') per cercare un correttore.")
            self._log_to_gui("ERRORE: Nessun Metodo Base valido per la ricerca correttore."); return

        storico_per_analisi = self._carica_e_valida_storico_comune(usa_filtri_data_globali=True)
        if not storico_per_analisi: return
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return

        min_tentativi_correttore_cfg = self.corr_cfg_min_tentativi.get()
        # ... (log parametri come prima) ...
        cerca_fisso_s = self.corr_cfg_cerca_fisso_semplice.get()
        cerca_estratto_s = self.corr_cfg_cerca_estratto_semplice.get()
        cerca_diff_ef = self.corr_cfg_cerca_diff_estr_fisso.get()
        cerca_diff_ee = self.corr_cfg_cerca_diff_estr_estr.get()
        cerca_mult_ef = self.corr_cfg_cerca_mult_estr_fisso.get()
        cerca_mult_ee = self.corr_cfg_cerca_mult_estr_estr.get()
        tipi_corr_log = []
        if cerca_fisso_s: tipi_corr_log.append("Fisso Singolo")
        if cerca_estratto_s: tipi_corr_log.append("Estratto Singolo")
        if cerca_diff_ef: tipi_corr_log.append("Estratto-Fisso (Diff)")
        if cerca_diff_ee: tipi_corr_log.append("Estratto-Estratto (Diff)")
        if cerca_mult_ef: tipi_corr_log.append("Estratto*Fisso (Mult)")
        if cerca_mult_ee: tipi_corr_log.append("Estratto*Estratto (Mult)")
        self._log_to_gui("Parametri Ricerca Correttore:")
        if metodo_1_valido_def: self._log_to_gui(f"  Metodo Base 1: {''.join(self._format_componente_per_display(c) for c in metodo_1_valido_def)}")
        if metodo_2_valido_def: self._log_to_gui(f"  Metodo Base 2: {''.join(self._format_componente_per_display(c) for c in metodo_2_valido_def)}")
        self._log_to_gui(f"  Opzioni Gioco: Ruote: {', '.join(ruote_gioco)}, Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        self._log_to_gui(f"  Min. Tentativi Correttore: {min_tentativi_correttore_cfg}")
        self._log_to_gui(f"  Tipi Correttore Selezionati: {', '.join(tipi_corr_log) if tipi_corr_log else 'Nessuno (controllare impostazioni)'}")


        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati_correttori_list = trova_miglior_correttore_per_metodo_complesso(
                storico_per_analisi,
                metodo_1_valido_def if metodo_1_valido_def else None,
                metodo_2_valido_def if metodo_2_valido_def else None,
                cerca_fisso_s, cerca_estratto_s, cerca_diff_ef, cerca_diff_ee,
                cerca_mult_ef, cerca_mult_ee,
                ruote_gioco, lookahead, indice_mese,
                min_tentativi_correttore_cfg, app_logger=self._log_to_gui,
            )
            
            self._log_to_gui("\n\n--- RISULTATI RICERCA CORRETTORI (LOG COMPLETO) ---")
            if not risultati_correttori_list:
                self._log_to_gui("Nessun correttore valido trovato che migliori il benchmark dei metodi base.")
                self.master.config(cursor="")
                messagebox.showinfo("Ricerca Correttore", "Nessun correttore valido trovato che migliori il benchmark.")
            else:
                miglior_risultato_correttore = risultati_correttori_list[0]
                lista_previsioni_popup_corr_vis = [] 
                metodi_grezzi_corretti_per_salvataggio = []
                
                data_riferimento_comune_popup_corr = storico_per_analisi[-1]['data'].strftime('%d/%m/%Y') if storico_per_analisi else None

                info_correttore_globale_str = (
                    f"Correttore Applicato: {miglior_risultato_correttore['tipo_correttore_descrittivo']} -> {miglior_risultato_correttore['dettaglio_correttore_str']}\n"
                    f"Operazione di Collegamento Base: '{miglior_risultato_correttore['operazione_collegamento_base']}'\n"
                    f"Performance del Metodo/i Corretto/i: {miglior_risultato_correttore['frequenza']:.2%} ({miglior_risultato_correttore['successi']}/{miglior_risultato_correttore['tentativi']} casi)"
                )
                self._log_to_gui(f"\n--- MIGLIOR CORRETTORE TROVATO ---")
                log_info_correttore_indentato = info_correttore_globale_str.replace('\n', '\n  ')
                self._log_to_gui(f"  {log_info_correttore_indentato}")


                if miglior_risultato_correttore.get('def_metodo_esteso_1'):
                    met1_est_def = miglior_risultato_correttore['def_metodo_esteso_1']
                    met1_est_str = "".join(self._format_componente_per_display(comp) for comp in met1_est_def)
                    self._log_to_gui(f"\n  Dettagli Metodo 1 Corretto:")
                    self._log_to_gui(f"    Definizione: {met1_est_str}")
                    ambata1_corr_live, abb1_corr_live = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                        storico_per_analisi, met1_est_def, ruote_gioco, data_riferimento_comune_popup_corr, "Metodo 1 Corretto"
                    )
                    lista_previsioni_popup_corr_vis.append({
                        "titolo_sezione": "--- PREVISIONE METODO 1 CORRETTO ---", "info_metodo_str": met1_est_str,
                        "ambata_prevista": ambata1_corr_live, "abbinamenti_dict": abb1_corr_live,
                        "performance_storica_str": "Vedi performance globale correttore"
                    })
                    metodi_grezzi_corretti_per_salvataggio.append({
                        "tipo_metodo_salvato": "complesso_corretto",
                        "definizione_metodo_base_originale_1": metodo_1_valido_def if metodo_1_valido_def else None,
                        "definizione_metodo_esteso_1": met1_est_def, 
                        "formula_testuale": met1_est_str,
                        "ambata_prevista": ambata1_corr_live, "abbinamenti": abb1_corr_live,
                        "tipo_correttore_descrittivo": miglior_risultato_correttore['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_risultato_correttore['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_risultato_correttore['operazione_collegamento_base'],
                        "successi": miglior_risultato_correttore['successi'],
                        "tentativi": miglior_risultato_correttore['tentativi'],
                        "frequenza": miglior_risultato_correttore['frequenza'],
                    })

                if miglior_risultato_correttore.get('def_metodo_esteso_2'):
                    met2_est_def = miglior_risultato_correttore['def_metodo_esteso_2']
                    met2_est_str = "".join(self._format_componente_per_display(comp) for comp in met2_est_def)
                    # ... (log e calcolo previsione per metodo 2 corretto) ...
                    self._log_to_gui(f"\n  Dettagli Metodo 2 Corretto:")
                    self._log_to_gui(f"    Definizione: {met2_est_str}")
                    ambata2_corr_live, abb2_corr_live = self._calcola_previsione_e_abbinamenti_metodo_complesso(
                        storico_per_analisi, met2_est_def, ruote_gioco, data_riferimento_comune_popup_corr, "Metodo 2 Corretto"
                    )
                    lista_previsioni_popup_corr_vis.append({
                        "titolo_sezione": "--- PREVISIONE METODO 2 CORRETTO ---", "info_metodo_str": met2_est_str,
                        "ambata_prevista": ambata2_corr_live, "abbinamenti_dict": abb2_corr_live,
                        "performance_storica_str": "Vedi performance globale correttore"
                    })
                    metodi_grezzi_corretti_per_salvataggio.append({
                        "tipo_metodo_salvato": "complesso_corretto",
                        "definizione_metodo_base_originale_2": metodo_2_valido_def if metodo_2_valido_def else None,
                        "definizione_metodo_esteso_2": met2_est_def,
                        "formula_testuale": met2_est_str,
                        "ambata_prevista": ambata2_corr_live, "abbinamenti": abb2_corr_live,
                        "tipo_correttore_descrittivo": miglior_risultato_correttore['tipo_correttore_descrittivo'],
                        "dettaglio_correttore_str": miglior_risultato_correttore['dettaglio_correttore_str'],
                        "operazione_collegamento_base": miglior_risultato_correttore['operazione_collegamento_base'],
                        "successi": miglior_risultato_correttore['successi'],
                        "tentativi": miglior_risultato_correttore['tentativi'],
                        "frequenza": miglior_risultato_correttore['frequenza'],
                    })
                
                self.master.config(cursor="")
                if not lista_previsioni_popup_corr_vis:
                     messagebox.showinfo("Ricerca Correttore", "Miglior correttore trovato, ma non ha prodotto previsioni valide.")
                else:
                    ruote_gioco_str_popup = "TUTTE" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                    info_correttore_per_popup_vis = {"testo_introduttivo": info_correttore_globale_str}
                    
                    self.mostra_popup_previsione(
                       titolo_popup="Previsione Metodo/i con Correttore Ottimale",
                       ruote_gioco_str=ruote_gioco_str_popup,
                       lista_previsioni_dettagliate=lista_previsioni_popup_corr_vis,
                       copertura_combinata_info=info_correttore_per_popup_vis, 
                       data_riferimento_previsione_str_comune=data_riferimento_comune_popup_corr,
                       metodi_grezzi_per_salvataggio=metodi_grezzi_corretti_per_salvataggio 
                    )
            self._log_to_gui("\n--- Ricerca Correttore Ottimale Completata ---")
        except Exception as e:
            self.master.config(cursor=""); messagebox.showerror("Errore Ricerca Correttore", f"Errore: {e}"); self._log_to_gui(f"ERRORE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch": self.master.config(cursor="")

    def avvia_verifica_giocata(self):
        # ... (corpo della funzione invariato) ...
        self._log_to_gui("\n" + "="*50 + "\nAVVIO VERIFICA GIOCATA MANUALE\n" + "="*50)
        numeri_str = self.numeri_verifica_var.get()
        try:
            numeri_da_verificare = sorted([int(n.strip()) for n in numeri_str.split(',') if n.strip()])
            if not numeri_da_verificare:
                messagebox.showerror("Errore Input", "Inserisci numeri validi da verificare."); self._log_to_gui("ERRORE: Nessun numero inserito per la verifica."); return
            if not all(1 <= n <= 90 for n in numeri_da_verificare):
                messagebox.showerror("Errore Input", "I numeri da verificare devono essere tra 1 e 90."); self._log_to_gui("ERRORE: Numeri non validi."); return
        except ValueError:
            messagebox.showerror("Errore Input", "Formato numeri non valido. Usa numeri separati da virgola."); self._log_to_gui("ERRORE: Formato numeri non valido."); return
        try:
            data_inizio_ver = self.date_inizio_verifica_entry.get_date()
        except ValueError:
            messagebox.showerror("Errore Input", "Seleziona una data di inizio verifica valida."); self._log_to_gui("ERRORE: Data inizio verifica non selezionata."); return
        colpi_ver = self.colpi_verifica_var.get()
        ruote_gioco_selezionate_ver, _, _ = self._get_parametri_gioco_comuni() 
        if ruote_gioco_selezionate_ver is None: return 
        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati):
            messagebox.showerror("Errore Input", "Seleziona una cartella archivio dati valida."); self._log_to_gui("ERRORE: Cartella dati non valida."); return
        self._log_to_gui(f"Parametri Verifica Manuale (avvio):")
        self._log_to_gui(f"  Numeri da Verificare: {numeri_da_verificare}")
        self._log_to_gui(f"  Data Inizio Verifica: {data_inizio_ver.strftime('%d/%m/%Y')}")
        self._log_to_gui(f"  Numero Colpi: {colpi_ver}")
        self._log_to_gui(f"  Ruote di Gioco: {', '.join(ruote_gioco_selezionate_ver)}")
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            self._log_to_gui("Caricamento storico completo per verifica...")
            storico_per_verifica_effettiva = carica_storico_completo(cartella_dati, app_logger=self._log_to_gui) 
            if not storico_per_verifica_effettiva:
                self.master.config(cursor="")
                messagebox.showinfo("Risultato Verifica", "Nessun dato storico caricato. Impossibile verificare.")
                self._log_to_gui("Nessun dato storico caricato per la verifica.")
                return
            stringa_risultati_popup = verifica_giocata_manuale(
                numeri_da_verificare,
                ruote_gioco_selezionate_ver,
                data_inizio_ver,
                colpi_ver,
                storico_per_verifica_effettiva,
                app_logger=self._log_to_gui 
            )
            self.master.config(cursor="") 
            self.mostra_popup_testo_semplice("Risultati Verifica Giocata", stringa_risultati_popup)
        except Exception as e:
            self.master.config(cursor="")
            messagebox.showerror("Errore Verifica", f"Si è verificato un errore durante la verifica: {e}")
            self._log_to_gui(f"ERRORE CRITICO VERIFICA GIOCATA MANUALE: {e}, {traceback.format_exc()}")
        finally:
            if self.master.cget('cursor') == "watch":
                self.master.config(cursor="")

    def mostra_popup_testo_semplice(self, titolo, contenuto_testo):
        # ... (corpo della funzione invariato) ...
        popup_window = tk.Toplevel(self.master)
        popup_window.title(titolo)
        num_righe = contenuto_testo.count('\n') + 1
        larghezza_stimata = 80 
        altezza_stimata_righe = max(10, min(30, num_righe + 4)) 
        popup_width = larghezza_stimata * 7 
        popup_height = altezza_stimata_righe * 15 
        popup_width = max(400, min(700, popup_width)) 
        popup_height = max(250, min(600, popup_height))
        popup_window.geometry(f"{popup_width}x{popup_height}")
        # popup_window.grab_set()
        popup_window.transient(self.master)
        text_widget = scrolledtext.ScrolledText(popup_window, wrap=tk.WORD, font=("Courier New", 9))
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, contenuto_testo)
        text_widget.config(state=tk.DISABLED)
        close_button = ttk.Button(popup_window, text="Chiudi", command=popup_window.destroy)
        close_button.pack(pady=10)
        self.master.eval(f'tk::PlaceWindow {str(popup_window)} center')
        # popup_window.wait_window()

# --- BLOCCO PRINCIPALE DI ESECUZIONE ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LottoAnalyzerApp(root)
    root.mainloop()