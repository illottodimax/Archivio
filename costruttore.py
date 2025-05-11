
import os
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
from itertools import combinations
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Listbox # Listbox non è usata direttamente qui ma potrebbe esserlo in mc_listbox_componenti
try:
    from tkcalendar import DateEntry
except ImportError:
    messagebox.showerror("Errore Dipendenza", "Il pacchetto 'tkcalendar' non è installato.\nPer favore, installalo con: pip install tkcalendar")
    sys.exit()
import json # Aggiunto per salvataggio/apertura impostazioni

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

# --- FUNZIONI LOGICHE ---
def regola_fuori_90(numero):
    if numero == 0: return 90
    if numero < 0:
        abs_val = abs(numero); resto = abs_val % 90
        return 90 - resto if resto != 0 else 90
    return (numero - 1) % 90 + 1

def parse_riga_estrazione(riga, nome_file_ruota, num_riga):
    try:
        parti = riga.strip().split('\t')
        if len(parti) != 7: return None, None
        data_str = parti[0]
        numeri_str = parti[2:7]
        numeri = sorted([int(n) for n in numeri_str])
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
        print(f"ERRORE INTERNO: Operazione '{operazione_str}' non supportata.")
        return -1, 0, 0, [] 
    op_func = OPERAZIONI[operazione_str]
    successi, tentativi = 0, 0; applicazioni_vincenti = []
    ambata_fissa_del_metodo = -1; prima_applicazione_valida = True
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
        if not estrazione_corrente.get(ruota_calcolo) or len(estrazione_corrente[ruota_calcolo]) <= pos_estratto_calcolo: continue
        numero_base = estrazione_corrente[ruota_calcolo][pos_estratto_calcolo]
        try: valore_operazione = op_func(numero_base, operando_fisso)
        except ZeroDivisionError: continue
        ambata_prevista_corrente = regola_fuori_90(valore_operazione)
        if prima_applicazione_valida:
            ambata_fissa_del_metodo = ambata_prevista_corrente
            prima_applicazione_valida = False
        tentativi += 1; trovato_in_questo_tentativo = False; dettagli_vincita_per_tentativo = []
        for k in range(1, lookahead + 1):
            if i + k >= len(storico): break
            estrazione_futura = storico[i + k]
            for ruota_verifica_effettiva in ruote_gioco_selezionate:
                if ambata_prevista_corrente in estrazione_futura.get(ruota_verifica_effettiva, []):
                    if not trovato_in_questo_tentativo: successi += 1; trovato_in_questo_tentativo = True
                    dettagli_vincita_per_tentativo.append({"ruota_vincita": ruota_verifica_effettiva, "numeri_ruota_vincita": estrazione_futura.get(ruota_verifica_effettiva, []), "data_riscontro": estrazione_futura['data'], "colpo_riscontro": k})
            if trovato_in_questo_tentativo and len(ruote_gioco_selezionate) == 1: break
        if trovato_in_questo_tentativo: applicazioni_vincenti.append({"data_applicazione": estrazione_corrente['data'], "estratto_base": numero_base, "operando": operando_fisso, "operazione": operazione_str, "ambata_prevista": ambata_prevista_corrente, "riscontri": dettagli_vincita_per_tentativo})
    return ambata_fissa_del_metodo, successi, tentativi, applicazioni_vincenti

def analizza_copertura_combinata(storico, top_metodi_info, ruote_gioco_selezionate, lookahead, indice_mese_filtro, app_logger=None):
    date_tentativi_combinati = set(); date_successi_combinati = set()
    op_func_cache = {op_str: OPERAZIONI[op_str] for op_str in OPERAZIONI}
    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro: continue
        ambate_previste_per_questa_estrazione = set(); almeno_un_metodo_applicabile_qui = False
        for metodo_info_dict in top_metodi_info:
            met_details = metodo_info_dict['metodo']
            ruota_calc = met_details['ruota_calcolo']; pos_estr = met_details['pos_estratto_calcolo']
            if not estrazione_corrente.get(ruota_calc) or len(estrazione_corrente[ruota_calc]) <= pos_estr: continue 
            almeno_un_metodo_applicabile_qui = True
            numero_base = estrazione_corrente[ruota_calc][pos_estr]
            op_str = met_details['operazione']; operando = met_details['operando_fisso']
            if op_str not in op_func_cache:
                if app_logger: app_logger(f"WARN: Operazione {op_str} non in cache per analisi combinata."); 
                continue 
            op_func = op_func_cache[op_str] 
            try: valore_operazione = op_func(numero_base, operando)
            except ZeroDivisionError: continue 
            ambata_prevista = regola_fuori_90(valore_operazione)
            ambate_previste_per_questa_estrazione.add(ambata_prevista)
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
    num_successi_combinati = len(date_successi_combinati); num_tentativi_combinati = len(date_tentativi_combinati)
    frequenza_combinata = num_successi_combinati / num_tentativi_combinati if num_tentativi_combinati > 0 else 0
    return num_successi_combinati, num_tentativi_combinati, frequenza_combinata

def trova_migliori_ambate_e_abbinamenti(storico, ruota_calcolo, pos_estratto_calcolo, ruote_gioco_selezionate, max_ambate_output=1, lookahead=1, indice_mese_filtro=None, min_tentativi_per_ambata=10, app_logger=None):
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=flush)
    risultati_ambate_grezzi = []; gioco_su_desc = "su " + ", ".join(ruote_gioco_selezionate) if len(ruote_gioco_selezionate) < len(RUOTE) else "su TUTTE le ruote"
    log_message(f"\nAnalisi metodi per ambata {gioco_su_desc} (da {ruota_calcolo}[pos.{pos_estratto_calcolo+1}]):")
    tot_metodi_testati = len(OPERAZIONI) * 90; metodi_processati = 0
    for op_str in OPERAZIONI:
        for operando in range(1, 91):
            metodi_processati += 1
            ambata_fissa_prodotta, successi, tentativi, applicazioni_vincenti_dett = analizza_metodo_sommativo_base(storico, ruota_calcolo, pos_estratto_calcolo, op_str, operando, ruote_gioco_selezionate, lookahead, indice_mese_filtro)
            if tentativi >= min_tentativi_per_ambata:
                frequenza = successi / tentativi if tentativi > 0 else 0
                risultati_ambate_grezzi.append({"metodo": {"operazione": op_str, "operando_fisso": operando, "ruota_calcolo": ruota_calcolo, "pos_estratto_calcolo": pos_estratto_calcolo}, "ambata_prodotta_dal_metodo": ambata_fissa_prodotta, "successi": successi, "tentativi": tentativi, "frequenza_ambata": frequenza, "applicazioni_vincenti_dettagliate": applicazioni_vincenti_dett})
    log_message("\n  Completata analisi metodi per ambata." + " "*50) 
    risultati_ambate_grezzi.sort(key=lambda x: (x["frequenza_ambata"], x["successi"]), reverse=True)
    top_n_metodi_per_combinata = risultati_ambate_grezzi[:max_ambate_output]
    if len(top_n_metodi_per_combinata) > 1:
        log_message(f"\n--- ANALISI COPERTURA COMBINATA PER TOP {len(top_n_metodi_per_combinata)} METODI ---")
        s_comb, t_comb, f_comb = analizza_copertura_combinata(storico, top_n_metodi_per_combinata, ruote_gioco_selezionate, lookahead, indice_mese_filtro, app_logger)
        if t_comb > 0:
            log_message(f"  Giocando simultaneamente le ambate prodotte dai {len(top_n_metodi_per_combinata)} migliori metodi:")
            log_message(f"  - Successi Complessivi (almeno un'ambata vincente): {s_comb}")
            log_message(f"  - Tentativi Complessivi (almeno un metodo applicabile): {t_comb}")
            log_message(f"  - Frequenza di Copertura Combinata: {f_comb:.2%}")
        else: log_message("  Nessun tentativo combinato applicabile per i metodi selezionati per l'analisi combinata.")
    top_ambate_final_con_abbinamenti = []
    log_message(f"\nAnalisi abbinamenti per le top {min(max_ambate_output, len(risultati_ambate_grezzi))} ambate...")
    for i, res_ambata_grezza in enumerate(risultati_ambate_grezzi[:max_ambate_output]):
        log_message(f"  Analizzando abbinamenti per il metodo {i+1} ({res_ambata_grezza['metodo']['operazione']} {res_ambata_grezza['metodo']['operando_fisso']})...")
        applicazioni_vincenti_per_abbinamenti = res_ambata_grezza["applicazioni_vincenti_dettagliate"]
        res_output_finale = dict(res_ambata_grezza); res_output_finale.pop("applicazioni_vincenti_dettagliate", None)
        if not applicazioni_vincenti_per_abbinamenti:
            log_message("    Nessuna applicazione vincente per questo metodo (per analisi abbinamenti).")
            res_output_finale["ambata_piu_frequente_dal_metodo"] = res_ambata_grezza.get("ambata_prodotta_dal_metodo", "N/D")
            res_output_finale["abbinamenti"] = {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "eventi_abbinamento_analizzati": 0}
            top_ambate_final_con_abbinamenti.append(res_output_finale); continue
        ambata_target_per_abbinamenti = res_ambata_grezza.get("ambata_prodotta_dal_metodo", None)
        if ambata_target_per_abbinamenti is None or ambata_target_per_abbinamenti == -1:
            log_message("    Impossibile determinare un'ambata consistente per questo metodo (per analisi abbinamenti).")
            res_output_finale["abbinamenti"] = {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "eventi_abbinamento_analizzati": 0}
            top_ambate_final_con_abbinamenti.append(res_output_finale); continue
        abbinamenti_calcolati = analizza_abbinamenti_per_metodo_complesso(applicazioni_vincenti_per_abbinamenti, ambata_target_per_abbinamenti, None)
        res_output_finale["ambata_piu_frequente_dal_metodo"] = ambata_target_per_abbinamenti
        res_output_finale["abbinamenti"] = abbinamenti_calcolati
        top_ambate_final_con_abbinamenti.append(res_output_finale)
    log_message("  Completata analisi abbinamenti.")
    return top_ambate_final_con_abbinamenti

def calcola_valore_metodo_complesso(estrazione_corrente, definizione_metodo, app_logger=None):
    if not definizione_metodo:
        if app_logger: app_logger("Errore: Definizione metodo complesso vuota.")
        return None
    valore_accumulato = 0; operazione_aritmetica_pendente = None
    for i, comp in enumerate(definizione_metodo):
        termine_attuale = 0
        if comp['tipo_termine'] == 'estratto':
            ruota_nome = comp['ruota']; pos = comp['posizione']
            if not estrazione_corrente.get(ruota_nome) or len(estrazione_corrente[ruota_nome]) <= pos: return None 
            termine_attuale = estrazione_corrente[ruota_nome][pos]
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
    successi = 0
    tentativi = 0
    applicazioni_vincenti = []
    ambate_prodotte_dal_metodo_corrente = Counter() 

    for i in range(len(storico) - lookahead):
        estrazione_corrente = storico[i]
        if indice_mese_filtro and estrazione_corrente['indice_mese'] != indice_mese_filtro:
            continue

        valore_calcolato_raw = calcola_valore_metodo_complesso(estrazione_corrente, definizione_metodo, app_logger)

        if valore_calcolato_raw is None: 
            continue 
        
        ambata_prevista_per_questa_applicazione = regola_fuori_90(valore_calcolato_raw)
        
        ambate_prodotte_dal_metodo_corrente[ambata_prevista_per_questa_applicazione] += 1
        
        tentativi += 1 
        
        trovato_successo_per_questa_applicazione = False
        dettagli_vincita_per_tentativo = []

        for k_lookahead in range(1, lookahead + 1):
            if i + k_lookahead >= len(storico): 
                break 
            
            estrazione_futura = storico[i + k_lookahead]
            
            for ruota_verifica in ruote_gioco_selezionate:
                if ambata_prevista_per_questa_applicazione in estrazione_futura.get(ruota_verifica, []):
                    if not trovato_successo_per_questa_applicazione:
                        successi += 1 
                        trovato_successo_per_questa_applicazione = True
                    
                    dettagli_vincita_per_tentativo.append({
                        "ruota_vincita": ruota_verifica,
                        "numeri_ruota_vincita": estrazione_futura.get(ruota_verifica, []),
                        "data_riscontro": estrazione_futura['data'],
                        "colpo_riscontro": k_lookahead
                    })
            
            if trovato_successo_per_questa_applicazione and len(ruote_gioco_selezionate) == 1:
                break 
        
        if trovato_successo_per_questa_applicazione:
            applicazioni_vincenti.append({
                "data_applicazione": estrazione_corrente['data'],
                "valore_calcolato_raw": valore_calcolato_raw,
                "ambata_prevista": ambata_prevista_per_questa_applicazione, 
                "riscontri": dettagli_vincita_per_tentativo
            })
            
    ambata_piu_frequente_del_metodo = None
    if ambate_prodotte_dal_metodo_corrente: 
        ambata_piu_frequente_del_metodo = ambate_prodotte_dal_metodo_corrente.most_common(1)[0][0]
            
    return ambata_piu_frequente_del_metodo, successi, tentativi, applicazioni_vincenti

def analizza_abbinamenti_per_metodo_complesso(applicazioni_vincenti_metodo_complesso, ambata_target_del_metodo, app_logger=None): 
    if not applicazioni_vincenti_metodo_complesso: 
        return {"ambo": [], "terno": [], "quaterna": [], "cinquina": [], "eventi_abbinamento_analizzati": 0}
    abbinamenti_per_ambo = Counter(); abbinamenti_per_terno = Counter(); abbinamenti_per_quaterna = Counter(); abbinamenti_per_cinquina = Counter()
    conteggio_eventi_per_abbinamenti = 0
    for app_vinc in applicazioni_vincenti_metodo_complesso:
        if app_vinc['ambata_prevista'] != ambata_target_del_metodo: 
            continue 
        for riscontro_info in app_vinc["riscontri"]:
            conteggio_eventi_per_abbinamenti += 1
            numeri_usciti_su_ruota_vincita = [n for n in riscontro_info["numeri_ruota_vincita"] if n != ambata_target_del_metodo]
            for num_abbinato in numeri_usciti_su_ruota_vincita: abbinamenti_per_ambo[num_abbinato] += 1
            if len(numeri_usciti_su_ruota_vincita) >= 2:
                for combo_2 in combinations(sorted(numeri_usciti_su_ruota_vincita), 2): abbinamenti_per_terno[combo_2] += 1
            if len(numeri_usciti_su_ruota_vincita) >= 3:
                for combo_3 in combinations(sorted(numeri_usciti_su_ruota_vincita), 3): abbinamenti_per_quaterna[combo_3] += 1
            if len(numeri_usciti_su_ruota_vincita) >= 4:
                for combo_4 in combinations(sorted(numeri_usciti_su_ruota_vincita), 4): abbinamenti_per_cinquina[combo_4] += 1
    return {
        "ambo": [{"numeri": [ab[0]], "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_ambo.most_common(5)],
        "terno": [{"numeri": list(ab[0]), "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_terno.most_common(5)],
        "quaterna": [{"numeri": list(ab[0]), "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_quaterna.most_common(5)],
        "cinquina": [{"numeri": list(ab[0]), "frequenza": ab[1]/conteggio_eventi_per_abbinamenti if conteggio_eventi_per_abbinamenti else 0, "conteggio": ab[1]} for ab in abbinamenti_per_cinquina.most_common(5)],
        "eventi_abbinamento_analizzati": conteggio_eventi_per_abbinamenti 
    }

def verifica_giocata_manuale(numeri_da_giocare, ruote_selezionate, data_inizio_controllo, num_colpi_controllo, storico_completo, app_logger=None):
    def log_message(msg):
        if app_logger: app_logger(msg)
        else: print(msg)
    log_message(f"\n--- VERIFICA GIOCATA MANUALE ---")
    log_message(f"Numeri da giocare: {numeri_da_giocare}")
    log_message(f"Ruote selezionate: {', '.join(ruote_selezionate)}")
    log_message(f"Data inizio controllo: {data_inizio_controllo}")
    log_message(f"Numero colpi controllo: {num_colpi_controllo}")
    if not numeri_da_giocare: log_message("ERRORE: Nessun numero inserito per la verifica."); return
    if not ruote_selezionate: log_message("ERRORE: Nessuna ruota selezionata per la verifica."); return
    if not data_inizio_controllo: log_message("ERRORE: Data inizio controllo non specificata."); return
    indice_partenza = -1
    for i, estrazione in enumerate(storico_completo):
        if estrazione['data'] >= data_inizio_controllo: indice_partenza = i; break
    if indice_partenza == -1: log_message(f"Nessuna estrazione trovata a partire dal {data_inizio_controllo}. Impossibile verificare."); return
    log_message(f"Controllo a partire dall'estrazione del {storico_completo[indice_partenza]['data']}:")
    trovato_esito_globale = False
    for colpo in range(num_colpi_controllo):
        indice_estrazione_corrente = indice_partenza + colpo
        if indice_estrazione_corrente >= len(storico_completo): log_message(f"Fine storico raggiunto al colpo {colpo+1} (su {num_colpi_controllo}). Controllo interrotto."); break
        estrazione_controllo = storico_completo[indice_estrazione_corrente]
        log_message(f"  Colpo {colpo + 1} (Data: {estrazione_controllo['data']}):")
        esito_trovato_in_questo_colpo_su_ruota_secca = False
        for ruota in ruote_selezionate:
            numeri_estratti_ruota = estrazione_controllo.get(ruota, [])
            if not numeri_estratti_ruota: continue
            vincenti_ambata = [n for n in numeri_da_giocare if n in numeri_estratti_ruota]
            if vincenti_ambata:
                log_message(f"    >> AMBATA SU {ruota.upper()}! Numeri usciti: {vincenti_ambata}")
                trovato_esito_globale = True
                if len(ruote_selezionate) == 1: 
                    esito_trovato_in_questo_colpo_su_ruota_secca = True
            if len(numeri_da_giocare) >= 2:
                numeri_giocati_presenti_nella_ruota = [n for n in numeri_da_giocare if n in numeri_estratti_ruota]
                num_corrispondenze = len(numeri_giocati_presenti_nella_ruota)
                sorte_trovata_in_questo_passaggio = False 
                if num_corrispondenze == 2: 
                    log_message(f"    >> AMBO SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}")
                    sorte_trovata_in_questo_passaggio = True
                elif num_corrispondenze == 3: 
                    log_message(f"    >> TERNO SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}")
                    sorte_trovata_in_questo_passaggio = True
                elif num_corrispondenze == 4: 
                    log_message(f"    >> QUATERNA SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}")
                    sorte_trovata_in_questo_passaggio = True
                elif num_corrispondenze == 5 and len(numeri_da_giocare) >= 5 : 
                    log_message(f"    >> CINQUINA SU {ruota.upper()}! Numeri: {sorted(numeri_giocati_presenti_nella_ruota)}")
                    sorte_trovata_in_questo_passaggio = True
                if sorte_trovata_in_questo_passaggio:
                    trovato_esito_globale = True
                    if len(ruote_selezionate) == 1:
                        esito_trovato_in_questo_colpo_su_ruota_secca = True
        if esito_trovato_in_questo_colpo_su_ruota_secca: 
            log_message(f"--- Esito trovato su ruota secca al colpo {colpo + 1}. Interruzione verifica colpi successivi. ---"); break 
    if not trovato_esito_globale: 
        log_message(f"\nNessun esito trovato per i numeri {numeri_da_giocare} entro {num_colpi_controllo} colpi.")
    log_message("--- Fine Verifica Giocata Manuale ---")

def costruisci_metodo_esteso(metodo_base_originale, operazione_collegamento, termine_correttore):
    if not metodo_base_originale or metodo_base_originale[-1]['operazione_successiva'] != '=':
        raise ValueError("Metodo base non valido o non terminato con '='.")
    metodo_esteso = [dict(comp) for comp in metodo_base_originale] 
    metodo_esteso[-1]['operazione_successiva'] = operazione_collegamento
    componente_correttore = dict(termine_correttore) 
    componente_correttore['operazione_successiva'] = '='
    metodo_esteso.append(componente_correttore)
    return metodo_esteso

def trova_miglior_correttore_per_metodo_complesso(storico, definizione_metodo_base, 
                                                  cerca_fisso, cerca_estratto, 
                                                  ruote_gioco_selezionate, lookahead, 
                                                  indice_mese_filtro, min_tentativi_per_correttore, 
                                                  app_logger=None):
    def log_message(msg, end='\n', flush=False):
        if app_logger: app_logger(msg, end=end, flush=flush)

    log_message("\nInizio ricerca correttori per metodo complesso...")
    risultati_correttori = []
    
    operazioni_da_testare_per_collegamento = ['+', '-', '*'] 

    if cerca_fisso:
        log_message("  Ricerca fissi correttori...")
        tot_fissi_da_testare = 90 * len(operazioni_da_testare_per_collegamento)
        fissi_processati_count = 0
        for op_collegamento in operazioni_da_testare_per_collegamento:
            for valore_fisso_correttore in range(1, 91):
                fissi_processati_count += 1
                if fissi_processati_count % 30 == 0 or fissi_processati_count == tot_fissi_da_testare: 
                    log_message(f"    Testando fisso correttore {fissi_processati_count}/{tot_fissi_da_testare}...", end='\r', flush=True)

                termine_correttore_attuale = {'tipo_termine': 'fisso', 'valore_fisso': valore_fisso_correttore}
                
                try:
                    definizione_metodo_esteso = costruisci_metodo_esteso(definizione_metodo_base, op_collegamento, termine_correttore_attuale)
                except ValueError as e:
                    if app_logger: app_logger(f"WARN: Errore costruzione metodo esteso con fisso {valore_fisso_correttore} op {op_collegamento}: {e}")
                    continue

                # MODIFICA: Recupera applicazioni_vincenti_estese
                ambata_prodotta_estesa, successi, tentativi, applicazioni_vincenti_estese = analizza_metodo_complesso_specifico(
                    storico, definizione_metodo_esteso, ruote_gioco_selezionate, 
                    lookahead, indice_mese_filtro, app_logger=None 
                )

                if tentativi >= min_tentativi_per_correttore and successi > 0: 
                    frequenza = successi / tentativi
                    risultati_correttori.append({
                        'metodo_esteso_def': definizione_metodo_esteso,
                        'tipo_correttore': 'Fisso',
                        'dettaglio_correttore_str': str(valore_fisso_correttore),
                        'operazione_collegamento': op_collegamento,
                        'ambata_risultante_metodo_esteso': ambata_prodotta_estesa,
                        'successi': successi,
                        'tentativi': tentativi,
                        'frequenza': frequenza,
                        'applicazioni_vincenti_metodo_esteso': applicazioni_vincenti_estese # AGGIUNTO
                    })
        log_message("\n    Completata ricerca fissi correttori." + " "*50) 

    if cerca_estratto:
        log_message("  Ricerca estratti correttori...")
        tot_estratti_da_testare = len(RUOTE) * 5 * len(operazioni_da_testare_per_collegamento)
        estratti_processati_count = 0
        for op_collegamento in operazioni_da_testare_per_collegamento:
            for ruota_correttore in RUOTE:
                for pos_correttore in range(5): 
                    estratti_processati_count +=1
                    if estratti_processati_count % 50 == 0 or estratti_processati_count == tot_estratti_da_testare: 
                        log_message(f"    Testando estratto correttore {estratti_processati_count}/{tot_estratti_da_testare}...", end='\r', flush=True)

                    termine_correttore_attuale = {'tipo_termine': 'estratto', 'ruota': ruota_correttore, 'posizione': pos_correttore}
                    try:
                        definizione_metodo_esteso = costruisci_metodo_esteso(definizione_metodo_base, op_collegamento, termine_correttore_attuale)
                    except ValueError as e:
                        if app_logger: app_logger(f"WARN: Errore costruzione metodo esteso con estratto {ruota_correttore}[{pos_correttore+1}] op {op_collegamento}: {e}")
                        continue
                        
                    # MODIFICA: Recupera applicazioni_vincenti_estese
                    ambata_prodotta_estesa, successi, tentativi, applicazioni_vincenti_estese = analizza_metodo_complesso_specifico(
                        storico, definizione_metodo_esteso, ruote_gioco_selezionate, 
                        lookahead, indice_mese_filtro, app_logger=None 
                    )

                    if tentativi >= min_tentativi_per_correttore and successi > 0:
                        frequenza = successi / tentativi
                        risultati_correttori.append({
                            'metodo_esteso_def': definizione_metodo_esteso,
                            'tipo_correttore': 'Estratto',
                            'dettaglio_correttore_str': f"{ruota_correttore}[{pos_correttore+1}]",
                            'operazione_collegamento': op_collegamento,
                            'ambata_risultante_metodo_esteso': ambata_prodotta_estesa, 
                            'successi': successi,
                            'tentativi': tentativi,
                            'frequenza': frequenza,
                            'applicazioni_vincenti_metodo_esteso': applicazioni_vincenti_estese # AGGIUNTO
                        })
        log_message("\n    Completata ricerca estratti correttori." + " "*50) 
           
    risultati_correttori.sort(key=lambda x: (x['frequenza'], x['successi']), reverse=True)
    log_message("Ricerca correttori per metodo complesso completata.")
    return risultati_correttori

# --- CLASSE PER LA GUI ---
class LottoAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Costruttore Metodi Lotto Avanzato")
        master.geometry("850x700") # Aumentata un po' l'altezza per più output

        self.cartella_dati_var = tk.StringVar()
        self.ruota_calcolo_var = tk.StringVar(value=RUOTE[0]) 
        self.posizione_estratto_var = tk.IntVar(value=1)
        self.ruote_gioco_vars = {ruota: tk.BooleanVar() for ruota in RUOTE}
        self.tutte_le_ruote_var = tk.BooleanVar(value=True) 
        self.lookahead_var = tk.IntVar(value=3) 
        self.indice_mese_var = tk.StringVar() 
        self.num_ambate_var = tk.IntVar(value=1) 
        self.min_tentativi_var = tk.IntVar(value=10)
        self.numeri_verifica_var = tk.StringVar() 
        self.colpi_verifica_var = tk.IntVar(value=9)
        self.storico_caricato = None 
        self.definizione_metodo_complesso_attuale = []
        self.mc_tipo_termine_var = tk.StringVar(value="estratto")
        self.mc_ruota_var = tk.StringVar(value=RUOTE[0])
        self.mc_posizione_var = tk.IntVar(value=1)
        self.mc_valore_fisso_var = tk.IntVar(value=1)
        self.mc_operazione_var = tk.StringVar(value='+')
        self.active_tab_ruote_checkbox_widgets = []

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
        tk.Button(cartella_frame, text="Nessuna", command=lambda: self.date_fine_entry_analisi.delete(0, tk.END)).grid(row=current_row_cc, column=2, sticky="w", padx=5, pady=2)

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
        
        output_controls_frame = ttk.Frame(master)
        output_controls_frame.pack(fill=tk.X, padx=10, pady=(5,0)) 
        output_label = tk.Label(output_controls_frame, text="Log e Risultati:", font=("Helvetica", 10, "bold"))
        output_label.pack(side=tk.LEFT, anchor="w") 
        self.btn_pulisci_output = tk.Button(output_controls_frame, text="Pulisci Output", command=self._clear_output_area_manual)
        self.btn_pulisci_output.pack(side=tk.LEFT, padx=10) 
        
        self.output_text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=90, height=25, font=("Courier New", 9)) # Aumentata altezza
        self.output_text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.output_text_area.config(state=tk.DISABLED)

        if self.notebook.tabs(): 
            self.master.after(100, lambda: self.on_tab_changed(None))

    def crea_gui_controlli_comuni(self, parent_frame_main_tab):
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
                                    if widget_in_frame.cget("text") in RUOTE: temp_widget_list.append(widget_in_frame)
                            if is_target_ruote_frame: self.active_tab_ruote_checkbox_widgets = temp_widget_list; break 
                    if self.active_tab_ruote_checkbox_widgets or (is_target_ruote_frame if 'is_target_ruote_frame' in locals() else False): break 
        except Exception as e: print(f"Errore in on_tab_changed: {e}")
        self.toggle_tutte_ruote()

    def toggle_tutte_ruote(self):
        stato_tutte_var = self.tutte_le_ruote_var.get()
        for nome_ruota in self.ruote_gioco_vars: self.ruote_gioco_vars[nome_ruota].set(stato_tutte_var)
        nuovo_stato_widget_figli = tk.DISABLED if stato_tutte_var else tk.NORMAL
        for cb_widget in self.active_tab_ruote_checkbox_widgets: cb_widget.config(state=nuovo_stato_widget_figli)

    def update_tutte_le_ruote_status(self):
        tutti_figli_selezionati = all(self.ruote_gioco_vars[ruota].get() for ruota in RUOTE)
        if self.tutte_le_ruote_var.get() != tutti_figli_selezionati: self.tutte_le_ruote_var.set(tutti_figli_selezionati)
    
    def crea_gui_metodi_semplici(self, parent_tab):
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
        main_frame = ttk.Frame(parent_tab, padding="5")
        main_frame.pack(expand=True, fill='both')
        self.crea_gui_controlli_comuni(main_frame)
        constructor_frame = ttk.LabelFrame(main_frame, text="Costruttore Metodo Complesso Base", padding="10")
        constructor_frame.pack(padx=10, pady=10, fill=tk.X)
        save_load_mc_frame = ttk.Frame(constructor_frame)
        save_load_mc_frame.pack(fill=tk.X, pady=(0,5), anchor='nw') 
        tk.Button(save_load_mc_frame, text="Salva Metodo Compl.", command=self.salva_metodo_complesso).pack(side=tk.LEFT, padx=5)
        tk.Button(save_load_mc_frame, text="Apri Metodo Compl.", command=self.apri_metodo_complesso).pack(side=tk.LEFT, padx=5)
        tk.Label(constructor_frame, text="Metodo Attuale:").pack(anchor="w")
        self.mc_listbox_componenti = tk.Listbox(constructor_frame, height=3, width=70) 
        self.mc_listbox_componenti.pack(fill=tk.X, expand=True, pady=(0,5))
        add_comp_frame = ttk.Frame(constructor_frame); add_comp_frame.pack(fill=tk.X)
        tk.Radiobutton(add_comp_frame, text="Estratto", variable=self.mc_tipo_termine_var, value="estratto", command=self._update_mc_input_state).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(add_comp_frame, text="Fisso", variable=self.mc_tipo_termine_var, value="fisso", command=self._update_mc_input_state).grid(row=0, column=1, sticky="w")
        self.mc_ruota_label = tk.Label(add_comp_frame, text="Ruota:"); self.mc_ruota_label.grid(row=1, column=0, sticky="e", padx=2)
        self.mc_ruota_combo = ttk.Combobox(add_comp_frame, textvariable=self.mc_ruota_var, values=RUOTE, state="readonly", width=10); self.mc_ruota_combo.grid(row=1, column=1, sticky="w", padx=2)
        self.mc_pos_label = tk.Label(add_comp_frame, text="Pos. (1-5):"); self.mc_pos_label.grid(row=1, column=2, sticky="e", padx=2)
        self.mc_pos_spinbox = tk.Spinbox(add_comp_frame, from_=1, to=5, textvariable=self.mc_posizione_var, width=4, state="readonly"); self.mc_pos_spinbox.grid(row=1, column=3, sticky="w", padx=2)
        self.mc_fisso_label = tk.Label(add_comp_frame, text="Valore Fisso (1-90):"); self.mc_fisso_label.grid(row=2, column=0, sticky="e", padx=2)
        self.mc_fisso_spinbox = tk.Spinbox(add_comp_frame, from_=1, to=90, textvariable=self.mc_valore_fisso_var, width=4, state="readonly"); self.mc_fisso_spinbox.grid(row=2, column=1, sticky="w", padx=2)
        tk.Label(add_comp_frame, text="Op. Successiva:").grid(row=3, column=0, sticky="e", padx=2)
        self.mc_op_combo = ttk.Combobox(add_comp_frame, textvariable=self.mc_operazione_var, values=list(OPERAZIONI_COMPLESSE.keys()) + ['='], state="readonly", width=4); self.mc_op_combo.grid(row=3, column=1, sticky="w", padx=2); self.mc_op_combo.set('+')
        tk.Button(add_comp_frame, text="Aggiungi Termine", command=self.aggiungi_componente_metodo).grid(row=3, column=2, columnspan=2, pady=5, padx=5, sticky="ew")
        buttons_frame = ttk.Frame(constructor_frame); buttons_frame.pack(fill=tk.X, pady=5)
        tk.Button(buttons_frame, text="Rimuovi Ultimo", command=self.rimuovi_ultimo_componente_metodo).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Pulisci Metodo", command=self.pulisci_metodo_complesso).pack(side=tk.LEFT, padx=5)
        self._update_mc_input_state()
        self._refresh_mc_listbox() 
        tk.Button(main_frame, text="Analizza Metodo Base Definito", command=self.avvia_analisi_metodo_complesso, font=("Helvetica", 10, "bold"), bg="lightcoral" ).pack(pady=(5,0), ipady=2, fill=tk.X, padx=10)
        correttore_frame = ttk.LabelFrame(main_frame, text="Ricerca Correttore per Metodo Base", padding="10")
        correttore_frame.pack(padx=10, pady=10, fill=tk.X)
        tk.Button(correttore_frame, text="Trova Correttore Ottimale", command=self.avvia_ricerca_correttore, font=("Helvetica", 11, "bold"), bg="gold" ).pack(pady=5, ipady=3)

    def crea_gui_verifica_manuale(self, parent_tab):
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

    def _update_mc_input_state(self):
        if self.mc_tipo_termine_var.get() == "estratto":
            self.mc_ruota_combo.config(state="readonly"); self.mc_pos_spinbox.config(state="readonly"); self.mc_fisso_spinbox.config(state="disabled")
            self.mc_ruota_label.config(state="normal"); self.mc_pos_label.config(state="normal"); self.mc_fisso_label.config(state="disabled")
        else:
            self.mc_ruota_combo.config(state="disabled"); self.mc_pos_spinbox.config(state="disabled"); self.mc_fisso_spinbox.config(state="readonly")
            self.mc_ruota_label.config(state="disabled"); self.mc_pos_label.config(state="disabled"); self.mc_fisso_label.config(state="normal")
            
    def _format_componente_per_display(self, componente):
        op_succ = componente['operazione_successiva']; op_str = f" {op_succ} " if op_succ and op_succ != '=' else ""
        if componente['tipo_termine'] == 'estratto': return f"{componente['ruota']}[{componente['posizione']+1}]{op_str}"
        elif componente['tipo_termine'] == 'fisso': return f"Fisso({componente['valore_fisso']}){op_str}"
        return "ERRORE_COMP"

    def _refresh_mc_listbox(self):
        if hasattr(self, 'mc_listbox_componenti') and self.mc_listbox_componenti.winfo_exists(): 
            self.mc_listbox_componenti.delete(0, tk.END); display_str = ""
            for comp in self.definizione_metodo_complesso_attuale: display_str += self._format_componente_per_display(comp)
            self.mc_listbox_componenti.insert(tk.END, display_str if display_str else "Nessun componente definito.")

    def aggiungi_componente_metodo(self):
        tipo = self.mc_tipo_termine_var.get(); op_succ = self.mc_operazione_var.get()
        if self.definizione_metodo_complesso_attuale and self.definizione_metodo_complesso_attuale[-1]['operazione_successiva'] == '=':
            messagebox.showwarning("Costruzione Metodo", "Il metodo è già terminato con '='."); return
        componente = {'tipo_termine': tipo, 'operazione_successiva': op_succ}
        if tipo == 'estratto': componente['ruota'] = self.mc_ruota_var.get(); componente['posizione'] = self.mc_posizione_var.get() - 1
        else: 
            val_fisso = self.mc_valore_fisso_var.get()
            if not (1 <= val_fisso <= 90): messagebox.showerror("Errore Input", "Valore fisso deve essere tra 1 e 90."); return
            componente['valore_fisso'] = val_fisso
        self.definizione_metodo_complesso_attuale.append(componente); self._refresh_mc_listbox()

    def rimuovi_ultimo_componente_metodo(self):
        if self.definizione_metodo_complesso_attuale: self.definizione_metodo_complesso_attuale.pop(); self._refresh_mc_listbox()

    def pulisci_metodo_complesso(self):
        self.definizione_metodo_complesso_attuale.clear(); self._refresh_mc_listbox()
    
    def _log_to_gui(self, message, end='\n', flush=False):
        self.output_text_area.config(state=tk.NORMAL)
        self.output_text_area.insert(tk.END, message + end)
        if flush or "\r" in message: self.output_text_area.see(tk.END); self.output_text_area.update_idletasks()
        self.output_text_area.config(state=tk.DISABLED); self.output_text_area.see(tk.END)

    def seleziona_cartella(self):
        cartella = filedialog.askdirectory(title="Seleziona cartella archivi")
        if cartella: self.cartella_dati_var.set(cartella)
        
    def _clear_output_area_manual(self): 
        self.output_text_area.config(state=tk.NORMAL)
        self.output_text_area.delete('1.0', tk.END)
        self.output_text_area.config(state=tk.DISABLED)
        print("DEBUG: Area output pulita manually.") 

    def _carica_e_valida_storico_comune(self):
        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati):
            messagebox.showerror("Errore Input", "Seleziona una cartella archivio dati valida."); self._log_to_gui("ERRORE: Cartella dati non valida."); return None
        try: data_inizio_storico = self.date_inizio_entry_analisi.get_date()
        except ValueError: data_inizio_storico = None 
        try: data_fine_storico = self.date_fine_entry_analisi.get_date()
        except ValueError: data_fine_storico = None
        if data_inizio_storico and data_fine_storico and data_fine_storico < data_inizio_storico:
            messagebox.showerror("Errore Input", "Data fine < data inizio per il caricamento dello storico."); self._log_to_gui("ERRORE: Data fine < data inizio per lo storico."); return None
        self.master.config(cursor="watch"); self.master.update_idletasks()
        storico = carica_storico_completo(cartella_dati, data_inizio_storico, data_fine_storico, app_logger=self._log_to_gui)
        self.master.config(cursor="")
        if not storico:
            messagebox.showinfo("Caricamento Dati", "Nessun dato storico caricato/filtrato."); self._log_to_gui("Nessun dato storico caricato/filtrato."); return None
        self.storico_caricato = storico; return storico

    def _get_parametri_gioco_comuni(self):
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
        impostazioni = {
            "tipo_metodo": "semplice", "ruota_calcolo_base": self.ruota_calcolo_var.get(),
            "posizione_estratto_base": self.posizione_estratto_var.get(), "num_ambate_dettagliare": self.num_ambate_var.get(),
            "min_tentativi_metodo": self.min_tentativi_var.get(), "tutte_le_ruote": self.tutte_le_ruote_var.get(),
            "ruote_gioco_selezionate": [r for r, v in self.ruote_gioco_vars.items() if v.get()],
            "lookahead": self.lookahead_var.get(), "indice_mese": self.indice_mese_var.get() }
        filepath = filedialog.asksaveasfilename( defaultextension=".json", filetypes=[("File JSON Imp. Semplici", "*.json"), ("Tutti i file", "*.*")], title="Salva Impostazioni Metodo Semplice" )
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(impostazioni, f, indent=4)
            self._log_to_gui(f"Impostazioni Metodo Semplice salvate in: {filepath}"); messagebox.showinfo("Salvataggio", "Impostazioni salvate!")
        except Exception as e: self._log_to_gui(f"Errore salvataggio imp. semplici: {e}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare:\n{e}")

    def apri_impostazioni_semplici(self):
        filepath = filedialog.askopenfilename( defaultextension=".json", filetypes=[("File JSON Imp. Semplici", "*.json"), ("Tutti i file", "*.*")], title="Apri Impostazioni Metodo Semplice" )
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: impostazioni = json.load(f)
            if impostazioni.get("tipo_metodo") != "semplice": messagebox.showerror("Errore Apertura", "File non valido."); return
            self.ruota_calcolo_var.set(impostazioni.get("ruota_calcolo_base", RUOTE[0]))
            self.posizione_estratto_var.set(impostazioni.get("posizione_estratto_base", 1))
            self.num_ambate_var.set(impostazioni.get("num_ambate_dettagliare", 1))
            self.min_tentativi_var.set(impostazioni.get("min_tentativi_metodo", 10))
            if "tutte_le_ruote" in impostazioni: self.tutte_le_ruote_var.set(impostazioni["tutte_le_ruote"])
            if "ruote_gioco_selezionate" in impostazioni and not self.tutte_le_ruote_var.get():
                for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(ruota in impostazioni["ruote_gioco_selezionate"])
            self.lookahead_var.set(impostazioni.get("lookahead", 3))
            self.indice_mese_var.set(impostazioni.get("indice_mese", ""))
            self.on_tab_changed(None) 
            self._log_to_gui(f"Impostazioni Metodo Semplice caricate da: {filepath}"); messagebox.showinfo("Apertura", "Impostazioni caricate!")
        except Exception as e: self._log_to_gui(f"Errore apertura imp. semplici: {e}"); messagebox.showerror("Errore Apertura", f"Impossibile aprire:\n{e}")

    def salva_metodo_complesso(self):
        if not self.definizione_metodo_complesso_attuale: messagebox.showwarning("Salvataggio", "Nessun metodo complesso definito."); return
        impostazioni = {
            "tipo_metodo": "complesso", "definizione_metodo": self.definizione_metodo_complesso_attuale,
            "tutte_le_ruote": self.tutte_le_ruote_var.get(),
            "ruote_gioco_selezionate": [r for r, v in self.ruote_gioco_vars.items() if v.get()],
            "lookahead": self.lookahead_var.get(), "indice_mese": self.indice_mese_var.get() }
        filepath = filedialog.asksaveasfilename( defaultextension=".lmc", filetypes=[("File Metodo Lotto Complesso", "*.lmc"), ("Tutti i file", "*.*")], title="Salva Metodo Complesso" )
        if not filepath: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(impostazioni, f, indent=4)
            self._log_to_gui(f"Metodo Complesso salvato in: {filepath}"); messagebox.showinfo("Salvataggio", "Metodo salvato!")
        except Exception as e: self._log_to_gui(f"Errore salvataggio metodo complesso: {e}"); messagebox.showerror("Errore Salvataggio", f"Impossibile salvare:\n{e}")

    def apri_metodo_complesso(self):
        filepath = filedialog.askopenfilename( defaultextension=".lmc", filetypes=[("File Metodo Lotto Complesso", "*.lmc"), ("Tutti i file", "*.*")], title="Apri Metodo Complesso" )
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f: impostazioni = json.load(f)
            if impostazioni.get("tipo_metodo") != "complesso" or "definizione_metodo" not in impostazioni:
                messagebox.showerror("Errore Apertura", "File non valido."); return
            self.definizione_metodo_complesso_attuale = impostazioni["definizione_metodo"]
            self._refresh_mc_listbox()
            if "tutte_le_ruote" in impostazioni: self.tutte_le_ruote_var.set(impostazioni["tutte_le_ruote"])
            if "ruote_gioco_selezionate" in impostazioni and not self.tutte_le_ruote_var.get():
                for ruota in RUOTE: self.ruote_gioco_vars[ruota].set(ruota in impostazioni["ruote_gioco_selezionate"])
            self.lookahead_var.set(impostazioni.get("lookahead", 3))
            self.indice_mese_var.set(impostazioni.get("indice_mese", ""))
            self.on_tab_changed(None) 
            self._log_to_gui(f"Metodo Complesso caricato da: {filepath}"); messagebox.showinfo("Apertura", "Metodo caricato!")
        except Exception as e: self._log_to_gui(f"Errore apertura metodo complesso: {e}"); messagebox.showerror("Errore Apertura", f"Impossibile aprire:\n{e}")

    def avvia_analisi_metodi_semplici(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA METODI SEMPLICI\n" + "="*50)
        storico_per_analisi = self._carica_e_valida_storico_comune()
        if not storico_per_analisi: return
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return
        ruota_calcolo = self.ruota_calcolo_var.get(); posizione_estratto = self.posizione_estratto_var.get() - 1
        num_ambate = self.num_ambate_var.get(); min_tentativi = self.min_tentativi_var.get()
        self._log_to_gui(f"Parametri Ricerca Metodi Semplici:\n  Ruota Base: {ruota_calcolo}, Posizione: {posizione_estratto+1}")
        self._log_to_gui(f"  Ruote Gioco: {', '.join(ruote_gioco)}\n  Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        self._log_to_gui(f"  Output Ambate: {num_ambate}, Min. Tentativi per Metodo: {min_tentativi}")
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            risultati = trova_migliori_ambate_e_abbinamenti(storico_per_analisi, ruota_calcolo, posizione_estratto, ruote_gioco, max_ambate_output=num_ambate, lookahead=lookahead, indice_mese_filtro=indice_mese, min_tentativi_per_ambata=min_tentativi, app_logger=self._log_to_gui)
            self._log_to_gui("\n\n--- RISULTATI FINALI RICERCA METODI SEMPLICI ---")
            if not risultati: self._log_to_gui("Nessun metodo semplice ha prodotto risultati sufficientemente frequenti.")
            else:
                for i, res in enumerate(risultati):
                    metodo = res['metodo']
                    self._log_to_gui(f"\n--- {i+1}° MIGLIOR METODO SEMPLICE PER AMBATA ---")
                    self._log_to_gui(f"  Metodo: {ruota_calcolo}[pos.{posizione_estratto+1}] {metodo['operazione']} {metodo['operando_fisso']}")
                    self._log_to_gui(f"  Ambata più frequente prodotta: {res.get('ambata_piu_frequente_dal_metodo', 'N/D')}")
                    self._log_to_gui(f"  Frequenza successo Ambata (metodo): {res['frequenza_ambata']:.2%} ({res['successi']}/{res['tentativi']} casi)")
                    desc_ruote_gioco = "TUTTE le ruote" if len(ruote_gioco) == len(RUOTE) else ", ".join(ruote_gioco)
                    if len(ruote_gioco) > 1 : self._log_to_gui(f"    (Conteggio successi su: {desc_ruote_gioco})")
                    abbinamenti = res.get("abbinamenti", {}); eventi_abbinamento = abbinamenti.get("eventi_abbinamento_analizzati", 0)
                    if eventi_abbinamento > 0:
                        self._log_to_gui(f"  Analizzati {eventi_abbinamento} eventi di vincita per abbinamenti con ambata '{res.get('ambata_piu_frequente_dal_metodo', 'N/D')}':")
                        for tipo_sorte, dati_sorte_lista in abbinamenti.items():
                            if tipo_sorte == "eventi_abbinamento_analizzati": continue
                            if dati_sorte_lista:
                                self._log_to_gui(f"    Migliori Abbinamenti per {tipo_sorte.upper().replace('_', ' ')}:")
                                mostrati = 0
                                for ab_info in dati_sorte_lista:
                                    if ab_info['conteggio'] > 0:
                                        numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri']))); self._log_to_gui(f"      - Numeri abbinati: [{numeri_ab_str}] -> Freq. {ab_info['frequenza']:.2%} (Conteggio: {ab_info['conteggio']})"); mostrati +=1
                                if mostrati == 0: self._log_to_gui(f"      Nessun abbinamento significativo per {tipo_sorte.upper()}.")
                    else: self._log_to_gui(f"  Nessun caso di successo del metodo ha prodotto l'ambata target '{res.get('ambata_piu_frequente_dal_metodo', 'N/D')}' per analizzare abbinamenti.")
            self._log_to_gui("\n--- Ricerca Metodi Semplici Completata ---")
            messagebox.showinfo("Analisi Completata", "Ricerca Metodi Semplici terminata. Vedi risultati.")
        except Exception as e: messagebox.showerror("Errore Analisi", f"Errore ricerca metodi semplici: {e}"); self._log_to_gui(f"ERRORE RICERCA METODI SEMPLICI: {e}"); import traceback; self._log_to_gui(traceback.format_exc())
        finally: self.master.config(cursor="")

    def avvia_analisi_metodo_complesso(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO ANALISI METODO COMPLESSO BASE\n" + "="*50)
        if not self.definizione_metodo_complesso_attuale: messagebox.showerror("Errore Input", "Definire almeno un componente."); self._log_to_gui("ERRORE: Metodo complesso non definito."); return
        if self.definizione_metodo_complesso_attuale[-1].get('operazione_successiva') != '=': messagebox.showerror("Errore Input Metodo", "Metodo deve terminare con '='."); self._log_to_gui("ERRORE: Metodo non terminato."); return
        storico_per_analisi = self._carica_e_valida_storico_comune()
        if not storico_per_analisi: return
        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: return
        metodo_str_display = "".join(self._format_componente_per_display(comp) for comp in self.definizione_metodo_complesso_attuale)
        self._log_to_gui(f"Parametri Analisi Metodo Complesso Base:\n  Metodo Definito: {metodo_str_display}")
        self._log_to_gui(f"  Ruote Gioco: {', '.join(ruote_gioco)}\n  Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            ambata_test, successi, tentativi, applicazioni_vincenti = analizza_metodo_complesso_specifico(storico_per_analisi, self.definizione_metodo_complesso_attuale, ruote_gioco, lookahead, indice_mese, self._log_to_gui)
            self._log_to_gui("\n--- RISULTATI ANALISI METODO COMPLESSO BASE ---")
            if tentativi == 0: self._log_to_gui("Metodo non applicabile.")
            else:
                frequenza_metodo = successi / tentativi if tentativi > 0 else 0
                self._log_to_gui(f"Metodo: {metodo_str_display}\nFreq: {frequenza_metodo:.2%} ({successi}/{tentativi})")
                if successi > 0:
                    # ambata_test è già l'ambata più frequente calcolata da analizza_metodo_complesso_specifico
                    ambata_target = ambata_test 
                    if ambata_target is not None:
                        self._log_to_gui(f"Ambata più frequente: {ambata_target}")
                        abbinamenti = analizza_abbinamenti_per_metodo_complesso(applicazioni_vincenti, ambata_target, self._log_to_gui)
                        eventi_abbinamento = abbinamenti.get("eventi_abbinamento_analizzati", 0)
                        if eventi_abbinamento > 0:
                            self._log_to_gui(f"  Analizzati {eventi_abbinamento} eventi di vincita per abbinamenti con ambata '{ambata_target}':")
                            for tipo_sorte, dati_sorte_lista in abbinamenti.items():
                                if tipo_sorte == "eventi_abbinamento_analizzati": continue
                                if dati_sorte_lista:
                                    self._log_to_gui(f"    Migliori Abbinamenti per {tipo_sorte.upper()}:")
                                    mostrati = 0
                                    for ab_info in dati_sorte_lista:
                                        if ab_info['conteggio'] > 0: 
                                            numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri'])))
                                            self._log_to_gui(f"      - Numeri abbinati: [{numeri_ab_str}] -> Freq. {ab_info['frequenza']:.2%} (Conteggio: {ab_info['conteggio']})")
                                            mostrati +=1
                                    if mostrati == 0: self._log_to_gui(f"      Nessun abbinamento significativo per {tipo_sorte.upper()}.")
                        else: self._log_to_gui(f"  Nessun caso di successo del metodo ha prodotto l'ambata target '{ambata_target}' per analizzare abbinamenti.")
                    else:
                         self._log_to_gui("Nessuna ambata prodotta consistentemente dal metodo.")
                else: self._log_to_gui("Il metodo non ha prodotto successi.")
            self._log_to_gui("\n--- Analisi Metodo Complesso Base Completata ---"); messagebox.showinfo("Analisi Completata", "Analisi Metodo Complesso Base terminata.")
        except Exception as e: messagebox.showerror("Errore Analisi Complessa", f"Errore: {e}"); self._log_to_gui(f"ERRORE ANALISI: {e}"); import traceback; self._log_to_gui(traceback.format_exc())
        finally: self.master.config(cursor="")

    def avvia_ricerca_correttore(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO RICERCA CORRETTORE PER METODO COMPLESSO\n" + "="*50)

        if not self.definizione_metodo_complesso_attuale:
            messagebox.showerror("Errore Input", "Definire prima un Metodo Base per cercare un correttore.")
            self._log_to_gui("ERRORE: Metodo Base non definito.")
            return
        
        if self.definizione_metodo_complesso_attuale[-1].get('operazione_successiva') != '=':
            messagebox.showerror("Errore Input", "Il Metodo Base deve essere terminato con l'operazione '=' prima di cercare un correttore.")
            self._log_to_gui("ERRORE: Metodo Base non terminato correttamente con '='.")
            return

        storico_per_analisi = self._carica_e_valida_storico_comune()
        if not storico_per_analisi:
            return 

        ruote_gioco, lookahead, indice_mese = self._get_parametri_gioco_comuni()
        if ruote_gioco is None: 
            return 
       
        min_tentativi_correttore = 5 

        metodo_base_str = "".join(self._format_componente_per_display(comp) for comp in self.definizione_metodo_complesso_attuale)
        self._log_to_gui(f"Ricerca correttore per Metodo Base: {metodo_base_str}")
        self._log_to_gui(f"  Opzioni di Gioco: Ruote: {', '.join(ruote_gioco)}, Colpi: {lookahead}, Ind.Mese: {indice_mese if indice_mese else 'Tutte'}")
        self._log_to_gui(f"  Min. Tentativi per validare un correttore: {min_tentativi_correttore}")

        try:
            self.master.config(cursor="watch")
            self.master.update_idletasks()

            metodo_base_per_ricerca = [dict(comp) for comp in self.definizione_metodo_complesso_attuale]

            risultati_correttori = trova_miglior_correttore_per_metodo_complesso(
                storico=storico_per_analisi,
                definizione_metodo_base=metodo_base_per_ricerca, 
                cerca_fisso=True,        
                cerca_estratto=True,     
                ruote_gioco_selezionate=ruote_gioco,
                lookahead=lookahead,
                indice_mese_filtro=indice_mese,
                min_tentativi_per_correttore=min_tentativi_correttore,
                app_logger=self._log_to_gui
            )
           
            self._log_to_gui("\n\n--- RISULTATI RICERCA CORRETTORI ---")
            if not risultati_correttori:
                self._log_to_gui("Nessun correttore valido trovato che migliori significativamente il metodo base o soddisfi i criteri minimi.")
            else:
                # MODIFICA: Mostra solo il miglior correttore e i suoi abbinamenti, o i top N se desiderato.
                # Per ora, mostriamo il migliore.
                self._log_to_gui(f"Trovati {len(risultati_correttori)} potenziali correttori. Mostro il migliore con analisi abbinamenti:")
                
                res_corr = risultati_correttori[0] # Prendiamo solo il primo (il migliore)
                
                metodo_esteso_str_display = "".join(self._format_componente_per_display(comp) for comp in res_corr['metodo_esteso_def'])
                
                self._log_to_gui(f"\n1. MIGLIOR CORRETTORE: Tipo '{res_corr['tipo_correttore']}', Dettaglio '{res_corr.get('dettaglio_correttore_str', '')}', Operazione di Collegamento '{res_corr['operazione_collegamento']}'")
                self._log_to_gui(f"   Metodo Esteso Risultante: {metodo_esteso_str_display}")
                
                ambata_estesa_output = res_corr.get('ambata_risultante_metodo_esteso', 'N/D')
                self._log_to_gui(f"   Ambata Prodotta dal Metodo Esteso: {ambata_estesa_output}")
                
                self._log_to_gui(f"   Frequenza Successo (Metodo Esteso): {res_corr['frequenza']:.2%} ({res_corr['successi']}/{res_corr['tentativi']} casi)")
                
                # --- INIZIO SEZIONE ABBINAMENTI PER IL MIGLIOR CORRETTORE ---
                ambata_target_per_abbinamenti = res_corr.get('ambata_risultante_metodo_esteso')
                applicazioni_vincenti_del_metodo_esteso = res_corr.get('applicazioni_vincenti_metodo_esteso')

                if ambata_target_per_abbinamenti is not None and applicazioni_vincenti_del_metodo_esteso:
                    self._log_to_gui(f"   --- Analisi Abbinamenti per ambata '{ambata_target_per_abbinamenti}' (Miglior Metodo Esteso) ---")
                    
                    abbinamenti_calcolati = analizza_abbinamenti_per_metodo_complesso(
                        applicazioni_vincenti_del_metodo_esteso,
                        ambata_target_per_abbinamenti, 
                        app_logger=self._log_to_gui
                    )
                    
                    eventi_abbinamento = abbinamenti_calcolati.get("eventi_abbinamento_analizzati", 0)
                    if eventi_abbinamento > 0:
                        self._log_to_gui(f"     Basato su {eventi_abbinamento} sortite vincenti dell'ambata '{ambata_target_per_abbinamenti}':")
                        for tipo_sorte, dati_sorte_lista in abbinamenti_calcolati.items():
                            if tipo_sorte == "eventi_abbinamento_analizzati": continue 
                            if dati_sorte_lista: 
                                self._log_to_gui(f"       Migliori Abbinamenti per {tipo_sorte.upper().replace('_', ' ')}:")
                                mostrati_per_sorte = 0
                                for ab_info in dati_sorte_lista:
                                    if ab_info['conteggio'] > 0:
                                        numeri_ab_str = ", ".join(map(str, sorted(ab_info['numeri'])))
                                        if tipo_sorte == "ambo":
                                            txt_numeri_abbinati = f"Numero abbinato: {numeri_ab_str}"
                                        else: 
                                            txt_numeri_abbinati = f"Numeri abbinati: [{numeri_ab_str}]"

                                        self._log_to_gui(f"         - {txt_numeri_abbinati} -> Freq. {ab_info['frequenza']:.2%} (Conteggio: {ab_info['conteggio']})")
                                        mostrati_per_sorte += 1
                                if mostrati_per_sorte == 0:
                                    self._log_to_gui(f"         Nessun abbinamento significativo trovato per {tipo_sorte.upper()}.")
                    else:
                        self._log_to_gui(f"     Nessun caso di successo del metodo esteso ha prodotto l'ambata target '{ambata_target_per_abbinamenti}' in modo utile per analizzare gli abbinamenti, o non ci sono state applicazioni vincenti sufficienti.")
                elif ambata_target_per_abbinamenti is None:
                    self._log_to_gui(f"   Impossibile determinare un'ambata consistente per il metodo esteso, quindi non si analizzano abbinamenti.")
                else: 
                    self._log_to_gui(f"   Non ci sono state applicazioni vincenti registrate per il metodo esteso per analizzare abbinamenti.")
                # --- FINE SEZIONE ABBINAMENTI ---
           
            self._log_to_gui("\n--- Ricerca Correttore Completata ---")
            messagebox.showinfo("Ricerca Correttore", "Ricerca del correttore terminata. Vedi risultati nell'area sottostante.")

        except Exception as e:
            messagebox.showerror("Errore Ricerca Correttore", f"Si è verificato un errore: {e}")
            self._log_to_gui(f"ERRORE DURANTE LA RICERCA DEL CORRETTORE: {e}")
            import traceback
            self._log_to_gui(traceback.format_exc())
        finally:
            self.master.config(cursor="")

    def avvia_verifica_giocata(self):
        self._log_to_gui("\n" + "="*50 + "\nAVVIO VERIFICA GIOCATA MANUALE\n" + "="*50)
        numeri_str = self.numeri_verifica_var.get()
        try:
            numeri_da_verificare = sorted([int(n.strip()) for n in numeri_str.split(',') if n.strip()])
            if not numeri_da_verificare: messagebox.showerror("Errore Input", "Inserisci numeri."); self._log_to_gui("ERRORE: Nessun numero."); return
            if not all(1 <= n <= 90 for n in numeri_da_verificare): messagebox.showerror("Errore Input", "Numeri 1-90."); self._log_to_gui("ERRORE: Numeri non validi."); return
        except ValueError: messagebox.showerror("Errore Input", "Formato numeri non valido."); self._log_to_gui("ERRORE: Formato numeri."); return
        try: data_inizio_ver = self.date_inizio_verifica_entry.get_date()
        except ValueError: messagebox.showerror("Errore Input", "Seleziona data inizio verifica."); self._log_to_gui("ERRORE: Data inizio non selezionata."); return
        colpi_ver = self.colpi_verifica_var.get()
        ruote_gioco_selezionate_ver, _, _ = self._get_parametri_gioco_comuni()
        if ruote_gioco_selezionate_ver is None: return
        cartella_dati = self.cartella_dati_var.get()
        if not cartella_dati or not os.path.isdir(cartella_dati): messagebox.showerror("Errore Input", "Seleziona cartella dati."); self._log_to_gui("ERRORE: Cartella dati non valida."); return
        self._log_to_gui(f"Parametri Verifica:\n  Numeri: {numeri_da_verificare}\n  Data Inizio: {data_inizio_ver}\n  Colpi: {colpi_ver}\n  Ruote: {', '.join(ruote_gioco_selezionate_ver)}")
        try:
            self.master.config(cursor="watch"); self.master.update_idletasks()
            self._log_to_gui("Caricamento storico per verifica...")
            storico_per_verifica = carica_storico_completo(cartella_dati, app_logger=self._log_to_gui) 
            if not storico_per_verifica: messagebox.showinfo("Risultato Verifica", "Nessun dato storico."); self._log_to_gui("Nessun dato storico per verifica."); return
            verifica_giocata_manuale(numeri_da_verificare, ruote_gioco_selezionate_ver, data_inizio_ver, colpi_ver, storico_per_verifica, app_logger=self._log_to_gui)
            messagebox.showinfo("Verifica Completata", "Verifica terminata.")
        except Exception as e: messagebox.showerror("Errore Verifica", f"Errore: {e}"); self._log_to_gui(f"ERRORE VERIFICA: {e}"); import traceback; self._log_to_gui(traceback.format_exc())
        finally: self.master.config(cursor="")

# --- BLOCCO PRINCIPALE DI ESECUZIONE ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LottoAnalyzerApp(root)
    root.mainloop()