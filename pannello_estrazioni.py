# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk # Manteniamo ttk per il Combobox
import os
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import traceback
from tkinter import filedialog, messagebox
import json

# --- Configurazione ---
# Percorso dati predefinito (sarà sovrascritto quando l'utente sceglie un percorso)
DEFAULT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pannello_config.json")

RUOTE_STANDARD = ["BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO",
                  "NAPOLI", "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE"]
RUOTE_MAP_CODICI = {
    'BARI': 'BA', 'CAGLIARI': 'CA', 'FIRENZE': 'FI', 'GENOVA': 'GE',
    'MILANO': 'MI', 'NAPOLI': 'NA', 'PALERMO': 'PA', 'ROMA': 'RO',
    'TORINO': 'TO', 'VENEZIA': 'VE', 'NAZIONALE': 'NZ'
}
RUOTE_ORDER = ['BA', 'CA', 'FI', 'GE', 'MI', 'NA', 'PA', 'RO', 'TO', 'VE', 'NZ']
HEADERS = ["Ruo", "E1", "E2", "E3", "E4", "E5", "Somma"]

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Colori e Sfondi come da screenshot desiderato
HIGHLIGHT_COLOR = 'yellow'
DEFAULT_BG_DATA = 'white'      # Sfondo celle numeri
DEFAULT_BG_HEADER = '#D3D3D3'  # Grigio scuro per header/ruota/somma
WINDOW_BG = '#E0E0FF'        # Sfondo finestra principale (lilla chiaro)
TOP_FRAME_BG = '#ADD8E6'       # Azzurro chiaro per frame superiore
BUTTON_GREEN_BG = '#4CAF50'    # Verde per bottone selezione cartella
TEXT_RED = '#FF0000'         # Rosso per data e crediti
TEXT_NAVY = 'navy'           # Blu scuro per status bar

# --- Funzioni di Gestione Configurazione ---
# (Invariate)
def load_config():
    """Carica la configurazione dal file JSON"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                if 'data_path' in config and os.path.exists(config['data_path']):
                    return config['data_path']
    except Exception as e:
        logging.warning(f"Errore caricamento configurazione: {e}")
    return DEFAULT_PATH

def save_config(data_path):
    """Salva la configurazione nel file JSON"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({'data_path': data_path}, f)
        return True
    except Exception as e:
        logging.error(f"Errore salvataggio configurazione: {e}")
        return False

# --- Funzioni di Caricamento Dati ---
# (Invariate - contengono la correzione RN/NZ)
def _carica_singolo_file_validato(file_path):
    try:
        if not os.path.exists(file_path):
            logging.error(f"H: File non trovato {file_path}")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        seen_rows=set()
        fmt_ok='%Y/%m/%d'
        num_cols=['Numero1','Numero2','Numero3','Numero4','Numero5']
        data_list=[]
        r_l=0; r_if=0; r_iv=0; r_id=0
        nome_ruota_atteso = os.path.splitext(os.path.basename(file_path))[0].upper()

        for ln, line in enumerate(lines, 1):
            r_l+=1
            line=line.strip()
            if not line: continue
            parts=line.split()
            if len(parts) != 7:
                r_if+=1
                continue
            data_str, ruota_str_orig, nums_orig = parts[0], parts[1], parts[2:7]
            try:
                data_dt_val = datetime.strptime(data_str, fmt_ok)
                [int(n) for n in nums_orig]
            except ValueError:
                r_iv+=1
                continue

            key = f"{data_str}_{nome_ruota_atteso}"
            if key in seen_rows:
                r_id+=1
                continue
            seen_rows.add(key)

            codice_atteso = RUOTE_MAP_CODICI.get(nome_ruota_atteso)
            if not codice_atteso:
                logging.warning(f"H [{os.path.basename(file_path)}: R{ln}] Ign: File '{nome_ruota_atteso}' non in MAPPA.")
                r_iv+=1
                continue

            codice_riga = ruota_str_orig.upper()

            if codice_riga != codice_atteso:
                is_roma_exception = (nome_ruota_atteso == 'ROMA' and codice_atteso == 'RO' and codice_riga == 'RM')
                is_nazionale_exception = (nome_ruota_atteso == 'NAZIONALE' and codice_atteso == 'NZ' and codice_riga == 'RN')

                if not is_roma_exception and not is_nazionale_exception:
                    if nome_ruota_atteso == 'NAZIONALE' and codice_riga == 'RN':
                         pass
                    else:
                        logging.warning(f"H [{os.path.basename(file_path)}: R{ln}] Ign: Codice riga ('{codice_riga}') != Atteso ('{codice_atteso}'). Riga: '{line}'")
                        r_iv+=1
                        continue

            numeri_validati_originali = []
            valid_row_numbers = True
            for n_str in nums_orig:
                try:
                    n_int = int(n_str)
                    if 1 <= n_int <= 90:
                        numeri_validati_originali.append(n_str)
                    else:
                        valid_row_numbers = False
                        break
                except ValueError:
                    valid_row_numbers = False
                    break

            if not valid_row_numbers:
                r_iv+=1
                continue

            row={'Data':data_dt_val, 'Ruota': codice_atteso} # Usa codice_atteso (NZ per Nazionale)
            for i, col in enumerate(num_cols):
                row[col] = numeri_validati_originali[i]
            data_list.append(row)

        if not data_list:
            return None
        df=pd.DataFrame(data_list)
        df['Data']=pd.to_datetime(df['Data'])
        df=df.sort_values(by='Data').reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"H: Errore grave durante caricamento file {file_path}: {e}", exc_info=True)
        return None

def mappa_file_ruote_e_carica(cartella_specifica):
    # (Invariata)
    mappa_ruote = {}
    file_cont = 0
    nomi_attesi = [f"{r}.TXT" for r in RUOTE_STANDARD]

    if not cartella_specifica or not os.path.isdir(cartella_specifica):
        logging.error(f"Cartella non valida: {cartella_specifica}")
        messagebox.showerror("Errore", f"Percorso non valido:\n{cartella_specifica}")
        return {}

    logging.info(f"Scansione: {cartella_specifica}...")
    try:
        for filename in os.listdir(cartella_specifica):
            filepath=os.path.join(cartella_specifica, filename)
            if os.path.isfile(filepath) and filename.upper().endswith(".TXT"):
                nome_ruota_base = os.path.splitext(filename)[0].upper()
                if nome_ruota_base in RUOTE_STANDARD:
                     mappa_ruote[nome_ruota_base] = filepath
                     file_cont += 1
        logging.info(f"Mappati {file_cont} file ruota standard.")
        if not mappa_ruote:
            messagebox.showerror("Nessun File", f"Nessun file ruota standard (*.txt) trovato in:\n{cartella_specifica}")
            return {}
    except Exception as e:
        logging.error(f"Errore scansione cartella: {e}", exc_info=True)
        messagebox.showerror("Errore", f"Errore scansione cartella:\n{e}")
        return {}

    estrazioni_finale = defaultdict(dict)
    estr_tot = 0
    logging.info("Inizio caricamento dati dai file mappati...")
    for nome_ruota, filepath in mappa_ruote.items():
        df_ruota = _carica_singolo_file_validato(filepath)
        if df_ruota is not None and not df_ruota.empty:
            for row in df_ruota.itertuples(index=False):
                date_str=row.Data.strftime('%Y/%m/%d')
                r_cod=row.Ruota
                nums=[row.Numero1,row.Numero2,row.Numero3,row.Numero4,row.Numero5]
                estrazioni_finale[date_str][r_cod] = nums
                estr_tot += 1
    date_count = len(estrazioni_finale)
    total_entries = sum(len(v) for v in estrazioni_finale.values())
    logging.info(f"Aggregazione completata. Caricate {total_entries} voci di estrazione per {date_count} date distinte.")
    return dict(estrazioni_finale)


# --- Variabili Globali GUI ---
# (Invariate)
root = None
table_labels = {}
estrazioni_caricate = {}
sorted_dates = []
current_date_index = -1
top_date_label = None
nav_entry_var = None
month_combo_var = None
month_combo = None
verifica_entry_var = None
status_label_var = None
data_path = None

# --- Funzioni GUI ---
# (Funzioni da clear_highlights a select_data_path INVARIATE nella logica,
# ma potrebbero essere influenzate dai cambiamenti in create_main_window)
def clear_highlights():
    if not table_labels: return
    for (ruota, header), label_widget in table_labels.items():
        if header.startswith("E"):
            label_widget.config(bg=DEFAULT_BG_DATA)

def highlight_numbers(numeri_da_evidenziare):
    if current_date_index < 0 or not sorted_dates: return 0
    target_date = sorted_dates[current_date_index]
    data_giorno = estrazioni_caricate.get(target_date, {})
    set_numeri_da_evidenziare = set(map(str, numeri_da_evidenziare))
    found_count = 0
    for ruota_code in RUOTE_ORDER:
        numeri_ruota = data_giorno.get(ruota_code, [])
        if len(numeri_ruota) == 5:
            for i, num_estratto in enumerate(numeri_ruota):
                if str(num_estratto) in set_numeri_da_evidenziare:
                    header_key = f"E{i+1}"
                    widget_key = (ruota_code, header_key)
                    if widget_key in table_labels:
                        table_labels[widget_key].config(bg=HIGHLIGHT_COLOR)
                        found_count += 1
    return found_count

def parse_validate_verify_numbers(input_str):
    numeri_da_cercare = set()
    if not input_str: return None
    try:
        parts = input_str.split()
        for part in parts:
            num_str = part.strip()
            if num_str:
                num_int = int(num_str)
                if 1 <= num_int <= 90:
                    numeri_da_cercare.add(str(num_int))
                else:
                    logging.warning(f"Numero fuori range (1-90) in verifica: {num_int}")
                    messagebox.showwarning("Input Errato", f"Il numero {num_int} è fuori dal range valido (1-90).", parent=root)
                    return None
        if not numeri_da_cercare : return None
        if len(numeri_da_cercare) > 10:
            logging.warning(f"Troppi numeri inseriti per la verifica (max 10): {len(numeri_da_cercare)}")
            messagebox.showwarning("Input Errato", f"Puoi verificare al massimo 10 numeri alla volta.\nHai inserito {len(numeri_da_cercare)} numeri.", parent=root)
            return None
        return numeri_da_cercare
    except ValueError:
        logging.warning(f"Input non numerico trovato nella verifica: '{input_str}'")
        messagebox.showerror("Input Errato", "L'input per la verifica contiene caratteri non numerici.\nInserire solo numeri da 1 a 90 separati da spazi.", parent=root)
        return None

def update_table_display(target_date):
    # (Logica invariata)
    global current_date_index
    clear_highlights()
    data_exists = bool(estrazioni_caricate and sorted_dates)

    if target_date is None:
         for r_code in RUOTE_ORDER:
            for hdr in HEADERS:
                key = (r_code, hdr)
                if key in table_labels:
                    table_labels[key].config(text=(r_code if hdr == "Ruo" else ""))
         if top_date_label:
            top_date_label.config(text="Nessun dato caricato")
         if month_combo_var:
             month_combo_var.set("")
         return

    data_found_for_date = data_exists and target_date in estrazioni_caricate

    if not data_exists or not data_found_for_date:
        for r_code in RUOTE_ORDER:
            for hdr in HEADERS:
                key = (r_code, hdr)
                if key in table_labels:
                    table_labels[key].config(text=(r_code if hdr == "Ruo" else ""))
        msg = "Dati non disponibili" if not data_exists else f"Dati non trovati per\n{target_date}"
        if top_date_label:
            top_date_label.config(text=msg)
        if month_combo_var:
             month_combo_var.set("")
        return

    try:
        current_date_index = sorted_dates.index(target_date)
    except ValueError:
        logging.error(f"Data target '{target_date}' non trovata in sorted_dates. Ritorno all'ultima data.")
        if not sorted_dates:
             update_table_display(None)
             return
        current_date_index = len(sorted_dates) - 1
        target_date = sorted_dates[current_date_index]

    try:
        date_obj=datetime.strptime(target_date,"%Y/%m/%d")
        # NOTA: L'originale usava abbreviazioni, lo screenshot nomi lunghi. Ripristino abbreviazioni.
        giorni=["lun","mar","mer","gio","ven","sab","dom"]
        giorno=giorni[date_obj.weekday()]
        # Formato data come da screenshot
        fmt_date=f"C. n.{current_date_index+1} di {giorno}\n{date_obj.strftime('%d/%m/%Y')}"
    except ValueError:
        fmt_date=f"Estrazione {current_date_index+1}\n{target_date}"

    if top_date_label:
        top_date_label.config(text=fmt_date)

    if month_combo_var and month_combo:
        month = target_date[5:7]
        if month in month_combo['values']:
             month_combo_var.set(month)
        else:
             month_combo_var.set("")

    data_giorno = estrazioni_caricate.get(target_date, {})
    for ruota_code in RUOTE_ORDER:
        numeri_ruota = data_giorno.get(ruota_code, [])
        somma_str = ""
        if len(numeri_ruota) == 5:
             try:
                somma_int = sum(int(n) for n in numeri_ruota)
                somma_str = str(somma_int)
             except (ValueError, TypeError) as e:
                logging.error(f"Errore nel calcolare somma per {ruota_code} data {target_date}: {numeri_ruota} - {e}")
                somma_str = "ERR"

        for header in HEADERS:
            key = (ruota_code, header)
            if key in table_labels:
                lbl=table_labels[key]
                txt=""
                if header == "Ruo":
                    txt = ruota_code
                elif header.startswith("E") and len(numeri_ruota) == 5:
                    try:
                        idx=int(header[1:])-1
                        if 0 <= idx < len(numeri_ruota):
                             txt = str(numeri_ruota[idx])
                        else:
                             txt = "?"
                    except (ValueError, IndexError):
                        txt = "?"
                elif header == "Somma":
                    txt = somma_str
                lbl.config(text=txt)

    if verifica_entry_var:
        numeri_in_entry = parse_validate_verify_numbers(verifica_entry_var.get())
        if numeri_in_entry:
            highlight_numbers(numeri_in_entry)

def go_last():
    # (Invariata)
    global current_date_index
    if sorted_dates:
        last_index = len(sorted_dates) - 1
        if current_date_index != last_index:
            update_table_display(sorted_dates[last_index])

def go_back_n_steps(event=None):
    # (Invariata)
    global current_date_index
    if not sorted_dates or current_date_index < 0: return
    try:
        input_value = nav_entry_var.get()
        steps_back = int(input_value)
        if steps_back <= 0:
            if steps_back == 0: return
            messagebox.showwarning("Input Errato", "Il numero di passi deve essere positivo.", parent=root)
            nav_entry_var.set("1")
            return
        target_index = current_date_index - steps_back
        target_index = max(0, target_index)
        if target_index != current_date_index:
            update_table_display(sorted_dates[target_index])
    except ValueError:
        logging.warning(f"Valore non valido nella casella passi: '{nav_entry_var.get()}'")
        messagebox.showwarning("Input Errato", "Inserire un numero valido nella casella dei passi.", parent=root)
        nav_entry_var.set("1")
    except Exception as e:
        logging.error(f"Errore in go_back_n_steps: {e}", exc_info=True)

def go_forward_n_steps(event=None):
    # (Invariata)
    global current_date_index
    if not sorted_dates or current_date_index < 0: return
    try:
        input_value = nav_entry_var.get()
        steps_forward = int(input_value)
        if steps_forward <= 0:
             if steps_forward == 0: return
             messagebox.showwarning("Input Errato", "Il numero di passi deve essere positivo.", parent=root)
             nav_entry_var.set("1")
             return
        target_index = current_date_index + steps_forward
        last_valid_index = len(sorted_dates) - 1
        target_index = min(target_index, last_valid_index)
        if target_index != current_date_index:
            update_table_display(sorted_dates[target_index])
    except ValueError:
        logging.warning(f"Valore non valido nella casella passi: '{nav_entry_var.get()}'")
        messagebox.showwarning("Input Errato", "Inserire un numero valido nella casella dei passi.", parent=root)
        nav_entry_var.set("1")
    except Exception as e:
        logging.error(f"Errore in go_forward_n_steps: {e}", exc_info=True)

def month_prev():
    # (Invariata)
    global current_date_index
    if current_date_index < 0 or not sorted_dates: return
    try:
        current_date_str = sorted_dates[current_date_index]
        current_dt = datetime.strptime(current_date_str, "%Y/%m/%d")
        first_of_current_month = current_dt.replace(day=1)
        last_of_prev_month = first_of_current_month - timedelta(days=1)
        target_year = last_of_prev_month.year
        target_month = last_of_prev_month.month
        found_index = find_last_date_in_month(target_year, target_month)
        if found_index != -1 and found_index != current_date_index:
            update_table_display(sorted_dates[found_index])
        elif found_index == -1:
             logging.info(f"Nessun dato trovato per il mese precedente: {target_year}/{target_month:02d}")
    except Exception as e:
        logging.error(f"Errore in month_prev: {e}", exc_info=True)

def month_next():
    # (Invariata)
    global current_date_index
    if current_date_index < 0 or not sorted_dates: return
    try:
        current_date_str = sorted_dates[current_date_index]
        current_dt = datetime.strptime(current_date_str, "%Y/%m/%d")
        try:
            days_in_current_month = pd.Timestamp(current_dt).days_in_month
        except AttributeError:
            import calendar
            days_in_current_month = calendar.monthrange(current_dt.year, current_dt.month)[1]
        first_of_current_month = current_dt.replace(day=1)
        first_of_next_month = first_of_current_month + timedelta(days=days_in_current_month)
        target_year = first_of_next_month.year
        target_month = first_of_next_month.month
        found_index = find_last_date_in_month(target_year, target_month)
        if found_index != -1 and found_index != current_date_index:
            update_table_display(sorted_dates[found_index])
        elif found_index == -1:
            logging.info(f"Nessun dato trovato per il mese successivo: {target_year}/{target_month:02d}")
    except Exception as e:
        logging.error(f"Errore in month_next: {e}", exc_info=True)

def find_last_date_in_month(year, month):
    # (Invariata)
    target_prefix = f"{year:04d}/{month:02d}/"
    last_found_index = -1
    for i in range(len(sorted_dates) - 1, -1, -1):
        if sorted_dates[i].startswith(target_prefix):
            last_found_index = i
            break
    return last_found_index

def on_month_selected(event=None):
    # (Invariata)
    global current_date_index
    if current_date_index < 0 or not sorted_dates or not month_combo_var: return
    try:
        selected_month_str = month_combo_var.get()
        if not selected_month_str: return
        selected_month = int(selected_month_str)
        current_date_str = sorted_dates[current_date_index]
        current_year = int(current_date_str[:4])
        found_index = find_last_date_in_month(current_year, selected_month)
        if found_index != -1:
            if found_index != current_date_index:
                 update_table_display(sorted_dates[found_index])
        else:
            actual_month_str = current_date_str[5:7]
            logging.info(f"Nessun dato trovato per {current_year}/{selected_month:02d}. Ripristino a {actual_month_str}.")
            month_combo_var.set(actual_month_str)
            messagebox.showinfo("Nessun Dato", f"Non sono presenti estrazioni per il mese {selected_month:02d} dell'anno {current_year}.", parent=root)
    except ValueError:
         logging.error(f"Errore nella selezione del mese: valore non valido '{month_combo_var.get()}'")
         if sorted_dates and current_date_index >=0:
             try:
                 current_date_str = sorted_dates[current_date_index]
                 actual_month_str = current_date_str[5:7]
                 month_combo_var.set(actual_month_str)
             except Exception: pass
    except Exception as e:
        logging.error(f"Errore generico in on_month_selected: {e}", exc_info=True)
        if sorted_dates and current_date_index >= 0:
             try:
                 current_date_str = sorted_dates[current_date_index]
                 actual_month_str = current_date_str[5:7]
                 month_combo_var.set(actual_month_str)
             except Exception: pass

def pulisci_verifica():
    # (Invariata)
    clear_highlights()
    if verifica_entry_var:
        verifica_entry_var.set("")
    if status_label_var:
        status_label_var.set("")

def verifica_numeri_sortiti():
    # (Invariata)
    if not verifica_entry_var: return
    if not estrazioni_caricate or not sorted_dates or current_date_index < 0:
        if status_label_var:
            status_label_var.set("Errore: Dati non caricati o nessuna data selezionata.")
        messagebox.showerror("Errore Verifica", "Non ci sono dati di estrazione caricati o nessuna data è selezionata per poter effettuare la verifica.", parent=root)
        return
    input_str = verifica_entry_var.get().strip()
    clear_highlights()
    if not input_str:
        if status_label_var:
            status_label_var.set("")
        return
    numeri_da_cercare = parse_validate_verify_numbers(input_str)
    if numeri_da_cercare is not None:
        found_count = highlight_numbers(numeri_da_cercare)
        if status_label_var:
            if found_count > 0:
                status_label_var.set(f"Verifica: {found_count} numeri evidenziati.")
            else:
                status_label_var.set("Verifica: Nessuno dei numeri inseriti è presente.")

def select_data_path():
    # (Invariata)
    global data_path, estrazioni_caricate, sorted_dates, current_date_index
    initial_dir = data_path if data_path and os.path.isdir(data_path) else os.path.dirname(DEFAULT_PATH)
    if not os.path.isdir(initial_dir):
        initial_dir = os.path.expanduser("~")
    folder_selected = filedialog.askdirectory(
        title="Seleziona la cartella contenente i file TXT delle estrazioni",
        initialdir=initial_dir,
        parent=root
    )
    if folder_selected:
        if not os.path.isdir(folder_selected):
            messagebox.showerror("Errore Percorso", f"Il percorso selezionato non è una cartella valida:\n{folder_selected}", parent=root)
            return
        data_path = folder_selected
        config_saved = save_config(data_path)
        if not config_saved:
             messagebox.showwarning("Errore Configurazione", "Impossibile salvare il percorso selezionato nel file di configurazione.", parent=root)
        if status_label_var:
            status_label_var.set(f"Caricamento dati da: {os.path.basename(data_path)}...")
            root.update_idletasks()
        estrazioni_caricate_new = mappa_file_ruote_e_carica(data_path)
        if estrazioni_caricate_new:
            estrazioni_caricate = estrazioni_caricate_new
            sorted_dates = sorted(estrazioni_caricate.keys())
            current_date_index = len(sorted_dates) - 1 if sorted_dates else -1
            if current_date_index >= 0:
                update_table_display(sorted_dates[current_date_index])
                if status_label_var:
                     status_label_var.set(f"Dati caricati ({len(sorted_dates)} date). Percorso: {os.path.basename(data_path)}")
            else:
                update_table_display(None)
                if status_label_var:
                    status_label_var.set(f"Caricamento completato, ma nessuna data trovata in: {os.path.basename(data_path)}")
        else:
            estrazioni_caricate = {}
            sorted_dates = []
            current_date_index = -1
            update_table_display(None)
            if status_label_var:
                 status_label_var.set(f"Errore o nessun dato valido trovato in: {os.path.basename(data_path)}")

# --- Creazione Finestra GUI ---
def create_main_window():
    """Crea la finestra principale della GUI."""
    global root, table_labels, top_date_label, nav_entry_var, month_combo_var
    global month_combo, verifica_entry_var, status_label_var

    # Gestione root esistente
    existing_root = getattr(tk, '_default_root', None)
    if existing_root and existing_root.winfo_exists():
        try:
            existing_root.destroy()
        except tk.TclError:
            logging.warning("Errore durante la distruzione della vecchia root Tk.")
            pass

    root = tk.Tk()
    tk._default_root = root

    root.title("Estrazioni Lotto") # Titolo originale
    root.configure(bg=WINDOW_BG) # Sfondo originale
    root.minsize(600, 450) # Dimensioni minime originali

    # --- Frame Superiore (Data e Selezione Percorso) ---
    top_frame = tk.Frame(root, bg=TOP_FRAME_BG) # Colore originale
    top_frame.pack(pady=10, padx=10, fill=tk.X)

    # Etichetta Data (Stile Originale)
    top_date_label = tk.Label(top_frame, text="Caricamento...", font=("Arial", 14, "bold"),
                               fg=TEXT_RED, bg=top_frame.cget('bg'), justify=tk.CENTER, height=2)
    top_date_label.pack(side=tk.LEFT, padx=(20, 10), pady=5, expand=True, fill=tk.X)

    # Pulsante Selezione Cartella (Stile Originale)
    select_path_button = tk.Button(
        top_frame,
        text="Selezione Cartella Dati", # Testo originale
        command=select_data_path,
        bg=BUTTON_GREEN_BG, # Colore originale
        fg="white",         # Colore testo originale
        font=("Arial", 10, "bold"),
        relief=tk.RAISED,
        bd=2
    )
    select_path_button.pack(side=tk.RIGHT, padx=10, pady=5)

    # --- Frame Tabella ---
    table_frame = tk.Frame(root, bd=1, relief=tk.SOLID) # Stile bordo originale
    table_frame.pack(pady=5, padx=10, fill=tk.X)
    header_bg = DEFAULT_BG_HEADER # Sfondo header/ruota/somma
    hdr_font = ("Arial", 10, "bold")
    data_font = ("Consolas", 11, "bold") # Font numeri originale (bold)
    city_font = ("Arial", 10, "bold")    # Font ruote/somma originale

    # Creazione Headers Tabella (Stile Originale)
    for col, hdr_txt in enumerate(HEADERS):
        w=6
        if hdr_txt=="Ruo": w=4
        elif hdr_txt=="Somma": w=7
        elif hdr_txt.startswith("E"): w=5
        # Usa RIDGE come nell'originale per header
        lbl = tk.Label(table_frame, text=hdr_txt, font=hdr_font, bg=header_bg, relief=tk.RIDGE, bd=1, width=w, padx=5, pady=3)
        lbl.grid(row=0, column=col, sticky="nsew")

    # Creazione Celle Dati Tabella (Stile Originale)
    table_labels = {}
    for row, r_code in enumerate(RUOTE_ORDER):
        for col, hdr in enumerate(HEADERS):
            bg=DEFAULT_BG_DATA # Default bianco per numeri
            fg="black"
            fnt=data_font     # Font numeri (Consolas bold)
            w=6
            anchor_val=tk.CENTER # Default numeri centrati

            if hdr=="Ruo":
                bg=header_bg # Sfondo grigio per ruota
                fnt=city_font
                w=4
                anchor_val=tk.CENTER
            elif hdr=="Somma":
                bg=header_bg # Sfondo grigio per somma
                fnt=city_font # Font come ruota per somma
                w=7
                anchor_val=tk.CENTER
            elif hdr.startswith("E"):
                w=5
                anchor_val=tk.CENTER

            # Usa RIDGE come nell'originale per tutte le celle
            lbl = tk.Label(table_frame, text="", font=fnt, bg=bg, fg=fg, relief=tk.RIDGE, bd=1, width=w, padx=5, pady=3, anchor=anchor_val)
            lbl.grid(row=row+1, column=col, sticky="nsew")
            table_labels[(r_code, hdr)] = lbl

    # Configura espansione colonne
    for i in range(len(HEADERS)):
        table_frame.grid_columnconfigure(i, weight=1)

    # --- Frame Navigazione Superiore (Passi e Mese) ---
    nav_frame_top = tk.Frame(root, bg=root.cget('bg')) # Usa sfondo finestra
    nav_frame_top.pack(pady=(10, 5), fill=tk.X, padx=10)

    # Frame Sinistro: Navigazione per passi (Stile Originale)
    nav_left = tk.Frame(nav_frame_top, bg=root.cget('bg'))
    nav_left.pack(side=tk.LEFT, padx=(0,0))

    tk.Button(nav_left, text="<", command=go_back_n_steps, font=("Arial", 10, "bold"), width=3).pack(side=tk.LEFT, padx=2) # bd e relief default
    nav_entry_var = tk.StringVar(value="1")
    entry_nav = tk.Entry(nav_left, textvariable=nav_entry_var, width=4, font=("Arial", 10, "bold"), justify='center', bd=2, relief=tk.SUNKEN) # SUNKEN originale
    entry_nav.pack(side=tk.LEFT, padx=2)
    entry_nav.bind("<Return>", go_forward_n_steps) # Cambiato binding invio a avanti
    entry_nav.bind("<KP_Enter>", go_forward_n_steps)
    tk.Button(nav_left, text=">", command=go_forward_n_steps, font=("Arial", 10, "bold"), width=3).pack(side=tk.LEFT, padx=2) # bd e relief default
    # Bottone Last stile originale
    tk.Button(nav_left, text="Last", command=go_last, font=("Arial", 10, "bold"), bg="#90EE90", fg="black", width=5, relief=tk.RAISED, bd=2).pack(side=tk.LEFT, padx=10)

    # Frame Destro: Navigazione per mese (Stile Originale)
    nav_right = tk.Frame(nav_frame_top, bg=root.cget('bg'))
    nav_right.pack(side=tk.RIGHT, padx=(0,10)) # Pad destra

    tk.Label(nav_right, text="Mese:", font=("Arial", 9), bg=root.cget('bg')).pack(side=tk.LEFT, padx=(0, 5)) # Label Mese:
    tk.Button(nav_right, text="<", command=month_prev, font=("Arial", 10, "bold"), width=3).pack(side=tk.LEFT, padx=2) # Bottone <
    month_combo_var = tk.StringVar()
    months = [f"{i:02d}" for i in range(1, 13)]
    # Usa ttk.Combobox come prima per dropdown
    month_combo = ttk.Combobox(nav_right, textvariable=month_combo_var, values=months, width=3, font=("Arial", 10), state="readonly", justify='center')
    month_combo.pack(side=tk.LEFT, padx=2)
    month_combo.bind("<<ComboboxSelected>>", on_month_selected)
    tk.Button(nav_right, text=">", command=month_next, font=("Arial", 10, "bold"), width=3).pack(side=tk.LEFT, padx=2) # Bottone >


    # --- Frame Verifica Numeri ---
    verify_frame = tk.Frame(root, bg=root.cget('bg'))
    verify_frame.pack(pady=(5, 0), fill=tk.X, padx=10)

    tk.Label(verify_frame, text="Verifica Numeri:", font=("Arial", 9), bg=root.cget('bg')).pack(side=tk.LEFT, padx=(0, 5)) # Testo label originale
    verifica_entry_var = tk.StringVar()
    verify_entry = tk.Entry(verify_frame, textvariable=verifica_entry_var, width=30, font=("Arial", 10)) # Larghezza originale
    verify_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    verify_entry.bind("<Return>", lambda event: verifica_numeri_sortiti())
    verify_entry.bind("<KP_Enter>", lambda event: verifica_numeri_sortiti())

    # Usa tk.Button come nell'originale (anche se ttk è stato usato per combobox)
    verify_button = tk.Button(verify_frame, text="Verifica", command=verifica_numeri_sortiti) # Stile default tk.Button
    verify_button.pack(side=tk.LEFT, padx=(0,5))
    clear_button = tk.Button(verify_frame, text="Pulisci", command=pulisci_verifica) # Stile default tk.Button
    clear_button.pack(side=tk.LEFT, padx=0)

    # --- Label Crediti (Stile Originale) ---
    credits_label = tk.Label(
        root,
        text="Il Lotto di Max",
        font=("Arial", 16, "bold italic"), # Font originale
        bg=root.cget('bg'), # Sfondo finestra
        fg=TEXT_RED # Colore rosso originale
    )
    credits_label.pack(pady=(5, 8)) # Padding originale

    # --- Etichetta di Stato (Stile Originale) ---
    status_label_var = tk.StringVar()
    status_initial_text = f"Percorso dati: {data_path}" if data_path else "Nessun percorso dati selezionato. Usa 'Selezione Cartella Dati'."
    status_label_var.set(status_initial_text)

    # Label direttamente in root, come nell'originale
    status_label = tk.Label(root, textvariable=status_label_var, font=("Arial", 9),
                           bg=WINDOW_BG, fg=TEXT_NAVY, anchor=tk.W) # Colori e anchor originali
    # Calcola wraplength dinamicamente (opzionale ma utile)
    # status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0,5)) # Pack originale
    # Per evitare il wrap automatico che potrebbe andare a capo male, lo togliamo momentaneamente
    status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0,5))


    # --- Update Iniziale dopo creazione GUI ---
    def schedule_initial_update():
        if root and root.winfo_exists():
            if sorted_dates:
                global current_date_index
                if current_date_index == -1:
                    current_date_index = len(sorted_dates) - 1
                current_date_index = max(0, min(len(sorted_dates) - 1, current_date_index))
                initial_date = sorted_dates[current_date_index]
                root.after(50, lambda d=initial_date: update_table_display(d))
            else:
                 root.after(50, lambda: update_table_display(None))
        else:
            logging.error("Impossibile schedulare update iniziale: la finestra root non esiste più.")

    root.after(10, schedule_initial_update)

    return root

# --- Esecuzione Principale ---
# (Invariata)
if __name__ == "__main__":
    data_path = load_config()
    estrazioni_caricate = {}
    sorted_dates = []
    current_date_index = -1
    root = None
    if data_path and os.path.isdir(data_path):
        print(f"Percorso dati caricato dalla configurazione: {data_path}")
        estrazioni_caricate = mappa_file_ruote_e_carica(data_path)
        if estrazioni_caricate:
            sorted_dates = sorted(estrazioni_caricate.keys())
            current_date_index = len(sorted_dates) - 1
            print(f"Caricate {len(estrazioni_caricate)} date, {len(sorted_dates)} date uniche ordinate.")
        else:
             print(f"Nessun dato valido trovato nel percorso configurato: {data_path}")
    else:
        print("Nessun percorso dati valido trovato nella configurazione o percorso non esistente.")
        data_path = None
    try:
        root = create_main_window()
        if root:
            root.mainloop()
        else:
             messagebox.showerror("Errore Critico", "Impossibile creare la finestra principale (create_main_window ha restituito None).")
             logging.error("create_main_window() ha restituito None.")
    except Exception as e:
        logging.error(f"Errore imprevisto nell'esecuzione principale della GUI: {e}", exc_info=True)
        try:
            messagebox.showerror("Errore GUI Imprevisto", f"Si è verificato un errore grave:\n{e}\n\nL'applicazione potrebbe dover essere chiusa.\nControlla i log per dettagli tecnici.")
        except Exception as msg_err:
            logging.error(f"Impossibile mostrare il messaggio di errore della GUI: {msg_err}")
        if root and isinstance(root, tk.Tk) and root.winfo_exists():
            try:
                root.destroy()
            except Exception as destroy_err:
                 logging.error(f"Errore durante il tentativo di chiudere la finestra Tkinter dopo un errore: {destroy_err}")