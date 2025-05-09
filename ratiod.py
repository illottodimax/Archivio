# -*- coding: utf-8 -*-
import requests
import datetime
import time
import copy
import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
import traceback

# --- Configurazione e Costanti ---
GITHUB_USER = "illottodimax"; GITHUB_REPO = "Archivio"; GITHUB_BRANCH = "main"
NOMI_RUOTE = ['BARI', 'CAGLIARI', 'FIRENZE', 'GENOVA', 'MILANO', 'NAPOLI', 'PALERMO', 'ROMA', 'TORINO', 'VENEZIA', 'NAZIONALE']
URL_RUOTE = {nome: f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{nome}.txt' for nome in NOMI_RUOTE}
ARCHIVI_CARICATI = {}
# Tag GUI
TAG_RED = 'red_text'
TAG_BOLD = 'bold_text'
TAG_RED_BOLD = 'red_bold_text'
# Costanti logica VB
ESTRAZIONI_SUCCESSIVE_VBS = 12
LIMITE_OCCORRENZE_SPIA_VBS = 12

# --- Variabili Globali per GUI ---
output_queue = queue.Queue()
analysis_running = False
selected_downloads_dir = os.getcwd() # Default cartella corrente, ma non mostrata
# folder_path_var = None # RIMOSSO

# --- Funzioni di Utilità Dati ---
# (Nessuna funzione qui)

# --- Funzioni di Output per la GUI ---
def gui_output(message, tags=None, clear=False, status=False, error=False):
    actual_tags = tuple(tags) if tags is not None else ()
    output_queue.put({'message': message, 'tags': actual_tags, 'clear': clear, 'status': status, 'error': error})
def gui_output_segments(segments, clear=False):
    output_queue.put({'segments': segments, 'clear': clear})

# --- PARSER ARCHIVIO ---
def parse_righe_archivio(righe, nome_ruota_str):
    archivio_ordine_file = []
    parsed_count = 0; skipped_count = 0
    for riga in righe:
        riga_pulita = riga.strip(); parsed_this_line = False
        if not riga_pulita: continue
        parti = riga_pulita.split()
        if len(parti) >= 7:
            data_str = parti[0].strip()
            try:
                y,m,d = map(int, data_str.split('/'))
                datetime.date(y, m, d)
                if not (1900 < y < 2100): raise ValueError("Anno fuori range")
                numeri_str = [n.strip() for n in parti[2:7]]
                numeri = [int(n) for n in numeri_str]
                if len(numeri) == 5 and all(1 <= num <= 90 for num in numeri):
                    archivio_ordine_file.append({'data': data_str, 'numeri': numeri})
                    parsed_count += 1; parsed_this_line = True
                else: pass
            except (ValueError, IndexError): pass
        if not parsed_this_line: skipped_count += 1
    if not archivio_ordine_file: return [], 0
    return archivio_ordine_file, parsed_count

# --- CARICA ARCHIVIO ---
def carica_archivio_ruota(nome_ruota_str):
    global selected_downloads_dir
    if not selected_downloads_dir:
         gui_output("ERRORE: Selezionare prima una cartella per gli archivi!", error=True)
         return None
    if not os.path.isdir(selected_downloads_dir):
        gui_output(f"ERRORE: La cartella selezionata '{selected_downloads_dir}' non esiste o non è una cartella valida!", error=True)
        return None

    if nome_ruota_str not in NOMI_RUOTE: return None
    if nome_ruota_str in ARCHIVI_CARICATI:
        gui_output(f"Utilizzo archivio {nome_ruota_str} già in memoria.")
        return ARCHIVI_CARICATI[nome_ruota_str]

    file_locale_path = os.path.join(selected_downloads_dir, f"{nome_ruota_str}.txt")
    archivio_crono = None

    if os.path.exists(file_locale_path):
        gui_output(f"Trovato archivio locale per {nome_ruota_str}. Carico...")
        try:
            with open(file_locale_path, 'r', encoding='utf-8') as f: righe_locali = f.readlines()
            archivio_crono, p_count = parse_righe_archivio(righe_locali, nome_ruota_str)
            if archivio_crono:
                gui_output(f"Archivio locale {nome_ruota_str} caricato ({p_count}).")
            else:
                gui_output(f"Archivio locale per {nome_ruota_str} vuoto/illeggibile. Tento download.", status=True)
        except Exception as e:
            gui_output(f"Errore lettura file locale {nome_ruota_str}: {e}. Tento download.", error=True)
            archivio_crono = None

    if archivio_crono is None:
        url = URL_RUOTE.get(nome_ruota_str)
        gui_output(f"Tentativo download archivio per {nome_ruota_str}...")
        try:
            headers = {'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
            response = requests.get(url, timeout=30, headers=headers); response.raise_for_status()
            contenuto_file = response.text; gui_output("Download completato.")
            righe_scaricate = contenuto_file.strip().split('\n')
            archivio_crono_dl, p_count_dl = parse_righe_archivio(righe_scaricate, nome_ruota_str)

            if archivio_crono_dl:
                gui_output(f"Archivio {nome_ruota_str} scaricato e parsificato ({p_count_dl}).")
                try:
                    with open(file_locale_path, 'w', encoding='utf-8') as f: f.write(contenuto_file)
                    gui_output(f"Archivio salvato/aggiornato in '{os.path.basename(file_locale_path)}'.")
                except IOError as e:
                    gui_output(f"ATTENZIONE: Impossibile salvare archivio in '{os.path.basename(file_locale_path)}': {e}", error=True)
                archivio_crono = archivio_crono_dl
            else:
                gui_output(f"ERRORE CRITICO: Nessuna estrazione valida per {nome_ruota_str} nel file scaricato.", error=True)
        except requests.exceptions.Timeout: gui_output(f"ERRORE Download {nome_ruota_str}: Timeout.", error=True)
        except requests.exceptions.RequestException as e: gui_output(f"ERRORE Download {nome_ruota_str}: {e}", error=True)
        except Exception as e: gui_output(f"ERRORE Generico download/parsing: {e}\n{traceback.format_exc()}", error=True)

    if archivio_crono:
        ARCHIVI_CARICATI[nome_ruota_str] = archivio_crono
        return archivio_crono
    else:
        ARCHIVI_CARICATI.pop(nome_ruota_str, None)
        gui_output(f"ERRORE FINALE: Impossibile caricare archivio per {nome_ruota_str}.", error=True)
        return None

# --- Funzioni Helper ---
def estrazione_fin_idx(archivio): return len(archivio) - 1 if archivio else -1
def estratto_da_archivio(archivio, indice_0based, posizione_1based):
    if not archivio or not (0 <= indice_0based < len(archivio)): return 0
    if not (1 <= posizione_1based <= 5): return 0
    try: return archivio[indice_0based]['numeri'][posizione_1based - 1]
    except (KeyError, IndexError, TypeError): return 0
def data_estrazione_da_archivio(archivio, indice_0based):
    if not archivio or not (0 <= indice_0based < len(archivio)): return "Data_Err"
    try: return archivio[indice_0based]['data']
    except (KeyError, IndexError, TypeError): return "Data_Err"

# --- Funzioni di Formattazione ---
def format2(numero):
    try: return f"{int(numero):02d}"
    except: return "00"

# --- Funzione di Analisi (Decine, con NUOVO header) ---
def analizza_ruota_lotto(nome_ruota_scelta, archivio_ruota_crono, num_estrazioni_da_analizzare):
    global analysis_running
    start_time_analysis = time.time()
    try:
        gui_output(f"\nInizio analisi DECINE per Ruota: {nome_ruota_scelta} ({num_estrazioni_da_analizzare} estrazioni)", tags=(TAG_BOLD,), clear=True)
        gui_output("-" * (len(nome_ruota_scelta) + 30))
        num_tot_estrazioni_archivio = len(archivio_ruota_crono)
        if num_estrazioni_da_analizzare <= 0: raise ValueError("N estrazioni non positivo.")
        if num_tot_estrazioni_archivio == 0: raise ValueError("Archivio vuoto.")
        n_effettivo = min(num_estrazioni_da_analizzare, num_tot_estrazioni_archivio)
        if n_effettivo != num_estrazioni_da_analizzare: gui_output(f"ATTENZIONE: Richieste {num_estrazioni_da_analizzare}, analizzo le ultime {n_effettivo}.", status=True)
        indice_ultima_estrazione = num_tot_estrazioni_archivio - 1
        indice_fine_ciclo = indice_ultima_estrazione
        indice_inizio_ciclo = indice_ultima_estrazione - n_effettivo + 1
        data_inizio_batch = data_estrazione_da_archivio(archivio_ruota_crono, indice_inizio_ciclo)
        data_fine_batch = data_estrazione_da_archivio(archivio_ruota_crono, indice_fine_ciclo)
        gui_output(f"Analizzo estrazioni dal {data_inizio_batch} al {data_fine_batch}")

        # --- NUOVA GENERAZIONE HEADER ALLINEATA ---
        header_prefix = " " * 11 # 11 spazi per corrispondere a "YYYY/MM/DD "
        header1_content = "0000000001 1111111112 2222222223 3333333334 4444444445 5555555556 6666666667 7777777778 8888888889"
        header2_content = "1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890 1234567890"

        gui_output(f"{header_prefix}{header1_content}")
        gui_output(f"{header_prefix}{header2_content}")
        gui_output("") # Riga vuota dopo l'header
        # --- FINE NUOVA GENERAZIONE HEADER ---

        estratti_precedenti_numeri = [-1] * 5

        # Ciclo principale
        for idx_corr in range(indice_inizio_ciclo, indice_fine_ciclo + 1):
            try:
                numeri_correnti = archivio_ruota_crono[idx_corr]['numeri']
                if len(numeri_correnti) != 5: raise IndexError("Estrazione corrente incompleta")
            except (KeyError, IndexError, TypeError) as e_data: gui_output(f"ERRORE dati indice {idx_corr}: {e_data}", error=True); continue

            idx_limite_storico = 0
            conteggio_numeri = {n: 0 for n in range(1, 91)}

            # Logica Conteggio per DECINE
            for idx_spia_0based in range(5):
                numero_spia = numeri_correnti[idx_spia_0based]
                if not (1 <= numero_spia <= 90): continue
                contatore_occ_spia = 0
                decina_spia = (numero_spia - 1) // 10
                pos_spia_1based = idx_spia_0based + 1
                for idx_storica in range(idx_corr - 1, idx_limite_storico - 1, -1):
                    num_storico = estratto_da_archivio(archivio_ruota_crono, idx_storica, pos_spia_1based)
                    if num_storico == numero_spia:
                        contatore_occ_spia += 1
                        for j in range(1, ESTRAZIONI_SUCCESSIVE_VBS + 1):
                            idx_succ = idx_storica + j
                            if idx_succ <= idx_corr:
                                for pos_succ_0based in range(5):
                                    num_succ = estratto_da_archivio(archivio_ruota_crono, idx_succ, pos_succ_0based + 1)
                                    pos_succ_1based = pos_succ_0based + 1
                                    if 1 <= num_succ <= 90 and (num_succ - 1) // 10 == decina_spia:
                                        pos_verifica_1based = (pos_spia_1based + j - 1) % 5 + 1
                                        pos_succ_next_1based = (pos_succ_1based % 5) + 1
                                        if pos_verifica_1based == pos_succ_1based or pos_verifica_1based == pos_succ_next_1based:
                                            conteggio_numeri[num_succ] += 1
                        if contatore_occ_spia >= LIMITE_OCCORRENZE_SPIA_VBS:
                            break

            # Logica Colorazione Estratti per DECINE (originale)
            colora_rosso = [False] * 5
            if idx_corr > indice_inizio_ciclo:
                for k in range(5):
                    num_c = numeri_correnti[k]
                    num_p = estratti_precedenti_numeri[k]
                    if 1 <= num_c <= 90 and 1 <= num_p <= 90 and (num_c - 1) // 10 == (num_p - 1) // 10:
                        colora_rosso[k] = True

            # Preparazione Output Riga
            segmenti_riga = []; data_str = data_estrazione_da_archivio(archivio_ruota_crono, idx_corr)
            # Lo spazio dopo la data qui è importante per l'allineamento dei dati
            segmenti_riga.append({'text': f"{data_str:<10s} ", 'tags': (TAG_BOLD,)}) # 10 spazi data + 1 spazio = 11 caratteri

            # Griglia 9x10 (Decine)
            for i in range(9):
                for j in range(1, 11):
                    num_reale = i * 10 + j; conteggio = conteggio_numeri[num_reale]
                    if conteggio == 0: display = "*"
                    elif conteggio < 10: display = str(conteggio)
                    else: display = chr(ord('A') + min(conteggio - 10, 25))
                    tags = (TAG_RED_BOLD,) if num_reale in numeri_correnti else (TAG_BOLD,)
                    segmenti_riga.append({'text': display, 'tags': tags})
                if i < 8: segmenti_riga.append({'text': " ", 'tags': ()})

            # Colorazione Estratti Finali
            segmenti_riga.append({'text': "  ", 'tags': ()})
            for k in range(5):
                tags = (TAG_RED_BOLD,) if colora_rosso[k] else (TAG_BOLD,)
                testo = format2(numeri_correnti[k]) + ("." if k < 4 else "")
                segmenti_riga.append({'text': testo, 'tags': tags})

            gui_output_segments(segmenti_riga)
            estratti_precedenti_numeri = list(numeri_correnti)

        # Fine Ciclo Analisi
        gui_output("-" * (len(nome_ruota_scelta) + 30))
        gui_output(f"Analisi DECINE per Ruota: {nome_ruota_scelta} completata.", tags=(TAG_BOLD,))
        end_time_analysis = time.time()
        gui_output(f"Tempo di esecuzione analisi: {end_time_analysis - start_time_analysis:.2f} secondi.")
        gui_output("Pronto.", status=True)

    except ValueError as ve: gui_output(f"ERRORE: {ve}", error=True, status=True)
    except Exception as e: gui_output(f"ERRORE IMPREVISTO: {e}\n{traceback.format_exc()}", error=True); gui_output("Analisi interrotta.", status=True)
    finally: analysis_running = False; output_queue.put({'command': 'analysis_finished'})

# --- Funzioni GUI ---
def update_gui():
    try:
        while not output_queue.empty():
            item = output_queue.get_nowait()
            if 'command' in item:
                if item['command'] == 'analysis_finished':
                    ruota_combo['state'] = 'readonly'; num_estrazioni_entry['state'] = 'normal'
                    analyze_button['state'] = 'normal'; analyze_button.config(text="Analizza")
                    folder_button['state'] = 'normal'
            elif 'status' in item and item['status']:
                 status_label.config(text=item['message'])
                 status_label.config(foreground='red' if item.get('error') else 'black')
            else: # Messaggi o segmenti
                output_text.config(state=tk.NORMAL)
                if item.get('clear'): output_text.delete('1.0', tk.END)
                if 'segments' in item:
                     for seg in item['segments']: output_text.insert(tk.END, seg['text'], tuple(seg.get('tags', [])))
                     output_text.insert(tk.END, '\n')
                elif 'message' in item:
                     tags = tuple(item.get('tags', []))
                     output_text.insert(tk.END, item['message'] + '\n', tags)
                output_text.see(tk.END)
                output_text.config(state=tk.DISABLED)
    except queue.Empty: pass
    finally: root.after(100, update_gui)

# --- MODIFICATO: select_folder senza aggiornare folder_path_var ---
def select_folder():
    global selected_downloads_dir #, folder_path_var # Rimosso folder_path_var
    initial_dir = selected_downloads_dir if selected_downloads_dir else os.getcwd()
    directory = filedialog.askdirectory(
        title="Seleziona la cartella per gli archivi Lotto",
        initialdir=initial_dir
    )
    if directory:
        selected_downloads_dir = directory
        # folder_path_var.set(selected_downloads_dir) # <<< RIMOSSO
        gui_output(f"Cartella archivi impostata.", status=True)
        ARCHIVI_CARICATI.clear()
        gui_output("Cache archivi svuotata.", status=True)

def start_analysis_thread():
    global analysis_running
    if analysis_running: messagebox.showwarning("Attenzione", "Analisi già in corso."); return

    if not selected_downloads_dir:
        messagebox.showerror("Errore", "Selezionare prima una cartella per gli archivi...")
        return

    if not os.path.isdir(selected_downloads_dir):
        messagebox.showerror("Errore", f"La cartella selezionata non esiste! ({selected_downloads_dir})")
        return

    nome_ruota_selezionata = ruota_combo.get()
    if not nome_ruota_selezionata: messagebox.showerror("Errore", "Seleziona ruota."); return
    try:
        num_estrazioni_str = num_estrazioni_entry.get()
        num_estrazioni_analizzare = int(num_estrazioni_str) if num_estrazioni_str else 30
        if num_estrazioni_analizzare <= 0: raise ValueError("N. estrazioni non positivo.")
        if not num_estrazioni_str: num_estrazioni_var.set("30")
    except ValueError as ve: messagebox.showerror("Errore", f"Numero estrazioni non valido: '{num_estrazioni_str}'. Deve essere un intero positivo."); return

    analysis_running = True
    ruota_combo['state'] = 'disabled'; num_estrazioni_entry['state'] = 'disabled'
    analyze_button['state'] = 'disabled'; analyze_button.config(text="Analisi...")
    folder_button['state'] = 'disabled'
    status_label.config(text=f"Avvio analisi DECINE per {nome_ruota_selezionata}...", foreground='black')
    gui_output("", clear=True)

    def analysis_task():
        archivio_crono = None
        try:
            gui_output(f"Caricamento archivio per {nome_ruota_selezionata}...", status=True)
            archivio_crono = carica_archivio_ruota(nome_ruota_selezionata)

            if archivio_crono and len(archivio_crono) >= 2:
                 try:
                     d0_str = data_estrazione_da_archivio(archivio_crono, 0)
                     d1_str = data_estrazione_da_archivio(archivio_crono, 1)
                     if d0_str != "Data_Err" and d1_str != "Data_Err":
                          if d0_str > d1_str:
                              gui_output("!!! ERRORE INTERNO: Ordine archivio errato post-caricamento !!!", error=True)
                              raise ValueError("Ordine archivio errato rilevato")
                 except Exception as e_ord: gui_output(f"WARNING: Verifica ordine fallita: {e_ord}", error=True)

            if archivio_crono:
                 gui_output(f"Archivio caricato ({len(archivio_crono)} estr.). Avvio calcoli DECINE...", status=True)
                 analizza_ruota_lotto(nome_ruota_selezionata, archivio_crono, num_estrazioni_analizzare)
            else:
                 gui_output("Analisi annullata: caricamento archivio fallito.", error=True, status=True)
                 if analysis_running:
                     analysis_running = False; output_queue.put({'command': 'analysis_finished'})

        except Exception as task_e:
            gui_output(f"ERRORE nel task di analisi: {task_e}", error=True, status=True)
            if analysis_running:
                 analysis_running = False; output_queue.put({'command': 'analysis_finished'})

    thread = threading.Thread(target=analysis_task, daemon=True); thread.start()


# --- Creazione GUI (SENZA Label/Entry per il percorso) ---
root = tk.Tk()
root.title("Analisi Statistica Lotto (Decine - VB Logic Emulation)")
root.geometry("1100x750")

# Frame per selezione cartella (contiene solo il bottone)
folder_frame = ttk.Frame(root, padding="10")
folder_frame.pack(side=tk.TOP, fill=tk.X)

# Pulsante per scegliere la cartella
folder_button = ttk.Button(folder_frame, text="Scegli Cartella Archivi...", command=select_folder)
folder_button.pack(side=tk.LEFT, padx=(5, 0))

# Frame Controlli Analisi (come prima)
controls_frame = ttk.Frame(root, padding="10")
controls_frame.pack(side=tk.TOP, fill=tk.X)
ttk.Label(controls_frame, text="Ruota:").pack(side=tk.LEFT, padx=5)
ruota_combo = ttk.Combobox(controls_frame, values=NOMI_RUOTE, state="readonly", width=15); ruota_combo.pack(side=tk.LEFT, padx=5)
if NOMI_RUOTE: ruota_combo.current(0)
ttk.Label(controls_frame, text="Ultime Estrazioni da Analizzare:").pack(side=tk.LEFT, padx=5)
num_estrazioni_var = tk.StringVar(value="30");
num_estrazioni_entry = ttk.Entry(controls_frame, textvariable=num_estrazioni_var, width=10); num_estrazioni_entry.pack(side=tk.LEFT, padx=5)
analyze_button = ttk.Button(controls_frame, text="Analizza", command=start_analysis_thread); analyze_button.pack(side=tk.LEFT, padx=10)

# Frame Output (come prima)
output_frame = ttk.Frame(root, padding="5")
output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
output_font_family = 'Courier New'
output_font_size = 11
default_font = (output_font_family, output_font_size)
bold_font = (output_font_family, output_font_size, 'bold')
output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.NONE, state=tk.DISABLED, height=25, width=120)
output_text.pack(fill=tk.BOTH, expand=True)
output_text.config(font=default_font)
output_text.tag_configure(TAG_RED, foreground="red")
output_text.tag_configure(TAG_BOLD, font=bold_font)
output_text.tag_configure(TAG_RED_BOLD, foreground="red", font=bold_font)

# Status Bar (come prima)
status_label = ttk.Label(root, text="Pronto.", padding="5", anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# --- Avvio ---
if __name__ == "__main__":
    gui_output("==================================================", tags=(TAG_BOLD,))
    gui_output("   Analisi Statistica Lotto (DECINE - VB Logic Emulation)    ", tags=(TAG_BOLD,))
    gui_output("==================================================", tags=(TAG_BOLD,))
    # gui_output(f"Cartella archivi iniziale: '{selected_downloads_dir}'") # RIMOSSO
    gui_output("Seleziona ruota, numero ultime estrazioni (default 30) e clicca 'Analizza'.")
    gui_output("Usa 'Scegli Cartella Archivi...' per selezionare la cartella con i file .txt.")
    gui_output("La cartella selezionata deve esistere.", status=True)
    root.after(100, update_gui)
    root.mainloop()