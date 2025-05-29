import tkinter as tk
from tkinter import messagebox, filedialog
import os
import requests
import zipfile
import io
import sys
# import datetime # Aggiunto se necessario per validazione date

# --- Variabili Globali ---
root = None
mappa_ruote = {}
data_inizio_entry = None
data_fine_entry = None
directory_label = None
PERCORSO_ESTRAZIONE = None

# --- Funzioni di Utilità ---

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

PERCORSO_ESTRAZIONE = get_base_path()

def select_directory():
    global PERCORSO_ESTRAZIONE, directory_label
    current_dir = PERCORSO_ESTRAZIONE
    directory = filedialog.askdirectory(title="Seleziona directory per i file", initialdir=current_dir)
    if directory:
        PERCORSO_ESTRAZIONE = directory
        if directory_label:
            directory_label.config(text=f"Directory Dati: {PERCORSO_ESTRAZIONE}")
        return True
    return False

def scarica_e_estrai_zip(url, destinazione):
    try:
        print(f"Tentativo di download da: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=60, headers=headers)
        response.raise_for_status()
        print(f"Download completato ({len(response.content)} bytes). Estrazione in: {destinazione}")
        os.makedirs(destinazione, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(destinazione)
        print("Estrazione completata.")
        return True
    except requests.exceptions.Timeout:
        messagebox.showerror("Errore", "Timeout durante il tentativo di connessione al server per il download.")
        return False
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Errore", f"Errore durante il download: {e}")
        return False
    except zipfile.BadZipFile as e:
        messagebox.showerror("Errore", f"Errore nel file ZIP scaricato: {e}")
        return False
    except Exception as e:
        messagebox.showerror("Errore", f"Errore sconosciuto download/estrazione: {e}")
        return False

def carica_dati_da_file(nome_file):
    if not os.path.exists(nome_file):
         print(f"File non trovato: {nome_file}")
         return []
    try:
        with open(nome_file, 'r', encoding='utf-8', errors='ignore') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        messagebox.showerror("Errore", f"Errore lettura file {os.path.basename(nome_file)}: {e}")
        return []

def salva_dati_su_file(dati_ruote_da_aggiungere):
    global mappa_ruote, PERCORSO_ESTRAZIONE

    if not mappa_ruote:
         mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    successo_parziale = True
    file_modificati_conteggio = 0
    for ruota, dati_nuovi in dati_ruote_da_aggiungere.items():
        if dati_nuovi:
            nome_file_relativo = mappa_ruote.get(ruota)
            if not nome_file_relativo:
                print(f"Attenzione: Chiave ruota '{ruota}' non trovata. Salto.")
                continue

            nome_file = os.path.join(PERCORSO_ESTRAZIONE, nome_file_relativo)
            try:
                # Crea la directory padre se non esiste
                os.makedirs(os.path.dirname(nome_file), exist_ok=True)
                # Apre in modalità append, crea il file se non esiste
                with open(nome_file, 'a', encoding='utf-8') as file:
                    for line in dati_nuovi:
                        if ruota == "NZ":
                            parts = line.split()
                            if len(parts) >= 7:
                                line = f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\t{parts[4]}\t{parts[5]}\t{parts[6]}"
                        file.write(line + '\n')
                file_modificati_conteggio +=1
                print(f"Aggiunte {len(dati_nuovi)} estrazioni a {os.path.basename(nome_file)}")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore scrittura file {os.path.basename(nome_file)}: {e}")
                successo_parziale = False
    
    return successo_parziale, file_modificati_conteggio


def close_window():
    global root
    if root:
        try:
            print("Tentativo di chiusura finestra (quit + destroy)...")
            root.quit()
            root.destroy()
            print("Finestra chiusa.")
            root = None
        except tk.TclError as e:
            print(f"Errore Tcl durante chiusura (finestra già distrutta?): {e}")
        except Exception as e:
            print(f"Errore imprevisto durante close_window: {e}")
    else:
        print("close_window chiamata ma 'root' è None.")


def aggiorna_file(data_inizio, data_fine, percorso_estrazione):
    global mappa_ruote

    mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    storico_file_path = os.path.join(percorso_estrazione, 'storico.txt')
    dati_storico = carica_dati_da_file(storico_file_path)

    if not dati_storico and os.path.exists(storico_file_path):
        messagebox.showerror("Errore", f"Errore durante la lettura di storico.txt in {percorso_estrazione}.")
        return
    if not dati_storico and not os.path.exists(storico_file_path):
         messagebox.showerror("Errore", f"Il file storico.txt non è stato trovato in {percorso_estrazione}.")
         return

    estrazioni_da_aggiungere = {ruota: [] for ruota in mappa_ruote.keys()}
    estrazioni_nuove_trovate_nel_range = False
    
    estrazioni_esistenti_per_ruota = {}
    for ruota_code, nome_file_relativo in mappa_ruote.items():
        file_ruota_path = os.path.join(percorso_estrazione, nome_file_relativo)
        estrazioni_esistenti_per_ruota[ruota_code] = set(
            ' '.join(line.split()) for line in carica_dati_da_file(file_ruota_path)
        )

    for line_storico in dati_storico:
        parts = line_storico.split()
        if len(parts) < 7: continue
        
        date_str = parts[0]
        ruota_code_storico = parts[1].upper()
        
        if not (data_inizio <= date_str <= data_fine):
            continue

        current_ruota_key = "NZ" if ruota_code_storico == "RN" else ruota_code_storico
        
        if current_ruota_key in mappa_ruote:
            line_storico_normalizzata = ' '.join(line_storico.strip().split())
            if line_storico_normalizzata not in estrazioni_esistenti_per_ruota.get(current_ruota_key, set()):
                estrazioni_da_aggiungere[current_ruota_key].append(line_storico.strip())
                estrazioni_nuove_trovate_nel_range = True

    estrazioni_da_aggiungere = {r: l for r, l in estrazioni_da_aggiungere.items() if l}

    if not estrazioni_nuove_trovate_nel_range:
         messagebox.showinfo("Info", f"Nessuna NUOVA estrazione da aggiungere trovata in storico.txt per l'intervallo {data_inizio} - {data_fine}.")
         return

    if not estrazioni_da_aggiungere: # Dopo il filtro duplicati
        messagebox.showinfo("Info", "Tutte le estrazioni nell'intervallo specificato sono già presenti nei file delle ruote.")
        return

    successo_scrittura, file_aggiornati_count = salva_dati_su_file(estrazioni_da_aggiungere)

    if successo_scrittura:
        if file_aggiornati_count > 0:
            messagebox.showinfo("Successo", f"Aggiunte nuove estrazioni a {file_aggiornati_count} file ruota in:\n{percorso_estrazione}\nPeriodo analizzato: {data_inizio} - {data_fine}.")
        else: # Dovrebbe essere coperto dai controlli precedenti ma per sicurezza
            messagebox.showinfo("Info", "Nessuna nuova estrazione è stata effettivamente aggiunta ai file delle ruote.")
    else:
        messagebox.showwarning("Attenzione", "L'aggiornamento incrementale non è stato completato correttamente\n(errori durante la scrittura di alcuni file).")


def aggiorna_tutti_file():
    global PERCORSO_ESTRAZIONE, data_inizio_entry, data_fine_entry

    url_storico = "https://www.igt.it/STORICO_ESTRAZIONI_LOTTO/storico.zip"
    print("Avvio download e estrazione...")
    if not scarica_e_estrai_zip(url_storico, PERCORSO_ESTRAZIONE):
        print("Download fallito o interrotto.")
        if not messagebox.askyesno("Download Fallito", "Download di storico.zip fallito.\nVuoi provare ad aggiornare usando un file storico.txt locale (se esistente)?"):
            return
        elif not os.path.exists(os.path.join(PERCORSO_ESTRAZIONE, 'storico.txt')):
            messagebox.showerror("Errore", "storico.txt locale non trovato. Impossibile procedere.")
            return
        print("Procedo con storico.txt locale.")


    storico_file_path = os.path.join(PERCORSO_ESTRAZIONE, 'storico.txt')
    if not os.path.exists(storico_file_path):
         messagebox.showerror("Errore", f"storico.txt non trovato in {PERCORSO_ESTRAZIONE} dopo il tentativo di download.")
         return

    print("Correzione RN -> NZ in storico.txt (se necessario)...")
    try:
        modificato = False
        new_content = []
        with open(storico_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                if ' RN ' in line:
                    new_content.append(line.replace(' RN ', ' NZ ', 1))
                    modificato = True
                else:
                    new_content.append(line)
        if modificato:
             with open(storico_file_path, 'w', encoding='utf-8') as file:
                 file.writelines(new_content)
             print("Correzione RN -> NZ applicata.")
        else:
             print("Nessuna occorrenza di ' RN ' trovata per la correzione.")
    except Exception as e:
        messagebox.showerror("Errore Correzione", f"Errore correzione RN->NZ: {e}")
        return

    if not data_inizio_entry or not data_fine_entry:
         messagebox.showerror("Errore Interno", "Widget date non accessibili.")
         return
    data_inizio = data_inizio_entry.get().strip()
    data_fine = data_fine_entry.get().strip()

    if not (len(data_inizio) == 10 and data_inizio[4] == '/' and data_inizio[7] == '/'):
        messagebox.showerror("Errore Formato Data", "Formato data inizio non valido. Usare AAAA/MM/GG.")
        return
    if not (len(data_fine) == 10 and data_fine[4] == '/' and data_fine[7] == '/'):
        messagebox.showerror("Errore Formato Data", "Formato data fine non valido. Usare AAAA/MM/GG.")
        return
    if data_inizio > data_fine:
        messagebox.showerror("Errore Date", "La data di inizio non può essere successiva alla data di fine.")
        return

    if not data_inizio: messagebox.showerror("Errore", "Inserisci data inizio."); return
    if not data_fine: messagebox.showerror("Errore", "Inserisci data fine."); return

    print(f"Avvio aggiornamento file per periodo {data_inizio} - {data_fine}...")
    aggiorna_file(data_inizio, data_fine, PERCORSO_ESTRAZIONE)

# La funzione crea_file() non è più necessaria perché salva_dati_su_file() crea i file
# se non esistono quando si tenta di aggiungere dati.
# def crea_file():
#     ... (codice precedente rimosso) ...

def main():
    global root, data_inizio_entry, data_fine_entry, directory_label, PERCORSO_ESTRAZIONE, mappa_ruote

    if root is not None:
        try:
            root.deiconify(); root.lift(); root.focus_force()
            print("Finestra Tkinter già esistente.")
            return
        except tk.TclError: root = None

    print(f"Percorso base rilevato: {PERCORSO_ESTRAZIONE}")
    mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    root = tk.Tk()
    root.title("Aggiornamento Dati Lotto")
    # MODIFICA: Leggermente meno alto dato che un pulsante è stato rimosso
    root.geometry("480x360") 

    BG_COLOR="#f0f0f0"; BTN_COLOR_OK="#4CAF50"; BTN_COLOR_INFO="#2196F3"; BTN_COLOR_WARN="#f44336"; BTN_COLOR_NEUTRAL="#e0e0e0"; FG_COLOR_WHITE="white"; FG_COLOR_DARK="#333333"; FONT_DEFAULT=("Segoe UI", 10); FONT_LABEL=("Segoe UI", 9); FONT_INFO=("Segoe UI", 8)
    root.config(bg=BG_COLOR)

    main_frame = tk.Frame(root, padx=15, pady=15, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(1, weight=1)

    dir_frame = tk.Frame(main_frame, bg=BG_COLOR)
    dir_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    dir_frame.columnconfigure(0, weight=1)
    directory_label = tk.Label(dir_frame, text=f"Directory Dati: {PERCORSO_ESTRAZIONE}", anchor='w', wraplength=400, justify=tk.LEFT, bg=BG_COLOR, fg=FG_COLOR_DARK, font=FONT_LABEL)
    directory_label.grid(row=0, column=0, sticky='ew', padx=(0, 5))
    tk.Button(dir_frame, text="Cambia...", command=select_directory, bg=BTN_COLOR_NEUTRAL, font=FONT_DEFAULT, width=10).grid(row=0, column=1, sticky='e')

    tk.Frame(main_frame, height=1, bg="#cccccc").grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)

    tk.Label(main_frame, text="Data Inizio (AAAA/MM/GG):", anchor='w', bg=BG_COLOR, fg=FG_COLOR_DARK, font=FONT_LABEL).grid(row=2, column=0, padx=5, pady=5, sticky='w')
    data_inizio_entry = tk.Entry(main_frame, width=15, font=FONT_DEFAULT)
    data_inizio_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
    data_inizio_entry.insert(0, "1939/01/07") 

    tk.Label(main_frame, text="Data Fine   (AAAA/MM/GG):", anchor='w', bg=BG_COLOR, fg=FG_COLOR_DARK, font=FONT_LABEL).grid(row=3, column=0, padx=5, pady=5, sticky='w')
    data_fine_entry = tk.Entry(main_frame, width=15, font=FONT_DEFAULT)
    data_fine_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
    # Potresti voler inserire la data odierna di default per la data fine:
    # from datetime import date
    # data_fine_entry.insert(0, date.today().strftime('%Y/%m/%d'))


    # --- MODIFICA: Rimosso il pulsante "Crea File Ruote Mancanti" ---
    # Il pulsante "Scarica e Aggiorna" gestisce la creazione dei file se necessario.
    tk.Button(main_frame, text="Scarica e Aggiorna File Ruote", command=aggiorna_tutti_file, bg=BTN_COLOR_OK, fg=FG_COLOR_WHITE, height=2, font=FONT_DEFAULT, relief=tk.FLAT).grid(row=4, column=0, columnspan=2, padx=5, pady=(15, 5), sticky='ew')
    # tk.Button(main_frame, text="Crea File Ruote Mancanti", command=crea_file, bg=BTN_COLOR_INFO, fg=FG_COLOR_WHITE, height=2, font=FONT_DEFAULT, relief=tk.FLAT).grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='ew') # RIMOSSO

    # Spostato il pulsante Chiudi più su nella griglia
    tk.Button(main_frame, text="Chiudi Finestra", command=close_window, bg=BTN_COLOR_WARN, fg=FG_COLOR_WHITE, font=FONT_DEFAULT, relief=tk.FLAT).grid(row=5, column=0, columnspan=2, padx=5, pady=(10, 5), sticky='ew') # Riga aggiornata da 6 a 5

    # --- MODIFICA: Aggiornata etichetta informativa ---
    tk.Label(main_frame, text="Il programma scarica l'archivio, lo elabora per il periodo indicato e aggiunge le estrazioni ai file delle ruote (creandoli se non esistono), evitando duplicati.", font=FONT_INFO, fg="#666666", bg=BG_COLOR, wraplength=440, justify=tk.LEFT).grid(row=6, column=0, columnspan=2, padx=5, pady=(10, 0), sticky='w') # Riga aggiornata da 7 a 6

    root.protocol("WM_DELETE_WINDOW", close_window)
    print("Avvio Tkinter mainloop...")
    root.mainloop()
    print("Tkinter mainloop terminato.")
    return True

if __name__ == "__main__":
    main()
    print("Script principale terminato.")