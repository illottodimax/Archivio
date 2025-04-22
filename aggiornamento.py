import tkinter as tk
from tkinter import messagebox, filedialog
import os
import requests
import zipfile
import io
import sys
# import datetime # Aggiunto se necessario per validazione date (non modificato qui)

# --- Variabili Globali ---
root = None
mappa_ruote = {}
data_inizio_entry = None
data_fine_entry = None
directory_label = None
PERCORSO_ESTRAZIONE = None # Verrà impostato all'avvio

# --- Funzioni di Utilità ---

def get_base_path():
    """Determina il percorso base."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

PERCORSO_ESTRAZIONE = get_base_path()

def select_directory():
    """Permette all'utente di selezionare una directory personalizzata"""
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
    """Scarica un file ZIP dall'URL e lo estrae nella destinazione."""
    try:
        print(f"Tentativo di download da: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'} # Aggiunto User-Agent semplice
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
    """Carica le linee da un file di testo."""
    if not os.path.exists(nome_file):
         print(f"File non trovato: {nome_file}")
         return [] # Ritorna lista vuota se non trovato (come originale)
    try:
        # Usa errors='ignore' per robustezza con encoding
        with open(nome_file, 'r', encoding='utf-8', errors='ignore') as file:
            # Rimuove spazi bianchi e linee vuote
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        messagebox.showerror("Errore", f"Errore lettura file {os.path.basename(nome_file)}: {e}")
        return [] # Ritorna lista vuota in caso di errore (come originale)

def salva_dati_su_file(dati_ruote):
    """Salva i dati forniti nei rispettivi file delle ruote (append)."""
    global mappa_ruote, PERCORSO_ESTRAZIONE

    if not mappa_ruote:
         mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    # Flag per tracciare se ci sono stati errori
    successo_parziale = True
    for ruota, dati in dati_ruote.items():
        if dati:
            nome_file_relativo = mappa_ruote.get(ruota)
            if not nome_file_relativo:
                print(f"Attenzione: Chiave ruota '{ruota}' non trovata. Salto.")
                continue

            nome_file = os.path.join(PERCORSO_ESTRAZIONE, nome_file_relativo)
            try:
                # Usa 'a' (append) come nel codice originale
                with open(nome_file, 'a', encoding='utf-8') as file:
                    for line in dati:
                        # Logica formattazione NZ originale
                        if ruota == "NZ":
                            parts = line.split()
                            if len(parts) >= 7:
                                line = f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\t{parts[4]}\t{parts[5]}\t{parts[6]}"
                        file.write(line + '\n')
            except Exception as e:
                messagebox.showerror("Errore", f"Errore scrittura file {os.path.basename(nome_file)}: {e}")
                successo_parziale = False # Segna che c'è stato un errore
    # Ritorna True solo se NON ci sono stati errori, False altrimenti
    return successo_parziale


# --- NUOVA FUNZIONE PER CHIUSURA SICURA ---
# MODIFICA 1: Introduzione funzione close_window
def close_window():
    """Funzione per chiudere la finestra Tkinter in modo sicuro."""
    global root
    if root:
        try:
            print("Tentativo di chiusura finestra (quit + destroy)...")
            root.quit()      # 1. Ferma il mainloop
            root.destroy()   # 2. Distrugge la finestra e i widget
            print("Finestra chiusa.")
            root = None # Azzera riferimento
        except tk.TclError as e:
            print(f"Errore Tcl durante chiusura (finestra già distrutta?): {e}")
        except Exception as e:
            print(f"Errore imprevisto durante close_window: {e}")
    else:
        print("close_window chiamata ma 'root' è None.")

# --- Funzioni Principali dell'Applicazione ---

def aggiorna_file(data_inizio, data_fine, percorso_estrazione):
    """Filtra storico.txt per data e aggiorna i file delle singole ruote."""
    global mappa_ruote # Usa mappa globale

    # Ricrea mappa ruote qui come nell'originale
    mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    tutti_dati_filtrati = {ruota: [] for ruota in mappa_ruote.keys()}
    storico_file_path = os.path.join(percorso_estrazione, 'storico.txt')
    dati_storico = carica_dati_da_file(storico_file_path)

    # Logica originale di gestione errore e filtraggio
    if not dati_storico and os.path.exists(storico_file_path): # Se carica_dati fallisce ma file esiste
        messagebox.showerror("Errore", f"Errore durante la lettura di storico.txt in {percorso_estrazione}.")
        return
    if not dati_storico and not os.path.exists(storico_file_path):
         messagebox.showerror("Errore", f"Il file storico.txt non è stato trovato in {percorso_estrazione}.")
         return

    date_trovate_nel_range = False
    for line in dati_storico:
        parts = line.split()
        if len(parts) < 7: continue
        date_str = parts[0]
        ruota_code = parts[1].upper() # Converti per sicurezza

        if data_inizio <= date_str <= data_fine:
            date_trovate_nel_range = True
            current_ruota = "NZ" if ruota_code == "RN" else ruota_code
            if current_ruota in tutti_dati_filtrati:
                 # Aggiungi linea originale (la formattazione NZ è in salva_dati)
                tutti_dati_filtrati[current_ruota].append(line.strip())

    tutti_dati_filtrati = {r: l for r, l in tutti_dati_filtrati.items() if l} # Rimuovi vuoti

    if not date_trovate_nel_range:
         messagebox.showinfo("Info", f"Nessuna estrazione trovata in storico.txt per l'intervallo {data_inizio} - {data_fine}.")
         return
    if not tutti_dati_filtrati:
        messagebox.showinfo("Info", "Nessun dato valido per le ruote note nell'intervallo specificato.")
        return

    # --- Logica Cancellazione Originale ---
    print("Cancellazione file ruote esistenti prima dell'aggiornamento...")
    cancellazione_ok = True
    for ruota_code in tutti_dati_filtrati.keys(): # Cancella solo se ci sono dati filtrati!
         nome_file_ruota = mappa_ruote.get(ruota_code)
         if nome_file_ruota:
             file_path = os.path.join(percorso_estrazione, nome_file_ruota)
             if os.path.exists(file_path):
                 try:
                     os.remove(file_path)
                     print(f"Rimosso: {os.path.basename(file_path)}")
                 except Exception as e:
                     messagebox.showerror("Errore", f"Impossibile rimuovere {os.path.basename(file_path)}: {e}")
                     cancellazione_ok = False; break # Interrompi se errore
    if not cancellazione_ok: return

    # --- Logica Salvataggio Originale (con append) ---
    # salva_dati_su_file ora ritorna False se ci sono stati errori
    successo_scrittura = salva_dati_su_file(tutti_dati_filtrati)

    if successo_scrittura:
        # SOLO SE NESSUN ERRORE DI SCRITTURA è stato mostrato da salva_dati_su_file
        messagebox.showinfo("Successo", f"Dati aggiunti per {len(tutti_dati_filtrati)} ruote in:\n{percorso_estrazione}\nPeriodo: {data_inizio} - {data_fine}.")
        # --- MODIFICA 2: Usa close_window in caso di successo ---
        close_window()
    else:
        # Se salva_dati_su_file ha mostrato errori, informa ma non chiudere
        messagebox.showwarning("Attenzione", "L'aggiornamento non è stato completato correttamente\n(errori durante la scrittura di alcuni file).")
        # Non chiamare close_window() qui


def aggiorna_tutti_file():
    """Gestisce il download, la correzione RN->NZ e chiama aggiorna_file."""
    global PERCORSO_ESTRAZIONE, data_inizio_entry, data_fine_entry

    # 1. Scarica e Estrai - URL originale
    url_storico = "https://www.igt.it/STORICO_ESTRAZIONI_LOTTO/storico.zip"
    print("Avvio download e estrazione...")
    if not scarica_e_estrai_zip(url_storico, PERCORSO_ESTRAZIONE):
        print("Download fallito o interrotto.")
        return

    # 2. Correggi RN -> NZ
    storico_file_path = os.path.join(PERCORSO_ESTRAZIONE, 'storico.txt')
    if not os.path.exists(storico_file_path):
         messagebox.showerror("Errore", f"storico.txt non trovato in {PERCORSO_ESTRAZIONE}")
         return

    print("Correzione RN -> NZ in storico.txt (se necessario)...")
    try:
        modificato = False
        new_content = []
        with open(storico_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                # Usa replace con count=1 per sicurezza
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
             print("Nessuna occorrenza di ' RN ' trovata.")
    except Exception as e:
        messagebox.showerror("Errore Correzione", f"Errore correzione RN->NZ: {e}")
        return

    # 3. Ottieni e Valida Date (rudimentale come originale)
    if not data_inizio_entry or not data_fine_entry:
         messagebox.showerror("Errore Interno", "Widget date non accessibili.")
         return
    data_inizio = data_inizio_entry.get().strip()
    data_fine = data_fine_entry.get().strip()

    if not data_inizio: messagebox.showerror("Errore", "Inserisci data inizio."); return
    if not data_fine: messagebox.showerror("Errore", "Inserisci data fine."); return
    # Aggiungere validazione formato/logica date se necessario

    # 5. Chiama aggiorna_file
    print(f"Avvio aggiornamento file per periodo {data_inizio} - {data_fine}...")
    aggiorna_file(data_inizio, data_fine, PERCORSO_ESTRAZIONE)
    # aggiorna_file gestirà la chiusura della finestra in caso di successo totale


def crea_file():
    """Crea i file .txt vuoti per ogni ruota se non esistono."""
    global mappa_ruote, PERCORSO_ESTRAZIONE

    # Ricrea mappa ruote qui come nell'originale
    mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    if not PERCORSO_ESTRAZIONE or not os.path.isdir(PERCORSO_ESTRAZIONE):
         if not select_directory():
             messagebox.showerror("Errore", "Percorso destinazione non valido.")
             return

    print(f"Verifica/creazione file mancanti in: {PERCORSO_ESTRAZIONE}")
    creati=0; esistenti=0; errori=[]
    for ruota_code, nome_file_ruota in mappa_ruote.items():
        nome_file_completo = os.path.join(PERCORSO_ESTRAZIONE, nome_file_ruota)
        try:
            if not os.path.exists(nome_file_completo):
                os.makedirs(os.path.dirname(nome_file_completo), exist_ok=True)
                with open(nome_file_completo, 'w', encoding='utf-8') as f: pass
                print(f"File CREATO: {os.path.basename(nome_file_completo)}")
                creati += 1
            else:
                 esistenti += 1
        except Exception as e:
            msg_err = f"Errore creazione/verifica {os.path.basename(nome_file_completo)}: {e}"
            print(msg_err); errori.append(msg_err)

    if errori:
         messagebox.showerror("Errore", "Errori durante creazione file:\n- " + "\n- ".join(errori))
    else:
         messagebox.showinfo("Successo", f"Verifica completata in {PERCORSO_ESTRAZIONE}.\nCreati: {creati} file.\nEsistenti: {esistenti} file.")
         # --- MODIFICA 3: Usa close_window in caso di successo ---
         close_window()


# --- Funzione Principale per l'Interfaccia Grafica ---
def main():
    """Crea e avvia l'interfaccia grafica Tkinter."""
    global root, data_inizio_entry, data_fine_entry, directory_label, PERCORSO_ESTRAZIONE, mappa_ruote

    # Controllo finestra esistente
    if root is not None:
        try:
            root.deiconify(); root.lift(); root.focus_force()
            print("Finestra Tkinter già esistente.")
            return
        except tk.TclError: root = None

    print(f"Percorso base rilevato: {PERCORSO_ESTRAZIONE}")
    # Popola mappa ruote globale all'inizio
    mappa_ruote = { "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt", "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt", "PA": "PALERMO.txt", "RM": "ROMA.txt", "NZ": "NAZIONALE.txt", "TO": "TORINO.txt", "VE": "VENEZIA.txt" }

    root = tk.Tk()
    root.title("Aggiornamento Dati Lotto")
    root.geometry("480x400")

    # --- Stile ---
    BG_COLOR="#f0f0f0"; BTN_COLOR_OK="#4CAF50"; BTN_COLOR_INFO="#2196F3"; BTN_COLOR_WARN="#f44336"; BTN_COLOR_NEUTRAL="#e0e0e0"; FG_COLOR_WHITE="white"; FG_COLOR_DARK="#333333"; FONT_DEFAULT=("Segoe UI", 10); FONT_LABEL=("Segoe UI", 9); FONT_INFO=("Segoe UI", 8)
    root.config(bg=BG_COLOR)

    # --- Frame Principale ---
    main_frame = tk.Frame(root, padx=15, pady=15, bg=BG_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(1, weight=1) # Colonna entry si espande

    # --- Sezione Directory ---
    dir_frame = tk.Frame(main_frame, bg=BG_COLOR)
    dir_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    dir_frame.columnconfigure(0, weight=1)
    directory_label = tk.Label(dir_frame, text=f"Directory Dati: {PERCORSO_ESTRAZIONE}", anchor='w', wraplength=400, justify=tk.LEFT, bg=BG_COLOR, fg=FG_COLOR_DARK, font=FONT_LABEL)
    directory_label.grid(row=0, column=0, sticky='ew', padx=(0, 5))
    tk.Button(dir_frame, text="Cambia...", command=select_directory, bg=BTN_COLOR_NEUTRAL, font=FONT_DEFAULT, width=10).grid(row=0, column=1, sticky='e')

    # --- Separatore ---
    tk.Frame(main_frame, height=1, bg="#cccccc").grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)

    # --- Sezione Date ---
    tk.Label(main_frame, text="Data Inizio (AAAA/MM/GG):", anchor='w', bg=BG_COLOR, fg=FG_COLOR_DARK, font=FONT_LABEL).grid(row=2, column=0, padx=5, pady=5, sticky='w')
    data_inizio_entry = tk.Entry(main_frame, width=15, font=FONT_DEFAULT)
    data_inizio_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
    tk.Label(main_frame, text="Data Fine   (AAAA/MM/GG):", anchor='w', bg=BG_COLOR, fg=FG_COLOR_DARK, font=FONT_LABEL).grid(row=3, column=0, padx=5, pady=5, sticky='w')
    data_fine_entry = tk.Entry(main_frame, width=15, font=FONT_DEFAULT)
    data_fine_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')

    # --- Pulsanti Azione ---
    tk.Button(main_frame, text="Scarica e Aggiorna File Ruote", command=aggiorna_tutti_file, bg=BTN_COLOR_OK, fg=FG_COLOR_WHITE, height=2, font=FONT_DEFAULT, relief=tk.FLAT).grid(row=4, column=0, columnspan=2, padx=5, pady=(15, 5), sticky='ew')
    tk.Button(main_frame, text="Crea File Ruote Mancanti", command=crea_file, bg=BTN_COLOR_INFO, fg=FG_COLOR_WHITE, height=2, font=FONT_DEFAULT, relief=tk.FLAT).grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

    # --- Pulsante Chiusura Esplicito (MODIFICATO) ---
    # MODIFICA 4: Cambia il comando del bottone "Chiudi"
    tk.Button(main_frame, text="Chiudi Finestra", command=close_window, bg=BTN_COLOR_WARN, fg=FG_COLOR_WHITE, font=FONT_DEFAULT, relief=tk.FLAT).grid(row=6, column=0, columnspan=2, padx=5, pady=(10, 5), sticky='ew')

    # --- Etichetta Informativa Inferiore ---
    tk.Label(main_frame, text="Nota: L'aggiornamento cancella e riscrive i file delle ruote nella directory selezionata per il periodo indicato.", font=FONT_INFO, fg="#666666", bg=BG_COLOR, wraplength=440, justify=tk.LEFT).grid(row=7, column=0, columnspan=2, padx=5, pady=(10, 0), sticky='w')

    # --- MODIFICA 5: Gestisci chiusura con 'X' ---
    root.protocol("WM_DELETE_WINDOW", close_window)

    # Centra finestra (codice opzionale)
    # root.update_idletasks(); width=root.winfo_width(); height=root.winfo_height(); x=(root.winfo_screenwidth()//2)-(width//2); y=(root.winfo_screenheight()//2)-(height//2); root.geometry(f'{width}x{height}+{x}+{y}')

    print("Avvio Tkinter mainloop...")
    root.mainloop()
    print("Tkinter mainloop terminato.")
    # Ritorna True come originale, anche se non strettamente necessario
    return True

# --- Blocco di Esecuzione ---
if __name__ == "__main__":
    main()
    print("Script principale terminato.")