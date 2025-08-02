import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import requests
import zipfile
import io
import sys
from datetime import date
import datetime # Aggiunto per la validazione della data

# --- Configurazione GitHub ---
# Inserisci qui le informazioni del tuo repository.
# Se cambi repository o utente, modifica solo queste righe.
GITHUB_USER = "illottodimax"
GITHUB_REPO = "Archivio"
GITHUB_BRANCH = "main"

# Mappa delle ruote usata in tutto lo script per coerenza.
# Associa il codice usato internamente al nome del file.
RUOTE_MAPPA = {
    "BA": "BARI.txt", "CA": "CAGLIARI.txt", "FI": "FIRENZE.txt",
    "GE": "GENOVA.txt", "MI": "MILANO.txt", "NA": "NAPOLI.txt",
    "PA": "PALERMO.txt", "RM": "ROMA.txt", "TO": "TORINO.txt",
    "VE": "VENEZIA.txt", "NZ": "NAZIONALE.txt"
}

# Costruisce dinamicamente gli URL per il download raw da GitHub
GITHUB_URLS = {
    codice_ruota: f'https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{nome_file}'
    for codice_ruota, nome_file in RUOTE_MAPPA.items()
}
# --- Fine Configurazione GitHub ---


# --- Variabili Globali ---
root = None
data_source_var = None
data_inizio_entry = None
data_fine_entry = None
directory_label = None
PERCORSO_ESTRAZIONE = None


# --- Funzioni di Utilità ---

def get_base_path():
    """Restituisce il percorso base dello script o dell'eseguibile."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

PERCORSO_ESTRAZIONE = get_base_path()

def select_directory():
    """Permette all'utente di selezionare una nuova directory per i dati."""
    global PERCORSO_ESTRAZIONE, directory_label
    directory = filedialog.askdirectory(title="Seleziona directory per i file", initialdir=PERCORSO_ESTRAZIONE)
    if directory:
        PERCORSO_ESTRAZIONE = directory
        if directory_label:
            directory_label.config(text=f"Directory Dati: {PERCORSO_ESTRAZIONE}")

def scarica_e_estrai_zip(url, destinazione):
    """Scarica ed estrae un file ZIP da un URL."""
    try:
        print(f"Tentativo di download ZIP da: {url}")
        # Usare uno User-Agent comune può aiutare a evitare blocchi 403
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=60, headers=headers)
        response.raise_for_status()
        print(f"Download ZIP completato ({len(response.content)} bytes). Estrazione in: {destinazione}")
        os.makedirs(destinazione, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(destinazione)
        print("Estrazione ZIP completata.")
        return True
    except requests.exceptions.HTTPError as e:
        messagebox.showerror("Errore Download", f"Errore HTTP durante il download da fonte ufficiale (Server ha risposto con errore):\n{e}")
        return False
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Errore Download", f"Errore di rete durante il download da fonte ufficiale:\n{e}")
        return False
    except Exception as e:
        messagebox.showerror("Errore", f"Errore imprevisto durante download/estrazione ZIP:\n{e}")
        return False

# --- NUOVA FUNZIONE PER SCARICARE DA GITHUB ---
def scarica_da_github(destinazione):
    """Scarica tutti i file delle ruote singolarmente da GitHub."""
    print("Avvio download da GitHub...")
    os.makedirs(destinazione, exist_ok=True)
    file_scaricati = 0
    file_falliti = []

    for codice_ruota, url in GITHUB_URLS.items():
        try:
            print(f"  -> Scaricando {RUOTE_MAPPA[codice_ruota]} da {url}...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            percorso_file_dest = os.path.join(destinazione, RUOTE_MAPPA[codice_ruota])
            # Scriviamo il contenuto in binario per evitare problemi di encoding
            with open(percorso_file_dest, 'wb') as f:
                f.write(response.content)
            
            file_scaricati += 1

        except requests.exceptions.RequestException as e:
            print(f"Errore scaricando {RUOTE_MAPPA[codice_ruota]}: {e}")
            file_falliti.append(RUOTE_MAPPA[codice_ruota])
        except Exception as e:
            print(f"Errore sconosciuto per {RUOTE_MAPPA[codice_ruota]}: {e}")
            file_falliti.append(RUOTE_MAPPA[codice_ruota])

    if not file_falliti and file_scaricati > 0:
        messagebox.showinfo("Successo", f"Aggiornamento da GitHub completato.\n{file_scaricati} file ruota sono stati scaricati e salvati in:\n{destinazione}")
        return True
    else:
        messaggio_errore = f"Aggiornamento da GitHub parzialmente fallito.\n\nFile scaricati: {file_scaricati}\nFile falliti: {len(file_falliti)}"
        if file_falliti:
            messaggio_errore += f"\n({', '.join(file_falliti)})"
        messagebox.showerror("Errore Aggiornamento GitHub", messaggio_errore)
        return False


def carica_dati_da_file(nome_file):
    """Carica le righe da un file di testo."""
    if not os.path.exists(nome_file):
         print(f"File non trovato: {nome_file}")
         return []
    try:
        with open(nome_file, 'r', encoding='utf-8', errors='ignore') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        messagebox.showerror("Errore Lettura File", f"Errore durante la lettura del file {os.path.basename(nome_file)}:\n{e}")
        return []

def salva_dati_su_file(dati_ruote_da_aggiungere):
    """Aggiunge nuove righe ai file delle ruote, evitando duplicati."""
    global PERCORSO_ESTRAZIONE

    successo_parziale = True
    file_modificati_conteggio = 0
    for ruota, dati_nuovi in dati_ruote_da_aggiungere.items():
        if dati_nuovi:
            nome_file_relativo = RUOTE_MAPPA.get(ruota)
            if not nome_file_relativo:
                print(f"Attenzione: Chiave ruota '{ruota}' non trovata nella mappa. Salto.")
                continue

            nome_file = os.path.join(PERCORSO_ESTRAZIONE, nome_file_relativo)
            try:
                os.makedirs(os.path.dirname(nome_file), exist_ok=True)
                with open(nome_file, 'a', encoding='utf-8') as file:
                    for line in dati_nuovi:
                        file.write(line + '\n')
                file_modificati_conteggio +=1
                print(f"Aggiunte {len(dati_nuovi)} estrazioni a {os.path.basename(nome_file)}")
            except Exception as e:
                messagebox.showerror("Errore Scrittura File", f"Errore durante la scrittura del file {os.path.basename(nome_file)}:\n{e}")
                successo_parziale = False
    
    return successo_parziale, file_modificati_conteggio


def aggiorna_file_da_storico(data_inizio, data_fine, percorso_estrazione):
    """Logica per aggiornare i file ruota partendo da un file storico.txt."""
    storico_file_path = os.path.join(percorso_estrazione, 'storico.txt')
    dati_storico = carica_dati_da_file(storico_file_path)

    if not dati_storico:
        messagebox.showerror("Errore", f"Impossibile leggere storico.txt o file vuoto in:\n{percorso_estrazione}")
        return

    estrazioni_da_aggiungere = {ruota: [] for ruota in RUOTE_MAPPA.keys()}
    estrazioni_nuove_trovate_nel_range = False
    
    estrazioni_esistenti_per_ruota = {}
    for ruota_code, nome_file_relativo in RUOTE_MAPPA.items():
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

        if ruota_code_storico == "RN":
            ruota_code_storico = "NZ"

        if ruota_code_storico in RUOTE_MAPPA:
            line_storico_normalizzata = ' '.join(line_storico.strip().split())
            if line_storico_normalizzata not in estrazioni_esistenti_per_ruota.get(ruota_code_storico, set()):
                estrazioni_da_aggiungere[ruota_code_storico].append(line_storico.strip())
                estrazioni_nuove_trovate_nel_range = True

    if not estrazioni_nuove_trovate_nel_range:
         messagebox.showinfo("Info", f"Nessuna NUOVA estrazione da aggiungere trovata in storico.txt per l'intervallo {data_inizio} - {data_fine}.")
         return

    successo_scrittura, file_aggiornati_count = salva_dati_su_file(estrazioni_da_aggiungere)

    if successo_scrittura and file_aggiornati_count > 0:
        messagebox.showinfo("Successo", f"Aggiunte nuove estrazioni a {file_aggiornati_count} file ruota in:\n{percorso_estrazione}\nPeriodo analizzato: {data_inizio} - {data_fine}.")
    elif successo_scrittura and file_aggiornati_count == 0:
        messagebox.showinfo("Info", "Tutte le estrazioni nell'intervallo specificato sono già presenti nei file delle ruote.")
    else:
        messagebox.showwarning("Attenzione", "L'aggiornamento incrementale non è stato completato correttamente.")


# <<< MODIFICA/AGGIUNTA >>>
# La funzione principale è stata riorganizzata per gestire le tre fonti dati.
def aggiorna_tutti_file():
    """Funzione principale chiamata dal pulsante, gestisce le tre fonti."""
    global PERCORSO_ESTRAZIONE, data_inizio_entry, data_fine_entry, data_source_var
    
    source = data_source_var.get()
    
    if source == 'GitHub':
        # Se la fonte è GitHub, scarichiamo semplicemente i file e finiamo.
        scarica_da_github(PERCORSO_ESTRAZIONE)
        return

    # Se la fonte è Ufficiale, scarica e scompatta prima di procedere.
    if source == 'Ufficiale':
        url_storico = "https://www.igt.it/STORICO_ESTRAZIONI_LOTTO/storico.zip"
        print("Avvio download da fonte ufficiale...")
        if not scarica_e_estrai_zip(url_storico, PERCORSO_ESTRAZIONE):
            print("Download da fonte ufficiale fallito.")
            # Chiediamo se vuole provare comunque con un file locale
            if not messagebox.askyesno("Download Fallito", "Il download da igt.it è fallito.\n\nVuoi provare ad aggiornare usando un file 'storico.txt' locale (se esistente)?"):
                return
        
        # Dopo il download (o se l'utente vuole procedere comunque), verifichiamo che storico.txt esista
        storico_file_path = os.path.join(PERCORSO_ESTRAZIONE, 'storico.txt')
        if not os.path.exists(storico_file_path):
            messagebox.showerror("Errore", f"'storico.txt' non trovato in:\n{PERCORSO_ESTRAZIONE}\n\nImpossibile procedere.")
            return

        # Correzione robusta RN -> NZ direttamente nel file scaricato
        print("Controllo e correzione RN -> NZ in storico.txt (se necessario)...")
        try:
            with open(storico_file_path, 'r+', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                # Usiamo spazi per evitare di sostituire 'RN' in altre parti del testo
                if ' RN ' in content:
                    print("Trovate occorrenze di ' RN ', applico la correzione.")
                    file.seek(0) # Torna all'inizio del file per sovrascrivere
                    file.write(content.replace(' RN ', ' NZ '))
                    file.truncate() # Rimuove il contenuto rimanente se il nuovo testo è più corto
                    print("Correzione RN -> NZ applicata a storico.txt.")
        except Exception as e:
            messagebox.showerror("Errore Correzione File", f"Errore durante la correzione RN->NZ in storico.txt:\n{e}")
            return

    # --- Logica comune per "Ufficiale" e "Locale" che usano storico.txt ---
    data_inizio = data_inizio_entry.get().strip()
    data_fine = data_fine_entry.get().strip()

    try:
        datetime.datetime.strptime(data_inizio, '%Y/%m/%d')
        datetime.datetime.strptime(data_fine, '%Y/%m/%d')
    except ValueError:
        messagebox.showerror("Errore Formato Data", "Formato data non valido. Usare AAAA/MM/GG.")
        return

    if data_inizio > data_fine:
        messagebox.showerror("Errore Date", "La data di inizio non può essere successiva alla data di fine.")
        return

    print(f"Avvio aggiornamento da file storico per periodo {data_inizio} - {data_fine}...")
    aggiorna_file_da_storico(data_inizio, data_fine, PERCORSO_ESTRAZIONE)


def toggle_date_fields_state(event=None):
    """Abilita o disabilita i campi data a seconda della fonte selezionata."""
    global data_source_var, data_inizio_entry, data_fine_entry
    if data_source_var.get() == 'GitHub':
        data_inizio_entry.config(state=tk.DISABLED)
        data_fine_entry.config(state=tk.DISABLED)
    else:
        data_inizio_entry.config(state=tk.NORMAL)
        data_fine_entry.config(state=tk.NORMAL)

def close_window():
    """Chiude la finestra Tkinter in modo pulito."""
    global root
    if root:
        root.destroy()
        root = None

def main():
    global root, data_source_var, data_inizio_entry, data_fine_entry, directory_label

    root = tk.Tk()
    root.title("Aggiornamento Dati Lotto")
    root.geometry("500x420")
    root.minsize(480, 360)

    # Stile per un aspetto più moderno
    style = ttk.Style(root)
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(1, weight=1)

    # --- Sezione Selezione Directory ---
    dir_frame = ttk.Frame(main_frame)
    dir_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    dir_frame.columnconfigure(0, weight=1)
    directory_label = ttk.Label(dir_frame, text=f"Directory Dati: {PERCORSO_ESTRAZIONE}", anchor='w', wraplength=400, justify=tk.LEFT)
    directory_label.grid(row=0, column=0, sticky='ew', padx=(0, 5))
    ttk.Button(dir_frame, text="Cambia...", command=select_directory).grid(row=0, column=1, sticky='e')
    
    ttk.Separator(main_frame, orient='horizontal').grid(row=1, column=0, columnspan=2, sticky='ew', pady=10)

    # --- Sezione Selezione Fonte Dati ---
    source_frame = ttk.LabelFrame(main_frame, text="1. Scegli la Fonte Dati", padding="10")
    source_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    
    # <<< MODIFICA/AGGIUNTA >>>: Aggiunta opzione "Ufficiale" e impostata come default.
    data_source_var = tk.StringVar(value="Ufficiale") 
    
    ttk.Radiobutton(source_frame, text="Fonte Ufficiale (igt.it)", variable=data_source_var, value="Ufficiale", command=toggle_date_fields_state).pack(anchor='w')
    ttk.Radiobutton(source_frame, text="Fonte Max Lotto (GitHub)", variable=data_source_var, value="GitHub", command=toggle_date_fields_state).pack(anchor='w')
    ttk.Radiobutton(source_frame, text="Locale (Usa 'storico.txt' esistente)", variable=data_source_var, value="Locale", command=toggle_date_fields_state).pack(anchor='w')
    
    # --- Sezione Date (per fonte ufficiale/locale) ---
    date_frame = ttk.LabelFrame(main_frame, text="2. Imposta Intervallo (se non usi GitHub)", padding="10")
    date_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
    date_frame.columnconfigure(1, weight=1)
    
    ttk.Label(date_frame, text="Data Inizio:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
    data_inizio_entry = ttk.Entry(date_frame, width=15)
    data_inizio_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
    data_inizio_entry.insert(0, "1939/01/07") 

    ttk.Label(date_frame, text="Data Fine:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
    data_fine_entry = ttk.Entry(date_frame, width=15)
    data_fine_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
    data_fine_entry.insert(0, date.today().strftime('%Y/%m/%d'))
    
    toggle_date_fields_state() # Imposta lo stato iniziale dei campi data

    # --- Sezione Pulsanti Azione ---
    action_frame = ttk.Frame(main_frame)
    action_frame.grid(row=4, column=0, columnspan=2, pady=(15, 5))
    action_frame.columnconfigure(0, weight=1)

    ttk.Button(action_frame, text="Avvia Aggiornamento", command=aggiorna_tutti_file, style="Accent.TButton").pack(fill='x', ipady=5, pady=(0, 5))
    ttk.Button(action_frame, text="Chiudi", command=close_window).pack(fill='x', ipady=5)
    
    style.configure("Accent.TButton", foreground="white", background="#0078D7") # Colore blu per il pulsante principale

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()

if __name__ == "__main__":
    main()
    print("Script principale terminato.")