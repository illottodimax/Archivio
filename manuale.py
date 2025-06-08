
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkcalendar import DateEntry
import os

# Mappatura NOME COMPLETO -> NOME FILE (usato per trovare il file)
# e NOME COMPLETO -> SIGLA (usato per scrivere nel file)
RUOTE_INFO = {
    'BARI': {'nome_file': 'Bari.txt', 'sigla': 'BA'},
    'CAGLIARI': {'nome_file': 'Cagliari.txt', 'sigla': 'CA'},
    'FIRENZE': {'nome_file': 'Firenze.txt', 'sigla': 'FI'},
    'GENOVA': {'nome_file': 'Genova.txt', 'sigla': 'GE'},
    'MILANO': {'nome_file': 'Milano.txt', 'sigla': 'MI'},
    'NAPOLI': {'nome_file': 'Napoli.txt', 'sigla': 'NA'},
    'PALERMO': {'nome_file': 'Palermo.txt', 'sigla': 'PA'},
    'ROMA': {'nome_file': 'Roma.txt', 'sigla': 'RO'},
    'TORINO': {'nome_file': 'Torino.txt', 'sigla': 'TO'},
    'VENEZIA': {'nome_file': 'Venezia.txt', 'sigla': 'VE'},
    'NAZIONALE': {'nome_file': 'Nazionale.txt', 'sigla': 'RN'} # Standard comune per la Nazionale
}

def aggiungi_estrazione():
    """
    Valida i dati inseriti dall'utente e li scrive nel file corretto.
    """
    global feedback_var
    
    feedback_var.set("")

    # 1. Recupero e Validazione Input
    cartella_path = cartella_entry.get()
    if not cartella_path or not os.path.isdir(cartella_path):
        messagebox.showerror("Errore", "Seleziona una cartella locale valida prima di aggiungere un'estrazione.")
        return
        
    try:
        data_estrazione = date_entry.get_date()
        ruota_selezionata_key = combo_ruota.get()
        numeri_str_raw = [e.get().strip() for e in entry_numeri]

        if not ruota_selezionata_key:
            messagebox.showerror("Input Mancante", "Selezionare una ruota.")
            return

        if any(not n_str.isdigit() for n_str in numeri_str_raw):
            messagebox.showerror("Input Invalido", "Tutti i campi numero devono contenere solo cifre.")
            return
            
        if len(numeri_str_raw) != 5 or any(n == "" for n in numeri_str_raw):
             messagebox.showerror("Input Invalido", "Inserire esattamente 5 numeri.")
             return
             
        numeri_int = [int(n) for n in numeri_str_raw]
        
        if not all(1 <= n <= 90 for n in numeri_int):
            messagebox.showerror("Input Invalido", "Tutti i numeri devono essere compresi tra 1 e 90.")
            return
            
        if len(set(numeri_int)) != 5:
            messagebox.showerror("Input Invalido", "I numeri inseriti devono essere tutti diversi.")
            return

    except Exception as e:
        messagebox.showerror("Errore Input", f"Errore durante la lettura dei dati: {e}")
        return

    # 2. Gestione File
    info_ruota = RUOTE_INFO[ruota_selezionata_key]
    nome_file_ruota = info_ruota['nome_file'].upper()
    sigla_ruota = info_ruota['sigla']
    percorso_file_completo = os.path.join(cartella_path, nome_file_ruota)

    if not os.path.exists(percorso_file_completo):
        risposta = messagebox.askyesno("File non Trovato", f"Il file '{nome_file_ruota}' non esiste nella cartella selezionata.\nVuoi crearlo ora?")
        if not risposta:
            feedback_var.set("Operazione annullata.")
            return
    
    # 3. Formattazione e Controllo Duplicati
    data_formattata = data_estrazione.strftime('%Y/%m/%d')
    numeri_formattati = [str(n) for n in numeri_int]
    
    # --- MODIFICA DEFINITIVA ---
    # Crea una lista con tutti gli elementi e uniscili con il tabulatore
    elementi_riga = [data_formattata, sigla_ruota] + numeri_formattati
    nuova_riga = "\t".join(elementi_riga) + "\n"
    
    chiave_duplicato = f"{data_formattata}_{sigla_ruota}"
    try:
        if os.path.exists(percorso_file_completo):
            with open(percorso_file_completo, 'r', encoding='utf-8') as f:
                for linea_esistente in f:
                    parti = linea_esistente.strip().split()
                    if len(parti) >= 2:
                        chiave_esistente = f"{parti[0]}_{parti[1].upper()}"
                        if chiave_esistente == chiave_duplicato:
                            messagebox.showerror("Estrazione Duplicata", f"Un'estrazione per la ruota {ruota_selezionata_key} in data {data_estrazione.strftime('%d/%m/%Y')} esiste già.")
                            return
    except Exception as e:
        messagebox.showerror("Errore Lettura", f"Impossibile leggere il file per il controllo duplicati:\n{e}")
        return

    # 4. Scrittura su File
    try:
        with open(percorso_file_completo, 'a+', encoding='utf-8') as f:
            if os.path.getsize(percorso_file_completo) > 0:
                f.seek(0, os.SEEK_END)
                f.seek(f.tell() - 1, os.SEEK_SET)
                if f.read(1) != '\n':
                    f.write('\n')
            f.write(nuova_riga)
        
        feedback_var.set(f"Estrazione aggiunta a '{nome_file_ruota}' con successo!")
        for entry in entry_numeri:
            entry.delete(0, tk.END)

    except Exception as e:
        messagebox.showerror("Errore Scrittura", f"Impossibile scrivere sul file:\n{e}")
        feedback_var.set("Errore durante il salvataggio.")

def seleziona_cartella():
    cartella_selezionata = filedialog.askdirectory(title="Seleziona la cartella con i file delle estrazioni")
    if cartella_selezionata:
        cartella_entry.config(state=tk.NORMAL)
        cartella_entry.delete(0, tk.END)
        cartella_entry.insert(0, cartella_selezionata)
        cartella_entry.config(state='readonly')
        with open("config_inserimento.txt", "w") as f:
            f.write(cartella_selezionata)

# --- Creazione GUI ---
root = tk.Tk()
root.title("Inserimento Manuale Estrazioni")
root.geometry("500x380") # Aumentata leggermente l'altezza per la nuova etichetta
root.minsize(450, 350)

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", font=("Segoe UI", 10), padding=5)
style.configure("TLabel", font=("Segoe UI", 10))
style.configure("Small.TLabel", font=("Segoe UI", 8), foreground="gray") # Stile per la nuova etichetta
style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Sezione Selezione Cartella
cartella_frame_outer = ttk.LabelFrame(main_frame, text=" 1. Cartella Archivi ", padding=10)
cartella_frame_outer.pack(fill=tk.X, pady=(0, 10))

cartella_entry = ttk.Entry(cartella_frame_outer, width=60, state='readonly')
cartella_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

btn_sfoglia = ttk.Button(cartella_frame_outer, text="Sfoglia...", command=seleziona_cartella)
btn_sfoglia.pack(side=tk.LEFT)

# Sezione Inserimento Dati
inserimento_frame = ttk.LabelFrame(main_frame, text=" 2. Dati Nuova Estrazione ", padding=15)
inserimento_frame.pack(pady=10, padx=10, fill=tk.X)

ttk.Label(inserimento_frame, text="Data Estrazione:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
date_entry = DateEntry(inserimento_frame, width=12, background='#3498db', foreground='white', borderwidth=2, date_pattern='dd/mm/yyyy')
date_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

ttk.Label(inserimento_frame, text="Ruota:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
combo_ruota = ttk.Combobox(inserimento_frame, values=list(RUOTE_INFO.keys()), state="readonly", width=15)
combo_ruota.grid(row=1, column=1, padx=5, pady=5, sticky='w')

ttk.Label(inserimento_frame, text="Numeri Estratti:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
numeri_frame = ttk.Frame(inserimento_frame)
numeri_frame.grid(row=2, column=1, padx=5, pady=5, sticky='w')
entry_numeri = []
for i in range(5):
    entry = ttk.Entry(numeri_frame, width=5, justify=tk.CENTER, font=("Segoe UI", 10))
    entry.pack(side=tk.LEFT, padx=3)
    entry_numeri.append(entry)

# >>>>> MODIFICA: Aggiunta etichetta di precisazione <<<<<
precisazione_label = ttk.Label(inserimento_frame, text="I numeretti vanno inseriti senza lo 0", style="Small.TLabel")
precisazione_label.grid(row=3, column=1, padx=5, pady=(0, 10), sticky='w')


# Pulsante e feedback
btn_aggiungi = ttk.Button(inserimento_frame, text="Aggiungi Estrazione", command=aggiungi_estrazione)
btn_aggiungi.grid(row=4, column=0, columnspan=2, pady=10, padx=5, ipady=4)

feedback_var = tk.StringVar()
feedback_label = ttk.Label(inserimento_frame, textvariable=feedback_var, style="Small.TLabel", foreground="blue")
feedback_label.grid(row=5, column=0, columnspan=2, pady=5)

# Carica l'ultimo percorso usato, se esiste
if os.path.exists("config_inserimento.txt"):
    with open("config_inserimento.txt", "r") as f:
        last_path = f.read().strip()
        if os.path.isdir(last_path):
            cartella_entry.config(state=tk.NORMAL)
            cartella_entry.insert(0, last_path)
            cartella_entry.config(state='readonly')
else:
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    default_folder_path = os.path.join(desktop_path, "NUMERICAL_EMPATHY_COMPLETO_2025")
    if os.path.isdir(default_folder_path):
        cartella_entry.config(state=tk.NORMAL)
        cartella_entry.insert(0, default_folder_path)
        cartella_entry.config(state='readonly')
        
root.mainloop()