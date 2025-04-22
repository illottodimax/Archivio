# spia_integration.py - Modulo di integrazione per spia2.py

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import pandas as pd
import numpy as np
import os
from tkcalendar import DateEntry
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import importlib.util
import itertools  # Necessario per le combinazioni
import spia2

# Riferimento alla finestra principale (sarà inizializzato nella funzione launch_spia_analysis_window)
root = None

def launch_spia_analysis_window():
    """Crea e mostra una nuova finestra (Toplevel) per l'analisi Spia."""
    import tkinter as tk
    from tkinter import messagebox
    
    # Verifica se esiste già un'istanza attiva
    global active_spia_window
    if 'active_spia_window' in globals() and active_spia_window is not None:
        try:
            if active_spia_window.winfo_exists():
                active_spia_window.lift()  # Porta in primo piano la finestra esistente
                return
        except:
            pass  # Se la verifica fallisce, continua e crea una nuova finestra
    
    # Ottieni un riferimento valido alla finestra principale
    try:
        if tk._default_root and tk._default_root.winfo_exists():
            root = tk._default_root
        else:
            # Se l'applicazione principale è stata distrutta, mostra un errore
            messagebox.showerror("Errore", "L'applicazione principale non è più disponibile.")
            return
    except Exception:
        messagebox.showerror("Errore", "Impossibile accedere all'applicazione principale.")
        return
    
    # --- Reinizializza le variabili di spia2.py se necessario ---
    try:
        import sys
        import importlib
        
        # Se spia2 è già stato importato, ricaricalo
        if 'spia2' in sys.modules:
            importlib.reload(sys.modules['spia2'])
    except Exception:
        pass  # Ignora errori nella reinizializzazione
    
    # --- Variabili globali specifiche della finestra Spia ---
    global risultati_globali_spia, info_ricerca_globale_spia, file_ruote_spia
    risultati_globali_spia = []
    info_ricerca_globale_spia = {}
    file_ruote_spia = {}
    
    # --- Finestra Toplevel ---
    try:
        spia_win = tk.Toplevel(root)
        spia_win.title("Analisi Numeri Spia (Modulo Integrato)")
        spia_win.geometry("1150x800")
        spia_win.minsize(1000, 650)
        spia_win.configure(bg="#f0f0f0")
        
        # Mantieni un riferimento globale alla finestra
        active_spia_window = spia_win
        
        # Funzione per gestire la chiusura pulita
        def on_closing():
            # Chiudi eventuali finestre secondarie
            for widget in spia_win.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    try:
                        widget.destroy()
                    except:
                        pass
            
            # Chiudi la finestra principale
            try:
                spia_win.destroy()
            except:
                pass
                
            # Chiudi esplicitamente le risorse matplotlib se utilizzate
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass
                
            # Ripulisci le variabili globali
            global active_spia_window, risultati_globali_spia, info_ricerca_globale_spia, file_ruote_spia
            active_spia_window = None
            risultati_globali_spia = []
            info_ricerca_globale_spia = {}
            file_ruote_spia = {}
            
            # Assicurati che tutte le risorse siano rilasciate
            try:
                import sys
                import gc
                
                # Rimuovi i moduli importati
                modules_to_remove = ['spia2']
                for module in modules_to_remove:
                    if module in sys.modules:
                        del sys.modules[module]
                
                # Forza la garbage collection
                gc.collect()
            except Exception:
                pass
        
        # Associa questa funzione all'evento di chiusura
        spia_win.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Imposta la finestra come transiente della finestra principale
        # Questo aiuta a gestire meglio la relazione tra le finestre
        if root != spia_win:
            spia_win.transient(root)
            
        return spia_win  # Ritorna il riferimento alla finestra creata
        
    except Exception as e:
        messagebox.showerror("Errore", f"Errore durante la creazione della finestra: {e}")
        return None
    
    # Funzione per gestire la chiusura pulita
    def on_closing():
        # Chiudi la finestra
        spia_win.destroy()
        
        # Assicurati che tutte le risorse siano rilasciate
        try:
            import sys
            # Rimuovi il modulo spia2 da sys.modules
            if 'spia2' in sys.modules:
                del sys.modules['spia2']
        except Exception:
            pass
    
    # Associa questa funzione all'evento di chiusura
    spia_win.protocol("WM_DELETE_WINDOW", on_closing)
    
    # --- Configurazione dello stile per questa finestra ---
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10), padding=5)
    style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
    style.configure("Small.TLabel", background="#f0f0f0", font=("Segoe UI", 8))
    style.configure("TEntry", padding=3)
    style.configure("TListbox", font=("Consolas", 10))
    style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background="#f0f0f0")
    style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 10))
    
    # --- Frame principale ---
    main_frame = ttk.Frame(spia_win, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # --- Frame per cartella ---
    cartella_frame = ttk.Frame(main_frame)
    cartella_frame.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(cartella_frame, text="Cartella Estrazioni:", style="Title.TLabel").pack(side=tk.LEFT, padx=(0, 5))
    cartella_entry = ttk.Entry(cartella_frame)
    cartella_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # --- Funzioni di gestione file ---
    def mappa_file_ruote_spia():
        """Legge la cartella e aggiorna la mappa globale file_ruote_spia."""
        global file_ruote_spia
        cartella = cartella_entry.get()
        file_ruote_spia.clear() # Svuota prima di riempire
        
        if not cartella or not os.path.isdir(cartella):
            return False
        
        ruote_valide = ['BARI', 'CAGLIARI', 'FIRENZE', 'GENOVA', 'MILANO',
                        'NAPOLI', 'PALERMO', 'ROMA', 'TORINO', 'VENEZIA', 'NAZIONALE']
        try:
            found = False
            for file in os.listdir(cartella):
                fp = os.path.join(cartella, file)
                if os.path.isfile(fp) and file.lower().endswith(".txt"):
                    nome_r = os.path.splitext(file)[0].upper()
                    if nome_r in ruote_valide:
                        file_ruote_spia[nome_r] = fp
                        found = True
            
            return found
            
        except Exception as e:
            messagebox.showerror("Errore Lettura Cartella", 
                               f"Impossibile leggere la cartella:\n{e}", parent=spia_win)
            traceback.print_exc()
            return False
    
    # --- Funzione aggiorna listbox ---
    def aggiorna_lista_file_gui(target_listbox):
        target_listbox.delete(0, tk.END)
        ruote_ordinate = sorted(file_ruote_spia.keys())
        for r in ruote_ordinate:
            target_listbox.insert(tk.END, r)
    
    # --- Funzione sfoglia ---
    def on_sfoglia_click():
        cartella_sel = filedialog.askdirectory(title="Seleziona Cartella Estrazioni", parent=spia_win)
        if cartella_sel:
            cartella_entry.delete(0, tk.END)
            cartella_entry.insert(0, cartella_sel)
            if mappa_file_ruote_spia():
                # Aggiorna liste file
                aggiorna_lista_file_gui(listbox_ruote_analisi)
                aggiorna_lista_file_gui(listbox_ruote_verifica)
                aggiorna_lista_file_gui(listbox_ruote_analisi_ant)
            else:
                listbox_ruote_analisi.delete(0, tk.END)
                listbox_ruote_verifica.delete(0, tk.END)
                listbox_ruote_analisi_ant.delete(0, tk.END)
    
    # --- Bottone sfoglia ---
    btn_sfoglia = ttk.Button(cartella_frame, text="Sfoglia...", command=on_sfoglia_click)
    btn_sfoglia.pack(side=tk.LEFT, padx=5)
    
    # --- Notebook principale ---
    notebook = ttk.Notebook(main_frame, style="TNotebook")
    notebook.pack(fill=tk.X, pady=10)
    
    # --- Tab Successivi ---
    tab_successivi = ttk.Frame(notebook, padding=10)
    notebook.add(tab_successivi, text=' Analisi Numeri Successivi (E/A/T) ')
    
    controls_frame_succ = ttk.Frame(tab_successivi)
    controls_frame_succ.pack(fill=tk.X)
    controls_frame_succ.columnconfigure(0, weight=1)
    controls_frame_succ.columnconfigure(1, weight=1)
    controls_frame_succ.columnconfigure(2, weight=0)
    controls_frame_succ.columnconfigure(3, weight=0)
    
    # Ruote Analisi (1)
    ruote_analisi_outer_frame = ttk.Frame(controls_frame_succ)
    ruote_analisi_outer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    ttk.Label(ruote_analisi_outer_frame, text="1. Ruote Analisi:", style="Title.TLabel").pack(anchor="w")
    ttk.Label(ruote_analisi_outer_frame, text="(CTRL/SHIFT per multipla)", style="Small.TLabel").pack(anchor="w", pady=(0, 5))
    
    ruote_analisi_list_frame = ttk.Frame(ruote_analisi_outer_frame)
    ruote_analisi_list_frame.pack(fill=tk.BOTH, expand=True)
    scrollbar_ruote_analisi = ttk.Scrollbar(ruote_analisi_list_frame)
    scrollbar_ruote_analisi.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox_ruote_analisi = tk.Listbox(ruote_analisi_list_frame, height=10, selectmode=tk.EXTENDED, 
                                      exportselection=False, font=("Consolas", 10), 
                                      selectbackground="#005A9E", selectforeground="white", 
                                      yscrollcommand=scrollbar_ruote_analisi.set)
    listbox_ruote_analisi.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_ruote_analisi.config(command=listbox_ruote_analisi.yview)
    
    # Ruote Verifica (3)
    ruote_verifica_outer_frame = ttk.Frame(controls_frame_succ)
    ruote_verifica_outer_frame.grid(row=0, column=1, sticky="nsew", padx=5)
    ttk.Label(ruote_verifica_outer_frame, text="3. Ruote Verifica:", style="Title.TLabel").pack(anchor="w")
    ttk.Label(ruote_verifica_outer_frame, text="(CTRL/SHIFT per multipla)", style="Small.TLabel").pack(anchor="w", pady=(0, 5))
    
    ruote_verifica_list_frame = ttk.Frame(ruote_verifica_outer_frame)
    ruote_verifica_list_frame.pack(fill=tk.BOTH, expand=True)
    scrollbar_ruote_verifica = ttk.Scrollbar(ruote_verifica_list_frame)
    scrollbar_ruote_verifica.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox_ruote_verifica = tk.Listbox(ruote_verifica_list_frame, height=10, selectmode=tk.EXTENDED, 
                                      exportselection=False, font=("Consolas", 10), 
                                      selectbackground="#005A9E", selectforeground="white", 
                                      yscrollcommand=scrollbar_ruote_verifica.set)
    listbox_ruote_verifica.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_ruote_verifica.config(command=listbox_ruote_verifica.yview)
    
    # Center controls (2, 4, 5)
    center_controls_frame_succ = ttk.Frame(controls_frame_succ)
    center_controls_frame_succ.grid(row=0, column=2, sticky="ns", padx=5)
    
    # Numeri Spia (2)
    spia_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 2. Numeri Spia (1-90) ", padding=5)
    spia_frame_succ.pack(fill=tk.X, pady=(0, 5))
    
    spia_entry_container_succ = ttk.Frame(spia_frame_succ)
    spia_entry_container_succ.pack(fill=tk.X, pady=5)
    
    entry_numeri_spia = []
    for i in range(5):
        entry = ttk.Entry(spia_entry_container_succ, width=5, justify=tk.CENTER, font=("Segoe UI", 10))
        entry.pack(side=tk.LEFT, padx=3, ipady=2)
        entry_numeri_spia.append(entry)
    
    # Estrazioni Successive (4)
    estrazioni_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 4. Estrazioni Successive ", padding=5)
    estrazioni_frame_succ.pack(fill=tk.X, pady=5)
    
    ttk.Label(estrazioni_frame_succ, text="Quante (1-18):", style="Small.TLabel").pack(anchor="w")
    estrazioni_entry_succ = ttk.Entry(estrazioni_frame_succ, width=5, justify=tk.CENTER, font=("Segoe UI", 10))
    estrazioni_entry_succ.pack(anchor="w", pady=2, ipady=2)
    estrazioni_entry_succ.insert(0, "5")
    
    # Verifica Esiti (5)
    verifica_frame_succ = ttk.LabelFrame(center_controls_frame_succ, text=" 5. Verifica Esiti ", padding=5)
    verifica_frame_succ.pack(fill=tk.X, pady=5)
    
    ttk.Label(verifica_frame_succ, text="Estrazioni Verifica (1-18):", style="Small.TLabel").pack(anchor="w")
    estrazioni_entry_verifica = ttk.Entry(verifica_frame_succ, width=5, justify=tk.CENTER, font=("Segoe UI", 10))
    estrazioni_entry_verifica.pack(anchor="w", pady=2, ipady=2)
    estrazioni_entry_verifica.insert(0, "9")
    
    # Buttons frame
    buttons_frame_succ = ttk.Frame(controls_frame_succ)
    buttons_frame_succ.grid(row=0, column=3, sticky="ns", padx=(10, 0))
    
    # --- Tab Antecedenti ---
    tab_antecedenti = ttk.Frame(notebook, padding=10)
    notebook.add(tab_antecedenti, text=' Analisi Numeri Antecedenti (Marker) ')
    
    controls_frame_ant = ttk.Frame(tab_antecedenti)
    controls_frame_ant.pack(fill=tk.X)
    controls_frame_ant.columnconfigure(0, weight=1)
    controls_frame_ant.columnconfigure(1, weight=0)
    controls_frame_ant.columnconfigure(2, weight=0)
    
    # Ruote da Analizzare (1)
    ruote_analisi_ant_outer_frame = ttk.Frame(controls_frame_ant)
    ruote_analisi_ant_outer_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    ttk.Label(ruote_analisi_ant_outer_frame, text="1. Ruote da Analizzare:", style="Title.TLabel").pack(anchor="w")
    ttk.Label(ruote_analisi_ant_outer_frame, text="(Obiettivo e antecedenti cercati qui)", style="Small.TLabel").pack(anchor="w", pady=(0, 5))
    
    ruote_analisi_ant_list_frame = ttk.Frame(ruote_analisi_ant_outer_frame)
    ruote_analisi_ant_list_frame.pack(fill=tk.BOTH, expand=True)
    scrollbar_ruote_analisi_ant = ttk.Scrollbar(ruote_analisi_ant_list_frame)
    scrollbar_ruote_analisi_ant.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox_ruote_analisi_ant = tk.Listbox(ruote_analisi_ant_list_frame, height=10, selectmode=tk.EXTENDED, 
                                         exportselection=False, font=("Consolas", 10), 
                                         selectbackground="#005A9E", selectforeground="white", 
                                         yscrollcommand=scrollbar_ruote_analisi_ant.set)
    listbox_ruote_analisi_ant.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_ruote_analisi_ant.config(command=listbox_ruote_analisi_ant.yview)
    
    # Center controls ant (2, 3)
    center_controls_frame_ant = ttk.Frame(controls_frame_ant)
    center_controls_frame_ant.grid(row=0, column=1, sticky="ns", padx=10)
    
    # Numeri Obiettivo (2)
    obiettivo_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 2. Numeri Obiettivo (1-90) ", padding=5)
    obiettivo_frame_ant.pack(fill=tk.X, pady=(0, 5))
    
    obiettivo_entry_container_ant = ttk.Frame(obiettivo_frame_ant)
    obiettivo_entry_container_ant.pack(fill=tk.X, pady=5)
    
    entry_numeri_obiettivo = []
    for i in range(5):
        entry = ttk.Entry(obiettivo_entry_container_ant, width=5, justify=tk.CENTER, font=("Segoe UI", 10))
        entry.pack(side=tk.LEFT, padx=3, ipady=2)
        entry_numeri_obiettivo.append(entry)
    
    # Estrazioni Precedenti (3)
    estrazioni_frame_ant = ttk.LabelFrame(center_controls_frame_ant, text=" 3. Estrazioni Precedenti ", padding=5)
    estrazioni_frame_ant.pack(fill=tk.X, pady=5)
    
    ttk.Label(estrazioni_frame_ant, text="Quante controllare (1-18):", style="Small.TLabel").pack(anchor="w")
    estrazioni_entry_ant = ttk.Entry(estrazioni_frame_ant, width=5, justify=tk.CENTER, font=("Segoe UI", 10))
    estrazioni_entry_ant.pack(anchor="w", pady=2, ipady=2)
    estrazioni_entry_ant.insert(0, "3")
    
    # Buttons ant
    buttons_frame_ant = ttk.Frame(controls_frame_ant)
    buttons_frame_ant.grid(row=0, column=2, sticky="ns", padx=(10, 0))
    
    # --- Date frame (common) ---
    common_controls_frame = ttk.Frame(main_frame)
    common_controls_frame.pack(fill=tk.X, pady=5)
    
    dates_frame = ttk.LabelFrame(common_controls_frame, text=" Periodo Analisi (Comune) ", padding=5)
    dates_frame.pack(side=tk.LEFT, padx=(0,10))
    dates_frame.columnconfigure(1, weight=1)
    
    ttk.Label(dates_frame, text="Da:", anchor="e").grid(row=0, column=0, padx=2, pady=2, sticky="w")
    start_date_default = datetime.date.today() - datetime.timedelta(days=365*3)
    start_date_entry = DateEntry(dates_frame, width=10, background='#3498db', foreground='white', 
                               borderwidth=2, date_pattern='yyyy-mm-dd', font=("Segoe UI", 9),
                               year=start_date_default.year, month=start_date_default.month, 
                               day=start_date_default.day)
    start_date_entry.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
    
    ttk.Label(dates_frame, text="A:", anchor="e").grid(row=1, column=0, padx=2, pady=2, sticky="w")
    end_date_entry = DateEntry(dates_frame, width=10, background='#3498db', foreground='white', 
                             borderwidth=2, date_pattern='yyyy-mm-dd', font=("Segoe UI", 9))
    end_date_entry.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
    
    # Common buttons frame
    common_buttons_frame = ttk.Frame(common_controls_frame)
    common_buttons_frame.pack(side=tk.LEFT, padx=10)
    
    # --- Funzioni di analisi e bottoni ---
    def aggiorna_risultati_globali(risultati_nuovi, info_ricerca=None, modalita="successivi"):
        global risultati_globali_spia, info_ricerca_globale_spia
        button_verifica_esiti.config(state=tk.DISABLED)
        button_visualizza.config(state=tk.DISABLED)
        
        if modalita == "successivi":
            risultati_globali_spia = risultati_nuovi
            info_ricerca_globale_spia = info_ricerca if info_ricerca else {}
            
            if risultati_globali_spia and info_ricerca_globale_spia.get('top_combinati') and info_ricerca_globale_spia.get('date_trigger_ordinate'):
                button_visualizza.config(state=tk.NORMAL)
                button_verifica_esiti.config(state=tk.NORMAL)
        else:
            risultati_globali_spia = []
            info_ricerca_globale_spia = {}
    
    def salva_risultati():
        risultato_text.config(state=tk.NORMAL)
        results_content = risultato_text.get(1.0, tk.END).strip()
        risultato_text.config(state=tk.DISABLED)
        
        default_msgs = ["Benvenuto", "Ricerca in corso...", "Nessun risultato", "Seleziona Ruota"]
        is_empty = not results_content or any(msg in results_content for msg in default_msgs)
        
        if is_empty:
            messagebox.showinfo("Nessun Risultato", "Niente da salvare.", parent=spia_win)
            return
            
        f_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")], 
                                            title="Salva Risultati", parent=spia_win)
        if f_path:
            try:
                with open(f_path, "w", encoding="utf-8") as f:
                    f.write(results_content)
                messagebox.showinfo("Salvataggio OK", f"Salvati in:\n{f_path}", parent=spia_win)
            except Exception as e:
                messagebox.showerror("Errore Salvataggio", f"Errore:\n{e}", parent=spia_win)
    
    # --- Importa le funzioni di analisi dal modulo spia2.py ---
    def importa_spia2():
        """Importa il modulo spia2.py e ritorna le funzioni necessarie."""
        try:
            module_path = os.path.join(os.path.dirname(__file__), "spia2.py")
            
            # Se il file non esiste in quel percorso, cerca nella directory corrente
            if not os.path.exists(module_path):
                module_path = "spia2.py"
                
            if not os.path.exists(module_path):
                raise FileNotFoundError(f"Il file spia2.py non è stato trovato")
            
            # Carica il modulo
            spec = importlib.util.spec_from_file_location("spia2", module_path)
            spia2 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(spia2)
            
            # Crea riferimenti alle funzioni che ci servono
            return {
                'carica_dati': spia2.carica_dati,
                'analizza_ruota_verifica': spia2.analizza_ruota_verifica,
                'analizza_antecedenti': spia2.analizza_antecedenti,
                'format_ambo_terno': spia2.format_ambo_terno,
                'visualizza_grafici': spia2.visualizza_grafici,
                'verifica_esiti_combinati': spia2.verifica_esiti_combinati
            }
        except Exception as e:
            messagebox.showerror("Errore Import", f"Impossibile importare spia2.py:\n{e}", parent=spia_win)
            traceback.print_exc()
            return None
    
    # Tenta di importare le funzioni di spia2.py
    spia2