import tkinter as tk
from tkinter import scrolledtext, filedialog
from datetime import datetime
import sys, os
from tkcalendar import DateEntry
from rit import LottoAnalyzer, RitardiAnalyzer, Backtester

# --- Funzioni di supporto ---
def create_analyzer_instance():
    """Crea un'istanza di LottoAnalyzer basata sulla scelta dell'utente."""
    source = source_var.get()
    path = path_var.get() if source == 'local' else None
    return LottoAnalyzer(data_source=source, local_path=path)

def select_folder():
    """Apre la finestra per scegliere la cartella e aggiorna l'etichetta."""
    # Partiamo dal Desktop
    initial_dir = os.path.join(os.path.expanduser('~'), 'Desktop')
    folder_selected = filedialog.askdirectory(initialdir=initial_dir)
    if folder_selected:
        path_var.set(folder_selected)

def update_source_choice(*args):
    """Abilita/disabilita il pulsante 'Scegli Cartella'."""
    if source_var.get() == 'local':
        folder_button.config(state=tk.NORMAL)
    else:
        folder_button.config(state=tk.DISABLED)

# --- Funzioni dei pulsanti principali ---
def do_live_analysis():
    output_text.delete('1.0', tk.END)
    print(">>> Avvio Analisi Live...\n")
    try:
        lotto_analyzer = create_analyzer_instance()
        ritardi_analyzer = RitardiAnalyzer(lotto_analyzer)
        ruote = list(lotto_analyzer.RUOTE_DISPONIBILI.keys())
        lotto_analyzer.carica_dati_per_ruote(ruote, print, force_reload=True)
        for ruota in ruote:
            report = ritardi_analyzer.genera_report_completo(ruota, top_n=5)
            print(report)
        print("\n--- ANALISI COMPLETATA ---")
    except Exception as e:
        print(f"ERRORE: {e}")

def do_historical_report():
    output_text.delete('1.0', tk.END)
    try:
        lotto_analyzer = create_analyzer_instance()
        data_test_str = data_entry.get()
        data_test = datetime.strptime(data_test_str, '%d/%m/%Y').date()
        print(f">>> Generazione Report Storico alla data: {data_test_str}...\n")
        ruote = list(lotto_analyzer.RUOTE_DISPONIBILI.keys())
        lotto_analyzer.carica_dati_per_ruote(ruote, print, force_reload=True)
        for ruota in ruote:
            estrazioni_complete = lotto_analyzer.estrazioni.get(ruota, [])
            estrazioni_storiche = [e for e in estrazioni_complete if e['data'] <= data_test]
            if not estrazioni_storiche:
                print(f"\n--- RUOTA {lotto_analyzer.RUOTE_DISPONIBILI[ruota].upper()} ---")
                print("Nessun dato storico trovato per questa data.")
                continue
            analyzer_storico = LottoAnalyzer(lotto_analyzer.data_source, lotto_analyzer.local_path)
            analyzer_storico.estrazioni[ruota] = estrazioni_storiche
            ritardi_analyzer_storico = RitardiAnalyzer(analyzer_storico)
            report = ritardi_analyzer_storico.genera_report_completo(ruota, top_n=5)
            print(report)
        print("\n--- ANALISI STORICA COMPLETATA ---")
    except Exception as e:
        print(f"ERRORE: {e}")

# Nel tuo file principale con l'interfaccia grafica
def do_backtest():
    output_text.delete('1.0', tk.END)
    # Rimuoviamo il print iniziale perché lo fa già la classe Backtester
    try:
        lotto_analyzer = create_analyzer_instance()
        backtester = Backtester(lotto_analyzer)
        
        data_test_str = data_entry.get()
        colpi_str = colpi_entry.get()
        top_n_str = top_n_entry.get()

        data_test = datetime.strptime(data_test_str, '%d/%m/%Y').date()
        colpi = int(colpi_str)
        top_n = int(top_n_str)
        
        ruote = list(lotto_analyzer.RUOTE_DISPONIBILI.keys())
        
        # --- MODIFICA FONDAMENTALE ---
        # 1. CHIAMA il metodo e SALVA la stringa restituita nella variabile 'report'.
        #    La funzione 'print' viene passata per i messaggi di caricamento.
        report = backtester.run_backtest(data_test, ruote, colpi, top_n, print)
        
        # 2. STAMPA la stringa del report che hai appena ricevuto.
        #    Se questa riga manca, non vedi nulla!
        print(report)
        # --- FINE MODIFICA FONDAMENTALE ---
        
        print("\n--- BACKTEST COMPLETATO ---")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERRORE: Controlla che la data e i numeri siano corretti.\n{e}")
# --- Interfaccia Grafica ---
root = tk.Tk()
root.title("LOTTORIT - Created by Max Lotto -")
root.geometry("800x700")

# Variabili per memorizzare le scelte dell'utente
source_var = tk.StringVar(value="github")
path_var = tk.StringVar(value="Nessuna cartella selezionata")
source_var.trace("w", update_source_choice)

# Frame per la scelta della fonte
source_frame = tk.LabelFrame(root, text="Sorgente Dati", padx=10, pady=10)
source_frame.pack(fill=tk.X, padx=10, pady=5)
tk.Radiobutton(source_frame, text="Internet (GitHub)", variable=source_var, value="github").pack(side=tk.LEFT)
tk.Radiobutton(source_frame, text="Locale", variable=source_var, value="local").pack(side=tk.LEFT, padx=20)
folder_button = tk.Button(source_frame, text="Scegli Cartella...", command=select_folder, state=tk.DISABLED)
folder_button.pack(side=tk.LEFT)
tk.Label(source_frame, textvariable=path_var, fg="blue").pack(side=tk.LEFT, padx=10)

# Titolo e pulsante live
title_label = tk.Label(root, text="Laboratorio Analisi Ritardi Lotto", font=("Helvetica", 16, "bold"))
title_label.pack(pady=5)
live_button = tk.Button(root, text="Analisi Live", command=do_live_analysis, bg="green", fg="white")
live_button.pack(fill=tk.X, padx=20, pady=5)
separator1 = tk.Frame(root, height=2, bd=1, relief=tk.SUNKEN)
separator1.pack(fill=tk.X, padx=5, pady=10)

# Sezione storica e backtest
backtest_label = tk.Label(root, text="Backtest e Analisi Storica", font=("Helvetica", 14))
backtest_label.pack()
data_frame = tk.Frame(root)
data_frame.pack(pady=2)
tk.Label(data_frame, text="Data di Riferimento:", width=25).pack(side=tk.LEFT)
data_entry = DateEntry(data_frame, width=12, locale='it_IT', date_pattern='dd/mm/yyyy')
data_entry.set_date(datetime(2025, 6, 1))
data_entry.pack(side=tk.LEFT)
colpi_frame = tk.Frame(root)
colpi_frame.pack(pady=2)
tk.Label(colpi_frame, text="Colpi da controllare (per Backtest):", width=25).pack(side=tk.LEFT)
colpi_entry = tk.Entry(colpi_frame)
colpi_entry.insert(0, "15")
colpi_entry.pack(side=tk.LEFT)
top_n_frame = tk.Frame(root)
top_n_frame.pack(pady=2)
tk.Label(top_n_frame, text="Analizza i primi N numeri:", width=25).pack(side=tk.LEFT)
top_n_entry = tk.Entry(top_n_frame)
top_n_entry.insert(0, "5")
top_n_entry.pack(side=tk.LEFT)
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
historical_report_button = tk.Button(button_frame, text="Mostra Report Storico", command=do_historical_report, bg="blue", fg="white")
historical_report_button.pack(side=tk.LEFT, padx=10)
backtest_button = tk.Button(button_frame, text="Avvia Backtest", command=do_backtest, bg="darkred", fg="white")
backtest_button.pack(side=tk.LEFT, padx=10)

# Output
separator2 = tk.Frame(root, height=2, bd=1, relief=tk.SUNKEN)
separator2.pack(fill=tk.X, padx=5, pady=10)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
output_text.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
class PrintRedirector:
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        self.widget.insert(tk.END, text); self.widget.see(tk.END)
    def flush(self): pass
sys.stdout = PrintRedirector(output_text)

root.mainloop()